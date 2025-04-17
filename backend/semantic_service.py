import re
import string
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Set
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import config

# Global model cache to avoid reloading models
_model_cache: Dict[str, SentenceTransformer] = {}

# Cache for reference text embeddings
_reference_embeddings: Dict[str, np.ndarray] = {}

# Common medical reference texts that can be pre-computed
_reference_texts: Dict[str, str] = {
    "normal_vitals": "Heart rate, blood pressure, respiratory rate and temperature within normal limits.",
    "fever": "Patient presents with elevated temperature, indicative of fever.",
    "respiratory_symptoms": "Patient exhibits cough, shortness of breath, and respiratory discomfort.",
    "normal_neurological": "Patient is alert and oriented, with normal neurological examination.",
    "normal_cardiac": "Regular heart rate and rhythm, no murmurs, gallops, or rubs.",
    "normal_respiratory": "Clear lung fields bilaterally, no wheezes, rales, or rhonchi."
}


def _get_model(model_type: str = "default") -> SentenceTransformer:
    """
    Get a sentence transformer model from cache or load it.
    
    Args:
        model_type: Type of model to load ("default", "clinical", etc.)
        
    Returns:
        SentenceTransformer model
    """
    model_name = config.get_model(model_type)
    
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    logger.info(f"Loading model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        # Fallback to default model
        if model_name != "all-MiniLM-L6-v2":
            logger.info("Falling back to default model")
            return _get_model("default")
        raise


def preprocess_text(text: str) -> str:
    """
    Preprocess text according to configuration settings.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    preprocessing_config = config.get_preprocessing_config()
    
    # Apply basic preprocessing
    if preprocessing_config.get("lowercase", True):
        text = text.lower()
        
    if preprocessing_config.get("normalize_whitespace", True):
        text = re.sub(r'\s+', ' ', text).strip()
        
    if preprocessing_config.get("remove_punctuation", True):
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Apply medical term normalization if enabled
    if preprocessing_config.get("normalize_medical_terms", True):
        text = normalize_medical_terms(text)
    
    return text


def normalize_medical_terms(text: str) -> str:
    """
    Normalize medical terms and abbreviations.
    
    Args:
        text: Input text with medical terms
        
    Returns:
        Text with normalized medical terms
    """
    abbreviations = config.get_medical_abbreviations()
    
    # Create a pattern to match whole words only
    pattern = r'\b(' + '|'.join(re.escape(abbr) for abbr in abbreviations.keys()) + r')\b'
    
    def replace_abbreviation(match):
        return abbreviations[match.group(0)]
    
    # Replace abbreviations
    text = re.sub(pattern, replace_abbreviation, text, flags=re.IGNORECASE)
    
    return text


def calculate_similarity(text1: str, text2: str, model_type: str = "default") -> float:
    """
    Calculate cosine similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        model_type: Type of model to use
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Get model
    model = _get_model(model_type)
    
    # Generate embeddings
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    
    # Calculate cosine similarity
    similarity = float(np.dot(embedding1, embedding2) / 
                      (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    return similarity


def get_reference_embedding(reference_key: str, model_type: str = "default") -> Optional[np.ndarray]:
    """
    Get pre-computed embedding for a reference text.
    
    Args:
        reference_key: Key of the reference text
        model_type: Type of model used
        
    Returns:
        Pre-computed embedding or None if not found
    """
    cache_key = f"{reference_key}_{model_type}"
    
    if cache_key in _reference_embeddings:
        return _reference_embeddings[cache_key]
    
    if reference_key in _reference_texts:
        model = _get_model(model_type)
        text = _reference_texts[reference_key]
        embedding = model.encode(text)
        _reference_embeddings[cache_key] = embedding
        return embedding
    
    return None


def validate_semantic(
    input_text: str, 
    reference_text: str, 
    module: str = None, 
    submodule: str = None,
    model_type: str = "default",
    custom_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate semantic similarity between two texts.
    
    Args:
        input_text: Input text to validate
        reference_text: Reference text to compare against
        module: Optional module name for domain-specific validation
        submodule: Optional submodule name for specific threshold
        model_type: Type of model to use
        custom_threshold: Custom threshold to override configuration
        
    Returns:
        Dictionary with similarity score and match result
    """
    # Preprocess texts
    processed_input = preprocess_text(input_text)
    processed_reference = preprocess_text(reference_text)
    
    # Use clinical model for medical text if available
    if module in ["diagnosis", "treatment"] and "clinical" in config.config.get("models", {}):
        model_type = "clinical"
    
    # Calculate similarity
    similarity_score = calculate_similarity(processed_input, processed_reference, model_type)
    
    # Determine threshold
    if custom_threshold is not None:
        threshold = custom_threshold
    elif module and submodule:
        threshold = config.get_module_threshold(module, submodule)
    else:
        threshold = config.get_threshold("default")
    
    # Check if score meets threshold
    match = similarity_score >= threshold
    
    # Log validation result
    logger.debug(f"Validation: score={similarity_score:.4f}, threshold={threshold:.4f}, match={match}")
    logger.debug(f"Input: '{input_text[:50]}...' Reference: '{reference_text[:50]}...'")
    
    return {
        "similarity": similarity_score,
        "match": match,
        "threshold": threshold,
        "model": config.get_model(model_type)
    }


def precompute_reference_embeddings() -> None:
    """Precompute embeddings for all reference texts with all models."""
    for model_type in config.config.get("models", {}):
        model = _get_model(model_type)
        for key, text in _reference_texts.items():
            cache_key = f"{key}_{model_type}"
            if cache_key not in _reference_embeddings:
                _reference_embeddings[cache_key] = model.encode(text)
    
    logger.info(f"Precomputed {len(_reference_embeddings)} reference embeddings")


# Precompute reference embeddings on module import
try:
    precompute_reference_embeddings()
except Exception as e:
    logger.warning(f"Failed to precompute reference embeddings: {str(e)}")

