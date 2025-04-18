import re
import time
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Set
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# Import our configuration and logging
from .config import config
from .logging_config import logger, medical_logger

# Global model cache to avoid reloading models
_model_cache: Dict[str, SentenceTransformer] = {}

# Cache for reference embeddings
_embedding_cache: Dict[str, np.ndarray] = {}

# Quality metrics tracking
_module_metrics = {
    "AMA": {"attempts": 0, "successful": 0, "errors": 0, "avg_time_ms": 0},
    "AI-MPN": {"attempts": 0, "successful": 0, "errors": 0, "avg_time_ms": 0},
    "TDBE": {"attempts": 0, "successful": 0, "errors": 0, "avg_time_ms": 0},
    "SMCC": {"attempts": 0, "successful": 0, "errors": 0, "avg_time_ms": 0},
    "ICSE": {"attempts": 0, "successful": 0, "errors": 0, "avg_time_ms": 0},
    "OIFC": {"attempts": 0, "successful": 0, "errors": 0, "avg_time_ms": 0}
}
# Common medical text patterns
_medical_patterns = {
    "measurements": r'\b\d+\.?\d*\s*(mg|g|ml|l|mmol|µg|mcg|IU|mEq)\b',
    "vitals": r'\b(temperature|pulse|blood pressure|BP|HR|RR|SpO2|SaO2)\b',
    "timing": r'\b(bid|tid|qid|daily|weekly|monthly|hourly|prn|as needed)\b'
}

# Medical term synonyms for better matching
_medical_synonyms = {
    # Temperature terms
    "fever": ["elevated temperature", "pyrexia", "hyperthermia", "febrile", "high temperature"],
    "temperature": ["temp", "body temperature", "fever"],
    
    # Respiratory terms
    "cough": ["tussis", "coughing"],
    "persistent cough": ["chronic cough", "ongoing cough", "continuous cough"],
    "respiratory": ["pulmonary", "lung", "airways", "breathing"],
    "shortness of breath": ["dyspnea", "breathlessness", "difficulty breathing"],
    
    # Infection terms
    "bacteria": ["bacterial", "bacteriological"],
    "infection": ["infectious process", "inflammatory process", "sepsis"],
    "culture": ["bacterial culture", "microbiological test", "bacteriological test"],
    
    # General medical terms
    "presents with": ["exhibits", "shows", "displays", "demonstrates", "manifests"],
    "indicative": ["suggestive", "consistent with", "compatible with", "characteristic of"],
    "positive": ["reactive", "detected", "present"],
    "negative": ["non-reactive", "not detected", "absent"]
}


# --- Medical Term Handling --- #

def expand_abbreviations(text: str) -> str:
    """
    Expand medical abbreviations in the text.
    
    Args:
        text: Input text with possible medical abbreviations
        
    Returns:
        Text with expanded abbreviations
    """
    if not text:
        return ""
        
    abbreviations = config.get_medical_abbreviations()
    if not abbreviations:
        return text
    
    # Create pattern for whole-word matching
    pattern = r'\b(' + '|'.join(re.escape(abbr) for abbr in abbreviations.keys()) + r')\b'
    
    def replace_abbreviation(match):
        return abbreviations[match.group(0)]
    
    # Replace abbreviations with their expansions
    return re.sub(pattern, replace_abbreviation, text, flags=re.IGNORECASE)


def normalize_medical_text(text: str, module: str) -> str:
    """
    Normalize medical text based on the specific domain module.
    
    Args:
        text: Input medical text
        module: Medical domain module
        
    Returns:
        Normalized text appropriate for the domain
    """
    if not text:
        return ""
    
    # Basic normalization for all domains
    text = text.lower().strip()
    
    # Expand abbreviations
    text = expand_abbreviations(text)
    
    # Apply medical term normalization
    # Standardize key medical terms
    text = text.replace("high fever", "elevated temperature")
    text = text.replace("febrile", "fever")
    text = text.replace("pyrexia", "fever")
    text = text.replace("bacterial", "bacteria")
    text = text.replace("bacteriological", "bacteria")
    
    # Module-specific normalization
    if module in ["AMA", "AI-MPN"]:
        # For diagnosis modules, preserve medical measurements
        text = re.sub(r'[^\w\s' + re.escape('.,;()[]{}+-*/') + ']', ' ', text)
        
        # Standardize measurements format
        text = re.sub(r'(\d+)\s*(/)\s*(\d+)', r'\1\2\3', text)  # Fix "120 / 80" to "120/80"
        
        # Standardize key diagnosis terms
        text = text.replace("respiratory infection", "pulmonary infection")
        text = text.replace("lung infection", "pulmonary infection")
        text = text.replace("chronic", "persistent")
        
    elif module in ["TDBE", "SMCC"]:
        # For treatment modules, preserve drug dosages
        text = re.sub(r'[^\w\s' + re.escape('.,;()[]{}+-*/') + ']', ' ', text)
        
        # Standardize dosage format (e.g., "5 mg" to "5mg")
        text = re.sub(r'(\d+)\s+(mg|ml|g|mcg|µg)', r'\1\2', text)
        
    elif module in ["ICSE", "OIFC"]:
        # For humanization modules, preserve more punctuation
        text = re.sub(r'[^\w\s' + re.escape('.,;:()[]{}?!-') + ']', ' ', text)
    
    # Normalize whitespace for all modules
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def handle_synonyms(text: str) -> List[str]:
    """
    Generate synonym variations for medical terms.
    
    Args:
        text: Input medical text
        
    Returns:
        List of text variations with synonyms
    """
    variations = [text]
    
    # First use the medical synonyms from config (if any)
    config_synonyms = config.get_medical_synonyms()
    if config_synonyms:
        for term, term_synonyms in config_synonyms.items():
            if term.lower() in text.lower():
                for synonym in term_synonyms:
                    # Create variation with this synonym replacement
                    variation = re.sub(r'\b' + re.escape(term) + r'\b', synonym, text, flags=re.IGNORECASE)
                    variations.append(variation)
    
    # Then use our more comprehensive medical synonyms
    for term, term_synonyms in _medical_synonyms.items():
        if term.lower() in text.lower():
            for synonym in term_synonyms:
                # Create variation with this synonym replacement
                variation = re.sub(r'\b' + re.escape(term) + r'\b', synonym, text, flags=re.IGNORECASE)
                if variation not in variations:
                    variations.append(variation)
    
    return variations


# --- Model Loading and Caching --- #

def load_model(model_name: str) -> SentenceTransformer:
    """
    Load the specified sentence transformer model.
    
    Args:
        model_name: Name of the model to load
    
    Returns:
        Loaded sentence transformer model
        
    Raises:
        ValueError: If the model cannot be loaded
    """
    # Check if model is already loaded
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        load_time = time.time() - start_time
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
        return model
    except Exception as e:
        error_msg = f"Failed to load model {model_name}: {str(e)}"
        logger.error(error_msg)
        medical_logger.log_error(
            module="system", 
            error_type="model_load_error", 
            error_message=error_msg
        )
        # Fall back to default model if available
        default_model = config.get_model()
        if model_name != default_model:
            logger.info(f"Falling back to default model: {default_model}")
            return load_model(default_model)
        
        raise RuntimeError(f"Failed to load any model: {str(e)}")


# --- Embedding and Similarity Calculation --- #

def get_embeddings(texts: List[str], model_name: str) -> List[np.ndarray]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the model to use
        
    Returns:
        List of embedding vectors
    """
    model = load_model(model_name)
    try:
        return model.encode(texts, convert_to_numpy=True)
    except Exception as e:
        error_msg = f"Error generating embeddings: {str(e)}"
        logger.error(error_msg)
        medical_logger.log_error(
            module="system", 
            error_type="embedding_error", 
            error_message=error_msg,
            details={"model": model_name, "text_count": len(texts)}
        )
        raise


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between 0 and 1
    """
    # Calculate cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    
    # Ensure the result is within [0, 1]
    return float(max(0.0, min(1.0, similarity)))


def calculate_enhanced_similarity(
    text1: str, 
    text2: str, 
    model: SentenceTransformer,
    embedding1: np.ndarray = None,
    embedding2: np.ndarray = None
) -> float:
    """
    Calculate similarity with enhanced medical synonym handling.
    
    Args:
        text1: First text
        text2: Second text
        model: SentenceTransformer model
        embedding1: Optional pre-computed embedding for text1
        embedding2: Optional pre-computed embedding for text2
        
    Returns:
        Best similarity score found
    """
    # Calculate base similarity if embeddings provided
    if embedding1 is not None and embedding2 is not None:
        similarity = calculate_similarity(embedding1, embedding2)
    else:
        # Generate base embeddings
        embedding1 = model.encode(text1)
        embedding2 = model.encode(text2)
        similarity = calculate_similarity(embedding1, embedding2)
    
    # Try with variations if similarity is below the variation threshold
    # Default to 0.8 if not defined in config
    variation_threshold = 0.8  # Default threshold for variation checking
    if similarity < variation_threshold:
        # Generate variations with medical synonyms
        text1_variations = handle_synonyms(text1)
        text2_variations = handle_synonyms(text2)
        
        # Try all combinations of variations
        for var1 in text1_variations[1:]:  # Skip the original
            emb1 = model.encode(var1)
            sim = calculate_similarity(emb1, embedding2)
            if sim > similarity:
                similarity = sim
                
        for var2 in text2_variations[1:]:  # Skip the original
            emb2 = model.encode(var2)
            sim = calculate_similarity(embedding1, emb2)
            if sim > similarity:
                similarity = sim
    
    return similarity


# --- Core Validation Functions --- #

def validate_semantic(
    input_text: str, 
    reference_text: str, 
    module: str,
    submodule: Optional[str] = None,
    model_type: Optional[str] = None,
    custom_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate semantic similarity between two texts with medical domain support.
    
    Args:
        input_text: Input text to validate
        reference_text: Reference text to compare against
        module: Medical domain module (AMA, AI-MPN, etc.)
        submodule: Optional submodule for more specific processing
        model_type: Optional specific model type to use
        custom_threshold: Optional custom threshold
        
    Returns:
        Dictionary with validation results
    """
    start_time = time.time()
    
    try:
        # Update metrics - use get() to handle unknown modules
        if module not in _module_metrics:
            logger.warning(f"Unknown module: {module}, using default")
            module = "AMA"  # Fallback to default module
        _module_metrics[module]["attempts"] += 1
        
        # Get appropriate model and threshold
        model_name = config.get_model_for_module(module)
        
        # Set threshold - use custom threshold if provided, otherwise get from config
        threshold = custom_threshold if custom_threshold is not None else config.get_threshold_for_module(module)
        
        # Normalize texts according to domain
        normalized_input = normalize_medical_text(input_text, module)
        normalized_reference = normalize_medical_text(reference_text, module)
        
        # Load model
        model = load_model(model_name)
        
        # Generate embeddings
        input_embedding = model.encode(normalized_input)
        reference_embedding = model.encode(normalized_reference)
        
        # Calculate enhanced similarity with synonym handling
        similarity = calculate_enhanced_similarity(
            normalized_input, 
            normalized_reference, 
            model,
            input_embedding, 
            reference_embedding
        )
        
        # Determine if match based on threshold
        match = similarity >= threshold
        
        # Calculate processing time
        duration_ms = (time.time() - start_time) * 1000
        
        # Update module metrics
        _module_metrics[module]["successful"] += 1
        _module_metrics[module]["avg_time_ms"] = (
            (_module_metrics[module]["avg_time_ms"] * (_module_metrics[module]["successful"] - 1) + duration_ms) / 
            _module_metrics[module]["successful"]
        )
        
        # Log validation with medical domain context
        medical_logger.log_validation(
            module=module,
            similarity=similarity,
            match=match,
            threshold=threshold,
            input_length=len(input_text),
            reference_length=len(reference_text),
            model_name=model_name,
            duration_ms=duration_ms
        )
        
        # Return validation result
        return {
            "similarity": similarity,
            "match": match,
            "threshold": threshold,
            "model": model_name,
            "processing_time_ms": duration_ms
        }
    
    except Exception as e:
        # Track error in metrics
        _module_metrics[module]["errors"] += 1
        
        # Log the error
        error_msg = f"Validation error for module {module}: {str(e)}"
        logger.error(error_msg)
        medical_logger.log_error(
            module=module, 
            error_type="validation_error", 
            error_message=error_msg,
            details={
                "input_length": len(input_text) if input_text else 0,
                "reference_length": len(reference_text) if reference_text else 0
            }
        )
        
        # Re-raise with additional context
        raise RuntimeError(f"Validation failed: {str(e)}")


# --- Quality Metrics --- #

def get_quality_metrics() -> Dict[str, Any]:
    """
    Get quality metrics for all modules.
    
    Returns:
        Dictionary of quality metrics
    """
    # Calculate overall metrics
    total_attempts = sum(m["attempts"] for m in _module_metrics.values())
    total_errors = sum(m["errors"] for m in _module_metrics.values())
    total_successful = sum(m["successful"] for m in _module_metrics.values())
    
    # Calculate error rates
    error_rate = total_errors / total_attempts if total_attempts > 0 else 0
    
    # Format for specific domains
    diagnostic_modules = ["AMA", "AI-MPN"]
    treatment_modules = ["TDBE", "SMCC"]
    humanization_modules = ["ICSE", "OIFC"]
    
    # Prepare detailed metrics
    metrics = {
        "modules": {
            module: {
                "accuracy": round(1.0 - (data["errors"] / data["attempts"]) if data["attempts"] > 0 else 1.0, 3),
                "avg_time_ms": round(data["avg_time_ms"], 2),
                "attempts": data["attempts"],
                "successful": data["successful"],
                "errors": data["errors"]
            }
            for module, data in _module_metrics.items()
        },
        "domains": {
            "diagnosis": {
                "accuracy": round(
                    1.0 - sum((_module_metrics[m]["errors"] for m in diagnostic_modules)) / 
                    max(1, sum((_module_metrics[m]["attempts"] for m in diagnostic_modules))), 
                    3
                )
            },
            "treatment": {
                "accuracy": round(
                    1.0 - sum((_module_metrics[m]["errors"] for m in treatment_modules)) / 
                    max(1, sum((_module_metrics[m]["attempts"] for m in treatment_modules))), 
                    3
                )
            },
            "humanization": {
                "accuracy": round(
                    1.0 - sum((_module_metrics[m]["errors"] for m in humanization_modules)) / 
                    max(1, sum((_module_metrics[m]["attempts"] for m in humanization_modules))), 
                    3
                )
            }
        },
        "overall": {
            "total_validations": total_attempts,
            "error_rate": round(error_rate, 3),
            "success_rate": round(1.0 - error_rate, 3),
            "models_loaded": len(_model_cache),
            "uptime_stats": medical_logger.get_stats()
        }
    }
    
    return metrics

# Try to load the default model on startup
try:
    default_model = config.get_model()
    logger.info(f"Preloading default model: {default_model}")
    load_model(default_model)
except Exception as e:
    # Non-fatal error, log and continue (model will be loaded on first request)
    logger.warning(f"Could not preload model: {str(e)}")
