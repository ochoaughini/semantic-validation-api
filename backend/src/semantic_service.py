"""
Semantic Validation Service

This module provides core functionality for semantic text validation in medical contexts.
It handles medical terminology, abbreviations, and domain-specific text processing.
"""

import re
import time
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Set, Union
from functools import lru_cache
try:
    from numpy.typing import NDArray
except ImportError:
    # Fallback for older numpy versions
    from typing import Any as NDArray
from sentence_transformers import SentenceTransformer

# Import our configuration and logging
from src.config import config
from src.logging_config import logger, medical_logger

# Type aliases for clarity
EmbeddingVector = NDArray[np.float32]
ModelCache = Dict[str, SentenceTransformer]
EmbeddingCache = Dict[str, EmbeddingVector]
MetricsData = Dict[str, Dict[str, Union[int, float]]]

# Custom exceptions
class ValidationError(Exception):
    """Base exception for validation errors."""
    pass

class ModelLoadError(ValidationError):
    """Raised when a model fails to load."""
    pass

class EmbeddingError(ValidationError):
    """Raised when embedding generation fails."""
    pass

# Global model cache to avoid reloading models
_model_cache: ModelCache = {}

# Module constants from config
VARIATION_THRESHOLD = config.VARIATION_THRESHOLD
DEFAULT_MODEL = config.get_model()

# Cache for reference embeddings
_embedding_cache: EmbeddingCache = {}

# Quality metrics tracking
_module_metrics: MetricsData = {
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

# Load medical synonyms from config or use defaults
_medical_synonyms = config.get_medical_synonyms() or {
    # Default fallback synonyms if config is empty
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
    
    # General medical terms
    "presents with": ["exhibits", "shows", "displays", "demonstrates"],
    "positive": ["reactive", "detected", "present"],
    "negative": ["non-reactive", "not detected", "absent"]
}


# --- Helper Functions --- #

def _validate_module(module: str) -> str:
    """Validate and normalize module name."""
    if module not in _module_metrics:
        logger.warning(f"Unknown module: {module}, falling back to default (AMA)")
        return "AMA"
    return module

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
        Loaded SentenceTransformer model
        
    Raises:
        RuntimeError: If both requested model and default model fail to load
        
    Note:
        If the requested model fails to load, will attempt to fall back
        to the default model (unless already trying the default model).
    """
    try:
        # First try requested model (cached)
        if model_name in _model_cache:
            return _model_cache[model_name]

        # Load model if not in cache
        logger.info(f"⌛ Loading model: {model_name}")
        start_time = time.time()
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        load_time = time.time() - start_time
        logger.info(f"✓ Model {model_name} loaded successfully in {load_time:.2f}s")
        return model
    except Exception as e:
        error_msg = f"Failed to load model {model_name}: {str(e)}"
        logger.error(error_msg)
        medical_logger.log_error(
            module="system", 
            error_type="model_load_error",
            error_message=error_msg
        )
        
        # Fall back to default model only if this wasn't already the default
        if model_name != DEFAULT_MODEL:
            logger.info(f"Falling back to default model: {DEFAULT_MODEL}")
            return load_model(DEFAULT_MODEL)
        
        # If we're already trying the default model, raise a more specific error
        raise ModelLoadError(f"Failed to load default model: {str(e)}") from e

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
            details={
                "model": model_name, 
                "text_count": len(texts), 
                "original_error": str(e)
            }
        )
        raise EmbeddingError(f"Error generating embeddings with model {model_name}: {str(e)}") from e

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
    # Validate inputs
    if not text1 or not text2:
        raise ValueError("Both texts must be provided")
    if model is None:
        raise ValueError("Model must be provided")
    if (embedding1 is not None and embedding2 is None) or (embedding1 is None and embedding2 is not None):
        raise ValueError("Both embeddings must be provided if one is")

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
    # Use module-level constant for performance
    if similarity < VARIATION_THRESHOLD:
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
    # Validate and normalize module first
    module = _validate_module(module)
    
    start_time = time.time()
    try:
        # Track attempt
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
            embedding1=input_embedding, 
            embedding2=reference_embedding
        )
        # Determine if match based on threshold
        match = similarity >= threshold
        
        # Calculate processing time
        duration_ms = (time.time() - start_time) * 1000
        # Update module metrics
        _module_metrics[module]["successful"] += 1
        
        # Calculate new average processing time
        current_avg = _module_metrics[module]["avg_time_ms"]
        previous_count = _module_metrics[module]["successful"] - 1
        _module_metrics[module]["avg_time_ms"] = (
            (current_avg * previous_count + duration_ms) / 
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
    
    except ValidationError as e:
        # Handle known validation errors
        _module_metrics[module]["errors"] += 1
        logger.error(f"Validation error in {module}: {str(e)}")
        raise
    except Exception as e:
        # Handle unexpected errors
        _module_metrics[module]["errors"] += 1
        
        # Log the error
        error_msg = f"Unexpected error in {module}: {str(e)}"
        logger.error(error_msg)
        medical_logger.log_error(
            module=module, 
            error_type="unexpected_error", 
            error_message=error_msg,
            details={
                "input_length": len(input_text) if input_text else 0,
                "reference_length": len(reference_text) if reference_text else 0,
                "original_error": str(e)
            }
        )
        
        # Re-raise with additional context
        raise ValidationError(f"Validation failed: {str(e)}") from e


# --- Quality Metrics --- #

def _prepare_module_metrics(
    module: str, 
    data: Dict[str, Union[int, float]]
) -> Dict[str, Union[int, float]]:
    """
    Prepare metrics for a single module.
    
    Args:
        module: Module identifier
        data: Raw metrics data
        
    Returns:
        Processed metrics dictionary
    """
    return {
        "accuracy": round(1.0 - (data["errors"] / data["attempts"]) 
                        if data["attempts"] > 0 else 1.0, 3),
        "avg_time_ms": round(data["avg_time_ms"], 2),
        "attempts": data["attempts"],
        "successful": data["successful"],
        "errors": data["errors"]
    }


def _calculate_domain_accuracy(modules: List[str]) -> float:
    """Calculate accuracy for a set of modules."""
    total_errors = sum(_module_metrics[m]["errors"] for m in modules)
    total_attempts = sum(_module_metrics[m]["attempts"] for m in modules)
    return round(1.0 - total_errors / max(1, total_attempts), 3)


def get_loaded_model_count() -> int:
    """Get the number of currently loaded models."""
    return len(_model_cache)


def get_quality_metrics() -> Dict[str, Any]:
    """
    Get quality metrics for all modules.
    
    Returns:
        Dictionary of quality metrics
    """
    # Calculate overall metrics
    total_attempts = sum(m["attempts"] for m in _module_metrics.values())
    
    # Calculate error rate only once
    error_rate = 0.0
    if total_attempts > 0:
        total_errors = sum(m["errors"] for m in _module_metrics.values())
        error_rate = total_errors / total_attempts
    # Format for specific domains
    diagnostic_modules = ["AMA", "AI-MPN"]
    treatment_modules = ["TDBE", "SMCC"]
    humanization_modules = ["ICSE", "OIFC"]
    
    # Prepare detailed metrics
    metrics = {
        "modules": {
            module: _prepare_module_metrics(module, data)
            for module, data in _module_metrics.items()
        },
        "domains": {
            "diagnosis": {
                "accuracy": _calculate_domain_accuracy(diagnostic_modules)
            },
            "treatment": {
                "accuracy": _calculate_domain_accuracy(treatment_modules)
            },
            "humanization": {
                "accuracy": _calculate_domain_accuracy(humanization_modules)
            }
        },
        "overall": {
            "total_validations": total_attempts,
            "error_rate": round(error_rate, 3),
            "success_rate": round(1.0 - error_rate, 3),
            "models_loaded": get_loaded_model_count(),
            "uptime_stats": medical_logger.get_stats()
        }
    }
    
    return metrics
# Initialize semantic validation service
logger.info("=" * 60)
logger.info("🔬 Semantic Validation Service Initialization")
logger.info("=" * 60)
logger.info(f"📚 Default model: {DEFAULT_MODEL}")
logger.info(f"🎯 Variation threshold: {VARIATION_THRESHOLD}")
try:
    load_model(DEFAULT_MODEL)
    logger.info(f"✓ Model preloaded successfully")
    logger.info("-" * 50)
except Exception as e:
    logger.warning(f"! Model preload failed: {str(e)}")
    logger.info("! Model will be loaded on first request")
    logger.info("-" * 50)
