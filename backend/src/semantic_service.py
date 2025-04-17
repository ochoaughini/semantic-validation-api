import re
import time
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Set
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# Import our configuration and logging
from .config import (
    get_medical_abbreviations,
    get_medical_synonyms,
    get_threshold_for_module,
    get_model_for_module,
    EMBEDDING_MODEL
)
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
        
    abbreviations = get_medical_abbreviations()
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
    
    # Module-specific normalization
    if module in ["AMA", "AI-MPN"]:
        # For diagnosis modules, preserve medical measurements
        text = re.sub(r'[^\w\s' + re.escape('.,;()[]{}+-*/') + ']', ' ', text)
        
        # Standardize measurements format
        text = re.sub(r'(\d+)\s*(/)\s*(\d+)', r'\1\2\3', text)  # Fix "120 / 80" to "120/80"
        
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
    synonyms = get_medical_synonyms()
    
    if not synonyms:
        return variations
    
    # For each term with synonyms, create variations
    for term, term_synonyms in synonyms.items():
        if term.lower() in text.lower():
            for synonym in term_synonyms:
                # Create variation with this synonym replacement
                variation = re.sub(r'\b' + re.escape(term) + r'\b', synonym, text, flags=re.IGNORECASE)
                variations.append(variation)
    
    return variations


# --- Model Loading and Caching --- #

def load_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache the sentence transformer model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded SentenceTransformer model
    """
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
        if model_name != EMBEDDING_MODEL:
            logger.info(f"Falling back to default model: {EMBEDDING_MODEL}")
            return load_model(EMBEDDING_MODEL)
        
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
    similarity = float(
        np.dot(embedding1, embedding2) / 
        (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    )
    
    # Ensure the result is within [0, 1]
    return max(0.0, min(1.0, similarity))


# --- Core Validation Functions --- #

def validate_semantic(
    input_text: str, 
    reference_text: str, 
    module: str,
    custom_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate semantic similarity between two texts with medical domain support.
    
    Args:
        input_text: Input text to validate
        reference_text: Reference text to compare against
        module: Medical domain module (AMA, AI-MPN, etc.)
        custom_threshold: Optional custom threshold
        
    Returns:
        Dictionary with validation results
    """
    start_time = time.time()
    
    try:
        # Update metrics
        _module_metrics[module]["attempts"] += 1
        
        # Get appropriate model and threshold
        model_name = get_model_for_module(module)
        threshold = custom_threshold if custom_threshold is not None else get_threshold_for_module(module)
        
        # Normalize texts according to domain
        normalized_input = normalize_medical_text(input_text, module)
        normalized_reference = normalize_medical_text(reference_text, module)
        
        # Generate embeddings
        input_embedding = get_embeddings([normalized_input], model_name)[0]
        reference_embedding = get_embeddings([normalized_reference], model_name)[0]
        
        # Calculate similarity
        similarity = calculate_similarity(input_embedding, reference_embedding)
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


# Initialize models on module import
try:
    logger.info(f"Preloading default model: {EMBEDDING_MODEL}")
    load_model(EMBEDDING_MODEL)
    logger.info("Semantic service initialized successfully")
except Exception as e:
    logger.error(f"Error initializing semantic service: {str(e)}")

