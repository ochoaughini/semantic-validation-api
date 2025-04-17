from typing import Dict, Any, Optional, List
import time
import re
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..auth import get_api_key
from ..semantic_service import validate_semantic
from ..logging_config import logger, medical_logger

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/humanization",
    tags=["humanization"],
    dependencies=[Depends(get_api_key)]
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# --- Models --- #

class HumanizationRequest(BaseModel):
    """Request model for humanization validation."""
    input_text: str = Field(..., min_length=1, max_length=4000, 
                           description="Patient communication text to validate")
    reference_text: str = Field(..., min_length=1, max_length=4000, 
                               description="Reference communication text to compare against")
    module: str = Field(..., description="Validation module: ICSE or OIFC")
    custom_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, 
                                             description="Optional custom threshold (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for context")
    
    @validator("module")
    def validate_module(cls, v):
        """Ensure module is a valid humanization module."""
        if v not in ["ICSE", "OIFC"]:
            raise ValueError(f"Invalid module for humanization. Must be 'ICSE' or 'OIFC', got '{v}'")
        return v

class HumanizationResponse(BaseModel):
    """Response model for humanization validation."""
    input: str = Field(..., description="Input communication text")
    reference: str = Field(..., description="Reference communication text")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    match: bool = Field(..., description="Whether the communication styles match")
    threshold: float = Field(..., description="Threshold used for matching")
    module: str = Field(..., description="Module used for validation")
    model: str = Field(..., description="Model used for validation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    readability: Optional[Dict[str, Any]] = Field(None, description="Readability metrics")
    medical_terms: Optional[Dict[str, int]] = Field(None, description="Medical terminology analysis")

# --- Helper Functions --- #

def analyze_readability(text: str) -> Dict[str, Any]:
    """
    Analyze the readability of text for patient communication.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of readability metrics
    """
    # Count sentences (basic approximation)
    sentences = len(re.split(r'[.!?]+', text.strip()))
    if sentences == 0:
        sentences = 1  # Avoid division by zero
        
    # Count words
    words = len(re.findall(r'\b\w+\b', text))
    if words == 0:
        words = 1  # Avoid division by zero
    
    # Count syllables (very approximate)
    syllables = len(re.findall(r'[aeiouy]+', text.lower()))
    
    # Calculate average words per sentence
    avg_words_per_sentence = words / sentences
    
    # Calculate average syllables per word
    avg_syllables_per_word = syllables / words
    
    # Approximate Flesch Reading Ease (higher is more readable)
    # Formula: 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    flesch_reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    flesch_reading_ease = max(0, min(100, flesch_reading_ease))  # Clamp to 0-100
    
    return {
        "sentences": sentences,
        "words": words,
        "syllables": syllables,
        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2),
        "flesch_reading_ease": round(flesch_reading_ease, 1),
        "readability_level": get_readability_level(flesch_reading_ease)
    }

def get_readability_level(flesch_score: float) -> str:
    """Get readability level description from Flesch score."""
    if flesch_score >= 90:
        return "Very Easy - 5th grade"
    elif flesch_score >= 80:
        return "Easy - 6th grade"
    elif flesch_score >= 70:
        return "Fairly Easy - 7th grade"
    elif flesch_score >= 60:
        return "Standard - 8th/9th grade"
    elif flesch_score >= 50:
        return "Fairly Difficult - 10th/12th grade"
    elif flesch_score >= 30:
        return "Difficult - College"
    else:
        return "Very Difficult - College Graduate"

def analyze_medical_terminology(text: str) -> Dict[str, int]:
    """
    Analyze medical terminology usage in patient communication.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of medical terminology statistics
    """
    # Common medical term patterns
    med_terms_patterns = {
        "technical_terms": r'\b(diagnosis|prognosis|etiology|pathology|sequelae|comorbidity|contraindication)\b',
        "anatomy_terms": r'\b(cardiac|pulmonary|hepatic|renal|neural|cerebral|vascular|lymphatic)\b',
        "latin_terms": r'\b(in situ|in vitro|in vivo|per os|post mortem|ad lib|stat)\b',
        "measurements": r'\b(mmHg|mmol|µg|mcg|mEq)\b',
        "abbreviations": r'\b([A-Z]{2,5})\b'  # General pattern for medical abbreviations
    }
    
    # Count occurrences of different types of medical terms
    results = {}
    for term_type, pattern in med_terms_patterns.items():
        results[term_type] = len(re.findall(pattern, text, re.IGNORECASE))
    
    # Calculate overall counts
    results["total_technical_terms"] = sum(results.values())
    results["text_length"] = len(text)
    
    # Calculate density (terms per 100 words)
    words = len(re.findall(r'\b\w+\b', text))
    results["term_density"] = round((results["total_technical_terms"] / max(1, words)) * 100, 2)
    
    return results

# --- Routes --- #

@router.post("/validate", response_model=HumanizationResponse)
@limiter.limit("50/minute")
async def validate_humanization(request: HumanizationRequest, req: Request):
    """
    Validate patient communication text against a reference.
    
    - **ICSE** (Interpersonal Communication Style Equivalence): Validates communication style
    - **OIFC** (Overall Information Fidelity Check): Checks information accuracy while preserving simplicity
    """
    try:
        start_time = time.time()
        logger.info(f"Humanization validation request: module={request.module}")
        
        # Humanization-specific processing
        if request.module == "ICSE":
            # For interpersonal communication style, analyze readability
            readability = analyze_readability(request.input_text)
            
            # Validate communication style
            result = validate_semantic(
                request.input_text,
                request.reference_text,
                request.module,
                request.custom_threshold
            )
            
            # Add readability analysis
            result["readability"] = readability
            result["medical_terms"] = None  # Not needed for ICSE
            
        elif request.module == "OIFC":
            # For information fidelity, analyze medical terminology
            medical_terms = analyze_medical_terminology(request.input_text)
            
            # Validate information accuracy
            result = validate_semantic(
                request.input_text,
                request.reference_text,
                request.module, 
                request.custom_threshold
            )
            
            # Add medical terminology analysis
            result["medical_terms"] = medical_terms
            result["readability"] = analyze_readability(request.input_text)
            
        else:
            # This should never happen due to pydantic validation
            raise HTTPException(
                status_code=400,
                detail=f"Invalid humanization module: {request.module}"
            )
            
        # Calculate total processing time
        processing_time = (time.time() - start_time) * 1000
            
        # Construct response
        return {
            "input": request.input_text,
            "reference": request.reference_text,
            "similarity": result["similarity"],
            "match": result["match"],
            "threshold": result["threshold"],
            "module": request.module,
            "model": result["model"],
            "processing_time_ms": processing_time,
            "readability": result["readability"],
            "medical_terms": result["medical_terms"]
        }
    
    except ValueError as ve:
        # Handle validation errors
        logger.warning(f"Humanization validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in humanization validation: {str(e)}", exc_info=True)
        medical_logger.log_error(
            module=request.module if hasattr(request, "module") else "unknown",
            error_type="humanization_validation_error",
            error_message=str(e),
            details={"input_length": len(request.input_text) if hasattr(request, "input_text") else 0}
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred during humanization validation"
        )

