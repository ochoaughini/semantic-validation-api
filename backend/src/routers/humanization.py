from typing import Dict, Any, Optional, List
import time
import re
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

# ---- Constants and Patterns ---- #

# Readability patterns compiled for performance
READABILITY_PATTERNS = {
    'sentence_end': re.compile(r'[.!?]+'),
    'words': re.compile(r'\b\w+\b'),
    'syllables': re.compile(r'[aeiouy]+', re.IGNORECASE),
    'complex_words': re.compile(r'\b\w{3,}\b')
}

# ---- Medical Terminology ---- #

# Professional terms that should be simplified
COMPLEX_TERMS = [
    "hypertension", "myocardial", "infarction", "hyperlipidemia",
    "cardiovascular", "endocrine", "gastrointestinal"
]

# Patient-friendly alternatives
SIMPLIFIED_TERMS = {
    "hypertension": "high blood pressure",
    "myocardial infarction": "heart attack",
    "hyperlipidemia": "high cholesterol",
    "cardiovascular": "heart and blood vessel",
    "endocrine": "hormone",
    "gastrointestinal": "stomach and intestine"
}
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

# Medical terminology patterns
MEDICAL_PATTERNS = {
    'abbreviations': re.compile(r'\b[A-Z]{2,5}\b'),  # Fixed pattern
    'technical': re.compile(r'\b(' + '|'.join(re.escape(term) for term in COMPLEX_TERMS) + r')\b', 
                          re.IGNORECASE)
}
# ---- Models ---- #

class ReadabilityMetrics(BaseModel):
    """Model for text readability metrics."""
    flesch_score: float = Field(..., description="Flesch Reading Ease score")
    grade_level: float = Field(..., description="Flesch-Kincaid Grade Level")
    complex_word_count: int = Field(..., description="Number of complex words")
    avg_words_per_sentence: float = Field(..., description="Average words per sentence")
    suggested_simplifications: Dict[str, str] = Field(
        default_factory=dict,
        description="Suggested term simplifications"
    )


class HumanizationElementsContext(BaseModel):
    """Model for humanization analysis context."""
    input: ReadabilityMetrics = Field(..., description="Input text metrics")
    reference: ReadabilityMetrics = Field(..., description="Reference text metrics")


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
    input: str = Field(..., description="Input humanized text")
    reference: str = Field(..., description="Reference humanized text")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    match: bool = Field(..., description="Whether the texts match semantically")
    threshold: float = Field(..., description="Threshold used for matching")
    module: str = Field(..., description="Module used for validation")
    model: str = Field(..., description="Model used for validation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    readability_metrics: Optional[HumanizationElementsContext] = Field(
        None, 
        description="Readability metrics for both texts"
    )

# ---- Readability Analysis ---- #

def analyze_readability(text: str) -> Dict[str, Any]:
    """
    Analyze the readability of text for patient communication.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of readability metrics
    """
    # Split into sentences
    sentences = [s.strip() for s in READABILITY_PATTERNS['sentence_end'].split(text) if s.strip()]
    sentence_count = max(1, len(sentences))
        
    # Count words
    words = READABILITY_PATTERNS['words'].findall(text)
    word_count = max(1, len(words))
    
    # Count syllables
    syllable_count = sum(len(READABILITY_PATTERNS['syllables'].findall(word)) for word in words)
    
    # Calculate metrics
    avg_words_per_sentence = word_count / sentence_count
    avg_syllables_per_word = syllable_count / word_count
    
    # Calculate Flesch Reading Ease
    flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
    
    # Calculate grade level using a more structured approach
    def get_grade_level(score: float) -> float:
        if score >= 90: return 5.0     # 5th grade
        if score >= 80: return 6.0     # 6th grade
        if score >= 70: return 7.0     # 7th grade
        if score >= 60: return 8.5     # 8-9th grade
        if score >= 50: return 10.5    # 10-11th grade
        if score >= 30: return 13.0    # College level
        return 16.0                    # College graduate level
    
    grade_level = get_grade_level(flesch_score)
    
    return {
        "flesch_score": round(flesch_score, 2),
        "grade_level": round(grade_level, 1),
        "complex_word_count": len([w for w in words if len(READABILITY_PATTERNS['syllables'].findall(w)) >= 3]),
        "avg_words_per_sentence": round(avg_words_per_sentence, 2),
        "suggested_simplifications": {
            term: SIMPLIFIED_TERMS.get(term, "") 
            for term in COMPLEX_TERMS if term.lower() in text.lower()
        }
    }

# ---- Medical Analysis ---- #

def analyze_medical_terminology(text: str) -> Dict[str, int]:
    """
    Analyze medical terminology usage in patient communication.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of medical terminology metrics
    """
    # Count occurrences of different types of medical terms
    results = {}
    
    # Count abbreviations
    results["abbreviations"] = len(MEDICAL_PATTERNS['abbreviations'].findall(text))
    
    # Count technical terms
    results["technical"] = len([term for term in COMPLEX_TERMS 
                               if term.lower() in text.lower()])
    
    # Calculate overall counts
    results["total_technical_terms"] = sum(results.values())
    results["text_length"] = len(text)
    
    # Calculate density (terms per 100 words)
    words = len(re.findall(r'\b\w+\b', text))
    results["term_density"] = round((results["total_technical_terms"] / max(1, words)) * 100, 2)
    
    return results

# ---- Helper Functions ---- #

def calculate_readability_metrics(text: str) -> Dict[str, Any]:
    """Calculate readability metrics for text."""
    # Reuse analyze_readability function to avoid duplicate logic
    return analyze_readability(text)


# ---- Routes ---- #

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
        
        # Calculate readability metrics for both texts
        input_metrics = calculate_readability_metrics(request.input_text)
        reference_metrics = calculate_readability_metrics(request.reference_text)
        
        # Use semantic validation
        result = validate_semantic(
            request.input_text,
            request.reference_text,
            module=request.module,
            custom_threshold=request.custom_threshold
        )
            
        # Calculate total processing time
        processing_time = (time.time() - start_time) * 1000
            
        # Construct response
        return HumanizationResponse(
            input=request.input_text,
            reference=request.reference_text,
            similarity=result["similarity"],
            match=result["match"],
            threshold=result["threshold"],
            module=request.module,
            model=result["model"],
            processing_time_ms=round(processing_time, 2),
            readability_metrics=HumanizationElementsContext(
                input=ReadabilityMetrics(**input_metrics),
                reference=ReadabilityMetrics(**reference_metrics)
            )
        )
    
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

