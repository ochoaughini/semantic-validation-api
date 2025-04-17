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
    prefix="/api/treatment",
    tags=["treatment"],
    dependencies=[Depends(get_api_key)]
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# --- Models --- #

class TreatmentRequest(BaseModel):
    """Request model for treatment validation."""
    input_text: str = Field(..., min_length=1, max_length=3000, 
                           description="Treatment description to validate")
    reference_text: str = Field(..., min_length=1, max_length=3000, 
                               description="Reference treatment text to compare against")
    module: str = Field(..., description="Validation module: TDBE or SMCC")
    custom_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, 
                                             description="Optional custom threshold (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    
    @validator("module")
    def validate_module(cls, v):
        """Ensure module is a valid treatment module."""
        if v not in ["TDBE", "SMCC"]:
            raise ValueError(f"Invalid module for treatment. Must be 'TDBE' or 'SMCC', got '{v}'")
        return v
    
    @validator("input_text", "reference_text")
    def validate_treatment_texts(cls, v):
        """Ensure treatment texts are valid and check for required elements."""
        if not v or not v.strip():
            raise ValueError("Treatment text cannot be empty")
            
        # For treatment texts, we do basic validation of content
        v = v.strip()
        
        # Check for suspicious characters that might indicate code injection
        if re.search(r'[<>{}$]', v):
            raise ValueError("Treatment text contains invalid characters")
            
        return v

class TreatmentResponse(BaseModel):
    """Response model for treatment validation."""
    input: str = Field(..., description="Input treatment text")
    reference: str = Field(..., description="Reference treatment text")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    match: bool = Field(..., description="Whether the treatments match semantically")
    threshold: float = Field(..., description="Threshold used for matching")
    module: str = Field(..., description="Module used for validation")
    model: str = Field(..., description="Model used for validation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    treatment_elements: Optional[Dict[str, bool]] = Field(None, description="Treatment elements present")

# --- Helper Functions --- #

def extract_treatment_elements(text: str) -> Dict[str, bool]:
    """
    Extract key treatment elements from the text.
    
    Args:
        text: Treatment text to analyze
        
    Returns:
        Dictionary of treatment elements and whether they are present
    """
    text = text.lower()
    
    # Check for key treatment elements
    elements = {
        "dosage": bool(re.search(r'\b\d+\.?\d*\s*(mg|g|ml|mcg|µg|units)\b', text)),
        "frequency": bool(re.search(r'\b(daily|weekly|monthly|hourly|once|twice|bid|tid|qid|prn)\b', text)),
        "route": bool(re.search(r'\b(oral|iv|intravenous|im|intramuscular|sc|subcutaneous|topical|inhaled)\b', text)),
        "duration": bool(re.search(r'\b(\d+\s*(day|week|month|hour|minute|year|min)s?)\b', text)),
        "drug_name": bool(re.search(r'\b(tablet|capsule|injection|solution|suspension|pill)\b', text))
    }
    
    return elements

# --- Routes --- #

@router.post("/validate", response_model=TreatmentResponse)
@limiter.limit("50/minute")
async def validate_treatment(request: TreatmentRequest, req: Request):
    """
    Validate medical treatment text against a reference.
    
    - **TDBE** (Treatment Description Basic Elements): Validates essential treatment details
    - **SMCC** (Synthesized Medical Care Comparison): Compares treatment approaches
    """
    try:
        start_time = time.time()
        logger.info(f"Treatment validation request: module={request.module}")
        
        # Treatment-specific preprocessing and validation
        if request.module == "TDBE":
            # For treatment description, we check for essential elements first
            treatment_elements = extract_treatment_elements(request.input_text)
            
            # Validate semantics
            result = validate_semantic(
                request.input_text,
                request.reference_text,
                request.module,
                request.custom_threshold
            )
            
            # Add treatment elements to the response
            result["treatment_elements"] = treatment_elements
            
        elif request.module == "SMCC":
            # For synthesized medical care, we handle longer texts
            logger.info("Processing SMCC validation for synthesized medical care")
            
            # Medical care comparison requires more context
            result = validate_semantic(
                request.input_text,
                request.reference_text,
                request.module,
                request.custom_threshold
            )
            
            # No specialized elements for SMCC
            result["treatment_elements"] = None
            
        else:
            # This should never happen due to pydantic validation
            raise HTTPException(
                status_code=400,
                detail=f"Invalid treatment module: {request.module}"
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
            "treatment_elements": result["treatment_elements"]
        }
    
    except ValueError as ve:
        # Handle validation errors
        logger.warning(f"Treatment validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in treatment validation: {str(e)}", exc_info=True)
        medical_logger.log_error(
            module=request.module if hasattr(request, "module") else "unknown",
            error_type="treatment_validation_error",
            error_message=str(e),
            details={"input_length": len(request.input_text) if hasattr(request, "input_text") else 0}
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred during treatment validation"
        )

