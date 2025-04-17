from typing import Dict, Any, Optional
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..auth import get_api_key
from ..semantic_service import validate_semantic
from ..logging_config import logger, medical_logger

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/diagnosis",
    tags=["diagnosis"],
    dependencies=[Depends(get_api_key)]
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# --- Models --- #

class DiagnosisRequest(BaseModel):
    """Request model for diagnosis validation."""
    input_text: str = Field(..., min_length=1, max_length=2000, 
                            description="Diagnosis text to validate")
    reference_text: str = Field(..., min_length=1, max_length=2000, 
                                description="Reference diagnosis text to compare against")
    module: str = Field(..., description="Validation module: AMA or AI-MPN")
    custom_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, 
                                              description="Optional custom threshold (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    
    @validator("module")
    def validate_module(cls, v):
        """Ensure module is a valid diagnosis module."""
        if v not in ["AMA", "AI-MPN"]:
            raise ValueError(f"Invalid module for diagnosis. Must be 'AMA' or 'AI-MPN', got '{v}'")
        return v
    
    @validator("input_text", "reference_text")
    def validate_texts(cls, v):
        """Ensure texts are not too long or empty."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class DiagnosisResponse(BaseModel):
    """Response model for diagnosis validation."""
    input: str = Field(..., description="Input diagnosis text")
    reference: str = Field(..., description="Reference diagnosis text")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    match: bool = Field(..., description="Whether the texts match semantically")
    threshold: float = Field(..., description="Threshold used for matching")
    module: str = Field(..., description="Module used for validation")
    model: str = Field(..., description="Model used for validation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

# --- Routes --- #

@router.post("/validate", response_model=DiagnosisResponse)
@limiter.limit("50/minute")
async def validate_diagnosis(request: DiagnosisRequest, req: Request):
    """
    Validate medical diagnosis text against a reference.
    
    - **AMA** (Accuracy of Medical Assessment): Validates accuracy of general diagnostic text
    - **AI-MPN** (Alignment with Physician Notes): Specific for aligning with professional notes
    """
    try:
        start_time = time.time()
        logger.info(f"Diagnosis validation request: module={request.module}")
        
        # Domain-specific preprocessing and validation
        if request.module == "AMA":
            # For general medical assessment, we use the default validation
            result = validate_semantic(
                request.input_text,
                request.reference_text,
                request.module,
                request.custom_threshold
            )
        elif request.module == "AI-MPN":
            # For physician note alignment, we use the clinical model
            logger.info("Using clinical model for AI-MPN validation")
            result = validate_semantic(
                request.input_text,
                request.reference_text,
                request.module,
                request.custom_threshold
            )
        else:
            # This should never happen due to pydantic validation, but as a safeguard
            raise HTTPException(
                status_code=400,
                detail=f"Invalid diagnosis module: {request.module}"
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
            "processing_time_ms": processing_time
        }
    
    except ValueError as ve:
        # Handle validation errors
        logger.warning(f"Diagnosis validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in diagnosis validation: {str(e)}", exc_info=True)
        medical_logger.log_error(
            module=request.module if hasattr(request, "module") else "unknown",
            error_type="diagnosis_validation_error",
            error_message=str(e),
            details={"input_length": len(request.input_text) if hasattr(request, "input_text") else 0}
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred during diagnosis validation"
        )

