from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, constr, validator, Field
import bleach
import re


class ModuleEnum(str, Enum):
    """Valid module types for semantic validation."""
    # Diagnosis modules
    AMA = "AMA"         # Accuracy of Medical Assessment
    AI_MPN = "AI-MPN"   # Alignment with Physician Notes
    
    # Treatment modules
    TDBE = "TDBE"       # Treatment Description Basic Elements
    SMCC = "SMCC"       # Synthesized Medical Care Comparison
    
    # Humanization modules
    ICSE = "ICSE"       # Interpersonal Communication Style Equivalence
    OIFC = "OIFC"       # Overall Information Fidelity Check

class ValidationRequest(BaseModel):
    """
    Base request model for validation endpoints.
    """
    input_text: constr(min_length=1, max_length=2000) = Field(
        ..., description="Text to be validated"
    )
    reference_text: constr(min_length=1, max_length=2000) = Field(
        ..., description="Reference text to compare against"
    )
    module: str = Field(..., description="Validation module")
    custom_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, 
                                             description="Optional custom threshold (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for tracking")
    
    @validator("input_text", "reference_text", pre=True)
    def sanitize_text(cls, v):
        """Sanitize text inputs to prevent HTML/script injection."""
        if isinstance(v, str):
            # Remove HTML tags and sanitize
            return bleach.clean(v, tags=[], strip=True)
        return v
    
    @validator("module")
    def validate_module(cls, v):
        """Validate that the module is supported."""
        try:
            # Convert string to enum to validate
            module_enum = ModuleEnum(v)
            return v
        except ValueError:
            valid_modules = [m.value for m in ModuleEnum]
            raise ValueError(f"Invalid module. Must be one of: {', '.join(valid_modules)}")

class ValidationResponse(BaseModel):
    """
    Base response model for validation endpoints.
    """
    input: str = Field(..., description="Input text that was validated")
    reference: str = Field(..., description="Reference text used for comparison")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score between 0 and 1")
    match: bool = Field(..., description="Whether the similarity meets the threshold")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold used for comparison")
    model: str = Field(..., description="Model used for generating embeddings")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class ErrorDetail(BaseModel):
    """
    Model for error details.
    """
    msg: str = Field(..., description="Error message")
    loc: Optional[List[str]] = Field(None, description="Error location")
    type: Optional[str] = Field(None, description="Error type")

class MetricsResponse(BaseModel):
    """
    Response model for quality metrics.
    """
    modules: Dict[str, Dict[str, Any]] = Field(..., description="Metrics per module")
    domains: Dict[str, Dict[str, float]] = Field(..., description="Metrics per domain")
    overall: Dict[str, Any] = Field(..., description="Overall metrics")


class ErrorDetail(BaseModel):
    """
    Model for error details.
    """
    msg: str = Field(..., description="Error message")
    loc: Optional[List[str]] = Field(None, description="Error location")
    type: Optional[str] = Field(None, description="Error type")
    """
    Model for health check response.
    """
    status: str = Field(..., description="Status of the API")
    version: str = Field(..., description="API version")

