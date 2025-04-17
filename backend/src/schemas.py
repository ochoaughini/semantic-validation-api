from typing import Dict, Any, Optional, List
from pydantic import BaseModel, constr, validator, Field
import bleach
import re

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
        valid_modules = ["AMA", "AI-MPN", "TDBE", "SMCC", "ICSE", "OIFC"]
        if v not in valid_modules:
            raise ValueError(f"Invalid module. Must be one of: {', '.join(valid_modules)}")
        return v

class ValidationResponse(BaseModel):
    """
    Base response model for validation endpoints.
    """
    similarity: float = Field(..., description="Similarity score between 0 and 1")
    match: bool = Field(..., description="Whether the similarity meets the threshold")

class ErrorDetail(BaseModel):
    """
    Model for error details.
    """
    msg: str = Field(..., description="Error message")
    loc: Optional[List[str]] = Field(None, description="Error location")
    type: Optional[str] = Field(None, description="Error type")

class ErrorResponse(BaseModel):
    """
    Model for API error responses.
    """
    detail: List[ErrorDetail] = Field(..., description="List of error details")

class HealthCheck(BaseModel):
    """
    Model for health check response.
    """
    status: str = Field(..., description="Status of the API")
    version: str = Field(..., description="API version")

