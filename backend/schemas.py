from typing import Dict, Any, Optional, List
from pydantic import BaseModel, validator, Field, constr
import bleach
from enum import Enum


class ModuleType(str, Enum):
    """Supported module types for semantic validation."""
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    HUMANIZATION = "humanization"
    GENERAL = "general"


class SubmoduleType(str, Enum):
    """Supported submodule types for semantic validation."""
    # Diagnosis submodules
    AMA = "AMA"         # Accuracy of Medical Assessment
    AI_MPN = "AI-MPN"   # Alignment with Physician Notes
    
    # Treatment submodules
    TDBE = "TDBE"       # Treatment Description Basic Elements
    SMCC = "SMCC"       # Synthesized Medical Care Comparison
    
    # Humanization submodules
    ICSE = "ICSE"       # Interpersonal Communication Style Equivalence
    OIFC = "OIFC"       # Overall Information Fidelity Check
    
    # General
    DEFAULT = "default"


class ValidationRequest(BaseModel):
    """Request model for semantic validation."""
    input_text: constr(min_length=1, max_length=5000) = Field(..., description="Text to be validated")
    reference_text: constr(min_length=1, max_length=5000) = Field(..., description="Reference text to compare against")
    module: Optional[str] = Field(None, description="Module type for domain-specific validation")
    submodule: Optional[str] = Field(None, description="Submodule type for specific threshold")
    model_type: Optional[str] = Field(None, description="Type of model to use (default, clinical, etc.)")
    custom_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Custom threshold to override configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for tracking or analytics")
    
    @validator('input_text', 'reference_text', pre=True)
    def sanitize_text(cls, v):
        """Sanitize text inputs to prevent HTML/script injection."""
        if isinstance(v, str):
            # Remove HTML tags and sanitize
            return bleach.clean(v, tags=[], strip=True)
        return v
    
    @validator('module')
    def validate_module(cls, v):
        """Validate the module value."""
        if v is not None and v not in [m.value for m in ModuleType]:
            valid_modules = ", ".join([m.value for m in ModuleType])
            raise ValueError(f"Invalid module. Must be one of: {valid_modules}")
        return v
    
    @validator('submodule')
    def validate_submodule(cls, v, values):
        """Validate the submodule based on the selected module."""
        if v is None:
            return v
            
        module = values.get('module')
        
        # If no module specified, any submodule is valid
        if module is None:
            return v
            
        # Check submodule validity for each module
        valid_submodules = []
        if module == ModuleType.DIAGNOSIS.value:
            valid_submodules = [SubmoduleType.AMA.value, SubmoduleType.AI_MPN.value]
        elif module == ModuleType.TREATMENT.value:
            valid_submodules = [SubmoduleType.TDBE.value, SubmoduleType.SMCC.value]
        elif module == ModuleType.HUMANIZATION.value:
            valid_submodules = [SubmoduleType.ICSE.value, SubmoduleType.OIFC.value]
        elif module == ModuleType.GENERAL.value:
            valid_submodules = [SubmoduleType.DEFAULT.value]
            
        if v not in valid_submodules:
            valid_str = ", ".join(valid_submodules)
            raise ValueError(f"Invalid submodule for {module}. Must be one of: {valid_str}")
            
        return v


class ValidationResponse(BaseModel):
    """Response model for semantic validation."""
    input: str = Field(..., description="Input text that was validated")
    reference: str = Field(..., description="Reference text used for comparison")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score between 0 and 1")
    match: bool = Field(..., description="Whether the similarity meets the threshold")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold used for comparison")
    model: str = Field(..., description="Model used for generating embeddings")
    module: Optional[str] = Field(None, description="Module used for validation")
    submodule: Optional[str] = Field(None, description="Submodule used for validation")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata from the request")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: List[Dict[str, Any]] = Field(..., description="List of error details")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of models loaded")

