from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, constr, validator, Field
import bleach
import re


class ModuleType(str, Enum):
    """High-level module categories for semantic validation."""
    DIAGNOSIS = "diagnosis"       # Medical diagnosis validation
    TREATMENT = "treatment"       # Treatment plan validation
    HUMANIZATION = "humanization" # Patient communication validation


class SubmoduleType(str, Enum):
    """Specific submodules for semantic validation."""
    # Diagnosis submodules
    AMA = "AMA"         # Accuracy of Medical Assessment
    AI_MPN = "AI-MPN"   # Alignment with Physician Notes
    
    # Treatment submodules
    TDBE = "TDBE"       # Treatment Description Basic Elements
    SMCC = "SMCC"       # Synthesized Medical Care Comparison
    
    # Humanization submodules
    ICSE = "ICSE"       # Interpersonal Communication Style Equivalence
    OIFC = "OIFC"       # Overall Information Fidelity Check


# Module to submodule mapping for validation
MODULE_SUBMODULE_MAP = {
    ModuleType.DIAGNOSIS: [SubmoduleType.AMA, SubmoduleType.AI_MPN],
    ModuleType.TREATMENT: [SubmoduleType.TDBE, SubmoduleType.SMCC],
    ModuleType.HUMANIZATION: [SubmoduleType.ICSE, SubmoduleType.OIFC],
}


# Keep ModuleEnum for backward compatibility
class ModuleEnum(str, Enum):
    """Valid module types for semantic validation (legacy, use SubmoduleType instead)."""
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
    module: str = Field(..., description="Validation module or submodule")
    submodule: Optional[str] = Field(None, description="Optional specific submodule")
    model_type: Optional[str] = Field(None, description="Optional specific model type to use")
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
            # First try as a high-level module
            try:
                ModuleType(v)
                return v
            except ValueError:
                # Then try as a submodule
                SubmoduleType(v)
                return v
        except ValueError:
            valid_modules = [m.value for m in ModuleType] + [m.value for m in SubmoduleType]
            raise ValueError(f"Invalid module. Must be one of: {', '.join(valid_modules)}")

    @validator("submodule", always=True)
    def validate_submodule(cls, v, values):
        """Validate that the submodule is valid for the given module."""
        if not v:
            return v
            
        # If main module is a high-level module, check submodule compatibility
        if "module" in values:
            try:
                # First try if the main module is a high-level module
                module = values["module"]
                if module in [m.value for m in ModuleType]:
                    module_type = ModuleType(module)
                    valid_submodules = [sm.value for sm in MODULE_SUBMODULE_MAP[module_type]]
                    if v not in valid_submodules:
                        raise ValueError(f"Invalid submodule '{v}' for module '{module}'. Valid options: {', '.join(valid_submodules)}")
            except (KeyError, ValueError):
                # Module might be a submodule already
                pass
                
        # Validate that the submodule exists
        try:
            SubmoduleType(v)
        except ValueError:
            valid_submodules = [m.value for m in SubmoduleType]
            raise ValueError(f"Invalid submodule. Must be one of: {', '.join(valid_submodules)}")
            
        return v

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
    module: str = Field(..., description="Module used for validation")
    submodule: Optional[str] = Field(None, description="Submodule used for validation")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata returned from processing")

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


class ErrorResponse(BaseModel):
    """
    Model for error responses.
    """
    detail: List[ErrorDetail] = Field(..., description="Error details")


class HealthResponse(BaseModel):
    """
    Model for health check response.
    """
    status: str = Field(..., description="Status of the API")
    version: str = Field(..., description="API version")
    models_loaded: Optional[int] = Field(None, description="Number of loaded models")

