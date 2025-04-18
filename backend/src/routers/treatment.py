from typing import Dict, Any, Optional, List
import time
import re
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

# Medical patterns compiled for performance
DRUG_FORMS = [
    "tablet", "capsule", "injection", "solution", "suspension", "pill",
    "cream", "ointment", "patch", "inhaler", "spray", "drops"
]
MEDICAL_PATTERNS = {
    'dosages': re.compile(r'\b\d+\.?\d*\s*(mg|g|ml|l|mmol|µg|mcg|IU|mEq)\b'),
    'timing': re.compile(r'\b(bid|tid|qid|daily|weekly|monthly|hourly|prn|as needed)\b'),
    'duration': re.compile(r'\b(\d+\s*(day|week|month|year)s?)\b'),
    'route': re.compile(r'\b(oral|iv|intravenous|im|intramuscular|sc|subcutaneous|topical|inhaled)\b', re.IGNORECASE),
    'drug_forms': re.compile(r'\b(' + '|'.join(DRUG_FORMS) + r')\b', re.IGNORECASE)
}

from ..auth import get_api_key
from ..semantic_service import validate_semantic
from ..logging_config import logger, medical_logger
from ..config import config

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/treatment",
    tags=["treatment"],
    dependencies=[Depends(get_api_key)]
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# --- Models --- #
class DrugInteraction(BaseModel):
    """Model for drug interaction details."""
    drug1: str = Field(..., description="First drug in interaction")
    drug2: str = Field(..., description="Second drug in interaction")
    type: str = Field(..., description="Type of interaction")


class TreatmentElements(BaseModel):
    """Model for extracted treatment elements."""
    dosages: List[str] = Field(default_factory=list, description="Dosage measurements")
    timing: List[str] = Field(default_factory=list, description="Timing instructions")
    duration: List[str] = Field(default_factory=list, description="Duration specifications")
    procedures: List[str] = Field(default_factory=list, description="Medical procedures")
    route: List[str] = Field(default_factory=list, description="Routes of administration")
    drug_forms: List[str] = Field(default_factory=list, description="Drug formulations")
    interactions: List[DrugInteraction] = Field(default_factory=list, description="Potential drug interactions")


class TreatmentElementsContext(BaseModel):
    """Model for treatment elements context."""
    input: TreatmentElements = Field(..., description="Elements from input text")
    reference: TreatmentElements = Field(..., description="Elements from reference text")


class TreatmentRequest(BaseModel):
    """Request model for treatment validation."""
    input_text: str = Field(..., min_length=1, max_length=2000, 
                          description="Treatment text to validate")
    reference_text: str = Field(..., min_length=1, max_length=2000, 
                              description="Reference treatment text to compare against")
    module: str = Field(..., description="Validation module (TDBE or SMCC)")
    submodule: Optional[str] = Field(None, description="Optional specific submodule")
    custom_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, 
                                            description="Optional custom threshold (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    @validator("module")
    def validate_module(cls, v):
        """Ensure module is a valid treatment module."""
        v = v.upper()  # Normalize input
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
    treatment_elements: Optional[TreatmentElementsContext] = Field(
        None, 
        description="Extracted treatment elements from both texts"
    )

# --- Helper Functions --- #

def check_drug_interactions(text: str) -> List[Dict[str, str]]:
    """
    Check for potential drug interactions in treatment text.
    
    Args:
        text: Treatment text to analyze
        
    Returns:
        List of potential drug interactions
    """
    interactions = []
    
    # Get medical terms from config
    medical_terms = config.get_medical_abbreviations()
    if not medical_terms:
        return interactions
        
    # Look for co-occurring medications
    drugs = set()
    for term, expansion in medical_terms.items():
        if "drug" in term.lower() or "medication" in term.lower():
            if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                drugs.add(term)
                
    # If multiple drugs found, flag for potential interaction
    if len(drugs) > 1:
        drug_list = sorted(list(drugs))
        for i, drug1 in enumerate(drug_list):
            for drug2 in drug_list[i+1:]:
                interactions.append({
                    "drug1": drug1,
                    "drug2": drug2,
                    "type": "co_administration"
                })
                
    return interactions


def extract_treatment_elements(text: str) -> Dict[str, List[str]]:
    """
    Extract structured information from treatment text.
    
    Args:
        text: Treatment text to analyze
        
    Returns:
        Dictionary of treatment elements extracted from text
    """
    elements = {
        "dosages": [],
        "timing": [],
        "duration": [],
        "procedures": [],
        "route": [],
        "drug_forms": [],
        "interactions": []  # Add interactions field
    }
    
    try:
        # Extract using compiled patterns
        for key, pattern in MEDICAL_PATTERNS.items():
            elements[key] = [m.group() for m in pattern.finditer(text)]
            
        # Extract procedures using medical terms
        medical_terms = config.get_medical_abbreviations()
        if medical_terms:
            procedure_terms = [
                term for term, _ in medical_terms.items() 
                if any(x in term.lower() for x in ["procedure", "therapy", "treatment"])
            ]
            if procedure_terms:
                procedure_pattern = re.compile(
                    r'\b(' + '|'.join(re.escape(p) for p in procedure_terms) + r')\b',
                    re.IGNORECASE
                )
                elements["procedures"] = [m.group() for m in procedure_pattern.finditer(text)]
        
        # Check for drug interactions
        elements["interactions"] = check_drug_interactions(text)
            
        return elements
    except Exception as e:
        logger.error(f"Error extracting treatment elements: {str(e)}")
        return elements  # Return empty elements on error


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
        
        # Extract treatment elements
        input_elements = extract_treatment_elements(request.input_text)
        reference_elements = extract_treatment_elements(request.reference_text)
        
        # Use semantic validation
        result = validate_semantic(
            request.input_text,
            request.reference_text,
            module=request.module,
            submodule=request.submodule,
            custom_threshold=request.custom_threshold
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Construct response
        return TreatmentResponse(
            input=request.input_text,
            reference=request.reference_text,
            similarity=result["similarity"],
            match=result["match"],
            threshold=result["threshold"],
            module=request.module,
            model=result["model"],
            processing_time_ms=processing_time,
            treatment_elements=TreatmentElementsContext(
                input=TreatmentElements(**input_elements),
                reference=TreatmentElements(**reference_elements)
            )
        )
        
    except ValueError as ve:
        # Handle validation errors
        logger.warning(f"Treatment validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Treatment validation error: {str(e)}", exc_info=True)
        medical_logger.log_error(
            module=request.module if hasattr(request, "module") else "unknown",
            error_type="treatment_validation_error",
            error_message=str(e),
            details={
                "input_length": len(request.input_text) if hasattr(request, "input_text") else 0,
                "submodule": request.submodule if hasattr(request, "submodule") else None
            }
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred during treatment validation"
        )

