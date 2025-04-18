import time
from typing import Dict, Any, Optional
import os
from fastapi import FastAPI, HTTPException, Depends, Security, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from loguru import logger

# Import our components
from .config import config
from .semantic_service import semantic_service
from .schemas import (
    ValidationRequest, ValidationResponse, ErrorResponse, HealthResponse,
    ModuleType, SubmoduleType
)

# API information
API_VERSION = "1.1.0"
API_TITLE = "Semantic Validation API"
API_DESCRIPTION = "API for validating semantic similarity between texts with advanced NLP models"

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration - Comprehensive list of allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Render domains
        "https://semantic-validation-api.onrender.com",       # Backend on Render
        "https://semantic-validation-app.onrender.com",       # Possible frontend on Render
        "https://semantic-validation.onrender.com",           # Alternative frontend name
        "https://*.onrender.com",                             # All Render subdomains
        
        # Local development
        "http://localhost:3000",                             # Local frontend development
        "http://localhost:5000",                             # Alternative local frontend
        "http://localhost:8080",                             # Local backend
        "http://127.0.0.1:3000",                            # Local IPv4 frontend
        "http://127.0.0.1:5000",                            # Alternative local IPv4
        "http://127.0.0.1:8080",                            # Local IPv4 backend
        
        # Allow frontend to be served from any domain during testing
        "*",                                                # Allow all origins temporarily
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],                # Include OPTIONS for preflight
    allow_headers=["Content-Type", "Authorization", "Origin", "Accept", "X-API-Key"],
)

# Optional API key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Set of valid API keys (for basic auth)
# In production, this should be stored securely
API_KEYS = {
    os.getenv("API_KEY", "dev-api-key-2025")
}

async def get_api_key(
    api_key: str = Security(api_key_header),
    require_auth: bool = True
):
    """Validate API key if authorization is required."""
    # Skip authentication if not required or in development
    if not require_auth or os.getenv("ENVIRONMENT") == "development":
        return None
        
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return api_key


# Helper function to time operation execution
async def timed_operation(func, *args, **kwargs):
    """Execute a function and measure execution time."""
    start_time = time.time()
    result = await func(*args, **kwargs) if callable(func) else func
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # ms
    return result, processing_time


# Main validation endpoint (compatible with existing frontend)
@app.post("/api/validate", response_model=ValidationResponse, response_model_exclude_none=True)
async def validate_text(
    request: ValidationRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Validate semantic similarity between two texts.
    This endpoint maintains compatibility with the existing frontend.
    """
    try:
        if not request.input_text or not request.reference_text:
            return JSONResponse(
                status_code=400,
                content={"detail": [{"msg": "Both input and reference texts are required"}]}
            )
        
        # Advanced semantic validation
        validation_result, processing_time = await timed_operation(
            semantic_service.validate_semantic,
            request.input_text,
            request.reference_text,
            request.module,
            request.submodule,
            request.model_type,
            request.custom_threshold
        )
        
        # Build response with all available information
        return ValidationResponse(
            input=request.input_text,
            reference=request.reference_text,
            similarity=validation_result["similarity"],
            match=validation_result["match"],
            threshold=validation_result["threshold"],
            model=validation_result["model"],
            module=request.module,
            submodule=request.submodule,
            processing_time_ms=processing_time,
            metadata=request.metadata
        )
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": [{"msg": str(e)}]}
        )


# Module-specific endpoints
@app.post("/api/diagnosis/validate", response_model=ValidationResponse, response_model_exclude_none=True)
async def validate_diagnosis(
    request: ValidationRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Validate semantic similarity for diagnosis texts."""
    # Override module to ensure correct domain-specific processing
    request.module = ModuleType.DIAGNOSIS.value
    return await validate_text(request, api_key)


@app.post("/api/treatment/validate", response_model=ValidationResponse, response_model_exclude_none=True)
async def validate_treatment(
    request: ValidationRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Validate semantic similarity for treatment texts."""
    # Override module to ensure correct domain-specific processing
    request.module = ModuleType.TREATMENT.value
    return await validate_text(request, api_key)


@app.post("/api/humanization/validate", response_model=ValidationResponse, response_model_exclude_none=True)
async def validate_humanization(
    request: ValidationRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Validate semantic similarity for humanization texts."""
    # Override module to ensure correct domain-specific processing
    request.module = ModuleType.HUMANIZATION.value
    return await validate_text(request, api_key)


# Enhanced health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and return status information."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=len(semantic_service._model_cache)
    )


# Root endpoint for basic information
@app.get("/")
async def root():
    """Get basic API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs_url": "/docs",
        "health_check": "/health"
    }


# Global error handlers for consistency with frontend expectations
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions and format them consistently."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": [{"msg": str(exc.detail)}]}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions and format them consistently."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": [{"msg": "Internal server error"}]}
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their processing time."""
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms)")
    
    return response


# Startup event to initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Using model: {config.get_model()}")
    
    # Log configured thresholds
    thresholds = config.config.get("thresholds", {})
    logger.info(f"Configured thresholds: {thresholds}")
    
    # Log environment
    environment = os.getenv("ENVIRONMENT", "production")
    logger.info(f"Running in {environment} environment")

