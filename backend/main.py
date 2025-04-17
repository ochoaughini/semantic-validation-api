from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Semantic Validation API",
    description="API for validating semantic similarity between texts",
    version="1.0.0"
)

# CORS configuration - Comprehensive list of allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://semantic-validation-api.onrender.com",       # Production on Render
        "http://localhost:3000",                             # Local frontend development
        "http://localhost:5000",                             # Alternative local frontend
        "http://localhost:8080",                             # Local backend
        "http://127.0.0.1:3000",                            # Local IPv4 frontend
        "http://127.0.0.1:5000",                            # Alternative local IPv4
        "http://127.0.0.1:8080",                            # Local IPv4 backend
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

class ValidationRequest(BaseModel):
    input_text: str
    reference_text: str

class ValidationResponse(BaseModel):
    input: str
    reference: str
    match: bool

@app.post("/api/validate", response_model=ValidationResponse)
async def validate_text(request: ValidationRequest):
    try:
        if not request.input_text or not request.reference_text:
            return JSONResponse(
                status_code=400,
                content={"detail": [{"msg": "Both input and reference texts are required"}]}
            )
        
        # Simple semantic validation
        input_clean = request.input_text.lower().strip()
        reference_clean = request.reference_text.lower().strip()
        
        # Compare texts
        match = input_clean == reference_clean
        
        return ValidationResponse(
            input=request.input_text,
            reference=request.reference_text,
            match=match
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": [{"msg": str(e)}]}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Global error handlers for consistency with frontend expectations
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": [{"msg": str(exc.detail)}]}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": [{"msg": "Internal server error"}]}
    )
