from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class ValidationRequest(BaseModel):
    input_text: str
    reference_text: str

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/api/validate")
def validate_texts(request: ValidationRequest):
    # lógica simples comparativa
    return {
        "input": request.input_text,
        "reference": request.reference_text,
        "match": request.input_text.lower() in request.reference_text.lower()
    }
