from fastapi import FastAPI
import numpy as np
from model import embed_text
from schemas import ValidationRequest

app = FastAPI()

# Limiar padrão de similaridade (75%)
DEFAULT_THRESHOLD = 0.75

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/api/validate")
def validate_texts(req: ValidationRequest, threshold: float = DEFAULT_THRESHOLD):
    # 1. Gerar embeddings semânticos
    emb1 = embed_text(req.input_text)
    emb2 = embed_text(req.reference_text)
    # 2. Calcular similaridade de cosseno
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    similarity = float(dot / norm)
    # 3. Verificar match com base no limiar
    match = similarity >= threshold
    # 4. Retornar resposta JSON
    return {
        "input_text": req.input_text,
        "reference_text": req.reference_text,
        "similarity": similarity,
        "match": match
    }
