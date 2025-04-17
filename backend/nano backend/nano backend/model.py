from sentence_transformers import SentenceTransformer

# Carregamento único do modelo de embeddings semânticos
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str):
    """
    Gera o embedding para a sentença fornecida.
    """
    return model.encode(text)
