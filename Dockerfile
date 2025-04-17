FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for sentence-transformers and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Environment settings
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV EMBEDDING_MODEL="all-MiniLM-L6-v2"
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Create cache directories with correct permissions
RUN mkdir -p /app/.cache/transformers /app/.cache/sentence_transformers

# Copy requirements first and install dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY backend/config.yaml /app/config.yaml
COPY backend/config.py /app/config.py
COPY backend/schemas.py /app/schemas.py
COPY backend/semantic_service.py /app/semantic_service.py
COPY backend/main.py /app/main.py

# Pre-download models with error handling
RUN pip install --no-cache-dir "sentence-transformers>=2.2.2" && \
    python -c "import os; os.environ['TOKENIZERS_PARALLELISM'] = 'false'; \
    from sentence_transformers import SentenceTransformer; \
    try: \
        model = SentenceTransformer('all-MiniLM-L6-v2'); \
        print('Model downloaded successfully!'); \
    except Exception as e: \
        print(f'Warning: Model pre-download failed: {e}'); \
        print('Will attempt to download at runtime');"

# Set permissions for application and cache dirs
RUN chown -R nobody:nogroup /app && \
    chmod -R 755 /app

# Switch to non-root user for security
USER nobody

# Expose port
EXPOSE ${PORT}

# Add curl for health check
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER nobody

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]

