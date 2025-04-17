FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for sentence-transformers and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    cmake \
    pkg-config \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Environment settings
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV EMBEDDING_MODEL="all-MiniLM-L6-v2"
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Make cache directories accessible to nobody user
RUN mkdir -p /app/.cache/transformers /app/.cache/sentence_transformers && \
    chmod -R 777 /app/.cache

# Copy requirements first (split installations for better caching)
COPY backend/requirements.txt /app/requirements.txt

# Install dependencies in separate steps for better layer caching
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.25.2 

RUN pip install --no-cache-dir pydantic==1.10.7 fastapi==0.95.2 uvicorn==0.22.0

RUN pip install --no-cache-dir PyYAML==6.0.1 python-dotenv==1.0.0 loguru==0.7.0 bleach==6.0.0

# Install sentence-transformers separately (don't download models yet)
RUN pip install --no-cache-dir "sentence-transformers>=2.2.2" "scikit-learn>=1.2.2" psutil

# Create necessary directory structure
RUN mkdir -p /app/src/routers /app/src/static

# Copy application code
COPY backend/src/config.yaml /app/src/config.yaml
COPY backend/src/config.py /app/src/config.py
COPY backend/src/schemas.py /app/src/schemas.py
COPY backend/src/semantic_service.py /app/src/semantic_service.py
COPY backend/src/auth.py /app/src/auth.py
COPY backend/src/logging_config.py /app/src/logging_config.py
COPY backend/main.py /app/main.py

# Copy router files
COPY backend/src/routers/ /app/src/routers/

# Copy static files
COPY backend/src/static/ /app/src/static/

# Create a startup script that downloads the model if needed
RUN echo '#!/bin/bash\n\
echo "Starting Semantic Validation API..."\n\
echo "Model will be downloaded on first request"\n\
echo "Running as user: $(id)"\n\
echo "Working directory: $(pwd)"\n\
echo "Cache directory permissions: $(ls -la /app/.cache)"\n\
echo "Directory structure:"\n\
echo "$(ls -la /app/src)"\n\
exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set permissions for application and cache dirs
RUN chown -R nobody:nogroup /app && \
    chmod -R 755 /app

# Switch to non-root user for security
USER nobody

# Expose port
EXPOSE ${PORT}

# Health check with longer start period to account for model download
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the application with the startup script
CMD ["/app/start.sh"]

