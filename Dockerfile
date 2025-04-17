FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Environment settings
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Copy all application code
COPY backend/config.yaml /app/config.yaml
COPY backend/config.py /app/config.py
COPY backend/schemas.py /app/schemas.py
COPY backend/semantic_service.py /app/semantic_service.py
COPY backend/main.py /app/main.py

# Pre-download models (optional, helps reduce first request latency)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create a directory for the non-root user and adjust permissions
RUN mkdir -p /app/.cache && chown -R nobody:nogroup /app

# Set correct permissions
RUN chmod -R 755 /app

# Security settings
USER nobody

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 2"]

