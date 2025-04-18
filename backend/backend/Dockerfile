# Use official Python 3.11 image (most stable with our dependencies)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# Set work directory
WORKDIR /app

# Install system dependencies required for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copy project
COPY . .

# Expose the port
EXPOSE ${PORT}

# Run application through gunicorn with uvicorn workers
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "--worker-class", "uvicorn.workers.UvicornWorker", "src.main:app"]

