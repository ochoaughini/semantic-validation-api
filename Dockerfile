FROM python:3.11-slim

WORKDIR /app

# Set build context to backend directory
WORKDIR /app/backend

# Copy backend files with correct paths
COPY backend/requirements.txt backend/main.py ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Security settings
USER nobody

# Expose port
EXPOSE 8080

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

