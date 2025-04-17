FROM python:3.11-slim

WORKDIR /app

# Create backend directory and set it as working directory
WORKDIR /app/backend

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/main.py ./main.py

# Run with proper security settings
USER nobody

# Expose port
EXPOSE 8080

# Start the application with proper settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

