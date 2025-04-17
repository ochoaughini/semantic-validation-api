FROM python:3.11-slim

WORKDIR /app

# Copy requirements first and install dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/main.py /app/main.py

# Security settings
USER nobody

# Environment variable for port
ENV PORT=8080

# Expose port
EXPOSE 8080

# Start the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 4

