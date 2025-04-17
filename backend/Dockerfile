FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./main.py .

# Run with proper security settings
USER nobody

# Expose port
EXPOSE 8080

# Start the application with proper settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
