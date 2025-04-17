FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy backend files individually with explicit paths
COPY backend/requirements.txt requirements.txt
COPY backend/main.py main.py

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Security settings
USER nobody

# Expose port
EXPOSE 8080

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

