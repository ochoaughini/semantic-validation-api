FROM python:3.11-slim

WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the main application file
COPY main.py /app/

# Security settings
USER nobody

# Expose the port
EXPOSE 8080

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

