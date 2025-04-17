FROM python:3.11-slim

WORKDIR /app

COPY .. .

RUN pip install --upgrade pip
RUN pip install -r backend/requirements.txt

CMD ["uvicorn", "backend.main:app", "--host", "0.0
