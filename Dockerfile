FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

RUN mkdir -p data/sample models

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000 8501

CMD ["python", "-m", "src.main"]
