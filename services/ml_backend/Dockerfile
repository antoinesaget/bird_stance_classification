FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/ml_backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src
COPY services /app/services

ENV PYTHONPATH=/app/src:/app
ENV MODEL_A_WEIGHTS=/data/birds_project/models/detector/yolo11m_pretrained/weights.pt

EXPOSE 9090

CMD ["uvicorn", "services.ml_backend.app.main:app", "--host", "0.0.0.0", "--port", "9090"]
