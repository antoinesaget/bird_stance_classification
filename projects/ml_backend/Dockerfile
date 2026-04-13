# Purpose: Build the production ML backend image used on the iats host.
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

COPY projects/ml_backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ops/src /app/ops/src
COPY shared/birdsys_core/src /app/shared/birdsys_core/src
COPY projects/ml_backend/src /app/projects/ml_backend/src

ENV PYTHONPATH=/app/ops/src:/app/shared/birdsys_core/src:/app/projects/ml_backend/src
ENV MODEL_A_WEIGHTS=/data/birds_project/models/detector/yolo11m_pretrained/weights.pt

EXPOSE 9090

CMD ["uvicorn", "birdsys.ml_backend.app.main:app", "--host", "0.0.0.0", "--port", "9090"]
