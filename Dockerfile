FROM python:3.13-slim

WORKDIR /app
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py config.py db.py embedding.py utils.py export_model_to_onnx.py ./
# Copy pre-exported ONNX model if available
COPY onnx_model/ ./onnx_model/ 2>/dev/null || true
ENV ONNX_MODEL_PATH=/app/onnx_model
CMD ["python", "server.py"]
