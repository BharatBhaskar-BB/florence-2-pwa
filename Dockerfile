# ── Stage 1: Build + download model ──────────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

WORKDIR /app

# Install Python deps (non-torch — torch already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/mit-han-lab/efficientvit.git

# Pre-download Florence-2-base model + processor (~500MB)
# This bakes the model into the image so startup is instant
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoProcessor; \
AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); \
AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); \
print('Model cached successfully')"

# Copy application code
COPY server.py .
COPY segmentor.py .
COPY pwa/ ./pwa/
COPY index.html .

EXPOSE 8100

CMD ["python", "server.py"]
