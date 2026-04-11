# ── Stage 1: Build + download model ──────────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

WORKDIR /app

# Install git (needed for pip git+ installs)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install Python deps (non-torch — torch already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/ChaoningZhang/MobileSAM.git

# Try EfficientViT-SAM (optional — needs scipy/meson which may fail in some base images)
RUN pip install --no-cache-dir scipy && \
    pip install --no-cache-dir git+https://github.com/mit-han-lab/efficientvit.git || \
    echo "WARNING: EfficientViT-SAM install failed — only MobileSAM will be available"

# Pre-download Florence-2-base model + processor (~500MB)
# This bakes the model into the image so startup is instant
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoProcessor; \
AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); \
AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); \
print('Model cached successfully')"

# Pre-download MobileSAM weights (~10MB)
RUN mkdir -p /app/weights && python -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
p = hf_hub_download(repo_id='dhkim2810/MobileSAM', filename='mobile_sam.pt'); \
shutil.copy2(p, '/app/weights/mobile_sam.pt'); \
print('MobileSAM weights cached')"

# Pre-download EfficientViT-SAM-L0 weights (~140MB) — only if package installed
RUN python -c "exec('try:\\n from huggingface_hub import hf_hub_download\\n import shutil\\n p = hf_hub_download(repo_id=\"han-cai/efficientvit-sam\", filename=\"l0.pt\")\\n shutil.copy2(p, \"/app/weights/efficientvit_sam_l0.pt\")\\n print(\"EfficientViT-SAM weights cached\")\\nexcept Exception as e:\\n print(f\"Skipping EfficientViT weights: {e}\")')"

# Copy application code
COPY server.py .
COPY segmentor.py .
COPY pwa/ ./pwa/
COPY index.html .

EXPOSE 8100

CMD ["python", "server.py"]
