# ── Stage 1: Build + download model ──────────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

WORKDIR /app

# Install git (needed for pip git+ installs)
RUN apt-get update && apt-get install -y --no-install-recommends git libxcb1 libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install Python deps (non-torch — torch already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/ChaoningZhang/MobileSAM.git

# Try SAM 2 (optional — for higher-quality segmentation)
# Use --no-deps to prevent sam2 from upgrading torch to an incompatible CUDA version
RUN pip install --no-cache-dir --no-deps sam2 || \
    echo "WARNING: SAM 2 install failed — only MobileSAM will be available"

# Force torch back to base image version (sam2/other deps may have upgraded it)
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

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

# Pre-download Grounding-DINO SwinT weights (~593MB)
RUN mkdir -p /app/weights && python -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
p = hf_hub_download(repo_id='ShilongLiu/GroundingDINO', filename='groundingdino_swint_ogc.pth'); \
shutil.copy2(p, '/app/weights/groundingdino_swint_ogc.pth'); \
print('GDINO SwinT weights cached')"

# Pre-download SAM 2 Tiny weights (~39MB) — only if package installed
RUN python -c "exec('try:\n from sam2.sam2_image_predictor import SAM2ImagePredictor\n SAM2ImagePredictor.from_pretrained(\"facebook/sam2.1-hiera-tiny\")\n print(\"SAM 2 Tiny weights cached\")\nexcept Exception as e:\n print(f\"Skipping SAM 2 weights: {e}\")')"

# Copy application code
COPY server.py .
COPY segmentor.py .
COPY gdino_detector.py .
COPY household_items.json .
COPY pwa/ ./pwa/
COPY index.html .

EXPOSE 8100

CMD ["python", "server.py"]
