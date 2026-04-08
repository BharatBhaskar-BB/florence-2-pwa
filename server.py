"""
Florence-2 Detection Server
============================
FastAPI backend serving Florence-2-base for object detection
and Gemini 2.0 Flash for batch inventory.
"""

import base64
import io
import json
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor

# ── Load Gemini API key ───────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    # Try loading from GDINO project .env
    env_path = os.path.expanduser("~/GDINO_LLM_PRIMING/backend/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("GEMINI_API_KEY="):
                GEMINI_API_KEY = line.strip().split("=", 1)[1]
                break

if GEMINI_API_KEY:
    print(f"Gemini API key loaded ({GEMINI_API_KEY[:8]}...)")
else:
    print("WARNING: No GEMINI_API_KEY found — /api/inventory will fail")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
print(f"Gemini model: {GEMINI_MODEL}")

# ── Model Loading ─────────────────────────────────────────────────────────────

MODEL_ID = "microsoft/Florence-2-base"

print(f"Loading {MODEL_ID}...")
t0 = time.time()

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=DTYPE
).to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Loaded in {time.time() - t0:.1f}s on {DEVICE} ({DTYPE})")

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="Florence-2 Detection Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    image: str  # base64-encoded JPEG
    task: str = "<OD>"  # <OD> or <DENSE_REGION_CAPTION>


class DetectResponse(BaseModel):
    bboxes: list[list[float]]
    labels: list[str]
    polygons: list[list[list[float]]] | None = None  # per-object polygon points
    time_ms: float
    image_width: int
    image_height: int


@app.post("/api/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    # Decode image
    img_bytes = base64.b64decode(req.image)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = image.size

    # Run inference
    t0 = time.time()
    inputs = processor(text=req.task, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs.pop("attention_mask", None)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024, num_beams=1, do_sample=False
        )

    result_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(result_text, task=req.task, image_size=image.size)
    elapsed = (time.time() - t0) * 1000

    task_result = parsed.get(req.task, {})
    bboxes = task_result.get("bboxes", [])
    labels = task_result.get("labels", [])

    return DetectResponse(
        bboxes=bboxes,
        labels=labels,
        polygons=None,
        time_ms=round(elapsed, 1),
        image_width=w,
        image_height=h,
    )


def run_inference(image, task_text):
    """Run a single Florence-2 inference and return parsed result."""
    inputs = processor(text=task_text, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs.pop("attention_mask", None)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024, num_beams=1, do_sample=False
        )
    result_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    return processor.post_process_generation(result_text, task=task_text, image_size=image.size)


@app.post("/detect_segment", response_model=DetectResponse)
async def detect_segment(req: DetectRequest):
    """OD first, then REGION_TO_SEGMENTATION for each detected box."""
    img_bytes = base64.b64decode(req.image)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = image.size

    t0 = time.time()

    # Step 1: Object detection
    od_parsed = run_inference(image, "<OD>")
    od_result = od_parsed.get("<OD>", {})
    bboxes = od_result.get("bboxes", [])
    labels = od_result.get("labels", [])

    # Step 2: Segmentation for each box
    polygons = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # Florence-2 expects location tokens: <loc_X> where X = coord * 999 / max_dim
        loc_x1 = int(x1 * 999 / w)
        loc_y1 = int(y1 * 999 / h)
        loc_x2 = int(x2 * 999 / w)
        loc_y2 = int(y2 * 999 / h)
        seg_task = f"<REGION_TO_SEGMENTATION>"
        seg_text = f"{seg_task}<loc_{loc_x1}><loc_{loc_y1}><loc_{loc_x2}><loc_{loc_y2}>"

        try:
            seg_parsed = run_inference(image, seg_text)
            seg_result = seg_parsed.get(seg_task, seg_parsed.get("<REGION_TO_SEGMENTATION>", {}))
            if "polygons" in seg_result and seg_result["polygons"]:
                # Each polygon is a list of [x, y] points
                poly = seg_result["polygons"][0]  # take first polygon
                # Convert flat list to [[x,y], [x,y], ...] if needed
                if poly and not isinstance(poly[0], list):
                    poly = [[poly[j], poly[j+1]] for j in range(0, len(poly) - 1, 2)]
                polygons.append(poly)
            else:
                polygons.append([])
        except Exception as e:
            print(f"Segmentation failed for box {bbox}: {e}")
            polygons.append([])

    elapsed = (time.time() - t0) * 1000

    return DetectResponse(
        bboxes=bboxes,
        labels=labels,
        polygons=polygons,
        time_ms=round(elapsed, 1),
        image_width=w,
        image_height=h,
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model": MODEL_ID}


# ── Gemini Batch Inventory ────────────────────────────────────────────────────

class InventoryRequest(BaseModel):
    frames: list[str]  # base64-encoded JPEG frames


class InventoryItem(BaseModel):
    name: str
    count: int = 1
    size: str = ""
    notes: str = ""


class InventoryResponse(BaseModel):
    items: list[InventoryItem]
    cost: float = 0.0


INVENTORY_PROMPT = """You are a professional moving estimator. Analyze these video frames from a room scan and create a complete inventory of all visible items.

Rules:
- List every distinct physical object you can see
- Count each item type by the MAXIMUM number visible in any single frame (not sum across frames)
- Use specific names (e.g., "dining chair" not just "chair", "floor lamp" not just "lamp")
- Include size estimate: "small", "medium", "large", or "extra-large"
- Do NOT count the same object twice across different frames — these are frames from a continuous camera pan
- Group similar items (e.g., "dining chair ×4" if you see 4 identical chairs)

Return JSON: {"items": [{"name": "item name", "count": N, "size": "small|medium|large|extra-large", "notes": "optional detail"}]}
"""


@app.post("/api/inventory", response_model=InventoryResponse)
async def inventory(req: InventoryRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build content parts: prompt + frames as PIL images
    parts = [INVENTORY_PROMPT]
    for i, b64 in enumerate(req.frames):
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        parts.append(f"Frame {i + 1}:")
        parts.append(img)

    t0 = time.time()
    response = await client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            max_output_tokens=8000,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    elapsed = time.time() - t0

    # Parse response
    text = response.text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        if "```" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
            data = json.loads(text)
        else:
            raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {text[:200]}")

    items = []
    for item in data.get("items", []):
        items.append(InventoryItem(
            name=item.get("name", "unknown").strip(),
            count=max(1, int(item.get("count", 1))),
            size=item.get("size", ""),
            notes=item.get("notes", ""),
        ))

    # Estimate cost (Gemini 2.0 Flash: $0.10/1M input, $0.40/1M output)
    input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    cost = (input_tokens * 0.10 + output_tokens * 0.40) / 1_000_000

    print(f"Gemini inventory: {len(items)} items, {elapsed:.1f}s, {input_tokens}+{output_tokens} tokens, ${cost:.4f}")

    return InventoryResponse(items=items, cost=round(cost, 4))


# ── Serve PWA frontend ────────────────────────────────────────────────────────

PWA_DIR = os.path.join(os.path.dirname(__file__), "pwa")

# Serve the old test UI at /test
@app.get("/test")
async def test_ui():
    return FileResponse("index.html")


# PWA static files and index
@app.get("/manifest.json")
async def manifest():
    return FileResponse(os.path.join(PWA_DIR, "manifest.json"))

@app.get("/sw.js")
async def service_worker():
    return FileResponse(os.path.join(PWA_DIR, "sw.js"), media_type="application/javascript")

@app.get("/style.css")
async def style():
    return FileResponse(os.path.join(PWA_DIR, "style.css"), media_type="text/css")

@app.get("/app.js")
async def app_js():
    return FileResponse(os.path.join(PWA_DIR, "app.js"), media_type="application/javascript")

@app.get("/icon-192.png")
async def icon_192():
    return FileResponse(os.path.join(PWA_DIR, "icon-192.png"), media_type="image/png")

@app.get("/icon-512.png")
async def icon_512():
    return FileResponse(os.path.join(PWA_DIR, "icon-512.png"), media_type="image/png")

@app.get("/apple-touch-icon.png")
async def apple_touch_icon():
    return FileResponse(os.path.join(PWA_DIR, "icon-192.png"), media_type="image/png")

@app.get("/favicon.png")
async def favicon():
    return FileResponse(os.path.join(PWA_DIR, "icon-192.png"), media_type="image/png")

@app.get("/")
async def index():
    return FileResponse(os.path.join(PWA_DIR, "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
