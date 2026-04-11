"""
Florence-2 Detection Server
============================
FastAPI backend serving Florence-2-base for object detection
and Gemini 2.0 Flash for batch inventory.
"""

import base64
import asyncio
import io
import json
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor

from segmentor import available_segmentors, get_predictor, segment_bboxes

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
    masks: list[list[list[float]]] | None = None  # SAM segmentation masks as polygons
    time_ms: float
    seg_time_ms: float = 0.0
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
    inputs = {k: v.to(DEVICE, DTYPE) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
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
    inputs = {k: v.to(DEVICE, DTYPE) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
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
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": MODEL_ID,
        "segmentors": available_segmentors(),
    }


# ── Sync detection helper ─────────────────────────────────────────────────────

def sync_detect(image_b64: str, task: str = "<OD>", segmentor: str = "none"):
    """Run Florence-2 detection synchronously (for use in thread pool)."""
    import numpy as np

    img_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = image.size

    t0 = time.time()
    inputs = processor(text=task, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE, DTYPE) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs.pop("attention_mask", None)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024, num_beams=1, do_sample=False
        )

    result_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(result_text, task=task, image_size=image.size)
    elapsed = (time.time() - t0) * 1000

    task_result = parsed.get(task, {})
    bboxes = task_result.get("bboxes", [])
    labels = task_result.get("labels", [])

    result = {
        "bboxes": bboxes,
        "labels": labels,
        "time_ms": round(elapsed, 1),
        "seg_time_ms": 0.0,
        "image_width": w,
        "image_height": h,
    }

    # Optional SAM segmentation on detected bboxes
    if segmentor != "none" and bboxes:
        try:
            pred = get_predictor(segmentor, DEVICE)
            image_np = np.array(image)
            masks, seg_ms = segment_bboxes(pred, image_np, bboxes)
            result["masks"] = masks
            result["seg_time_ms"] = round(seg_ms, 1)
        except Exception as e:
            print(f"Segmentation error ({segmentor}): {e}")

    return result


# ── WebSocket Detection ──────────────────────────────────────────────────────

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")

    latest_frame = None
    frame_available = asyncio.Event()
    connected = True

    async def receiver():
        nonlocal latest_frame, connected
        try:
            while connected:
                raw = await websocket.receive_text()
                latest_frame = raw  # always overwrite — keep only latest
                frame_available.set()
        except WebSocketDisconnect:
            connected = False
            frame_available.set()
        except Exception:
            connected = False
            frame_available.set()

    async def processor():
        nonlocal latest_frame, connected
        try:
            while connected:
                await frame_available.wait()
                frame_available.clear()

                if not connected:
                    break

                if latest_frame is None:
                    continue

                raw = latest_frame
                latest_frame = None  # consume it

                data = json.loads(raw)
                frame_id = data.get("frame_id", 0)
                image_b64 = data["image"]
                task = data.get("task", "<OD>")
                segmentor = data.get("segmentor", "none")

                result = await asyncio.to_thread(sync_detect, image_b64, task, segmentor)
                result["frame_id"] = frame_id

                if connected:
                    await websocket.send_json(result)
        except WebSocketDisconnect:
            connected = False
        except Exception as e:
            print(f"WS processor error: {e}")
            connected = False

    recv_task = asyncio.create_task(receiver())
    proc_task = asyncio.create_task(processor())

    await asyncio.gather(recv_task, proc_task, return_exceptions=True)
    print("WebSocket client disconnected")


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


INVENTORY_PROMPT = """You are a professional moving inventory specialist. These are frames from a continuous camera pan around a room. Your job is to create a complete inventory of ALL distinct physical objects visible across all frames.

IMPORTANT — This is a PANNING camera scan:
- The camera moves continuously, so objects appear in some frames and NOT in others.
- Identify ALL unique objects by tracking them across frames. An object seen in frames 1-3 and a DIFFERENT object seen in frame 11 are SEPARATE items — count both.
- If two objects look different (different color, size, or shape), they are SEPARATE items even if they are the same category.
- For identical objects grouped together in ONE frame (e.g., 4 dining chairs around a table), count them all.
- Do NOT double-count the SAME individual object appearing across multiple consecutive frames.

Detection rules:
- List EVERY distinct physical object — furniture, electronics, bags, bottles, containers, clothing, decor, appliances, etc.
- Be thorough: look carefully at all areas of each frame including edges, backgrounds, and partially visible objects.
- Use specific descriptive names (e.g., "white water bottle", "red metallic bottle", "black backpack").
- Include color in the name when it helps distinguish items.
- Size estimate: "small", "medium", "large", or "extra-large".
- Ignore fixed room features: walls, floor, ceiling, doors, windows, power outlets, light switches.

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


# ── Upload Artifacts ──────────────────────────────────────────────────────────

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


@app.post("/api/upload-artifacts")
async def upload_artifacts(
    metadata: str = Form(...),
    video: UploadFile | None = File(None),
    frames: list[UploadFile] = File(default=[]),
):
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_dir = os.path.join(ARTIFACTS_DIR, f"scan_{ts}")
    frames_dir = os.path.join(scan_dir, "keyframes")
    os.makedirs(frames_dir, exist_ok=True)

    # Save video
    if video and video.filename:
        video_path = os.path.join(scan_dir, video.filename)
        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)
        print(f"Saved video: {video_path} ({len(content) / 1024 / 1024:.1f}MB)")

    # Save key frames
    for frame in frames:
        if frame.filename:
            frame_path = os.path.join(frames_dir, frame.filename)
            content = await frame.read()
            with open(frame_path, "wb") as f:
                f.write(content)
    print(f"Saved {len(frames)} key frames to {frames_dir}")

    # Save metadata (inventory result + stats)
    meta_path = os.path.join(scan_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(json.loads(metadata), f, indent=2)

    print(f"Artifacts saved to {scan_dir}")
    return {"status": "ok", "path": f"artifacts/scan_{ts}", "frames": len(frames)}


# ── Serve PWA frontend ────────────────────────────────────────────────────────

PWA_DIR = os.path.join(os.path.dirname(__file__), "pwa")

# Serve the old test UI at /test
@app.get("/test")
async def test_ui():
    return FileResponse("index.html")


# PWA static files — served at root, /florence-scanner/, and custom ROUTE_PREFIX
_ROUTE_PREFIX = os.environ.get("ROUTE_PREFIX", "").rstrip("/")
_prefixes = ["", "/florence-scanner"]
if _ROUTE_PREFIX and _ROUTE_PREFIX not in _prefixes:
    _prefixes.append(_ROUTE_PREFIX)

def _pwa_file(name, media_type=None):
    path = os.path.join(PWA_DIR, name)
    return FileResponse(path, media_type=media_type) if media_type else FileResponse(path)

_static_files = {
    "manifest.json": "application/json",
    "sw.js": "application/javascript",
    "style.css": "text/css",
    "app.js": "application/javascript",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
}

for _prefix in _prefixes:
    for _name, _mime in _static_files.items():
        def _make_handler(n=_name, m=_mime):
            async def handler():
                return _pwa_file(n, m)
            return handler
        app.get(f"{_prefix}/{_name}")(_make_handler())

    # Index page
    def _make_index(prefix=_prefix):
        async def index():
            return FileResponse(os.path.join(PWA_DIR, "index.html"))
        return index
    app.get(f"{_prefix}/" if _prefix else "/")(_make_index())

    # Icon aliases
    for _alias, _target in [("apple-touch-icon.png", "icon-192.png"), ("favicon.png", "icon-192.png")]:
        def _make_alias(t=_target):
            async def handler():
                return _pwa_file(t, "image/png")
            return handler
        app.get(f"{_prefix}/{_alias}")(_make_alias())

    # API endpoints under prefix (for local testing without Caddy)
    if _prefix:
        app.post(f"{_prefix}/api/detect", response_model=DetectResponse)(detect)
        app.get(f"{_prefix}/api/health")(health)
        app.post(f"{_prefix}/api/inventory", response_model=InventoryResponse)(inventory)
        app.post(f"{_prefix}/api/upload-artifacts")(upload_artifacts)
        app.websocket(f"{_prefix}/ws/detect")(ws_detect)


if __name__ == "__main__":
    ssl_kwargs = {}
    cert = os.path.join(os.path.dirname(__file__), "cert.pem")
    key = os.path.join(os.path.dirname(__file__), "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
        print(f"HTTPS enabled — https://0.0.0.0:8100")
    uvicorn.run(app, host="0.0.0.0", port=8100, **ssl_kwargs)
