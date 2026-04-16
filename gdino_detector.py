"""
Grounding-DINO SwinT Detector — standalone module for live detection.

Loads GDINO with household item prompts, batched at PROMPT_BATCH_SIZE=20.
Returns the same {bboxes, labels} format as Florence-2 so the frontend
rendering code works unchanged.
"""

import json
import os
import time

import numpy as np
import torch
from torchvision.ops import nms

PROMPT_BATCH_SIZE = 20
BOX_THRESHOLD = 0.50
TEXT_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.50  # cross-batch dedup
MIN_BOX_AREA = 100  # px²

# ── Paths ──────────────────────────────────────────────────────────────────────

_WEIGHTS_PATH = os.environ.get(
    "GDINO_WEIGHTS", os.path.join(os.path.dirname(__file__), "weights", "groundingdino_swint_ogc.pth")
)
_ITEMS_PATH = os.path.join(os.path.dirname(__file__), "household_items.json")

# ── Module-level state (lazy loaded) ──────────────────────────────────────────

_model = None
_device = None
_prompts: list[str] = []
_transform = None
_warmup_count = 0
WARMUP_THRESHOLD = 3


def _load_prompts() -> list[str]:
    """Load household item list from config file."""
    global _prompts
    if _prompts:
        return _prompts
    with open(_ITEMS_PATH) as f:
        data = json.load(f)
    _prompts = data["items"]
    print(f"[GDINO] Loaded {len(_prompts)} household prompts")
    return _prompts


def _load_model():
    """Load Grounding-DINO SwinT model (called once at startup)."""
    global _model, _device, _transform

    import groundingdino
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import load_model

    if torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"

    # Resolve config path from the installed package
    pkg_dir = os.path.dirname(groundingdino.__file__)
    config_path = os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")

    if not os.path.isfile(_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"GDINO weights not found at: {_WEIGHTS_PATH}\n"
            f"Download: https://github.com/IDEA-Research/GroundingDINO/"
            f"releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        )

    t0 = time.time()
    _model = load_model(config_path, _WEIGHTS_PATH, device=_device)
    print(f"[GDINO] Loaded on {_device} in {time.time() - t0:.1f}s")

    # Pre-build transform (reused for every frame)
    _transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load prompts
    _load_prompts()


def ensure_loaded():
    """Ensure model + prompts are loaded. Safe to call multiple times."""
    if _model is None:
        _load_model()


def warmup_status() -> dict:
    """Return warmup progress for the frontend."""
    return {
        "warmup": min(_warmup_count, WARMUP_THRESHOLD),
        "warmup_total": WARMUP_THRESHOLD,
    }


def _build_caption(prompts: list[str]) -> str:
    """Convert prompt list to GDINO caption format: 'chair . table . sofa'."""
    return " . ".join(p.strip().lower() for p in prompts)


def _match_prompt(phrase: str, batch_prompts: list[str]) -> str:
    """Map GDINO output phrase back to original prompt.
    
    GDINO's phrase extraction can be partial or reordered,
    so we try: exact → substring → fallback to raw phrase.
    """
    phrase_lower = phrase.lower().strip()
    for p in batch_prompts:
        if p.lower() == phrase_lower:
            return p
    for p in batch_prompts:
        if phrase_lower in p.lower() or p.lower() in phrase_lower:
            return p
    return phrase


def detect(image_pil, conf_threshold: float = BOX_THRESHOLD) -> dict:
    """Run GDINO detection on a PIL image.
    
    Returns dict matching Florence-2 format:
        {bboxes: [[x1,y1,x2,y2], ...], labels: [...], time_ms: float, ...}
    """
    global _warmup_count
    ensure_loaded()

    from groundingdino.util.inference import predict

    w, h = image_pil.size
    image_transformed, _ = _transform(image_pil, None)

    t0 = time.time()

    all_bboxes = []
    all_labels = []
    all_scores = []

    # Batch prompts (20 per call) for reliable phrase extraction
    for batch_start in range(0, len(_prompts), PROMPT_BATCH_SIZE):
        batch = _prompts[batch_start:batch_start + PROMPT_BATCH_SIZE]
        caption = _build_caption(batch)

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=_model,
                image=image_transformed,
                caption=caption,
                box_threshold=conf_threshold,
                text_threshold=TEXT_THRESHOLD,
                device=_device,
            )

        if boxes is None or len(boxes) == 0:
            continue

        boxes_np = boxes.cpu().numpy()
        logits_np = logits.cpu().numpy()

        for i in range(len(boxes_np)):
            cx, cy, bw, bh = boxes_np[i]
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            if float(logits_np[i]) < conf_threshold:
                continue
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue

            phrase = phrases[i].strip().lower()
            label = _match_prompt(phrase, batch)
            all_bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            all_labels.append(label)
            all_scores.append(float(logits_np[i]))

    # ── Cross-batch NMS: deduplicate overlapping boxes from different batches ──
    if len(all_bboxes) > 1:
        boxes_t = torch.tensor(all_bboxes, dtype=torch.float32)
        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        keep = nms(boxes_t, scores_t, NMS_IOU_THRESHOLD)
        keep = keep.tolist()
        all_bboxes = [all_bboxes[i] for i in keep]
        all_labels = [all_labels[i] for i in keep]

    elapsed = (time.time() - t0) * 1000
    _warmup_count += 1

    return {
        "bboxes": all_bboxes,
        "labels": all_labels,
        "time_ms": round(elapsed, 1),
        "seg_time_ms": 0.0,
        "image_width": w,
        "image_height": h,
        **warmup_status(),
    }
