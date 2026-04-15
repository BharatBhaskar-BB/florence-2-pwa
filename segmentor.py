"""
Lightweight SAM wrappers for bbox-prompted segmentation.
Supports MobileSAM and SAM 2 Tiny with lazy loading.

Optimizations:
- Batched box prediction (one predict call for all bboxes)
- Image embedding cache (skip set_image if frame unchanged)
"""

import hashlib
import os
import time

import numpy as np
import torch

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Cache loaded predictors
_predictors: dict = {}

# Image embedding cache — skip set_image() when frame is similar
_embedding_cache: dict = {}  # {segmentor_name: {"hash": str, "features": ...}}


def available_segmentors() -> list[str]:
    """Return list of segmentor names whose packages are importable."""
    avail = ["none"]
    try:
        import mobile_sam  # noqa: F401
        avail.append("mobilesam")
    except ImportError:
        pass
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        avail.append("sam2-tiny")
    except ImportError:
        pass
    return avail


def _download_weights(repo_id: str, filename: str, local_name: str) -> str:
    """Download weights from HuggingFace Hub if not cached locally."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    local_path = os.path.join(WEIGHTS_DIR, local_name)
    if os.path.exists(local_path):
        return local_path

    print(f"Downloading {local_name} from {repo_id}...")
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
    # Symlink or copy to our weights dir
    if downloaded != local_path:
        import shutil
        shutil.copy2(downloaded, local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"Downloaded {local_name} ({size_mb:.1f}MB)")
    return local_path


def _load_mobilesam(device: torch.device):
    """Load MobileSAM predictor."""
    from mobile_sam import sam_model_registry, SamPredictor

    ckpt = _download_weights("dhkim2810/MobileSAM", "mobile_sam.pt", "mobile_sam.pt")
    sam = sam_model_registry["vit_t"](checkpoint=ckpt)
    # Force fp32 to avoid CUDA dtype mismatch in set_image
    sam.to(device).float().eval()
    return SamPredictor(sam)


def _load_sam2_tiny(device: torch.device):
    """Load SAM 2.1 Hiera-Tiny predictor via HuggingFace."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
    pred.model.to(device).float().eval()
    return pred


def get_predictor(name: str, device: torch.device):
    """Get or lazily create a SAM predictor by name."""
    if name in _predictors:
        return _predictors[name]

    t0 = time.time()
    if name == "mobilesam":
        pred = _load_mobilesam(device)
    elif name == "sam2-tiny":
        pred = _load_sam2_tiny(device)
    else:
        raise ValueError(f"Unknown segmentor: {name}")

    _predictors[name] = pred
    print(f"Loaded {name} in {time.time() - t0:.1f}s on {device}")
    return pred


def _image_hash(image_np: np.ndarray) -> str:
    """Fast perceptual hash: downsample to 32x32, hash the bytes."""
    from PIL import Image
    small = Image.fromarray(image_np).resize((32, 32), Image.NEAREST)
    return hashlib.md5(np.array(small).tobytes()).hexdigest()


def _set_image_cached(predictor, image_np: np.ndarray, name: str) -> bool:
    """Call set_image only if the frame changed. Returns True if embedding was recomputed."""
    img_hash = _image_hash(image_np)
    cached = _embedding_cache.get(name)
    if cached and cached["hash"] == img_hash:
        return False  # skip — embedding still valid
    predictor.set_image(image_np)
    _embedding_cache[name] = {"hash": img_hash}
    return True


def mask_to_polygon(mask: np.ndarray, simplify: float = 2.0) -> list[list[float]]:
    """Convert a binary mask to simplified polygon contour points."""
    import cv2

    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, simplify, True)
    poly = approx.squeeze().tolist()
    if not poly:
        return []
    # Ensure list of [x, y] (not flat when single point)
    if isinstance(poly[0], (int, float)):
        return [poly]
    return poly


def segment_bboxes(
    predictor,
    image_np: np.ndarray,
    bboxes: list[list[float]],
    simplify: float = 2.0,
    name: str = "mobilesam",
) -> tuple[list[list[list[float]]], float]:
    """Run segmentation for all bboxes on one image.

    Uses embedding cache to skip set_image when frame is similar.
    Uses batched predict_torch when available for all boxes at once.

    Returns: (polygons, seg_time_ms)
        polygons: list of polygon point lists, one per bbox
    """
    t0 = time.time()

    # Set image with caching
    recomputed = _set_image_cached(predictor, image_np, name)

    if not bboxes:
        return [], (time.time() - t0) * 1000

    # Try batched prediction (all boxes at once)
    polygons = _predict_batched(predictor, bboxes, simplify)

    elapsed_ms = (time.time() - t0) * 1000
    tag = "recomputed" if recomputed else "cached"
    if elapsed_ms > 50:
        print(f"SAM {name}: {len(bboxes)} boxes, {elapsed_ms:.0f}ms (embed {tag})")
    return polygons, elapsed_ms


def _predict_batched(predictor, bboxes, simplify):
    """Predict masks for all bboxes. Uses predict_torch for batch if available."""
    device = predictor.model.device

    # MobileSAM / SAM support predict_torch with batched boxes
    if hasattr(predictor, 'predict_torch'):
        try:
            boxes_np = np.array(bboxes, dtype=np.float32)
            boxes_t = torch.as_tensor(boxes_np, device=device)
            transformed = predictor.transform.apply_boxes_torch(boxes_t, predictor.original_size)

            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
            # masks shape: (N, 1, H, W)
            masks_np = masks.squeeze(1).cpu().numpy()
            return [mask_to_polygon(m, simplify) for m in masks_np]
        except Exception as e:
            print(f"Batched predict_torch failed, falling back: {e}")

    # Fallback: predict one-by-one
    polygons = []
    for bbox in bboxes:
        try:
            box_np = np.array(bbox, dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box_np, multimask_output=False
            )
            polygons.append(mask_to_polygon(masks[0], simplify))
        except Exception as e:
            print(f"Segmentation failed for bbox {bbox}: {e}")
            polygons.append([])
    return polygons
