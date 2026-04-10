"""
Lightweight SAM wrappers for bbox-prompted segmentation.
Supports MobileSAM and EfficientViT-SAM with lazy loading.
"""

import os
import time

import numpy as np
import torch

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Cache loaded predictors
_predictors: dict = {}


def available_segmentors() -> list[str]:
    """Return list of segmentor names whose packages are importable."""
    avail = ["none"]
    try:
        import mobile_sam  # noqa: F401
        avail.append("mobilesam")
    except ImportError:
        pass
    try:
        from efficientvit.sam_model_zoo import create_sam_model  # noqa: F401
        avail.append("efficientvit-sam")
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
    sam.to(device).eval()
    return SamPredictor(sam)


def _load_efficientvit_sam(device: torch.device):
    """Load EfficientViT-SAM-L0 predictor."""
    from efficientvit.sam_model_zoo import create_sam_model
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

    ckpt = _download_weights(
        "han-cai/efficientvit-sam", "l0.pt", "efficientvit_sam_l0.pt"
    )
    sam = create_sam_model(name="l0", weight_url=ckpt)
    sam.to(device).eval()
    return EfficientViTSamPredictor(sam)


def get_predictor(name: str, device: torch.device):
    """Get or lazily create a SAM predictor by name."""
    if name in _predictors:
        return _predictors[name]

    t0 = time.time()
    if name == "mobilesam":
        pred = _load_mobilesam(device)
    elif name == "efficientvit-sam":
        pred = _load_efficientvit_sam(device)
    else:
        raise ValueError(f"Unknown segmentor: {name}")

    _predictors[name] = pred
    print(f"Loaded {name} in {time.time() - t0:.1f}s on {device}")
    return pred


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
) -> tuple[list[list[list[float]]], float]:
    """Run segmentation for all bboxes on one image.

    Returns: (polygons, seg_time_ms)
        polygons: list of polygon point lists, one per bbox
    """
    t0 = time.time()
    predictor.set_image(image_np)

    polygons = []
    for bbox in bboxes:
        try:
            box_np = np.array(bbox, dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box_np, multimask_output=False
            )
            poly = mask_to_polygon(masks[0], simplify)
            polygons.append(poly)
        except Exception as e:
            print(f"Segmentation failed for bbox {bbox}: {e}")
            polygons.append([])

    elapsed_ms = (time.time() - t0) * 1000
    return polygons, elapsed_ms
