"""
Florence-2 Live Detection Test Script
======================================
Tests Microsoft Florence-2-base for zero-shot object detection on household
room frames. Covers Step 1 & 2 from the testing plan:
  - Step 1: Detection quality on room frames (<OD>, <DENSE_REGION_CAPTION>, <CAPTION>)
  - Step 2: Latency benchmark on M4 Pro Mac

Usage:
  # Test with a single image
  python test_florence2.py --image path/to/frame.jpg

  # Test with frames extracted from a video (stride=30 by default)
  python test_florence2.py --video path/to/video.mp4

  # Test with all videos in a directory
  python test_florence2.py --video-dir /Users/bharat.bhaskar/GDINO_LLM_PRIMING/test_videos

  # Adjust number of frames per video and benchmark iterations
  python test_florence2.py --video path/to/video.mp4 --max-frames 10 --benchmark-iters 20

  # Use the large model for accuracy comparison
  python test_florence2.py --video path/to/video.mp4 --model florence-2-large
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_VARIANTS = {
    "florence-2-base": "microsoft/Florence-2-base",
    "florence-2-large": "microsoft/Florence-2-large",
}

TASKS = ["<OD>", "<DENSE_REGION_CAPTION>", "<CAPTION>"]

# Colors for bounding box visualization (RGB)
BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (0, 128, 255), (128, 0, 255), (200, 200, 200), (100, 100, 100),
]


# ── Frame Extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, max_frames: int = 10, stride: int = 30) -> list[str]:
    """Extract frames from a video using ffmpeg at a given stride."""
    tmpdir = tempfile.mkdtemp(prefix="florence2_frames_")

    # Get total frame count
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        total_frames = int(subprocess.check_output(
            probe_cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).decode().strip())
    except (subprocess.CalledProcessError, ValueError):
        # Fallback: estimate from duration
        dur_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path,
        ]
        duration = float(subprocess.check_output(
            dur_cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).decode().strip())
        total_frames = int(duration * 30)  # assume 30fps

    # Calculate which frames to extract
    frame_indices = list(range(0, total_frames, stride))[:max_frames]
    if not frame_indices:
        frame_indices = [0]

    print(f"  Video: {Path(video_path).name} | {total_frames} total frames | extracting {len(frame_indices)} frames (stride={stride})")

    extracted = []
    for i, frame_idx in enumerate(frame_indices):
        out_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"select=eq(n\\,{frame_idx})",
            "-vsync", "vfr",
            "-frames:v", "1",
            "-q:v", "2",
            out_path,
        ]
        subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.exists(out_path):
            extracted.append(out_path)

    return extracted


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(model_key: str) -> tuple:
    """Load Florence-2 model and processor."""
    model_name = MODEL_VARIANTS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    print(f"{'='*60}")

    t0 = time.time()

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS doesn't support float16 for all ops
        print(f"  Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print(f"  Device: CUDA ({torch.cuda.get_device_name()})")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print(f"  Device: CPU")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    load_time = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {load_time:.1f}s | {param_count:.0f}M params | dtype={dtype}")

    return model, processor, device, dtype


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, processor, device, image: Image.Image, task: str) -> dict:
    """Run a single inference pass and return parsed results."""
    inputs = processor(text=task, images=image, return_tensors="pt")
    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    # Remove attention_mask — Florence-2's generate() builds its own after merging
    # image features with text embeddings; the processor's text-only mask conflicts.
    inputs.pop("attention_mask", None)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
        )

    result_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        result_text, task=task, image_size=image.size
    )
    return parsed


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_detections(image: Image.Image, detections: dict, task: str) -> Image.Image:
    """Draw bounding boxes and labels on a copy of the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    if task in detections and "bboxes" in detections[task]:
        bboxes = detections[task]["bboxes"]
        labels = detections[task].get("labels", [f"obj_{i}" for i in range(len(bboxes))])

        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            x1, y1, x2, y2 = bbox

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 16), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font)

    return img


# ── Testing ───────────────────────────────────────────────────────────────────

def test_frame(
    model, processor, device, image_path: str, output_dir: str, frame_label: str
) -> dict:
    """Run all tasks on a single frame, save visualized output, return results."""
    image = Image.open(image_path).convert("RGB")
    print(f"\n  Frame: {frame_label} ({image.size[0]}x{image.size[1]})")

    results = {}

    for task in TASKS:
        t0 = time.time()
        parsed = run_inference(model, processor, device, image, task)
        elapsed = time.time() - t0

        results[task] = {
            "parsed": parsed,
            "time_ms": elapsed * 1000,
        }

        if task == "<OD>":
            od = parsed.get("<OD>", {})
            n_objects = len(od.get("bboxes", []))
            labels = od.get("labels", [])
            unique_labels = sorted(set(labels))
            print(f"    {task}: {n_objects} objects in {elapsed*1000:.0f}ms")
            print(f"      Labels: {', '.join(unique_labels)}")

            # Save annotated image
            annotated = draw_detections(image, parsed, task)
            out_path = os.path.join(output_dir, f"{frame_label}_OD.jpg")
            annotated.save(out_path, quality=90)
            print(f"      Saved: {out_path}")

        elif task == "<DENSE_REGION_CAPTION>":
            drc = parsed.get("<DENSE_REGION_CAPTION>", {})
            n_regions = len(drc.get("bboxes", []))
            captions = drc.get("labels", [])
            print(f"    {task}: {n_regions} regions in {elapsed*1000:.0f}ms")
            for cap in captions[:5]:
                print(f"      - {cap}")
            if len(captions) > 5:
                print(f"      ... and {len(captions) - 5} more")

            # Save annotated image
            annotated = draw_detections(image, parsed, task)
            out_path = os.path.join(output_dir, f"{frame_label}_DRC.jpg")
            annotated.save(out_path, quality=90)

        elif task == "<CAPTION>":
            caption = parsed.get("<CAPTION>", "")
            print(f"    {task}: \"{caption}\" ({elapsed*1000:.0f}ms)")

    return results


def run_benchmark(
    model, processor, device, image_path: str, n_iters: int = 20
) -> dict:
    """Benchmark <OD> inference latency."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text="<OD>", images=image, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs.pop("attention_mask", None)

    print(f"\n{'='*60}")
    print(f"Latency Benchmark: <OD> task, {n_iters} iterations")
    print(f"  Image: {image.size[0]}x{image.size[1]}")
    print(f"{'='*60}")

    # Warmup (3 passes)
    print("  Warming up (3 passes)...")
    for _ in range(3):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1024, num_beams=1, do_sample=False)

    # Timed runs
    times = []
    for i in range(n_iters):
        t0 = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1024, num_beams=1, do_sample=False)
        elapsed = time.time() - t0
        times.append(elapsed * 1000)
        print(f"    Run {i+1:2d}/{n_iters}: {elapsed*1000:.0f}ms")

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    median = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]

    print(f"\n  Results:")
    print(f"    Avg:    {avg:.0f}ms")
    print(f"    Median: {median:.0f}ms")
    print(f"    Min:    {mn:.0f}ms")
    print(f"    Max:    {mx:.0f}ms")
    print(f"    P95:    {p95:.0f}ms")
    print(f"    FPS:    {1000/avg:.1f}")

    verdict = "PASS" if avg < 500 else "MARGINAL" if avg < 1000 else "FAIL"
    print(f"\n  Verdict: {verdict} (<500ms=PASS, 500-1000ms=MARGINAL, >1000ms=FAIL)")

    return {
        "avg_ms": avg,
        "median_ms": median,
        "min_ms": mn,
        "max_ms": mx,
        "p95_ms": p95,
        "fps": 1000 / avg,
        "verdict": verdict,
        "n_iters": n_iters,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Florence-2 Live Detection Test")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to a single test image")
    input_group.add_argument("--video", type=str, help="Path to a test video")
    input_group.add_argument("--video-dir", type=str, help="Directory of test videos")

    parser.add_argument(
        "--model", type=str, default="florence-2-base",
        choices=list(MODEL_VARIANTS.keys()),
        help="Model variant (default: florence-2-base)",
    )
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames per video (default: 10)")
    parser.add_argument("--stride", type=int, default=30, help="Frame extraction stride (default: 30)")
    parser.add_argument("--benchmark-iters", type=int, default=20, help="Benchmark iterations (default: 20)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for annotated images")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency benchmark")

    args = parser.parse_args()

    # Setup output directory
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "florence2_output"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Collect image paths
    image_paths = []
    frame_labels = []

    if args.image:
        image_paths.append(args.image)
        frame_labels.append(Path(args.image).stem)

    elif args.video:
        print(f"\nExtracting frames from video...")
        frames = extract_frames(args.video, max_frames=args.max_frames, stride=args.stride)
        image_paths.extend(frames)
        video_stem = Path(args.video).stem
        frame_labels.extend([f"{video_stem}_f{i}" for i in range(len(frames))])

    elif args.video_dir:
        video_exts = {".mp4", ".mov", ".avi", ".mkv"}
        videos = sorted([
            f for f in Path(args.video_dir).iterdir()
            if f.suffix.lower() in video_exts
        ])
        print(f"\nFound {len(videos)} videos in {args.video_dir}")
        for vpath in videos:
            print(f"\nExtracting frames from {vpath.name}...")
            frames = extract_frames(str(vpath), max_frames=args.max_frames, stride=args.stride)
            video_stem = vpath.stem
            image_paths.extend(frames)
            frame_labels.extend([f"{video_stem}_f{i}" for i in range(len(frames))])

    if not image_paths:
        print("ERROR: No frames to process.")
        sys.exit(1)

    print(f"\nTotal frames to process: {len(image_paths)}")

    # Load model
    model, processor, device, dtype = load_model(args.model)

    # Run detection on all frames
    print(f"\n{'='*60}")
    print(f"Running detection on {len(image_paths)} frames...")
    print(f"{'='*60}")

    all_results = {}
    all_labels = set()

    for img_path, label in zip(image_paths, frame_labels):
        result = test_frame(model, processor, device, img_path, output_dir, label)
        all_results[label] = result

        # Collect unique labels across all frames
        od = result.get("<OD>", {}).get("parsed", {}).get("<OD>", {})
        all_labels.update(od.get("labels", []))

    # Summary
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Frames processed: {len(all_results)}")
    print(f"  Unique object classes found: {len(all_labels)}")
    print(f"  Classes: {', '.join(sorted(all_labels))}")

    od_times = [
        r["<OD>"]["time_ms"]
        for r in all_results.values()
        if "<OD>" in r
    ]
    if od_times:
        print(f"\n  <OD> inference times across frames:")
        print(f"    Avg: {sum(od_times)/len(od_times):.0f}ms")
        print(f"    Min: {min(od_times):.0f}ms")
        print(f"    Max: {max(od_times):.0f}ms")

    # Benchmark
    if not args.skip_benchmark and image_paths:
        benchmark_image = image_paths[len(image_paths) // 2]  # use middle frame
        benchmark_results = run_benchmark(
            model, processor, device, benchmark_image, n_iters=args.benchmark_iters
        )
        all_results["_benchmark"] = benchmark_results

    # Save full results as JSON
    results_path = os.path.join(output_dir, "results.json")

    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  Full results saved: {results_path}")

    # Save label inventory
    labels_path = os.path.join(output_dir, "detected_labels.txt")
    with open(labels_path, "w") as f:
        for label in sorted(all_labels):
            f.write(f"{label}\n")
    print(f"  Label inventory saved: {labels_path}")

    print(f"\n  Annotated images saved in: {output_dir}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
