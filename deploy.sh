#!/usr/bin/env bash
set -euo pipefail

# ── BundleBox Deploy Script ──────────────────────────────────────────────────
# Usage:
#   1. SSH into your GCP L4 instance
#   2. Clone/copy this repo
#   3. Set GEMINI_API_KEY:  export GEMINI_API_KEY=your_key_here
#   4. Run:  bash deploy.sh

echo "═══ BundleBox Deploy ═══"

# ── Preflight checks ────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install with:"
    echo "  curl -fsSL https://get.docker.com | sh"
    exit 1
fi

if ! docker compose version &>/dev/null; then
    echo "ERROR: docker compose v2 not found."
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers + nvidia-container-toolkit."
    echo "  sudo apt install -y nvidia-container-toolkit && sudo systemctl restart docker"
    exit 1
fi

if [[ ! -f .env ]]; then
    echo "ERROR: .env file not found."
    echo "  cp .env.example .env  # then fill in your keys"
    exit 1
fi

if ! grep -q 'GEMINI_API_KEY=.' .env || grep -q 'your_gemini_api_key_here' .env; then
    echo "WARNING: GEMINI_API_KEY not set in .env — /api/inventory will fail."
    read -p "Continue without Gemini? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# ── Build & Run ──────────────────────────────────────────────────────────────
echo ""
echo "Building Docker image (first build downloads ~3GB, takes 3-5 min)..."
docker compose build

echo ""
echo "Starting BundleBox..."
docker compose up -d

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  BundleBox is starting up!"
echo ""
echo "  Wait ~30s for model to load, then open:"
EXTERNAL_IP=$(curl -s -m 3 http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google" 2>/dev/null || echo "<your-instance-ip>")
echo "  http://${EXTERNAL_IP}:8100"
echo ""
echo "  Health check:"
echo "  curl http://localhost:8100/api/health"
echo ""
echo "  Logs:"
echo "  docker compose logs -f"
echo ""
echo "  Stop:"
echo "  docker compose down"
echo "═══════════════════════════════════════════════════════"
