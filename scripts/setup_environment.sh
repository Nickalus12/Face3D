#!/bin/bash
# Face3D Environment Setup Script
# Run from project root: bash scripts/setup_environment.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# ── Step 1: Install Miniconda if not present ──
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."

    # Download Miniconda for Windows
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    INSTALLER="$TEMP/miniconda_installer.exe"

    curl -Lo "$INSTALLER" "$MINICONDA_URL"
    echo "Downloaded Miniconda installer."
    echo ""
    echo "=== MANUAL STEP REQUIRED ==="
    echo "Run the installer: $INSTALLER"
    echo "After installing, restart your terminal and re-run this script."
    echo "=== ===================== ==="
    exit 1
fi

echo "Conda found: $(conda --version)"

# ── Step 2: Create conda environment ──
if conda env list | grep -q "face3d"; then
    echo "Environment 'face3d' already exists. Updating..."
    conda env update -f "$PROJECT_ROOT/environment.yaml" --prune
else
    echo "Creating 'face3d' environment..."
    conda env create -f "$PROJECT_ROOT/environment.yaml"
fi

echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate face3d

# ── Step 3: Install Depth Anything 3 from GitHub ──
echo "Installing Depth Anything 3..."
pip install git+https://github.com/bytedance-seed/depth-anything-3.git 2>/dev/null || \
    echo "WARNING: Depth Anything 3 install failed. Install manually later."

# ── Step 4: Verify CUDA ──
echo ""
echo "Verifying CUDA..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# ── Step 5: Verify key imports ──
echo ""
echo "Verifying imports..."
python -c "
errors = []
for pkg in ['cv2', 'mediapipe', 'open3d', 'scipy', 'yaml', 'tqdm']:
    try:
        __import__(pkg)
        print(f'  OK: {pkg}')
    except ImportError:
        errors.append(pkg)
        print(f'  MISSING: {pkg}')

# Check gsplat (needs CUDA compile)
try:
    import gsplat
    print(f'  OK: gsplat')
except Exception as e:
    errors.append('gsplat')
    print(f'  MISSING: gsplat ({e})')

if errors:
    print(f'\nMissing packages: {errors}')
    print('Try: pip install ' + ' '.join(errors))
else:
    print('\nAll packages verified!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "Activate with: conda activate face3d"
echo "Run pipeline:  python scripts/run_pipeline.py --video path/to/video.mp4"
echo "=========================================="
