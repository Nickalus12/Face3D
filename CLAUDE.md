# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Face3D** — 3D face reconstruction from Samsung Galaxy S25 Ultra captures. Takes video + Expert RAW photos + Sensor Logger data and produces photorealistic 3D Gaussian Splat models with animated FLAME mesh, UV-textured mesh, and turntable renders. Includes a Tauri 2.0 desktop app for visualization and pipeline control.

## Architecture

### Two major components:

1. **Python Pipeline** (`src/`, `scripts/`) — 15-stage reconstruction engine
2. **Tauri Desktop App** (`face3d-app/`) — React + Rust visualization/control app

### Pipeline (15 stages, orchestrated by `scripts/run_pipeline.py`)

```
Input: --content-dir folder OR --video + --photos + --sensor-log
  │
  ├─ Stage 0:   src/capture/data_organizer.py    → Auto-detect & organize inputs by type/lens
  ├─ Stage 1:   src/capture/frame_extractor.py   → 8K→1080p pre-convert, frame extraction
  │             src/capture/photo_processor.py   → Expert RAW DNG processing (rawpy DHT)
  │             src/capture/multi_res_processor.py → 200MP multi-resolution output
  ├─ Stage 2:   src/capture/color_correction.py  → S-Log3→sRGB (auto-detect, LUT fast path)
  ├─ Stage 3:   src/capture/frame_filter.py      → Blur/exposure filter, smart frame selection
  ├─ Stage 4:   src/sensors/sensor_logger.py     → Parse Sensor Logger ZIPs (183Hz quaternions)
  │             src/sensors/imu_parser.py         → Fallback: IMU from video metadata
  ├─ Stage 5:   src/sensors/orientation.py        → Madgwick/Complementary filter, COLMAP priors
  │             (skipped if Sensor Logger has pre-fused quaternions)
  ├─ Stage 6:   src/depth/da3_unified.py          → DA3: depth + poses + point cloud (primary)
  │             src/reconstruction/colmap_runner.py → COLMAP SfM (fallback)
  ├─ Stage 7-8: (auto-skipped when DA3 unified is used)
  ├─ Stage 9:   src/reconstruction/landmarks.py   → MediaPipe 478-point face landmarks
  ├─ Stage 10:  src/reconstruction/flame_fitter.py → FLAME parametric face fitting
  ├─ Stage 11:  src/reconstruction/face_segmentation.py → Face masks + GrabCut refinement
  ├─ Stage 12:  src/splatting/initializer.py       → FLAME-bound Gaussian init or COLMAP sparse
  ├─ Stage 13:  src/splatting/trainer.py           → 2DGS training (gsplat rasterization_2dgs)
  └─ Stage 14:  src/splatting/exporter.py          → PLY + SuGaR mesh + texture + animation + report
```

### Tauri Desktop App (`face3d-app/`)

```
face3d-app/
├── src/                      # React + TypeScript frontend
│   ├── components/           # 21 React components (Viewer3D, Console, Gallery, etc.)
│   ├── hooks/                # Reactive data hooks (useGpuInfo, useModelLoader, etc.)
│   ├── store/                # Zustand stores (pipeline, session, settings, toast, view)
│   └── lib/                  # IPC layer (api.ts, cache.ts, events.ts, plyLoader.ts)
└── src-tauri/src/            # Rust backend
    ├── main.rs               # Tauri builder + plugin registration
    ├── commands/              # 7 command modules (pipeline, sessions, gpu, sensors, etc.)
    ├── state.rs               # Shared state (pipeline process, GPU cache)
    └── error.rs               # Typed error handling (thiserror)
```

### Data Flow
```
content/New/ (mixed video, photos, sensor ZIPs)
  → data/raw/{session}/video/, photos/{wide,main}/, sensors/
    → data/processed/{session}/frames, depth, colmap, landmarks, flame, masks
      → data/output/{session}/gaussians.ply, mesh.obj, texture.png, renders/, animations/, report.html
```

Stage completion tracked via `.stage_N_complete` marker files — fully resumable.

## Commands

### Python Pipeline

```bash
# Environment
conda activate face3d

# CUDA env (required for gsplat JIT compilation)
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
export PATH="$CUDA_HOME/bin:/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

# Run full pipeline (recommended: --content-dir auto-detects everything)
python scripts/run_pipeline.py --content-dir content/New --session natalie

# Explicit inputs
python scripts/run_pipeline.py --video face.mp4 --photos *.jpg --sensor-log sensors.zip --session test

# Resume / re-run specific stages
python scripts/run_pipeline.py --video face.mp4 --session test --start-stage 13 --force

# Quick smoke test (synthetic data, 11 seconds)
python scripts/quick_test.py

# Benchmarks
python scripts/benchmark.py --quick
python scripts/benchmark.py --full --report
```

### Tests

```bash
pip install -r requirements-dev.txt

# Run all fast tests
python -m pytest tests/ -v

# Skip slow/GPU tests
python -m pytest tests/ -m "not slow and not cuda"

# Run only property-based tests
python -m pytest tests/ -m property_based

# Run benchmarks
python -m pytest tests/test_benchmarks.py --benchmark-enable

# Single test
python -m pytest tests/test_pipeline_stages.py::test_colmap_io_roundtrip -v
```

### Tauri Desktop App

```bash
cd face3d-app
npm install
npm run tauri dev          # Dev mode with hot reload
npm run tauri build        # Production build
npm run build              # Frontend only (Vite)
npx tsc --noEmit           # TypeScript check
```

## Key Technical Details

### Import Convention
All Python modules use **absolute imports from `src/`** (which is on `sys.path`):
```python
from utils.colmap_io import read_cameras_binary    # NOT from src.utils or ..utils
from splatting.trainer import GaussianTrainer       # NOT from src.splatting
from depth.da3_unified import run_da3_unified       # NOT relative imports
```

### MediaPipe Version Pin
MediaPipe **must be 0.10.14**. Versions 0.10.21+ removed `mp.solutions.face_mesh/face_detection/selfie_segmentation`. The codebase has compatibility shims but the pinned version is required.

### gsplat 2DGS Specifics (v1.4.0+)
- `rasterization_2dgs()` with `distloss=True` requires `render_mode="RGB+ED"`
- 2DGS uses `key_for_gradient="gradient_2dgs"` in DefaultStrategy (not `"means2d"`)
- `absgrad=True` in the rasterization call but `absgrad=False` in the strategy (gsplat 1.4.0 incompatibility: `gradient_2dgs` tensor has no `.absgrad` attribute)
- `backgrounds` parameter is NOT auto-expanded for 2DGS — omit it when using `render_mode="RGB+ED"`
- `packed=False` is 25-30% faster; `packed=True` uses up to 4x less memory (for large scenes or <12GB VRAM)
- gsplat achieves up to 4x overall memory reduction vs reference 3DGS through fused kernels

### Export Formats
Stage 14 supports multiple export formats via `splatting.export.formats` config:
- `"ply"`: Standard PLY via `gsplat.export_splats()` (fallback: custom writer). Compatible with SuperSplat, playcanvas, polycam.
- `"splat"`: Compact binary for antimatter15 web viewer.
- `"compressed"`: Compressed PLY (quantised SH/positions) + PngCompression (236MB -> 16.5MB for 1M Gaussians).
- `"gltf"`: glTF 2.0 / GLB mesh export (requires `trimesh` or `pygltflib`). Exports the SuGaR/TSDF mesh with optional texture.

### 8K Video Handling
Videos wider than 1920px are auto-converted to 1080p H.264 before frame extraction. The 8K original is preserved for texture baking. This prevents the pipeline from hanging on stage 1 (8K HEVC decode is extremely slow).

### Sensor Logger Integration
Samsung Sensor Logger ZIPs contain CSV files with pre-fused orientation quaternions at 183Hz. These are used directly as COLMAP rotation priors (bypassing the Madgwick filter). The barometer data provides camera height (Y-translation) priors.

### FLAME Model
Located at `Models/Flame/flame2023_Open.pkl`. Loaded via pickle with `encoding='latin1'` fallback. The `mediapipe_landmark_embedding.npz` maps MediaPipe landmarks to FLAME mesh vertices via barycentric coordinates.

### Training Defaults (optimized for RTX 3080 16GB)
- 3,000 iterations (not 30K — dense init + progressive resolution converges fast)
- `packed=False` (30% faster than True; True saves up to 4x memory for large scenes)
- `sh_degree_max=1` (sufficient for faces, 15-25% faster than SH3)
- `max_num_gaussians=300,000` (VRAM safety cap)
- 3-stage progressive resolution: 1/4 → 1/2 → full
- Staged loss: L1+depth first, add DSSIM at 15%, add normal+distortion at 40%

### Tauri IPC
Frontend calls Rust via `invoke()` from `@tauri-apps/api/core`. The `src/lib/api.ts` provides a type-safe wrapper with retry logic, caching (TTL-based), and dev-mode mock data fallback (checks `window.__TAURI_INTERNALS__`).

### Parallel Processing
CPU-bound stages use `ProcessPoolExecutor` via `src/utils/parallel.py`. MediaPipe is NOT thread/process-safe — always runs on the main process. Images >1920px are auto-downscaled before MediaPipe (200MP photos were taking 33s each without this).

## Config

`config/pipeline.yaml` — all hyperparameters. Key sections:
- `capture.max_frames`: 80 (default for 10-15s clips)
- `reconstruction.use_da3_unified`: true (set false to use COLMAP instead)
- `reconstruction.da3.process_res`: 0 (auto-select 336 for face close-ups, 504 for wide)
- `splatting.training.iterations`: 3000
- `splatting.export.formats`: ["ply", "splat"] (export formats: ply, splat, compressed, gltf)
- `photos.multi_lens.main_200mp_weight`: 5.0 (200MP photos weighted 5x during training)

## Key Tech Stack

- **Depth Anything 3** — depth + camera poses + confidence in one pass
- **COLMAP 3.9.1** (CUDA) — fallback SfM at `COLMAP/COLMAP-3.9.1-windows-cuda/COLMAP.bat`
- **FLAME** — parametric face model (5023 vertices, 300 shape + 100 expression params)
- **gsplat 1.4.0** — 2DGS rasterization, DefaultStrategy/MCMCStrategy, SelectiveAdam
- **MediaPipe 0.10.14** — 478-point face mesh (pinned version)
- **Tauri 2.0** — desktop app (React + Rust)
- **PyTorch 2.5.1** + CUDA 12.1, GPU: RTX 3080 16GB
