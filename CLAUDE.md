# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Face3D** — Next-gen 3D face reconstruction pipeline. Takes Samsung Galaxy S25 Ultra Pro Video (multi-angle face capture) and produces photorealistic 3D face models via 2D Gaussian Splatting with FLAME-bound initialization and SuGaR mesh extraction.

## Architecture

### Pipeline (14 stages, orchestrated by `scripts/run_pipeline.py`)

```
S25 Ultra Video (.mp4) + optional IMU sensor data
  │
  ├─ Stage 1-3: src/capture/       → Frame extraction, LOG color correction, quality filtering
  ├─ Stage 4-5: src/sensors/       → IMU parsing, Madgwick filter, COLMAP rotation priors
  ├─ Stage 6:   src/depth/da3_unified.py → DA3: depth + poses + point cloud (replaces COLMAP+DA2)
  │             (fallback: src/reconstruction/colmap_runner.py → COLMAP SfM)
  ├─ Stage 7-8: (auto-skipped when DA3 unified is used)
  ├─ Stage 9-11: src/reconstruction/ → MediaPipe landmarks, FLAME fitting, face segmentation
  ├─ Stage 12:  src/splatting/initializer.py → FLAME-bound Gaussian init (GaussianAvatars-style)
  ├─ Stage 13:  src/splatting/trainer.py → 2DGS training with face losses
  └─ Stage 14:  src/splatting/exporter.py → SuGaR mesh + compressed PLY + turntable renders
```

### Key Innovations (vs vanilla 3DGS)
- **DA3 unified**: Replaces COLMAP + Depth Anything + alignment with single DA3 inference (depth + poses + intrinsics in one pass)
- **2D Gaussian Splatting**: Flat disc Gaussians for better surface reconstruction (gsplat `rasterization_2dgs`)
- **FLAME-bound Gaussians**: ~30K Gaussians bound to FLAME mesh triangles + ~10K free for hair/ears, enabling animation
- **Face losses**: Normal consistency + distortion + LPIPS + depth supervision + opacity entropy
- **Appearance embedding**: Per-frame learned correction for exposure/white-balance drift
- **Progressive training**: Start at half resolution, switch to full at 50% iterations
- **SuGaR mesh**: Poisson surface reconstruction from flat Gaussians, clean watertight meshes
- **Compression**: gsplat PngCompression (236MB → 16.5MB per 1M Gaussians)

### Data Flow
```
data/raw/{session}/video.mp4
  → data/processed/{session}/frames, depth, landmarks, flame, masks, colmap/
    → data/output/{session}/gaussians.ply, gaussians_compressed.ply, mesh.ply, mesh.obj, renders/
```

Stage completion tracked via `.stage_N_complete` marker files — fully resumable.

### Key Source Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `src/capture/` | Video ingestion | `extract_frames_motion_based`, `batch_color_correct`, `filter_frames` |
| `src/sensors/` | IMU integration | `MadgwickFilter`, `ComplementaryFilter`, `parse_imu_from_video` |
| `src/depth/` | Depth + poses | `run_da3_unified` (primary), `DepthEstimator`, `batch_align` |
| `src/reconstruction/` | Face reconstruction | `run_colmap` (fallback), `FaceLandmarkDetector`, `FLAMEModel`, `FLAMEFitter`, `FaceSegmenter` |
| `src/splatting/` | Gaussian Splatting | `initialize_from_flame_binding`, `GaussianTrainer` (2DGS), `extract_mesh_sugar`, `export_compressed` |
| `src/utils/` | Shared utilities | `colmap_io`, `camera` (S25 Ultra profiles) |

### FLAME Model Files (`Models/Flame/`)
- `flame2023_Open.pkl` — Commercial-use FLAME model (default)
- `FLAME2023/flame2023.pkl` — Academic model with improved eye region
- `FLAME_masks.pkl` — Vertex region masks
- `mediapipe_landmark_embedding.npz` — MediaPipe → FLAME vertex mapping
- `FLAME_texture.npz` — UV texture space

### Config (`config/`)
- `pipeline.yaml` — Master config (all hyperparameters)
- `camera_profiles/s25_ultra.yaml` — S25 Ultra lens/IMU specs

## Commands

```bash
# Environment setup
conda env create -f environment.yaml
conda activate face3d

# Run full pipeline
python scripts/run_pipeline.py --video path/to/video.mp4

# Resume from specific stage
python scripts/run_pipeline.py --video path/to/video.mp4 --session SESSION_ID --start-stage 7

# Force re-run a stage
python scripts/run_pipeline.py --video path/to/video.mp4 --session SESSION_ID --start-stage 13 --force
```

## Key Tech Stack

- **Depth Anything 3** — Unified depth + camera poses + confidence (replaces COLMAP for pose estimation)
- **COLMAP 3.9.1** (CUDA) — Fallback SfM, binary at `COLMAP/COLMAP-3.9.1-windows-cuda/COLMAP.bat`
- **FLAME** — Parametric face model (5023 vertices, 300 shape + 100 expression params)
- **gsplat 1.4.0** — 2DGS + 3DGS rasterization, DefaultStrategy/MCMCStrategy, compression, export
- **MediaPipe 0.10.14** — 478-point face mesh landmarks (pinned — newer versions broke API)
- **PyTorch 2.5.1** + CUDA 12.1 — GPU compute
- **GPU**: RTX 3080 16GB — VRAM cap at 500K Gaussians, chunk processing for DA3

### CUDA Environment (required for gsplat JIT compilation)
```
CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
PATH includes VS 2022 Community cl.exe (14.38) and CUDA bin
TORCH_CUDA_ARCH_LIST="8.6"
```

## CAD (separate from pipeline)
- `dryer_bearing.py` — Fusion 360 script for GE dryer bearing (WE3M26). Runs inside Fusion 360, not standalone.
- `Designs/Dryer/` — Exported .3mf models
