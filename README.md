<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.x-76b900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/License-GPL--3.0-blue?logo=gnu&logoColor=white" />
</p>

<h1 align="center">Face3D</h1>

<p align="center">
  <strong>Next-gen 3D face reconstruction from phone video.</strong><br>
  Samsung Galaxy S25 Ultra &rarr; 2D Gaussian Splatting &rarr; Animated 3D Face
</p>

<p align="center">
  <em>Capture a 20-second orbit video. Get a photorealistic, animatable 3D face model.</em>
</p>

---

## What It Does

Face3D takes a **video** of someone's face (shot on a Samsung Galaxy S25 Ultra) and reconstructs a full **3D Gaussian Splat model** with:

- **Photorealistic rendering** from any angle via 2D Gaussian Splatting
- **Clean triangle mesh** with 4K UV texture via SuGaR extraction
- **Facial animation** driven by FLAME expression parameters (smile, surprise, etc.)
- **Compressed export** for web/AR viewing (15x smaller files)

Optionally feed in **Expert RAW photos** alongside video for even sharper texture detail.

---

## Pipeline

```
                        Samsung Galaxy S25 Ultra
                     Video (.mp4) + Photos (.jpg/.dng)
                                  |
                    +-------------+-------------+
                    |                           |
            Frame Extraction              Photo Processing
            Motion-based sampling         Expert RAW / DNG
            HW-accelerated HEVC          EXIF lens matching
            S-Log3 color correction       High-res anchors
                    |                           |
                    +-------------+-------------+
                                  |
                        Quality Filtering
                        Blur / exposure / face detection
                        Smart frame selection (farthest-point)
                                  |
                    +-------------+-------------+
                    |                           |
              Depth Anything 3            COLMAP (fallback)
              Depth + Poses + Confidence  Structure-from-Motion
              Single inference pass       Face-optimized settings
              Replaces COLMAP             SuperPoint/SuperGlue
                    |                           |
                    +-------------+-------------+
                                  |
                      Face Reconstruction
                      MediaPipe 478 landmarks
                      FLAME parametric fitting
                      Face segmentation + GrabCut
                                  |
                    +-------------+-------------+
                    |                           |
          FLAME-Bound Gaussians          Free Gaussians
          30K on mesh triangles          10K for hair/ears
          Animatable via FLAME           Move freely
          Barycentric coordinates        No mesh binding
                    |                           |
                    +-------------+-------------+
                                  |
                      2D Gaussian Splatting
                      Normal consistency loss
                      Distortion regularization
                      LPIPS perceptual loss
                      Depth supervision
                      Appearance embedding
                      Progressive training
                                  |
                    +-------------+-------------+
                    |             |             |
              Gaussian PLY   SuGaR Mesh   Animation
              + compressed   + 4K texture  Smile, surprise
              15x smaller    UV-baked      Head turn
                             .obj + .mtl   FLAME-driven
                                  |
                          Quality Report
                          HTML with comparisons
                          Turntable GIF
                          Stage timing
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate face3d

# Install CUDA-dependent packages
pip install gsplat pycolmap
pip install git+https://github.com/bytedance-seed/depth-anything-3.git --no-deps
pip install addict evo moviepy==1.0.3 einops huggingface_hub safetensors

# Download FLAME model (required, free registration)
# https://flame.is.tue.mpg.de → Download → place in Models/Flame/
```

### 2. Capture

Record a **20-second orbit video** of the subject's face with your S25 Ultra:
- Pro Video mode, 4K or 8K, 30fps
- Slowly orbit from ear to ear (frontal → left profile → back → right profile → frontal)
- Keep ~1.5m distance, maintain consistent height
- Even diffuse lighting (overcast or ring light)
- Subject keeps a neutral expression and stays still

Optionally take **5-10 Expert RAW photos** at key angles (front, 3/4, profile).

### 3. Run

```bash
# Video only
python scripts/run_pipeline.py --video path/to/face_video.mp4

# Video + Expert RAW photos (higher quality textures)
python scripts/run_pipeline.py --video face_video.mp4 --photos photo1.jpg photo2.jpg

# Resume from a specific stage
python scripts/run_pipeline.py --video face_video.mp4 --session SESSION_ID --start-stage 13

# Force re-run a stage
python scripts/run_pipeline.py --video face_video.mp4 --session SESSION_ID --start-stage 10 --end-stage 10 --force
```

### 4. Output

```
data/output/{session}/
├── gaussians.ply                # Full Gaussian splat model
├── gaussians_compressed.ply     # Compressed (15x smaller)
├── mesh.ply                     # SuGaR Poisson mesh
├── mesh_textured.obj            # UV-textured mesh
├── texture.png                  # 4K face texture map
├── renders/                     # Turntable views + video
├── animations/                  # Smile, surprise, head turn videos
├── previews/                    # Depth maps, landmarks, masks
├── report.html                  # Self-contained quality report
└── metrics.jsonl                # Per-stage metrics
```

---

## Architecture

### Modules

| Module | Purpose |
|--------|---------|
| `src/capture/` | Video frame extraction, Expert RAW processing, quality filtering, smart frame selection |
| `src/sensors/` | Samsung IMU parsing, Madgwick orientation filter, COLMAP rotation priors |
| `src/depth/` | Depth Anything 3 unified (depth + poses), DA2 fallback, scale alignment |
| `src/reconstruction/` | COLMAP SfM, MediaPipe landmarks, FLAME fitting, face segmentation |
| `src/splatting/` | 2DGS training, FLAME-bound initialization, SuGaR mesh, texture baking, animation |
| `src/utils/` | COLMAP I/O, camera profiles, quality reports, visualization |

### Key Technologies

| Component | What | Why |
|-----------|------|-----|
| **[Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3)** | Depth + camera poses in one pass | Replaces COLMAP, 35% better pose accuracy |
| **[gsplat](https://github.com/nerfstudio-project/gsplat)** | 2D Gaussian Splatting | Flat-disc Gaussians for better surface reconstruction |
| **[FLAME](https://flame.is.tue.mpg.de)** | Parametric face model | 5,023 vertices, expression-driven animation |
| **[MediaPipe](https://github.com/google-ai-edge/mediapipe)** | 478-point face landmarks | Bridge between 2D detection and 3D FLAME fitting |
| **[COLMAP](https://colmap.github.io)** | Structure-from-Motion | Fallback when DA3 is unavailable |

### Training Innovations

- **2D Gaussian Splatting** — flat disc primitives for better face surfaces
- **6 loss functions** — L1, D-SSIM, normal consistency, distortion, LPIPS, depth supervision
- **FLAME-bound initialization** — Gaussians on mesh triangles with barycentric coordinates
- **Appearance embedding** — per-frame learned exposure/white-balance correction
- **Progressive training** — start at half resolution, scale up at 50% iterations
- **Photo-weighted sampling** — Expert RAW views sampled 3x more during training

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1080 (8GB) | RTX 3080+ (16GB) |
| VRAM | 8 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB free | 50 GB free |
| OS | Windows 10/11 | Windows 11 |
| CUDA | 11.8+ | 12.x |
| Phone | Any with video | Samsung Galaxy S25 Ultra |

---

## Configuration

All parameters are in `config/pipeline.yaml`:

```yaml
# Key settings to tune
capture:
  max_frames: 300
  filtering:
    blur_threshold: 30.0       # Lower = accept softer frames
    require_face: true

splatting:
  training:
    iterations: 7000           # 3000 = quick test, 30000 = max quality
    max_num_gaussians: 500000  # VRAM cap for 16GB GPU
  export:
    texture_resolution: 4096   # UV texture map size
    turntable_views: 60
```

---

## Error Handling

The pipeline is designed to be resilient:

- **OOM Recovery** — automatically reduces Gaussian count and retries (up to 3x)
- **NaN Detection** — restores last good checkpoint, halves learning rate
- **Gradient Clipping** — prevents training explosions
- **Stage Validation** — checks outputs between stages, fails fast with helpful messages
- **Graceful Fallbacks** — DA3 → COLMAP, 2DGS → 3DGS, FLAME → sparse points, SuGaR → TSDF

---

## Project Structure

```
Face3D/
├── config/
│   ├── pipeline.yaml              # Master configuration
│   ├── expressions.yaml           # FLAME expression presets
│   └── camera_profiles/
│       └── s25_ultra.yaml         # Samsung S25 Ultra specs
├── scripts/
│   ├── run_pipeline.py            # Pipeline orchestrator (14 stages)
│   └── setup_environment.sh       # Automated env setup
├── src/
│   ├── capture/                   # Video + photo ingestion
│   ├── sensors/                   # IMU integration
│   ├── depth/                     # DA3 + DA2 depth estimation
│   ├── reconstruction/            # COLMAP, FLAME, segmentation
│   ├── splatting/                 # Training, export, animation
│   └── utils/                     # Shared utilities
├── Models/                        # Pretrained weights (not in repo)
│   └── Flame/                     # FLAME model files
├── data/                          # Pipeline I/O (not in repo)
│   ├── raw/                       # Input videos + photos
│   ├── processed/                 # Intermediate results
│   └── output/                    # Final outputs
├── environment.yaml               # Conda environment spec
├── LICENSE                        # GPL-3.0
└── README.md
```

---

## License

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for details.

Third-party dependencies have their own licenses:
- **FLAME** — Requires separate registration at [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de)
- **Depth Anything 3** — Apache 2.0
- **gsplat** — Apache 2.0
- **MediaPipe** — Apache 2.0
- **COLMAP** — BSD 3-Clause

---

<p align="center">
  Built by <a href="https://www.linkedin.com/in/nickalusbrewer/">Nickalus Brewer</a>
</p>
