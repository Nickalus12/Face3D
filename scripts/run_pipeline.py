#!/usr/bin/env python3
"""Face3D Pipeline Orchestrator

Runs the full face reconstruction pipeline from S25 Ultra video to Gaussian Splats.
Each stage checks for existing outputs and skips if already complete (resumable).

Usage:
    python scripts/run_pipeline.py --video path/to/video.mp4
    python scripts/run_pipeline.py --video path/to/video.mp4 --session my_session
    python scripts/run_pipeline.py --session my_session --start-stage 4  # Resume from stage 4
    python scripts/run_pipeline.py --config config/pipeline.yaml --video path/to/video.mp4
    python scripts/run_pipeline.py --content-dir content/New  # Auto-detect all inputs
"""

import os
import sys

# ── CUDA environment setup (MUST happen before ANY CUDA/gsplat imports) ──
if os.name == "nt":
    from pathlib import Path as _P
    _cuda_home = os.environ.get("CUDA_HOME", "")
    if not _cuda_home:
        for _ver in ("v12.8", "v12.6", "v12.4", "v12.1"):
            _c = f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{_ver}"
            if _P(_c).exists():
                os.environ["CUDA_HOME"] = _c
                _cuda_home = _c
                break
    if _cuda_home:
        _cb = str(_P(_cuda_home) / "bin")
        if _cb not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _cb + ";" + os.environ.get("PATH", "")
        _mb = _P("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC")
        if _mb.exists():
            for _mv in sorted(_mb.iterdir(), reverse=True):
                _cl = _mv / "bin/Hostx64/x64/cl.exe"
                if _cl.exists():
                    _cd = str(_cl.parent)
                    if _cd not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = _cd + ";" + os.environ.get("PATH", "")
                    break
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

import argparse
import json
import logging
import shutil
import struct
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

# Disable PIL decompression bomb check globally (200MP Samsung photos = 199M pixels)
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# CUDA env already set at top of file (before any imports)
# Early CUDA/gsplat validation — fail within 2 seconds, not after 20 minutes
from splatting.cuda_check import check_gsplat_cuda as _check_cuda
_cuda_ok, _cuda_err, _cuda_backend = _check_cuda()
if not _cuda_ok:
    print(f"[WARNING] gsplat CUDA not available: {_cuda_err}")
    print("[WARNING] GPU training (Stage 13) will fail. Fix CUDA setup first.")

from utils.timing import PipelineProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,  # Force stdout — Tauri captures both stdout+stderr causing duplicates
    force=True,
)
log = logging.getLogger("face3d")


# ---------------------------------------------------------------------------
# Config & session helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_config(config: dict, video_paths: list[Path]):
    """Validate configuration before pipeline starts. Exits on fatal errors."""
    errors = []
    warnings = []

    # Check video files exist and are readable
    for vp in video_paths:
        if not vp.exists():
            errors.append(f"Video file does not exist: {vp}")
        else:
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(vp)],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode != 0:
                    warnings.append(f"ffprobe could not read video: {vp} ({result.stderr.strip()})")
                else:
                    dur = float(result.stdout.strip())
                    log.info("Video %s: duration %.1fs", vp.name, dur)
            except FileNotFoundError:
                warnings.append("ffprobe not found on PATH; skipping video probe")
            except Exception as e:
                warnings.append(f"ffprobe check failed for {vp}: {e}")

    # Check CUDA availability
    device = config.get("device", "cuda")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("Config specifies device='cuda' but CUDA is not available")
            else:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
                log.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)
        except ImportError:
            errors.append("PyTorch not installed; cannot use CUDA device")

    # Check FLAME model files
    flame_cfg = config.get("reconstruction", {}).get("flame", {})
    for key, label in [("model_path", "FLAME model"), ("embedding_path", "MediaPipe embedding")]:
        path = flame_cfg.get(key)
        if path and not Path(path).exists():
            warnings.append(f"{label} not found: {path}")

    # Check COLMAP binary (only if not using DA3)
    use_da3 = config.get("reconstruction", {}).get("use_da3_unified", True)
    if not use_da3:
        colmap_bin = config.get("reconstruction", {}).get("colmap", {}).get("binary")
        if colmap_bin and not Path(colmap_bin).exists():
            errors.append(f"COLMAP binary not found: {colmap_bin}")

    for w in warnings:
        log.warning("Config: %s", w)

    if errors:
        for e in errors:
            log.error("Config: %s", e)
        log.error("Fix the above configuration errors before running the pipeline.")
        sys.exit(1)


def setup_session(config: dict, session_id: str | None, video_paths: list[str]) -> dict:
    """Create session directory structure and return session info.

    Creates the full professional folder layout under data/{raw,processed,output}/{session}/.
    """
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_root = Path(config["session"]["data_root"])
    raw_dir = data_root / "raw" / session_id
    proc_dir = data_root / "processed" / session_id
    out_dir = data_root / "output" / session_id

    for d in [
        # Raw input structure
        raw_dir / "video",
        raw_dir / "photos" / "wide" / "dng",
        raw_dir / "photos" / "wide" / "jpg",
        raw_dir / "photos" / "main" / "dng",
        raw_dir / "photos" / "main" / "jpg",
        raw_dir / "photos" / "other",
        raw_dir / "sensors" / "video_synced",
        raw_dir / "sensors" / "photo_session",
        # Processed structure
        proc_dir / "frames",
        proc_dir / "frames_srgb",
        proc_dir / "photos" / "wide_processed",
        proc_dir / "photos" / "main_quarter",
        proc_dir / "photos" / "main_fullres",
        proc_dir / "photos" / "main_face_crops",
        proc_dir / "sensors",
        proc_dir / "depth",
        proc_dir / "depth_aligned",
        proc_dir / "colmap" / "sparse",
        proc_dir / "landmarks",
        proc_dir / "face_masks",
        proc_dir / "flame",
        # Output structure
        out_dir / "renders",
        out_dir / "animations",
        out_dir / "previews",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "session_id": session_id,
        "raw_dir": raw_dir,
        "proc_dir": proc_dir,
        "out_dir": out_dir,
        "video_paths": [Path(v) for v in video_paths],
    }


def stage_complete(marker_path: Path) -> bool:
    return marker_path.exists()


def mark_stage_complete(marker_path: Path):
    marker_path.write_text(datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

def log_metrics(session: dict, stage_num: int, elapsed_s: float, metrics: dict):
    """Append a metrics line to data/output/{session}/metrics.jsonl."""
    metrics_path = session["out_dir"] / "metrics.jsonl"
    entry = {
        "stage": stage_num,
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": round(elapsed_s, 1),
        "metrics": metrics,
    }
    with open(metrics_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Stage output validation
# ---------------------------------------------------------------------------

def _count_files(directory: Path, pattern: str) -> int:
    return len(list(directory.glob(pattern)))


def _read_colmap_images_bin(images_bin: Path) -> int:
    """Read COLMAP images.bin and return number of registered images."""
    try:
        with open(images_bin, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
        return num_images
    except Exception:
        return 0


def _read_colmap_points_bin(points_bin: Path) -> int:
    """Read COLMAP points3D.bin and return number of 3D points."""
    try:
        with open(points_bin, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
        return num_points
    except Exception:
        return 0


def validate_stage_output(stage_num: int, config: dict, session: dict) -> dict:
    """Validate outputs after a stage completes. Returns metrics dict.

    Raises RuntimeError on fatal validation failures.
    Returns a dict of metrics to be logged.
    """
    proc = session["proc_dir"]
    metrics = {}

    if stage_num == 1:
        n_frames = _count_files(proc / "frames", "*.png") + _count_files(proc / "frames", "*.jpg")
        metrics["frames_extracted"] = n_frames
        if n_frames < 10:
            raise RuntimeError(
                f"Not enough frames extracted ({n_frames}). "
                "Try a longer video or lower motion threshold."
            )
        log.info("Validation: %d frames extracted", n_frames)

    elif stage_num == 3:
        selected_json = proc / "selected_frames.json"
        n_selected = 0
        if selected_json.exists():
            with open(selected_json) as f:
                selected = json.load(f)
            n_selected = len(selected) if isinstance(selected, list) else selected.get("count", 0)
        # Also count remaining files in frames_srgb as a fallback metric
        n_srgb = _count_files(proc / "frames_srgb", "*.png") + _count_files(proc / "frames_srgb", "*.jpg")
        n_selected = max(n_selected, n_srgb)
        metrics["frames_selected"] = n_selected
        if n_selected < 5:
            log.warning(
                "Only %d frames passed filtering. Will retry with lower blur_threshold (20).",
                n_selected,
            )
            # Signal retry needed
            metrics["_retry_filter"] = True
        log.info("Validation: %d frames passed quality filter", n_selected)

    elif stage_num == 6:
        sparse_dir = proc / "colmap" / "sparse" / "0"
        da3_marker = proc / ".da3_unified_complete"

        if da3_marker.exists():
            # DA3 unified path -- read marker for stats
            marker_text = da3_marker.read_text()
            metrics["method"] = "da3_unified"
            for part in marker_text.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        metrics[k] = int(v)
                    except ValueError:
                        metrics[k] = v
            log.info("Validation (DA3): %s", marker_text)
        elif sparse_dir.exists():
            cameras_bin = sparse_dir / "cameras.bin"
            images_bin = sparse_dir / "images.bin"
            points_bin = sparse_dir / "points3D.bin"
            if not cameras_bin.exists() or not images_bin.exists():
                raise RuntimeError(
                    "COLMAP reconstruction failed: cameras.bin or images.bin missing in "
                    f"{sparse_dir}. Check that frames have enough texture/overlap."
                )
            n_registered = _read_colmap_images_bin(images_bin)
            n_points = _read_colmap_points_bin(points_bin) if points_bin.exists() else 0
            n_input = (
                _count_files(proc / "frames_srgb", "*.png")
                + _count_files(proc / "frames_srgb", "*.jpg")
            )
            metrics["method"] = "colmap"
            metrics["registered_images"] = n_registered
            metrics["input_images"] = n_input
            metrics["sparse_points"] = n_points
            log.info(
                "Validation: COLMAP registered %d/%d images, %d sparse points",
                n_registered, n_input, n_points,
            )
            if n_registered < 3:
                raise RuntimeError(
                    f"COLMAP only registered {n_registered} images (need >= 3). "
                    "Ensure the video has sufficient camera motion and scene overlap. "
                    "Try increasing max_num_features or using sequential matching."
                )
        else:
            raise RuntimeError(
                f"No reconstruction output found. Expected {sparse_dir} or DA3 marker."
            )

    elif stage_num == 7:
        n_depth = _count_files(proc / "depth", "*.npy")
        metrics["depth_maps"] = n_depth
        if n_depth < 3:
            raise RuntimeError(
                f"Only {n_depth} depth maps produced (need >= 3). "
                "Check depth estimation logs."
            )
        log.info("Validation: %d depth maps generated", n_depth)

    elif stage_num == 9:
        lm_dir = proc / "landmarks"
        n_json = _count_files(lm_dir, "*.json")
        n_with_face = 0
        n_no_face = 0
        for jf in lm_dir.glob("*.json"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if data and (isinstance(data, list) and len(data) > 0) or (isinstance(data, dict) and data.get("landmarks")):
                    n_with_face += 1
                else:
                    n_no_face += 1
            except Exception:
                n_no_face += 1
        metrics["landmark_files"] = n_json
        metrics["faces_detected"] = n_with_face
        metrics["no_face"] = n_no_face
        log.info(
            "Validation: %d landmark files (%d with faces, %d without)",
            n_json, n_with_face, n_no_face,
        )

    elif stage_num == 10:
        flame_params = proc / "flame" / "flame_params.npz"
        metrics["flame_fitted"] = flame_params.exists()
        if not flame_params.exists():
            log.warning("FLAME params file not found at %s. Continuing without FLAME.", flame_params)

    elif stage_num == 12:
        init_path = proc / "gaussians_init.pt"
        if init_path.exists():
            import torch
            data = torch.load(str(init_path), weights_only=False, map_location="cpu")
            n_gaussians = len(data.get("positions", []))
            metrics["initial_gaussians"] = n_gaussians
            if n_gaussians < 100:
                log.warning(
                    "Only %d initial Gaussians (< 100). Quality may be poor. "
                    "Consider using a denser point cloud or FLAME mesh.",
                    n_gaussians,
                )
            log.info("Validation: %d initial Gaussians", n_gaussians)

    return metrics


# ---------------------------------------------------------------------------
# Helper: DA3 unified
# ---------------------------------------------------------------------------

def _use_da3_unified(config: dict) -> bool:
    """Check if the pipeline should use the unified DA3 stage."""
    return config.get("reconstruction", {}).get("use_da3_unified", True)


def _run_da3_unified_stage(config: dict, session: dict) -> bool:
    """Run the unified DA3 stage replacing COLMAP + depth + alignment."""
    from depth.da3_unified import run_da3_unified, da3_provides_enough_views

    depth_cfg = config.get("depth", {})
    recon_cfg = config.get("reconstruction", {})

    model_size = depth_cfg.get("model_size", "vitl")
    size_to_da3 = {"vitl": "da3-large", "vitb": "da3-base", "vits": "da3-small", "vitg": "da3-giant"}
    model_name = size_to_da3.get(model_size, model_size)
    if depth_cfg.get("da3_model"):
        model_name = depth_cfg["da3_model"]

    da3_cfg = recon_cfg.get("da3", {})
    process_res = da3_cfg.get("process_res", 0)
    use_ray_pose = da3_cfg.get("use_ray_pose", True)
    chunk_size = da3_cfg.get("chunk_size", 16)
    conf_threshold = da3_cfg.get("conf_threshold", 0.0)
    conf_percentile = da3_cfg.get("conf_percentile", 30.0)
    deduplicate = da3_cfg.get("deduplicate", True)
    dedup_threshold = da3_cfg.get("dedup_threshold", 5)
    use_streaming_ply = da3_cfg.get("use_streaming_ply", True)
    checkpoint_version = da3_cfg.get("checkpoint_version", "auto")
    use_streaming = da3_cfg.get("use_streaming", False)
    da3_direct_gaussians = da3_cfg.get("da3_direct_gaussians", False)
    oom_fallback = da3_cfg.get("oom_fallback", "reduce_resolution")

    summary = run_da3_unified(
        frames_dir=session["proc_dir"] / "frames_srgb",
        output_dir=session["proc_dir"],
        model_name=model_name,
        process_res=process_res,
        use_ray_pose=use_ray_pose,
        chunk_size=chunk_size,
        conf_threshold=conf_threshold,
        deduplicate=deduplicate,
        dedup_threshold=dedup_threshold,
        conf_percentile=conf_percentile,
        use_streaming_ply=use_streaming_ply,
        checkpoint_version=checkpoint_version,
        use_streaming=use_streaming,
        da3_direct_gaussians=da3_direct_gaussians,
        oom_fallback=oom_fallback,
    )

    log.info("DA3 unified summary: %s", summary)

    if not da3_provides_enough_views(session["proc_dir"], min_views=5):
        log.error("DA3 did not produce enough views for reconstruction")
        return False

    (session["proc_dir"] / ".da3_unified_complete").write_text(
        f"frames={summary['num_frames']} poses={summary['num_valid_poses']} "
        f"points={summary['num_points']}"
    )

    return True


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _organize_raw_folder(session: dict) -> None:
    """Create professional folder structure under data/raw/{session}/ and
    copy source files into it.

    Structure:
        data/raw/{session}/
        +-- video/                  # Source video files
        +-- photos/
        |   +-- wide/dng/           # Wide lens Expert RAW
        |   +-- wide/jpg/
        |   +-- main/dng/           # 200MP main sensor
        |   +-- main/jpg/
        |   +-- other/
        +-- sensors/
        |   +-- video_synced/       # Sensor log matched to video
        |   +-- photo_session/      # Sensor log during photo capture
        +-- session_manifest.json
    """
    raw_dir = session["raw_dir"]

    # Create subdirectories
    dirs = {
        "video": raw_dir / "video",
        "photos_wide_dng": raw_dir / "photos" / "wide" / "dng",
        "photos_wide_jpg": raw_dir / "photos" / "wide" / "jpg",
        "photos_main_dng": raw_dir / "photos" / "main" / "dng",
        "photos_main_jpg": raw_dir / "photos" / "main" / "jpg",
        "photos_other": raw_dir / "photos" / "other",
        "sensors_video": raw_dir / "sensors" / "video_synced",
        "sensors_photo": raw_dir / "sensors" / "photo_session",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Copy video files
    for vp in session["video_paths"]:
        dst = dirs["video"] / vp.name
        if not dst.exists():
            shutil.copy2(vp, dst)
            log.info("  Copied video: %s", vp.name)

    # Copy photos (will be classified by lens after EXIF analysis)
    for pp in session.get("photo_paths", []):
        pp = Path(pp)
        suffix = pp.suffix.lower()
        # Temporarily copy to 'other' — stage 0 will reclassify after EXIF
        dst = dirs["photos_other"] / pp.name
        if not dst.exists():
            shutil.copy2(pp, dst)

    # Copy sensor logs
    for sp in session.get("sensor_log_paths", []):
        sp = Path(sp)
        dst = dirs["sensors_video"] / sp.name
        if not dst.exists():
            shutil.copy2(sp, dst)
            log.info("  Copied sensor log: %s", sp.name)


def _reclassify_photos_to_raw(session: dict, manifest: dict) -> None:
    """Move photos from other/ into the correct lens subdirectory under raw/.

    Uses the lens classification from the manifest produced by
    organize_capture_session.
    """
    raw_dir = session["raw_dir"]
    photos_other = raw_dir / "photos" / "other"

    lens_dir_map = {
        "wide_23mm": "wide",
        "main_200mp": "main",
        "ultrawide_13mm": "other",
        "telephoto_70mm": "other",
        "supertelephoto_200mm": "other",
        "unknown": "other",
    }

    for lens_name, photo_list in manifest.get("photos", {}).items():
        subdir = lens_dir_map.get(lens_name, "other")
        for entry in photo_list:
            stem = entry.get("stem", "")
            for ext_key, ext_subdir in [("dng", "dng"), ("jpg", "jpg")]:
                src_path = entry.get(ext_key)
                if src_path is None:
                    continue
                src = Path(src_path)
                # File might be in 'other' from initial copy, or still at original
                other_copy = photos_other / src.name
                target_dir = raw_dir / "photos" / subdir / ext_subdir
                target_dir.mkdir(parents=True, exist_ok=True)
                dst = target_dir / src.name
                if dst.exists():
                    continue
                if other_copy.exists():
                    shutil.move(str(other_copy), str(dst))
                elif src.exists():
                    shutil.copy2(src, dst)


def _compute_capture_quality_score(manifest: dict) -> dict:
    """Compute a 0-100 capture quality score based on available data.

    Scores are based on:
    - Video duration and resolution (0-20)
    - Photo count and resolution mix (0-25)
    - Sensor data availability (0-20)
    - Multi-lens coverage (0-15)
    - Lighting sensor availability for consistency check (0-10)
    - Overall data richness bonus (0-10)

    Returns dict with total_score, breakdown, and assessment string.
    """
    breakdown = {}
    total = 0

    # --- Video quality (0-20) ---
    video = manifest.get("video", {})
    video_score = 0
    dur = video.get("duration", 0)
    res = video.get("resolution", [0, 0])
    if dur > 0:
        video_score += min(5, dur / 6)  # 30s = 5pts
        pixels = res[0] * res[1]
        if pixels >= 7680 * 4320:  # 8K
            video_score += 10
        elif pixels >= 3840 * 2160:  # 4K
            video_score += 7
        elif pixels >= 1920 * 1080:  # 1080p
            video_score += 4
        fps = video.get("fps", 0)
        if fps >= 60:
            video_score += 5
        elif fps >= 30:
            video_score += 3
    breakdown["video"] = round(min(20, video_score), 1)
    total += breakdown["video"]

    # --- Photo quality (0-25) ---
    photos = manifest.get("photos", {})
    n_total_photos = sum(len(v) for v in photos.values())
    n_200mp = len(photos.get("main_200mp", []))
    photo_score = 0
    if n_total_photos > 0:
        photo_score += min(10, n_total_photos)  # 10 photos = 10pts
        # 200MP bonus
        if n_200mp > 0:
            photo_score += min(10, n_200mp * 2)  # 5 200MP photos = 10pts
        # Wide-lens photos
        n_wide = len(photos.get("wide_23mm", []))
        if n_wide > 0:
            photo_score += min(5, n_wide)
    breakdown["photos"] = round(min(25, photo_score), 1)
    total += breakdown["photos"]

    # --- Sensor data (0-20) ---
    sensors = manifest.get("sensors", {})
    sensor_score = 0
    if sensors.get("video_synced"):
        sensor_score += 5  # Has a synced sensor log
    for key in ("accelerometer_hz", "gyroscope_hz", "orientation_hz"):
        if sensors.get(key, 0) > 0:
            sensor_score += 3
    if sensors.get("has_gps"):
        sensor_score += 3
    if sensors.get("has_light"):
        sensor_score += 3
    breakdown["sensors"] = round(min(20, sensor_score), 1)
    total += breakdown["sensors"]

    # --- Multi-lens coverage (0-15) ---
    n_lenses = len(photos)
    lens_score = min(15, n_lenses * 5)
    breakdown["multi_lens"] = round(lens_score, 1)
    total += breakdown["multi_lens"]

    # --- Lighting consistency (0-10) ---
    # This will be refined in Stage 4 when light sensor data is parsed
    light_score = 5 if sensors.get("has_light") else 0
    breakdown["lighting"] = round(min(10, light_score), 1)
    total += breakdown["lighting"]

    # --- Data richness bonus (0-10) ---
    richness = 0
    if n_total_photos > 0 and dur > 0:
        richness += 3  # Both video and photos
    if sensors.get("video_synced") and n_total_photos > 0:
        richness += 3  # Sensors + photos
    if n_200mp > 0 and sensors.get("video_synced"):
        richness += 4  # Full pipeline data
    breakdown["richness"] = round(min(10, richness), 1)
    total += breakdown["richness"]

    total = round(min(100, total), 1)

    # Assessment string
    if total >= 85:
        assessment = f"Excellent ({total}/100) — full multi-source capture"
    elif total >= 65:
        assessment = f"Good ({total}/100) — solid data for reconstruction"
    elif total >= 40:
        assessment = f"Fair ({total}/100) — reconstruction possible but may lack detail"
    elif total >= 20:
        assessment = f"Minimal ({total}/100) — consider adding photos or sensor data"
    else:
        assessment = f"Basic ({total}/100) — video-only capture, limited quality"

    return {
        "total_score": total,
        "breakdown": breakdown,
        "assessment": assessment,
    }


def stage_0_organize_inputs(config: dict, session: dict) -> bool:
    """Stage 0: Organize multi-source inputs (video, photos, sensor logs).

    Creates a professional folder structure under data/raw/{session}/,
    classifies photos by lens, matches sensor logs to video, validates
    temporal overlap, creates multi-resolution outputs for 200MP photos,
    and writes a session_manifest.json.
    """
    marker = session["proc_dir"] / ".stage_0_complete"
    if stage_complete(marker):
        log.info("Stage 0: Input organization already complete, skipping")
        # Reload manifest into session
        manifest_path = session["proc_dir"] / "session_manifest.json"
        if manifest_path.exists():
            session["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return True

    photo_paths = session.get("photo_paths", [])
    sensor_log_paths = session.get("sensor_log_paths", [])

    # Skip if no multi-source data
    if not photo_paths and not sensor_log_paths:
        log.info("Stage 0: No photos or sensor logs provided, skipping organization")
        session["manifest"] = None
        mark_stage_complete(marker)
        return True

    log.info("Stage 0: Organizing multi-source capture data...")

    # Step 1: Create professional folder structure and copy source files
    _organize_raw_folder(session)

    from capture.data_organizer import (
        organize_capture_session,
        compute_multi_lens_camera_models,
        estimate_photo_timestamps,
    )

    # Step 2: Classify and create manifest
    manifest = organize_capture_session(
        video_paths=[str(v) for v in session["video_paths"]],
        photo_paths=[str(p) for p in photo_paths],
        sensor_log_paths=[str(s) for s in sensor_log_paths],
        output_dir=session["proc_dir"],
    )
    session["manifest"] = manifest

    # Step 3: Reclassify photos into lens-specific raw subdirectories
    _reclassify_photos_to_raw(session, manifest)

    # Step 4: Match sensor logs to video/photo sessions in raw structure
    sensors_info = manifest.get("sensors", {})
    video_synced = sensors_info.get("video_synced")
    photo_session_sensor = sensors_info.get("photo_session")
    if video_synced:
        # Copy the matched sensor log to video_synced/
        src = Path(video_synced)
        dst = session["raw_dir"] / "sensors" / "video_synced" / src.name
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
    if photo_session_sensor:
        src = Path(photo_session_sensor)
        dst = session["raw_dir"] / "sensors" / "photo_session" / src.name
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)

    # Also save the manifest into raw/ for reference
    raw_manifest_path = session["raw_dir"] / "session_manifest.json"
    raw_manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )

    # Step 5: Compute camera models for each lens
    camera_models = compute_multi_lens_camera_models(manifest)
    session["camera_models"] = camera_models

    cam_models_path = session["proc_dir"] / "camera_models.json"
    cam_models_path.write_text(
        json.dumps(camera_models, indent=2, default=str), encoding="utf-8",
    )
    log.info("Camera models saved: %s", cam_models_path)

    # Step 6: Estimate photo timestamps relative to video
    if photo_paths:
        photo_timestamps = estimate_photo_timestamps(
            photo_paths=photo_paths,
            video_start_time=None,  # Will be refined after video metadata is read
        )
        ts_path = session["proc_dir"] / "photo_timestamps.json"
        ts_path.write_text(
            json.dumps(photo_timestamps, indent=2, default=str), encoding="utf-8",
        )

    # Step 7: Process 200MP photos into multi-resolution tiers
    # NOTE: This is slow (~30s per DNG at 200MP). Skipped if outputs already exist.
    photos_200mp = manifest.get("photos", {}).get("main_200mp", [])
    if photos_200mp:
        # Check if already processed — skip if outputs exist (saves minutes)
        multi_res_output = session["proc_dir"] / "photos_200mp" / "quarter"
        existing_quarter = list(multi_res_output.glob("*.png")) if multi_res_output.exists() else []
        _skip_200mp = len(existing_quarter) >= len(photos_200mp)

        if _skip_200mp:
            log.info("Stage 0: 200MP photos already processed (%d files), skipping", len(existing_quarter))
            session["multi_res_200mp"] = {}
        else:
            log.info("Stage 0: Processing %d 200MP photos into multi-resolution tiers (this may take several minutes)...", len(photos_200mp))
            from capture.multi_res_processor import process_200mp_photos

            # Collect DNG paths (prefer DNG over JPG for 200MP)
            dng_paths = []
            for p in photos_200mp:
                dng = p.get("dng")
                jpg = p.get("jpg")
                if dng:
                    dng_paths.append(dng)
                elif jpg:
                    dng_paths.append(jpg)

            if dng_paths:
                multi_res_dir = session["proc_dir"] / "photos" / "main_quarter"
                fullres_dir = session["proc_dir"] / "photos" / "main_fullres"
                face_crops_dir = session["proc_dir"] / "photos" / "main_face_crops"

                # Configure output tiers to use the professional directory structure
                multi_res_results = process_200mp_photos(
                    photo_paths=dng_paths,
                    output_dir=session["proc_dir"] / "photos_200mp",
                )
                session["multi_res_200mp"] = multi_res_results

            # Also create symlinks/copies in the professional structure
            for tier_name, tier_dir in [
                ("full", fullres_dir),
                ("quarter", multi_res_dir),
                ("face_crops", face_crops_dir),
            ]:
                tier_dir.mkdir(parents=True, exist_ok=True)
                for entry in multi_res_results.get(tier_name, []):
                    src = Path(entry["path"])
                    dst = tier_dir / src.name
                    if src.exists() and not dst.exists():
                        shutil.copy2(src, dst)

            log.info(
                "Stage 0: 200MP processing complete - %d full, %d quarter, %d face crops",
                len(multi_res_results.get("full", [])),
                len(multi_res_results.get("quarter", [])),
                len(multi_res_results.get("face_crops", [])),
            )

    # Create processed/sensors/ directory structure
    sensors_proc_dir = session["proc_dir"] / "sensors"
    sensors_proc_dir.mkdir(parents=True, exist_ok=True)

    # Create processed/photos/ subdirs
    for subdir in ("wide_processed",):
        (session["proc_dir"] / "photos" / subdir).mkdir(parents=True, exist_ok=True)

    # Log summary
    n_wide = len(manifest.get("photos", {}).get("wide_23mm", []))
    n_main = len(manifest.get("photos", {}).get("main_200mp", []))
    n_other = sum(
        len(v) for k, v in manifest.get("photos", {}).items()
        if k not in ("wide_23mm", "main_200mp")
    )
    log.info(
        "Stage 0: Found %d video(s), %d wide photo(s), %d 200MP photo(s), "
        "%d other photo(s), %d sensor log(s)",
        manifest["summary"]["num_videos"],
        n_wide, n_main, n_other,
        manifest["summary"]["num_sensor_logs"],
    )

    # Compute capture quality score
    quality = _compute_capture_quality_score(manifest)
    log.info("Stage 0: Capture Quality — %s", quality["assessment"])
    for category, score in quality["breakdown"].items():
        if score > 0:
            log.info("  %s: %.1f pts", category, score)

    # Save quality score
    quality_path = session["proc_dir"] / "capture_quality.json"
    quality_path.write_text(
        json.dumps(quality, indent=2), encoding="utf-8",
    )

    mark_stage_complete(marker)
    return True


def stage_1_extract_frames(config: dict, session: dict) -> bool:
    """Extract frames from video with motion-based sampling."""
    marker = session["proc_dir"] / ".stage_1_complete"
    if stage_complete(marker):
        log.info("Stage 1: Frame extraction already complete, skipping")
        return True

    log.info("Stage 1: Extracting frames from video...")
    from capture import extract_frames
    import subprocess as _sp

    cap_cfg = config["capture"]
    frames_dir = session["proc_dir"] / "frames"

    # Pre-convert 8K/4K video to 1080p for fast extraction
    converted_videos = []
    for video_path in session["video_paths"]:
        try:
            probe_out = _sp.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "stream=width",
                 "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True, timeout=10,
            )
            # ffprobe CSV output may have trailing comma (e.g., "7680,")
            raw = probe_out.stdout.strip().split("\n")[0].strip().rstrip(",")
            width = int(raw) if raw else 0
        except Exception:
            width = 0

        if width > 1920:
            converted = session["proc_dir"] / "video_1080p.mp4"
            if not converted.exists():
                log.info("Stage 1: Pre-converting %dx video to 1080p (one-time, speeds up extraction 10x)...", width)
                try:
                    # Scale so longest dimension is 1920, preserve aspect ratio
                    scale_filter = "scale=1920:-2" if width >= 1920 else "scale=-2:1920"
                    # Try CUDA hwaccel first, fall back to CPU decode
                    cmd_base = [
                        "ffmpeg", "-y",
                        "-i", str(video_path),
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                        "-vf", scale_filter,
                        "-an", str(converted),
                    ]
                    try:
                        # Try with CUDA hardware decode (much faster for 8K HEVC)
                        cmd_cuda = ["ffmpeg", "-y", "-hwaccel", "cuda",
                                    "-i", str(video_path),
                                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                                    "-vf", scale_filter, "-an", str(converted)]
                        _sp.run(cmd_cuda, capture_output=True, timeout=300, check=True)
                    except Exception:
                        log.info("Stage 1: CUDA hwaccel unavailable, using CPU decode")
                        _sp.run(cmd_base, capture_output=True, timeout=600, check=True)
                    log.info("Stage 1: Converted to 1080p in %s", converted)
                except Exception as e:
                    log.warning("Stage 1: 1080p conversion failed (%s), using original", e)
                    converted = video_path
            else:
                log.info("Stage 1: Using existing 1080p conversion")
            converted_videos.append(converted)
        else:
            converted_videos.append(video_path)

    for video_path in converted_videos:
        if cap_cfg.get("motion_based", True):
            from capture.frame_extractor import extract_frames_motion_based
            extract_frames_motion_based(
                video_path=video_path,
                output_dir=frames_dir,
                min_flow_threshold=cap_cfg.get("min_flow_threshold", 2.0),
                max_frames=cap_cfg.get("max_frames", 80),
                scene_change_only=cap_cfg.get("scene_change_only", True),
            )
        else:
            extract_frames(
                video_path=video_path,
                output_dir=frames_dir,
                target_fps=cap_cfg.get("target_fps", 2),
                max_frames=cap_cfg.get("max_frames", 80),
            )

        # Detect LOG profile for this video and store in session for stage 2
        from capture.frame_extractor import detect_log_profile
        log_profile = detect_log_profile(video_path)
        session["detected_log_profile"] = log_profile
        if log_profile:
            log.info("Stage 1: Detected LOG profile '%s' for %s", log_profile, video_path.name)
        else:
            log.info("Stage 1: No LOG profile detected — color correction will be skipped")

    # --- Process Expert RAW photos if provided ---
    photo_paths = session.get("photo_paths", [])
    photos_cfg = config.get("photos", {})
    session_manifest = session.get("manifest")

    if photo_paths and photos_cfg.get("enabled", True):
        log.info("Stage 1: Processing %d Expert RAW photos...", len(photo_paths))
        from capture.photo_processor import process_photos

        # If we have a manifest from Stage 0, organize photos by lens
        if session_manifest and "photos" in session_manifest:
            # Process wide-lens photos (same lens as video — direct texture anchors)
            wide_photos = session_manifest["photos"].get("wide_23mm", [])
            if wide_photos:
                wide_paths = []
                for p in wide_photos:
                    dng = p.get("dng")
                    jpg = p.get("jpg")
                    if dng:
                        wide_paths.append(dng)
                    elif jpg:
                        wide_paths.append(jpg)

                if wide_paths:
                    wide_dir = session["proc_dir"] / "photos_wide"
                    processed_wide, wide_meta = process_photos(
                        photo_paths=wide_paths,
                        output_dir=wide_dir,
                    )
                    log.info("Stage 1: Processed %d wide-lens photos", len(processed_wide))

            # Process 200MP quarter-res versions (already generated in Stage 0)
            multi_res = session.get("multi_res_200mp", {})
            quarter_photos = multi_res.get("quarter", [])
            if quarter_photos:
                log.info("Stage 1: %d 200MP quarter-res photos available from Stage 0", len(quarter_photos))

            # Collect all processed photos for downstream
            all_photo_paths = []
            all_photo_meta = []

            # Wide-lens photos
            if wide_photos:
                wide_dir = session["proc_dir"] / "photos_wide"
                if wide_dir.exists():
                    for pp in sorted(wide_dir.glob("photo_*.png")):
                        all_photo_paths.append(pp)

            # 200MP quarter-res photos
            quarter_dir = session["proc_dir"] / "photos_200mp" / "quarter"
            if quarter_dir.exists():
                for pp in sorted(quarter_dir.glob("*.png")):
                    all_photo_paths.append(pp)

            session["processed_photo_paths"] = all_photo_paths
        else:
            # No manifest — process all photos uniformly (legacy path)
            photos_dir = session["proc_dir"] / "photos"
            processed_photos, photo_metadata = process_photos(
                photo_paths=photo_paths,
                output_dir=photos_dir,
            )
            session["processed_photo_paths"] = processed_photos
            session["photo_metadata"] = photo_metadata
            all_photo_paths = processed_photos
            log.info("Stage 1: Processed %d photos -> %s", len(processed_photos), photos_dir)

        # Copy processed photos into frames_srgb so they are included
        # in downstream stages (DA3, COLMAP, training). Photos are named
        # photo_NNNN.png to distinguish from video frame_NNNNNN.png files.
        srgb_dir = session["proc_dir"] / "frames_srgb"
        srgb_dir.mkdir(parents=True, exist_ok=True)
        for pp in all_photo_paths:
            dst = srgb_dir / pp.name
            if not dst.exists():
                shutil.copy2(pp, dst)
                log.debug("Copied photo %s to frames_srgb/", pp.name)

        # Save a photo manifest so downstream stages know which images are photos
        # and their source lens for weighted training
        photo_manifest_data = {
            "photo_names": [p.name for p in all_photo_paths],
            "weight": photos_cfg.get("weight", 3.0),
            "count": len(all_photo_paths),
        }

        # If we have multi-source data, add per-lens weight info
        if session_manifest and "photos" in session_manifest:
            wide_names = []
            main_200mp_names = []
            face_crop_names = []

            wide_dir = session["proc_dir"] / "photos_wide"
            if wide_dir.exists():
                wide_names = [f.name for f in sorted(wide_dir.glob("photo_*.png"))]

            quarter_dir = session["proc_dir"] / "photos_200mp" / "quarter"
            if quarter_dir.exists():
                main_200mp_names = [f.name for f in sorted(quarter_dir.glob("*.png"))]

            face_crops_dir = session["proc_dir"] / "photos_200mp" / "face_crops"
            if face_crops_dir.exists():
                face_crop_names = [f.name for f in sorted(face_crops_dir.glob("*.tiff"))]

            photo_manifest_data["lens_groups"] = {
                "wide_23mm": {
                    "names": wide_names,
                    "weight": 3.0,
                },
                "main_200mp": {
                    "names": main_200mp_names,
                    "weight": 5.0,
                },
            }
            photo_manifest_data["face_crops"] = {
                "names": face_crop_names,
                "dir": str(face_crops_dir) if face_crops_dir.exists() else None,
                "weight": 5.0,
            }

        manifest_path = session["proc_dir"] / "photo_manifest.json"
        manifest_path.write_text(
            json.dumps(photo_manifest_data, indent=2), encoding="utf-8",
        )
        log.info("Photo manifest saved: %s", manifest_path)
    else:
        session["processed_photo_paths"] = []
        session["photo_metadata"] = []

    mark_stage_complete(marker)
    return True


def stage_2_color_correct(config: dict, session: dict) -> bool:
    """Apply LOG to sRGB color correction for COLMAP.

    Automatically skips colour correction when the video was not recorded
    with a LOG profile (detected in stage 1).  Also respects the manual
    ``color_correction.enabled`` config flag.
    """
    marker = session["proc_dir"] / ".stage_2_complete"
    if stage_complete(marker):
        log.info("Stage 2: Color correction already complete, skipping")
        return True

    cc_cfg = config["capture"]["color_correction"]

    # Determine whether the video is LOG-encoded
    is_log = session.get("detected_log_profile") is not None

    if not cc_cfg.get("enabled", True):
        log.info("Stage 2: Color correction disabled in config, copying frames as-is")
        src = session["proc_dir"] / "frames"
        dst = session["proc_dir"] / "frames_srgb"
        for f in src.glob("*.png"):
            shutil.copy2(f, dst / f.name)
    else:
        if is_log:
            log.info("Stage 2: Applying LOG -> sRGB color correction...")
        else:
            log.info("Stage 2: No LOG profile — frames will be copied without correction")

        from capture.color_correction import batch_color_correct
        batch_color_correct(
            frames_dir=session["proc_dir"] / "frames",
            output_srgb_dir=session["proc_dir"] / "frames_srgb",
            is_log=is_log,
        )

    mark_stage_complete(marker)
    return True


def stage_3_filter_frames(config: dict, session: dict) -> bool:
    """Filter out blurry, overexposed, or faceless frames.

    Uses *quick_mode* by default (blur + exposure only at half resolution,
    no face detection) for speed.  Full face-aware filtering is deferred
    to the pose-aware frame selection step after DA3/COLMAP.
    """
    marker = session["proc_dir"] / ".stage_3_complete"
    if stage_complete(marker):
        log.info("Stage 3: Frame filtering already complete, skipping")
        return True

    log.info("Stage 3: Filtering frames for quality...")
    from capture.frame_filter import filter_frames

    filt_cfg = config["capture"]["filtering"]
    blur_threshold = filt_cfg.get("blur_threshold", 100.0)
    quick_mode = filt_cfg.get("quick_mode", True)

    filter_frames(
        frames_dir=session["proc_dir"] / "frames_srgb",
        output_json=session["proc_dir"] / "selected_frames.json",
        blur_threshold=blur_threshold,
        exposure_low=filt_cfg.get("exposure_low", 30),
        exposure_high=filt_cfg.get("exposure_high", 225),
        require_face=filt_cfg.get("require_face", True),
        quick_mode=quick_mode,
    )

    mark_stage_complete(marker)
    return True


def stage_4_parse_sensors(config: dict, session: dict) -> bool:
    """Parse IMU data from video metadata, sidecar files, or Sensor Logger ZIP."""
    marker = session["proc_dir"] / ".stage_4_complete"
    if stage_complete(marker):
        log.info("Stage 4: Sensor parsing already complete, skipping")
        return True

    sensor_cfg = config["sensors"]
    if not sensor_cfg["imu"].get("enabled", True):
        log.info("Stage 4: IMU disabled, skipping")
        mark_stage_complete(marker)
        return True

    # -- Check for Sensor Logger ZIP (much richer data with pre-fused orientation) --
    sensor_log_path = session.get("sensor_log_path")

    # Auto-detect: look for .zip files in content/New/ if not explicitly provided
    if sensor_log_path is None:
        content_dir = PROJECT_ROOT / "content" / "New"
        if content_dir.exists():
            zips = sorted(content_dir.glob("*.zip"))
            if zips:
                sensor_log_path = zips[0]
                log.info("Stage 4: Auto-detected Sensor Logger ZIP: %s", sensor_log_path)

    if sensor_log_path is not None and Path(sensor_log_path).exists():
        log.info("Stage 4: Parsing Sensor Logger data from %s", sensor_log_path)
        from sensors.sensor_logger import (
            parse_sensor_logger_zip,
            align_sensor_to_video,
            estimate_camera_trajectory,
            compute_lighting_profile,
        )

        try:
            import numpy as np

            sensors_dir = session["proc_dir"] / "sensors"
            sensors_dir.mkdir(parents=True, exist_ok=True)

            npz_path, summary = parse_sensor_logger_zip(
                zip_path=sensor_log_path,
                output_dir=sensors_dir,
            )
            log.info(
                "Stage 4: Parsed %d sensors, %d total samples",
                len(summary["sensors"]), summary["total_samples"],
            )

            # Align sensor data to video frames
            for video_path in session["video_paths"]:
                sensor_data = align_sensor_to_video(
                    sensor_npz_path=npz_path,
                    video_path=video_path,
                )
                session["sensor_logger_data"] = sensor_data

                # Save aligned sensor data to disk for downstream stages
                aligned_npz_path = sensors_dir / "aligned_sensor_data.npz"
                aligned_save = {}
                for key, val in sensor_data.items():
                    if isinstance(val, np.ndarray):
                        aligned_save[key] = val
                    elif isinstance(val, (int, float)):
                        aligned_save[key] = np.array(val)
                np.savez_compressed(str(aligned_npz_path), **aligned_save)
                log.info("Stage 4: Saved aligned sensor data to %s", aligned_npz_path)

                # Analyse camera trajectory
                if "frame_quaternions" in sensor_data:
                    trajectory = estimate_camera_trajectory(
                        orientation_quats=sensor_data["frame_quaternions"],
                        timestamps=sensor_data["frame_timestamps"],
                    )
                    session["trajectory_assessment"] = trajectory
                    log.info("Stage 4: %s", trajectory["coverage_assessment"])

                    # Save trajectory analysis
                    traj_path = sensors_dir / "trajectory_analysis.json"
                    traj_path.write_text(
                        json.dumps(trajectory, indent=2, default=str),
                        encoding="utf-8",
                    )

                # Compute and save lighting profile
                if "frame_light" in sensor_data:
                    lighting = compute_lighting_profile(
                        light_lux=sensor_data["frame_light"],
                        timestamps=sensor_data["frame_timestamps"],
                        frame_timestamps=sensor_data["frame_timestamps"],
                    )
                    n_outliers = int(lighting["outlier_mask"].sum())
                    log.info(
                        "Stage 4: Lighting — median=%.0f lux, range=%.0f-%.0f, %d outlier frames",
                        lighting["median_lux"],
                        lighting["lux_range"][0], lighting["lux_range"][1],
                        n_outliers,
                    )

                    # Save lighting profile to disk for Stage 13
                    lighting_npz_path = sensors_dir / "lighting_profile.npz"
                    np.savez_compressed(
                        str(lighting_npz_path),
                        frame_lux=lighting["frame_lux"],
                        frame_adjustment_factors=lighting["frame_adjustment_factors"],
                        outlier_mask=lighting["outlier_mask"],
                        median_lux=np.array(lighting["median_lux"]),
                        lux_min=np.array(lighting["lux_range"][0]),
                        lux_max=np.array(lighting["lux_range"][1]),
                    )
                    log.info("Stage 4: Saved lighting profile to %s", lighting_npz_path)

            # Mark that we have Sensor Logger data (skip Madgwick in stage 5)
            (session["proc_dir"] / ".sensor_logger_available").write_text(
                str(npz_path)
            )

        except Exception as e:
            log.warning("Sensor Logger parsing failed (non-fatal): %s", e)
            log.info("Falling back to video-embedded IMU parsing")
            sensor_log_path = None  # fall through to legacy path

    # -- Legacy: parse IMU from video metadata --
    if sensor_log_path is None:
        log.info("Stage 4: Parsing IMU/sensor data from video metadata...")
        from sensors import parse_imu_from_video

        for video_path in session["video_paths"]:
            try:
                parse_imu_from_video(
                    video_path=video_path,
                    output_path=session["proc_dir"] / "imu_data.npz",
                )
            except Exception as e:
                log.warning("IMU parsing failed (non-fatal): %s", e)
                log.info("Continuing without IMU data")

    mark_stage_complete(marker)
    return True


# ---------------------------------------------------------------------------
# Stage 5 sensor-derived prior enhancements
# ---------------------------------------------------------------------------

def _enhance_priors_with_barometer(
    priors_path: Path,
    sensor_data: dict,
    frame_names: list[str],
) -> None:
    """Add Y-translation constraints from barometer pressure data.

    Barometer pressure deltas convert to altitude changes
    (1 hPa ~= 8.43m at sea level).  This is the only translation
    prior available from sensor data — even gyroscopes cannot
    provide positional information.

    Rewrites the priors file in-place with tighter TY_STD where
    barometer data is available.
    """
    import numpy as np

    # Load barometer data from the aligned NPZ
    aligned_npz = priors_path.parent.parent / "sensors" / "aligned_sensor_data.npz"
    if not aligned_npz.exists():
        return

    try:
        data = np.load(str(aligned_npz), allow_pickle=True)
    except Exception:
        return

    # Check for barometer in the raw sensor data
    raw_npz_path = priors_path.parent.parent / "sensors" / "sensor_logger_data.npz"
    if not raw_npz_path.exists():
        return

    try:
        raw = np.load(str(raw_npz_path), allow_pickle=True)
    except Exception:
        return

    if "baro_timestamps" not in raw or "baro_pressure" not in raw:
        return

    baro_t = raw["baro_timestamps"]
    baro_p = raw["baro_pressure"]

    if len(baro_t) < 2 or len(baro_p) < 2:
        return

    # Interpolate barometer to frame timestamps
    frame_t = data.get("frame_timestamps")
    if frame_t is None or len(frame_t) == 0:
        return

    t_clamped = np.clip(frame_t, baro_t[0], baro_t[-1])
    frame_pressure = np.interp(t_clamped, baro_t, baro_p)

    # Convert pressure delta to height (meters)
    # Pressure decreases with altitude: 1 hPa drop ~= 8.43m rise
    reference_pressure = frame_pressure[0]
    height_m = (reference_pressure - frame_pressure) * 8.43

    # Check if there's meaningful height variation (> 2cm)
    height_range = float(np.ptp(height_m))
    if height_range < 0.02:
        log.info("Stage 5: Barometer — negligible height variation (%.1f mm), skipping Y-translation priors",
                 height_range * 1000)
        return

    log.info(
        "Stage 5: Barometer — %.1f cm height variation detected, adding Y-translation priors",
        height_range * 100,
    )

    # Re-read existing priors and add TY values with moderate uncertainty
    # TY_STD: ~5cm uncertainty (barometer is noisy for small changes)
    ty_std = max(0.05, height_range * 0.3)

    lines = priors_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    frame_idx = 0

    for line in lines:
        if line.startswith("#") or not line.strip():
            new_lines.append(line)
            continue

        parts = line.split()
        if len(parts) >= 14 and frame_idx < len(height_m):
            # Replace TY (index 6) and TY_STD (index 12) with barometer values
            parts[6] = f"{height_m[frame_idx]:.6f}"
            parts[12] = f"{ty_std:.4f}"
            new_lines.append(" ".join(parts))
            frame_idx += 1
        else:
            new_lines.append(line)

    priors_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    log.info("Stage 5: Updated %d priors with barometer-derived Y-translation (TY_STD=%.3f m)",
             frame_idx, ty_std)


def _compute_gravity_alignment(sensor_data: dict, session: dict) -> None:
    """Compute gravity-derived "up" direction from accelerometer data.

    When the camera is approximately stationary between movements, the
    accelerometer reading is dominated by gravity.  The mean acceleration
    vector gives us the gravity direction, which we save for aligning the
    reconstruction so that "up" in the model matches real-world "up".
    """
    import numpy as np

    if "frame_accel" not in sensor_data:
        return

    accel = sensor_data["frame_accel"]  # Nx3
    if len(accel) < 2:
        return

    # Average gravity direction (negate because accelerometer measures
    # reaction force: phone resting on table reads +9.8 on Z)
    gravity = accel.mean(axis=0)
    gravity_norm = np.linalg.norm(gravity)
    if gravity_norm < 1.0:
        log.warning("Stage 5: Gravity vector magnitude too low (%.2f), skipping alignment", gravity_norm)
        return

    up_direction = -gravity / gravity_norm

    # Save gravity alignment data
    sensors_dir = session["proc_dir"] / "sensors"
    sensors_dir.mkdir(parents=True, exist_ok=True)
    gravity_path = sensors_dir / "gravity_alignment.npz"
    np.savez(
        str(gravity_path),
        up_direction=up_direction,
        gravity_vector=gravity,
        gravity_magnitude=np.array(gravity_norm),
    )

    log.info(
        "Stage 5: Gravity alignment — up=[%.3f, %.3f, %.3f], |g|=%.2f m/s^2",
        up_direction[0], up_direction[1], up_direction[2], gravity_norm,
    )


def stage_5_compute_priors(config: dict, session: dict) -> bool:
    """Compute rotation priors from IMU for COLMAP.

    When Sensor Logger data is available (pre-fused quaternions), this stage
    uses those directly — they are far more accurate than Madgwick-filtered
    IMU because Samsung already fused accel+gyro+mag with their Kalman filter.
    """
    marker = session["proc_dir"] / ".stage_5_complete"
    if stage_complete(marker):
        log.info("Stage 5: Prior computation already complete, skipping")
        return True

    # -- Check for Sensor Logger pre-fused orientations --
    sensor_logger_marker = session["proc_dir"] / ".sensor_logger_available"
    if sensor_logger_marker.exists():
        log.info("Stage 5: Using Sensor Logger pre-fused quaternions (skipping Madgwick)")

        from sensors.sensor_logger import (
            align_sensor_to_video,
            generate_rotation_priors_from_sensor_logger,
        )

        npz_path = Path(sensor_logger_marker.read_text().strip())
        frame_names = sorted(
            [f.name for f in (session["proc_dir"] / "frames_srgb").glob("*.png")]
            + [f.name for f in (session["proc_dir"] / "frames_srgb").glob("*.jpg")]
        )

        if not frame_names:
            log.warning("Stage 5: No frames found in frames_srgb, skipping priors")
            mark_stage_complete(marker)
            return True

        try:
            # Re-align if not already in session (e.g. resuming from stage 5)
            sensor_data = session.get("sensor_logger_data")
            if sensor_data is None:
                for video_path in session["video_paths"]:
                    sensor_data = align_sensor_to_video(
                        sensor_npz_path=npz_path,
                        video_path=video_path,
                    )
                    break

            if sensor_data is not None and "frame_quaternions" in sensor_data:
                priors_file = session["proc_dir"] / "colmap" / "image_priors.txt"
                generate_rotation_priors_from_sensor_logger(
                    sensor_data=sensor_data,
                    frame_names=frame_names,
                    output_path=priors_file,
                )
                # Also save a copy in sensors/ for reference
                sensors_priors = session["proc_dir"] / "sensors" / "rotation_priors.txt"
                sensors_priors.parent.mkdir(parents=True, exist_ok=True)
                if not sensors_priors.exists():
                    shutil.copy2(priors_file, sensors_priors)

                # Log trajectory assessment
                from sensors.sensor_logger import estimate_camera_trajectory
                trajectory = estimate_camera_trajectory(
                    orientation_quats=sensor_data["frame_quaternions"],
                    timestamps=sensor_data["frame_timestamps"],
                )
                log.info("Stage 5: Trajectory — %s", trajectory["coverage_assessment"])
                log.info("Stage 5: Wrote COLMAP priors from Sensor Logger (pre-fused, tighter uncertainty)")

                # --- Barometer → height priors (Y-translation constraint) ---
                # Pressure deltas convert to altitude changes: 1 hPa ~= 8.43m
                # This gives us the ONLY translation prior from sensor data.
                _enhance_priors_with_barometer(priors_file, sensor_data, frame_names)

                # --- Gravity alignment from accelerometer ---
                # Save the gravity-derived "up" direction so post-reconstruction
                # alignment (stage 6 or stage 14) can orient the face correctly.
                _compute_gravity_alignment(sensor_data, session)
            else:
                log.warning("Stage 5: Sensor Logger data missing quaternions, falling back to Madgwick")
                sensor_logger_marker.unlink(missing_ok=True)
                # Fall through to legacy path below

        except Exception as e:
            log.warning("Stage 5: Sensor Logger priors failed: %s — falling back to Madgwick", e)
            sensor_logger_marker.unlink(missing_ok=True)

        # If we successfully wrote priors, mark complete and return
        priors_path = session["proc_dir"] / "colmap" / "image_priors.txt"
        if priors_path.exists():
            mark_stage_complete(marker)
            return True

    # -- Legacy: Madgwick filter from video-embedded IMU --
    sensor_cfg = config["sensors"]
    imu_path = session["proc_dir"] / "imu_data.npz"
    if not imu_path.exists():
        log.info("Stage 5: No IMU data available, skipping")
        mark_stage_complete(marker)
        return True

    log.info("Stage 5: Computing rotation priors from IMU...")
    try:
        import numpy as np
        from sensors import compute_rotations, generate_colmap_priors

        imu_data = np.load(str(imu_path))
        rotations = compute_rotations(
            accel=imu_data["accel"],
            gyro=imu_data["gyro"],
            timestamps=imu_data["timestamps"],
        )

        if sensor_cfg.get("generate_colmap_priors", True):
            generate_colmap_priors(
                rotations=rotations,
                frame_names=sorted(
                    [f.name for f in (session["proc_dir"] / "frames_srgb").glob("*.png")]
                ),
                output_path=session["proc_dir"] / "colmap" / "image_priors.txt",
            )
    except Exception as e:
        log.warning("IMU prior computation failed (non-fatal): %s", e)
        log.info("Continuing without rotation priors")

    mark_stage_complete(marker)
    return True


def stage_6_colmap(config: dict, session: dict) -> bool:
    """Run COLMAP SfM or unified DA3 if configured."""
    marker = session["proc_dir"] / ".stage_6_complete"
    if stage_complete(marker):
        log.info("Stage 6: Reconstruction already complete, skipping")
        return True

    # Check for photo manifest (photos are already in frames_srgb from stage 1)
    photo_manifest_path = session["proc_dir"] / "photo_manifest.json"
    has_photos = photo_manifest_path.exists()
    if has_photos:
        photo_manifest = json.loads(photo_manifest_path.read_text(encoding="utf-8"))
        log.info("Stage 6: %d Expert RAW photos will be included in reconstruction",
                 photo_manifest.get("count", 0))

    # ---- Unified DA3 path ----
    if _use_da3_unified(config):
        log.info("Stage 6: Using unified DA3 (replaces COLMAP + depth + alignment)")
        if has_photos:
            log.info("Stage 6: Photos are in frames_srgb and will be processed by DA3 alongside video frames")
        try:
            success = _run_da3_unified_stage(config, session)
            if success:
                mark_stage_complete(marker)
                mark_stage_complete(session["proc_dir"] / ".stage_7_complete")
                mark_stage_complete(session["proc_dir"] / ".stage_8_complete")
                # Run frame selection post-step (normally in stage 8)
                _select_training_frames(config, session)
                return True
            else:
                log.warning("DA3 unified stage failed, falling back to COLMAP pipeline")
        except Exception as e:
            log.warning("DA3 unified stage raised an error: %s", e)
            log.info("Falling back to COLMAP pipeline")

    # ---- Legacy COLMAP path ----
    log.info("Stage 6: Running COLMAP SfM...")
    from reconstruction import run_colmap

    colmap_cfg = config["reconstruction"]["colmap"]
    priors_path = session["proc_dir"] / "colmap" / "image_priors.txt"

    run_colmap(
        images_dir=session["proc_dir"] / "frames_srgb",
        workspace_dir=session["proc_dir"] / "colmap",
        colmap_binary=colmap_cfg["binary"],
        camera_model=colmap_cfg.get("camera_model", "OPENCV"),
        single_camera=colmap_cfg.get("single_camera", True),
        max_image_size=colmap_cfg.get("max_image_size", 3840),
        max_num_features=colmap_cfg.get("max_num_features", 8192),
        image_priors_path=priors_path if priors_path.exists() else None,
    )

    # Register photos into COLMAP reconstruction if available
    if has_photos:
        colmap_model_dir = session["proc_dir"] / "colmap" / "sparse" / "0"

        # Check if we have multi-lens data (from Stage 0)
        has_multi_lens = (
            photo_manifest.get("lens_groups") is not None
            and colmap_model_dir.exists()
        )

        if has_multi_lens:
            log.info("Stage 6: Multi-lens photo registration (wide + 200MP)...")
            from capture.multi_res_processor import register_multi_lens_photos

            wide_dir = session["proc_dir"] / "photos_wide"
            quarter_dir = session["proc_dir"] / "photos_200mp" / "quarter"

            # Load camera models if available
            cam_models_path = session["proc_dir"] / "camera_models.json"
            camera_models = None
            if cam_models_path.exists():
                camera_models = json.loads(cam_models_path.read_text(encoding="utf-8"))

            try:
                updated_dir = register_multi_lens_photos(
                    wide_photos_dir=wide_dir,
                    main_photos_dir=quarter_dir,
                    colmap_model_dir=colmap_model_dir,
                    colmap_binary=colmap_cfg["binary"],
                    camera_models=camera_models,
                )
                log.info("Stage 6: Multi-lens photos registered, model at %s", updated_dir)
            except Exception as e:
                log.warning("Stage 6: Multi-lens registration failed (continuing without): %s", e)
        else:
            # Legacy single-lens path
            photos_dir = session["proc_dir"] / "photos"
            if photos_dir.exists() and colmap_model_dir.exists():
                log.info("Stage 6: Registering Expert RAW photos into COLMAP reconstruction...")
                from capture.photo_processor import register_photos_into_reconstruction
                try:
                    updated_dir = register_photos_into_reconstruction(
                        photo_dir=photos_dir,
                        colmap_model_dir=colmap_model_dir,
                        colmap_binary=colmap_cfg["binary"],
                    )
                    log.info("Stage 6: Photos registered, updated model at %s", updated_dir)
                except Exception as e:
                    log.warning("Stage 6: Photo registration failed (continuing without): %s", e)

    mark_stage_complete(marker)
    return True


def stage_7_depth(config: dict, session: dict) -> bool:
    """Run monocular depth estimation.

    Skipped automatically when the unified DA3 stage already ran (stage 6).
    """
    marker = session["proc_dir"] / ".stage_7_complete"
    if stage_complete(marker):
        log.info("Stage 7: Depth estimation already complete, skipping")
        return True

    if (session["proc_dir"] / ".da3_unified_complete").exists():
        log.info("Stage 7: Skipped -- DA3 unified stage already produced depth maps")
        mark_stage_complete(marker)
        return True

    log.info("Stage 7: Running depth estimation...")
    from depth import DepthEstimator

    depth_cfg = config["depth"]
    frames_dir = session["proc_dir"] / "frames_srgb"
    depth_dir = session["proc_dir"] / "depth"

    frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        log.error("No frames found in %s", frames_dir)
        return False

    estimator = DepthEstimator(
        model_name=depth_cfg.get("model", "auto"),
        model_size=depth_cfg.get("model_size", "vitl"),
    )

    log.info("Depth backend: %s (provides poses: %s)",
             estimator.backend.upper(), estimator.provides_poses)

    estimator.estimate_batch(
        image_paths=frame_paths,
        output_dir=depth_dir,
        batch_size=depth_cfg.get("batch_size", 4),
    )

    if estimator.provides_poses:
        pose_files = list(depth_dir.glob("*_pose.npz"))
        if pose_files:
            log.info("DA3 produced %d camera pose files", len(pose_files))
            (session["proc_dir"] / ".da3_poses_available").write_text(str(len(pose_files)))

    # Save depth map preview visualizations
    _save_depth_previews(session)

    mark_stage_complete(marker)
    return True


def stage_8_align_depth(config: dict, session: dict) -> bool:
    """Align monocular depth to COLMAP metric scale.

    Skipped when the unified DA3 stage already ran (stage 6).
    """
    marker = session["proc_dir"] / ".stage_8_complete"
    if stage_complete(marker):
        log.info("Stage 8: Depth alignment already complete, skipping")
        return True

    if (session["proc_dir"] / ".da3_unified_complete").exists():
        log.info("Stage 8: Skipped -- DA3 unified stage already produced aligned depth")
        mark_stage_complete(marker)
        return True

    depth_cfg = config["depth"]
    if not depth_cfg.get("align_to_colmap", True):
        log.info("Stage 8: Depth alignment disabled, skipping")
        mark_stage_complete(marker)
        return True

    log.info("Stage 8: Aligning depth maps to COLMAP scale...")
    from depth import batch_align

    batch_align(
        depth_dir=session["proc_dir"] / "depth",
        colmap_model_dir=session["proc_dir"] / "colmap" / "sparse" / "0",
        output_dir=session["proc_dir"] / "depth_aligned",
    )

    # --- Post-step: select best training frames via farthest-point sampling ---
    _select_training_frames(config, session)

    mark_stage_complete(marker)
    return True


def _select_training_frames(config: dict, session: dict) -> None:
    """Select a spatially diverse subset of frames for downstream stages.

    Called as a post-step of stage 8 (after depth alignment or DA3 unified).
    Writes ``selected_training_frames.json`` into the session proc dir.
    """
    from capture.frame_filter import select_best_frames

    selection_cfg = config.get("capture", {}).get("frame_selection", {})
    target_count = selection_cfg.get("target_count", 50)
    min_baseline = selection_cfg.get("min_baseline", 0.05)

    colmap_model_dir = session["proc_dir"] / "colmap" / "sparse" / "0"
    if not colmap_model_dir.exists():
        colmap_model_dir = None

    depth_dir = session["proc_dir"] / "depth"
    if not depth_dir.exists():
        depth_dir = None

    output_json = session["proc_dir"] / "selected_training_frames.json"

    log.info("Stage 8 post-step: Selecting best training frames (target=%d)...", target_count)
    selected = select_best_frames(
        frames_dir=session["proc_dir"] / "frames_srgb",
        poses_dir=session["proc_dir"] / "depth",
        output_json=output_json,
        target_count=target_count,
        min_baseline=min_baseline,
        colmap_model_dir=colmap_model_dir,
        depth_dir=depth_dir,
    )
    log.info("Selected %d training frames -> %s", len(selected), output_json)


def stage_9_landmarks(config: dict, session: dict) -> bool:
    """Detect facial landmarks on all frames using InsightFace (with MediaPipe fallback)."""
    marker = session["proc_dir"] / ".stage_9_complete"
    if stage_complete(marker):
        log.info("Stage 9: Landmark detection already complete, skipping")
        return True

    log.info("Stage 9: Detecting facial landmarks...")

    # Collect all frame + photo paths
    frames_dir = session["proc_dir"] / "frames_srgb"
    if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
        frames_dir = session["proc_dir"] / "frames"
    image_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))

    landmarks_dir = session["proc_dir"] / "landmarks"
    preview_dir = session["out_dir"] / "previews"

    # Try InsightFace first (better accuracy, especially on side profiles)
    try:
        from reconstruction.insightface_detector import detect_batch, generate_previews

        log.info("Stage 9: Using InsightFace (buffalo_l, 106+68 landmarks)")
        summary = detect_batch(
            image_paths=image_paths,
            output_dir=landmarks_dir,
            max_image_size=1920,
        )
        log.info(
            "Stage 9: InsightFace detected %d/%d faces (mean score %.3f)",
            summary["detected"], summary["total"], summary["mean_score"],
        )

        # Generate more preview images (every 8th frame)
        generate_previews(image_paths, landmarks_dir, preview_dir, num_previews=10)

    except ImportError:
        log.warning("Stage 9: InsightFace not available, falling back to MediaPipe")
        from reconstruction import FaceLandmarkDetector
        detector = FaceLandmarkDetector()
        detector.detect_batch(
            frames_dir=frames_dir,
            output_dir=landmarks_dir,
        )
        _save_landmark_previews(session)

    mark_stage_complete(marker)
    return True


def stage_10_flame(config: dict, session: dict) -> bool:
    """Fit FLAME parametric face model.

    Checks for matching landmarks + camera params before attempting the fit.
    Falls back gracefully with clear diagnostics when data is insufficient.
    """
    marker = session["proc_dir"] / ".stage_10_complete"
    if stage_complete(marker):
        log.info("Stage 10: FLAME fitting already complete, skipping")
        return True

    log.info("Stage 10: Fitting FLAME face model...")
    import json as _json
    from reconstruction import fit_flame_to_sequence

    flame_cfg = config["reconstruction"]["flame"]
    landmarks_dir = session["proc_dir"] / "landmarks"
    colmap_model_dir = session["proc_dir"] / "colmap" / "sparse" / "0"

    # ── Pre-flight: count available data ──────────────────────────────
    landmark_files = sorted(landmarks_dir.glob("*.json")) if landmarks_dir.exists() else []
    num_landmarks_total = len(landmark_files)
    num_detected = 0
    for lf in landmark_files:
        try:
            with open(lf, "r", encoding="utf-8") as fh:
                ld = _json.load(fh)
            if ld.get("detected", False):
                num_detected += 1
        except Exception:
            pass

    cameras_bin = colmap_model_dir / "cameras.bin"
    images_bin = colmap_model_dir / "images.bin"
    has_colmap = cameras_bin.exists() and images_bin.exists()

    # Count COLMAP images
    num_colmap_images = 0
    if has_colmap:
        try:
            import struct
            with open(images_bin, "rb") as fh:
                num_colmap_images = struct.unpack("<Q", fh.read(8))[0]
        except Exception:
            pass

    log.info(
        "Stage 10 pre-flight: %d landmark files (%d with detections), "
        "COLMAP model %s (%d images)",
        num_landmarks_total, num_detected,
        "found" if has_colmap else "MISSING", num_colmap_images,
    )

    if num_detected < 3:
        log.warning(
            "FLAME fitting skipped: only %d frames with face detections "
            "(minimum 3 required). Run landmark detection (stage 9) on more frames.",
            num_detected,
        )
        mark_stage_complete(marker)
        return True

    if not has_colmap:
        log.warning(
            "FLAME fitting skipped: no COLMAP model found at %s. "
            "Camera poses are required for multi-view FLAME fitting.",
            colmap_model_dir,
        )
        mark_stage_complete(marker)
        return True

    if num_colmap_images < 3:
        log.warning(
            "FLAME fitting may struggle: only %d images in COLMAP model "
            "(ideally need 5+). Will attempt anyway.",
            num_colmap_images,
        )

    # ── Run FLAME fitting ──────────────────────────────────────────────
    try:
        result = fit_flame_to_sequence(
            frames_dir=session["proc_dir"] / "frames_srgb",
            landmarks_dir=landmarks_dir,
            colmap_model_dir=colmap_model_dir,
            flame_model_path=flame_cfg["model_path"],
            embedding_path=flame_cfg["embedding_path"],
            output_dir=session["proc_dir"] / "flame",
            fitting_config=flame_cfg.get("fitting"),
        )
        converged = result.get("converged", False)
        final_loss = result.get("loss", float("inf"))
        log.info(
            "FLAME fitting complete: loss=%.4f, converged=%s",
            final_loss, converged,
        )
    except RuntimeError as e:
        log.warning(
            "FLAME fitting failed: %s. "
            "This is non-fatal -- the pipeline will continue without a fitted "
            "FLAME model. Gaussian initialization will fall back to point cloud.",
            e,
        )
    except Exception as e:
        log.warning("FLAME fitting failed with unexpected error: %s", e)
        log.info("Continuing without FLAME model")

    mark_stage_complete(marker)
    return True


def stage_11_segment(config: dict, session: dict) -> bool:
    """Segment face regions in all frames."""
    marker = session["proc_dir"] / ".stage_11_complete"
    if stage_complete(marker):
        log.info("Stage 11: Face segmentation already complete, skipping")
        return True

    log.info("Stage 11: Segmenting face regions...")
    from reconstruction import FaceSegmenter

    segmenter = FaceSegmenter()
    segmenter.segment_batch(
        frames_dir=session["proc_dir"] / "frames_srgb",
        output_dir=session["proc_dir"] / "face_masks",
    )

    # Save mask overlay previews
    _save_mask_previews(session)

    mark_stage_complete(marker)
    return True


def stage_12_init_gaussians(config: dict, session: dict) -> bool:
    """Initialize Gaussians from FLAME binding, FLAME mesh, or point cloud.

    Initialization priority:
      1. flame_binding -- GaussianAvatars-style triangle-bound Gaussians (best)
      2. flame_mesh -- Uniform surface sampling on fitted FLAME mesh
      3. colmap_sparse -- Sparse COLMAP points (fallback)
    """
    marker = session["proc_dir"] / ".stage_12_complete"
    if stage_complete(marker):
        log.info("Stage 12: Gaussian initialization already complete, skipping")
        return True

    log.info("Stage 12: Initializing Gaussians...")
    import numpy as np
    import torch
    from splatting.initializer import (
        initialize_from_flame_binding,
        initialize_from_flame_mesh,
        initialize_from_pointcloud,
        initialize_from_colmap_sparse,
    )

    splat_cfg = config["splatting"]
    flame_cfg = config["reconstruction"]["flame"]
    init_path = session["proc_dir"] / "gaussians_init.pt"
    binding_meta_path = session["proc_dir"] / "gaussians_binding_meta.npz"
    gaussians = None
    metadata = None

    flame_model_path = Path(flame_cfg["model_path"])
    flame_params_path = session["proc_dir"] / "flame" / "flame_params.npz"

    # --- Try FLAME-bound initialization ---
    if flame_model_path.exists() and flame_params_path.exists():
        log.info("Attempting FLAME-bound Gaussian initialization...")
        try:
            gaussians, metadata = initialize_from_flame_binding(
                flame_model_path=flame_model_path,
                flame_params_path=flame_params_path,
                num_gaussians_per_triangle=splat_cfg.get("num_gaussians_per_triangle", 3),
            )
            log.info(
                "FLAME-bound init: %d bound + %d free Gaussians",
                metadata["num_bound"], metadata["num_free"],
            )
        except Exception as e:
            log.warning("FLAME-bound initialization failed: %s", e)
            log.info("Falling back to FLAME mesh surface sampling...")
            gaussians = None

    # --- Fallback: FLAME mesh surface sampling ---
    if gaussians is None:
        flame_mesh = session["proc_dir"] / "flame" / "fitted_mesh.ply"
        if flame_mesh.exists():
            log.info("Using FLAME mesh surface sampling for initialization")
            try:
                gaussians = initialize_from_flame_mesh(
                    mesh_path=flame_mesh,
                    num_samples=splat_cfg.get("num_surface_samples", 200000),
                )
            except Exception as e:
                log.warning("FLAME mesh initialization failed: %s", e)

    # --- Fallback: point cloud ---
    if gaussians is None:
        pc_path = session["proc_dir"] / "dense_points.ply"
        if pc_path.exists():
            log.info("Using dense point cloud for initialization")
            try:
                gaussians = initialize_from_pointcloud(
                    pointcloud_path=pc_path, output_path=pc_path,
                )
            except Exception as e:
                log.warning("Point cloud initialization failed: %s", e)

    # --- Last resort: COLMAP sparse ---
    if gaussians is None:
        log.info("Using COLMAP sparse point cloud for initialization (last resort)")
        gaussians = initialize_from_colmap_sparse(
            colmap_model_dir=session["proc_dir"] / "colmap" / "sparse" / "0",
        )

    # Downsample if exceeding max_num_gaussians to avoid OOM during training
    max_gs = splat_cfg.get("training", {}).get("max_num_gaussians", 500000)
    n_gs = len(gaussians.positions)
    if n_gs > max_gs:
        log.info("Downsampling %d Gaussians to %d (max_num_gaussians cap)", n_gs, max_gs)
        import torch as _torch
        indices = _torch.randperm(n_gs)[:max_gs]
        gaussians.positions = gaussians.positions[indices]
        gaussians.colors_sh = gaussians.colors_sh[indices]
        gaussians.scales = gaussians.scales[indices]
        gaussians.rotations = gaussians.rotations[indices]
        gaussians.opacities = gaussians.opacities[indices]

    save_dict = {
        "positions": gaussians.positions,
        "colors_sh": gaussians.colors_sh,
        "scales": gaussians.scales,
        "rotations": gaussians.rotations,
        "opacities": gaussians.opacities,
    }
    torch.save(save_dict, str(init_path))
    log.info("Initialized %d Gaussians, saved to %s", len(gaussians.positions), init_path)

    if metadata is not None:
        np.savez(
            str(binding_meta_path),
            triangle_indices=metadata["triangle_indices"],
            bary_coords=metadata["bary_coords"],
            faces=metadata["faces"],
            num_bound=metadata["num_bound"],
            num_free=metadata["num_free"],
        )
        log.info("Saved FLAME binding metadata to %s", binding_meta_path)

    mark_stage_complete(marker)
    return True


def stage_13_train_gaussians(config: dict, session: dict) -> bool:
    """Train 3D Gaussian Splatting with OOM recovery and NaN detection."""
    marker = session["out_dir"] / ".stage_13_complete"
    if stage_complete(marker):
        log.info("Stage 13: Gaussian training already complete, skipping")
        return True

    log.info("Stage 13: Training 3D Gaussian Splatting...")
    import torch
    from splatting.trainer import GaussianTrainer, TrainingConfig
    from splatting.camera_utils import load_cameras_from_colmap
    from splatting.initializer import GaussianModel

    splat_cfg = config["splatting"]
    train_params = splat_cfg.get("training", {})
    max_num_gaussians = train_params.get("max_num_gaussians", 500000)

    # Load cameras
    cameras = load_cameras_from_colmap(session["proc_dir"] / "colmap" / "sparse" / "0")

    # Build photo weight info for weighted view sampling
    # Supports per-lens weights: 200MP face crops = 5x, wide DNG = 3x, video = 1x
    photo_manifest_path = session["proc_dir"] / "photo_manifest.json"
    photo_names = set()
    photo_weight = 1.0
    per_image_weights = {}  # name -> weight (for multi-lens data)
    if photo_manifest_path.exists():
        manifest = json.loads(photo_manifest_path.read_text(encoding="utf-8"))
        photo_names = set(manifest.get("photo_names", []))
        photos_cfg = config.get("photos", {})
        photo_weight = photos_cfg.get("weight", manifest.get("weight", 3.0))

        # Check for per-lens weight groups (from multi-source Stage 0)
        lens_groups = manifest.get("lens_groups")
        if lens_groups:
            for lens_name, group_info in lens_groups.items():
                group_weight = group_info.get("weight", photo_weight)
                for name in group_info.get("names", []):
                    per_image_weights[name] = group_weight

            # Log per-lens summary
            for lens_name, group_info in lens_groups.items():
                n = len(group_info.get("names", []))
                w = group_info.get("weight", photo_weight)
                if n > 0:
                    log.info(
                        "Stage 13: %d %s photo views weighted %.1fx",
                        n, lens_name, w,
                    )

            # Use the max weight as the default photo_weight for
            # backward-compatible trainer API
            if per_image_weights:
                photo_weight = max(per_image_weights.values())
        elif photo_names:
            log.info("Stage 13: %d photo views will be weighted %.1fx during training",
                     len(photo_names), photo_weight)

    # Load per-frame lighting data from sensor logger (if available)
    # Frames with unusual lighting get lower loss weight during training
    lighting_weights = None
    lighting_npz_path = session["proc_dir"] / "sensors" / "lighting_profile.npz"
    if lighting_npz_path.exists():
        import numpy as np
        lighting_data = np.load(str(lighting_npz_path))
        adjustment_factors = lighting_data["frame_adjustment_factors"]
        outlier_mask = lighting_data["outlier_mask"]
        median_lux = float(lighting_data["median_lux"])

        # Build per-frame loss weights: outlier frames get reduced weight (0.3x),
        # normal frames get 1.0.  This prevents inconsistent lighting from
        # corrupting the appearance model.
        lighting_weights = np.where(outlier_mask, 0.3, 1.0).astype(np.float32)
        n_outliers = int(outlier_mask.sum())
        log.info(
            "Stage 13: Lighting-aware training — median %.0f lux, %d outlier frames "
            "will be down-weighted to 0.3x",
            median_lux, n_outliers,
        )

    # Load initialized Gaussians
    init_path = session["proc_dir"] / "gaussians_init.pt"
    init_data = torch.load(str(init_path), weights_only=False)

    # OOM retry loop: up to 3 retries, halving max_num_gaussians each time
    max_oom_retries = 3
    for oom_attempt in range(max_oom_retries + 1):
        gaussians = GaussianModel(
            positions=init_data["positions"].clone(),
            colors_sh=init_data["colors_sh"].clone(),
            scales=init_data["scales"].clone(),
            rotations=init_data["rotations"].clone(),
            opacities=init_data["opacities"].clone(),
        )
        log.info("Loaded %d Gaussians for training (max_num_gaussians=%d)",
                 len(gaussians.positions), max_num_gaussians)

        tc = TrainingConfig(
            iterations=train_params.get("iterations", 3000),
            max_num_gaussians=max_num_gaussians,
        )
        trainer = GaussianTrainer(config=tc)

        try:
            trainer.train(
                gaussians=gaussians,
                cameras=cameras,
                images_dir=session["proc_dir"] / "frames_srgb",
                depth_dir=session["proc_dir"] / "depth_aligned",
                masks_dir=session["proc_dir"] / "face_masks",
                output_dir=session["out_dir"],
                photo_names=photo_names if photo_names else None,
                photo_weight=photo_weight,
                per_image_weights=per_image_weights if per_image_weights else None,
                lighting_weights=lighting_weights,
            )
            # Training succeeded
            break
        except torch.cuda.OutOfMemoryError:
            if oom_attempt < max_oom_retries:
                max_num_gaussians //= 2
                log.warning(
                    "OOM at stage 13 (attempt %d/%d). "
                    "Reducing max_num_gaussians to %d and retrying...",
                    oom_attempt + 1, max_oom_retries, max_num_gaussians,
                )
                torch.cuda.empty_cache()
            else:
                log.error(
                    "OOM at stage 13 after %d retries (max_num_gaussians=%d). "
                    "Reduce input resolution or number of training images.",
                    max_oom_retries, max_num_gaussians,
                )
                raise

    mark_stage_complete(marker)
    return True


def stage_14_export(config: dict, session: dict) -> bool:
    """Export final model, renders, and metrics.

    Each sub-step (PLY, mesh, texture, compressed, renders, animation,
    report) runs inside its own try/except so one failure does not block
    the rest of the export pipeline.
    """
    marker = session["out_dir"] / ".stage_14_complete"
    if stage_complete(marker):
        log.info("Stage 14: Export already complete, skipping")
        return True

    log.info("Stage 14: Exporting results...")
    import torch
    from splatting.initializer import GaussianModel

    export_cfg = config["splatting"]["export"]
    out_dir = session["out_dir"]

    # ---- Load checkpoint ------------------------------------------------
    ckpt_path = out_dir / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = out_dir / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = out_dir / "best.pt"
    if not ckpt_path.exists():
        log.error("No checkpoint found in %s", out_dir)
        return False

    log.info("Loading checkpoint from %s", ckpt_path)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # The trainer saves splats as a ParameterDict with keys:
    #   means, scales, quats, opacities, sh0, shN
    # plus metadata keys: iteration, loss, psnr, optimizer_states
    sh_all = torch.cat([state["sh0"], state["shN"]], dim=1)  # (N, 16, 3)
    opacities = state["opacities"]
    if opacities.dim() == 1:
        opacities = opacities.unsqueeze(-1)  # GaussianModel expects (N, 1)

    gaussians = GaussianModel(
        positions=state["means"],
        colors_sh=sh_all,
        scales=state["scales"],
        rotations=state["quats"],
        opacities=opacities,
    )
    log.info(
        "Loaded %d Gaussians from checkpoint (iteration %s, PSNR %.2f)",
        gaussians.num_gaussians,
        state.get("iteration", "?"),
        state.get("psnr", 0.0),
    )

    # Track which sub-steps succeeded
    substep_ok = {}

    # ---- 1. Gaussian export (multi-format via gsplat native API) --------
    export_formats = export_cfg.get("formats", ["ply"])
    # Separate Gaussian formats from mesh-only formats (gltf)
    gaussian_formats = [f for f in export_formats if f in ("ply", "splat", "compressed")]
    try:
        from splatting.exporter import export_gsplat_native

        exported = export_gsplat_native(
            gaussians, out_dir / "gaussians", formats=gaussian_formats,
        )
        substep_ok["ply"] = "ply" in exported
        for fmt, path in exported.items():
            log.info("Exported %s: %s", fmt, path)
    except Exception as e:
        log.error("Gaussian export failed: %s", e, exc_info=True)
        # Fallback to legacy PLY writer
        try:
            from splatting.exporter import export_gaussians_ply

            export_gaussians_ply(gaussians, out_dir / "gaussians.ply")
            substep_ok["ply"] = True
        except Exception as e2:
            log.error("Legacy PLY export also failed: %s", e2, exc_info=True)
            substep_ok["ply"] = False

    # ---- 2. Mesh extraction: SuGaR first, TSDF fallback ----------------
    mesh_path = out_dir / "mesh"
    try:
        from splatting.exporter import extract_mesh_sugar

        extract_mesh_sugar(gaussians, mesh_path)
        log.info("SuGaR mesh extraction succeeded")
        substep_ok["mesh"] = True
    except Exception as e:
        log.warning("SuGaR mesh extraction failed (%s); falling back to TSDF", e)
        try:
            from splatting.exporter import export_mesh_from_gaussians

            export_mesh_from_gaussians(gaussians, out_dir / "mesh.obj")
            substep_ok["mesh"] = True
        except Exception as e2:
            log.error("TSDF mesh extraction also failed: %s", e2)
            substep_ok["mesh"] = False

    # ---- 3. Texture baking onto extracted mesh --------------------------
    #
    # Texture source priority (highest quality first):
    #   1. Full-res 200MP photos / face crops — pore-level detail
    #   2. Wide-lens DNG photos — secondary coverage
    #   3. Video frames — fill remaining gaps
    try:
        mesh_obj = out_dir / "mesh.obj"
        mesh_ply = out_dir / "mesh.ply"
        mesh_file = mesh_obj if mesh_obj.exists() else mesh_ply
        if mesh_file.exists():
            from splatting.texture_baker import bake_texture
            from splatting.camera_utils import load_cameras_from_colmap

            colmap_dir = session["proc_dir"] / "colmap" / "sparse" / "0"
            if not colmap_dir.exists():
                colmap_dir = session["proc_dir"] / "colmap" / "sparse"

            # Determine best texture source images
            face_crops_dir = session["proc_dir"] / "photos_200mp" / "face_crops"
            full_200mp_dir = session["proc_dir"] / "photos_200mp" / "full"
            wide_photos_dir = session["proc_dir"] / "photos_wide"
            frames_dir = session["proc_dir"] / "frames"

            if full_200mp_dir.exists() and any(full_200mp_dir.iterdir()):
                images_dir = full_200mp_dir
                log.info("Stage 14: Using full-res 200MP photos for texture baking (primary)")
            elif face_crops_dir.exists() and any(face_crops_dir.iterdir()):
                images_dir = face_crops_dir
                log.info("Stage 14: Using 200MP face crops for texture baking (primary)")
            elif wide_photos_dir.exists() and any(wide_photos_dir.iterdir()):
                images_dir = wide_photos_dir
                log.info("Stage 14: Using wide-lens photos for texture baking")
            else:
                images_dir = frames_dir
                log.info("Stage 14: Using video frames for texture baking")

            cameras = load_cameras_from_colmap(colmap_dir)

            # Check for FLAME texture UVs
            flame_tex = PROJECT_ROOT / "Models" / "Flame" / "FLAME_texture.npz"
            flame_texture_path = flame_tex if flame_tex.exists() else None

            tex_res = export_cfg.get("texture_resolution", 2048)

            # Increase texture resolution when 200MP data is available
            if full_200mp_dir.exists() and any(full_200mp_dir.iterdir()):
                tex_res = max(tex_res, 4096)
                log.info("Stage 14: Increased texture resolution to %d for 200MP data", tex_res)

            bake_texture(
                mesh_path=mesh_file,
                cameras=cameras,
                images_dir=images_dir,
                output_dir=out_dir,
                texture_resolution=tex_res,
                prefer_photos=True,
                flame_texture_path=flame_texture_path,
            )
            log.info("Texture baking complete")
            substep_ok["texture"] = True
        else:
            log.info("No mesh found; skipping texture baking")
            substep_ok["texture"] = None  # skipped, not failed
    except Exception as e:
        log.warning("Texture baking failed: %s", e, exc_info=True)
        substep_ok["texture"] = False

    # ---- 4. Compressed export (PngCompression) ---------------------------
    # Only run if "compressed" is in formats or always as a bonus step
    if "compressed" in export_formats:
        try:
            from splatting.exporter import export_compressed

            export_compressed(gaussians, out_dir / "gaussians_compressed")
            substep_ok["compressed"] = True
        except Exception as e:
            log.warning("Compressed export failed: %s", e)
            substep_ok["compressed"] = False
    else:
        try:
            from splatting.exporter import export_compressed

            export_compressed(gaussians, out_dir / "gaussians_compressed")
            substep_ok["compressed"] = True
        except Exception as e:
            log.warning("Compressed export failed: %s", e)
            substep_ok["compressed"] = False

    # ---- 4b. glTF 2.0 export (mesh + texture) --------------------------
    if "gltf" in export_formats:
        try:
            from splatting.exporter import export_to_gltf

            mesh_obj = out_dir / "mesh.obj"
            mesh_ply = out_dir / "mesh.ply"
            mesh_for_gltf = mesh_obj if mesh_obj.exists() else (
                mesh_ply if mesh_ply.exists() else None
            )
            texture_for_gltf = out_dir / "texture.png"
            if not texture_for_gltf.exists():
                texture_for_gltf = None

            gltf_path = export_to_gltf(
                mesh_path=mesh_for_gltf,
                gaussians=gaussians,
                output_path=out_dir / "model.glb",
                texture_path=texture_for_gltf,
            )
            substep_ok["gltf"] = gltf_path is not None
            if gltf_path:
                log.info("Exported glTF to %s", gltf_path)
        except Exception as e:
            log.warning("glTF export failed: %s", e)
            substep_ok["gltf"] = False

    # ---- 4c. Draco / quantized compression --------------------------------
    compression_cfg = export_cfg.get("compression", {})
    if compression_cfg.get("enabled", True):
        try:
            from splatting.exporter import compress_exported_files

            # Build exported dict from what we know succeeded
            exported_files = {}
            ply_path = out_dir / "gaussians.ply"
            splat_path = out_dir / "gaussians.splat"
            if ply_path.exists():
                exported_files["ply"] = ply_path
            if splat_path.exists():
                exported_files["splat"] = splat_path

            if exported_files:
                comp_results = compress_exported_files(
                    exported=exported_files,
                    gaussians=gaussians,
                    compression_config=compression_cfg,
                )
                substep_ok["draco_compression"] = bool(comp_results)
                for fmt, stats in comp_results.items():
                    log.info(
                        "Compressed %s: %.1fx reduction (%s)",
                        fmt,
                        stats.get("ratio", 1.0),
                        stats.get("backend_used", "unknown"),
                    )
            else:
                log.info("No exported files to compress")
                substep_ok["draco_compression"] = None
        except Exception as e:
            log.warning("Draco/quantized compression failed: %s", e)
            substep_ok["draco_compression"] = False

    # ---- 5. Turntable renders -------------------------------------------
    try:
        from splatting.exporter import render_novel_views

        render_novel_views(
            gaussians,
            output_dir=out_dir / "renders",
            num_views=export_cfg.get("turntable_views", 60),
        )
        substep_ok["renders"] = True
    except Exception as e:
        log.error("Turntable rendering failed: %s", e, exc_info=True)
        substep_ok["renders"] = False

    # ---- 6. Animation generation (optional, requires FLAME binding) -----
    try:
        binding_meta_path = session["proc_dir"] / "gaussians_binding_meta.npz"
        flame_params_path = session["proc_dir"] / "flame" / "flame_params.npz"
        flame_cfg = config.get("reconstruction", {}).get("flame", {})
        flame_model_path_str = flame_cfg.get("model_path")

        if flame_model_path_str is None:
            log.info("No FLAME model_path in config; skipping animation generation")
            substep_ok["animation"] = None
        else:
            flame_model_path = Path(flame_model_path_str)
            if (
                binding_meta_path.exists()
                and flame_params_path.exists()
                and flame_model_path.exists()
            ):
                log.info("FLAME binding metadata found — generating demo animations...")
                from splatting.animator import FaceAnimator

                animator = FaceAnimator(
                    flame_model_path=flame_model_path,
                    flame_params=flame_params_path,
                    binding_metadata=binding_meta_path,
                    gaussians=gaussians,
                    device=config.get("device", "cuda"),
                )

                expressions_yaml = PROJECT_ROOT / "config" / "expressions.yaml"
                animator.generate_predefined_animations(
                    output_dir=out_dir / "animations",
                    fps=30,
                    presets_path=expressions_yaml if expressions_yaml.exists() else None,
                )
                log.info("Animation generation complete: %s", out_dir / "animations")
                substep_ok["animation"] = True
            else:
                log.info("No FLAME binding metadata; skipping animation generation")
                substep_ok["animation"] = None
    except Exception as e:
        log.warning("Animation generation failed (non-fatal): %s", e)
        substep_ok["animation"] = False

    # ---- 7. Quality report, comparison grid, turntable GIF --------------
    try:
        _generate_final_report(config, session)
        substep_ok["report"] = True
    except Exception as e:
        log.warning("Quality report generation failed: %s", e)
        substep_ok["report"] = False

    # ---- Summary --------------------------------------------------------
    failed = [k for k, v in substep_ok.items() if v is False]
    skipped = [k for k, v in substep_ok.items() if v is None]
    passed = [k for k, v in substep_ok.items() if v is True]
    log.info(
        "Stage 14 sub-step results: passed=%s, skipped=%s, failed=%s",
        passed, skipped, failed,
    )

    mark_stage_complete(marker)
    return True


# ---------------------------------------------------------------------------
# Visualization helpers (called from stage functions)
# ---------------------------------------------------------------------------

def _pick_evenly_spaced(file_list: list[Path], count: int = 3) -> list[Path]:
    """Pick *count* evenly-spaced items from a sorted list."""
    import numpy as np
    if not file_list:
        return []
    n = min(count, len(file_list))
    indices = np.linspace(0, len(file_list) - 1, n, dtype=int)
    return [file_list[i] for i in indices]


def _save_depth_previews(session: dict, num_previews: int = 3) -> None:
    """Save colormapped depth map previews after stage 7."""
    try:
        from utils.visualization import visualize_depth_map
    except ImportError:
        log.debug("visualization module not available, skipping depth previews")
        return

    depth_dir = session["proc_dir"] / "depth"
    previews_dir = session["out_dir"] / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(depth_dir.glob("*.npy"))
    selected = _pick_evenly_spaced(npy_files, num_previews)

    for depth_path in selected:
        try:
            out = previews_dir / f"depth_{depth_path.stem}.png"
            visualize_depth_map(depth_path, out)
        except Exception as e:
            log.debug("Failed to visualize depth %s: %s", depth_path.name, e)

    if selected:
        log.info("Saved %d depth preview(s) to %s", len(selected), previews_dir)


def _save_landmark_previews(session: dict, num_previews: int = 3) -> None:
    """Save landmark overlay previews after stage 9."""
    try:
        from utils.visualization import visualize_landmarks
    except ImportError:
        log.debug("visualization module not available, skipping landmark previews")
        return

    frames_dir = session["proc_dir"] / "frames_srgb"
    landmarks_dir = session["proc_dir"] / "landmarks"
    previews_dir = session["out_dir"] / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    lm_files = sorted(landmarks_dir.glob("*.json"))
    selected = _pick_evenly_spaced(lm_files, num_previews)

    for lm_path in selected:
        # Find matching frame
        stem = lm_path.stem
        frame_path = frames_dir / f"{stem}.png"
        if not frame_path.exists():
            frame_path = frames_dir / f"{stem}.jpg"
        if not frame_path.exists():
            continue
        try:
            out = previews_dir / f"landmarks_{stem}.png"
            visualize_landmarks(frame_path, lm_path, out)
        except Exception as e:
            log.debug("Failed to visualize landmarks for %s: %s", stem, e)

    if selected:
        log.info("Saved %d landmark preview(s) to %s", len(selected), previews_dir)


def _save_mask_previews(session: dict, num_previews: int = 3) -> None:
    """Save mask overlay previews after stage 11."""
    try:
        from utils.visualization import visualize_mask_overlay
    except ImportError:
        log.debug("visualization module not available, skipping mask previews")
        return

    frames_dir = session["proc_dir"] / "frames_srgb"
    masks_dir = session["proc_dir"] / "face_masks"
    previews_dir = session["out_dir"] / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(
        list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.npy"))
    )
    selected = _pick_evenly_spaced(mask_files, num_previews)

    for mask_path in selected:
        stem = mask_path.stem
        frame_path = frames_dir / f"{stem}.png"
        if not frame_path.exists():
            frame_path = frames_dir / f"{stem}.jpg"
        if not frame_path.exists():
            continue
        try:
            out = previews_dir / f"mask_{stem}.png"
            visualize_mask_overlay(frame_path, mask_path, out)
        except Exception as e:
            log.debug("Failed to visualize mask for %s: %s", stem, e)

    if selected:
        log.info("Saved %d mask preview(s) to %s", len(selected), previews_dir)


def _generate_final_report(config: dict, session: dict) -> None:
    """Generate quality report, comparison grid, and turntable GIF after stage 14."""
    previews_dir = session["out_dir"] / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    # Comparison grid
    try:
        from utils.quality_report import generate_comparison_grid
        generate_comparison_grid(
            images_dir=session["proc_dir"] / "frames_srgb",
            renders_dir=session["out_dir"] / "renders",
            output_path=previews_dir / "comparison_grid.png",
            num_comparisons=5,
        )
    except Exception as e:
        log.warning("Failed to generate comparison grid: %s", e)

    # Turntable GIF
    try:
        from utils.visualization import create_turntable_gif
        create_turntable_gif(
            renders_dir=session["out_dir"] / "renders",
            output_path=previews_dir / "turntable.gif",
            fps=config.get("splatting", {}).get("export", {}).get("gif_fps", 15),
            max_size=512,
        )
    except Exception as e:
        log.warning("Failed to create turntable GIF: %s", e)

    # HTML quality report
    try:
        from utils.quality_report import generate_quality_report
        generate_quality_report(
            session_dir=session["out_dir"],
            output_path=session["out_dir"] / "report.html",
        )
    except Exception as e:
        log.warning("Failed to generate quality report: %s", e)


# ---------------------------------------------------------------------------
# Pipeline summary
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format seconds as 'Xm Ys'."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _file_size_mb(path: Path) -> str:
    """Return file size as 'X.XMB' or 'N/A'."""
    if path.exists():
        return f"{path.stat().st_size / (1024 * 1024):.1f}MB"
    return "N/A"


def print_pipeline_summary(session: dict, total_elapsed: float):
    """Print a formatted summary of the pipeline run."""
    proc = session["proc_dir"]
    out = session["out_dir"]
    metrics_path = out / "metrics.jsonl"

    # Collect metrics from jsonl
    stage_metrics = {}
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            try:
                entry = json.loads(line)
                stage_metrics[entry["stage"]] = entry.get("metrics", {})
            except (json.JSONDecodeError, KeyError):
                pass

    n_extracted = stage_metrics.get(1, {}).get("frames_extracted", "?")
    n_filtered = stage_metrics.get(3, {}).get("frames_selected", "?")

    # Reconstruction info
    s6 = stage_metrics.get(6, {})
    recon_method = s6.get("method", "unknown")
    n_registered = s6.get("registered_images", s6.get("poses", "?"))

    # Depth info
    s7 = stage_metrics.get(7, {})
    n_depth = s7.get("depth_maps", "?")

    # FLAME info
    s10 = stage_metrics.get(10, {})
    flame_fitted = s10.get("flame_fitted", False)

    # Gaussian info
    s12 = stage_metrics.get(12, {})
    n_init_gauss = s12.get("initial_gaussians", "?")
    s13 = stage_metrics.get(13, {})
    n_final_gauss = s13.get("final_gaussians", n_init_gauss)
    train_iters = s13.get("iterations", "?")
    final_psnr = s13.get("final_psnr", "?")

    # Output files
    ply_path = out / "gaussians.ply"
    mesh_ply = out / "mesh" / "mesh.ply"
    mesh_obj = out / "mesh.obj"
    mesh_file = mesh_ply if mesh_ply.exists() else mesh_obj
    n_renders = _count_files(out / "renders", "*.png")

    summary = f"""
{'=' * 40}
  Face3D Pipeline Complete
{'=' * 40}
Session:    {session['session_id']}
Duration:   {_format_duration(total_elapsed)}
Frames:     {n_extracted} extracted -> {n_filtered} filtered -> {n_registered} registered
Recon:      {recon_method.upper()} ({n_depth} depth maps)
FLAME:      {'fitted' if flame_fitted else 'not fitted'}
Gaussians:  {n_final_gauss} trained ({train_iters} iterations{f', final PSNR {final_psnr}' if final_psnr != '?' else ''})
Output:     {out}
  - gaussians.ply ({_file_size_mb(ply_path)})
  - mesh ({_file_size_mb(mesh_file)})
  - renders/ ({n_renders} turntable views)
  - metrics.jsonl
{'=' * 40}
"""
    log.info(summary)


# ---------------------------------------------------------------------------
# Stage registry & main
# ---------------------------------------------------------------------------

STAGES = [
    (0, "Organize inputs", stage_0_organize_inputs),
    (1, "Extract frames", stage_1_extract_frames),
    (2, "Color correction", stage_2_color_correct),
    (3, "Filter frames", stage_3_filter_frames),
    (4, "Parse sensors", stage_4_parse_sensors),
    (5, "Compute priors", stage_5_compute_priors),
    (6, "COLMAP SfM / DA3 Unified", stage_6_colmap),
    (7, "Depth estimation", stage_7_depth),
    (8, "Align depth", stage_8_align_depth),
    (9, "Detect landmarks", stage_9_landmarks),
    (10, "Fit FLAME", stage_10_flame),
    (11, "Segment face", stage_11_segment),
    (12, "Initialize Gaussians", stage_12_init_gaussians),
    (13, "Train Gaussians", stage_13_train_gaussians),
    (14, "Export", stage_14_export),
]


def _auto_detect_content_dir(content_dir: Path) -> dict:
    """Scan a directory for videos, photos, and sensor logs.

    Returns a dict with keys: videos, photos, sensor_logs (lists of Paths).
    """
    if not content_dir.exists():
        log.error("Content directory does not exist: %s", content_dir)
        sys.exit(1)

    videos = []
    photos = []
    sensor_logs = []

    for f in sorted(content_dir.iterdir()):
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        if suffix in (".mp4", ".mov"):
            videos.append(f)
        elif suffix == ".dng":
            photos.append(f)
        elif suffix in (".jpg", ".jpeg"):
            # Include JPGs — either they have a matching DNG or are standalone
            photos.append(f)
        elif suffix == ".zip":
            # Validate it looks like a Sensor Logger archive
            try:
                import zipfile as _zf
                with _zf.ZipFile(f, "r") as zf:
                    names = zf.namelist()
                    sensor_csvs = (
                        "Accelerometer.csv", "Gyroscope.csv",
                        "Orientation.csv", "Magnetometer.csv",
                        "Light.csv", "Barometer.csv",
                    )
                    has_sensor_csv = any(
                        n.endswith(sn) for n in names for sn in sensor_csvs
                    )
                if has_sensor_csv:
                    sensor_logs.append(f)
                else:
                    log.debug("ZIP %s does not appear to be a Sensor Logger archive", f.name)
            except Exception:
                log.debug("Could not read ZIP %s, skipping", f.name)

    # Classify photos by type for the summary log
    n_dng = sum(1 for p in photos if p.suffix.lower() == ".dng")
    n_jpg = sum(1 for p in photos if p.suffix.lower() in (".jpg", ".jpeg"))
    log.info(
        "Auto-detected from %s: %d video(s), %d DNG(s), %d JPG(s), %d sensor log(s)",
        content_dir, len(videos), n_dng, n_jpg, len(sensor_logs),
    )

    return {"videos": videos, "photos": photos, "sensor_logs": sensor_logs}


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate a list of Paths by their resolved absolute path."""
    seen = set()
    unique = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def resolve_inputs(args) -> dict:
    """Normalize all input sources into a single dict.

    Merges --content-dir auto-detection with explicit --video, --photos,
    --sensor-log arguments. Returns:
        {"videos": [Path], "photos": [Path], "sensor_logs": [Path]}
    """
    videos = []
    photos = []
    sensor_logs = []

    # Auto-detect from --content-dir
    if args.content_dir:
        detected = _auto_detect_content_dir(Path(args.content_dir))
        videos.extend(detected["videos"])
        photos.extend(detected["photos"])
        sensor_logs.extend(detected["sensor_logs"])

    # Merge explicit --video
    if args.video:
        videos.extend(Path(v) for v in args.video)

    # Merge explicit --photos (with glob expansion)
    if args.photos:
        import glob as glob_mod
        for pattern in args.photos:
            expanded = glob_mod.glob(pattern)
            if expanded:
                photos.extend(Path(p) for p in expanded)
            else:
                p = Path(pattern)
                if p.exists():
                    photos.append(p)
                else:
                    log.warning("Photo path not found: %s", pattern)

    # Merge explicit --sensor-log
    if args.sensor_log:
        for sl in args.sensor_log:
            sl_path = Path(sl)
            if sl_path.exists():
                sensor_logs.append(sl_path)
            else:
                log.warning("Sensor Logger ZIP not found: %s", sl_path)

    return {
        "videos": _deduplicate_paths(videos),
        "photos": _deduplicate_paths(photos),
        "sensor_logs": _deduplicate_paths(sensor_logs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Face3D Pipeline — 3D face reconstruction from S25 Ultra captures",
        epilog=(
            "Examples:\n"
            "  python scripts/run_pipeline.py --content-dir content/New --session natalie\n"
            "  python scripts/run_pipeline.py --video face.mp4\n"
            "  python scripts/run_pipeline.py --video face.mp4 --photos *.dng --sensor-log sensors.zip\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", nargs="+", default=None, help="Input video file(s)")
    parser.add_argument("--content-dir", default=None,
                        help="Directory with mixed video/photos/sensors (auto-detect all)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "pipeline.yaml"))
    parser.add_argument("--session", default=None, help="Session ID (auto-generated if omitted)")
    parser.add_argument("--start-stage", type=int, default=0, help="Stage to start from")
    parser.add_argument("--end-stage", type=int, default=14, help="Stage to end at")
    parser.add_argument("--force", action="store_true", help="Force re-run even if stage is complete")
    parser.add_argument("--photos", nargs="*", default=None,
                        help="Expert RAW photos for high-quality texture (glob patterns or file paths)")
    parser.add_argument("--sensor-log", nargs="*", default=None,
                        help="Sensor Logger ZIP file(s) for high-quality IMU/orientation data")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # Resolve all inputs from --content-dir and/or explicit args
    inputs = resolve_inputs(args)

    if not inputs["videos"]:
        parser.error(
            "No video files found. Use --video or --content-dir with a "
            "directory containing .mp4/.mov files."
        )

    # Validate config and inputs
    validate_config(config, inputs["videos"])

    # Setup session with organized folder structure
    session = setup_session(config, args.session, [str(v) for v in inputs["videos"]])
    session["photo_paths"] = inputs["photos"]
    session["sensor_log_paths"] = inputs["sensor_logs"]
    session["sensor_log_path"] = inputs["sensor_logs"][0] if inputs["sensor_logs"] else None

    # Log what we found
    if inputs["photos"]:
        log.info("Photos: %d files", len(inputs["photos"]))
    if inputs["sensor_logs"]:
        log.info("Sensor Logger: %d file(s) — %s", len(inputs["sensor_logs"]),
                 ", ".join(p.name for p in inputs["sensor_logs"]))

    log.info("Session: %s", session["session_id"])
    log.info("Videos: %s", [str(v) for v in session["video_paths"]])
    log.info("Output: %s", session["out_dir"])

    profiler = PipelineProfiler(session_id=session["session_id"])

    t0 = time.time()
    for stage_num, stage_name, stage_fn in STAGES:
        if stage_num < args.start_stage or stage_num > args.end_stage:
            continue

        if args.force:
            for marker_dir in [session["proc_dir"], session["out_dir"]]:
                marker = marker_dir / f".stage_{stage_num}_complete"
                marker.unlink(missing_ok=True)

        log.info("=" * 60)
        log.info("Stage %d: %s", stage_num, stage_name)
        log.info("=" * 60)

        profiler.start_stage(stage_name, stage_num=stage_num)
        stage_start = time.time()
        stage_error = None
        try:
            success = stage_fn(config, session)
            if not success:
                stage_error = "returned False"
                log.error("Stage %d failed", stage_num)
                profiler.end_stage(stage_name, error=stage_error)
                sys.exit(1)
        except Exception as exc:
            stage_error = str(exc)
            log.exception("Stage %d (%s) failed with error", stage_num, stage_name)
            profiler.end_stage(stage_name, error=stage_error)
            profiler.summary()
            sys.exit(1)

        elapsed = time.time() - stage_start

        # Free GPU memory after GPU-heavy stages to avoid VRAM pressure on next stage
        # Stages 6 (DA3/COLMAP), 13 (training), 14 (export) are the main GPU consumers
        if stage_num in (6, 13, 14):
            try:
                import torch
                if torch.cuda.is_available():
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    log.debug("Freed GPU memory after stage %d", stage_num)
            except ImportError:
                pass

        # Validate stage outputs and collect metrics
        try:
            metrics = validate_stage_output(stage_num, config, session)
        except RuntimeError as e:
            log.error("Stage %d validation failed: %s", stage_num, e)
            profiler.end_stage(stage_name, error=str(e))
            profiler.summary()
            sys.exit(1)

        # Handle stage 3 retry with lower blur_threshold
        if stage_num == 3 and metrics.get("_retry_filter"):
            log.info("Retrying stage 3 with blur_threshold=20...")
            # Remove marker and re-run
            marker_path = session["proc_dir"] / ".stage_3_complete"
            marker_path.unlink(missing_ok=True)

            from capture.frame_filter import filter_frames
            filt_cfg = config["capture"]["filtering"]
            filter_frames(
                frames_dir=session["proc_dir"] / "frames_srgb",
                output_json=session["proc_dir"] / "selected_frames.json",
                blur_threshold=20.0,
                exposure_low=filt_cfg.get("exposure_low", 30),
                exposure_high=filt_cfg.get("exposure_high", 225),
                require_face=filt_cfg.get("require_face", True),
            )
            mark_stage_complete(marker_path)

            # Re-validate
            metrics = validate_stage_output(3, config, session)
            del metrics["_retry_filter"]
            if metrics.get("frames_selected", 0) < 5:
                log.error(
                    "Still only %d frames after retry. Video may not have enough usable frames.",
                    metrics.get("frames_selected", 0),
                )
                sys.exit(1)
            elapsed = time.time() - stage_start

        # Remove internal flags from metrics before logging
        metrics.pop("_retry_filter", None)

        # Determine item count for throughput calculation
        num_items = 0
        for key in ("frames_extracted", "frames_selected", "depth_maps",
                     "landmark_files", "initial_gaussians", "registered_images"):
            if key in metrics and isinstance(metrics[key], int):
                num_items = metrics[key]
                break

        profiler.end_stage(stage_name, num_items=num_items)

        log_metrics(session, stage_num, elapsed, metrics)
        log.info("Stage %d completed in %.1fs", stage_num, elapsed)

    total = time.time() - t0
    log.info("Pipeline complete in %.1fs", total)

    # Print profiler summary table
    profiler.summary()

    # Save profiling JSON
    profiler.save_json(session["out_dir"] / "pipeline_timing.json")

    # Print formatted summary
    print_pipeline_summary(session, total)


if __name__ == "__main__":
    main()
