#!/usr/bin/env python3
"""Face3D Pipeline Orchestrator

Runs the full face reconstruction pipeline from S25 Ultra video to Gaussian Splats.
Each stage checks for existing outputs and skips if already complete (resumable).

Usage:
    python scripts/run_pipeline.py --video path/to/video.mp4
    python scripts/run_pipeline.py --video path/to/video.mp4 --session my_session
    python scripts/run_pipeline.py --session my_session --start-stage 4  # Resume from stage 4
    python scripts/run_pipeline.py --config config/pipeline.yaml --video path/to/video.mp4
"""

import argparse
import json
import logging
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
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
    """Create session directory structure and return session info."""
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_root = Path(config["session"]["data_root"])
    raw_dir = data_root / "raw" / session_id
    proc_dir = data_root / "processed" / session_id
    out_dir = data_root / "output" / session_id

    for d in [
        raw_dir,
        proc_dir / "frames",
        proc_dir / "frames_srgb",
        proc_dir / "depth",
        proc_dir / "colmap" / "sparse",
        proc_dir / "landmarks",
        proc_dir / "face_masks",
        proc_dir / "flame",
        out_dir / "renders",
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
    process_res = da3_cfg.get("process_res", 504)
    use_ray_pose = da3_cfg.get("use_ray_pose", False)
    chunk_size = da3_cfg.get("chunk_size", 8)
    conf_threshold = da3_cfg.get("conf_threshold", 0.3)

    summary = run_da3_unified(
        frames_dir=session["proc_dir"] / "frames_srgb",
        output_dir=session["proc_dir"],
        model_name=model_name,
        process_res=process_res,
        use_ray_pose=use_ray_pose,
        chunk_size=chunk_size,
        conf_threshold=conf_threshold,
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

def stage_1_extract_frames(config: dict, session: dict) -> bool:
    """Extract frames from video with motion-based sampling."""
    marker = session["proc_dir"] / ".stage_1_complete"
    if stage_complete(marker):
        log.info("Stage 1: Frame extraction already complete, skipping")
        return True

    log.info("Stage 1: Extracting frames from video...")
    from capture import extract_frames

    cap_cfg = config["capture"]
    frames_dir = session["proc_dir"] / "frames"

    for video_path in session["video_paths"]:
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
    if photo_paths and photos_cfg.get("enabled", True):
        log.info("Stage 1: Processing %d Expert RAW photos...", len(photo_paths))
        from capture.photo_processor import process_photos

        photos_dir = session["proc_dir"] / "photos"
        processed_photos, photo_metadata = process_photos(
            photo_paths=photo_paths,
            output_dir=photos_dir,
        )
        session["processed_photo_paths"] = processed_photos
        session["photo_metadata"] = photo_metadata
        log.info("Stage 1: Processed %d photos -> %s", len(processed_photos), photos_dir)

        # Also copy processed photos into frames_srgb so they are included
        # in downstream stages (DA3, COLMAP, training). Photos are named
        # photo_NNNN.png to distinguish from video frame_NNNNNN.png files.
        srgb_dir = session["proc_dir"] / "frames_srgb"
        srgb_dir.mkdir(parents=True, exist_ok=True)
        for pp in processed_photos:
            dst = srgb_dir / pp.name
            if not dst.exists():
                shutil.copy2(pp, dst)
                log.debug("Copied photo %s to frames_srgb/", pp.name)

        # Save a manifest so downstream stages know which images are photos
        manifest_path = session["proc_dir"] / "photo_manifest.json"
        manifest = {
            "photo_names": [p.name for p in processed_photos],
            "weight": photos_cfg.get("weight", 3.0),
            "count": len(processed_photos),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
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
    """Parse IMU data from video metadata or sidecar files."""
    marker = session["proc_dir"] / ".stage_4_complete"
    if stage_complete(marker):
        log.info("Stage 4: Sensor parsing already complete, skipping")
        return True

    sensor_cfg = config["sensors"]
    if not sensor_cfg["imu"].get("enabled", True):
        log.info("Stage 4: IMU disabled, skipping")
        mark_stage_complete(marker)
        return True

    log.info("Stage 4: Parsing IMU/sensor data...")
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


def stage_5_compute_priors(config: dict, session: dict) -> bool:
    """Compute rotation priors from IMU for COLMAP."""
    marker = session["proc_dir"] / ".stage_5_complete"
    if stage_complete(marker):
        log.info("Stage 5: Prior computation already complete, skipping")
        return True

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
        photos_dir = session["proc_dir"] / "photos"
        colmap_model_dir = session["proc_dir"] / "colmap" / "sparse" / "0"
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
    """Detect facial landmarks on all frames."""
    marker = session["proc_dir"] / ".stage_9_complete"
    if stage_complete(marker):
        log.info("Stage 9: Landmark detection already complete, skipping")
        return True

    log.info("Stage 9: Detecting facial landmarks...")
    from reconstruction import FaceLandmarkDetector

    detector = FaceLandmarkDetector()
    detector.detect_batch(
        frames_dir=session["proc_dir"] / "frames_srgb",
        output_dir=session["proc_dir"] / "landmarks",
    )

    # Save landmark overlay previews
    _save_landmark_previews(session)

    mark_stage_complete(marker)
    return True


def stage_10_flame(config: dict, session: dict) -> bool:
    """Fit FLAME parametric face model."""
    marker = session["proc_dir"] / ".stage_10_complete"
    if stage_complete(marker):
        log.info("Stage 10: FLAME fitting already complete, skipping")
        return True

    log.info("Stage 10: Fitting FLAME face model...")
    from reconstruction import fit_flame_to_sequence

    flame_cfg = config["reconstruction"]["flame"]
    try:
        fit_flame_to_sequence(
            frames_dir=session["proc_dir"] / "frames_srgb",
            landmarks_dir=session["proc_dir"] / "landmarks",
            colmap_model_dir=session["proc_dir"] / "colmap" / "sparse" / "0",
            flame_model_path=flame_cfg["model_path"],
            embedding_path=flame_cfg["embedding_path"],
            output_dir=session["proc_dir"] / "flame",
        )
    except Exception as e:
        log.warning("FLAME fitting failed (non-fatal for non-face data): %s", e)
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
    photo_manifest_path = session["proc_dir"] / "photo_manifest.json"
    photo_names = set()
    photo_weight = 1.0
    if photo_manifest_path.exists():
        manifest = json.loads(photo_manifest_path.read_text(encoding="utf-8"))
        photo_names = set(manifest.get("photo_names", []))
        photos_cfg = config.get("photos", {})
        photo_weight = photos_cfg.get("weight", manifest.get("weight", 3.0))
        if photo_names:
            log.info("Stage 13: %d photo views will be weighted %.1fx during training",
                     len(photo_names), photo_weight)

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

    # ---- 1. Standard PLY export ----------------------------------------
    try:
        from splatting.exporter import export_gaussians_ply

        export_gaussians_ply(gaussians, out_dir / "gaussians.ply")
        substep_ok["ply"] = True
    except Exception as e:
        log.error("PLY export failed: %s", e, exc_info=True)
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
            images_dir = session["proc_dir"] / "frames"

            cameras = load_cameras_from_colmap(colmap_dir)

            # Check for FLAME texture UVs
            flame_tex = PROJECT_ROOT / "Models" / "Flame" / "FLAME_texture.npz"
            flame_texture_path = flame_tex if flame_tex.exists() else None

            tex_res = export_cfg.get("texture_resolution", 2048)
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

    # ---- 4. Compressed export -------------------------------------------
    try:
        from splatting.exporter import export_compressed

        export_compressed(gaussians, out_dir / "gaussians_compressed")
        substep_ok["compressed"] = True
    except Exception as e:
        log.warning("Compressed export failed: %s", e)
        substep_ok["compressed"] = False

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


def main():
    parser = argparse.ArgumentParser(description="Face3D Pipeline")
    parser.add_argument("--video", nargs="+", required=True, help="Input video file(s)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "pipeline.yaml"))
    parser.add_argument("--session", default=None, help="Session ID (auto-generated if omitted)")
    parser.add_argument("--start-stage", type=int, default=1, help="Stage to start from")
    parser.add_argument("--end-stage", type=int, default=14, help="Stage to end at")
    parser.add_argument("--force", action="store_true", help="Force re-run even if stage is complete")
    parser.add_argument("--photos", nargs="*", default=None,
                        help="Expert RAW photos for high-quality texture (glob patterns or file paths)")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # Validate config and inputs before anything else
    video_paths = [Path(v) for v in args.video]
    validate_config(config, video_paths)

    session = setup_session(config, args.session, args.video)

    # Resolve photo paths (expand glob patterns)
    if args.photos:
        import glob as glob_mod
        photo_paths = []
        for pattern in args.photos:
            expanded = glob_mod.glob(pattern)
            if expanded:
                photo_paths.extend(Path(p) for p in expanded)
            else:
                # Treat as literal path
                p = Path(pattern)
                if p.exists():
                    photo_paths.append(p)
                else:
                    log.warning("Photo path not found: %s", pattern)
        session["photo_paths"] = photo_paths
        if photo_paths:
            log.info("Photos: %d files provided", len(photo_paths))
    else:
        session["photo_paths"] = []

    log.info("Session: %s", session["session_id"])
    log.info("Videos: %s", [str(v) for v in session["video_paths"]])
    log.info("Output: %s", session["out_dir"])

    t0 = time.time()
    for stage_num, stage_name, stage_fn in STAGES:
        if stage_num < args.start_stage or stage_num > args.end_stage:
            continue

        if args.force:
            for marker_dir in [session["proc_dir"], session["out_dir"]]:
                marker = marker_dir / f".stage_{stage_num}_complete"
                marker.unlink(missing_ok=True)

        log.info("=" * 60)
        log.info("Stage %d/14: %s", stage_num, stage_name)
        log.info("=" * 60)

        stage_start = time.time()
        try:
            success = stage_fn(config, session)
            if not success:
                log.error("Stage %d failed", stage_num)
                sys.exit(1)
        except Exception:
            log.exception("Stage %d (%s) failed with error", stage_num, stage_name)
            sys.exit(1)

        elapsed = time.time() - stage_start

        # Validate stage outputs and collect metrics
        try:
            metrics = validate_stage_output(stage_num, config, session)
        except RuntimeError as e:
            log.error("Stage %d validation failed: %s", stage_num, e)
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

        log_metrics(session, stage_num, elapsed, metrics)
        log.info("Stage %d completed in %.1fs", stage_num, elapsed)

    total = time.time() - t0
    log.info("Pipeline complete in %.1fs", total)

    # Print formatted summary
    print_pipeline_summary(session, total)


if __name__ == "__main__":
    main()
