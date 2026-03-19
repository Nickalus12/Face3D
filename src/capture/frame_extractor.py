"""Frame extraction from Samsung Galaxy S25 Ultra Pro Video (H.265/HEVC).

Supports fixed-FPS and motion-based adaptive extraction with optical flow,
plus lens metadata parsing for multi-lens grouping.

Includes hardware-accelerated decoding (NVIDIA CUDA), ffmpeg-native frame
selection filters, ffprobe-based EXIF/focal-length extraction, and adaptive
motion thresholds calibrated from initial optical flow statistics.

Also provides LOG profile detection via video metadata so that downstream
colour correction can be skipped for non-LOG footage.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.timing import timed

logger = logging.getLogger(__name__)

# Samsung S25 Ultra lens focal lengths (35mm equiv) mapped to sensor crop info
S25_ULTRA_LENSES = {
    13: "ultrawide",
    23: "wide",
    70: "telephoto",
    200: "supertelephoto",
}

_KNOWN_FOCAL_LENGTHS = sorted(S25_ULTRA_LENSES.keys())


def _snap_focal_length(focal_mm: float) -> int:
    """Snap a raw focal length to the nearest known S25 Ultra lens."""
    return min(_KNOWN_FOCAL_LENGTHS, key=lambda k: abs(k - focal_mm))


def _check_cuda_available() -> bool:
    """Check whether ffmpeg was built with CUDA/NVDEC support."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=10,
        )
        return "cuda" in result.stdout.lower()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# Cache the CUDA availability check so it only runs once per process.
_CUDA_AVAILABLE: Optional[bool] = None


def _is_cuda_available() -> bool:
    """Return cached result of CUDA availability probe."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        _CUDA_AVAILABLE = _check_cuda_available()
        if _CUDA_AVAILABLE:
            logger.info("NVIDIA CUDA hwaccel available for ffmpeg decoding")
        else:
            logger.debug("CUDA hwaccel not available; using CPU decoding")
    return _CUDA_AVAILABLE


def _hwaccel_input_flags() -> list[str]:
    """Return ffmpeg input flags for hardware-accelerated HEVC decoding.

    Falls back to an empty list when CUDA is not available.
    """
    if _is_cuda_available():
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    return []


def _probe_video(video_path: Path) -> dict:
    """Use ffprobe to extract stream and format metadata."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _get_fps(probe_data: dict) -> float:
    """Extract frame rate from ffprobe output."""
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            r_frame_rate = stream.get("r_frame_rate", "30/1")
            if "/" in r_frame_rate:
                num, den = r_frame_rate.split("/")
                return float(num) / float(den) if float(den) != 0 else 30.0
            return float(r_frame_rate)
    return 30.0


def _get_duration(probe_data: dict) -> float:
    """Extract duration in seconds from ffprobe output."""
    fmt = probe_data.get("format", {})
    if "duration" in fmt:
        return float(fmt["duration"])
    for stream in probe_data.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    return 0.0


def _extract_focal_length_ffprobe(video_path: Path) -> Optional[float]:
    """Extract focal length from video/image metadata using ffprobe.

    Parses ``-show_streams`` JSON for FocalLength and
    FocalLengthIn35mmFormat tags embedded by Samsung camera firmware.

    Returns:
        Focal length in mm (35 mm-equivalent preferred), or ``None``.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)

        # Collect all tag dictionaries (format-level + per-stream)
        tag_dicts: list[dict] = []
        tag_dicts.append(data.get("format", {}).get("tags", {}))
        for stream in data.get("streams", []):
            tag_dicts.append(stream.get("tags", {}))
            for sd in stream.get("side_data_list", []):
                if isinstance(sd, dict):
                    tag_dicts.append(sd)

        # Prefer 35 mm-equivalent focal length, fall back to raw
        priority_keys = [
            "FocalLengthIn35mmFormat",
            "FocalLengthIn35mmFilm",
            "com.android.capture.focalLength",
            "com.samsung.android.capture.focalLength",
            "FocalLength",
            "focal_length",
        ]
        for key in priority_keys:
            for tags in tag_dicts:
                if key in tags:
                    raw = str(tags[key])
                    cleaned = re.sub(r"[^\d.]", "", raw)
                    if cleaned:
                        return float(cleaned)
    except (subprocess.TimeoutExpired, ValueError, json.JSONDecodeError, OSError):
        pass
    return None


def _extract_lens_from_exif(frame_path: Path) -> Optional[str]:
    """Attempt to read EXIF focal length from a frame using ffprobe.

    For frames extracted from video, EXIF is generally unavailable per-frame.
    This is a best-effort extraction that works when metadata is embedded.
    """
    focal = _extract_focal_length_ffprobe(frame_path)
    if focal is not None:
        snapped = _snap_focal_length(focal)
        return S25_ULTRA_LENSES.get(snapped)
    return None


def _detect_lens_from_video_metadata(video_path: Path) -> Optional[str]:
    """Try to detect the lens used from video-level metadata tags."""
    try:
        focal = _extract_focal_length_ffprobe(video_path)
        if focal is not None:
            snapped = _snap_focal_length(focal)
            return S25_ULTRA_LENSES.get(snapped)
    except (ValueError, OSError) as exc:
        logger.debug("Could not detect lens from video metadata: %s", exc)
    return None


def detect_log_profile(video_path: str | Path) -> Optional[str]:
    """Detect whether a video was recorded with a LOG color profile.

    Inspects video-level and stream-level metadata tags for Samsung LOG
    indicators (e.g. ``com.samsung.android.capture.colorSpace``,
    ``ColorSpace``, filename hints).

    Returns:
        The LOG profile name (e.g. ``"slog3"``) if detected, or ``None``
        when the video appears to use a standard (Rec.709 / sRGB) profile.
    """
    video_path = Path(video_path)

    # Quick filename heuristic (often Samsung LOG files contain "LOG" in name)
    if "log" in video_path.stem.lower():
        logger.info("LOG profile detected via filename hint: %s", video_path.name)
        return "slog3"

    try:
        probe = _probe_video(video_path)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None

    # Collect all tag dictionaries from format + streams
    tag_dicts: list[dict] = []
    tag_dicts.append(probe.get("format", {}).get("tags", {}))
    for stream in probe.get("streams", []):
        tag_dicts.append(stream.get("tags", {}))
        for sd in stream.get("side_data_list", []):
            if isinstance(sd, dict):
                tag_dicts.append(sd)

    # Samsung-specific LOG indicators
    log_keys = [
        "com.samsung.android.capture.colorSpace",
        "com.samsung.android.sceneMode",
        "com.android.capture.colorSpace",
        "ColorSpace",
        "color_space",
    ]

    log_indicators = {"log", "slog", "slog3", "s-log3", "hlg", "flat", "cine"}

    for key in log_keys:
        for tags in tag_dicts:
            value = tags.get(key, "")
            if isinstance(value, str) and any(ind in value.lower() for ind in log_indicators):
                logger.info("LOG profile detected via metadata '%s': '%s'", key, value)
                return "slog3"

    # Check color_transfer / color_trc in stream info
    for stream in probe.get("streams", []):
        trc = stream.get("color_transfer", stream.get("color_trc", "")).lower()
        if any(ind in trc for ind in ("slog", "log", "hlg", "arib-std-b67")):
            logger.info("LOG profile detected via color_transfer: '%s'", trc)
            return "slog3"

    logger.info("No LOG profile detected for %s — standard color assumed", video_path.name)
    return None


def _compute_optical_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute mean optical flow magnitude between two grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def _calibrate_motion_threshold(
    cap: cv2.VideoCapture,
    small_size: tuple[int, int],
    calibration_frames: int = 50,
    multiplier: float = 1.5,
) -> tuple[float, np.ndarray]:
    """Compute an adaptive motion threshold from the first *calibration_frames*.

    Reads up to *calibration_frames* consecutive frame pairs, computes optical
    flow magnitude for each, and returns ``median_flow * multiplier`` as the
    threshold.  The capture position is reset to frame 0 afterwards.

    Args:
        cap: An opened ``cv2.VideoCapture`` (position will be reset).
        small_size: (width, height) to resize frames before flow computation.
        calibration_frames: Number of frames to sample.
        multiplier: Factor applied to the median flow to produce the threshold.

    Returns:
        Tuple of (threshold, first_frame_gray_at_small_size).
    """
    original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, first = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        return 2.0, np.empty(0)

    prev_gray = cv2.cvtColor(cv2.resize(first, small_size), cv2.COLOR_BGR2GRAY)
    first_gray = prev_gray.copy()
    flow_values: list[float] = []

    for _ in range(calibration_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(cv2.resize(frame, small_size), cv2.COLOR_BGR2GRAY)
        flow_mag = _compute_optical_flow_magnitude(prev_gray, curr_gray)
        flow_values.append(flow_mag)
        prev_gray = curr_gray

    # Reset capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if flow_values:
        median_flow = float(np.median(flow_values))
        threshold = median_flow * multiplier
        logger.info(
            "Adaptive threshold calibration: median_flow=%.3f, threshold=%.3f "
            "(from %d frame pairs)",
            median_flow, threshold, len(flow_values),
        )
        # Ensure a sane minimum so we don't save every single frame
        return max(threshold, 0.5), first_gray
    return 2.0, first_gray


def _save_lens_metadata(
    frame_paths: list[Path],
    video_path: Path,
    output_dir: Path,
) -> Path:
    """Save lens grouping metadata as JSON alongside extracted frames."""
    video_lens = _detect_lens_from_video_metadata(video_path)

    lens_data: dict[str, object] = {
        "video_source": str(video_path),
        "detected_lens": video_lens or "unknown",
        "lens_lookup": S25_ULTRA_LENSES,
        "frames": {},
    }

    for fp in frame_paths:
        per_frame_lens = _extract_lens_from_exif(fp) or video_lens or "unknown"
        lens_data["frames"][fp.name] = {  # type: ignore[index]
            "path": str(fp),
            "lens": per_frame_lens,
        }

    meta_path = output_dir / "lens_metadata.json"
    meta_path.write_text(json.dumps(lens_data, indent=2), encoding="utf-8")
    logger.info("Lens metadata saved to %s", meta_path)
    return meta_path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    target_fps: float = 2.0,
    max_frames: int = 80,
) -> list[Path]:
    """Extract frames from an S25 Ultra H.265/HEVC video at a target FPS.

    Uses ffmpeg with ``-vf select=not(mod(n\\,N))`` for efficient uniform
    frame sampling at the demuxer level, avoiding decode of unwanted frames.
    Attempts NVIDIA CUDA hardware-accelerated HEVC decoding and falls back
    to CPU if unavailable.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted PNG frames will be saved.
        target_fps: Desired extraction rate in frames per second.
        max_frames: Maximum number of frames to extract (default 80).

    Returns:
        Sorted list of paths to the extracted PNG frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    probe = _probe_video(video_path)
    source_fps = _get_fps(probe)
    duration = _get_duration(probe)

    # Clamp target_fps so we don't upsample
    effective_fps = min(target_fps, source_fps)
    estimated_frames = int(duration * effective_fps)
    if estimated_frames > max_frames and duration > 0:
        effective_fps = max_frames / duration
        logger.info(
            "Clamping FPS from %.2f to %.2f to stay within %d max frames",
            target_fps, effective_fps, max_frames,
        )

    # Compute the frame step for ffmpeg select filter (pick every Nth frame)
    frame_step = max(1, round(source_fps / effective_fps))

    frame_pattern = str(output_dir / "frame_%06d.png")

    # Build ffmpeg command with optional CUDA hwaccel and select filter
    hwaccel_flags = _hwaccel_input_flags()

    cmd = ["ffmpeg", "-y"]
    cmd.extend(hwaccel_flags)
    cmd.extend(["-i", str(video_path)])

    # Use select filter for efficient uniform sampling.  vsync=vfr ensures
    # only selected frames are written (avoids duplicated output frames).
    # Downscale to 1920px wide to avoid massive 8K PNGs (~30MB each).
    # DA3 processes at 504px anyway, so full 8K is wasted.
    select_expr = f"not(mod(n\\,{frame_step}))"
    vf_parts = [f"select='{select_expr}'"]

    # Downscale if wider than 1920px (keeps aspect ratio)
    if probe.get("streams"):
        for s in probe["streams"]:
            if s.get("codec_type") == "video":
                w = int(s.get("width", 0))
                if w > 1920:
                    vf_parts.append("scale=1920:-2")
                    logger.info("Downscaling %dx to 1920px wide during extraction", w)
                break

    cmd.extend([
        "-vf", ",".join(vf_parts),
        "-vsync", "vfr",
        "-vframes", str(max_frames),
        "-pix_fmt", "rgb24",
        frame_pattern,
    ])

    logger.info(
        "Extracting frames: source=%.1ffps, target=%.2ffps, duration=%.1fs, "
        "frame_step=%d, cuda=%s",
        source_fps, effective_fps, duration, frame_step,
        bool(hwaccel_flags),
    )

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        if hwaccel_flags:
            logger.warning(
                "CUDA-accelerated decode failed; retrying with CPU decoding"
            )
            # Rebuild without hwaccel
            cmd_cpu = ["ffmpeg", "-y", "-i", str(video_path)]
            cmd_cpu.extend([
                "-vf", f"select='{select_expr}'",
                "-vsync", "vfr",
                "-vframes", str(max_frames),
                "-pix_fmt", "rgb24",
                frame_pattern,
            ])
            subprocess.run(cmd_cpu, capture_output=True, text=True, check=True)
        else:
            raise

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    logger.info("Extracted %d frames to %s", len(frame_paths), output_dir)

    if frame_paths:
        _save_lens_metadata(frame_paths, video_path, output_dir)

    return frame_paths


@timed
def extract_frames_motion_based(
    video_path: str | Path,
    output_dir: str | Path,
    min_flow_threshold: float = 2.0,
    max_frames: int = 80,
    adaptive_threshold: bool = True,
    calibration_frames: int = 50,
    calibration_multiplier: float = 1.5,
    scene_change_prefill: bool = True,
    scene_threshold: float = 0.1,
    scene_change_only: bool = True,
) -> list[Path]:
    """Extract frames only when sufficient camera motion is detected.

    Ideal for face capture sessions where the camera orbits the subject.

    When *scene_change_only* is ``True`` (default), extraction uses only
    ffmpeg native scene-change detection, which is ~10x faster than
    Python optical flow because scene analysis runs during decode.  The
    optical-flow pass 2 is skipped entirely.

    When *scene_change_only* is ``False``, falls back to the legacy
    two-pass approach: ffmpeg scene-change keyframes first, then
    optical-flow gap-filling.

    When *adaptive_threshold* is ``True``, the first *calibration_frames*
    are used to compute optical flow statistics and the effective threshold
    is set to ``median_flow * calibration_multiplier``, adapting
    automatically to different capture speeds.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where selected PNG frames will be saved.
        min_flow_threshold: Minimum mean optical flow magnitude (pixels)
            required to keep a frame. Used as a fallback when adaptive
            calibration produces too few flow samples.
        max_frames: Maximum number of frames to save (default 80).
        adaptive_threshold: If ``True``, calibrate the threshold from the
            first *calibration_frames* frames.
        calibration_frames: Number of frames to sample for calibration.
        calibration_multiplier: Factor applied to median flow for threshold.
        scene_change_prefill: If ``True``, first extract scene-change
            keyframes via ffmpeg before optical-flow selection.
        scene_threshold: Scene change threshold for ffmpeg select filter
            (0.0-1.0). Lower values detect more changes.
        scene_change_only: If ``True`` (default), skip optical-flow pass
            entirely and rely only on ffmpeg scene detection for speed.

    Returns:
        Sorted list of paths to the saved PNG frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # --- Pass 1 (optional): scene-change keyframe extraction via ffmpeg ---
    scene_frames_saved = 0
    # When scene_change_only is True, skip the scene detection pass entirely
    # and go straight to uniform extraction. Scene detection is useless for
    # smooth face orbit videos (finds 0 scenes) and wastes a full video decode.
    if scene_change_only and scene_change_prefill:
        scene_change_prefill = False
        logger.info("Skipping scene-change pass (scene_change_only=True, using uniform extraction directly)")
    if scene_change_prefill:
        scene_dir = output_dir / "_scene_change_pass"
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_pattern = str(scene_dir / "scene_%06d.png")

        hwaccel_flags = _hwaccel_input_flags()
        cmd = ["ffmpeg", "-y"]
        cmd.extend(hwaccel_flags)
        # Downscale during scene extraction if video is wider than 1920px
        scene_vf = f"select='gt(scene\\,{scene_threshold})'"
        probe = _probe_video(video_path)
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video" and int(s.get("width", 0)) > 1920:
                scene_vf += ",scale=1920:-2"
                break
        cmd.extend([
            "-i", str(video_path),
            "-vf", scene_vf,
            "-vsync", "vfr",
            "-vframes", str(max_frames),
            "-pix_fmt", "rgb24",
            scene_pattern,
        ])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            if hwaccel_flags:
                cmd_cpu = ["ffmpeg", "-y", "-i", str(video_path)]
                cmd_cpu.extend([
                    "-vf", f"select='gt(scene\\,{scene_threshold})'",
                    "-vsync", "vfr",
                    "-vframes", str(max_frames),
                    "-pix_fmt", "rgb24",
                    scene_pattern,
                ])
                try:
                    subprocess.run(cmd_cpu, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as exc:
                    logger.warning("Scene-change prefill failed: %s", exc)

        # Move scene-change frames into the main output dir
        scene_files = sorted(scene_dir.glob("scene_*.png"))
        for sf in scene_files:
            dest = output_dir / f"frame_{scene_frames_saved:06d}.png"
            sf.replace(dest)
            scene_frames_saved += 1

        # Clean up temp dir
        try:
            scene_dir.rmdir()
        except OSError:
            pass

        logger.info(
            "Scene-change prefill: saved %d keyframes", scene_frames_saved,
        )

        if scene_frames_saved >= max_frames:
            saved_paths = sorted(output_dir.glob("frame_*.png"))
            if saved_paths:
                _save_lens_metadata(saved_paths, video_path, output_dir)
            return saved_paths

        # In scene_change_only mode, skip the expensive optical flow pass.
        # If scene detection yielded too few frames, supplement with uniform
        # sampling via ffmpeg select filter.
        if scene_change_only:
            if scene_frames_saved >= 10:
                logger.info(
                    "Scene-change only mode: %d frames extracted, skipping optical flow",
                    scene_frames_saved,
                )
                saved_paths = sorted(output_dir.glob("frame_*.png"))
                if saved_paths:
                    _save_lens_metadata(saved_paths, video_path, output_dir)
                return saved_paths
            else:
                # Too few scene changes — fall back to uniform FPS extraction
                logger.info(
                    "Scene-change only mode: only %d frames, supplementing with uniform extraction",
                    scene_frames_saved,
                )
                remaining_budget = max_frames - scene_frames_saved
                probe = _probe_video(video_path)
                source_fps = _get_fps(probe)
                duration = _get_duration(probe)
                supplement_fps = remaining_budget / max(duration, 1.0)
                supplement_fps = min(supplement_fps, source_fps)
                frame_step = max(1, round(source_fps / supplement_fps))

                supp_dir = output_dir / "_supplement_pass"
                supp_dir.mkdir(parents=True, exist_ok=True)
                supp_pattern = str(supp_dir / "supp_%06d.png")

                hwaccel_flags = _hwaccel_input_flags()
                cmd = ["ffmpeg", "-y"]
                cmd.extend(hwaccel_flags)
                # Downscale if 8K+ during supplement extraction
                supp_vf = f"select='not(mod(n\\,{frame_step}))'"
                for s in probe.get("streams", []):
                    if s.get("codec_type") == "video" and int(s.get("width", 0)) > 1920:
                        supp_vf += ",scale=1920:-2"
                        break
                cmd.extend([
                    "-i", str(video_path),
                    "-vf", supp_vf,
                    "-vsync", "vfr",
                    "-vframes", str(remaining_budget),
                    "-pix_fmt", "rgb24",
                    supp_pattern,
                ])
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError:
                    if hwaccel_flags:
                        cmd_cpu = ["ffmpeg", "-y", "-i", str(video_path)]
                        cmd_cpu.extend([
                            "-vf", supp_vf,
                            "-vsync", "vfr",
                            "-vframes", str(remaining_budget),
                            "-pix_fmt", "rgb24",
                            supp_pattern,
                        ])
                        subprocess.run(cmd_cpu, capture_output=True, text=True, check=True)

                supp_files = sorted(supp_dir.glob("supp_*.png"))
                for sf in supp_files:
                    dest = output_dir / f"frame_{scene_frames_saved:06d}.png"
                    sf.replace(dest)
                    scene_frames_saved += 1
                try:
                    supp_dir.rmdir()
                except OSError:
                    pass

                saved_paths = sorted(output_dir.glob("frame_*.png"))
                if saved_paths:
                    _save_lens_metadata(saved_paths, video_path, output_dir)
                logger.info(
                    "Scene-change + supplement: %d total frames extracted",
                    len(saved_paths),
                )
                return saved_paths

    # --- Pass 2: optical-flow based frame selection ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale dimensions for flow computation
    scale_factor = 0.5
    ret_peek, peek_frame = cap.read()
    if not ret_peek:
        cap.release()
        return sorted(output_dir.glob("frame_*.png"))
    h, w = peek_frame.shape[:2]
    small_size = (int(w * scale_factor), int(h * scale_factor))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Adaptive threshold calibration
    effective_threshold = min_flow_threshold
    if adaptive_threshold:
        calibrated, _ = _calibrate_motion_threshold(
            cap, small_size,
            calibration_frames=calibration_frames,
            multiplier=calibration_multiplier,
        )
        effective_threshold = calibrated
        logger.info(
            "Using adaptive motion threshold: %.3f (original: %.3f)",
            effective_threshold, min_flow_threshold,
        )

    logger.info(
        "Motion-based extraction: %d total frames, threshold=%.3f",
        total_frames_in_video, effective_threshold,
    )

    saved_paths: list[Path] = sorted(output_dir.glob("frame_*.png"))
    saved_count = len(saved_paths)  # account for scene-change prefill
    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0

    remaining = max_frames - saved_count
    if remaining <= 0:
        cap.release()
        if saved_paths:
            _save_lens_metadata(saved_paths, video_path, output_dir)
        return saved_paths

    # Always save the first frame if we haven't saved any yet
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return saved_paths

    # Determine if we need to downscale (8K → 1080p)
    save_w, save_h = w, h
    if w > 1920:
        save_w = 1920
        save_h = int(h * 1920 / w)
        save_h = save_h - (save_h % 2)  # ensure even
        logger.info("Downscaling frames from %dx%d to %dx%d during extraction", w, h, save_w, save_h)

    if saved_count == 0:
        first_path = output_dir / f"frame_{saved_count:06d}.png"
        save_frame = cv2.resize(first_frame, (save_w, save_h)) if save_w != w else first_frame
        cv2.imwrite(str(first_path), save_frame)
        saved_paths.append(first_path)
        saved_count += 1

    prev_gray = cv2.cvtColor(
        cv2.resize(first_frame, small_size), cv2.COLOR_BGR2GRAY
    )
    frame_idx += 1

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        curr_small = cv2.resize(frame, small_size)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        flow_mag = _compute_optical_flow_magnitude(prev_gray, curr_gray)

        if flow_mag >= effective_threshold:
            out_path = output_dir / f"frame_{saved_count:06d}.png"
            save_frame = cv2.resize(frame, (save_w, save_h)) if save_w != w else frame
            cv2.imwrite(str(out_path), save_frame)
            saved_paths.append(out_path)
            saved_count += 1
            prev_gray = curr_gray

            if saved_count % 50 == 0:
                logger.info("Saved %d / %d max frames", saved_count, max_frames)
        else:
            # Still update prev_gray periodically to avoid drift
            if frame_idx % 10 == 0:
                prev_gray = curr_gray

        frame_idx += 1

    cap.release()
    logger.info(
        "Motion-based extraction complete: %d frames saved from %d decoded",
        saved_count, frame_idx,
    )

    if saved_paths:
        _save_lens_metadata(saved_paths, video_path, output_dir)

    return saved_paths
