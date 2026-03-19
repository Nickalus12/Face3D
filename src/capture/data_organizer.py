"""Multi-source data organizer for Face3D pipeline.

Classifies and validates multi-lens, multi-resolution, multi-modality capture
data (8K video, Expert RAW DNGs from multiple S25 Ultra lenses, Sensor Logger
exports) and produces a session manifest that downstream stages consume.

Supports:
- Samsung Galaxy S25 Ultra wide (23mm equiv, 6.3mm physical) video + photos
- 200MP main sensor Expert RAW photos (16320x12240)
- Sensor Logger ZIP archives (accelerometer, gyroscope, orientation, etc.)
"""

from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lens classification by 35mm-equivalent focal length
_LENS_CLASSIFICATION = {
    (10, 18): "ultrawide_13mm",
    (19, 28): "wide_23mm",
    (55, 85): "telephoto_70mm",
    (150, 250): "supertelephoto_200mm",
}

# Physical focal lengths (mm) for each lens class
_PHYSICAL_FOCAL = {
    "ultrawide_13mm": 1.95,
    "wide_23mm": 6.3,
    "telephoto_70mm": 18.6,
    "supertelephoto_200mm": 31.0,
}

# Sensor widths (mm) per lens class
_SENSOR_WIDTH_MM = {
    "ultrawide_13mm": 5.6,
    "wide_23mm": 8.0,
    "telephoto_70mm": 5.6,
    "supertelephoto_200mm": 5.6,
}

# Known Sensor Logger CSV filenames
_SENSOR_LOG_FILES = {
    "Accelerometer.csv": "accelerometer",
    "Gyroscope.csv": "gyroscope",
    "Orientation.csv": "orientation",
    "Magnetometer.csv": "magnetometer",
    "Barometer.csv": "barometer",
    "Location.csv": "gps",
    "Light.csv": "light",
}


def _classify_lens(focal_length_35mm: float | None) -> str:
    """Classify a 35mm-equivalent focal length into a known S25 Ultra lens.

    Returns lens name string or "unknown".
    """
    if focal_length_35mm is None:
        return "unknown"
    for (lo, hi), name in _LENS_CLASSIFICATION.items():
        if lo <= focal_length_35mm <= hi:
            return name
    return "unknown"


def _classify_lens_by_resolution(width: int, height: int) -> str | None:
    """Fallback lens classification using image resolution.

    The 200MP main sensor produces 16320x12240 (or rotated 12240x16320).
    The wide sensor in photo mode produces 5712x4284.
    """
    pixels = width * height
    if pixels > 150_000_000:  # >150MP -> 200MP main sensor
        return "main_200mp"
    if pixels > 20_000_000:  # >20MP -> wide lens photos
        return "wide_23mm"
    return None


def _extract_exif(photo_path: Path) -> dict[str, Any]:
    """Extract EXIF metadata from a photo file (JPG, DNG, or TIFF).

    Returns dict with focal_length, focal_length_35mm, datetime,
    width, height, make, model, orientation, iso, exposure_time.

    Uses exifread as primary reader (handles Samsung Expert RAW DNG),
    falls back to Pillow for formats exifread can't handle.
    """
    metadata: dict[str, Any] = {
        "width": 0, "height": 0,
        "focal_length": None, "focal_length_35mm": None,
        "datetime": None, "datetime_original": None,
        "exposure_time": None, "iso": None,
        "make": "", "model": "",
        "orientation": 1,
    }

    # ── Primary: exifread (handles DNG, TIFF, JPG without loading pixels) ──
    try:
        import exifread

        with open(photo_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        if tags:
            # Dimensions
            w = tags.get("Image ImageWidth") or tags.get("EXIF ExifImageWidth")
            h = tags.get("Image ImageLength") or tags.get("EXIF ExifImageLength")
            if w:
                metadata["width"] = int(str(w))
            if h:
                metadata["height"] = int(str(h))

            # Make / Model
            make = tags.get("Image Make")
            model = tags.get("Image Model")
            if make:
                metadata["make"] = str(make).strip()
            if model:
                metadata["model"] = str(model).strip()

            # Orientation
            orient = tags.get("Image Orientation")
            if orient:
                # exifread returns "Rotated 90 CW" etc. — map to int
                orient_str = str(orient)
                orient_map = {
                    "Horizontal (normal)": 1, "Mirrored horizontal": 2,
                    "Rotated 180": 3, "Mirrored vertical": 4,
                    "Mirrored horizontal then rotated 90 CCW": 5,
                    "Rotated 90 CW": 6, "Mirrored horizontal then rotated 90 CW": 7,
                    "Rotated 90 CCW": 8,
                }
                metadata["orientation"] = orient_map.get(orient_str, 1)

            # Focal length (raw mm)
            fl = tags.get("EXIF FocalLength")
            if fl:
                val = fl.values[0] if hasattr(fl, "values") else fl
                if hasattr(val, "num") and hasattr(val, "den"):
                    metadata["focal_length"] = float(val.num) / float(val.den) if val.den else None
                else:
                    try:
                        metadata["focal_length"] = float(str(val))
                    except (ValueError, TypeError):
                        pass

            # 35mm equivalent
            fl35 = tags.get("EXIF FocalLengthIn35mmFilm")
            if fl35:
                try:
                    metadata["focal_length_35mm"] = float(str(fl35))
                except (ValueError, TypeError):
                    pass

            # DateTime
            for tag_key, meta_key in [
                ("EXIF DateTimeOriginal", "datetime_original"),
                ("Image DateTime", "datetime"),
                ("EXIF DateTimeDigitized", "datetime"),
            ]:
                dt_tag = tags.get(tag_key)
                if dt_tag and metadata.get(meta_key) is None:
                    try:
                        metadata[meta_key] = datetime.strptime(
                            str(dt_tag), "%Y:%m:%d %H:%M:%S"
                        ).isoformat()
                    except (ValueError, TypeError):
                        pass

            # Exposure time
            et = tags.get("EXIF ExposureTime")
            if et:
                val = et.values[0] if hasattr(et, "values") else et
                if hasattr(val, "num") and hasattr(val, "den"):
                    metadata["exposure_time"] = float(val.num) / float(val.den) if val.den else None
                else:
                    try:
                        metadata["exposure_time"] = float(str(val))
                    except (ValueError, TypeError):
                        pass

            # ISO
            iso = tags.get("EXIF ISOSpeedRatings")
            if iso:
                try:
                    metadata["iso"] = int(str(iso))
                except (ValueError, TypeError):
                    pass

            # If we got valid data, return it
            if metadata["width"] > 0 and metadata["height"] > 0:
                return metadata

    except ImportError:
        pass  # exifread not installed, fall through to Pillow
    except Exception as e:
        logger.debug("exifread failed for %s: %s, trying Pillow", photo_path.name, e)

    # ── Fallback: Pillow (for formats exifread can't handle) ──
    try:
        import PIL.Image
        from PIL.ExifTags import TAGS

        PIL.Image.MAX_IMAGE_PIXELS = None
        img = PIL.Image.open(photo_path)
        metadata["width"] = img.width
        metadata["height"] = img.height

        try:
            exif_raw = img._getexif() or {}
        except Exception:
            exif_raw = {}
        img.close()

        exif = {}
        for tag_id, value in exif_raw.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            exif[tag_name] = value

        if not metadata["make"]:
            metadata["make"] = exif.get("Make", "")
        if not metadata["model"]:
            metadata["model"] = exif.get("Model", "")

        fl = exif.get("FocalLength")
        if fl is not None and metadata["focal_length"] is None:
            if hasattr(fl, "numerator"):
                metadata["focal_length"] = float(fl.numerator) / float(fl.denominator)
            else:
                metadata["focal_length"] = float(fl)

        fl35 = exif.get("FocalLengthIn35mmFilm")
        if fl35 is not None and metadata["focal_length_35mm"] is None:
            metadata["focal_length_35mm"] = float(fl35)

        for tag_key, meta_key in [
            ("DateTimeOriginal", "datetime_original"),
            ("DateTime", "datetime"),
        ]:
            dt_str = exif.get(tag_key)
            if dt_str and metadata.get(meta_key) is None:
                try:
                    metadata[meta_key] = datetime.strptime(
                        str(dt_str), "%Y:%m:%d %H:%M:%S"
                    ).isoformat()
                except (ValueError, TypeError):
                    pass

        return metadata

    except Exception as e:
        logger.warning("Failed to extract EXIF from %s: %s", photo_path, e)
        return metadata


def _probe_video(video_path: Path) -> dict[str, Any]:
    """Probe video file for duration, resolution, fps using ffprobe."""
    import subprocess

    info: dict[str, Any] = {
        "path": str(video_path),
        "duration": 0.0,
        "resolution": [0, 0],
        "fps": 0.0,
        "lens": "wide_23mm",  # S25 Ultra Pro Video always uses wide lens
    }

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)

            # Duration from format or stream
            fmt_dur = data.get("format", {}).get("duration")
            if fmt_dur:
                info["duration"] = float(fmt_dur)

            streams = data.get("streams", [])
            if streams:
                s = streams[0]
                info["resolution"] = [int(s.get("width", 0)), int(s.get("height", 0))]
                # Parse frame rate fraction
                fps_str = s.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    info["fps"] = float(num) / float(den)
                else:
                    info["fps"] = float(fps_str)

                if not info["duration"] and s.get("duration"):
                    info["duration"] = float(s["duration"])

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.warning("ffprobe failed for %s: %s", video_path, e)

    return info


def _analyze_sensor_zip(zip_path: Path) -> dict[str, Any]:
    """Analyze a Sensor Logger ZIP archive.

    Returns dict with available sensors, sample rates, time range, and flags.
    """
    result: dict[str, Any] = {
        "path": str(zip_path),
        "sensors": {},
        "time_range_ns": [0, 0],
        "has_gps": False,
        "has_light": False,
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            for csv_name, sensor_key in _SENSOR_LOG_FILES.items():
                # Sensor Logger may nest under a subfolder
                matching = [n for n in names if n.endswith(csv_name)]
                if not matching:
                    continue

                csv_entry = matching[0]
                with zf.open(csv_entry) as f:
                    lines = f.read().decode("utf-8", errors="replace").splitlines()

                if len(lines) < 2:
                    continue

                header = lines[0].strip().split(",")
                # Find time column (usually first column, named "time" or
                # "seconds_elapsed" or epoch nanoseconds)
                time_col_idx = 0
                for i, h in enumerate(header):
                    if h.strip().lower() in ("time", "epocn(ns)", "epoch (ns)"):
                        time_col_idx = i
                        break

                # Sample a few rows to compute rate and time range
                timestamps = []
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) > time_col_idx:
                        try:
                            timestamps.append(float(parts[time_col_idx]))
                        except ValueError:
                            pass

                if not timestamps:
                    continue

                timestamps_arr = np.array(timestamps)
                t_min = float(timestamps_arr[0])
                t_max = float(timestamps_arr[-1])

                # Compute sample rate
                n_samples = len(timestamps_arr)
                if n_samples > 1:
                    # If values are > 1e15, they're epoch nanoseconds
                    if t_min > 1e15:
                        dt_s = (t_max - t_min) / 1e9
                    elif t_min > 1e12:
                        dt_s = (t_max - t_min) / 1e6  # milliseconds
                    else:
                        dt_s = t_max - t_min  # seconds

                    hz = (n_samples - 1) / dt_s if dt_s > 0 else 0.0
                else:
                    hz = 0.0

                result["sensors"][sensor_key] = {
                    "file": csv_entry,
                    "samples": n_samples,
                    "hz": round(hz, 1),
                    "time_range": [t_min, t_max],
                }

                # Track global time range
                if result["time_range_ns"][0] == 0 or t_min < result["time_range_ns"][0]:
                    result["time_range_ns"][0] = t_min
                if t_max > result["time_range_ns"][1]:
                    result["time_range_ns"][1] = t_max

            result["has_gps"] = "gps" in result["sensors"]
            result["has_light"] = "light" in result["sensors"]

    except Exception as e:
        logger.error("Failed to analyze sensor ZIP %s: %s", zip_path, e)

    return result


def _match_sensor_to_video(
    sensor_analyses: list[dict],
    video_info: dict,
) -> tuple[str | None, str | None]:
    """Match sensor log ZIPs to video by timestamp overlap.

    Returns (video_synced_path, photo_session_path) or (None, None).
    """
    if not sensor_analyses:
        return None, None

    video_duration = video_info.get("duration", 0.0)
    if video_duration <= 0:
        # Can't match without video duration; return first available
        if len(sensor_analyses) >= 1:
            return sensor_analyses[0]["path"], (
                sensor_analyses[1]["path"] if len(sensor_analyses) > 1 else None
            )
        return None, None

    # Score each sensor log by how well its duration matches the video
    best_match = None
    best_score = -1.0

    for sa in sensor_analyses:
        # Check if it has gyroscope (most important for video sync)
        if "gyroscope" not in sa.get("sensors", {}):
            continue

        gyro = sa["sensors"]["gyroscope"]
        t_range = gyro.get("time_range", [0, 0])
        t_min, t_max = t_range

        # Compute sensor duration
        if t_min > 1e15:
            sensor_dur = (t_max - t_min) / 1e9
        elif t_min > 1e12:
            sensor_dur = (t_max - t_min) / 1e6
        else:
            sensor_dur = t_max - t_min

        # Score: how close is sensor duration to video duration?
        # Perfect match = 1.0, off by 2x = 0.5
        if sensor_dur > 0:
            ratio = min(sensor_dur, video_duration) / max(sensor_dur, video_duration)
            score = ratio
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_match = sa["path"]

    # The other sensor logs are for the photo session
    photo_session = None
    for sa in sensor_analyses:
        if sa["path"] != best_match:
            photo_session = sa["path"]
            break

    return best_match, photo_session


def organize_capture_session(
    video_paths: list[str | Path],
    photo_paths: list[str | Path],
    sensor_log_paths: list[str | Path],
    output_dir: str | Path,
) -> dict:
    """Organize and validate multi-source capture data.

    Classifies photos by lens, matches sensor logs to video, validates
    temporal overlap, and produces a ``session_manifest.json``.

    Args:
        video_paths: One or more video file paths.
        photo_paths: All photo file paths (JPG, DNG, PNG, TIFF).
        sensor_log_paths: Sensor Logger ZIP archive paths.
        output_dir: Directory to write the manifest and organized data.

    Returns:
        The manifest dict (also written to ``output_dir/session_manifest.json``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Video info ----
    video_info_list = []
    for vp in video_paths:
        vp = Path(vp)
        if not vp.exists():
            logger.warning("Video file not found: %s", vp)
            continue
        info = _probe_video(vp)
        video_info_list.append(info)
        logger.info(
            "Video: %s (%.1fs, %dx%d, %.1f fps)",
            vp.name, info["duration"],
            info["resolution"][0], info["resolution"][1], info["fps"],
        )

    primary_video = video_info_list[0] if video_info_list else {}

    # ---- Classify photos by lens ----
    photos_by_lens: dict[str, list[dict]] = {}
    photo_timestamps: list[float] = []

    # Group DNG/JPG pairs: same stem = one logical photo
    photo_groups: dict[str, dict[str, Path]] = {}
    for pp in photo_paths:
        pp = Path(pp)
        if not pp.exists():
            logger.warning("Photo not found: %s", pp)
            continue
        stem = pp.stem
        suffix = pp.suffix.lower()
        if stem not in photo_groups:
            photo_groups[stem] = {}
        if suffix == ".dng":
            photo_groups[stem]["dng"] = pp
        elif suffix in (".jpg", ".jpeg"):
            photo_groups[stem]["jpg"] = pp
        elif suffix in (".png", ".tiff", ".tif"):
            photo_groups[stem]["other"] = pp

    for stem, group in photo_groups.items():
        # Prefer JPG for EXIF (faster to read), DNG as raw source
        exif_source = group.get("jpg") or group.get("dng") or group.get("other")
        if exif_source is None:
            continue

        exif = _extract_exif(exif_source)

        # Classify lens
        lens = _classify_lens(exif.get("focal_length_35mm"))
        if lens == "unknown":
            # Fallback: classify by resolution
            res_lens = _classify_lens_by_resolution(exif["width"], exif["height"])
            if res_lens:
                lens = res_lens
            else:
                lens = "unknown"

        # Build photo entry
        entry: dict[str, Any] = {
            "stem": stem,
            "resolution": [exif["width"], exif["height"]],
        }
        if "jpg" in group:
            entry["jpg"] = str(group["jpg"])
        if "dng" in group:
            entry["dng"] = str(group["dng"])
        if "other" in group:
            entry["other"] = str(group["other"])

        entry["focal_length"] = exif.get("focal_length")
        entry["focal_length_35mm"] = exif.get("focal_length_35mm")
        entry["iso"] = exif.get("iso")
        entry["exposure_time"] = exif.get("exposure_time")
        entry["datetime"] = exif.get("datetime_original") or exif.get("datetime")

        if entry["datetime"]:
            try:
                dt = datetime.fromisoformat(entry["datetime"])
                photo_timestamps.append(dt.timestamp())
            except (ValueError, TypeError):
                pass

        photos_by_lens.setdefault(lens, []).append(entry)

    # Log classification results
    for lens_name, photos in photos_by_lens.items():
        resolutions = set()
        for p in photos:
            resolutions.add(f"{p['resolution'][0]}x{p['resolution'][1]}")
        logger.info(
            "Photos [%s]: %d files, resolutions: %s",
            lens_name, len(photos), ", ".join(sorted(resolutions)),
        )

    # ---- Analyze sensor logs ----
    sensor_analyses = []
    for sp in sensor_log_paths:
        sp = Path(sp)
        if not sp.exists():
            logger.warning("Sensor log not found: %s", sp)
            continue
        analysis = _analyze_sensor_zip(sp)
        sensor_analyses.append(analysis)
        sensor_names = list(analysis.get("sensors", {}).keys())
        logger.info(
            "Sensor log: %s (%d sensors: %s)",
            sp.name, len(sensor_names), ", ".join(sensor_names),
        )

    # Match sensor logs to video
    video_synced, photo_session_sensor = _match_sensor_to_video(
        sensor_analyses, primary_video,
    )

    # Build sensor summary
    sensors_manifest: dict[str, Any] = {
        "video_synced": video_synced,
        "photo_session": photo_session_sensor,
    }

    # Add sample rates from the video-synced sensor log
    for sa in sensor_analyses:
        if sa["path"] == video_synced:
            for sensor_key, sensor_info in sa.get("sensors", {}).items():
                sensors_manifest[f"{sensor_key}_hz"] = sensor_info["hz"]
            sensors_manifest["has_gps"] = sa.get("has_gps", False)
            sensors_manifest["has_light"] = sa.get("has_light", False)
            break

    # ---- Timestamp ranges ----
    timestamp_ranges: dict[str, Any] = {}

    # Video timestamp range (we don't have absolute epoch, use duration)
    if primary_video.get("duration", 0) > 0:
        timestamp_ranges["video_duration_s"] = primary_video["duration"]

    # Sensor timestamp range
    for sa in sensor_analyses:
        if sa["path"] == video_synced:
            timestamp_ranges["sensors"] = sa["time_range_ns"]
            break

    # Photo timestamp range
    if photo_timestamps:
        timestamp_ranges["photos"] = [
            min(photo_timestamps),
            max(photo_timestamps),
        ]

    # ---- Build manifest ----
    manifest: dict[str, Any] = {
        "video": primary_video,
        "additional_videos": video_info_list[1:] if len(video_info_list) > 1 else [],
        "photos": {
            lens: photos for lens, photos in photos_by_lens.items()
        },
        "sensors": sensors_manifest,
        "sensor_details": sensor_analyses,
        "timestamp_ranges": timestamp_ranges,
        "summary": {
            "num_videos": len(video_info_list),
            "num_photos_total": sum(len(v) for v in photos_by_lens.values()),
            "num_lenses": len(photos_by_lens),
            "lens_types": list(photos_by_lens.keys()),
            "num_sensor_logs": len(sensor_analyses),
        },
    }

    # Write manifest
    manifest_path = output_dir / "session_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Session manifest written to %s", manifest_path)

    return manifest


def compute_multi_lens_camera_models(manifest: dict) -> dict[str, dict[str, Any]]:
    """Create separate camera intrinsic models for each lens in the capture.

    Each lens gets a pinhole camera model with focal length in pixels,
    principal point at image center, and physical parameters.

    Args:
        manifest: Session manifest from ``organize_capture_session``.

    Returns:
        Dict keyed by lens name, each containing:
        - ``fx``, ``fy``: focal length in pixels
        - ``cx``, ``cy``: principal point
        - ``width``, ``height``: image resolution
        - ``focal_mm``: physical focal length
        - ``sensor_width_mm``: sensor width
        - ``model``: camera model string for COLMAP (``"PINHOLE"`` or ``"OPENCV"``)
    """
    from utils.camera import focal_length_pixels

    camera_models: dict[str, dict[str, Any]] = {}

    # Video camera model
    video = manifest.get("video", {})
    if video.get("resolution", [0, 0])[0] > 0:
        video_lens = video.get("lens", "wide_23mm")
        focal_mm = _PHYSICAL_FOCAL.get(video_lens, 6.3)
        sensor_w = _SENSOR_WIDTH_MM.get(video_lens, 8.0)
        w, h = video["resolution"]

        fx = focal_length_pixels(focal_mm, sensor_w, w)
        fy = fx  # Square pixels
        camera_models["video"] = {
            "fx": fx,
            "fy": fy,
            "cx": w / 2.0,
            "cy": h / 2.0,
            "width": w,
            "height": h,
            "focal_mm": focal_mm,
            "sensor_width_mm": sensor_w,
            "model": "OPENCV",
            "lens_name": video_lens,
        }
        logger.info(
            "Video camera: fx=%.1f px (%.1fmm on %.1fmm sensor, %dx%d)",
            fx, focal_mm, sensor_w, w, h,
        )

    # Photo camera models (one per lens group)
    photos = manifest.get("photos", {})
    for lens_name, photo_list in photos.items():
        if not photo_list:
            continue

        # Use first photo's resolution as representative
        rep = photo_list[0]
        w, h = rep["resolution"]
        if w == 0 or h == 0:
            continue

        # Determine physical focal length
        if lens_name in _PHYSICAL_FOCAL:
            focal_mm = _PHYSICAL_FOCAL[lens_name]
            sensor_w = _SENSOR_WIDTH_MM.get(lens_name, 8.0)
        elif lens_name == "main_200mp":
            # 200MP main sensor
            focal_mm = 6.3
            sensor_w = 8.0
        elif rep.get("focal_length"):
            focal_mm = rep["focal_length"]
            # Estimate sensor width from crop factor
            fl35 = rep.get("focal_length_35mm", focal_mm * 6.7)
            crop_factor = fl35 / focal_mm if focal_mm > 0 else 6.7
            sensor_w = 36.0 / crop_factor  # 36mm is full frame width
        else:
            logger.warning("Cannot determine focal length for lens %s, skipping", lens_name)
            continue

        fx = focal_length_pixels(focal_mm, sensor_w, w)
        fy = fx

        camera_models[f"photo_{lens_name}"] = {
            "fx": fx,
            "fy": fy,
            "cx": w / 2.0,
            "cy": h / 2.0,
            "width": w,
            "height": h,
            "focal_mm": focal_mm,
            "sensor_width_mm": sensor_w,
            "model": "OPENCV",
            "lens_name": lens_name,
        }
        logger.info(
            "Photo camera [%s]: fx=%.1f px (%.1fmm on %.1fmm sensor, %dx%d)",
            lens_name, fx, focal_mm, sensor_w, w, h,
        )

    return camera_models


def estimate_photo_timestamps(
    photo_paths: list[str | Path],
    video_start_time: str | datetime | None = None,
    sensor_timestamps: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Estimate each photo's timestamp relative to the video start.

    Uses EXIF DateTimeOriginal to compute the offset from video start.
    This enables associating photos with nearby video frames for pose
    estimation.

    Args:
        photo_paths: List of photo file paths.
        video_start_time: Video start time as ISO string or datetime.
            If None, returns absolute timestamps only.
        sensor_timestamps: Optional list of sensor epoch timestamps
            for cross-referencing.

    Returns:
        List of dicts with keys:
        - ``path``: photo file path
        - ``datetime_str``: EXIF datetime string
        - ``epoch_s``: absolute epoch seconds (float)
        - ``video_offset_s``: seconds from video start (None if no video start)
        - ``nearest_frame_idx``: estimated nearest video frame index (None if
          no fps info)
    """
    # Parse video start time
    video_start_epoch: float | None = None
    if video_start_time is not None:
        if isinstance(video_start_time, str):
            try:
                video_start_epoch = datetime.fromisoformat(video_start_time).timestamp()
            except ValueError:
                logger.warning("Cannot parse video_start_time: %s", video_start_time)
        elif isinstance(video_start_time, datetime):
            video_start_epoch = video_start_time.timestamp()

    results: list[dict[str, Any]] = []

    for pp in photo_paths:
        pp = Path(pp)
        exif = _extract_exif(pp)

        entry: dict[str, Any] = {
            "path": str(pp),
            "datetime_str": None,
            "epoch_s": None,
            "video_offset_s": None,
            "nearest_frame_idx": None,
        }

        dt_str = exif.get("datetime_original") or exif.get("datetime")
        if dt_str:
            entry["datetime_str"] = dt_str
            try:
                dt = datetime.fromisoformat(dt_str)
                entry["epoch_s"] = dt.timestamp()

                if video_start_epoch is not None:
                    offset = dt.timestamp() - video_start_epoch
                    entry["video_offset_s"] = round(offset, 3)
            except (ValueError, TypeError):
                pass

        results.append(entry)

    # Sort by timestamp
    results.sort(key=lambda x: x.get("epoch_s") or 0)

    # Log summary
    with_ts = sum(1 for r in results if r["epoch_s"] is not None)
    logger.info(
        "Photo timestamps: %d/%d have EXIF datetime", with_ts, len(results),
    )
    if video_start_epoch and with_ts > 0:
        offsets = [r["video_offset_s"] for r in results if r["video_offset_s"] is not None]
        if offsets:
            logger.info(
                "Photo offsets from video start: %.1fs to %.1fs",
                min(offsets), max(offsets),
            )

    return results
