"""Quality-based frame filtering for 3D face reconstruction.

Evaluates blur, exposure, and face presence to reject frames that would
degrade COLMAP matching or downstream reconstruction quality.

Optimisations include downscaled face detection, face-region sharpness
evaluation, and minimum face-size enforcement.

Also provides pose-aware frame selection (``select_best_frames``) for
choosing a spatially diverse subset of frames after camera poses are
available (from COLMAP or DA3).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Guard mediapipe import - it may not be installed in all environments
_MEDIAPIPE_AVAILABLE = False
_mp_face_detection = None

try:
    import mediapipe as mp
    # MediaPipe >= 0.10.21 moved solutions to a separate import
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        _mp_face_detection = mp.solutions.face_detection
    else:
        from mediapipe.python.solutions import face_detection as _mp_face_detection
    _MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    logger.warning(
        "mediapipe face detection not available. Face filtering will be disabled. "
        "Install with: pip install mediapipe"
    )

# Default width for downscaled face detection (MediaPipe doesn't need full res)
_FACE_DETECT_WIDTH = 640

# Minimum fraction of the frame area the face bounding box must cover
_MIN_FACE_AREA_RATIO = 0.20


def _get_face_bbox(
    detection: object,
    frame_h: int,
    frame_w: int,
) -> tuple[int, int, int, int]:
    """Extract pixel-coordinate bounding box from a MediaPipe detection.

    Returns:
        (x, y, w, h) in pixel coordinates clamped to the frame dimensions.
    """
    bbox = detection.location_data.relative_bounding_box  # type: ignore[union-attr]
    x = max(0, int(bbox.xmin * frame_w))
    y = max(0, int(bbox.ymin * frame_h))
    w = min(int(bbox.width * frame_w), frame_w - x)
    h = min(int(bbox.height * frame_h), frame_h - y)
    return x, y, w, h


def check_blur(
    frame: np.ndarray,
    threshold: float = 100.0,
    face_bbox: Optional[tuple[int, int, int, int]] = None,
    half_res: bool = True,
) -> tuple[bool, float]:
    """Detect blur using the variance of the Laplacian.

    When *face_bbox* is provided the sharpness metric is computed only
    over the face region, which gives a much more relevant signal for
    face-reconstruction pipelines (the background may be intentionally
    blurred with shallow DOF).

    When *half_res* is ``True`` (default) and no face bbox is provided,
    the frame is downscaled 2x before computing the Laplacian.  Because
    Laplacian variance scales linearly with resolution, the threshold is
    halved internally for a consistent result with ~4x fewer pixels.

    Args:
        frame: BGR uint8 image (as loaded by OpenCV).
        threshold: Minimum Laplacian variance to consider the frame sharp.
            Typical values range from 50 (lenient) to 200 (strict).
        face_bbox: Optional (x, y, w, h) face bounding box in pixels.
            When given, blur is evaluated only within this region.
        half_res: If ``True``, compute blur at half resolution for speed
            (only when face_bbox is not provided).

    Returns:
        Tuple of (is_sharp, laplacian_variance). ``is_sharp`` is ``True``
        when the variance meets or exceeds *threshold*.  The returned
        variance is scaled back to full-resolution equivalent.
    """
    if face_bbox is not None:
        bx, by, bw, bh = face_bbox
        # Guard against degenerate boxes
        if bw > 10 and bh > 10:
            roi = frame[by:by + bh, bx:bx + bw]
        else:
            roi = frame
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return variance >= threshold, variance

    # No face bbox — optionally use half resolution for speed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if half_res and gray.shape[0] > 500 and gray.shape[1] > 500:
        h, w = gray.shape[:2]
        gray_half = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        variance_half = float(cv2.Laplacian(gray_half, cv2.CV_64F).var())
        # Scale variance back to approximate full-res equivalent
        variance = variance_half * 2.0
        return variance >= threshold, variance

    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return variance >= threshold, variance


def check_exposure(
    frame: np.ndarray,
    low: int = 30,
    high: int = 225,
    dark_ratio_limit: float = 0.5,
    bright_ratio_limit: float = 0.5,
) -> tuple[bool, dict[str, float]]:
    """Check for over- or under-exposure via histogram analysis.

    Computes the fraction of pixels falling below *low* (underexposed)
    and above *high* (overexposed). The frame is rejected if either
    fraction exceeds its respective limit.

    Args:
        frame: BGR uint8 image.
        low: Intensity below which a pixel is considered underexposed.
        high: Intensity above which a pixel is considered overexposed.
        dark_ratio_limit: Maximum tolerable fraction of dark pixels.
        bright_ratio_limit: Maximum tolerable fraction of bright pixels.

    Returns:
        Tuple of (exposure_ok, stats). ``exposure_ok`` is ``True`` when
        neither the dark nor bright pixel ratio exceeds its limit.
        ``stats`` contains ``mean_intensity``, ``dark_ratio``, and
        ``bright_ratio``.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    mean_intensity = float(gray.mean())

    dark_pixels = int(np.sum(gray < low))
    bright_pixels = int(np.sum(gray > high))

    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels

    stats = {
        "mean_intensity": round(mean_intensity, 2),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
    }

    exposure_ok = dark_ratio <= dark_ratio_limit and bright_ratio <= bright_ratio_limit
    return exposure_ok, stats


def check_face_present(
    frame: np.ndarray,
    min_confidence: float = 0.5,
    detector: object | None = None,
    min_face_area_ratio: float = _MIN_FACE_AREA_RATIO,
) -> tuple[bool, Optional[float], Optional[tuple[int, int, int, int]]]:
    """Detect whether a sufficiently large face is present.

    Uses MediaPipe Face Detection on a downscaled copy of the frame for
    efficiency.  After detection, enforces that the face bounding box
    covers at least *min_face_area_ratio* of the frame area --- faces
    that are too small (subject too far away) would not contribute
    meaningful geometry to face reconstruction.

    For batch usage, pass a pre-initialized ``FaceDetection`` instance
    via *detector* to avoid reinitializing the model on every call.

    Args:
        frame: BGR uint8 image.
        min_confidence: Minimum detection confidence in [0, 1]. Only used
            when *detector* is ``None`` (a temporary detector is created).
        detector: Optional pre-initialized ``mp.solutions.face_detection
            .FaceDetection`` instance.
        min_face_area_ratio: Minimum fraction of the frame area covered by
            the face bounding box.  Set to ``0.0`` to disable the check.

    Returns:
        Tuple of (face_found, confidence, face_bbox).
        ``face_found`` is ``True`` when a face of sufficient size is
        detected.  ``face_bbox`` is ``(x, y, w, h)`` in **original**
        frame pixel coordinates, or ``None``.
    """
    if not _MEDIAPIPE_AVAILABLE or _mp_face_detection is None:
        logger.debug("MediaPipe unavailable, skipping face detection")
        return True, None, None  # Permissive fallback

    orig_h, orig_w = frame.shape[:2]

    # Downscale for faster inference (MediaPipe doesn't need full res)
    if orig_w > _FACE_DETECT_WIDTH:
        scale = _FACE_DETECT_WIDTH / orig_w
        small_h = int(orig_h * scale)
        small = cv2.resize(frame, (_FACE_DETECT_WIDTH, small_h))
    else:
        small = frame
        scale = 1.0

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    if detector is not None:
        results = detector.process(rgb)
    else:
        with _mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence,
        ) as temp_detector:
            results = temp_detector.process(rgb)

    if results.detections:
        best = max(results.detections, key=lambda d: d.score[0])
        confidence = float(best.score[0])

        # Compute face bbox in original frame coordinates
        small_h_actual, small_w_actual = small.shape[:2]
        sx, sy, sw, sh = _get_face_bbox(best, small_h_actual, small_w_actual)

        # Scale back to original resolution
        face_bbox = (
            int(sx / scale),
            int(sy / scale),
            int(sw / scale),
            int(sh / scale),
        )

        # Enforce minimum face size
        face_area = face_bbox[2] * face_bbox[3]
        frame_area = orig_w * orig_h
        face_ratio = face_area / frame_area if frame_area > 0 else 0.0

        if face_ratio < min_face_area_ratio:
            logger.debug(
                "Face too small: %.1f%% of frame (need %.1f%%)",
                face_ratio * 100, min_face_area_ratio * 100,
            )
            return False, confidence, face_bbox

        return True, confidence, face_bbox

    return False, None, None


def filter_frames(
    frames_dir: str | Path,
    output_json: str | Path,
    blur_threshold: float = 100.0,
    exposure_low: int = 30,
    exposure_high: int = 225,
    require_face: bool = True,
    face_confidence: float = 0.5,
    min_face_area_ratio: float = _MIN_FACE_AREA_RATIO,
    quick_mode: bool = False,
    max_workers: int | None = None,
) -> list[Path]:
    """Filter a directory of frames by quality and write selection results.

    Each frame is evaluated for blur, exposure, and face presence. Frames
    that pass all active checks are included in the output selection.
    A JSON report is written with per-frame results and rejection reasons.

    Blur is evaluated over the detected face region (when available) so
    that shallow-DOF background blur does not cause false rejections.

    When *quick_mode* is ``True``, face detection is skipped entirely
    (regardless of *require_face*) and blur is computed at half resolution
    for speed.  This is useful as a fast pre-filter before DA3, with full
    face-aware filtering deferred to the pose-aware selection stage.

    Args:
        frames_dir: Directory containing input frames (PNG/JPG).
        output_json: Path to write the filter results JSON.
        blur_threshold: Laplacian variance threshold for blur detection.
        exposure_low: Pixel intensity below which counts as underexposed.
        exposure_high: Pixel intensity above which counts as overexposed.
        require_face: Whether to reject frames with no detected face.
        face_confidence: Minimum MediaPipe detection confidence.
        min_face_area_ratio: Minimum fraction of the frame the face
            bounding box must cover.
        quick_mode: If ``True``, skip face detection and use half-res
            blur checks for maximum speed.
        max_workers: Number of parallel worker processes for blur and
            exposure checks in quick_mode. Defaults to ``cpu_count - 1``.

    Returns:
        List of paths to frames that passed all quality checks.
    """
    frames_dir = Path(frames_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    frame_files = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    if not frame_files:
        logger.warning("No image files found in %s", frames_dir)
        report = {"selected": [], "rejected": [], "summary": {"total": 0}}
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return []

    effective_require_face = require_face and not quick_mode
    if quick_mode:
        logger.info(
            "Filtering %d frames from %s (QUICK MODE: blur+exposure only, half-res)",
            len(frame_files), frames_dir,
        )
    else:
        logger.info("Filtering %d frames from %s", len(frame_files), frames_dir)

    selected: list[Path] = []
    report_frames: list[dict] = []

    # --- Quick mode: parallel blur+exposure checks (no MediaPipe) ---
    if quick_mode:
        from src.utils.parallel import parallel_map

        def _check_frame_quick(frame_path: Path) -> dict:
            """Blur + exposure check for a single frame (process-safe)."""
            frame = cv2.imread(str(frame_path))
            if frame is None:
                return {
                    "file": frame_path.name,
                    "path": str(frame_path),
                    "selected": False,
                    "reasons": ["unreadable"],
                }
            reasons: list[str] = []
            details: dict[str, object] = {}

            is_sharp, lap_var = check_blur(
                frame, threshold=blur_threshold, face_bbox=None, half_res=True,
            )
            details["laplacian_variance"] = round(lap_var, 2)
            if not is_sharp:
                reasons.append(f"blur (variance={lap_var:.1f} < {blur_threshold})")

            exposure_ok, exp_stats = check_exposure(
                frame, low=exposure_low, high=exposure_high,
            )
            details["exposure"] = exp_stats
            if not exposure_ok:
                reasons.append(
                    f"exposure (dark={exp_stats['dark_ratio']:.2%}, "
                    f"bright={exp_stats['bright_ratio']:.2%})"
                )

            return {
                "file": frame_path.name,
                "path": str(frame_path),
                "selected": len(reasons) == 0,
                "reasons": reasons,
                "details": details,
            }

        report_frames = parallel_map(
            _check_frame_quick,
            frame_files,
            max_workers=max_workers,
            desc="Filtering (quick)",
            use_threads=False,
        )

        for entry, frame_path in zip(report_frames, frame_files):
            if entry is not None and entry.get("selected", False):
                selected.append(frame_path)

    else:
        # --- Full mode: sequential with MediaPipe face detection ---
        # Initialize a single MediaPipe detector for the entire batch to avoid
        # reinitializing the model on every frame.
        face_detector_ctx = None
        face_detector = None
        if effective_require_face and _MEDIAPIPE_AVAILABLE and _mp_face_detection is not None:
            face_detector_ctx = _mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=face_confidence,
            )
            face_detector = face_detector_ctx.__enter__()

        try:
            for i, frame_path in enumerate(frame_files):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    logger.warning("Cannot read frame, skipping: %s", frame_path)
                    report_frames.append({
                        "file": frame_path.name,
                        "path": str(frame_path),
                        "selected": False,
                        "reasons": ["unreadable"],
                    })
                    continue

                reasons: list[str] = []
                details: dict[str, object] = {}
                face_bbox: Optional[tuple[int, int, int, int]] = None

                # Face detection check (run first so we can use the bbox for blur)
                if effective_require_face:
                    face_found, conf, face_bbox = check_face_present(
                        frame,
                        min_confidence=face_confidence,
                        detector=face_detector,
                        min_face_area_ratio=min_face_area_ratio,
                    )
                    details["face_confidence"] = conf
                    if face_bbox is not None:
                        face_area = face_bbox[2] * face_bbox[3]
                        frame_area = frame.shape[1] * frame.shape[0]
                        details["face_area_ratio"] = round(
                            face_area / frame_area if frame_area > 0 else 0.0, 4,
                        )
                    if not face_found:
                        if conf is not None and face_bbox is not None:
                            reasons.append("face_too_small")
                        else:
                            reasons.append("no_face_detected")

                # Blur check (use face region when available for more relevant metric)
                is_sharp, lap_var = check_blur(
                    frame, threshold=blur_threshold, face_bbox=face_bbox,
                    half_res=False,
                )
                details["laplacian_variance"] = round(lap_var, 2)
                if not is_sharp:
                    reasons.append(f"blur (variance={lap_var:.1f} < {blur_threshold})")

                # Exposure check
                exposure_ok, exp_stats = check_exposure(
                    frame, low=exposure_low, high=exposure_high
                )
                details["exposure"] = exp_stats
                if not exposure_ok:
                    reasons.append(
                        f"exposure (dark={exp_stats['dark_ratio']:.2%}, "
                        f"bright={exp_stats['bright_ratio']:.2%})"
                    )

                is_selected = len(reasons) == 0

                report_frames.append({
                    "file": frame_path.name,
                    "path": str(frame_path),
                    "selected": is_selected,
                    "reasons": reasons if reasons else [],
                    "details": details,
                })

                if is_selected:
                    selected.append(frame_path)

                if (i + 1) % 50 == 0:
                    logger.info(
                        "Filtered %d / %d frames (%d selected so far)",
                        i + 1, len(frame_files), len(selected),
                    )
        finally:
            if face_detector_ctx is not None:
                face_detector_ctx.__exit__(None, None, None)

    # Build summary
    reason_counts: dict[str, int] = {}
    for entry in report_frames:
        for reason in entry["reasons"]:
            tag = reason.split("(")[0].strip().rstrip()
            reason_counts[tag] = reason_counts.get(tag, 0) + 1

    report = {
        "summary": {
            "total": len(frame_files),
            "selected": len(selected),
            "rejected": len(frame_files) - len(selected),
            "rejection_reasons": reason_counts,
            "settings": {
                "blur_threshold": blur_threshold,
                "exposure_low": exposure_low,
                "exposure_high": exposure_high,
                "require_face": require_face,
                "face_confidence": face_confidence,
                "min_face_area_ratio": min_face_area_ratio,
                "mediapipe_available": _MEDIAPIPE_AVAILABLE,
                "quick_mode": quick_mode,
            },
        },
        "frames": report_frames,
    }

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Frame filtering complete: %d/%d selected. Report: %s",
        len(selected), len(frame_files), output_json,
    )

    return selected


# ---------------------------------------------------------------------------
# Pose-aware frame selection (called after camera poses are available)
# ---------------------------------------------------------------------------

def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _load_poses_colmap(colmap_model_dir: Path) -> dict[str, np.ndarray]:
    """Load camera positions from a COLMAP sparse model.

    Returns:
        Dict mapping image filename -> camera centre (3,) in world coords.
    """
    images_bin = colmap_model_dir / "images.bin"
    if not images_bin.exists():
        return {}

    from utils.colmap_io import read_images_binary

    images = read_images_binary(images_bin)
    poses: dict[str, np.ndarray] = {}
    for img in images.values():
        R = _qvec_to_rotmat(img.qvec)
        # Camera centre in world coords: C = -R^T @ t
        centre = -R.T @ img.tvec
        poses[img.name] = centre
    return poses


def _load_poses_da3(poses_dir: Path) -> dict[str, np.ndarray]:
    """Load camera positions from DA3 pose .npz files.

    Each file is expected to contain a 'pose' key with a 4x4 camera-to-world
    matrix, or 'R' and 't' keys.

    Returns:
        Dict mapping image filename -> camera centre (3,) in world coords.
    """
    poses: dict[str, np.ndarray] = {}
    pose_files = sorted(poses_dir.glob("*_pose.npz"))
    for pf in pose_files:
        data = np.load(str(pf))
        # Derive image name: strip _pose suffix
        stem = pf.stem
        if stem.endswith("_pose"):
            stem = stem[:-5]
        # Try common key layouts
        if "pose" in data:
            # 4x4 camera-to-world
            c2w = data["pose"]
            centre = c2w[:3, 3]
        elif "R" in data and "t" in data:
            R = data["R"]
            t = data["t"]
            centre = -R.T @ t.ravel()
        else:
            continue
        # Match to common image extensions
        for ext in (".png", ".jpg", ".jpeg"):
            poses[stem + ext] = centre
    return poses


def _load_confidence_maps(depth_dir: Path) -> dict[str, float]:
    """Load mean confidence per frame from DA3 confidence maps.

    Returns:
        Dict mapping image filename -> mean confidence in [0, 1].
    """
    conf_scores: dict[str, float] = {}
    conf_files = sorted(depth_dir.glob("*_conf.npz"))
    if not conf_files:
        conf_files = sorted(depth_dir.glob("*_confidence.npz"))
    for cf in conf_files:
        data = np.load(str(cf))
        # Try common key names
        for key in ("confidence", "conf", "map"):
            if key in data:
                conf_scores[cf.stem.replace("_conf", "").replace("_confidence", "")] = float(
                    data[key].mean()
                )
                break
    return conf_scores


def select_best_frames(
    frames_dir: str | Path,
    poses_dir: str | Path,
    output_json: str | Path,
    target_count: int = 50,
    min_baseline: float = 0.05,
    colmap_model_dir: str | Path | None = None,
    depth_dir: str | Path | None = None,
) -> list[str]:
    """Select a spatially diverse subset of frames using farthest-point sampling.

    After camera poses are available (from COLMAP or DA3), this picks the
    best *target_count* frames by maximising view diversity: a greedy
    farthest-point sampling algorithm starts with the frame that has the
    most baseline coverage, then iteratively adds the frame whose minimum
    distance to all already-selected frames is largest.

    If DA3 confidence maps exist, frame distances are weighted by mean
    confidence (high-confidence views are preferred).

    Args:
        frames_dir: Directory containing input frames (used to enumerate
            available frame filenames).
        poses_dir: Directory that may contain DA3 ``*_pose.npz`` files.
            Also acts as the fallback when *colmap_model_dir* is ``None``.
        output_json: Path to write ``selected_training_frames.json``.
        target_count: Desired number of output frames.
        min_baseline: Minimum distance between any two selected camera
            centres (in scene units). Pairs closer than this are considered
            redundant.
        colmap_model_dir: Optional path to COLMAP sparse model directory
            (e.g. ``colmap/sparse/0``). Poses are loaded from here first;
            DA3 poses are used for any frames not in the COLMAP model.
        depth_dir: Optional directory containing DA3 confidence maps
            (``*_conf.npz``). When present, confidence is used as a
            tie-breaker.

    Returns:
        List of selected frame filenames (basenames, sorted).
    """
    frames_dir = Path(frames_dir)
    poses_dir = Path(poses_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Enumerate available frames
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    all_frame_names = sorted(
        f.name for f in frames_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    if not all_frame_names:
        logger.warning("No frames found in %s", frames_dir)
        output_json.write_text(json.dumps({"selected": [], "reason": "no_frames"}, indent=2))
        return []

    # --- Load poses from COLMAP and/or DA3 ---
    poses: dict[str, np.ndarray] = {}

    if colmap_model_dir is not None:
        colmap_model_dir = Path(colmap_model_dir)
        colmap_poses = _load_poses_colmap(colmap_model_dir)
        poses.update(colmap_poses)
        logger.info("Loaded %d poses from COLMAP", len(colmap_poses))

    da3_poses = _load_poses_da3(poses_dir)
    # DA3 poses fill in any frames not already covered by COLMAP
    for name, centre in da3_poses.items():
        if name not in poses:
            poses[name] = centre
    if da3_poses:
        logger.info("Loaded %d poses from DA3 (new: %d)",
                     len(da3_poses), len(da3_poses) - len(set(da3_poses) & set(poses)))

    # Keep only frames that have poses
    posed_frames = [f for f in all_frame_names if f in poses]
    if not posed_frames:
        logger.warning(
            "No camera poses found for any frames. Returning all %d frames.",
            len(all_frame_names),
        )
        result = all_frame_names
        output_json.write_text(json.dumps({
            "selected": result,
            "reason": "no_poses_available",
            "total_frames": len(all_frame_names),
        }, indent=2))
        return result

    logger.info("%d / %d frames have camera poses", len(posed_frames), len(all_frame_names))

    # If we already have fewer than target, return all posed frames
    if len(posed_frames) <= target_count:
        logger.info("Already at or below target (%d <= %d), selecting all posed frames",
                     len(posed_frames), target_count)
        output_json.write_text(json.dumps({
            "selected": posed_frames,
            "total_frames": len(all_frame_names),
            "posed_frames": len(posed_frames),
            "target_count": target_count,
            "strategy": "all_posed",
        }, indent=2))
        return posed_frames

    # --- Load confidence weights (optional) ---
    conf_weights: dict[str, float] = {}
    if depth_dir is not None:
        conf_weights = _load_confidence_maps(Path(depth_dir))
        if conf_weights:
            logger.info("Loaded confidence scores for %d frames", len(conf_weights))

    # --- Build position matrix ---
    n = len(posed_frames)
    positions = np.zeros((n, 3), dtype=np.float64)
    for i, name in enumerate(posed_frames):
        positions[i] = poses[name]

    # Confidence weight per frame (default 1.0)
    weights = np.ones(n, dtype=np.float64)
    for i, name in enumerate(posed_frames):
        stem = Path(name).stem
        if stem in conf_weights:
            weights[i] = max(0.1, conf_weights[stem])  # clamp to avoid zeroing out

    # --- Pairwise distance matrix ---
    # positions: (n, 3) -> diff: (n, n, 3) -> dist: (n, n)
    diff = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # --- Greedy farthest-point sampling ---
    # Seed: frame with largest sum of weighted baselines (most central diversity)
    weighted_coverage = (dist_matrix * weights[None, :]).sum(axis=1)
    seed_idx = int(np.argmax(weighted_coverage))

    selected_indices = [seed_idx]
    # min_dist_to_selected[i] = min distance from frame i to any selected frame
    min_dist_to_selected = dist_matrix[seed_idx].copy()
    # Apply confidence weighting: higher confidence -> effectively "closer" threshold
    # so high-confidence frames are preferred when distances are similar
    min_dist_weighted = min_dist_to_selected * weights

    remaining = set(range(n)) - {seed_idx}

    while len(selected_indices) < target_count and remaining:
        # Pick the frame with the largest weighted min-distance to selected set
        best_idx = -1
        best_dist = -1.0
        for idx in remaining:
            if min_dist_weighted[idx] > best_dist:
                best_dist = min_dist_weighted[idx]
                best_idx = idx

        if best_idx < 0:
            break

        # Skip if below minimum baseline (all remaining are too close)
        if min_dist_to_selected[best_idx] < min_baseline and len(selected_indices) >= 10:
            logger.info(
                "Stopping at %d frames: remaining frames are within min_baseline=%.4f",
                len(selected_indices), min_baseline,
            )
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

        # Update min distances
        new_dists = dist_matrix[best_idx]
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_dists)
        min_dist_weighted = min_dist_to_selected * weights

    selected_names = sorted([posed_frames[i] for i in selected_indices])

    # --- Write results ---
    result = {
        "selected": selected_names,
        "total_frames": len(all_frame_names),
        "posed_frames": len(posed_frames),
        "target_count": target_count,
        "actual_count": len(selected_names),
        "min_baseline": min_baseline,
        "strategy": "farthest_point_sampling",
        "confidence_weighted": bool(conf_weights),
    }
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info(
        "Selected %d / %d frames via farthest-point sampling (target=%d). Saved to %s",
        len(selected_names), len(posed_frames), target_count, output_json,
    )

    return selected_names
