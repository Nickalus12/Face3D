"""Data validation contracts for Face3D pipeline stage-to-stage data.

Each validator checks a specific data structure produced by a pipeline stage
and returns a ValidationResult with pass/fail status and detailed diagnostics.

Validators are lightweight — they use only numpy (no GPU, no torch required)
and can be called at stage boundaries to catch data corruption early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frame validation (Stage 1-3 outputs)
# ---------------------------------------------------------------------------

_VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def validate_frames(
    frames_dir: Path | str,
    *,
    min_frames: int = 1,
    expected_extensions: set[str] | None = None,
    min_width: int = 64,
    min_height: int = 64,
) -> ValidationResult:
    """Validate that a frames directory contains usable image files.

    Checks:
        - Directory exists and contains image files
        - At least ``min_frames`` images present
        - Each image is readable and meets minimum dimensions

    Args:
        frames_dir: Path to directory containing frame images.
        min_frames: Minimum number of frames required.
        expected_extensions: Allowed file extensions (default: common image formats).
        min_width: Minimum image width in pixels.
        min_height: Minimum image height in pixels.
    """
    import cv2

    frames_dir = Path(frames_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not frames_dir.exists():
        return ValidationResult(valid=False, errors=[f"Frames directory does not exist: {frames_dir}"])

    if not frames_dir.is_dir():
        return ValidationResult(valid=False, errors=[f"Path is not a directory: {frames_dir}"])

    exts = expected_extensions or _VALID_IMAGE_EXTENSIONS
    image_files = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    if len(image_files) < min_frames:
        errors.append(f"Found {len(image_files)} frames, need at least {min_frames}")

    if len(image_files) == 0:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Spot-check first, middle, and last frames
    check_indices = {0, len(image_files) // 2, len(image_files) - 1}
    for idx in sorted(check_indices):
        path = image_files[idx]
        img = cv2.imread(str(path))
        if img is None:
            errors.append(f"Unreadable image: {path.name}")
            continue
        h, w = img.shape[:2]
        if w < min_width or h < min_height:
            errors.append(f"{path.name}: dimensions {w}x{h} below minimum {min_width}x{min_height}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Depth map validation (Stage 6-8 outputs)
# ---------------------------------------------------------------------------

def validate_depth_maps(
    depths: np.ndarray | list[np.ndarray],
) -> ValidationResult:
    """Validate depth map arrays.

    Checks per map:
        - 2D array
        - float32 dtype
        - All values positive and finite (no NaN/Inf)

    Args:
        depths: Single depth map (H, W) or list of depth maps.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if isinstance(depths, np.ndarray) and depths.ndim == 2:
        depths = [depths]
    elif isinstance(depths, np.ndarray) and depths.ndim == 3:
        depths = [depths[i] for i in range(depths.shape[0])]

    if len(depths) == 0:
        return ValidationResult(valid=False, errors=["No depth maps provided"])

    for i, d in enumerate(depths):
        prefix = f"depth[{i}]"
        if not isinstance(d, np.ndarray):
            errors.append(f"{prefix}: not a numpy array")
            continue
        if d.ndim != 2:
            errors.append(f"{prefix}: expected 2D, got {d.ndim}D shape {d.shape}")
            continue
        if not np.issubdtype(d.dtype, np.floating):
            errors.append(f"{prefix}: expected float dtype, got {d.dtype}")
        if np.any(np.isnan(d)):
            nan_count = int(np.isnan(d).sum())
            errors.append(f"{prefix}: contains {nan_count} NaN values")
        if np.any(np.isinf(d)):
            inf_count = int(np.isinf(d).sum())
            errors.append(f"{prefix}: contains {inf_count} Inf values")
        if np.any(d < 0):
            neg_count = int((d < 0).sum())
            errors.append(f"{prefix}: contains {neg_count} negative values")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Quaternion validation (orientation / COLMAP poses)
# ---------------------------------------------------------------------------

def validate_quaternions(
    quats: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate unit quaternions (w, x, y, z convention).

    Checks:
        - Shape is (N, 4)
        - All values finite
        - Unit norm within ``tolerance``

    Args:
        quats: (N, 4) quaternion array.
        tolerance: Allowed deviation from unit norm.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(quats, np.ndarray):
        return ValidationResult(valid=False, errors=["Quaternions must be a numpy array"])

    if quats.ndim != 2 or quats.shape[1] != 4:
        errors.append(f"Expected shape (N, 4), got {quats.shape}")
        return ValidationResult(valid=False, errors=errors)

    if quats.shape[0] == 0:
        errors.append("Empty quaternion array")
        return ValidationResult(valid=False, errors=errors)

    if not np.all(np.isfinite(quats)):
        non_finite = int(np.sum(~np.isfinite(quats)))
        errors.append(f"Contains {non_finite} non-finite values")

    norms = np.linalg.norm(quats, axis=1)
    deviations = np.abs(norms - 1.0)
    max_dev = float(deviations.max())
    if max_dev > tolerance:
        bad_count = int(np.sum(deviations > tolerance))
        errors.append(
            f"{bad_count}/{len(quats)} quaternions deviate from unit norm "
            f"(max deviation: {max_dev:.2e}, tolerance: {tolerance:.0e})"
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# COLMAP model validation (Stage 6 output)
# ---------------------------------------------------------------------------

def validate_colmap_model(
    cameras: dict,
    images: dict,
    points3D: dict,
) -> ValidationResult:
    """Validate a COLMAP sparse model (cameras, images, points3D dicts).

    Checks:
        - At least 1 camera, 1 image
        - Camera intrinsic params are positive and finite
        - Image quaternions are unit-length and finite
        - Image camera_id references a valid camera
        - 3D point positions are finite

    Args:
        cameras: Dict mapping camera_id -> camera object (with .params, .width, .height).
        images: Dict mapping image_id -> image object (with .qvec, .tvec, .camera_id).
        points3D: Dict mapping point_id -> point object (with .xyz).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if len(cameras) == 0:
        errors.append("No cameras in model")
    if len(images) == 0:
        errors.append("No images in model")

    # Validate cameras
    for cam_id, cam in cameras.items():
        params = np.asarray(cam.params)
        if not np.all(np.isfinite(params)):
            errors.append(f"Camera {cam_id}: non-finite intrinsic params")
        if np.any(params[:2] <= 0):
            errors.append(f"Camera {cam_id}: focal length <= 0")

    # Validate images
    camera_ids = set(cameras.keys())
    for img_id, img in images.items():
        qvec = np.asarray(img.qvec)
        tvec = np.asarray(img.tvec)

        if not np.all(np.isfinite(qvec)):
            errors.append(f"Image {img_id}: non-finite quaternion")
        elif abs(np.linalg.norm(qvec) - 1.0) > 1e-4:
            errors.append(
                f"Image {img_id}: quaternion norm = {np.linalg.norm(qvec):.6f} (not unit)"
            )

        if not np.all(np.isfinite(tvec)):
            errors.append(f"Image {img_id}: non-finite translation")

        if img.camera_id not in camera_ids:
            errors.append(f"Image {img_id}: references missing camera {img.camera_id}")

    # Validate 3D points
    if len(points3D) == 0 and len(images) > 0:
        warnings.append("No 3D points in model (may be valid for pose-only reconstruction)")

    non_finite_pts = 0
    for pt_id, pt in points3D.items():
        if not np.all(np.isfinite(pt.xyz)):
            non_finite_pts += 1
    if non_finite_pts > 0:
        errors.append(f"{non_finite_pts} 3D points have non-finite coordinates")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# FLAME parameter validation (Stage 10 output)
# ---------------------------------------------------------------------------

def validate_flame_params(
    shape: np.ndarray,
    expression: np.ndarray,
    pose: np.ndarray,
    vertices: np.ndarray,
    *,
    shape_bound: float = 10.0,
    expression_bound: float = 10.0,
    expected_vertices: int = 5023,
) -> ValidationResult:
    """Validate FLAME model parameters and output vertices.

    Checks:
        - All arrays are finite
        - Shape/expression coefficients within reasonable bounds
        - Vertices shape is (N, 3) with N == ``expected_vertices``

    Args:
        shape: Shape coefficients, typically (1, K) or (K,).
        expression: Expression coefficients, typically (1, K) or (K,).
        pose: Pose parameters (jaw, global rotation), typically (1, P) or (P,).
        vertices: Output vertices, expected (expected_vertices, 3) or (1, expected_vertices, 3).
        shape_bound: Max absolute value for shape coefficients.
        expression_bound: Max absolute value for expression coefficients.
        expected_vertices: Expected vertex count (5023 for FLAME).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Flatten batch dim if present
    _shape = np.asarray(shape).squeeze()
    _expr = np.asarray(expression).squeeze()
    _pose = np.asarray(pose).squeeze()
    _verts = np.asarray(vertices)
    if _verts.ndim == 3:
        _verts = _verts.squeeze(0)

    # Finiteness
    if not np.all(np.isfinite(_shape)):
        errors.append("Shape params contain NaN/Inf")
    if not np.all(np.isfinite(_expr)):
        errors.append("Expression params contain NaN/Inf")
    if not np.all(np.isfinite(_pose)):
        errors.append("Pose params contain NaN/Inf")
    if not np.all(np.isfinite(_verts)):
        errors.append("Vertices contain NaN/Inf")

    # Bounds
    if np.all(np.isfinite(_shape)) and np.max(np.abs(_shape)) > shape_bound:
        warnings.append(
            f"Shape coefficients max |value| = {np.max(np.abs(_shape)):.2f} "
            f"exceeds typical bound {shape_bound}"
        )
    if np.all(np.isfinite(_expr)) and np.max(np.abs(_expr)) > expression_bound:
        warnings.append(
            f"Expression coefficients max |value| = {np.max(np.abs(_expr)):.2f} "
            f"exceeds typical bound {expression_bound}"
        )

    # Vertex shape
    if _verts.ndim != 2 or _verts.shape[1] != 3:
        errors.append(f"Vertices expected shape (N, 3), got {_verts.shape}")
    elif _verts.shape[0] != expected_vertices:
        errors.append(
            f"Expected {expected_vertices} vertices, got {_verts.shape[0]}"
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Gaussian splat validation (Stage 12-13 output)
# ---------------------------------------------------------------------------

def validate_gaussians(
    means: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    quats: np.ndarray,
) -> ValidationResult:
    """Validate Gaussian splat parameters.

    Checks:
        - Positions (means): shape (N, 3), all finite
        - Scales: shape (N, 3), all finite (log-space, so any value is ok)
        - Opacities: shape (N, 1) or (N,), all finite
        - Quaternions: shape (N, 4), unit norm, finite

    Args:
        means: (N, 3) Gaussian center positions.
        scales: (N, 3) log-scale values.
        opacities: (N, 1) or (N,) logit-space opacities.
        quats: (N, 4) rotation quaternions (w, x, y, z).
    """
    errors: list[str] = []
    warnings: list[str] = []

    means = np.asarray(means)
    scales = np.asarray(scales)
    opacities = np.asarray(opacities).reshape(-1)
    quats = np.asarray(quats)

    n = means.shape[0] if means.ndim >= 1 else 0

    # Means
    if means.ndim != 2 or means.shape[1] != 3:
        errors.append(f"Means expected shape (N, 3), got {means.shape}")
    elif not np.all(np.isfinite(means)):
        bad = int(np.sum(~np.isfinite(means).all(axis=1)))
        errors.append(f"Means: {bad}/{n} positions contain NaN/Inf")

    # Scales
    if scales.ndim != 2 or scales.shape[1] != 3:
        errors.append(f"Scales expected shape (N, 3), got {scales.shape}")
    elif not np.all(np.isfinite(scales)):
        bad = int(np.sum(~np.isfinite(scales).all(axis=1)))
        errors.append(f"Scales: {bad}/{n} values contain NaN/Inf")

    # Opacities
    if not np.all(np.isfinite(opacities)):
        bad = int(np.sum(~np.isfinite(opacities)))
        errors.append(f"Opacities: {bad}/{n} values contain NaN/Inf")

    # Quaternions
    if quats.ndim != 2 or quats.shape[1] != 4:
        errors.append(f"Quaternions expected shape (N, 4), got {quats.shape}")
    else:
        if not np.all(np.isfinite(quats)):
            bad = int(np.sum(~np.isfinite(quats).all(axis=1)))
            errors.append(f"Quaternions: {bad}/{n} contain NaN/Inf")
        else:
            norms = np.linalg.norm(quats, axis=1)
            devs = np.abs(norms - 1.0)
            if devs.max() > 0.01:
                bad = int(np.sum(devs > 0.01))
                errors.append(
                    f"Quaternions: {bad}/{n} not unit-norm "
                    f"(max deviation: {devs.max():.4f})"
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Sensor data validation (Stage 4-5 output)
# ---------------------------------------------------------------------------

def validate_sensor_data(
    data_dict: dict[str, np.ndarray],
) -> ValidationResult:
    """Validate parsed sensor data from Sensor Logger.

    Checks:
        - Timestamps are monotonically non-decreasing
        - Sampling rate is consistent (no gaps > 10x median interval)
        - Sensor values are finite
        - Quaternion columns (if present) are unit-norm

    Args:
        data_dict: Dict with keys like 'accel_timestamps', 'accel_xyz',
            'orientation_timestamps', 'orientation_quats', etc.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not data_dict:
        return ValidationResult(valid=False, errors=["Empty sensor data dict"])

    # Check each timestamp array
    timestamp_keys = [k for k in data_dict if k.endswith("_timestamps")]
    if not timestamp_keys:
        # Also accept 'timestamps' directly
        if "timestamps" in data_dict:
            timestamp_keys = ["timestamps"]

    for ts_key in timestamp_keys:
        ts = np.asarray(data_dict[ts_key])
        ts_key.replace("_timestamps", "")

        if ts.ndim != 1:
            errors.append(f"{ts_key}: expected 1D, got {ts.ndim}D")
            continue

        if len(ts) < 2:
            warnings.append(f"{ts_key}: only {len(ts)} sample(s)")
            continue

        if not np.all(np.isfinite(ts)):
            errors.append(f"{ts_key}: contains non-finite timestamps")
            continue

        # Monotonicity
        diffs = np.diff(ts)
        if np.any(diffs < 0):
            bad_count = int(np.sum(diffs < 0))
            errors.append(f"{ts_key}: {bad_count} non-monotonic timestamp steps")

        # Sampling rate consistency
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) > 2:
            median_dt = float(np.median(positive_diffs))
            if median_dt > 0:
                max_gap = float(positive_diffs.max())
                if max_gap > 10.0 * median_dt:
                    warnings.append(
                        f"{ts_key}: max gap {max_gap:.4f}s is >{10}x median interval "
                        f"{median_dt:.4f}s (possible dropout)"
                    )

    # Check XYZ sensor arrays for finiteness
    xyz_keys = [k for k in data_dict if k.endswith("_xyz")]
    for key in xyz_keys:
        arr = np.asarray(data_dict[key])
        if not np.all(np.isfinite(arr)):
            nan_count = int(np.sum(np.isnan(arr)))
            inf_count = int(np.sum(np.isinf(arr)))
            errors.append(f"{key}: {nan_count} NaN, {inf_count} Inf values")

    # Check quaternion arrays
    quat_keys = [k for k in data_dict if k.endswith("_quats")]
    for key in quat_keys:
        arr = np.asarray(data_dict[key])
        if arr.ndim != 2 or arr.shape[1] != 4:
            errors.append(f"{key}: expected shape (N, 4), got {arr.shape}")
            continue
        if not np.all(np.isfinite(arr)):
            errors.append(f"{key}: contains non-finite quaternion values")
        else:
            norms = np.linalg.norm(arr, axis=1)
            devs = np.abs(norms - 1.0)
            if devs.max() > 1e-3:
                bad = int(np.sum(devs > 1e-3))
                warnings.append(
                    f"{key}: {bad}/{len(arr)} quaternions deviate from unit norm "
                    f"(max: {devs.max():.4e})"
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
