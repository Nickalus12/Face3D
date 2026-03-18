"""Align monocular relative depth maps to metric scale using COLMAP sparse points."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from utils.colmap_io import (
    Camera,
    Image,
    Point3D,
    get_intrinsics_matrix,
    qvec_to_rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

logger = logging.getLogger(__name__)

# Reasonable depth clamp range for face capture at arm's length (meters)
_DEPTH_MIN_DEFAULT = 0.1
_DEPTH_MAX_DEFAULT = 10.0


def _ransac_scale_shift(
    mono_depths: np.ndarray,
    metric_depths: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_iterations: int = 1000,
    inlier_threshold: float = 0.05,
    min_samples: int = 3,
    rng_seed: Optional[int] = 42,
) -> Tuple[float, float, np.ndarray]:
    """RANSAC-based robust estimation of scale and shift.

    Fits ``s * mono + b = metric`` using random subsets and keeps the
    model with the most inliers. This is far more robust than plain
    least-squares when outliers exist (reflective surfaces, hair, etc.).

    Args:
        mono_depths: (N,) monocular depth samples.
        metric_depths: (N,) corresponding metric (COLMAP) depth samples.
        weights: (N,) optional per-point confidence weights (e.g. from DA3).
        n_iterations: Number of RANSAC iterations.
        inlier_threshold: Absolute residual threshold to count as inlier.
        min_samples: Points sampled per iteration (must be >= 2).
        rng_seed: Random seed for reproducibility.

    Returns:
        (scale, shift, inlier_mask) where inlier_mask is boolean (N,).
    """
    rng = np.random.default_rng(rng_seed)
    n = len(mono_depths)
    min_samples = max(min_samples, 2)

    best_s, best_b = 1.0, 0.0
    best_inlier_count = 0
    best_inlier_mask = np.zeros(n, dtype=bool)

    for _ in range(n_iterations):
        idx = rng.choice(n, size=min_samples, replace=False)
        A_sub = np.column_stack([mono_depths[idx], np.ones(min_samples)])
        b_sub = metric_depths[idx]

        # Optionally apply weights to the subset
        if weights is not None:
            W_sub = np.diag(weights[idx])
            result = np.linalg.lstsq(W_sub @ A_sub, W_sub @ b_sub, rcond=None)
        else:
            result = np.linalg.lstsq(A_sub, b_sub, rcond=None)

        s_cand, b_cand = result[0]

        # Count inliers on the full set
        residuals = np.abs(s_cand * mono_depths + b_cand - metric_depths)
        inlier_mask = residuals < inlier_threshold
        inlier_count = inlier_mask.sum()

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_s, best_b = s_cand, b_cand
            best_inlier_mask = inlier_mask

    # Refit on all inliers for final precision
    if best_inlier_count >= 2:
        A_in = np.column_stack([
            mono_depths[best_inlier_mask],
            np.ones(best_inlier_count),
        ])
        b_in = metric_depths[best_inlier_mask]
        if weights is not None:
            W_in = np.diag(weights[best_inlier_mask])
            result = np.linalg.lstsq(W_in @ A_in, W_in @ b_in, rcond=None)
        else:
            result = np.linalg.lstsq(A_in, b_in, rcond=None)
        best_s, best_b = result[0]

    return best_s, best_b, best_inlier_mask


def align_depth_to_colmap(
    depth_map: np.ndarray,
    colmap_points_3d: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: Tuple[np.ndarray, np.ndarray],
    confidence_map: Optional[np.ndarray] = None,
    use_ransac: bool = True,
    ransac_iterations: int = 1000,
    ransac_inlier_threshold: float = 0.05,
    depth_clamp_min: float = _DEPTH_MIN_DEFAULT,
    depth_clamp_max: float = _DEPTH_MAX_DEFAULT,
) -> np.ndarray:
    """Align a relative depth map to metric scale using COLMAP sparse points.

    Solves for scale *s* and shift *b* that minimize::

        ||s * depth_mono + b - depth_colmap||^2

    When *use_ransac* is True (default), a RANSAC estimator is used for
    robustness against outlier correspondences (reflective surfaces,
    hair, etc.). When a *confidence_map* is provided (e.g. from DA3),
    high-confidence points receive greater weight in the fit.

    The final aligned depth is clamped to [depth_clamp_min, depth_clamp_max]
    to remove physically implausible values for face-distance capture.

    Args:
        depth_map: (H, W) relative (inverse) depth from monocular estimator.
        colmap_points_3d: (N, 3) COLMAP sparse 3D points visible in this frame.
        camera_intrinsics: (3, 3) intrinsics matrix K.
        camera_extrinsics: Tuple of (R, t) where R is (3,3) rotation and
            t is (3,) translation.
        confidence_map: Optional (H, W) confidence from DA3, values in [0, 1].
            Used as per-point weights in the alignment.
        use_ransac: Use RANSAC for robust scale/shift fitting.
        ransac_iterations: Number of RANSAC iterations.
        ransac_inlier_threshold: RANSAC inlier residual threshold (meters).
        depth_clamp_min: Minimum valid depth after alignment (meters).
        depth_clamp_max: Maximum valid depth after alignment (meters).

    Returns:
        (H, W) aligned metric depth map, clamped to valid range.
    """
    R, t = camera_extrinsics
    t = t.ravel()

    # Project COLMAP 3D points into camera frame to get metric depth
    points_cam = (R @ colmap_points_3d.T).T + t  # (N, 3)
    metric_depths = points_cam[:, 2]  # z-component is depth

    # Filter points with positive depth
    valid = metric_depths > 0
    points_cam = points_cam[valid]
    metric_depths = metric_depths[valid]
    colmap_points_3d = colmap_points_3d[valid]

    if len(metric_depths) < 3:
        logger.warning(
            "Only %d valid COLMAP points for alignment (need >= 3), "
            "returning unscaled depth",
            len(metric_depths),
        )
        return depth_map.copy()

    # Project to pixel coordinates to sample monocular depth
    K = camera_intrinsics
    points_proj = (K @ points_cam.T).T
    pixels = points_proj[:, :2] / points_proj[:, 2:3]

    H, W = depth_map.shape[:2]
    px = np.round(pixels[:, 0]).astype(int)
    py = np.round(pixels[:, 1]).astype(int)

    # Keep only points that fall within image bounds
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px = px[in_bounds]
    py = py[in_bounds]
    metric_depths = metric_depths[in_bounds]

    if len(metric_depths) < 3:
        logger.warning("Too few in-bounds points for alignment (%d)", len(metric_depths))
        return depth_map.copy()

    # Sample monocular depth at sparse point locations
    mono_depths = depth_map[py, px]

    # Filter out zero/invalid monocular depth values
    valid_mono = mono_depths > 0
    mono_depths = mono_depths[valid_mono]
    metric_depths = metric_depths[valid_mono]
    px = px[valid_mono]
    py = py[valid_mono]

    if len(metric_depths) < 3:
        logger.warning("Too few valid mono-depth samples for alignment (%d)", len(metric_depths))
        return depth_map.copy()

    # Build per-point weights from confidence map if available
    weights = None
    if confidence_map is not None:
        if confidence_map.shape[:2] == (H, W):
            weights = confidence_map[py, px].astype(np.float64)
            # Avoid zero weights -- floor at a small epsilon
            weights = np.maximum(weights, 1e-6)
            logger.debug("Using confidence weights: min=%.4f, max=%.4f", weights.min(), weights.max())
        else:
            logger.warning(
                "Confidence map shape %s does not match depth %s, ignoring",
                confidence_map.shape,
                depth_map.shape,
            )

    # Solve for scale and shift
    if use_ransac and len(metric_depths) >= 6:
        s, b, inlier_mask = _ransac_scale_shift(
            mono_depths,
            metric_depths,
            weights=weights,
            n_iterations=ransac_iterations,
            inlier_threshold=ransac_inlier_threshold,
        )
        n_inliers = inlier_mask.sum()
        logger.debug(
            "RANSAC alignment: scale=%.4f, shift=%.4f, inliers=%d/%d",
            s, b, n_inliers, len(metric_depths),
        )
    else:
        # Weighted least squares: W @ A @ x = W @ b
        A = np.column_stack([mono_depths, np.ones_like(mono_depths)])
        if weights is not None:
            W = np.diag(weights)
            result = np.linalg.lstsq(W @ A, W @ metric_depths, rcond=None)
        else:
            result = np.linalg.lstsq(A, metric_depths, rcond=None)
        s, b = result[0]
        logger.debug("LeastSq alignment: scale=%.4f, shift=%.4f, points=%d", s, b, len(metric_depths))

    # Scale/shift sanity checks
    if s < 0:
        logger.warning(
            "Negative scale %.4f (inverted depth), flipping to %.4f", s, abs(s),
        )
        s = abs(s)
        b = -b

    if abs(s) > 100:
        logger.warning(
            "Very large scale %.4f — potential misalignment. Clamping to 100.",
            s,
        )
        s = np.clip(s, -100.0, 100.0)

    if abs(b) > 100:
        logger.warning(
            "Very large shift %.4f — potential misalignment. Clamping to [-100, 100].",
            b,
        )
        b = np.clip(b, -100.0, 100.0)

    aligned = s * depth_map + b

    # Clamp to physically reasonable range
    aligned = np.clip(aligned, depth_clamp_min, depth_clamp_max)

    return aligned.astype(np.float32)


def compute_confidence(
    depth_map: np.ndarray,
    aligned_depth: np.ndarray,
    colmap_depths: np.ndarray,
    pixel_coords: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel confidence based on alignment residuals.

    Args:
        depth_map: (H, W) original monocular depth.
        aligned_depth: (H, W) scale-aligned depth.
        colmap_depths: (N,) metric depths from COLMAP sparse points.
        pixel_coords: (N, 2) pixel coordinates of sparse points.

    Returns:
        (H, W) confidence map in [0, 1].
    """
    H, W = aligned_depth.shape[:2]

    if len(colmap_depths) == 0:
        return np.zeros((H, W), dtype=np.float32)

    px = np.round(pixel_coords[:, 0]).astype(int)
    py = np.round(pixel_coords[:, 1]).astype(int)
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py = px[in_bounds], py[in_bounds]
    colmap_depths = colmap_depths[in_bounds]

    # Compute residuals at sparse points
    aligned_at_sparse = aligned_depth[py, px]
    residuals = np.abs(aligned_at_sparse - colmap_depths)

    if len(residuals) == 0 or np.median(residuals) == 0:
        return np.ones((H, W), dtype=np.float32)

    # Use median absolute residual as uncertainty reference
    median_res = np.median(residuals)

    # Per-pixel confidence: higher when depth is smooth and within expected range
    # Use gradient magnitude as a proxy for uncertainty
    grad_x = np.gradient(aligned_depth, axis=1)
    grad_y = np.gradient(aligned_depth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize gradient magnitude relative to median residual
    confidence = np.exp(-grad_mag / (median_res + 1e-8))
    confidence = np.clip(confidence, 0.0, 1.0)

    return confidence.astype(np.float32)


def batch_align(
    depth_dir: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    use_ransac: bool = True,
    ransac_iterations: int = 1000,
    ransac_inlier_threshold: float = 0.05,
    depth_clamp_min: float = _DEPTH_MIN_DEFAULT,
    depth_clamp_max: float = _DEPTH_MAX_DEFAULT,
    confidence_dir: Optional[Path] = None,
) -> None:
    """Align all depth maps in a directory using a COLMAP sparse model.

    Args:
        depth_dir: Directory containing .npy depth maps (named by image stem).
        colmap_model_dir: Directory containing cameras.bin, images.bin, points3D.bin.
        output_dir: Directory to write aligned depth maps and confidence maps.
        use_ransac: Use RANSAC-based robust alignment.
        ransac_iterations: Number of RANSAC iterations per frame.
        ransac_inlier_threshold: RANSAC inlier threshold in meters.
        depth_clamp_min: Minimum depth clamp (meters).
        depth_clamp_max: Maximum depth clamp (meters).
        confidence_dir: Optional directory with DA3 confidence maps (.npy, same stems).
            If provided, confidence values are used as weights during alignment.
    """
    from tqdm import tqdm

    depth_dir = Path(depth_dir)
    colmap_model_dir = Path(colmap_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if confidence_dir is not None:
        confidence_dir = Path(confidence_dir)

    # Load COLMAP model — handle missing or empty model gracefully
    cameras_bin = colmap_model_dir / "cameras.bin"
    images_bin = colmap_model_dir / "images.bin"

    if not cameras_bin.exists() or not images_bin.exists():
        logger.warning(
            "COLMAP model missing at %s (cameras.bin exists=%s, images.bin exists=%s). "
            "Copying raw depth maps as-is without alignment.",
            colmap_model_dir, cameras_bin.exists(), images_bin.exists(),
        )
        # Copy raw depth maps without alignment
        import shutil
        depth_files = sorted(depth_dir.glob("*.npy"))
        for df in depth_files:
            if df.name.endswith("_conf.npy"):
                continue
            stem = df.stem
            out_path = output_dir / f"{stem}.npy"
            conf_path = output_dir / f"{stem}_confidence.npy"
            if not out_path.exists():
                shutil.copy2(str(df), str(out_path))
            if not conf_path.exists():
                depth_data = np.load(str(df))
                np.save(str(conf_path), np.ones_like(depth_data, dtype=np.float32) * 0.5)
        logger.info("Copied %d raw depth maps to %s (no alignment)", len(depth_files), output_dir)
        return

    logger.info("Loading COLMAP sparse model from %s", colmap_model_dir)
    cameras = read_cameras_binary(cameras_bin)
    images = read_images_binary(images_bin)
    points3d = read_points3d_binary(colmap_model_dir / "points3D.bin")

    if not images:
        logger.warning(
            "COLMAP model has 0 registered images. "
            "Copying raw depth maps as-is without alignment.",
        )
        import shutil
        depth_files = sorted(depth_dir.glob("*.npy"))
        for df in depth_files:
            if df.name.endswith("_conf.npy"):
                continue
            stem = df.stem
            out_path = output_dir / f"{stem}.npy"
            conf_path = output_dir / f"{stem}_confidence.npy"
            if not out_path.exists():
                shutil.copy2(str(df), str(out_path))
            if not conf_path.exists():
                depth_data = np.load(str(df))
                np.save(str(conf_path), np.ones_like(depth_data, dtype=np.float32) * 0.5)
        logger.info("Copied %d raw depth maps to %s (no alignment)", len(depth_files), output_dir)
        return

    # Build point3d_id -> xyz lookup
    p3d_xyz = {pid: pt.xyz for pid, pt in points3d.items()}

    aligned_count = 0
    skipped_count = 0
    copy_count = 0

    for image in tqdm(images.values(), desc="Aligning depth maps"):
        stem = Path(image.name).stem
        depth_path = depth_dir / f"{stem}.npy"
        out_path = output_dir / f"{stem}.npy"
        conf_path = output_dir / f"{stem}_confidence.npy"

        if out_path.exists() and conf_path.exists():
            skipped_count += 1
            continue

        if not depth_path.exists():
            logger.debug("No depth map for %s, skipping", image.name)
            continue

        depth_map = np.load(str(depth_path))

        # Load external confidence map (e.g. from DA3) if available
        ext_confidence = None
        if confidence_dir is not None:
            ext_conf_path = confidence_dir / f"{stem}.npy"
            if ext_conf_path.exists():
                ext_confidence = np.load(str(ext_conf_path)).astype(np.float64)

        # Gather COLMAP 3D points visible in this image
        visible_mask = image.point3D_ids >= 0
        visible_p3d_ids = image.point3D_ids[visible_mask]
        visible_xys = image.xys[visible_mask]

        colmap_pts = []
        for p3d_id in visible_p3d_ids:
            if p3d_id in p3d_xyz:
                colmap_pts.append(p3d_xyz[p3d_id])

        if len(colmap_pts) < 3:
            logger.warning(
                "Image %s: only %d sparse points, copying unaligned depth as output",
                image.name, len(colmap_pts),
            )
            # Write unaligned depth copy so downstream stages have a file for every frame
            np.save(str(out_path), depth_map)
            np.save(str(conf_path), np.ones_like(depth_map, dtype=np.float32) * 0.3)
            copy_count += 1
            continue

        colmap_pts = np.array(colmap_pts, dtype=np.float64)

        # Camera parameters
        camera = cameras[image.camera_id]
        K = get_intrinsics_matrix(camera)
        R = qvec_to_rotmat(image.qvec)
        t = image.tvec

        # Align with RANSAC and optional confidence weighting
        aligned = align_depth_to_colmap(
            depth_map,
            colmap_pts,
            K,
            (R, t),
            confidence_map=ext_confidence,
            use_ransac=use_ransac,
            ransac_iterations=ransac_iterations,
            ransac_inlier_threshold=ransac_inlier_threshold,
            depth_clamp_min=depth_clamp_min,
            depth_clamp_max=depth_clamp_max,
        )

        # Compute confidence
        pts_cam = (R @ colmap_pts.T).T + t.ravel()
        metric_depths = pts_cam[:, 2]
        pts_proj = (K @ pts_cam.T).T
        pixel_coords = pts_proj[:, :2] / pts_proj[:, 2:3]
        valid = metric_depths > 0
        confidence = compute_confidence(
            depth_map, aligned, metric_depths[valid], pixel_coords[valid]
        )

        np.save(str(out_path), aligned)
        np.save(str(conf_path), confidence)
        aligned_count += 1

    logger.info(
        "Alignment complete: %d aligned, %d copied (0 COLMAP pts), %d skipped (cached)",
        aligned_count,
        copy_count,
        skipped_count,
    )
