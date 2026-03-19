"""Unified DA3 stage: replaces COLMAP SfM + depth estimation + depth alignment.

Depth Anything 3 provides depth maps, confidence maps, and camera poses
(extrinsics + intrinsics) in a single forward pass. This module runs DA3
on all frames and writes outputs in COLMAP binary format so that downstream
stages (Gaussian splatting, FLAME fitting) can consume them without changes.
"""

import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# COLMAP camera model IDs
_PINHOLE_MODEL_ID = 1  # PINHOLE: fx, fy, cx, cy


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def _write_cameras_binary(path: Path, camera_data: dict) -> None:
    """Write cameras.bin in COLMAP binary format.

    Args:
        path: Output file path.
        camera_data: Dict with keys 'camera_id', 'width', 'height', 'params' (fx, fy, cx, cy).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate camera params are finite
    params = camera_data["params"]
    for i, p in enumerate(params):
        if not np.isfinite(p):
            logger.warning("Camera param[%d] is not finite (%.4f), clamping to 0", i, p)
            params[i] = 0.0

    with open(path, "wb") as f:
        # Number of cameras
        f.write(struct.pack("<Q", 1))
        # camera_id (uint32), model_id (int32), width (uint64), height (uint64)
        f.write(struct.pack("<Ii",
                            camera_data["camera_id"],
                            _PINHOLE_MODEL_ID))
        f.write(struct.pack("<QQ",
                            camera_data["width"],
                            camera_data["height"]))
        # PINHOLE has 4 params: fx, fy, cx, cy
        f.write(struct.pack("<4d", params[0], params[1], params[2], params[3]))


def _write_images_binary(path: Path, images: List[dict]) -> None:
    """Write images.bin in COLMAP binary format.

    Handles edge case of 0 registered images by writing a valid empty file.
    Validates that all quaternions are normalized.

    Args:
        path: Output file path.
        images: List of dicts with keys: 'image_id', 'qvec', 'tvec', 'camera_id', 'name'.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(images) == 0:
        logger.warning("Writing empty images.bin (0 registered images)")

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            # image_id (uint32)
            f.write(struct.pack("<I", img["image_id"]))

            # qvec (4 doubles: w, x, y, z) — ensure normalized
            qvec = np.array(img["qvec"], dtype=np.float64)
            qnorm = np.linalg.norm(qvec)
            if qnorm < 1e-10 or not np.all(np.isfinite(qvec)):
                logger.warning(
                    "Image %s: degenerate quaternion (norm=%.6f), using identity",
                    img["name"], qnorm,
                )
                qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                qvec = qvec / qnorm
            f.write(struct.pack("<4d", qvec[0], qvec[1], qvec[2], qvec[3]))

            # tvec (3 doubles) — validate finite
            tvec = np.array(img["tvec"], dtype=np.float64)
            if not np.all(np.isfinite(tvec)):
                logger.warning("Image %s: non-finite tvec, clamping to zero", img["name"])
                tvec = np.where(np.isfinite(tvec), tvec, 0.0)
            f.write(struct.pack("<3d", tvec[0], tvec[1], tvec[2]))

            # camera_id (uint32)
            f.write(struct.pack("<I", img["camera_id"]))
            # name (null-terminated string)
            f.write(img["name"].encode("utf-8") + b"\x00")
            # num_points2d = 0 (no 2D-3D correspondences from DA3)
            f.write(struct.pack("<Q", 0))


def _write_points3d_binary(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write points3D.bin in COLMAP binary format.

    Handles edge case of 0 points by writing a valid empty file.
    Validates all coordinates are finite.

    Args:
        path: Output file path.
        points: (N, 3) float64 array of 3D point positions.
        colors: (N, 3) uint8 array of RGB colors.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)

    if n == 0:
        logger.info("Writing empty points3D.bin (0 points)")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", 0))
        return

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            # Validate coordinates are finite
            pt = points[i]
            if not np.all(np.isfinite(pt)):
                pt = np.where(np.isfinite(pt), pt, 0.0)

            # point3D_id (uint64)
            f.write(struct.pack("<Q", i + 1))
            # xyz (3 doubles)
            f.write(struct.pack("<3d", pt[0], pt[1], pt[2]))
            # rgb (3 uint8)
            f.write(struct.pack("<3B", colors[i, 0], colors[i, 1], colors[i, 2]))
            # error (double)
            f.write(struct.pack("<d", 0.0))
            # track_length = 0 (no image observations for depth-unprojected points)
            f.write(struct.pack("<Q", 0))


def _unproject_depth_to_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5,
    stride: int = 4,
    depth_min: float = 0.05,
    depth_max: float = 20.0,
) -> tuple:
    """Unproject a depth map to 3D points in world coordinates.

    Args:
        depth: (H, W) metric depth map.
        intrinsics: (3, 3) camera intrinsic matrix.
        extrinsics: (4, 4) or (3, 4) world-to-camera transformation.
        image: (H, W, 3) RGB image for point colors.
        confidence: Optional (H, W) confidence map for filtering.
        conf_threshold: Minimum confidence to keep a point.
        stride: Sub-sample every N pixels to reduce point count.
        depth_min: Minimum valid depth.
        depth_max: Maximum valid depth.

    Returns:
        (points_world, colors) where points_world is (M, 3) float64
        and colors is (M, 3) uint8.
    """
    H, W = depth.shape[:2]
    img_H, img_W = image.shape[:2]

    # Build pixel grid at stride intervals, clamped to valid range for ALL arrays
    max_y = min(H, img_H) - 1
    max_x = min(W, img_W) - 1
    ys = np.arange(0, max_y + 1, stride)
    xs = np.arange(0, max_x + 1, stride)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.flatten()
    yy = yy.flatten()

    # Sample depth and build validity mask
    d = depth[yy, xx].astype(np.float64)
    valid = (d > depth_min) & (d < depth_max) & np.isfinite(d)

    if confidence is not None:
        # Clamp indices for confidence array which may differ in size
        cy = np.clip(yy, 0, confidence.shape[0] - 1)
        cx = np.clip(xx, 0, confidence.shape[1] - 1)
        conf = confidence[cy, cx]
        valid &= conf > conf_threshold

    xx = xx[valid]
    yy = yy[valid]
    d = d[valid]

    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    # Unproject to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_cam = (xx.astype(np.float64) - cx) * d / fx
    y_cam = (yy.astype(np.float64) - cy) * d / fy
    z_cam = d

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (M, 3)

    # Transform to world coordinates: P_world = R^T @ (P_cam - t)
    if extrinsics.shape == (4, 4):
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
    else:  # (3, 4)
        R = extrinsics[:, :3]
        t = extrinsics[:, 3]

    # world-to-camera: P_cam = R @ P_world + t
    # So: P_world = R^T @ (P_cam - t)
    points_world = (R.T @ (points_cam - t).T).T

    # Sample colors
    if image.ndim == 3 and image.shape[2] == 3:
        colors = image[yy, xx]  # (M, 3)
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(d), 3), 128, dtype=np.uint8)

    return points_world.astype(np.float64), colors


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a simple PLY point cloud file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", float(points[i, 0]), float(points[i, 1]), float(points[i, 2])))
            f.write(struct.pack("<BBB", int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))


def _record_da2_fallback_frame(
    fpath: Path,
    depth_dir: Path,
    all_frame_names: list,
    all_depths: list,
    all_confs: list,
    all_extrinsics: list,
    all_intrinsics: list,
    all_valid_pose: list,
    da2_fallback_frames: list,
    log: logging.Logger,
) -> None:
    """Record a frame that needs DA2 fallback due to DA3 OOM at chunk_size=1.

    Writes a placeholder depth map and marks the frame for later DA2 processing.
    """
    da2_fallback_frames.append(fpath)
    all_frame_names.append(fpath.name)

    img = cv2.imread(str(fpath))
    if img is not None:
        h, w = img.shape[:2]
    else:
        h, w = 720, 1280

    # Try DA2 inline if available
    try:
        from src.depth.depth_estimator import DepthEstimator
        estimator = DepthEstimator(model_type="v2")
        da2_result = estimator.estimate(str(fpath))
        if da2_result is not None:
            depth_i = da2_result["depth"].astype(np.float32)
            log.info("DA2 fallback succeeded for %s", fpath.name)
            all_depths.append(depth_i)
            all_confs.append(np.ones_like(depth_i) * 0.5)  # Lower confidence for DA2
            np.save(str(depth_dir / f"{fpath.stem}.npy"), depth_i)
            np.save(str(depth_dir / f"{fpath.stem}_conf.npy"),
                    np.ones_like(depth_i, dtype=np.float32) * 0.5)
            all_extrinsics.append(np.eye(4, dtype=np.float64))
            f_est = max(depth_i.shape) * 0.8
            all_intrinsics.append(np.array([
                [f_est, 0.0, depth_i.shape[1] / 2.0],
                [0.0, f_est, depth_i.shape[0] / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64))
            all_valid_pose.append(False)
            return
    except Exception as da2_err:
        log.warning("DA2 fallback also failed for %s: %s", fpath.name, da2_err)

    # Final fallback: placeholder zeros
    placeholder_depth = np.zeros((h, w), dtype=np.float32)
    all_depths.append(placeholder_depth)
    all_confs.append(np.zeros((h, w), dtype=np.float32))
    all_extrinsics.append(np.eye(4, dtype=np.float64))
    f_est = max(h, w) * 0.8
    all_intrinsics.append(np.array([
        [f_est, 0.0, w / 2.0],
        [0.0, f_est, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64))
    all_valid_pose.append(False)
    np.save(str(depth_dir / f"{fpath.stem}.npy"), placeholder_depth)
    np.save(str(depth_dir / f"{fpath.stem}_conf.npy"),
            np.zeros((h, w), dtype=np.float32))


def _process_remaining_chunk(
    model,
    remaining_paths: List[Path],
    remaining_str: List[str],
    process_res: int,
    use_ray_pose: bool,
    sub_chunk_size: int,
    depth_dir: Path,
    all_frame_names: list,
    all_depths: list,
    all_confs: list,
    all_extrinsics: list,
    all_intrinsics: list,
    all_valid_pose: list,
    da2_fallback_frames: list,
    log: logging.Logger,
) -> None:
    """Process remaining frames in sub-chunks after an OOM reduction."""
    for sub_start in range(0, len(remaining_paths), sub_chunk_size):
        sub_paths = remaining_paths[sub_start:sub_start + sub_chunk_size]
        sub_str = remaining_str[sub_start:sub_start + sub_chunk_size]
        try:
            pred = model.inference(
                image=sub_str,
                process_res=process_res,
                process_res_method="upper_bound_resize",
                use_ray_pose=use_ray_pose,
            )
            for i, fpath in enumerate(sub_paths):
                all_frame_names.append(fpath.name)
                depth_i = pred.depth[i].astype(np.float32)
                all_depths.append(depth_i)
                np.save(str(depth_dir / f"{fpath.stem}.npy"), depth_i)

                if pred.conf is not None:
                    conf_i = pred.conf[i].astype(np.float32)
                else:
                    conf_i = np.ones_like(depth_i)
                all_confs.append(conf_i)
                np.save(str(depth_dir / f"{fpath.stem}_conf.npy"), conf_i)

                if pred.extrinsics is not None:
                    ext_i = pred.extrinsics[i].astype(np.float64)
                    if ext_i.shape == (3, 4):
                        ext_full = np.eye(4, dtype=np.float64)
                        ext_full[:3, :] = ext_i
                        ext_i = ext_full
                    all_extrinsics.append(ext_i)
                    all_valid_pose.append(True)
                else:
                    all_extrinsics.append(np.eye(4, dtype=np.float64))
                    all_valid_pose.append(False)

                if pred.intrinsics is not None:
                    all_intrinsics.append(pred.intrinsics[i].astype(np.float64))
                else:
                    h, w = depth_i.shape[:2]
                    f_est = max(h, w) * 0.8
                    all_intrinsics.append(np.array([
                        [f_est, 0.0, w / 2.0],
                        [0.0, f_est, h / 2.0],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64))
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.warning("DA3 OOM on remaining sub-chunk, falling back to DA2 for %d frames", len(sub_paths))
            for fpath in sub_paths:
                _record_da2_fallback_frame(
                    fpath, depth_dir, all_frame_names, all_depths, all_confs,
                    all_extrinsics, all_intrinsics, all_valid_pose,
                    da2_fallback_frames, log,
                )
        except Exception as e:
            log.error("DA3 sub-chunk failed: %s", e)
            for fpath in sub_paths:
                all_frame_names.append(fpath.name)
                img = cv2.imread(str(fpath))
                h, w = (img.shape[:2]) if img is not None else (720, 1280)
                placeholder = np.zeros((h, w), dtype=np.float32)
                all_depths.append(placeholder)
                all_confs.append(np.zeros((h, w), dtype=np.float32))
                all_extrinsics.append(np.eye(4, dtype=np.float64))
                f_est = max(h, w) * 0.8
                all_intrinsics.append(np.array([
                    [f_est, 0.0, w / 2.0],
                    [0.0, f_est, h / 2.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64))
                all_valid_pose.append(False)
                np.save(str(depth_dir / f"{fpath.stem}.npy"), placeholder)
                np.save(str(depth_dir / f"{fpath.stem}_conf.npy"),
                        np.zeros((h, w), dtype=np.float32))


def _validate_poses(
    all_extrinsics: List[np.ndarray],
    all_valid_pose: list,
    all_frame_names: List[str],
) -> int:
    """Validate DA3 poses and filter out degenerate ones.

    Checks for all-zero rotation, NaN values, and filters frames with
    low confidence (below 10th percentile of rotation magnitudes).

    Args:
        all_extrinsics: List of (4, 4) extrinsic matrices (modified in-place).
        all_valid_pose: List of booleans (modified in-place).
        all_frame_names: Frame names for logging.

    Returns:
        Number of valid poses after filtering.
    """
    rotation_norms = []
    for idx in range(len(all_extrinsics)):
        if not all_valid_pose[idx]:
            continue

        ext = all_extrinsics[idx]
        R = ext[:3, :3]
        t = ext[:3, 3]

        # Check for NaN/Inf in the extrinsic matrix
        if not np.all(np.isfinite(ext)):
            logger.warning("Frame %s: degenerate pose (NaN/Inf), marking invalid", all_frame_names[idx])
            all_valid_pose[idx] = False
            continue

        # Check for all-zero rotation (degenerate)
        r_norm = np.linalg.norm(R)
        if r_norm < 1e-6:
            logger.warning("Frame %s: all-zero rotation matrix, marking invalid", all_frame_names[idx])
            all_valid_pose[idx] = False
            continue

        rotation_norms.append((idx, r_norm))

    # Filter out frames below 10th percentile of rotation magnitude
    # (these are likely unreliable poses)
    if len(rotation_norms) >= 5:
        norms = np.array([rn for _, rn in rotation_norms])
        p10 = np.percentile(norms, 10)
        for idx, r_norm in rotation_norms:
            if r_norm < p10:
                logger.debug(
                    "Frame %s: rotation norm %.4f below 10th percentile (%.4f), marking invalid",
                    all_frame_names[idx], r_norm, p10,
                )
                all_valid_pose[idx] = False

    n_valid = sum(all_valid_pose)
    n_total = len(all_frame_names)
    logger.info("DA3 produced %d valid poses out of %d frames", n_valid, n_total)
    return n_valid


def _validate_point_cloud(points: np.ndarray, colors: np.ndarray) -> tuple:
    """Validate and clean a point cloud, filtering NaN/Inf coordinates.

    Args:
        points: (N, 3) point positions.
        colors: (N, 3) point colors.

    Returns:
        (cleaned_points, cleaned_colors) with invalid points removed.
    """
    if len(points) == 0:
        return points, colors

    # Filter NaN/Inf coordinates
    valid_mask = np.all(np.isfinite(points), axis=1)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning("Point cloud: filtered %d NaN/Inf points out of %d", n_invalid, len(points))
        points = points[valid_mask]
        colors = colors[valid_mask]

    if len(points) == 0:
        logger.warning("Point cloud: all points were NaN/Inf")
        return points, colors

    # Check point count
    if len(points) < 1000:
        logger.warning("Point cloud has only %d points (< 1000), reconstruction may be poor", len(points))

    # Compute and log bounding box / scene scale
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    scene_scale = np.linalg.norm(bbox_size)
    logger.info(
        "Point cloud: %d points, bbox min=[%.2f, %.2f, %.2f], max=[%.2f, %.2f, %.2f], scale=%.3f",
        len(points),
        bbox_min[0], bbox_min[1], bbox_min[2],
        bbox_max[0], bbox_max[1], bbox_max[2],
        scene_scale,
    )

    return points, colors


def run_da3_unified(
    frames_dir: Path,
    output_dir: Path,
    model_name: str = "da3-large",
    process_res: int = 504,
    use_ray_pose: bool = False,
    chunk_size: int = 8,
    pointcloud_stride: int = 4,
    conf_threshold: float = 0.3,
    device: str = "cuda",
) -> dict:
    """Run DA3 as a unified replacement for COLMAP + depth estimation + alignment.

    Performs depth estimation with camera pose recovery in a single pass.
    Writes outputs in COLMAP binary format for downstream compatibility.

    Args:
        frames_dir: Directory containing input frames (.png or .jpg).
        output_dir: Root output directory. Will contain:
            - depth/ : per-frame .npy depth maps and confidence maps
            - colmap/sparse/0/ : cameras.bin, images.bin, points3D.bin
            - dense_points.ply : dense point cloud
        model_name: DA3 model preset name (e.g. "da3-large", "da3-giant").
        process_res: Processing resolution for DA3 (504 is default, higher = better but slower).
        use_ray_pose: Use ray-based pose estimation instead of camera decoder.
        chunk_size: Number of frames to process per DA3 inference call.
            8 works well for RTX 3080 16GB with process_res=504.
        pointcloud_stride: Pixel stride for dense point cloud generation.
            Lower = denser cloud but more memory. 4 is a good default.
        conf_threshold: Minimum confidence for points in the dense cloud.
        device: Torch device string.

    Returns:
        Summary dict with stats: num_frames, num_points, mean_confidence,
        num_valid_poses, model_name, process_res.
    """
    from tqdm import tqdm

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # Collect frame paths
    frame_paths = sorted(
        list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
    )
    if not frame_paths:
        raise FileNotFoundError(f"No .png or .jpg frames found in {frames_dir}")

    logger.info("DA3 unified stage: %d frames, model=%s, process_res=%d, use_ray_pose=%s",
                len(frame_paths), model_name, process_res, use_ray_pose)

    # Create output directories
    depth_dir = output_dir / "depth"
    colmap_dir = output_dir / "colmap" / "sparse" / "0"
    depth_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Load DA3 model
    from depth_anything_3.api import DepthAnything3

    _DA3_MODELS = {
        "da3-small": "depth-anything/DA3-Small",
        "da3-base": "depth-anything/DA3-Base",
        "da3-large": "depth-anything/DA3-Large",
        "da3-giant": "depth-anything/DA3-Giant",
        "da3metric-large": "depth-anything/DA3Metric-Large",
        "da3mono-large": "depth-anything/DA3Mono-Large",
    }
    hf_id = _DA3_MODELS.get(model_name, model_name)

    logger.info("Loading DA3 model: %s", hf_id)
    model = DepthAnything3.from_pretrained(hf_id)
    model = model.to(device=device)
    logger.info("DA3 model loaded on %s", device)

    # Process all frames in chunks
    # We accumulate per-frame results and then do a single pass to write COLMAP + point cloud
    all_depths = []       # List of (H, W) arrays
    all_confs = []        # List of (H, W) arrays
    all_extrinsics = []   # List of (4, 4) arrays
    all_intrinsics = []   # List of (3, 3) arrays
    all_frame_names = []  # Corresponding filenames
    all_valid_pose = []   # Whether the pose is usable

    num_chunks = (len(frame_paths) + chunk_size - 1) // chunk_size

    # Track DA2 fallback frames for reporting
    da2_fallback_frames = []

    for chunk_start in tqdm(range(0, len(frame_paths), chunk_size),
                            desc="DA3 unified inference",
                            total=num_chunks):
        chunk_paths = frame_paths[chunk_start:chunk_start + chunk_size]
        chunk_str_paths = [str(p) for p in chunk_paths]

        # Retry with halving chunk size on OOM
        current_chunk_size = len(chunk_paths)
        oom_retry = True
        while oom_retry:
            oom_retry = False
            try:
                prediction = model.inference(
                    image=chunk_str_paths[:current_chunk_size] if current_chunk_size < len(chunk_str_paths) else chunk_str_paths,
                    process_res=process_res,
                    process_res_method="upper_bound_resize",
                    use_ray_pose=use_ray_pose,
                )

                # Process only the frames that were in this sub-chunk
                sub_chunk_paths = chunk_paths[:current_chunk_size] if current_chunk_size < len(chunk_paths) else chunk_paths
                for i, fpath in enumerate(sub_chunk_paths):
                    frame_name = fpath.name
                    all_frame_names.append(frame_name)

                    # Depth map
                    depth_i = prediction.depth[i].astype(np.float32)
                    all_depths.append(depth_i)
                    np.save(str(depth_dir / f"{fpath.stem}.npy"), depth_i)

                    # Confidence map
                    if prediction.conf is not None:
                        conf_i = prediction.conf[i].astype(np.float32)
                    else:
                        conf_i = np.ones_like(depth_i)
                    all_confs.append(conf_i)
                    np.save(str(depth_dir / f"{fpath.stem}_conf.npy"), conf_i)

                    # Camera extrinsics (4x4 world-to-camera)
                    if prediction.extrinsics is not None:
                        ext_i = prediction.extrinsics[i].astype(np.float64)
                        # DA3 may return (3, 4) or (4, 4). Normalize to (4, 4).
                        if ext_i.shape == (3, 4):
                            ext_full = np.eye(4, dtype=np.float64)
                            ext_full[:3, :] = ext_i
                            ext_i = ext_full
                        all_extrinsics.append(ext_i)
                        all_valid_pose.append(True)
                    else:
                        all_extrinsics.append(np.eye(4, dtype=np.float64))
                        all_valid_pose.append(False)

                    # Camera intrinsics (3x3)
                    if prediction.intrinsics is not None:
                        ixt_i = prediction.intrinsics[i].astype(np.float64)
                        all_intrinsics.append(ixt_i)
                    else:
                        # Fallback: estimate from image dimensions
                        h, w = depth_i.shape[:2]
                        f_est = max(h, w) * 0.8
                        all_intrinsics.append(np.array([
                            [f_est, 0.0, w / 2.0],
                            [0.0, f_est, h / 2.0],
                            [0.0, 0.0, 1.0],
                        ], dtype=np.float64))

                # If we reduced chunk size, process remaining frames in next sub-chunks
                if current_chunk_size < len(chunk_paths):
                    remaining = chunk_paths[current_chunk_size:]
                    remaining_str = [str(p) for p in remaining]
                    # Recursively process remaining with same reduced chunk size
                    _process_remaining_chunk(
                        model, remaining, remaining_str, process_res, use_ray_pose,
                        current_chunk_size, depth_dir, all_frame_names, all_depths,
                        all_confs, all_extrinsics, all_intrinsics, all_valid_pose,
                        da2_fallback_frames, logger,
                    )

            except torch.cuda.OutOfMemoryError:
                # Clear VRAM before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                new_chunk_size = max(1, current_chunk_size // 2)
                logger.warning(
                    "DA3 OOM on chunk of %d images, reducing chunk size to %d",
                    current_chunk_size, new_chunk_size,
                )

                if new_chunk_size >= 1 and current_chunk_size > 1:
                    current_chunk_size = new_chunk_size
                    oom_retry = True
                    continue

                # OOM at chunk_size=1 — fall back to DA2 for these frames
                logger.warning(
                    "DA3 OOM at chunk_size=1, falling back to DA2 for %d frames at offset %d",
                    len(chunk_paths), chunk_start,
                )
                for fpath in chunk_paths:
                    _record_da2_fallback_frame(
                        fpath, depth_dir, all_frame_names, all_depths, all_confs,
                        all_extrinsics, all_intrinsics, all_valid_pose,
                        da2_fallback_frames, logger,
                    )

            except Exception as e:
                logger.error("DA3 chunk failed at offset %d: %s", chunk_start, e)
                # Record failures with identity poses
                for fpath in chunk_paths:
                    all_frame_names.append(fpath.name)
                    # Try to at least get image dimensions for a placeholder
                    img = cv2.imread(str(fpath))
                    if img is not None:
                        h, w = img.shape[:2]
                    else:
                        h, w = 720, 1280
                    placeholder_depth = np.zeros((h, w), dtype=np.float32)
                    all_depths.append(placeholder_depth)
                    all_confs.append(np.zeros((h, w), dtype=np.float32))
                    all_extrinsics.append(np.eye(4, dtype=np.float64))
                    f_est = max(h, w) * 0.8
                    all_intrinsics.append(np.array([
                        [f_est, 0.0, w / 2.0],
                        [0.0, f_est, h / 2.0],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64))
                    all_valid_pose.append(False)
                    np.save(str(depth_dir / f"{fpath.stem}.npy"), placeholder_depth)
                    np.save(str(depth_dir / f"{fpath.stem}_conf.npy"),
                            np.zeros((h, w), dtype=np.float32))

        # Free VRAM between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if da2_fallback_frames:
        logger.warning(
            "DA3 OOM fallback: %d frames need DA2 depth estimation: %s",
            len(da2_fallback_frames),
            [f.name for f in da2_fallback_frames],
        )

    n_frames = len(all_frame_names)

    # Validate poses: check for degenerate rotation, NaN, low confidence
    n_valid = _validate_poses(all_extrinsics, all_valid_pose, all_frame_names)
    logger.info("DA3 inference complete: %d frames, %d with valid poses", n_frames, n_valid)

    # ---- Write COLMAP binary model ----
    # Use the first valid intrinsics as the shared camera model.
    # DA3 intrinsics correspond to the depth map resolution, which may differ
    # from the original image resolution. We read the first original frame to
    # get the actual image dimensions and scale intrinsics accordingly.
    first_orig = cv2.imread(str(frame_paths[0]))
    if first_orig is not None:
        orig_h, orig_w = first_orig.shape[:2]
    else:
        orig_h, orig_w = all_depths[0].shape[:2]

    # DA3 depth maps may be at process_res; intrinsics are at depth map resolution.
    depth_h, depth_w = all_depths[0].shape[:2]

    # Compute scale factors between depth map and original image
    scale_x = orig_w / depth_w
    scale_y = orig_h / depth_h

    ref_K = all_intrinsics[0]
    fx_scaled = ref_K[0, 0] * scale_x
    fy_scaled = ref_K[1, 1] * scale_y
    cx_scaled = ref_K[0, 2] * scale_x
    cy_scaled = ref_K[1, 2] * scale_y

    camera_data = {
        "camera_id": 1,
        "width": orig_w,
        "height": orig_h,
        "params": [fx_scaled, fy_scaled, cx_scaled, cy_scaled],
    }
    _write_cameras_binary(colmap_dir / "cameras.bin", camera_data)
    logger.info("Wrote cameras.bin (PINHOLE: fx=%.1f fy=%.1f cx=%.1f cy=%.1f, %dx%d)",
                fx_scaled, fy_scaled, cx_scaled, cy_scaled, orig_w, orig_h)

    # Write images.bin — one entry per frame with valid pose
    image_entries = []
    for idx in range(n_frames):
        if not all_valid_pose[idx]:
            continue

        ext = all_extrinsics[idx]  # (4, 4) world-to-camera
        R = ext[:3, :3]
        t = ext[:3, 3]

        qvec = _rotation_matrix_to_quaternion(R)

        image_entries.append({
            "image_id": idx + 1,
            "qvec": qvec,
            "tvec": t,
            "camera_id": 1,
            "name": all_frame_names[idx],
        })

    _write_images_binary(colmap_dir / "images.bin", image_entries)
    logger.info("Wrote images.bin with %d posed images", len(image_entries))

    # ---- Generate dense point cloud ----
    logger.info("Generating dense point cloud from DA3 depth maps...")
    all_points = []
    all_colors = []

    for idx in tqdm(range(n_frames), desc="Unprojecting depth to 3D"):
        if not all_valid_pose[idx]:
            continue

        depth_i = all_depths[idx]
        conf_i = all_confs[idx]
        ext_i = all_extrinsics[idx]
        K_i = all_intrinsics[idx]  # At depth map resolution

        # Read original image for colors (resize to depth map resolution)
        img = cv2.imread(str(frames_dir / all_frame_names[idx]))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (depth_h, depth_w):
            img_rgb = cv2.resize(img_rgb, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)

        pts, cols = _unproject_depth_to_points(
            depth=depth_i,
            intrinsics=K_i,
            extrinsics=ext_i,
            image=img_rgb,
            confidence=conf_i,
            conf_threshold=conf_threshold,
            stride=pointcloud_stride,
        )
        if len(pts) > 0:
            all_points.append(pts)
            all_colors.append(cols)

    if all_points:
        merged_points = np.concatenate(all_points, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)

        # Validate point cloud: filter NaN/Inf, log bounding box and scale
        merged_points, merged_colors = _validate_point_cloud(merged_points, merged_colors)

        # Subsample if point cloud is very large (>2M points)
        max_points = 2_000_000
        if len(merged_points) > max_points:
            logger.info("Subsampling point cloud from %d to %d points",
                        len(merged_points), max_points)
            rng = np.random.default_rng(42)
            indices = rng.choice(len(merged_points), size=max_points, replace=False)
            indices.sort()
            merged_points = merged_points[indices]
            merged_colors = merged_colors[indices]

        # Write PLY
        ply_path = output_dir / "dense_points.ply"
        _write_ply(ply_path, merged_points, merged_colors)
        logger.info("Dense point cloud: %d points -> %s", len(merged_points), ply_path)

        # Also write a subset into points3D.bin for COLMAP compatibility
        # Use up to 100k points for the COLMAP sparse model
        max_colmap_pts = 100_000
        if len(merged_points) > max_colmap_pts:
            rng = np.random.default_rng(123)
            c_indices = rng.choice(len(merged_points), size=max_colmap_pts, replace=False)
            c_indices.sort()
            colmap_pts = merged_points[c_indices]
            colmap_cols = merged_colors[c_indices]
        else:
            colmap_pts = merged_points
            colmap_cols = merged_colors

        _write_points3d_binary(colmap_dir / "points3D.bin", colmap_pts, colmap_cols)
        logger.info("Wrote points3D.bin with %d points", len(colmap_pts))
        num_total_points = len(merged_points)
    else:
        # Write empty points3D.bin
        _write_points3d_binary(
            colmap_dir / "points3D.bin",
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
        )
        num_total_points = 0
        logger.warning("No points generated — all frames had invalid poses or depth")

    # Also save depth maps that are already metric-scale (DA3 produces metric depth)
    # into a depth_aligned directory so stage 13 (training) can find them
    aligned_dir = output_dir / "depth_aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n_frames):
        stem = Path(all_frame_names[idx]).stem
        src = depth_dir / f"{stem}.npy"
        dst = aligned_dir / f"{stem}.npy"
        if src.exists() and not dst.exists():
            # DA3 depth is already metric — just copy/link
            depth_data = np.load(str(src))
            # Clamp to reasonable face-capture range
            depth_data = np.clip(depth_data, 0.1, 10.0)
            np.save(str(dst), depth_data)

        # Also copy confidence as the alignment confidence
        conf_src = depth_dir / f"{stem}_conf.npy"
        conf_dst = aligned_dir / f"{stem}_confidence.npy"
        if conf_src.exists() and not conf_dst.exists():
            import shutil
            shutil.copy2(str(conf_src), str(conf_dst))

    logger.info("Copied %d metric depth maps to %s", n_frames, aligned_dir)

    # Compute summary stats
    valid_confs = [c for c, v in zip(all_confs, all_valid_pose) if v]
    mean_conf = float(np.mean([c.mean() for c in valid_confs])) if valid_confs else 0.0

    summary = {
        "num_frames": n_frames,
        "num_valid_poses": n_valid,
        "num_points": num_total_points,
        "mean_confidence": mean_conf,
        "model_name": model_name,
        "process_res": process_res,
        "use_ray_pose": use_ray_pose,
        "colmap_dir": str(colmap_dir),
        "depth_dir": str(depth_dir),
        "aligned_dir": str(aligned_dir),
    }

    logger.info("DA3 unified stage complete: %d frames, %d poses, %d points, mean_conf=%.3f",
                n_frames, n_valid, num_total_points, mean_conf)

    # Clean up model to free VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def da3_provides_enough_views(output_dir: Path, min_views: int = 5) -> bool:
    """Check if DA3 produced enough reliable poses for reconstruction.

    Reads the images.bin written by run_da3_unified and counts the number
    of registered images.

    Args:
        output_dir: The output_dir that was passed to run_da3_unified.
        min_views: Minimum number of posed views required.

    Returns:
        True if at least min_views cameras were registered.
    """
    images_bin = Path(output_dir) / "colmap" / "sparse" / "0" / "images.bin"
    if not images_bin.exists():
        logger.warning("images.bin not found at %s", images_bin)
        return False

    try:
        with open(images_bin, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
        logger.info("DA3 produced %d posed views (min required: %d)", num_images, min_views)
        return num_images >= min_views
    except Exception as e:
        logger.error("Failed to read images.bin: %s", e)
        return False
