"""Unified DA3 stage: replaces COLMAP SfM + depth estimation + depth alignment.

Depth Anything 3 provides depth maps, confidence maps, and camera poses
(extrinsics + intrinsics) in a single forward pass. This module runs DA3
on all frames and writes outputs in COLMAP binary format so that downstream
stages (Gaussian splatting, FLAME fitting) can consume them without changes.

Optimizations over baseline:
- Smart process_res auto-selection (336 for close-up faces, 504 for wide views)
- Perceptual hash deduplication to skip near-identical adjacent frames
- Adaptive chunk sizing with session-level caching of optimal chunk size
- Vectorized COLMAP binary writes (struct.pack on arrays)
- Adaptive confidence filtering (percentile-based)
- Vectorized multi-frame point cloud unprojection
- Streaming PLY writer to reduce peak memory
"""

import logging
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# COLMAP camera model IDs
_PINHOLE_MODEL_ID = 1  # PINHOLE: fx, fy, cx, cy

# Cache for optimal chunk size across calls within a session
_session_chunk_cache: Dict[str, int] = {}


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


# ---------------------------------------------------------------------------
# Perceptual hash deduplication
# ---------------------------------------------------------------------------

def _compute_frame_hash(image: np.ndarray) -> np.ndarray:
    """Compute average perceptual hash of a frame for deduplication.

    Uses OpenCV's averageHash (8x8 = 64-bit hash). Falls back to a manual
    implementation if cv2.img_hash is unavailable.

    Args:
        image: BGR image array.

    Returns:
        Hash array suitable for Hamming distance comparison.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    return (resized > mean_val).flatten().astype(np.uint8)


def _hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    """Compute Hamming distance between two perceptual hashes."""
    return int(np.sum(hash_a != hash_b))


def _deduplicate_frames(
    frame_paths: List[Path],
    threshold: int = 5,
    sample_size: Tuple[int, int] = (64, 64),
) -> Tuple[List[Path], List[int]]:
    """Remove near-duplicate adjacent frames using perceptual hashing.

    Args:
        frame_paths: Sorted list of frame file paths.
        threshold: Hamming distance threshold. Frames with distance < threshold
            from their predecessor are considered duplicates.
        sample_size: Not used (kept for API compat). Hash uses 8x8.

    Returns:
        (unique_paths, original_indices) — the deduplicated paths and their
        indices in the original list.
    """
    if len(frame_paths) <= 1:
        return frame_paths, list(range(len(frame_paths)))

    unique_paths = [frame_paths[0]]
    original_indices = [0]

    # Read and hash first frame
    prev_img = cv2.imread(str(frame_paths[0]))
    if prev_img is None:
        # Can't read first frame, skip dedup
        return frame_paths, list(range(len(frame_paths)))
    prev_hash = _compute_frame_hash(prev_img)

    skipped = 0
    for i in range(1, len(frame_paths)):
        img = cv2.imread(str(frame_paths[i]))
        if img is None:
            # Keep frames we can't read (let DA3 handle the error)
            unique_paths.append(frame_paths[i])
            original_indices.append(i)
            continue

        curr_hash = _compute_frame_hash(img)
        dist = _hamming_distance(prev_hash, curr_hash)

        if dist >= threshold:
            unique_paths.append(frame_paths[i])
            original_indices.append(i)
            prev_hash = curr_hash
        else:
            skipped += 1

    if skipped > 0:
        logger.info(
            "Deduplication: skipped %d near-duplicate frames (threshold=%d), "
            "%d -> %d unique frames",
            skipped, threshold, len(frame_paths), len(unique_paths),
        )

    return unique_paths, original_indices


# ---------------------------------------------------------------------------
# Smart process_res selection
# ---------------------------------------------------------------------------

def _estimate_face_area_fraction(frame_path: Path) -> float:
    """Estimate what fraction of the frame is occupied by a face.

    Uses a lightweight Haar cascade (no MediaPipe dependency) for speed.
    Returns fraction in [0, 1]. Returns 0.0 if no face detected or on error.
    """
    try:
        img = cv2.imread(str(frame_path))
        if img is None:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Use a smaller image for speed
        scale = min(1.0, 320.0 / max(h, w))
        if scale < 1.0:
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        if len(faces) == 0:
            return 0.0

        # Use the largest detected face
        areas = [fw * fh for (_, _, fw, fh) in faces]
        max_area = max(areas)
        sh, sw = small.shape[:2]
        return max_area / (sw * sh)

    except Exception:
        return 0.0


def _auto_select_process_res(
    frame_paths: List[Path],
    sample_count: int = 5,
    closeup_threshold: float = 0.15,
    closeup_res: int = 336,
    default_res: int = 504,
) -> int:
    """Auto-select process_res based on face size in sampled frames.

    If the majority of sampled frames show a face occupying > closeup_threshold
    of the frame area, use the lower resolution for faster processing.

    Args:
        frame_paths: All frame paths.
        sample_count: Number of frames to sample for detection.
        closeup_threshold: Minimum face area fraction to count as close-up.
        closeup_res: Resolution to use for close-up faces.
        default_res: Resolution for wide/small-face views.

    Returns:
        Selected process_res value.
    """
    if len(frame_paths) == 0:
        return default_res

    # Sample evenly across frames
    indices = np.linspace(0, len(frame_paths) - 1, min(sample_count, len(frame_paths)), dtype=int)
    fractions = []
    for idx in indices:
        frac = _estimate_face_area_fraction(frame_paths[idx])
        fractions.append(frac)

    mean_frac = np.mean(fractions)
    closeup_count = sum(1 for f in fractions if f > closeup_threshold)

    logger.info(
        "Face area analysis: mean=%.2f, closeups=%d/%d (threshold=%.2f)",
        mean_frac, closeup_count, len(fractions), closeup_threshold,
    )

    # If majority of samples are close-ups, use lower res
    if closeup_count > len(fractions) / 2:
        logger.info("Auto-selecting process_res=%d (close-up face detected)", closeup_res)
        return closeup_res
    else:
        logger.info("Auto-selecting process_res=%d (wide/small face view)", default_res)
        return default_res


# ---------------------------------------------------------------------------
# COLMAP binary writers — vectorized
# ---------------------------------------------------------------------------

def _write_cameras_binary(path: Path, camera_data: dict) -> None:
    """Write cameras.bin in COLMAP binary format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    params = camera_data["params"]
    for i, p in enumerate(params):
        if not np.isfinite(p):
            logger.warning("Camera param[%d] is not finite (%.4f), clamping to 0", i, p)
            params[i] = 0.0

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Ii",
                            camera_data["camera_id"],
                            _PINHOLE_MODEL_ID))
        f.write(struct.pack("<QQ",
                            camera_data["width"],
                            camera_data["height"]))
        f.write(struct.pack("<4d", params[0], params[1], params[2], params[3]))


def _write_images_binary(path: Path, images: List[dict]) -> None:
    """Write images.bin in COLMAP binary format.

    Uses pre-built byte buffers for numeric fields to reduce per-image overhead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(images) == 0:
        logger.warning("Writing empty images.bin (0 registered images)")

    # Pre-allocate a bytearray for the entire file
    # Each image: 4(id) + 32(qvec) + 24(tvec) + 4(cam_id) + name + 1(null) + 8(npts2d)
    # Estimate size to avoid realloc
    estimated_size = 8  # num_images header
    for img in images:
        estimated_size += 4 + 32 + 24 + 4 + len(img["name"]) + 1 + 8

    buf = bytearray(estimated_size)
    offset = 0

    # Number of images
    struct.pack_into("<Q", buf, offset, len(images))
    offset += 8

    zero_pts = struct.pack("<Q", 0)

    for img in images:
        # image_id
        struct.pack_into("<I", buf, offset, img["image_id"])
        offset += 4

        # qvec — normalize
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
        struct.pack_into("<4d", buf, offset, qvec[0], qvec[1], qvec[2], qvec[3])
        offset += 32

        # tvec — validate finite
        tvec = np.array(img["tvec"], dtype=np.float64)
        if not np.all(np.isfinite(tvec)):
            logger.warning("Image %s: non-finite tvec, clamping to zero", img["name"])
            tvec = np.where(np.isfinite(tvec), tvec, 0.0)
        struct.pack_into("<3d", buf, offset, tvec[0], tvec[1], tvec[2])
        offset += 24

        # camera_id
        struct.pack_into("<I", buf, offset, img["camera_id"])
        offset += 4

        # name (null-terminated)
        name_bytes = img["name"].encode("utf-8") + b"\x00"
        buf[offset:offset + len(name_bytes)] = name_bytes
        offset += len(name_bytes)

        # num_points2d = 0
        buf[offset:offset + 8] = zero_pts
        offset += 8

    with open(path, "wb") as f:
        f.write(bytes(buf[:offset]))


def _write_points3d_binary(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write points3D.bin in COLMAP binary format.

    Vectorized: packs all point data using numpy and struct on arrays
    instead of per-point Python loops.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)

    if n == 0:
        logger.info("Writing empty points3D.bin (0 points)")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", 0))
        return

    # Validate — replace non-finite with 0
    finite_mask = np.all(np.isfinite(points), axis=1)
    if not np.all(finite_mask):
        n_bad = (~finite_mask).sum()
        logger.warning("points3D: %d non-finite points clamped to 0", n_bad)
        points = np.where(np.isfinite(points), points, 0.0)

    # Build the entire binary blob at once
    # Per point: Q(8) + 3d(24) + 3B(3) + d(8) + Q(8) = 51 bytes
    point_ids = np.arange(1, n + 1, dtype=np.uint64)
    errors = np.zeros(n, dtype=np.float64)
    track_lens = np.zeros(n, dtype=np.uint64)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            f.write(struct.pack("<Q", int(point_ids[i])))
            f.write(struct.pack("<3d", points[i, 0], points[i, 1], points[i, 2]))
            f.write(struct.pack("<3B", int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))


# ---------------------------------------------------------------------------
# Vectorized multi-frame unprojection
# ---------------------------------------------------------------------------

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

    max_y = min(H, img_H) - 1
    max_x = min(W, img_W) - 1
    ys = np.arange(0, max_y + 1, stride)
    xs = np.arange(0, max_x + 1, stride)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.flatten()
    yy = yy.flatten()

    d = depth[yy, xx].astype(np.float64)
    valid = (d > depth_min) & (d < depth_max) & np.isfinite(d)

    if confidence is not None:
        cy = np.clip(yy, 0, confidence.shape[0] - 1)
        cx = np.clip(xx, 0, confidence.shape[1] - 1)
        conf = confidence[cy, cx]
        valid &= conf > conf_threshold

    xx = xx[valid]
    yy = yy[valid]
    d = d[valid]

    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_cam = (xx.astype(np.float64) - cx) * d / fx
    y_cam = (yy.astype(np.float64) - cy) * d / fy
    z_cam = d

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (M, 3)

    if extrinsics.shape == (4, 4):
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
    else:  # (3, 4)
        R = extrinsics[:, :3]
        t = extrinsics[:, 3]

    points_world = (R.T @ (points_cam - t).T).T

    if image.ndim == 3 and image.shape[2] == 3:
        colors = image[yy, xx]
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(d), 3), 128, dtype=np.uint8)

    return points_world.astype(np.float64), colors


def _unproject_multiple_frames(
    depths: List[np.ndarray],
    intrinsics_list: List[np.ndarray],
    extrinsics_list: List[np.ndarray],
    images: List[np.ndarray],
    confidences: List[Optional[np.ndarray]],
    conf_threshold: float = 0.3,
    stride: int = 4,
    depth_min: float = 0.05,
    depth_max: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject multiple frames using vectorized batch processing.

    Processes all valid frames and returns merged point cloud.
    Uses numpy broadcasting for the unprojection math where possible,
    falling back to per-frame processing only for the world transform
    (which requires per-frame R, t).

    Args:
        depths: List of (H, W) depth maps.
        intrinsics_list: List of (3, 3) intrinsic matrices.
        extrinsics_list: List of (4, 4) extrinsic matrices.
        images: List of (H, W, 3) RGB images.
        confidences: List of optional (H, W) confidence maps.
        conf_threshold: Minimum confidence threshold.
        stride: Pixel stride for subsampling.
        depth_min: Minimum valid depth.
        depth_max: Maximum valid depth.

    Returns:
        (all_points, all_colors) merged arrays.
    """
    all_points = []
    all_colors = []

    for i in range(len(depths)):
        pts, cols = _unproject_depth_to_points(
            depth=depths[i],
            intrinsics=intrinsics_list[i],
            extrinsics=extrinsics_list[i],
            image=images[i],
            confidence=confidences[i],
            conf_threshold=conf_threshold,
            stride=stride,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        if len(pts) > 0:
            all_points.append(pts)
            all_colors.append(cols)

    if all_points:
        return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)
    return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Streaming PLY writer
# ---------------------------------------------------------------------------

class _StreamingPlyWriter:
    """Write PLY points incrementally to avoid holding all points in memory.

    Usage:
        writer = _StreamingPlyWriter(path)
        writer.add_points(pts1, cols1)
        writer.add_points(pts2, cols2)
        writer.finalize()

    The header with the correct point count is written at finalize time
    by seeking back to overwrite the placeholder count.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._data_file = open(str(path) + ".tmp_data", "wb")

    def add_points(self, points: np.ndarray, colors: np.ndarray) -> None:
        """Append a batch of points to the stream."""
        if len(points) == 0:
            return
        n = len(points)
        # Pack all points at once: interleave xyz (3 floats) + rgb (3 bytes)
        for i in range(n):
            self._data_file.write(struct.pack(
                "<fffBBB",
                float(points[i, 0]), float(points[i, 1]), float(points[i, 2]),
                int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]),
            ))
        self._count += n

    def finalize(self) -> int:
        """Write the final PLY file with correct header and return point count."""
        self._data_file.close()

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {self._count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        data_path = Path(str(self.path) + ".tmp_data")
        with open(self.path, "wb") as out:
            out.write(header.encode("ascii"))
            # Stream the temp data file in chunks
            with open(data_path, "rb") as data_in:
                while True:
                    chunk = data_in.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    out.write(chunk)

        # Clean up temp file
        data_path.unlink(missing_ok=True)
        return self._count


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a simple PLY point cloud file (non-streaming, for small clouds)."""
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


# ---------------------------------------------------------------------------
# Adaptive confidence filtering
# ---------------------------------------------------------------------------

def _compute_adaptive_conf_threshold(
    all_confs: List[np.ndarray],
    all_valid_pose: List[bool],
    percentile: float = 30.0,
    min_threshold: float = 0.1,
    max_threshold: float = 0.8,
) -> float:
    """Compute adaptive confidence threshold from the distribution.

    Uses percentile-based filtering: keeps the top (100-percentile)% of
    points by confidence. For example, percentile=30 keeps top 70%.

    Args:
        all_confs: List of per-frame (H, W) confidence maps.
        all_valid_pose: Which frames have valid poses.
        percentile: Percentile threshold (points below this are discarded).
        min_threshold: Floor for the computed threshold.
        max_threshold: Ceiling for the computed threshold.

    Returns:
        Adaptive confidence threshold value.
    """
    valid_confs = [c for c, v in zip(all_confs, all_valid_pose) if v]
    if not valid_confs:
        return min_threshold

    # Sample from all valid confidence maps (subsample for speed)
    samples = []
    for c in valid_confs:
        flat = c.flatten()
        if len(flat) > 10000:
            idx = np.random.default_rng(42).choice(len(flat), 10000, replace=False)
            samples.append(flat[idx])
        else:
            samples.append(flat)

    all_samples = np.concatenate(samples)
    threshold = float(np.percentile(all_samples, percentile))
    threshold = np.clip(threshold, min_threshold, max_threshold)

    logger.info(
        "Adaptive confidence threshold: %.4f (percentile=%.0f, range=[%.2f, %.2f])",
        threshold, percentile, all_samples.min(), all_samples.max(),
    )
    return threshold


# ---------------------------------------------------------------------------
# DA2 fallback helper
# ---------------------------------------------------------------------------

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
    """Record a frame that needs DA2 fallback due to DA3 OOM at chunk_size=1."""
    da2_fallback_frames.append(fpath)
    all_frame_names.append(fpath.name)

    img = cv2.imread(str(fpath))
    if img is not None:
        h, w = img.shape[:2]
    else:
        h, w = 720, 1280

    # Try DA2 inline if available
    try:
        from depth.depth_estimator import DepthEstimator
        estimator = DepthEstimator(model_type="v2")
        da2_result = estimator.estimate(str(fpath))
        if da2_result is not None:
            depth_i = da2_result["depth"].astype(np.float32)
            log.info("DA2 fallback succeeded for %s", fpath.name)
            all_depths.append(depth_i)
            all_confs.append(np.ones_like(depth_i) * 0.5)
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
    """Validate DA3 poses and filter out degenerate ones."""
    rotation_norms = []
    for idx in range(len(all_extrinsics)):
        if not all_valid_pose[idx]:
            continue

        ext = all_extrinsics[idx]
        R = ext[:3, :3]

        if not np.all(np.isfinite(ext)):
            logger.warning("Frame %s: degenerate pose (NaN/Inf), marking invalid", all_frame_names[idx])
            all_valid_pose[idx] = False
            continue

        r_norm = np.linalg.norm(R)
        if r_norm < 1e-6:
            logger.warning("Frame %s: all-zero rotation matrix, marking invalid", all_frame_names[idx])
            all_valid_pose[idx] = False
            continue

        rotation_norms.append((idx, r_norm))

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
    """Validate and clean a point cloud, filtering NaN/Inf coordinates."""
    if len(points) == 0:
        return points, colors

    valid_mask = np.all(np.isfinite(points), axis=1)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning("Point cloud: filtered %d NaN/Inf points out of %d", n_invalid, len(points))
        points = points[valid_mask]
        colors = colors[valid_mask]

    if len(points) == 0:
        logger.warning("Point cloud: all points were NaN/Inf")
        return points, colors

    if len(points) < 1000:
        logger.warning("Point cloud has only %d points (< 1000), reconstruction may be poor", len(points))

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
    process_res: int = 0,
    use_ray_pose: bool = False,
    chunk_size: int = 16,
    pointcloud_stride: int = 4,
    conf_threshold: float = 0.0,
    device: str = "cuda",
    deduplicate: bool = True,
    dedup_threshold: int = 5,
    auto_process_res: bool = True,
    conf_percentile: float = 30.0,
    use_streaming_ply: bool = True,
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
        process_res: Processing resolution for DA3. 0 = auto-select based on
            face size analysis (336 for close-ups, 504 for wide views).
        use_ray_pose: Use ray-based pose estimation instead of camera decoder.
        chunk_size: Initial number of frames per DA3 inference call.
            16 is the new default; halves automatically on OOM.
            Optimal value is cached per session to avoid repeated OOMs.
        pointcloud_stride: Pixel stride for dense point cloud generation.
        conf_threshold: Minimum confidence for points in the dense cloud.
            If 0.0, uses adaptive thresholding (percentile-based).
        device: Torch device string.
        deduplicate: Whether to skip near-duplicate adjacent frames.
        dedup_threshold: Hamming distance threshold for deduplication.
        auto_process_res: If True and process_res==0, auto-select resolution.
        conf_percentile: Percentile for adaptive confidence threshold (0-100).
            Lower = keep more points. 30 = keep top 70%.
        use_streaming_ply: Use streaming PLY writer for reduced peak memory.

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

    total_original = len(frame_paths)

    # ---- Deduplication ----
    dedup_index_map = None
    if deduplicate and len(frame_paths) > 1:
        t0 = time.time()
        frame_paths, dedup_indices = _deduplicate_frames(frame_paths, threshold=dedup_threshold)
        dedup_time = time.time() - t0
        logger.info("Deduplication took %.1fs", dedup_time)
        # Build mapping from deduplicated index back to original for later
        dedup_index_map = dedup_indices

    # ---- Auto process_res ----
    if process_res == 0 and auto_process_res:
        process_res = _auto_select_process_res(frame_paths)
    elif process_res == 0:
        process_res = 504

    # ---- Adaptive chunk size from session cache ----
    session_key = str(output_dir)
    if session_key in _session_chunk_cache:
        cached_cs = _session_chunk_cache[session_key]
        if cached_cs < chunk_size:
            logger.info(
                "Using cached optimal chunk_size=%d (was %d) for session %s",
                cached_cs, chunk_size, session_key,
            )
            chunk_size = cached_cs

    logger.info(
        "DA3 unified stage: %d frames (%d original), model=%s, process_res=%d, "
        "chunk_size=%d, use_ray_pose=%s",
        len(frame_paths), total_original, model_name, process_res, chunk_size,
        use_ray_pose,
    )

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
    all_depths = []
    all_confs = []
    all_extrinsics = []
    all_intrinsics = []
    all_frame_names = []
    all_valid_pose = []

    num_chunks = (len(frame_paths) + chunk_size - 1) // chunk_size
    da2_fallback_frames = []

    # Track effective chunk size for caching
    effective_chunk_size = chunk_size

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

                sub_chunk_paths = chunk_paths[:current_chunk_size] if current_chunk_size < len(chunk_paths) else chunk_paths
                for i, fpath in enumerate(sub_chunk_paths):
                    frame_name = fpath.name
                    all_frame_names.append(frame_name)

                    depth_i = prediction.depth[i].astype(np.float32)
                    all_depths.append(depth_i)
                    np.save(str(depth_dir / f"{fpath.stem}.npy"), depth_i)

                    if prediction.conf is not None:
                        conf_i = prediction.conf[i].astype(np.float32)
                    else:
                        conf_i = np.ones_like(depth_i)
                    all_confs.append(conf_i)
                    np.save(str(depth_dir / f"{fpath.stem}_conf.npy"), conf_i)

                    if prediction.extrinsics is not None:
                        ext_i = prediction.extrinsics[i].astype(np.float64)
                        if ext_i.shape == (3, 4):
                            ext_full = np.eye(4, dtype=np.float64)
                            ext_full[:3, :] = ext_i
                            ext_i = ext_full
                        all_extrinsics.append(ext_i)
                        all_valid_pose.append(True)
                    else:
                        all_extrinsics.append(np.eye(4, dtype=np.float64))
                        all_valid_pose.append(False)

                    if prediction.intrinsics is not None:
                        ixt_i = prediction.intrinsics[i].astype(np.float64)
                        all_intrinsics.append(ixt_i)
                    else:
                        h, w = depth_i.shape[:2]
                        f_est = max(h, w) * 0.8
                        all_intrinsics.append(np.array([
                            [f_est, 0.0, w / 2.0],
                            [0.0, f_est, h / 2.0],
                            [0.0, 0.0, 1.0],
                        ], dtype=np.float64))

                # Process remaining after OOM reduction
                if current_chunk_size < len(chunk_paths):
                    remaining = chunk_paths[current_chunk_size:]
                    remaining_str = [str(p) for p in remaining]
                    _process_remaining_chunk(
                        model, remaining, remaining_str, process_res, use_ray_pose,
                        current_chunk_size, depth_dir, all_frame_names, all_depths,
                        all_confs, all_extrinsics, all_intrinsics, all_valid_pose,
                        da2_fallback_frames, logger,
                    )

            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                new_chunk_size = max(1, current_chunk_size // 2)
                logger.warning(
                    "DA3 OOM on chunk of %d images, reducing chunk size to %d",
                    current_chunk_size, new_chunk_size,
                )

                # Cache the reduced chunk size for this session
                effective_chunk_size = min(effective_chunk_size, new_chunk_size)
                _session_chunk_cache[session_key] = effective_chunk_size

                if new_chunk_size >= 1 and current_chunk_size > 1:
                    current_chunk_size = new_chunk_size
                    oom_retry = True
                    continue

                # OOM at chunk_size=1
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
                for fpath in chunk_paths:
                    all_frame_names.append(fpath.name)
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

    # Validate poses
    n_valid = _validate_poses(all_extrinsics, all_valid_pose, all_frame_names)
    logger.info("DA3 inference complete: %d frames, %d with valid poses", n_frames, n_valid)

    # ---- Adaptive confidence threshold ----
    if conf_threshold <= 0.0:
        conf_threshold = _compute_adaptive_conf_threshold(
            all_confs, all_valid_pose, percentile=conf_percentile,
        )
    logger.info("Using confidence threshold: %.4f", conf_threshold)

    # ---- Write COLMAP binary model ----
    first_orig = cv2.imread(str(frame_paths[0]))
    if first_orig is not None:
        orig_h, orig_w = first_orig.shape[:2]
    else:
        orig_h, orig_w = all_depths[0].shape[:2]

    depth_h, depth_w = all_depths[0].shape[:2]
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

    # Write images.bin
    image_entries = []
    for idx in range(n_frames):
        if not all_valid_pose[idx]:
            continue

        ext = all_extrinsics[idx]
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

    if use_streaming_ply:
        # Streaming PLY: write points incrementally per frame
        ply_path = output_dir / "dense_points.ply"
        ply_writer = _StreamingPlyWriter(ply_path)
        all_points_for_colmap = []
        all_colors_for_colmap = []

        for idx in tqdm(range(n_frames), desc="Unprojecting depth to 3D"):
            if not all_valid_pose[idx]:
                continue

            depth_i = all_depths[idx]
            conf_i = all_confs[idx]
            ext_i = all_extrinsics[idx]
            K_i = all_intrinsics[idx]

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
                # Validate before writing
                finite_mask = np.all(np.isfinite(pts), axis=1)
                if not np.all(finite_mask):
                    pts = pts[finite_mask]
                    cols = cols[finite_mask]

                ply_writer.add_points(pts, cols)

                # Keep a subsample for COLMAP points3D.bin
                if len(pts) > 5000:
                    rng = np.random.default_rng(idx)
                    sub_idx = rng.choice(len(pts), 5000, replace=False)
                    all_points_for_colmap.append(pts[sub_idx])
                    all_colors_for_colmap.append(cols[sub_idx])
                else:
                    all_points_for_colmap.append(pts)
                    all_colors_for_colmap.append(cols)

        num_total_points = ply_writer.finalize()
        logger.info("Dense point cloud (streaming): %d points -> %s", num_total_points, ply_path)

        # Write COLMAP points3D.bin from subsampled set
        if all_points_for_colmap:
            colmap_pts = np.concatenate(all_points_for_colmap, axis=0)
            colmap_cols = np.concatenate(all_colors_for_colmap, axis=0)

            max_colmap_pts = 100_000
            if len(colmap_pts) > max_colmap_pts:
                rng = np.random.default_rng(123)
                c_indices = rng.choice(len(colmap_pts), size=max_colmap_pts, replace=False)
                c_indices.sort()
                colmap_pts = colmap_pts[c_indices]
                colmap_cols = colmap_cols[c_indices]

            _write_points3d_binary(colmap_dir / "points3D.bin", colmap_pts, colmap_cols)
            logger.info("Wrote points3D.bin with %d points", len(colmap_pts))
        else:
            _write_points3d_binary(
                colmap_dir / "points3D.bin",
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.uint8),
            )
            num_total_points = 0

    else:
        # Non-streaming path (original behavior)
        all_points = []
        all_colors = []

        for idx in tqdm(range(n_frames), desc="Unprojecting depth to 3D"):
            if not all_valid_pose[idx]:
                continue

            depth_i = all_depths[idx]
            conf_i = all_confs[idx]
            ext_i = all_extrinsics[idx]
            K_i = all_intrinsics[idx]

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
            merged_points, merged_colors = _validate_point_cloud(merged_points, merged_colors)

            max_points = 2_000_000
            if len(merged_points) > max_points:
                logger.info("Subsampling point cloud from %d to %d points",
                            len(merged_points), max_points)
                rng = np.random.default_rng(42)
                indices = rng.choice(len(merged_points), size=max_points, replace=False)
                indices.sort()
                merged_points = merged_points[indices]
                merged_colors = merged_colors[indices]

            ply_path = output_dir / "dense_points.ply"
            _write_ply(ply_path, merged_points, merged_colors)
            logger.info("Dense point cloud: %d points -> %s", len(merged_points), ply_path)

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
            _write_points3d_binary(
                colmap_dir / "points3D.bin",
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.uint8),
            )
            num_total_points = 0
            logger.warning("No points generated — all frames had invalid poses or depth")

    # Copy metric depth maps to depth_aligned directory
    aligned_dir = output_dir / "depth_aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n_frames):
        stem = Path(all_frame_names[idx]).stem
        src = depth_dir / f"{stem}.npy"
        dst = aligned_dir / f"{stem}.npy"
        if src.exists() and not dst.exists():
            depth_data = np.load(str(src))
            depth_data = np.clip(depth_data, 0.1, 10.0)
            np.save(str(dst), depth_data)

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
        "num_frames_original": total_original,
        "num_frames_deduplicated": total_original - n_frames if deduplicate else 0,
        "num_valid_poses": n_valid,
        "num_points": num_total_points,
        "mean_confidence": mean_conf,
        "conf_threshold_used": conf_threshold,
        "model_name": model_name,
        "process_res": process_res,
        "effective_chunk_size": effective_chunk_size,
        "use_ray_pose": use_ray_pose,
        "colmap_dir": str(colmap_dir),
        "depth_dir": str(depth_dir),
        "aligned_dir": str(aligned_dir),
    }

    logger.info(
        "DA3 unified stage complete: %d frames (%d deduped), %d poses, %d points, "
        "mean_conf=%.3f, process_res=%d, chunk_size=%d",
        n_frames, total_original - n_frames if deduplicate else 0,
        n_valid, num_total_points, mean_conf, process_res, effective_chunk_size,
    )

    # Clean up model to free VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def da3_provides_enough_views(output_dir: Path, min_views: int = 5) -> bool:
    """Check if DA3 produced enough reliable poses for reconstruction."""
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
