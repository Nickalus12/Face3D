"""Dense point cloud generation from aligned depth maps and camera poses."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from utils.colmap_io import (
    get_intrinsics_matrix,
    qvec_to_rotmat,
    read_cameras_binary,
    read_images_binary,
)

logger = logging.getLogger(__name__)


def depth_to_pointcloud(
    depth_map: np.ndarray,
    color_image: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Unproject a depth map to a colored 3D point cloud.

    Uses fully vectorized numpy unprojection (no per-pixel loops) and
    samples RGB colors from the color image at each valid depth pixel.

    Args:
        depth_map: (H, W) float32 aligned metric depth map.
        color_image: (H, W, 3) uint8 BGR or RGB color image.
        intrinsics: (3, 3) camera intrinsics matrix K.
        extrinsics: (4, 4) world-to-camera transform [R|t; 0 0 0 1].

    Returns:
        Open3D PointCloud with colors sampled from color_image.
    """
    H, W = depth_map.shape[:2]

    # Resize color image to match depth if needed
    if color_image.shape[:2] != (H, W):
        color_image = cv2.resize(color_image, (W, H), interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB if needed (OpenCV loads as BGR)
    if len(color_image.shape) == 3 and color_image.shape[2] == 3:
        color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    else:
        color_rgb = color_image

    # Vectorized unprojection: create full pixel grid and unproject all at once
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map.flatten().astype(np.float64)
    x = (u.flatten().astype(np.float64) - cx) * z / fx
    y = (v.flatten().astype(np.float64) - cy) * z / fy

    # Stack into (H*W, 3) then filter valid depths
    points_cam = np.stack([x, y, z], axis=-1)
    valid = z > 0
    points_cam = points_cam[valid]
    colors = color_rgb.reshape(-1, 3)[valid].astype(np.float64) / 255.0

    # Transform to world coordinates: X_world = R^T @ (X_cam - t)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (R.T @ (points_cam - t).T).T

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def _pairwise_colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_radii: List[float] = None,
    max_iter_per_scale: int = 50,
) -> np.ndarray:
    """Multi-scale colored ICP between two point clouds.

    Runs Open3D's ``registration_colored_icp`` at progressively finer
    voxel scales for robust alignment that leverages both geometry and
    color information.

    Args:
        source: Source point cloud (will be transformed).
        target: Target / reference point cloud.
        voxel_radii: Voxel radii from coarse to fine.
        max_iter_per_scale: Maximum ICP iterations per scale level.

    Returns:
        (4, 4) rigid transformation matrix that aligns *source* to *target*.
    """
    if voxel_radii is None:
        voxel_radii = [0.04, 0.02, 0.01]

    current_transform = np.eye(4)

    for radius in voxel_radii:
        # Downsample both clouds at this scale
        src_down = source.voxel_down_sample(radius)
        tgt_down = target.voxel_down_sample(radius)

        # Normals are required for colored ICP
        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )
        tgt_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )

        result = o3d.pipelines.registration.registration_colored_icp(
            src_down,
            tgt_down,
            radius,
            current_transform,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=max_iter_per_scale,
            ),
        )
        current_transform = result.transformation

    return current_transform


def merge_pointclouds(
    pointclouds: List[o3d.geometry.PointCloud],
    voxel_size: float = 0.005,
    use_colored_icp: bool = True,
    icp_voxel_radii: List[float] = None,
) -> o3d.geometry.PointCloud:
    """Merge multiple point clouds with optional multi-scale colored ICP and voxel downsampling.

    When *use_colored_icp* is True, each successive point cloud is
    refined against the running merged cloud using multi-scale colored
    ICP before concatenation. This significantly improves alignment
    compared to naive concatenation+downsampling.

    Args:
        pointclouds: List of Open3D point clouds to merge.
        voxel_size: Voxel size for final downsampling (in world units).
        use_colored_icp: Whether to refine alignment with colored ICP.
        icp_voxel_radii: Voxel radii for multi-scale ICP (coarse-to-fine).

    Returns:
        Merged and downsampled Open3D PointCloud.
    """
    if not pointclouds:
        return o3d.geometry.PointCloud()

    if icp_voxel_radii is None:
        icp_voxel_radii = [0.04, 0.02, 0.01]

    merged = o3d.geometry.PointCloud(pointclouds[0])

    for i, pcd in enumerate(pointclouds[1:], start=1):
        if use_colored_icp and len(merged.points) > 0 and len(pcd.points) > 0:
            try:
                transform = _pairwise_colored_icp(
                    pcd, merged, voxel_radii=icp_voxel_radii
                )
                pcd_aligned = o3d.geometry.PointCloud(pcd)
                pcd_aligned.transform(transform)
                merged += pcd_aligned
            except Exception as e:
                logger.warning(
                    "Colored ICP failed for cloud %d, falling back to direct merge: %s",
                    i, e,
                )
                merged += pcd
        else:
            merged += pcd

    logger.info("Merged %d clouds: %d total points", len(pointclouds), len(merged.points))

    if voxel_size > 0:
        merged = merged.voxel_down_sample(voxel_size=voxel_size)
        logger.info("After voxel downsampling (%.4f): %d points", voxel_size, len(merged.points))

    return merged


def generate_dense_pointcloud(
    depth_dir: Path,
    frames_dir: Path,
    colmap_model_dir: Path,
    output_path: Path,
    statistical_outlier_nb: int = 20,
    statistical_outlier_std: float = 2.0,
    radius_outlier_nb: int = 16,
    radius_outlier_radius: float = 0.05,
    voxel_size: float = 0.005,
    use_colored_icp: bool = True,
    icp_voxel_radii: List[float] = None,
    normal_radius: float = 0.1,
    normal_max_nn: int = 30,
    normal_orient_k: int = 15,
) -> o3d.geometry.PointCloud:
    """Full pipeline: generate a dense point cloud from aligned depth maps.

    Loads all aligned depth maps, corresponding color frames, and COLMAP
    camera poses. Generates per-frame point clouds, merges them with
    optional multi-scale colored ICP refinement, filters outliers with
    both statistical and radius removal, estimates and orients normals,
    and saves as .ply.

    Args:
        depth_dir: Directory with aligned .npy depth maps.
        frames_dir: Directory with color images (matching stems).
        colmap_model_dir: Directory with cameras.bin, images.bin.
        output_path: Output .ply file path.
        statistical_outlier_nb: Number of neighbors for statistical outlier removal.
        statistical_outlier_std: Std ratio threshold for statistical outlier removal.
        radius_outlier_nb: Minimum neighbors within radius for radius outlier removal.
            Set to 0 to disable.
        radius_outlier_radius: Search radius for radius outlier removal.
        voxel_size: Voxel size for downsampling.
        use_colored_icp: Use multi-scale colored ICP during merge.
        icp_voxel_radii: Voxel radii for multi-scale ICP (default [0.04, 0.02, 0.01]).
        normal_radius: Search radius for normal estimation.
        normal_max_nn: Maximum neighbors for normal estimation.
        normal_orient_k: k for consistent normal orientation via tangent plane.

    Returns:
        Final Open3D PointCloud with estimated normals.
    """
    from tqdm import tqdm

    depth_dir = Path(depth_dir)
    frames_dir = Path(frames_dir)
    colmap_model_dir = Path(colmap_model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if icp_voxel_radii is None:
        icp_voxel_radii = [0.04, 0.02, 0.01]

    # Load COLMAP model
    logger.info("Loading COLMAP model from %s", colmap_model_dir)
    cameras = read_cameras_binary(colmap_model_dir / "cameras.bin")
    images = read_images_binary(colmap_model_dir / "images.bin")

    # Common image extensions to search for
    image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

    pointclouds = []
    skipped = 0

    for image in tqdm(images.values(), desc="Generating point clouds"):
        stem = Path(image.name).stem

        # Find aligned depth map
        depth_path = depth_dir / f"{stem}.npy"
        if not depth_path.exists():
            skipped += 1
            continue

        # Find color frame
        color_path = None
        for ext in image_extensions:
            candidate = frames_dir / f"{stem}{ext}"
            if candidate.exists():
                color_path = candidate
                break
        # Also try the original name from COLMAP
        if color_path is None:
            candidate = frames_dir / image.name
            if candidate.exists():
                color_path = candidate

        if color_path is None:
            logger.debug("No color frame for %s, skipping", stem)
            skipped += 1
            continue

        # Load data
        depth_map = np.load(str(depth_path))
        color_image = cv2.imread(str(color_path))
        if color_image is None:
            logger.warning("Failed to read color image: %s", color_path)
            skipped += 1
            continue

        # Build camera matrices
        camera = cameras[image.camera_id]
        K = get_intrinsics_matrix(camera)
        R = qvec_to_rotmat(image.qvec)
        t = image.tvec

        # Build 4x4 extrinsics matrix
        extrinsics = np.eye(4, dtype=np.float64)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t

        # Generate point cloud for this frame
        pcd = depth_to_pointcloud(depth_map, color_image, K, extrinsics)
        if len(pcd.points) > 0:
            pointclouds.append(pcd)

    logger.info(
        "Generated %d frame point clouds (%d skipped)",
        len(pointclouds),
        skipped,
    )

    if not pointclouds:
        logger.error("No point clouds generated, cannot create output")
        return o3d.geometry.PointCloud()

    # Merge all frame point clouds (with optional colored ICP refinement)
    merged = merge_pointclouds(
        pointclouds,
        voxel_size=voxel_size,
        use_colored_icp=use_colored_icp,
        icp_voxel_radii=icp_voxel_radii,
    )

    # Statistical outlier removal
    if statistical_outlier_nb > 0 and len(merged.points) > statistical_outlier_nb:
        logger.info(
            "Removing statistical outliers (nb=%d, std=%.1f)",
            statistical_outlier_nb,
            statistical_outlier_std,
        )
        cleaned, inlier_idx = merged.remove_statistical_outlier(
            nb_neighbors=statistical_outlier_nb,
            std_ratio=statistical_outlier_std,
        )
        removed = len(merged.points) - len(cleaned.points)
        logger.info("Removed %d statistical outlier points (%.1f%%)", removed, 100.0 * removed / len(merged.points))
        merged = cleaned

    # Radius outlier removal -- catches isolated clusters that statistical removal misses
    if radius_outlier_nb > 0 and len(merged.points) > radius_outlier_nb:
        logger.info(
            "Removing radius outliers (nb=%d, radius=%.3f)",
            radius_outlier_nb,
            radius_outlier_radius,
        )
        cleaned, inlier_idx = merged.remove_radius_outlier(
            nb_points=radius_outlier_nb,
            radius=radius_outlier_radius,
        )
        removed = len(merged.points) - len(cleaned.points)
        logger.info("Removed %d radius outlier points (%.1f%%)", removed, 100.0 * removed / len(merged.points))
        merged = cleaned

    # Estimate normals on the final merged cloud
    if len(merged.points) > 0:
        logger.info(
            "Estimating normals (radius=%.3f, max_nn=%d)",
            normal_radius,
            normal_max_nn,
        )
        merged.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
        )
        # Orient normals consistently using tangent plane propagation
        try:
            merged.orient_normals_consistent_tangent_plane(k=normal_orient_k)
            logger.info("Normals oriented consistently (k=%d)", normal_orient_k)
        except Exception as e:
            logger.warning("Failed to orient normals consistently: %s", e)

    # Save
    o3d.io.write_point_cloud(str(output_path), merged)
    logger.info("Saved dense point cloud to %s (%d points)", output_path, len(merged.points))

    return merged
