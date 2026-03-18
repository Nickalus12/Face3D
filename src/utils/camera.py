"""Camera utilities and Samsung S25 Ultra lens database."""

from typing import Dict

import numpy as np

# Samsung Galaxy S25 Ultra lens database
# Each entry: focal_length_mm, sensor_width_mm, sensor_height_mm, megapixels, fov_degrees
S25_ULTRA_LENSES: Dict[str, Dict] = {
    "wide": {
        "focal_length_mm": 6.3,
        "sensor_width_mm": 8.0,
        "sensor_height_mm": 6.0,
        "resolution": (8160, 6120),
        "megapixels": 200,
        "fov_degrees": 85,
        "aperture": 1.7,
        "equivalent_focal_length_mm": 23,
        "description": "200MP wide main sensor (HP2)",
    },
    "ultrawide": {
        "focal_length_mm": 2.2,
        "sensor_width_mm": 5.6,
        "sensor_height_mm": 4.2,
        "resolution": (4000, 3000),
        "megapixels": 50,
        "fov_degrees": 120,
        "aperture": 1.9,
        "equivalent_focal_length_mm": 13,
        "description": "50MP ultrawide sensor",
    },
    "telephoto_3x": {
        "focal_length_mm": 18.6,
        "sensor_width_mm": 5.6,
        "sensor_height_mm": 4.2,
        "resolution": (4000, 3000),
        "megapixels": 50,
        "fov_degrees": 36,
        "aperture": 2.4,
        "equivalent_focal_length_mm": 67,
        "description": "50MP 3x telephoto sensor",
    },
    "telephoto_5x": {
        "focal_length_mm": 31.0,
        "sensor_width_mm": 5.6,
        "sensor_height_mm": 4.2,
        "resolution": (4000, 3000),
        "megapixels": 50,
        "fov_degrees": 22,
        "aperture": 2.8,
        "equivalent_focal_length_mm": 115,
        "description": "50MP 5x periscope telephoto sensor",
    },
}


def focal_length_pixels(focal_length_mm: float, sensor_width_mm: float, image_width_px: int) -> float:
    """Convert physical focal length to pixel units."""
    return focal_length_mm * image_width_px / sensor_width_mm


def project_points(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project 3D world points to 2D image coordinates.

    Args:
        points_3d: (N, 3) world-space points.
        K: (3, 3) intrinsics matrix.
        R: (3, 3) rotation matrix (world-to-camera).
        t: (3,) translation vector (world-to-camera).

    Returns:
        (N, 2) array of projected 2D pixel coordinates.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()

    # Transform to camera frame: X_cam = R @ X_world + t
    points_cam = (R @ points_3d.T).T + t  # (N, 3)

    # Project: x = K @ X_cam  (homogeneous)
    points_proj = (K @ points_cam.T).T  # (N, 3)

    # Dehomogenize
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d


def unproject_points(
    points_2d: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Unproject 2D image points + depth to 3D world coordinates.

    Args:
        points_2d: (N, 2) pixel coordinates.
        depth: (N,) depth values at each pixel.
        K: (3, 3) intrinsics matrix.
        R: (3, 3) rotation matrix (world-to-camera).
        t: (3,) translation vector (world-to-camera).

    Returns:
        (N, 3) array of 3D world points.
    """
    points_2d = np.asarray(points_2d, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()

    N = points_2d.shape[0]

    # Homogeneous pixel coords
    ones = np.ones((N, 1), dtype=np.float64)
    pixels_h = np.hstack([points_2d, ones])  # (N, 3)

    # Unproject to camera frame (normalized)
    K_inv = np.linalg.inv(K)
    rays_cam = (K_inv @ pixels_h.T).T  # (N, 3)

    # Scale by depth
    points_cam = rays_cam * depth[:, np.newaxis]  # (N, 3)

    # Transform to world frame: X_world = R^T @ (X_cam - t)
    R_inv = R.T
    points_world = (R_inv @ (points_cam - t).T).T  # (N, 3)
    return points_world
