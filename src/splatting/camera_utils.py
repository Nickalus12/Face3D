"""Camera utilities for Gaussian splatting: loading, conversion, and generation."""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Camera:
    """Camera representation compatible with gsplat rasterization.

    Stores intrinsic and extrinsic parameters needed to build the
    viewmat (4x4 world-to-camera) and K (3x3 intrinsic) matrices
    that gsplat.rasterization() expects.
    """

    R: np.ndarray              # (3, 3) rotation matrix (world-to-camera)
    T: np.ndarray              # (3,) translation vector (world-to-camera)
    FoVx: float                # horizontal field of view in radians
    FoVy: float                # vertical field of view in radians
    width: int
    height: int
    image_path: str | None = None
    uid: int = 0

    @property
    def fx(self) -> float:
        return self.width / (2.0 * math.tan(self.FoVx / 2.0))

    @property
    def fy(self) -> float:
        return self.height / (2.0 * math.tan(self.FoVy / 2.0))

    def get_viewmat(self, device: torch.device | str = "cuda") -> torch.Tensor:
        """Build a 4x4 world-to-camera view matrix for gsplat."""
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = self.R
        viewmat[:3, 3] = self.T
        return torch.from_numpy(viewmat).to(device)

    def get_K(self, device: torch.device | str = "cuda") -> torch.Tensor:
        """Build a 3x3 intrinsic matrix for gsplat."""
        K = np.array([
            [self.fx, 0.0, self.width / 2.0],
            [0.0, self.fy, self.height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return torch.from_numpy(K).to(device)


def focal_to_fov(focal: float, pixels: int) -> float:
    """Convert focal length in pixels to field of view in radians."""
    return 2.0 * math.atan(pixels / (2.0 * focal))


def load_cameras_from_colmap(colmap_model_dir: Path) -> list[Camera]:
    """Load cameras and images from a COLMAP model directory.

    Reads cameras.bin (or .txt) and images.bin (or .txt) to produce
    a list of Camera objects with correct intrinsic and extrinsic params.

    Args:
        colmap_model_dir: Directory containing cameras.bin/txt and images.bin/txt.

    Returns:
        List of Camera dataclass instances.
    """
    # Load intrinsics
    cameras_bin = colmap_model_dir / "cameras.bin"
    cameras_txt = colmap_model_dir / "cameras.txt"
    if cameras_bin.exists():
        intrinsics = _read_cameras_binary(cameras_bin)
    elif cameras_txt.exists():
        intrinsics = _read_cameras_text(cameras_txt)
    else:
        raise FileNotFoundError(f"No cameras.bin or cameras.txt in {colmap_model_dir}")

    # Load extrinsics (image poses)
    images_bin = colmap_model_dir / "images.bin"
    images_txt = colmap_model_dir / "images.txt"
    if images_bin.exists():
        image_data = _read_images_binary(images_bin)
    elif images_txt.exists():
        image_data = _read_images_text(images_txt)
    else:
        raise FileNotFoundError(f"No images.bin or images.txt in {colmap_model_dir}")

    cameras: list[Camera] = []
    for img_id, (qvec, tvec, cam_id, name) in image_data.items():
        if cam_id not in intrinsics:
            logger.warning("Image %s references unknown camera %d, skipping", name, cam_id)
            continue

        model, width, height, params = intrinsics[cam_id]
        fx, fy, cx, cy = _extract_focal(model, params, width, height)

        R = _qvec_to_rotmat(qvec)
        T = np.array(tvec, dtype=np.float32)

        cam = Camera(
            R=R,
            T=T,
            FoVx=focal_to_fov(fx, width),
            FoVy=focal_to_fov(fy, height),
            width=width,
            height=height,
            image_path=name,
            uid=img_id,
        )
        cameras.append(cam)

    cameras.sort(key=lambda c: c.uid)
    logger.info("Loaded %d cameras from COLMAP model at %s", len(cameras), colmap_model_dir)
    return cameras


def generate_turntable_cameras(
    center: np.ndarray,
    radius: float,
    num_views: int = 60,
    height: float = 0.0,
    fov: float = 0.7854,
    image_width: int = 800,
    image_height: int = 800,
) -> list[Camera]:
    """Generate cameras evenly spaced on a circle for turntable rendering.

    Args:
        center: (3,) center point the cameras look at.
        radius: Distance from center to each camera.
        num_views: Number of views around the circle.
        height: Vertical offset of cameras above center.
        fov: Field of view in radians (used for both x and y).
        image_width: Output image width.
        image_height: Output image height.

    Returns:
        List of Camera instances arranged in a turntable.
    """
    cameras: list[Camera] = []
    center = np.asarray(center, dtype=np.float32)

    for i in range(num_views):
        angle = 2.0 * math.pi * i / num_views
        # Camera position
        cam_pos = center + np.array([
            radius * math.cos(angle),
            height,
            radius * math.sin(angle),
        ], dtype=np.float32)

        # Look-at: camera looks toward center
        forward = center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Up vector (world Y)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # Right vector
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Recompute up
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)

        # World-to-camera rotation: rows are right, -up, forward
        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        T = (-R @ cam_pos).astype(np.float32)

        cameras.append(Camera(
            R=R,
            T=T,
            FoVx=fov,
            FoVy=fov,
            width=image_width,
            height=image_height,
            uid=i,
        ))

    return cameras


# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

# Camera model IDs
_CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}


def _extract_focal(
    model: str, params: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from COLMAP camera model parameters."""
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f = params[0]
        return f, f, params[1], params[2]
    elif model in ("PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
                    "FULL_OPENCV", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"):
        return params[0], params[1], params[2], params[3]
    elif model == "FOV":
        return params[0], params[1], params[2], params[3]
    else:
        # Fallback: assume first param is focal length
        f = params[0]
        return f, f, width / 2.0, height / 2.0


def _read_cameras_binary(path: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    """Read cameras.bin -> {cam_id: (model_name, width, height, params)}."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model_name = _CAMERA_MODELS.get(model_id, "PINHOLE")
            num_params = _NUM_PARAMS.get(model_name, 4)
            params = list(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            cameras[cam_id] = (model_name, width, height, params)
    return cameras


def _read_cameras_text(path: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    """Read cameras.txt -> {cam_id: (model_name, width, height, params)}."""
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = (model_name, width, height, params)
    return cameras


def _read_images_binary(
    path: Path,
) -> dict[int, tuple[list[float], list[float], int, str]]:
    """Read images.bin -> {img_id: (qvec, tvec, camera_id, name)}."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = list(struct.unpack("<dddd", f.read(32)))
            tvec = list(struct.unpack("<ddd", f.read(24)))
            cam_id = struct.unpack("<I", f.read(4))[0]
            # Read image name (null-terminated string)
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes += ch
            name = name_bytes.decode("utf-8")
            # Read 2D points (skip)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # Each point2D: x(double) y(double) point3D_id(long long)
            f.read(num_points2d * 24)
            images[img_id] = (qvec, tvec, cam_id, name)
    return images


def _read_images_text(
    path: Path,
) -> dict[int, tuple[list[float], list[float], int, str]]:
    """Read images.txt -> {img_id: (qvec, tvec, camera_id, name)}."""
    images = {}
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    # images.txt has pairs of lines: metadata line + points2D line
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        img_id = int(parts[0])
        qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
        cam_id = int(parts[8])
        name = parts[9]
        images[img_id] = (qvec, tvec, cam_id, name)
    return images


def _qvec_to_rotmat(qvec: list[float]) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)
    return R
