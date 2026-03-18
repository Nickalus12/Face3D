"""Read/write COLMAP binary model files (cameras, images, points3D)."""

import struct
from collections import namedtuple
from pathlib import Path
from typing import Dict

import numpy as np

# COLMAP camera model IDs -> (model_name, num_params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    """Read cameras.bin and return a dict mapping camera_id -> Camera."""
    path = Path(path)
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<IiQQ", f.read(24))
            model_name, num_params = CAMERA_MODELS[model_id]
            params = np.array(
                struct.unpack(f"<{num_params}d", f.read(8 * num_params)),
                dtype=np.float64,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_images_binary(path: Path) -> Dict[int, Image]:
    """Read images.bin and return a dict mapping image_id -> Image."""
    path = Path(path)
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read null-terminated image name
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            name = "".join(name_chars)

            # Read 2D points
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            xys = np.empty((num_points2d, 2), dtype=np.float64)
            point3d_ids = np.empty(num_points2d, dtype=np.int64)
            for j in range(num_points2d):
                x, y = struct.unpack("<2d", f.read(16))
                p3d_id = struct.unpack("<q", f.read(8))[0]
                xys[j] = [x, y]
                point3d_ids[j] = p3d_id

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_points3d_binary(path: Path) -> Dict[int, Point3D]:
    """Read points3D.bin and return a dict mapping point3d_id -> Point3D."""
    path = Path(path)
    points3d = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            rgb = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]

            track_length = struct.unpack("<Q", f.read(8))[0]
            image_ids = np.empty(track_length, dtype=np.int32)
            point2d_idxs = np.empty(track_length, dtype=np.int32)
            for j in range(track_length):
                img_id, pt2d_idx = struct.unpack("<II", f.read(8))
                image_ids[j] = img_id
                point2d_idxs[j] = pt2d_idx

            points3d[point3d_id] = Point3D(
                id=point3d_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2d_idxs,
            )
    return points3d


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)
    return R


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
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
    return np.array([w, x, y, z], dtype=np.float64)


def get_intrinsics_matrix(camera: Camera) -> np.ndarray:
    """Extract 3x3 intrinsics matrix K from a COLMAP Camera.

    Supports SIMPLE_PINHOLE (f, cx, cy), PINHOLE (fx, fy, cx, cy),
    SIMPLE_RADIAL (f, cx, cy, k), RADIAL (f, cx, cy, k1, k2),
    and OPENCV/FULL_OPENCV (fx, fy, cx, cy, ...).
    """
    params = camera.params
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                         "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif camera.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE",
                           "FULL_OPENCV", "THIN_PRISM_FISHEYE", "FOV"):
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K
