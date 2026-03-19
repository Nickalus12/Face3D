"""Pytest configuration and shared fixtures for Face3D pipeline tests.

Architecture:
    - sys.path setup so ``src/`` modules are importable with absolute imports
    - Synthetic data generators with deterministic seeds for reproducibility
    - COLMAP binary model factory for integration tests
    - GPU / model availability markers with auto-skip
    - Validation helpers for property-based invariant checking
    - Snapshot / golden-file comparison utilities for regression detection

Design principles:
    - Every fixture uses a fixed seed for deterministic reproducibility
    - All temp data is scoped to ``tmp_path`` (pytest auto-cleanup)
    - Heavy fixtures (FLAME model, depth estimator) are session-scoped
    - Fixtures compose: ``sample_colmap_model`` can feed into ``sample_gaussians``
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup -- ensure src/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Expose PROJECT_ROOT to all tests
MODELS_DIR = PROJECT_ROOT / "Models"
FLAME_DIR = MODELS_DIR / "Flame"


# ---------------------------------------------------------------------------
# Auto-register custom markers so ``pytest --strict-markers`` works
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers for the Face3D test suite."""
    config.addinivalue_line("markers", "slow: marks tests as slow (>5s)")
    config.addinivalue_line("markers", "cuda: marks tests requiring CUDA GPU")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
    config.addinivalue_line("markers", "integration: marks integration tests spanning multiple stages")
    config.addinivalue_line("markers", "property_based: marks property-based / fuzzing tests")
    config.addinivalue_line("markers", "regression: marks golden-file / snapshot regression tests")


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _torch_cuda_available() -> bool:
    """Check CUDA availability without importing torch at module level."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _flame_model_path() -> Optional[Path]:
    """Return the first available FLAME model path, or None."""
    candidates = [
        FLAME_DIR / "flame2023_Open.pkl",
        FLAME_DIR / "FLAME2023" / "flame2023.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Session-scoped fixtures (loaded once per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root():
    """Absolute path to the Face3D project root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def flame_model_path():
    """Path to FLAME model, or skip if not found."""
    path = _flame_model_path()
    if path is None:
        pytest.skip("FLAME model file not found in Models/Flame/")
    return path


# ---------------------------------------------------------------------------
# Test data factory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_session_dir(tmp_path):
    """Create a temporary session directory tree mirroring the pipeline layout.

    Returns the session root (``tmp_path / "session"``).
    """
    session = tmp_path / "session"
    for sub in (
        "frames",
        "frames_srgb",
        "depth",
        "depth_aligned",
        "colmap/sparse/0",
        "landmarks",
        "face_masks",
        "flame",
    ):
        (session / sub).mkdir(parents=True, exist_ok=True)
    return session


@pytest.fixture
def sample_frames(tmp_path):
    """Generate 5 synthetic 640x480 BGR test frames with a face-like ellipse.

    Each frame has slight colour variation so blur / exposure checks are
    meaningful.  Frames are saved as PNG files in ``tmp_path / "frames"``.

    Returns a list of :class:`pathlib.Path` to the saved frames.
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    rng = np.random.default_rng(42)

    for i in range(5):
        # Light grey background with per-frame noise
        frame = rng.integers(100, 160, size=(480, 640, 3), dtype=np.uint8)

        # Draw a face-like skin-toned ellipse in the centre
        centre = (320, 240)
        axes = (110, 140)
        skin_bgr = (120 + i * 10, 160 + i * 5, 200 + i * 3)
        cv2.ellipse(frame, centre, axes, 0, 0, 360, skin_bgr, -1)

        # Eyes (two small dark circles)
        cv2.circle(frame, (285, 210), 12, (40, 40, 40), -1)
        cv2.circle(frame, (355, 210), 12, (40, 40, 40), -1)

        # Mouth (small ellipse)
        cv2.ellipse(frame, (320, 290), (30, 10), 0, 0, 360, (80, 80, 120), -1)

        path = frames_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)

    return paths


@pytest.fixture
def sample_colmap_model(tmp_path):
    """Create a minimal COLMAP binary model with 5 cameras and 100 3-D points.

    The cameras form a half-circle arc (valid trajectory geometry) so that
    tests can validate camera-pose consistency and baseline diversity.

    Writes ``cameras.bin``, ``images.bin``, and ``points3D.bin`` into
    ``tmp_path / "colmap" / "sparse" / "0"``.

    Returns the model directory path.
    """
    model_dir = tmp_path / "colmap" / "sparse" / "0"
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    num_cameras = 5
    num_points = 100

    # --- cameras.bin (PINHOLE model, id=1, 4 params: fx, fy, cx, cy) ---
    cameras_path = model_dir / "cameras.bin"
    with open(cameras_path, "wb") as f:
        f.write(struct.pack("<Q", num_cameras))
        for cam_id in range(1, num_cameras + 1):
            model_id = 1  # PINHOLE
            width, height = 640, 480
            fx, fy, cx, cy = 525.0, 525.0, 320.0, 240.0
            f.write(struct.pack("<IiQQ", cam_id, model_id, width, height))
            f.write(struct.pack("<4d", fx, fy, cx, cy))

    # --- images.bin (cameras on a semicircular arc for geometric validity) ---
    images_path = model_dir / "images.bin"
    with open(images_path, "wb") as f:
        f.write(struct.pack("<Q", num_cameras))
        for img_id in range(1, num_cameras + 1):
            # Place cameras on a semicircular arc looking at the origin
            angle = math.pi * (img_id - 1) / (num_cameras - 1)
            cam_pos = np.array([
                2.0 * math.cos(angle),
                0.0,
                2.0 * math.sin(angle),
            ])
            # Look-at rotation (camera looks toward origin)
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right) + 1e-12
            up = np.cross(right, forward)
            R = np.stack([right, -up, forward], axis=0)  # world-to-camera
            tvec = -R @ cam_pos

            # Convert R to quaternion (w, x, y, z)
            from utils.colmap_io import rotmat_to_qvec
            qvec = rotmat_to_qvec(R)
            if qvec[0] < 0:
                qvec = -qvec

            f.write(struct.pack("<I", img_id))
            f.write(struct.pack("<4d", *qvec))
            f.write(struct.pack("<3d", *tvec))
            f.write(struct.pack("<I", img_id))  # camera_id = img_id

            name = f"frame_{img_id - 1:06d}.png"
            f.write(name.encode("utf-8"))
            f.write(b"\x00")

            # 2D keypoints
            num_pts_2d = 10
            f.write(struct.pack("<Q", num_pts_2d))
            for j in range(num_pts_2d):
                x2d = rng.uniform(0, 640)
                y2d = rng.uniform(0, 480)
                pt3d_id = int(rng.integers(1, num_points + 1))
                f.write(struct.pack("<2d", x2d, y2d))
                f.write(struct.pack("<q", pt3d_id))

    # --- points3D.bin (points clustered near origin = face region) ---
    points_path = model_dir / "points3D.bin"
    with open(points_path, "wb") as f:
        f.write(struct.pack("<Q", num_points))
        for pt_id in range(1, num_points + 1):
            # Cluster points in a face-sized region (~25cm around origin)
            xyz = rng.normal(0, 0.12, size=3)
            rgb = rng.integers(100, 220, size=3, dtype=np.uint8)
            error = rng.uniform(0.1, 2.0)

            f.write(struct.pack("<Q", pt_id))
            f.write(struct.pack("<3d", *xyz))
            f.write(struct.pack("<3B", *rgb))
            f.write(struct.pack("<d", error))

            track_len = 2
            f.write(struct.pack("<Q", track_len))
            for _ in range(track_len):
                img_id_ref = int(rng.integers(1, num_cameras + 1))
                pt2d_idx = int(rng.integers(0, 10))
                f.write(struct.pack("<II", img_id_ref, pt2d_idx))

    return model_dir


@pytest.fixture
def sample_depth_maps(tmp_path):
    """Generate 5 random depth maps as .npy files.

    Values are positive (0.5--5.0 m) to match realistic monocular depth.
    """
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(99)
    paths = []
    for i in range(5):
        depth = rng.uniform(0.5, 5.0, size=(480, 640)).astype(np.float32)
        path = depth_dir / f"frame_{i:06d}.npy"
        np.save(str(path), depth)
        paths.append(path)

    return paths


@pytest.fixture
def sample_gaussians():
    """Create a small GaussianModel (50 Gaussians) for quick tests.

    All parameters are valid: unit quaternions, finite positions, log-scales.
    """
    import torch
    from splatting.initializer import GaussianModel, SH_COEFFS

    rng = np.random.default_rng(77)
    n = 50

    positions = rng.normal(0, 0.1, size=(n, 3)).astype(np.float32)
    colors_sh = np.zeros((n, SH_COEFFS, 3), dtype=np.float32)
    colors_sh[:, 0, :] = rng.uniform(0.3, 0.8, size=(n, 3))
    scales = rng.uniform(-5, -2, size=(n, 3)).astype(np.float32)  # log-scale

    # Valid unit quaternions
    quats_raw = rng.normal(size=(n, 4)).astype(np.float32)
    quats = quats_raw / np.linalg.norm(quats_raw, axis=1, keepdims=True)

    # Logit opacities (sigmoid -> ~0.5--0.9)
    opacities = rng.uniform(0.0, 2.0, size=(n, 1)).astype(np.float32)

    return GaussianModel(
        positions=torch.from_numpy(positions),
        colors_sh=torch.from_numpy(colors_sh),
        scales=torch.from_numpy(scales),
        rotations=torch.from_numpy(quats),
        opacities=torch.from_numpy(opacities),
    )


@pytest.fixture
def sample_imu_data():
    """Generate synthetic 100 Hz IMU data (1 second, gentle orbit)."""
    rng = np.random.default_rng(55)
    n = 100
    timestamps = np.linspace(0.0, 1.0, n)

    # Gentle orbit: gravity + small rotation
    accel = np.zeros((n, 3), dtype=np.float64)
    accel[:, 2] = -9.81  # gravity in z
    accel += rng.normal(0, 0.05, size=(n, 3))

    gyro = np.zeros((n, 3), dtype=np.float64)
    gyro[:, 1] = 0.5  # constant yaw rate ~30 deg/s
    gyro += rng.normal(0, 0.01, size=(n, 3))

    return {
        "timestamps": timestamps,
        "accel_xyz": accel,
        "gyro_xyz": gyro,
    }


@pytest.fixture
def synthetic_video(tmp_path):
    """Create a tiny synthetic .avi video (20 frames at 10 fps).

    Returns the path to the video file.
    """
    video_path = tmp_path / "test_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    rng = np.random.default_rng(7)
    for _ in range(20):
        frame = rng.integers(60, 200, size=(240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


# ---------------------------------------------------------------------------
# CUDA skip fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cuda_available():
    """Fixture that skips the test if no CUDA GPU is available."""
    if not _torch_cuda_available():
        pytest.skip("CUDA GPU not available")


# ---------------------------------------------------------------------------
# Validation helpers (importable by all test modules)
# ---------------------------------------------------------------------------

class Invariants:
    """Reusable invariant checks for Face3D data structures.

    Usage in tests::

        Invariants.assert_valid_quaternions(qvecs)
        Invariants.assert_valid_depth_map(depth)
    """

    @staticmethod
    def assert_valid_quaternions(qvecs: np.ndarray, atol: float = 1e-5):
        """Assert quaternions are unit-length (w, x, y, z)."""
        assert qvecs.ndim == 2 and qvecs.shape[1] == 4
        norms = np.linalg.norm(qvecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=atol,
                                   err_msg="Quaternions are not unit-length")

    @staticmethod
    def assert_valid_rotation_matrix(R: np.ndarray, atol: float = 1e-6):
        """Assert R is a proper rotation (det=1, R^T R = I)."""
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=atol,
                                   err_msg="Rotation matrix det != 1")
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=atol,
                                   err_msg="Rotation matrix is not orthogonal")

    @staticmethod
    def assert_valid_depth_map(depth: np.ndarray):
        """Assert depth is 2-D, float, all positive, no NaN/Inf."""
        assert depth.ndim == 2, f"Depth map should be 2-D, got {depth.ndim}-D"
        assert np.issubdtype(depth.dtype, np.floating), f"Expected float dtype, got {depth.dtype}"
        assert np.all(np.isfinite(depth)), "Depth contains NaN or Inf"
        assert np.all(depth >= 0), "Depth contains negative values"

    @staticmethod
    def assert_valid_gaussians(model):
        """Assert Gaussian model parameters satisfy physical invariants."""
        import torch

        pos = model.positions
        assert torch.all(torch.isfinite(pos)), "Gaussian positions contain NaN/Inf"

        # Quaternions should be roughly unit-length
        qnorms = torch.norm(model.rotations, dim=1)
        assert torch.allclose(qnorms, torch.ones_like(qnorms), atol=0.01), \
            f"Quaternion norms deviate from 1: min={qnorms.min():.4f}, max={qnorms.max():.4f}"

        # Opacities should be finite (logit space, any value is valid)
        assert torch.all(torch.isfinite(model.opacities)), "Opacities contain NaN/Inf"

        # Scales should be finite (log-space)
        assert torch.all(torch.isfinite(model.scales)), "Scales contain NaN/Inf"

    @staticmethod
    def assert_face_scale(points: np.ndarray, expected_diameter_m: float = 0.25, tolerance: float = 5.0):
        """Assert reconstructed points span a human-face-scale region.

        Args:
            points: (N, 3) point positions in metres.
            expected_diameter_m: Expected face diameter (~25 cm).
            tolerance: Multiplicative tolerance factor.
        """
        extent = points.max(axis=0) - points.min(axis=0)
        diameter = np.linalg.norm(extent)
        assert diameter < expected_diameter_m * tolerance, \
            f"Point cloud diameter {diameter:.3f}m >> expected ~{expected_diameter_m}m"
        assert diameter > expected_diameter_m / tolerance, \
            f"Point cloud diameter {diameter:.3f}m << expected ~{expected_diameter_m}m"

    @staticmethod
    def assert_valid_ply(path: Path):
        """Assert a PLY file has a valid header and non-zero vertex count."""
        assert path.exists(), f"PLY file does not exist: {path}"
        assert path.stat().st_size > 100, f"PLY file is suspiciously small: {path.stat().st_size} bytes"
        with open(path, "rb") as f:
            header = b""
            while b"end_header" not in header:
                chunk = f.read(1)
                if not chunk:
                    raise AssertionError("PLY file missing end_header")
                header += chunk
        header_str = header.decode("ascii", errors="replace")
        assert "ply" in header_str.lower(), "File does not start with PLY magic"
        assert "element vertex" in header_str, "PLY missing vertex element"


# ---------------------------------------------------------------------------
# Numerical comparison helpers
# ---------------------------------------------------------------------------

def arrays_close(a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if two numpy arrays are element-wise close (for snapshot tests)."""
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, rtol=rtol, atol=atol)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Both images should be uint8 [0, 255] or float32 [0, 1].
    """
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute simplified SSIM between two grayscale images."""
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())
