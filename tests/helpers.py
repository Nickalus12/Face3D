"""Reusable validation helpers and quality metrics for Face3D tests.

Import from test files::

    from tests.helpers import Invariants, compute_psnr, compute_ssim

Or if running from the tests/ directory::

    from helpers import Invariants, compute_psnr, compute_ssim
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


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

        qnorms = torch.norm(model.rotations, dim=1)
        assert torch.allclose(qnorms, torch.ones_like(qnorms), atol=0.01), \
            f"Quaternion norms deviate from 1: min={qnorms.min():.4f}, max={qnorms.max():.4f}"

        assert torch.all(torch.isfinite(model.opacities)), "Opacities contain NaN/Inf"
        assert torch.all(torch.isfinite(model.scales)), "Scales contain NaN/Inf"

    @staticmethod
    def assert_face_scale(points: np.ndarray, expected_diameter_m: float = 0.25, tolerance: float = 5.0):
        """Assert reconstructed points span a human-face-scale region."""
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


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute simplified SSIM between two images."""
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
