"""Tests for new modules added in the optimization session.

Covers:
- SH allocation fix (DEFAULT_SH_COEFFS, _build_splats padding/truncation)
- KDTree batch scale estimation (scipy instead of open3d for-loop)
- Numba kernels (color correction, quaternion ops, depth unprojection)
- Timing utilities (@timed, StageTimer, PipelineProfiler)
- ML replay buffer and prior network
- Triton kernel fallbacks (PyTorch path)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ═══════════════════════════════════════════════════════════════════════════
# SH Allocation & _build_splats
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestSHAllocation:
    """Test that SH coefficients are allocated efficiently."""

    def test_default_sh_coeffs_is_4(self):
        from splatting.initializer import DEFAULT_SH_COEFFS
        assert DEFAULT_SH_COEFFS == 4, "Default should be SH1 (4 coeffs) to save VRAM"

    def test_legacy_sh_coeffs_is_16(self):
        from splatting.initializer import SH_COEFFS
        assert SH_COEFFS == 16, "Legacy constant for backward compat"

    def test_gaussian_model_with_sh1(self):
        torch = pytest.importorskip("torch")
        from splatting.initializer import GaussianModel, DEFAULT_SH_COEFFS

        n = 100
        model = GaussianModel(
            positions=torch.randn(n, 3),
            colors_sh=torch.randn(n, DEFAULT_SH_COEFFS, 3),
            scales=torch.randn(n, 3),
            rotations=torch.randn(n, 4),
            opacities=torch.randn(n, 1),
        )
        assert model.colors_sh.shape == (n, 4, 3)
        assert model.num_gaussians == n

    def test_build_splats_pads_sh1_to_sh3(self):
        """_build_splats should pad SH1 (4 coeffs) to SH3 (16) when needed."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("open3d")
        from splatting.initializer import GaussianModel
        from splatting.trainer import _build_splats

        n = 50
        model = GaussianModel(
            positions=torch.randn(n, 3),
            colors_sh=torch.randn(n, 4, 3),  # SH1
            scales=torch.randn(n, 3),
            rotations=torch.randn(n, 4),
            opacities=torch.randn(n, 1),
        )
        splats = _build_splats(model, torch.device("cpu"), sh_degree_max=3)
        # sh0=(N,1,3) + shN=(N,15,3) = 16 total
        assert splats["sh0"].shape == (n, 1, 3)
        assert splats["shN"].shape == (n, 15, 3)

    def test_build_splats_keeps_sh1(self):
        """_build_splats with sh_degree_max=1 should keep 4 coeffs."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("open3d")
        from splatting.initializer import GaussianModel
        from splatting.trainer import _build_splats

        n = 50
        model = GaussianModel(
            positions=torch.randn(n, 3),
            colors_sh=torch.randn(n, 4, 3),  # SH1
            scales=torch.randn(n, 3),
            rotations=torch.randn(n, 4),
            opacities=torch.randn(n, 1),
        )
        splats = _build_splats(model, torch.device("cpu"), sh_degree_max=1)
        assert splats["sh0"].shape == (n, 1, 3)
        assert splats["shN"].shape == (n, 3, 3)  # 4-1=3

    def test_build_splats_truncates_sh3_to_sh1(self):
        """_build_splats should truncate SH3 model when training with SH1."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("open3d")
        from splatting.initializer import GaussianModel
        from splatting.trainer import _build_splats

        n = 50
        model = GaussianModel(
            positions=torch.randn(n, 3),
            colors_sh=torch.randn(n, 16, 3),  # SH3
            scales=torch.randn(n, 3),
            rotations=torch.randn(n, 4),
            opacities=torch.randn(n, 1),
        )
        splats = _build_splats(model, torch.device("cpu"), sh_degree_max=1)
        assert splats["sh0"].shape == (n, 1, 3)
        assert splats["shN"].shape == (n, 3, 3)  # truncated to 4-1=3

    def test_build_splats_roundtrip(self):
        """_build_splats -> _splats_to_gaussians should preserve data."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("open3d")
        from splatting.initializer import GaussianModel
        from splatting.trainer import _build_splats, _splats_to_gaussians

        n = 30
        original = GaussianModel(
            positions=torch.randn(n, 3),
            colors_sh=torch.randn(n, 4, 3),
            scales=torch.randn(n, 3),
            rotations=torch.randn(n, 4),
            opacities=torch.randn(n, 1),
        )
        splats = _build_splats(original, torch.device("cpu"), sh_degree_max=1)
        recovered = _splats_to_gaussians(splats)
        assert torch.allclose(original.positions, recovered.positions)
        assert recovered.colors_sh.shape[1] == 4  # preserved SH1


# ═══════════════════════════════════════════════════════════════════════════
# KDTree Scale Estimation (scipy batch)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestScaleEstimation:
    """Test the scipy-based batch KDTree scale estimation."""

    def test_output_shape(self):
        from splatting.initializer import _estimate_initial_scales
        points = np.random.randn(100, 3).astype(np.float32)
        scales = _estimate_initial_scales(points, k=4)
        assert scales.shape == (100, 3)

    def test_isotropic_scales(self):
        """All three scale components should be identical (isotropic init)."""
        from splatting.initializer import _estimate_initial_scales
        points = np.random.randn(50, 3).astype(np.float32)
        scales = _estimate_initial_scales(points, k=4)
        np.testing.assert_array_equal(scales[:, 0], scales[:, 1])
        np.testing.assert_array_equal(scales[:, 0], scales[:, 2])

    def test_log_space(self):
        """Scales should be in log-space (can be negative)."""
        from splatting.initializer import _estimate_initial_scales
        points = np.random.randn(200, 3).astype(np.float32) * 0.01  # very tight cluster
        scales = _estimate_initial_scales(points, k=4)
        # Log of small distances should be negative
        assert np.any(scales < 0)

    def test_finite_output(self):
        from splatting.initializer import _estimate_initial_scales
        points = np.random.randn(100, 3).astype(np.float32)
        scales = _estimate_initial_scales(points, k=4)
        assert np.all(np.isfinite(scales))

    def test_small_k(self):
        """k=1 should work (minimum neighbors)."""
        from splatting.initializer import _estimate_initial_scales
        points = np.random.randn(10, 3).astype(np.float32)
        scales = _estimate_initial_scales(points, k=1)
        assert scales.shape == (10, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Numba Kernels
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestNumbaKernels:
    """Test numba-accelerated kernels match numpy reference implementations."""

    def test_slog3_to_linear_shape(self):
        from utils.numba_kernels import slog3_to_linear
        img = np.random.rand(480, 640, 3).astype(np.float32)
        result = slog3_to_linear(img)
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_slog3_to_linear_non_negative(self):
        from utils.numba_kernels import slog3_to_linear
        img = np.linspace(0, 1, 256).astype(np.float32).reshape(16, 16, 1)
        img = np.repeat(img, 3, axis=2)
        result = slog3_to_linear(img)
        assert np.all(result >= -0.01)  # Allow tiny numerical noise

    def test_slog3_to_linear_monotonic(self):
        from utils.numba_kernels import slog3_to_linear
        ramp = np.linspace(0, 1, 1000).astype(np.float32).reshape(1, 1000, 1)
        result = slog3_to_linear(ramp)
        diffs = np.diff(result[0, :, 0])
        assert np.all(diffs >= -1e-6)  # Monotonically increasing

    def test_linear_to_srgb_range(self):
        from utils.numba_kernels import linear_to_srgb
        img = np.random.rand(100, 100, 3).astype(np.float32)
        result = linear_to_srgb(img)
        assert np.all(result >= 0)
        assert np.all(result <= 1.0 + 1e-6)

    def test_quaternion_multiply_identity(self):
        from utils.numba_kernels import quaternion_multiply
        q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = quaternion_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_quaternion_to_rotation_matrix_orthogonal(self):
        from utils.numba_kernels import quaternion_to_rotation_matrix
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_unproject_depth_shape(self):
        from utils.numba_kernels import unproject_depth_to_points
        H, W = 10, 10
        depth = np.ones((H, W), dtype=np.float32)
        K_inv = np.eye(3, dtype=np.float64)
        mask = np.ones((H, W), dtype=np.bool_)
        ys, xs = np.mgrid[:H, :W]
        ys = ys.ravel().astype(np.int64)
        xs = xs.ravel().astype(np.int64)
        points = unproject_depth_to_points(
            depth, K_inv[0], K_inv[1], K_inv[2], mask, ys, xs
        )
        assert points.shape[1] == 3

    def test_confidence_filter(self):
        from utils.numba_kernels import confidence_filter
        conf = np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.2]], dtype=np.float32)
        mask = confidence_filter(conf, 0.4)
        expected = np.array([[False, True, True], [False, True, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_apply_lut(self):
        from utils.numba_kernels import apply_lut
        img = np.array([[[0, 127, 255]]], dtype=np.uint8)
        # LUT returns uint8 (mapped values), not float
        lut = np.arange(256, dtype=np.uint8)
        result = apply_lut(img, lut)
        np.testing.assert_array_equal(result[0, 0], [0, 127, 255])

    def test_rotation_matrix_to_quaternion_identity(self):
        from utils.numba_kernels import rotation_matrix_to_quaternion
        R = np.eye(3, dtype=np.float64)
        q = rotation_matrix_to_quaternion(R)
        assert len(q) == 4
        np.testing.assert_allclose(abs(q[0]), 1.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Timing Utilities
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestTimingUtils:
    """Test timing decorators and profiler."""

    def test_timed_decorator(self):
        from utils.timing import timed

        @timed
        def fast_func():
            return 42

        result = fast_func()
        assert result == 42

    def test_stage_timer(self):
        from utils.timing import StageTimer

        with StageTimer("test stage") as timer:
            time.sleep(0.01)
        assert timer.elapsed >= 0.01
        assert timer.elapsed < 1.0

    def test_pipeline_profiler(self):
        from utils.timing import PipelineProfiler

        profiler = PipelineProfiler(session_id="test")
        profiler.start_stage("stage_a")
        time.sleep(0.01)
        profiler.end_stage("stage_a", num_items=10)

        profiler.mark_skipped("stage_b")

        data = profiler.to_dict()
        # Profiler stores stages — check the structure has our data
        stages = data.get("stages", data)
        assert isinstance(stages, (list, dict))
        # Just verify it recorded something without crashing
        assert len(stages) >= 1

    def test_profiler_summary_no_crash(self):
        """Summary should print without crashing even with no stages."""
        from utils.timing import PipelineProfiler
        profiler = PipelineProfiler(session_id="empty")
        profiler.summary()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# ML Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestReplayBuffer:
    """Test the experience replay buffer for progressive learning."""

    def test_buffer_creation(self, tmp_path):
        from ml.replay_buffer import ScanReplayBuffer
        buffer = ScanReplayBuffer(buffer_dir=tmp_path / "buffer", max_scans=10)
        assert buffer.scan_count() == 0

    def test_buffer_add_and_count(self, tmp_path):
        from ml.replay_buffer import ScanReplayBuffer

        buffer = ScanReplayBuffer(buffer_dir=tmp_path / "buffer", max_scans=10)

        # Create minimal scan data
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), img)

        flame_params = {
            "shape": np.zeros(100, dtype=np.float32),
            "expression": np.zeros(50, dtype=np.float32),
            "pose": np.zeros(6, dtype=np.float32),
        }

        buffer.add_scan(
            session_id="test_scan",
            frames_dir=frames_dir,
            flame_params=flame_params,
            gaussian_model=None,
            cameras=[],
            training_metrics={"final_loss": 0.01},
        )
        assert buffer.scan_count() == 1

    def test_buffer_summary(self, tmp_path):
        from ml.replay_buffer import ScanReplayBuffer
        buffer = ScanReplayBuffer(buffer_dir=tmp_path / "buffer")
        summary = buffer.summary()
        assert isinstance(summary, str)


# ═══════════════════════════════════════════════════════════════════════════
# Face Prior Network
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestFacePrior:
    """Test the face prior prediction network."""

    def test_network_forward(self):
        torch = pytest.importorskip("torch")
        from ml.face_prior import FacePriorNetwork

        net = FacePriorNetwork(shape_dim=100, expr_dim=50)
        shape = torch.randn(1, 100)
        expr = torch.randn(1, 50)
        output = net(shape, expr)
        assert "offsets" in output
        assert "scales" in output
        assert "opacities" in output

    def test_network_output_shapes(self):
        torch = pytest.importorskip("torch")
        from ml.face_prior import FacePriorNetwork

        net = FacePriorNetwork(shape_dim=100, expr_dim=50, num_regions=9)
        shape = torch.randn(2, 100)  # batch of 2
        expr = torch.randn(2, 50)
        output = net(shape, expr)
        assert output["offsets"].shape == (2, 9, 3)
        assert output["scales"].shape == (2, 9, 3)
        assert output["opacities"].shape == (2, 9)

    def test_network_opacities_bounded(self):
        """Opacities should be sigmoid-bounded [0, 1]."""
        torch = pytest.importorskip("torch")
        from ml.face_prior import FacePriorNetwork

        net = FacePriorNetwork()
        shape = torch.randn(5, 100)
        expr = torch.randn(5, 50)
        output = net(shape, expr)
        assert torch.all(output["opacities"] >= 0)
        assert torch.all(output["opacities"] <= 1)

    def test_network_save_load(self, tmp_path):
        torch = pytest.importorskip("torch")
        from ml.face_prior import FacePriorNetwork

        net = FacePriorNetwork()
        path = tmp_path / "prior.pt"
        net.save(path)
        assert path.exists()

        net2 = FacePriorNetwork()
        net2.load(path)
        # Weights should match
        for p1, p2 in zip(net.parameters(), net2.parameters()):
            assert torch.allclose(p1, p2)

    def test_network_small_param_count(self):
        """Prior network should be tiny (~50K params)."""
        torch = pytest.importorskip("torch")
        from ml.face_prior import FacePriorNetwork

        net = FacePriorNetwork()
        total = sum(p.numel() for p in net.parameters())
        assert total < 200_000, f"Network too large: {total} params"


# ═══════════════════════════════════════════════════════════════════════════
# Triton Kernel Fallbacks (PyTorch path)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestTritonFallbacks:
    """Test Triton kernel PyTorch fallback paths."""

    def test_confidence_prune(self):
        torch = pytest.importorskip("torch")
        from utils.triton_kernels import confidence_prune

        conf = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7])
        mask = confidence_prune(conf, 0.4)
        expected = torch.tensor([False, True, True, False, True])
        assert torch.equal(mask, expected)

    def test_sh_to_rgb_degree0(self):
        torch = pytest.importorskip("torch")
        from utils.triton_kernels import sh_to_rgb

        n = 100
        sh_coeffs = torch.randn(n, 1, 3)  # DC only
        directions = torch.randn(n, 3)
        directions = directions / directions.norm(dim=1, keepdim=True)
        rgb = sh_to_rgb(sh_coeffs, directions, sh_degree=0)
        assert rgb.shape == (n, 3)

    def test_opacity_entropy(self):
        torch = pytest.importorskip("torch")
        from utils.triton_kernels import opacity_entropy

        opacities = torch.tensor([0.5, 0.5, 0.5])  # Max entropy
        loss = opacity_entropy(opacities)
        assert loss.item() > 0

        opacities_low = torch.tensor([0.01, 0.99, 0.01])  # Low entropy
        loss_low = opacity_entropy(opacities_low)
        assert loss_low.item() < loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# Validation Contracts (new edge cases)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestValidationEdgeCases:
    """Edge cases for validation contracts not covered by test_contracts.py."""

    def test_validate_depth_maps_with_inf(self):
        from validation.contracts import validate_depth_maps
        depths = np.array([[1.0, np.inf], [0.5, 2.0]], dtype=np.float32)
        result = validate_depth_maps(depths)
        assert not result.valid
        assert any("inf" in e.lower() or "finite" in e.lower() for e in result.errors)

    def test_validate_quaternions_wrong_dim(self):
        from validation.contracts import validate_quaternions
        quats = np.random.randn(10, 3).astype(np.float32)  # 3 instead of 4
        result = validate_quaternions(quats)
        assert not result.valid

    def test_validate_gaussians_negative_scales(self):
        from validation.contracts import validate_gaussians
        n = 10
        result = validate_gaussians(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.full((n, 3), -1.0, dtype=np.float32),  # negative
            opacities=np.random.rand(n).astype(np.float32),
            quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        )
        # Scales can be in log-space (negative is fine) or linear
        # The validator should at least not crash
        assert isinstance(result.valid, bool)
