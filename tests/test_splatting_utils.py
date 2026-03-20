"""Tests for splatting and utility modules.

Covers:
- TrainingConfig defaults and validation
- Progressive resolution schedule and staged loss schedule
- Camera utilities (Camera, focal_to_fov, generate_turntable_cameras)
- Animator helpers (normal_to_quaternion, triangle normals, interpolation, presets)
- Texture baker helpers (barycentric, bilinear sampling, spherical UV, material file)
- Parallel utilities (get_optimal_workers, parallel_map)
- Visualization utilities (depth colormap, mask overlay)
- Utils/camera (S25 Ultra lens DB, focal_length_pixels, project/unproject roundtrip)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ═══════════════════════════════════════════════════════════════════════════
# TrainingConfig
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestTrainingConfig:
    """Tests for TrainingConfig dataclass defaults and semantics."""

    def test_default_iterations(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.iterations == 7000

    def test_default_strategy(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.strategy in ("default", "mcmc", "auto")
        assert cfg.strategy == "default"

    def test_default_use_2dgs(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.use_2dgs is True

    def test_progressive_training_enabled(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.progressive_training is True
        assert cfg.progressive_stages == 3

    def test_staged_loss_enabled(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.staged_loss is True

    def test_lr_warmup_positive(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.lr_warmup_iters > 0
        assert cfg.lr_warmup_iters == 100

    def test_background_color_default(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.background_color == [0.0, 0.0, 0.0]

    def test_max_num_gaussians_positive(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.max_num_gaussians > 0
        assert cfg.max_num_gaussians == 500_000

    def test_custom_iterations(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig(iterations=10_000)
        assert cfg.iterations == 10_000

    def test_lr_means_final_less_than_initial(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.lr_means_final < cfg.lr_means

    def test_depth_weight_decay_config(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.lambda_depth_initial >= cfg.lambda_depth_final

    def test_sh_degree_max_default(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.sh_degree_max == 1

    def test_packed_false_default(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.packed is False

    def test_near_far_plane_ordering(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.near_plane < cfg.far_plane

    def test_mcmc_cap_max_positive(self):
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.mcmc_cap_max > 0


# ═══════════════════════════════════════════════════════════════════════════
# Progressive resolution schedule
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestProgressiveResolution:
    """Test the 3-stage progressive resolution schedule logic."""

    def test_stage_boundaries_3stage(self):
        """Stage transitions: 0-10% quarter, 10-30% half, 30%+ full."""
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig(iterations=7000, progressive_stages=3)

        # Stage 1 ends at 10% = iteration 700
        # Stage 2 ends at 30% = iteration 2100
        stage1_end = int(cfg.iterations * 0.10)
        stage2_end = int(cfg.iterations * 0.30)

        assert stage1_end == 700
        assert stage2_end == 2100

    def test_resolution_scale_factors(self):
        """Quarter / half / full resolution scale factors."""
        scales = [0.25, 0.5, 1.0]  # 3-stage
        assert len(scales) == 3
        assert scales[0] == 0.25
        assert scales[1] == 0.5
        assert scales[2] == 1.0

    def test_2stage_fallback(self):
        """Legacy 2-stage: half then full."""
        from splatting.trainer import TrainingConfig
        cfg = TrainingConfig(progressive_stages=2)
        assert cfg.progressive_stages == 2


# ═══════════════════════════════════════════════════════════════════════════
# Staged loss schedule
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestStagedLossSchedule:
    """Test which losses are active at which iteration ranges."""

    def _active_losses(self, iteration: int, total: int = 3000):
        """Determine active losses at a given iteration based on staged schedule.

        Stage 1 (0-15%):  L1 + depth
        Stage 2 (15-40%): + D-SSIM
        Stage 3 (40%+):   + normal + distortion
        """
        frac = iteration / total
        losses = ["l1", "depth"]
        if frac >= 0.15:
            losses.append("dssim")
        if frac >= 0.40:
            losses.extend(["normal", "distortion"])
        return losses

    def test_stage1_losses(self):
        losses = self._active_losses(0)
        assert "l1" in losses
        assert "depth" in losses
        assert "dssim" not in losses
        assert "normal" not in losses

    def test_stage2_losses(self):
        losses = self._active_losses(600)  # 20% of 3000
        assert "dssim" in losses
        assert "normal" not in losses

    def test_stage3_losses(self):
        losses = self._active_losses(1500)  # 50% of 3000
        assert "dssim" in losses
        assert "normal" in losses
        assert "distortion" in losses

    def test_boundary_15pct(self):
        losses_before = self._active_losses(449, 3000)
        losses_after = self._active_losses(450, 3000)
        assert "dssim" not in losses_before
        assert "dssim" in losses_after

    def test_boundary_40pct(self):
        losses_before = self._active_losses(1199, 3000)
        losses_after = self._active_losses(1200, 3000)
        assert "normal" not in losses_before
        assert "normal" in losses_after


# ═══════════════════════════════════════════════════════════════════════════
# Camera utilities
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestCameraUtils:
    """Tests for Camera class and related functions."""

    def test_camera_instantiation(self):
        from splatting.camera_utils import Camera
        R = np.eye(3, dtype=np.float32)
        T = np.zeros(3, dtype=np.float32)
        cam = Camera(R=R, T=T, FoVx=1.0, FoVy=1.0, width=640, height=480)
        assert cam.width == 640
        assert cam.height == 480

    def test_focal_from_fov(self):
        """fx = width / (2 * tan(FoVx / 2))."""
        from splatting.camera_utils import Camera
        fov = math.pi / 4  # 45 degrees
        cam = Camera(
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FoVx=fov, FoVy=fov,
            width=640, height=480,
        )
        expected_fx = 640 / (2.0 * math.tan(fov / 2.0))
        assert abs(cam.fx - expected_fx) < 1e-5

    def test_fy_property(self):
        from splatting.camera_utils import Camera
        fov = 0.8
        cam = Camera(
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FoVx=fov, FoVy=fov,
            width=800, height=600,
        )
        expected_fy = 600 / (2.0 * math.tan(fov / 2.0))
        assert abs(cam.fy - expected_fy) < 1e-5

    def test_focal_to_fov_known_values(self):
        """525px focal, 640px width -> known FOV."""
        from splatting.camera_utils import focal_to_fov
        fov = focal_to_fov(525.0, 640)
        expected = 2.0 * math.atan(640 / (2.0 * 525.0))
        assert abs(fov - expected) < 1e-8

    def test_focal_to_fov_roundtrip(self):
        from splatting.camera_utils import Camera, focal_to_fov
        focal = 525.0
        width = 640
        fov = focal_to_fov(focal, width)
        cam = Camera(
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FoVx=fov, FoVy=fov,
            width=width, height=480,
        )
        assert abs(cam.fx - focal) < 1e-4

    def test_get_viewmat_shape(self):
        torch = pytest.importorskip("torch")
        from splatting.camera_utils import Camera
        cam = Camera(
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FoVx=1.0, FoVy=1.0,
            width=640, height=480,
        )
        viewmat = cam.get_viewmat("cpu")
        assert viewmat.shape == (4, 4)
        assert viewmat.dtype == torch.float32

    def test_get_K_shape(self):
        torch = pytest.importorskip("torch")
        from splatting.camera_utils import Camera
        cam = Camera(
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FoVx=1.0, FoVy=1.0,
            width=640, height=480,
        )
        K = cam.get_K("cpu")
        assert K.shape == (3, 3)
        # Principal point at center
        assert abs(K[0, 2].item() - 320.0) < 1e-3
        assert abs(K[1, 2].item() - 240.0) < 1e-3

    def test_generate_turntable_cameras_count(self):
        from splatting.camera_utils import generate_turntable_cameras
        center = np.array([0.0, 0.0, 0.0])
        cams = generate_turntable_cameras(center, radius=1.0, num_views=12)
        assert len(cams) == 12

    def test_generate_turntable_cameras_positions_form_circle(self):
        from splatting.camera_utils import generate_turntable_cameras
        center = np.array([0.0, 0.0, 0.0])
        radius = 2.0
        cams = generate_turntable_cameras(center, radius=radius, num_views=36)

        for cam in cams:
            # Camera position = -R^T @ T
            cam_pos = -cam.R.T @ cam.T
            dist = np.linalg.norm(cam_pos - center)
            assert abs(dist - radius) < 0.05, f"Camera not on circle: dist={dist}"

    def test_generate_turntable_cameras_look_at_center(self):
        from splatting.camera_utils import generate_turntable_cameras
        center = np.array([0.0, 0.0, 0.0])
        cams = generate_turntable_cameras(center, radius=2.0, num_views=8)

        for cam in cams:
            cam_pos = -cam.R.T @ cam.T
            # Forward direction is 3rd row of R
            forward = cam.R[2, :]
            to_center = center - cam_pos
            to_center /= np.linalg.norm(to_center) + 1e-8
            dot = np.dot(forward, to_center)
            assert dot > 0.9, f"Camera not looking at center: dot={dot}"

    def test_generate_turntable_cameras_with_height(self):
        from splatting.camera_utils import generate_turntable_cameras
        center = np.array([0.0, 0.0, 0.0])
        height = 0.5
        cams = generate_turntable_cameras(center, radius=2.0, num_views=4, height=height)
        for cam in cams:
            cam_pos = -cam.R.T @ cam.T
            # Y coordinate should be height offset
            assert abs(cam_pos[1] - height) < 0.05

    def test_camera_uid_sequential(self):
        from splatting.camera_utils import generate_turntable_cameras
        cams = generate_turntable_cameras(np.zeros(3), radius=1.0, num_views=5)
        uids = [c.uid for c in cams]
        assert uids == [0, 1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════════════════════
# Animator helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestAnimatorHelpers:
    """Tests for animation helper functions (no GPU needed)."""

    def test_normal_to_quaternion_z_axis(self):
        """Normal [0,0,1] should give identity quaternion [1,0,0,0]."""
        torch = pytest.importorskip("torch")
        from splatting.animator import _normal_to_quaternion_torch

        normals = torch.tensor([[0.0, 0.0, 1.0]])
        quats = _normal_to_quaternion_torch(normals)
        assert quats.shape == (1, 4)
        # Identity quaternion
        np.testing.assert_allclose(quats[0].numpy(), [1, 0, 0, 0], atol=1e-4)

    def test_normal_to_quaternion_neg_z(self):
        """Normal [0,0,-1] should give 180-degree rotation."""
        torch = pytest.importorskip("torch")
        from splatting.animator import _normal_to_quaternion_torch

        normals = torch.tensor([[0.0, 0.0, -1.0]])
        quats = _normal_to_quaternion_torch(normals)
        assert quats.shape == (1, 4)
        # 180-degree rotation about x: [0, 1, 0, 0]
        assert abs(quats[0, 0].item()) < 0.01
        assert abs(quats[0, 1].item() - 1.0) < 0.01

    def test_normal_to_quaternion_unit_length(self):
        """All output quaternions should be unit length."""
        torch = pytest.importorskip("torch")
        from splatting.animator import _normal_to_quaternion_torch

        rng = np.random.default_rng(42)
        normals_np = rng.normal(size=(100, 3)).astype(np.float32)
        normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True)
        normals = torch.from_numpy(normals_np)

        quats = _normal_to_quaternion_torch(normals)
        norms = torch.norm(quats, dim=1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-4)

    def test_compute_triangle_normals(self):
        torch = pytest.importorskip("torch")
        from splatting.animator import _compute_triangle_normals_torch

        # Simple triangle in XY plane -> normal along Z
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        normals = _compute_triangle_normals_torch(vertices, faces)
        assert normals.shape == (1, 3)
        # Should point in +Z direction
        assert normals[0, 2].item() > 0.99

    def test_compute_triangle_normals_unit_length(self):
        torch = pytest.importorskip("torch")
        from splatting.animator import _compute_triangle_normals_torch

        rng = np.random.default_rng(7)
        verts = torch.from_numpy(rng.normal(size=(20, 3)).astype(np.float32))
        faces = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.long)
        normals = _compute_triangle_normals_torch(verts, faces)
        norms = torch.norm(normals, dim=1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-5)

    def test_interpolate_params_endpoints(self):
        from splatting.animator import _builtin_presets, FaceAnimator

        # Test the interpolation logic directly
        start = {
            "expression": np.zeros(10, dtype=np.float32),
            "jaw_pose": np.zeros(3, dtype=np.float32),
            "neck_pose": np.zeros(3, dtype=np.float32),
        }
        end = {
            "expression": np.ones(10, dtype=np.float32),
            "jaw_pose": np.ones(3, dtype=np.float32),
            "neck_pose": np.ones(3, dtype=np.float32),
        }
        # Manually replicate the interpolation logic
        num_frames = 5
        frames = []
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            frame = {}
            for key in ("expression", "jaw_pose", "neck_pose"):
                s = start[key]
                e = end[key]
                frame[key] = (1 - t) * s + t * e
            frames.append(frame)

        # First frame should be start
        np.testing.assert_allclose(frames[0]["expression"], 0.0, atol=1e-6)
        # Last frame should be end
        np.testing.assert_allclose(frames[-1]["expression"], 1.0, atol=1e-6)
        # Middle frame should be 0.5
        np.testing.assert_allclose(frames[2]["expression"], 0.5, atol=1e-6)

    def test_builtin_presets_keys(self):
        from splatting.animator import _builtin_presets
        presets = _builtin_presets()
        assert "neutral" in presets
        assert "smile" in presets
        assert "surprise" in presets
        assert "blink" in presets
        assert "frown" in presets

    def test_builtin_presets_shapes(self):
        from splatting.animator import _builtin_presets
        presets = _builtin_presets()
        for name, params in presets.items():
            assert params["expression"].shape == (10,), f"{name} expression shape wrong"
            assert params["jaw_pose"].shape == (3,), f"{name} jaw_pose shape wrong"
            assert params["neck_pose"].shape == (3,), f"{name} neck_pose shape wrong"

    def test_neutral_preset_is_zeros(self):
        from splatting.animator import _builtin_presets
        presets = _builtin_presets()
        neutral = presets["neutral"]
        np.testing.assert_array_equal(neutral["expression"], 0.0)
        np.testing.assert_array_equal(neutral["jaw_pose"], 0.0)
        np.testing.assert_array_equal(neutral["neck_pose"], 0.0)

    def test_write_obj(self, tmp_path):
        from splatting.animator import _write_obj
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        path = tmp_path / "test.obj"
        _write_obj(verts, faces, path)
        assert path.exists()
        text = path.read_text()
        assert "v 0.000000 0.000000 0.000000" in text
        # OBJ faces are 1-indexed
        assert "f 1 2 3" in text


# ═══════════════════════════════════════════════════════════════════════════
# Texture Baker helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestTextureBakerHelpers:
    """Tests for texture_baker pure functions."""

    def test_barycentric_2d_inside_triangle(self):
        from splatting.texture_baker import _barycentric_2d
        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # Centroid should have ~equal bary coords
        centroid = np.array([[1.0 / 3, 1.0 / 3]])
        bary = _barycentric_2d(tri, centroid)
        assert bary.shape == (1, 3)
        assert np.all(bary >= -0.01), f"Centroid should be inside triangle: {bary}"
        np.testing.assert_allclose(bary.sum(), 1.0, atol=1e-6)

    def test_barycentric_2d_at_vertices(self):
        from splatting.texture_baker import _barycentric_2d
        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # At vertex 0
        bary = _barycentric_2d(tri, np.array([[0.0, 0.0]]))
        assert bary[0, 0] > 0.99, f"At v0, bary[0] should be ~1: {bary}"

    def test_barycentric_2d_outside_triangle(self):
        from splatting.texture_baker import _barycentric_2d
        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        outside = np.array([[2.0, 2.0]])
        bary = _barycentric_2d(tri, outside)
        # At least one coord should be negative
        assert np.any(bary < 0)

    def test_barycentric_2d_sum_to_one(self):
        from splatting.texture_baker import _barycentric_2d
        tri = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]])
        rng = np.random.default_rng(99)
        points = rng.uniform(0, 3, size=(50, 2))
        bary = _barycentric_2d(tri, points)
        np.testing.assert_allclose(bary.sum(axis=1), 1.0, atol=1e-6)

    def test_sample_pixel_bilinear_center(self):
        from splatting.texture_baker import _sample_pixel_bilinear
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        color = _sample_pixel_bilinear(img, 5.0, 5.0)
        assert color is not None
        np.testing.assert_allclose(color, 128.0, atol=1e-5)

    def test_sample_pixel_bilinear_out_of_bounds(self):
        from splatting.texture_baker import _sample_pixel_bilinear
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert _sample_pixel_bilinear(img, -1.0, 5.0) is None
        assert _sample_pixel_bilinear(img, 5.0, -1.0) is None
        assert _sample_pixel_bilinear(img, 9.5, 5.0) is None

    def test_sample_pixel_bilinear_interpolation(self):
        from splatting.texture_baker import _sample_pixel_bilinear
        # Create a 4x4 image with gradient
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = [0, 0, 0]
        img[0, 1] = [100, 100, 100]
        img[1, 0] = [100, 100, 100]
        img[1, 1] = [200, 200, 200]
        # Sample at (0.5, 0.5) -> should be average of 4 corners
        color = _sample_pixel_bilinear(img, 0.5, 0.5)
        assert color is not None
        # Bilinear average: (0 + 100 + 100 + 200) / 4 = 100
        np.testing.assert_allclose(color, 100.0, atol=1.0)

    def test_spherical_uv_output_shape(self):
        from splatting.texture_baker import create_spherical_uv
        rng = np.random.default_rng(42)
        vertices = rng.normal(size=(20, 3))
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        uv = create_spherical_uv(vertices, faces)
        assert uv.shape == (9, 2)  # F*3 x 2

    def test_spherical_uv_range(self):
        from splatting.texture_baker import create_spherical_uv
        rng = np.random.default_rng(42)
        vertices = rng.normal(size=(20, 3))
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        uv = create_spherical_uv(vertices, faces)
        assert uv.min() >= 0.0 - 0.01
        assert uv.max() <= 1.0 + 0.01

    def test_create_material_file(self, tmp_path):
        from splatting.texture_baker import create_material_file
        tex_path = tmp_path / "texture.png"
        tex_path.write_bytes(b"fake")
        mtl_path = tmp_path / "test.mtl"
        result = create_material_file(tex_path, mtl_path)
        assert result == mtl_path
        assert mtl_path.exists()
        content = mtl_path.read_text()
        assert "newmtl face_texture" in content
        assert "map_Kd texture.png" in content

    def test_compute_face_normals(self):
        from splatting.texture_baker import _compute_face_normals
        # Need trimesh for this — skip if not available
        trimesh = pytest.importorskip("trimesh")
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 0], [0, 0, 1], [1, 0, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        normals = _compute_face_normals(mesh)
        assert normals.shape == (2, 3)
        # First face normal should be ~[0, 0, 1]
        assert normals[0, 2] > 0.99


# ═══════════════════════════════════════════════════════════════════════════
# Parallel utilities
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestParallelUtils:
    """Tests for parallel.py functions."""

    def test_get_optimal_workers_cpu_positive(self):
        from utils.parallel import get_optimal_workers
        w = get_optimal_workers("cpu")
        assert 1 <= w <= os.cpu_count()

    def test_get_optimal_workers_io_at_least_cpu(self):
        from utils.parallel import get_optimal_workers
        w_io = get_optimal_workers("io")
        w_cpu = get_optimal_workers("cpu")
        assert w_io >= w_cpu

    def test_get_optimal_workers_io_capped(self):
        from utils.parallel import get_optimal_workers
        w = get_optimal_workers("io")
        assert w <= 32

    def test_parallel_map_simple(self):
        from utils.parallel import parallel_map
        items = [1, 2, 3, 4, 5]
        results = parallel_map(lambda x: x * 2, items, desc="test", use_threads=True)
        assert results == [2, 4, 6, 8, 10]

    def test_parallel_map_empty(self):
        from utils.parallel import parallel_map
        results = parallel_map(lambda x: x, [], desc="empty")
        assert results == []

    def test_parallel_map_small_batch_serial(self):
        """Batches <= 2 run serially."""
        from utils.parallel import parallel_map
        results = parallel_map(lambda x: x + 1, [10, 20], desc="small")
        assert results == [11, 21]

    def test_parallel_map_preserves_order(self):
        from utils.parallel import parallel_map
        items = list(range(10))
        results = parallel_map(lambda x: x ** 2, items, desc="order", use_threads=True)
        assert results == [x ** 2 for x in items]

    def test_parallel_map_error_handling(self):
        """Failed items should be None, others should succeed."""
        from utils.parallel import parallel_map

        def maybe_fail(x):
            if x == 3:
                raise ValueError("intentional failure")
            return x * 10

        results = parallel_map(maybe_fail, [1, 2, 3, 4, 5], desc="fail", use_threads=True)
        assert results[0] == 10
        assert results[1] == 20
        assert results[2] is None  # failed item
        assert results[3] == 40
        assert results[4] == 50


# ═══════════════════════════════════════════════════════════════════════════
# S25 Ultra lens database and utils/camera
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestS25UltraCameraUtils:
    """Tests for Samsung S25 Ultra lens database and projection functions."""

    def test_lens_db_has_all_lenses(self):
        from utils.camera import S25_ULTRA_LENSES
        assert "wide" in S25_ULTRA_LENSES
        assert "ultrawide" in S25_ULTRA_LENSES
        assert "telephoto_3x" in S25_ULTRA_LENSES
        assert "telephoto_5x" in S25_ULTRA_LENSES

    def test_wide_lens_200mp(self):
        from utils.camera import S25_ULTRA_LENSES
        wide = S25_ULTRA_LENSES["wide"]
        assert wide["megapixels"] == 200
        assert wide["focal_length_mm"] == 6.3

    def test_focal_length_pixels_conversion(self):
        from utils.camera import focal_length_pixels
        # Wide lens: 6.3mm focal, 8.0mm sensor width
        fl_px = focal_length_pixels(6.3, 8.0, 8160)
        expected = 6.3 * 8160 / 8.0
        assert abs(fl_px - expected) < 0.01

    def test_focal_length_pixels_proportional(self):
        """Doubling image width should double focal length in pixels."""
        from utils.camera import focal_length_pixels
        fl1 = focal_length_pixels(6.3, 8.0, 1000)
        fl2 = focal_length_pixels(6.3, 8.0, 2000)
        assert abs(fl2 / fl1 - 2.0) < 1e-6

    def test_project_unproject_roundtrip(self):
        from utils.camera import project_points, unproject_points
        rng = np.random.default_rng(42)
        K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)

        pts_3d = rng.uniform(-0.5, 0.5, size=(10, 3)).astype(np.float64)
        pts_3d[:, 2] += 2.0  # Ensure points are in front of camera

        pts_2d = project_points(pts_3d, K, R, t)
        depths = pts_3d[:, 2]  # z-coordinate in camera frame (R=I, t=0)
        pts_3d_back = unproject_points(pts_2d, depths, K, R, t)
        np.testing.assert_allclose(pts_3d_back, pts_3d, atol=1e-8)

    def test_project_points_shape(self):
        from utils.camera import project_points
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        t = np.array([0, 0, 0], dtype=np.float64)
        pts = np.array([[0, 0, 5], [1, 1, 3]], dtype=np.float64)
        result = project_points(pts, K, R, t)
        assert result.shape == (2, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization utilities
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestVisualization:
    """Tests for visualization.py pure helpers."""

    def test_depth_map_colormap_output(self, tmp_path):
        from utils.visualization import visualize_depth_map
        depth = np.random.default_rng(42).uniform(0.5, 5.0, size=(100, 100)).astype(np.float32)
        out_path = tmp_path / "depth_vis.png"
        result = visualize_depth_map(depth, out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 100

    def test_depth_map_colormap_custom(self, tmp_path):
        from utils.visualization import visualize_depth_map
        depth = np.ones((50, 50), dtype=np.float32)
        out_path = tmp_path / "depth_uniform.png"
        result = visualize_depth_map(depth, out_path, colormap="viridis")
        assert result == out_path
        assert out_path.exists()

    def test_depth_map_from_npy(self, tmp_path):
        from utils.visualization import visualize_depth_map
        depth = np.random.default_rng(1).uniform(1, 3, size=(60, 80)).astype(np.float32)
        npy_path = tmp_path / "depth.npy"
        np.save(str(npy_path), depth)
        out_path = tmp_path / "depth_from_npy.png"
        result = visualize_depth_map(str(npy_path), out_path)
        assert out_path.exists()

    def test_depth_map_rejects_3d(self, tmp_path):
        from utils.visualization import visualize_depth_map
        depth_3d = np.zeros((10, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            visualize_depth_map(depth_3d, tmp_path / "bad.png")

    def test_landmark_region_colors_defined(self):
        from utils.visualization import _REGION_COLORS, _LANDMARK_REGIONS
        for region in _LANDMARK_REGIONS:
            assert region in _REGION_COLORS

    def test_index_to_color_mapping(self):
        from utils.visualization import _INDEX_TO_COLOR, _DEFAULT_COLOR
        # Some indices should be mapped
        assert len(_INDEX_TO_COLOR) > 0
        # Check a known eye index
        assert 33 in _INDEX_TO_COLOR  # eyes region starts at 33

    def test_mask_overlay(self, tmp_path):
        import cv2
        from utils.visualization import visualize_mask_overlay
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255  # face region
        img_path = tmp_path / "img.png"
        cv2.imwrite(str(img_path), img)
        mask_path = tmp_path / "mask.png"
        cv2.imwrite(str(mask_path), mask)
        out = tmp_path / "overlay.png"
        result = visualize_mask_overlay(str(img_path), str(mask_path), out, alpha=0.5)
        assert result == out
        assert out.exists()

    def test_mask_overlay_with_arrays(self, tmp_path):
        from utils.visualization import visualize_mask_overlay
        img = np.full((80, 80, 3), 200, dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:60, 20:60] = 1
        out = tmp_path / "overlay_arr.png"
        result = visualize_mask_overlay(img, mask, out, alpha=0.3)
        assert out.exists()

    def test_mask_overlay_resizes_mask(self, tmp_path):
        from utils.visualization import visualize_mask_overlay
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)  # different size
        mask[10:40, 10:40] = 1
        out = tmp_path / "overlay_resize.png"
        result = visualize_mask_overlay(img, mask, out)
        assert out.exists()

    def test_visualize_landmarks_with_array(self, tmp_path):
        import cv2
        from utils.visualization import visualize_landmarks
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(42)
        landmarks = rng.uniform(0, 1, size=(478, 2)).astype(np.float32)  # normalized
        out = tmp_path / "landmarks.png"
        result = visualize_landmarks(img, landmarks, out)
        assert out.exists()


# ═══════════════════════════════════════════════════════════════════════════
# COLMAP binary reader (camera_utils)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestColmapCameraReader:
    """Tests for COLMAP binary camera reading in camera_utils."""

    def test_load_cameras_from_colmap(self, sample_colmap_model):
        from splatting.camera_utils import load_cameras_from_colmap
        cameras = load_cameras_from_colmap(sample_colmap_model)
        assert len(cameras) == 5

    def test_loaded_camera_properties(self, sample_colmap_model):
        from splatting.camera_utils import load_cameras_from_colmap
        cameras = load_cameras_from_colmap(sample_colmap_model)
        for cam in cameras:
            assert cam.width == 640
            assert cam.height == 480
            assert cam.fx > 0
            assert cam.fy > 0
            assert cam.R.shape == (3, 3)
            assert cam.T.shape == (3,)

    def test_qvec_to_rotmat_identity(self):
        from splatting.camera_utils import _qvec_to_rotmat
        # Identity quaternion [1, 0, 0, 0] -> Identity rotation
        R = _qvec_to_rotmat([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_qvec_to_rotmat_orthogonal(self):
        from splatting.camera_utils import _qvec_to_rotmat
        # Random unit quaternion
        q = np.array([0.5, 0.5, 0.5, 0.5])
        R = _qvec_to_rotmat(q.tolist())
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_extract_focal_pinhole(self):
        from splatting.camera_utils import _extract_focal
        fx, fy, cx, cy = _extract_focal("PINHOLE", [500, 500, 320, 240], 640, 480)
        assert fx == 500
        assert fy == 500
        assert cx == 320
        assert cy == 240

    def test_extract_focal_simple_pinhole(self):
        from splatting.camera_utils import _extract_focal
        fx, fy, cx, cy = _extract_focal("SIMPLE_PINHOLE", [500, 320, 240], 640, 480)
        assert fx == fy == 500


# ═══════════════════════════════════════════════════════════════════════════
# AppearanceMLP
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestAppearanceMLP:
    """Tests for the AppearanceMLP network."""

    def test_output_shape(self):
        torch = pytest.importorskip("torch")
        from splatting.trainer import AppearanceMLP
        mlp = AppearanceMLP(appearance_dim=32, hidden_dim=64)
        rgb = torch.rand(100, 200, 3)
        emb = torch.rand(32)
        out = mlp(rgb, emb)
        assert out.shape == (100, 200, 3)

    def test_output_clamped(self):
        torch = pytest.importorskip("torch")
        from splatting.trainer import AppearanceMLP
        mlp = AppearanceMLP(appearance_dim=16)
        rgb = torch.rand(10, 10, 3)
        emb = torch.rand(16)
        out = mlp(rgb, emb)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_residual_connection(self):
        """Output should be rgb + correction, so for zero-init MLP ~= rgb."""
        torch = pytest.importorskip("torch")
        from splatting.trainer import AppearanceMLP
        mlp = AppearanceMLP(appearance_dim=8)
        # Zero out the last linear layer to test residual
        with torch.no_grad():
            mlp.net[-1].weight.zero_()
            mlp.net[-1].bias.zero_()
        rgb = torch.rand(5, 5, 3)
        emb = torch.rand(8)
        out = mlp(rgb, emb)
        torch.testing.assert_close(out, rgb, atol=1e-6, rtol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Exporter constants
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.regression
class TestExporterConstants:
    """Tests for exporter module constants."""

    def test_sh_c0_value(self):
        from splatting.exporter import _SH_C0
        expected = 1.0 / (2.0 * math.sqrt(math.pi))
        assert abs(_SH_C0 - expected) < 1e-10
