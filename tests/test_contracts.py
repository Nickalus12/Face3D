"""Tests for data validation contracts.

Validates each contract function with:
    - Valid synthetic data (should pass)
    - Invalid data: NaN, wrong shapes, out of bounds
    - Edge cases: empty arrays, single element, large values
"""

from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_quaternions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate N random unit quaternions (w, x, y, z)."""
    raw = rng.normal(size=(n, 4)).astype(np.float64)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
# validate_frames
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateFrames:

    @pytest.mark.regression
    def test_valid_frames(self, sample_frames, tmp_path):
        from validation.contracts import validate_frames

        result = validate_frames(sample_frames[0].parent)
        assert result.valid
        assert len(result.errors) == 0

    @pytest.mark.regression
    def test_missing_directory(self, tmp_path):
        from validation.contracts import validate_frames

        result = validate_frames(tmp_path / "nonexistent")
        assert not result.valid
        assert any("does not exist" in e for e in result.errors)

    @pytest.mark.regression
    def test_empty_directory(self, tmp_path):
        from validation.contracts import validate_frames

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = validate_frames(empty_dir)
        assert not result.valid
        assert any("0 frames" in e for e in result.errors)

    @pytest.mark.regression
    def test_unreadable_image(self, tmp_path):
        from validation.contracts import validate_frames

        frames_dir = tmp_path / "bad_frames"
        frames_dir.mkdir()
        # Write garbage as a .png file
        (frames_dir / "bad.png").write_bytes(b"\x00" * 50)
        result = validate_frames(frames_dir)
        assert not result.valid
        assert any("Unreadable" in e for e in result.errors)

    @pytest.mark.regression
    def test_too_small_image(self, tmp_path):
        from validation.contracts import validate_frames

        frames_dir = tmp_path / "tiny_frames"
        frames_dir.mkdir()
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "tiny.png"), tiny)
        result = validate_frames(frames_dir, min_width=64, min_height=64)
        assert not result.valid
        assert any("below minimum" in e for e in result.errors)

    @pytest.mark.regression
    def test_single_valid_frame(self, tmp_path):
        from validation.contracts import validate_frames

        frames_dir = tmp_path / "one_frame"
        frames_dir.mkdir()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000000.png"), img)
        result = validate_frames(frames_dir, min_frames=1)
        assert result.valid


# ═══════════════════════════════════════════════════════════════════════════
# validate_depth_maps
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateDepthMaps:

    @pytest.mark.regression
    def test_valid_depth_maps(self):
        from validation.contracts import validate_depth_maps

        rng = np.random.default_rng(42)
        depths = [rng.uniform(0.5, 5.0, size=(480, 640)).astype(np.float32) for _ in range(3)]
        result = validate_depth_maps(depths)
        assert result.valid

    @pytest.mark.regression
    def test_single_depth_map(self):
        from validation.contracts import validate_depth_maps

        depth = np.ones((100, 100), dtype=np.float32)
        result = validate_depth_maps(depth)
        assert result.valid

    @pytest.mark.regression
    def test_depth_with_nan(self):
        from validation.contracts import validate_depth_maps

        depth = np.ones((100, 100), dtype=np.float32)
        depth[50, 50] = np.nan
        result = validate_depth_maps(depth)
        assert not result.valid
        assert any("NaN" in e for e in result.errors)

    @pytest.mark.regression
    def test_depth_with_inf(self):
        from validation.contracts import validate_depth_maps

        depth = np.ones((100, 100), dtype=np.float32)
        depth[0, 0] = np.inf
        result = validate_depth_maps(depth)
        assert not result.valid
        assert any("Inf" in e for e in result.errors)

    @pytest.mark.regression
    def test_depth_with_negative(self):
        from validation.contracts import validate_depth_maps

        depth = np.ones((100, 100), dtype=np.float32)
        depth[10, 10] = -0.5
        result = validate_depth_maps(depth)
        assert not result.valid
        assert any("negative" in e for e in result.errors)

    @pytest.mark.regression
    def test_depth_wrong_dtype(self):
        from validation.contracts import validate_depth_maps

        depth = np.ones((100, 100), dtype=np.int32)
        result = validate_depth_maps(depth)
        assert not result.valid
        assert any("float dtype" in e for e in result.errors)

    @pytest.mark.regression
    def test_depth_3d_shape(self):
        from validation.contracts import validate_depth_maps

        # 1D should fail
        depth = np.ones((100,), dtype=np.float32)
        result = validate_depth_maps([depth])
        assert not result.valid
        assert any("2D" in e for e in result.errors)

    @pytest.mark.regression
    def test_empty_list(self):
        from validation.contracts import validate_depth_maps

        result = validate_depth_maps([])
        assert not result.valid

    @pytest.mark.regression
    def test_batch_depth_3d_array(self):
        from validation.contracts import validate_depth_maps

        # 3D array treated as batch
        depths = np.ones((3, 100, 100), dtype=np.float32)
        result = validate_depth_maps(depths)
        assert result.valid


# ═══════════════════════════════════════════════════════════════════════════
# validate_quaternions
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateQuaternions:

    @pytest.mark.regression
    def test_valid_quaternions(self):
        from validation.contracts import validate_quaternions

        rng = np.random.default_rng(42)
        quats = _make_unit_quaternions(10, rng)
        result = validate_quaternions(quats)
        assert result.valid

    @pytest.mark.regression
    def test_single_quaternion(self):
        from validation.contracts import validate_quaternions

        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = validate_quaternions(q)
        assert result.valid

    @pytest.mark.regression
    def test_non_unit_quaternions(self):
        from validation.contracts import validate_quaternions

        quats = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
        result = validate_quaternions(quats)
        assert not result.valid
        assert any("unit norm" in e for e in result.errors)

    @pytest.mark.regression
    def test_nan_quaternion(self):
        from validation.contracts import validate_quaternions

        quats = np.array([[1.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 0.0]])
        result = validate_quaternions(quats)
        assert not result.valid
        assert any("non-finite" in e for e in result.errors)

    @pytest.mark.regression
    def test_wrong_shape(self):
        from validation.contracts import validate_quaternions

        quats = np.ones((5, 3))
        result = validate_quaternions(quats)
        assert not result.valid
        assert any("(N, 4)" in e for e in result.errors)

    @pytest.mark.regression
    def test_empty_array(self):
        from validation.contracts import validate_quaternions

        quats = np.zeros((0, 4))
        result = validate_quaternions(quats)
        assert not result.valid

    @pytest.mark.regression
    def test_tolerance_parameter(self):
        from validation.contracts import validate_quaternions

        # Slightly off unit norm but within loose tolerance
        quats = np.array([[1.001, 0.0, 0.0, 0.0]])
        result_strict = validate_quaternions(quats, tolerance=1e-6)
        result_loose = validate_quaternions(quats, tolerance=0.01)
        assert not result_strict.valid
        assert result_loose.valid


# ═══════════════════════════════════════════════════════════════════════════
# validate_colmap_model
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateColmapModel:

    def _make_simple_model(self):
        """Create minimal valid COLMAP-like model dicts."""
        from types import SimpleNamespace

        cameras = {
            1: SimpleNamespace(params=np.array([525.0, 525.0, 320.0, 240.0]),
                               width=640, height=480, model="PINHOLE"),
        }
        images = {
            1: SimpleNamespace(
                qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                tvec=np.array([0.0, 0.0, 0.0]),
                camera_id=1,
                name="frame_000000.png",
            ),
        }
        points3D = {
            1: SimpleNamespace(xyz=np.array([0.0, 0.0, 1.0])),
        }
        return cameras, images, points3D

    @pytest.mark.regression
    def test_valid_model(self):
        from validation.contracts import validate_colmap_model

        cameras, images, points3D = self._make_simple_model()
        result = validate_colmap_model(cameras, images, points3D)
        assert result.valid

    @pytest.mark.regression
    def test_no_cameras(self):
        from validation.contracts import validate_colmap_model

        _, images, points3D = self._make_simple_model()
        result = validate_colmap_model({}, images, points3D)
        assert not result.valid

    @pytest.mark.regression
    def test_no_images(self):
        from validation.contracts import validate_colmap_model

        cameras, _, points3D = self._make_simple_model()
        result = validate_colmap_model(cameras, {}, points3D)
        assert not result.valid

    @pytest.mark.regression
    def test_missing_camera_reference(self):
        from validation.contracts import validate_colmap_model
        from types import SimpleNamespace

        cameras, images, points3D = self._make_simple_model()
        images[1].camera_id = 999  # nonexistent
        result = validate_colmap_model(cameras, images, points3D)
        assert not result.valid
        assert any("missing camera" in e for e in result.errors)

    @pytest.mark.regression
    def test_nan_point_position(self):
        from validation.contracts import validate_colmap_model
        from types import SimpleNamespace

        cameras, images, points3D = self._make_simple_model()
        points3D[1].xyz = np.array([np.nan, 0.0, 0.0])
        result = validate_colmap_model(cameras, images, points3D)
        assert not result.valid

    @pytest.mark.regression
    def test_non_unit_image_quaternion(self):
        from validation.contracts import validate_colmap_model

        cameras, images, points3D = self._make_simple_model()
        images[1].qvec = np.array([5.0, 0.0, 0.0, 0.0])
        result = validate_colmap_model(cameras, images, points3D)
        assert not result.valid

    @pytest.mark.regression
    def test_no_points_warning(self):
        from validation.contracts import validate_colmap_model

        cameras, images, _ = self._make_simple_model()
        result = validate_colmap_model(cameras, images, {})
        assert result.valid  # warning only, not an error
        assert len(result.warnings) > 0

    @pytest.mark.regression
    def test_negative_focal_length(self):
        from validation.contracts import validate_colmap_model

        cameras, images, points3D = self._make_simple_model()
        cameras[1].params = np.array([-525.0, 525.0, 320.0, 240.0])
        result = validate_colmap_model(cameras, images, points3D)
        assert not result.valid
        assert any("focal length" in e for e in result.errors)


# ═══════════════════════════════════════════════════════════════════════════
# validate_flame_params
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateFlameParams:

    @pytest.mark.regression
    def test_valid_params(self):
        from validation.contracts import validate_flame_params

        rng = np.random.default_rng(42)
        shape = rng.normal(0, 1, size=(1, 100)).astype(np.float32)
        expr = rng.normal(0, 0.5, size=(1, 50)).astype(np.float32)
        pose = np.zeros((1, 6), dtype=np.float32)
        verts = rng.normal(0, 0.1, size=(5023, 3)).astype(np.float32)

        result = validate_flame_params(shape, expr, pose, verts)
        assert result.valid

    @pytest.mark.regression
    def test_wrong_vertex_count(self):
        from validation.contracts import validate_flame_params

        shape = np.zeros((100,), dtype=np.float32)
        expr = np.zeros((50,), dtype=np.float32)
        pose = np.zeros((6,), dtype=np.float32)
        verts = np.zeros((1000, 3), dtype=np.float32)

        result = validate_flame_params(shape, expr, pose, verts)
        assert not result.valid
        assert any("5023" in e for e in result.errors)

    @pytest.mark.regression
    def test_nan_vertices(self):
        from validation.contracts import validate_flame_params

        shape = np.zeros((100,), dtype=np.float32)
        expr = np.zeros((50,), dtype=np.float32)
        pose = np.zeros((6,), dtype=np.float32)
        verts = np.zeros((5023, 3), dtype=np.float32)
        verts[100] = np.nan

        result = validate_flame_params(shape, expr, pose, verts)
        assert not result.valid
        assert any("NaN" in e for e in result.errors)

    @pytest.mark.regression
    def test_extreme_shape_warns(self):
        from validation.contracts import validate_flame_params

        shape = np.full((100,), 15.0, dtype=np.float32)  # exceeds default bound of 10
        expr = np.zeros((50,), dtype=np.float32)
        pose = np.zeros((6,), dtype=np.float32)
        verts = np.zeros((5023, 3), dtype=np.float32)

        result = validate_flame_params(shape, expr, pose, verts)
        assert result.valid  # warning only
        assert len(result.warnings) > 0

    @pytest.mark.regression
    def test_batch_dim_squeezed(self):
        from validation.contracts import validate_flame_params

        shape = np.zeros((1, 100), dtype=np.float32)
        expr = np.zeros((1, 50), dtype=np.float32)
        pose = np.zeros((1, 6), dtype=np.float32)
        verts = np.zeros((1, 5023, 3), dtype=np.float32)  # batch dim

        result = validate_flame_params(shape, expr, pose, verts)
        assert result.valid

    @pytest.mark.regression
    def test_inf_pose(self):
        from validation.contracts import validate_flame_params

        shape = np.zeros((100,), dtype=np.float32)
        expr = np.zeros((50,), dtype=np.float32)
        pose = np.array([np.inf, 0, 0, 0, 0, 0], dtype=np.float32)
        verts = np.zeros((5023, 3), dtype=np.float32)

        result = validate_flame_params(shape, expr, pose, verts)
        assert not result.valid


# ═══════════════════════════════════════════════════════════════════════════
# validate_gaussians
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateGaussians:

    def _make_valid_gaussians(self, n: int = 50):
        rng = np.random.default_rng(77)
        means = rng.normal(0, 0.1, size=(n, 3)).astype(np.float32)
        scales = rng.uniform(-5, -2, size=(n, 3)).astype(np.float32)
        opacities = rng.uniform(0, 2, size=(n, 1)).astype(np.float32)
        raw_q = rng.normal(size=(n, 4)).astype(np.float32)
        quats = raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True)
        return means, scales, opacities, quats

    @pytest.mark.regression
    def test_valid_gaussians(self):
        from validation.contracts import validate_gaussians

        means, scales, opacities, quats = self._make_valid_gaussians()
        result = validate_gaussians(means, scales, opacities, quats)
        assert result.valid

    @pytest.mark.regression
    def test_nan_positions(self):
        from validation.contracts import validate_gaussians

        means, scales, opacities, quats = self._make_valid_gaussians()
        means[0] = np.nan
        result = validate_gaussians(means, scales, opacities, quats)
        assert not result.valid

    @pytest.mark.regression
    def test_inf_scales(self):
        from validation.contracts import validate_gaussians

        means, scales, opacities, quats = self._make_valid_gaussians()
        scales[5] = np.inf
        result = validate_gaussians(means, scales, opacities, quats)
        assert not result.valid

    @pytest.mark.regression
    def test_non_unit_quaternions(self):
        from validation.contracts import validate_gaussians

        means, scales, opacities, _ = self._make_valid_gaussians()
        quats = np.ones((50, 4), dtype=np.float32) * 5.0  # far from unit
        result = validate_gaussians(means, scales, opacities, quats)
        assert not result.valid

    @pytest.mark.regression
    def test_wrong_means_shape(self):
        from validation.contracts import validate_gaussians

        _, scales, opacities, quats = self._make_valid_gaussians()
        means = np.zeros((50, 2), dtype=np.float32)
        result = validate_gaussians(means, scales, opacities, quats)
        assert not result.valid

    @pytest.mark.regression
    def test_single_gaussian(self):
        from validation.contracts import validate_gaussians

        means, scales, opacities, quats = self._make_valid_gaussians(n=1)
        result = validate_gaussians(means, scales, opacities, quats)
        assert result.valid

    @pytest.mark.regression
    def test_large_gaussian_count(self):
        from validation.contracts import validate_gaussians

        means, scales, opacities, quats = self._make_valid_gaussians(n=100_000)
        result = validate_gaussians(means, scales, opacities, quats)
        assert result.valid


# ═══════════════════════════════════════════════════════════════════════════
# validate_sensor_data
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateSensorData:

    def _make_valid_sensor_data(self):
        rng = np.random.default_rng(55)
        n = 100
        ts = np.linspace(0.0, 1.0, n)
        accel = np.zeros((n, 3), dtype=np.float64)
        accel[:, 2] = -9.81
        accel += rng.normal(0, 0.05, size=(n, 3))

        raw_q = rng.normal(size=(n, 4))
        quats = raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True)

        return {
            "accel_timestamps": ts,
            "accel_xyz": accel,
            "orientation_timestamps": ts.copy(),
            "orientation_quats": quats,
        }

    @pytest.mark.regression
    def test_valid_sensor_data(self):
        from validation.contracts import validate_sensor_data

        data = self._make_valid_sensor_data()
        result = validate_sensor_data(data)
        assert result.valid

    @pytest.mark.regression
    def test_empty_dict(self):
        from validation.contracts import validate_sensor_data

        result = validate_sensor_data({})
        assert not result.valid

    @pytest.mark.regression
    def test_non_monotonic_timestamps(self):
        from validation.contracts import validate_sensor_data

        data = self._make_valid_sensor_data()
        data["accel_timestamps"][50] = 0.0  # jump backward
        result = validate_sensor_data(data)
        assert not result.valid
        assert any("non-monotonic" in e for e in result.errors)

    @pytest.mark.regression
    def test_nan_in_accel(self):
        from validation.contracts import validate_sensor_data

        data = self._make_valid_sensor_data()
        data["accel_xyz"][10, 1] = np.nan
        result = validate_sensor_data(data)
        assert not result.valid

    @pytest.mark.regression
    def test_wrong_quat_shape(self):
        from validation.contracts import validate_sensor_data

        data = self._make_valid_sensor_data()
        data["orientation_quats"] = np.ones((100, 3))  # wrong shape
        result = validate_sensor_data(data)
        assert not result.valid

    @pytest.mark.regression
    def test_sampling_gap_warning(self):
        from validation.contracts import validate_sensor_data

        data = self._make_valid_sensor_data()
        # Insert a huge gap
        ts = data["accel_timestamps"].copy()
        ts[50:] += 100.0  # massive jump
        data["accel_timestamps"] = ts
        result = validate_sensor_data(data)
        # Should still be valid but with a warning
        assert result.valid
        assert len(result.warnings) > 0

    @pytest.mark.regression
    def test_bare_timestamps_key(self):
        from validation.contracts import validate_sensor_data

        data = {
            "timestamps": np.linspace(0.0, 1.0, 50),
            "accel_xyz": np.zeros((50, 3)),
        }
        result = validate_sensor_data(data)
        assert result.valid

    @pytest.mark.regression
    def test_single_sample_warning(self):
        from validation.contracts import validate_sensor_data

        data = {
            "accel_timestamps": np.array([0.0]),
            "accel_xyz": np.zeros((1, 3)),
        }
        result = validate_sensor_data(data)
        assert result.valid  # single sample is valid but warns
        assert len(result.warnings) > 0
