"""Comprehensive tests for depth estimation, reconstruction, and sensor modules.

Covers:
- DA3 unified: rotation-to-quaternion, perceptual hashing, dedup, confidence filtering,
  unprojection, streaming PLY, pose validation, point cloud validation
- Depth alignment: RANSAC scale/shift, rotation delta
- Point cloud: IncrementalPlyWriter
- COLMAP runner: vocab tree detection, rotation-to-quaternion, model selection
- Landmarks: landmark quality scoring
- FLAME fitter: frame stem normalization, frontal frame selection, yaw estimation,
  landmark weights, scale estimation, quat-to-rotation
- Face segmentation: mask quality scoring, largest component, morphological ops
- COLMAP priors: rotation_matrix_to_colmap_quat, prior file generation
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Import helper: bypass package __init__.py which pulls in heavy deps
# (open3d via depth/__init__, mediapipe via reconstruction/__init__).
# ---------------------------------------------------------------------------

_mod_cache: dict[str, object] = {}


def _load(dotted: str):
    """Load a single source module by dotted path, bypassing __init__.py."""
    if dotted in _mod_cache:
        return _mod_cache[dotted]

    parts = dotted.split(".")
    file_path = SRC_DIR.joinpath(*parts).with_suffix(".py")
    if not file_path.exists():
        pytest.skip(f"Source file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(dotted, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        pytest.skip(f"Cannot import {dotted}: {exc}")

    _mod_cache[dotted] = mod
    return mod


# Convenience accessors
def _da3():
    return _load("depth.da3_unified")


def _alignment():
    return _load("depth.depth_alignment")


def _pointcloud():
    return _load("depth.pointcloud")


def _colmap_runner():
    return _load("reconstruction.colmap_runner")


def _landmarks():
    return _load("reconstruction.landmarks")


def _flame_fitter():
    return _load("reconstruction.flame_fitter")


def _face_seg():
    return _load("reconstruction.face_segmentation")


def _colmap_priors():
    return _load("sensors.colmap_priors")


# ===========================================================================
# DA3 Unified tests
# ===========================================================================

@pytest.mark.regression
class TestDA3RotationMatrixToQuaternion:
    """Tests for _rotation_matrix_to_quaternion in da3_unified."""

    def test_identity_matrix(self):
        fn = _da3()._rotation_matrix_to_quaternion
        q = fn(np.eye(3))
        np.testing.assert_allclose(np.abs(q), [1, 0, 0, 0], atol=1e-10)

    def test_90_deg_rotation_x(self):
        fn = _da3()._rotation_matrix_to_quaternion
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        q = fn(R)
        assert q.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
        R_back = _quat_to_rotmat(q)
        np.testing.assert_allclose(R_back, R, atol=1e-10)

    def test_90_deg_rotation_y(self):
        fn = _da3()._rotation_matrix_to_quaternion
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
        q = fn(R)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
        np.testing.assert_allclose(_quat_to_rotmat(q), R, atol=1e-10)

    def test_90_deg_rotation_z(self):
        fn = _da3()._rotation_matrix_to_quaternion
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        q = fn(R)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
        np.testing.assert_allclose(_quat_to_rotmat(q), R, atol=1e-10)

    def test_180_deg_rotation_x(self):
        fn = _da3()._rotation_matrix_to_quaternion
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        q = fn(R)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
        np.testing.assert_allclose(_quat_to_rotmat(q), R, atol=1e-10)

    def test_arbitrary_rotation(self):
        fn = _da3()._rotation_matrix_to_quaternion
        angle = np.pi / 4
        axis = np.array([1, 1, 1], dtype=np.float64) / np.sqrt(3)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        q = fn(R)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
        np.testing.assert_allclose(_quat_to_rotmat(q), R, atol=1e-8)

    def test_output_is_unit_quaternion_many(self):
        fn = _da3()._rotation_matrix_to_quaternion
        rng = np.random.default_rng(42)
        for _ in range(20):
            A = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            q = fn(Q)
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)


def _quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


@pytest.mark.regression
class TestDA3PerceptualHash:
    """Tests for _compute_frame_hash and _hamming_distance."""

    def test_identical_images_zero_distance(self):
        m = _da3()
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        h1 = m._compute_frame_hash(img)
        h2 = m._compute_frame_hash(img)
        assert m._hamming_distance(h1, h2) == 0

    def test_hash_shape(self):
        m = _da3()
        img = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        h = m._compute_frame_hash(img)
        assert h.shape == (64,)
        assert h.dtype == np.uint8

    def test_different_images_nonzero_distance(self):
        m = _da3()
        rng = np.random.default_rng(10)
        img1 = rng.integers(0, 128, (64, 64, 3), dtype=np.uint8)
        img2 = rng.integers(128, 255, (64, 64, 3), dtype=np.uint8)
        h1 = m._compute_frame_hash(img1)
        h2 = m._compute_frame_hash(img2)
        assert m._hamming_distance(h1, h2) > 0

    def test_grayscale_input(self):
        m = _da3()
        gray = np.full((64, 64), 100, dtype=np.uint8)
        h = m._compute_frame_hash(gray)
        assert h.shape == (64,)

    def test_similar_images_small_distance(self):
        m = _da3()
        base = np.full((64, 64, 3), 128, dtype=np.uint8)
        noisy = base.copy()
        noisy[10:20, 10:20] = 135
        h1 = m._compute_frame_hash(base)
        h2 = m._compute_frame_hash(noisy)
        assert m._hamming_distance(h1, h2) < 20


@pytest.mark.regression
class TestDA3AdaptiveConfidence:
    """Tests for _compute_adaptive_conf_threshold."""

    def test_basic_percentile(self):
        fn = _da3()._compute_adaptive_conf_threshold
        rng = np.random.default_rng(42)
        confs = [rng.uniform(0.0, 1.0, (10, 10)).astype(np.float32) for _ in range(3)]
        thresh = fn(confs, [True, True, True], percentile=50.0)
        assert 0.1 <= thresh <= 0.8

    def test_no_valid_frames_returns_min(self):
        fn = _da3()._compute_adaptive_conf_threshold
        confs = [np.ones((10, 10), dtype=np.float32)]
        thresh = fn(confs, [False], min_threshold=0.1)
        assert thresh == 0.1

    def test_clamp_to_min_max(self):
        fn = _da3()._compute_adaptive_conf_threshold
        confs = [np.zeros((10, 10), dtype=np.float32)]
        thresh = fn(confs, [True], percentile=50.0, min_threshold=0.2, max_threshold=0.9)
        assert thresh >= 0.2

    def test_high_confidence_maps(self):
        fn = _da3()._compute_adaptive_conf_threshold
        confs = [np.ones((10, 10), dtype=np.float32) * 0.95 for _ in range(5)]
        thresh = fn(confs, [True] * 5, percentile=30.0, min_threshold=0.1, max_threshold=0.8)
        assert thresh <= 0.8


@pytest.mark.regression
class TestDA3Unprojection:
    """Tests for _unproject_depth_to_points."""

    def test_identity_camera(self):
        fn = _da3()._unproject_depth_to_points
        H, W = 8, 8
        depth = np.ones((H, W), dtype=np.float32) * 2.0
        K = np.array([[4, 0, 4], [0, 4, 4], [0, 0, 1]], dtype=np.float64)
        ext = np.eye(4, dtype=np.float64)
        img = np.full((H, W, 3), 128, dtype=np.uint8)
        pts, cols = fn(depth, K, ext, img, stride=1)
        assert pts.shape[1] == 3
        assert len(pts) > 0
        np.testing.assert_allclose(pts[:, 2], 2.0, atol=1e-10)

    def test_empty_with_zero_depth(self):
        fn = _da3()._unproject_depth_to_points
        depth = np.zeros((8, 8), dtype=np.float32)
        K = np.eye(3, dtype=np.float64) * 100; K[2, 2] = 1
        ext = np.eye(4, dtype=np.float64)
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        pts, cols = fn(depth, K, ext, img, stride=1)
        assert pts.shape == (0, 3)

    def test_stride_reduces_points(self):
        fn = _da3()._unproject_depth_to_points
        H, W = 16, 16
        depth = np.ones((H, W), dtype=np.float32)
        K = np.array([[10, 0, 8], [0, 10, 8], [0, 0, 1]], dtype=np.float64)
        ext = np.eye(4, dtype=np.float64)
        img = np.full((H, W, 3), 128, dtype=np.uint8)
        pts1, _ = fn(depth, K, ext, img, stride=1)
        pts4, _ = fn(depth, K, ext, img, stride=4)
        assert len(pts4) < len(pts1)

    def test_confidence_filtering(self):
        fn = _da3()._unproject_depth_to_points
        H, W = 8, 8
        depth = np.ones((H, W), dtype=np.float32)
        K = np.array([[4, 0, 4], [0, 4, 4], [0, 0, 1]], dtype=np.float64)
        ext = np.eye(4, dtype=np.float64)
        img = np.full((H, W, 3), 128, dtype=np.uint8)
        conf = np.ones((H, W), dtype=np.float32) * 0.1
        pts, _ = fn(depth, K, ext, img, confidence=conf, conf_threshold=0.5, stride=1)
        assert len(pts) == 0

    def test_3x4_extrinsics(self):
        fn = _da3()._unproject_depth_to_points
        depth = np.ones((4, 4), dtype=np.float32) * 1.5
        K = np.array([[4, 0, 2], [0, 4, 2], [0, 0, 1]], dtype=np.float64)
        ext = np.eye(4, dtype=np.float64)[:3, :]
        img = np.full((4, 4, 3), 100, dtype=np.uint8)
        pts, _ = fn(depth, K, ext, img, stride=1)
        assert pts.shape[1] == 3 and len(pts) > 0


@pytest.mark.regression
class TestDA3PoseValidation:
    """Tests for _validate_poses."""

    def test_all_valid(self):
        fn = _da3()._validate_poses
        exts = [np.eye(4, dtype=np.float64) for _ in range(3)]
        valid = [True, True, True]
        assert fn(exts, valid, ["f0", "f1", "f2"]) == 3

    def test_nan_pose_invalidated(self):
        fn = _da3()._validate_poses
        ext_nan = np.eye(4, dtype=np.float64); ext_nan[0, 0] = np.nan
        valid = [True, True, True]
        fn([np.eye(4), ext_nan, np.eye(4)], valid, ["f0", "f1", "f2"])
        assert valid[1] is False

    def test_zero_rotation_invalidated(self):
        fn = _da3()._validate_poses
        ext_zero = np.zeros((4, 4), dtype=np.float64); ext_zero[3, 3] = 1.0
        valid = [True, True]
        fn([np.eye(4), ext_zero], valid, ["f0", "f1"])
        assert valid[1] is False

    def test_already_invalid_skipped(self):
        fn = _da3()._validate_poses
        valid = [False]
        assert fn([np.eye(4)], valid, ["f0"]) == 0


@pytest.mark.regression
class TestDA3PointCloudValidation:
    """Tests for _validate_point_cloud."""

    def test_valid_points_unchanged(self):
        fn = _da3()._validate_point_cloud
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        p, c = fn(pts, cols)
        np.testing.assert_array_equal(p, pts)

    def test_nan_points_filtered(self):
        fn = _da3()._validate_point_cloud
        pts = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, 9]], dtype=np.float64)
        cols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        p, c = fn(pts, cols)
        assert len(p) == 2

    def test_empty_input(self):
        fn = _da3()._validate_point_cloud
        p, c = fn(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8))
        assert len(p) == 0


@pytest.mark.regression
class TestDA3StreamingPlyWriter:
    """Tests for _StreamingPlyWriter."""

    def test_write_and_read_ply(self, tmp_path):
        Writer = _da3()._StreamingPlyWriter
        ply_path = tmp_path / "test.ply"
        writer = Writer(ply_path)
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        writer.add_points(pts, cols)
        count = writer.finalize()
        assert count == 2
        assert ply_path.exists()
        with open(ply_path, "rb") as f:
            header = b""
            while b"end_header" not in header:
                header += f.read(1)
            assert b"element vertex 2" in header

    def test_multiple_batches(self, tmp_path):
        Writer = _da3()._StreamingPlyWriter
        ply_path = tmp_path / "multi.ply"
        writer = Writer(ply_path)
        for i in range(5):
            writer.add_points(
                np.array([[float(i), 0, 0]], dtype=np.float64),
                np.array([[128, 128, 128]], dtype=np.uint8),
            )
        assert writer.finalize() == 5

    def test_empty_batch_skipped(self, tmp_path):
        Writer = _da3()._StreamingPlyWriter
        ply_path = tmp_path / "empty.ply"
        writer = Writer(ply_path)
        writer.add_points(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
        writer.add_points(np.array([[1, 2, 3]], dtype=np.float64),
                          np.array([[100, 100, 100]], dtype=np.uint8))
        assert writer.finalize() == 1


@pytest.mark.regression
class TestDA3WritePly:
    def test_write_small_cloud(self, tmp_path):
        ply_path = tmp_path / "small.ply"
        pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        cols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        _da3()._write_ply(ply_path, pts, cols)
        assert ply_path.exists() and ply_path.stat().st_size > 0


@pytest.mark.regression
class TestDA3ColmapBinaryWriters:
    def test_write_cameras_binary(self, tmp_path):
        path = tmp_path / "cameras.bin"
        cam = {"camera_id": 1, "width": 640, "height": 480,
               "params": [525.0, 525.0, 320.0, 240.0]}
        _da3()._write_cameras_binary(path, cam)
        with open(path, "rb") as f:
            assert struct.unpack("<Q", f.read(8))[0] == 1

    def test_write_images_binary(self, tmp_path):
        path = tmp_path / "images.bin"
        images = [{"image_id": 1, "qvec": [1, 0, 0, 0], "tvec": [0, 0, 0],
                    "camera_id": 1, "name": "frame_000000.png"}]
        _da3()._write_images_binary(path, images)
        with open(path, "rb") as f:
            assert struct.unpack("<Q", f.read(8))[0] == 1

    def test_write_empty_images(self, tmp_path):
        path = tmp_path / "images_empty.bin"
        _da3()._write_images_binary(path, [])
        with open(path, "rb") as f:
            assert struct.unpack("<Q", f.read(8))[0] == 0

    def test_write_points3d_binary(self, tmp_path):
        path = tmp_path / "points3D.bin"
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        _da3()._write_points3d_binary(path, pts, cols)
        with open(path, "rb") as f:
            assert struct.unpack("<Q", f.read(8))[0] == 2

    def test_write_empty_points3d(self, tmp_path):
        path = tmp_path / "pts_empty.bin"
        _da3()._write_points3d_binary(path, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
        with open(path, "rb") as f:
            assert struct.unpack("<Q", f.read(8))[0] == 0

    def test_degenerate_quaternion_handled(self, tmp_path):
        path = tmp_path / "images_degen.bin"
        images = [{"image_id": 1, "qvec": [0, 0, 0, 0], "tvec": [0, 0, 0],
                    "camera_id": 1, "name": "test.png"}]
        _da3()._write_images_binary(path, images)
        assert path.exists()


@pytest.mark.regression
class TestDA3ProvidesEnoughViews:
    def test_missing_file_returns_false(self, tmp_path):
        assert _da3().da3_provides_enough_views(tmp_path, min_views=5) is False

    def test_enough_views(self, tmp_path):
        d = tmp_path / "colmap" / "sparse" / "0"; d.mkdir(parents=True)
        with open(d / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", 10))
        assert _da3().da3_provides_enough_views(tmp_path, min_views=5) is True

    def test_not_enough_views(self, tmp_path):
        d = tmp_path / "colmap" / "sparse" / "0"; d.mkdir(parents=True)
        with open(d / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", 3))
        assert _da3().da3_provides_enough_views(tmp_path, min_views=5) is False


@pytest.mark.regression
class TestMultiFrameUnprojection:
    def test_merge_two_frames(self):
        fn = _da3()._unproject_multiple_frames
        H, W = 4, 4
        K = np.array([[4, 0, 2], [0, 4, 2], [0, 0, 1]], dtype=np.float64)
        ext = np.eye(4, dtype=np.float64)
        depth = np.ones((H, W), dtype=np.float32)
        img = np.full((H, W, 3), 128, dtype=np.uint8)
        pts, cols = fn([depth, depth], [K, K], [ext, ext], [img, img], [None, None], stride=2)
        assert pts.shape[1] == 3 and len(pts) > 0

    def test_empty_depths(self):
        pts, cols = _da3()._unproject_multiple_frames([], [], [], [], [])
        assert pts.shape == (0, 3)


# ===========================================================================
# Depth Alignment tests
# ===========================================================================

@pytest.mark.regression
class TestRANSACScaleShift:
    def test_perfect_linear(self):
        fn = _alignment()._ransac_scale_shift
        mono = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metric = 2.0 * mono + 0.5
        s, b, mask = fn(mono, metric, n_iterations=500, inlier_threshold=0.1)
        np.testing.assert_allclose(s, 2.0, atol=0.01)
        np.testing.assert_allclose(b, 0.5, atol=0.01)
        assert mask.sum() == 5

    def test_with_outliers(self):
        fn = _alignment()._ransac_scale_shift
        rng = np.random.default_rng(42)
        n = 50
        mono = rng.uniform(1, 5, n)
        metric = 3.0 * mono + 1.0
        outlier_idx = rng.choice(n, 10, replace=False)
        metric[outlier_idx] += rng.uniform(5, 20, 10)
        s, b, mask = fn(mono, metric, n_iterations=2000, inlier_threshold=0.5)
        np.testing.assert_allclose(s, 3.0, atol=0.3)
        np.testing.assert_allclose(b, 1.0, atol=0.5)

    def test_initial_guess_used(self):
        fn = _alignment()._ransac_scale_shift
        mono = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        metric = 2.0 * mono + 1.0
        s, b, _ = fn(mono, metric, n_iterations=100,
                      initial_guess=(2.0, 1.0), inlier_threshold=0.1)
        np.testing.assert_allclose(s, 2.0, atol=0.05)
        np.testing.assert_allclose(b, 1.0, atol=0.05)

    def test_weighted_fit(self):
        fn = _alignment()._ransac_scale_shift
        mono = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        metric = 1.5 * mono + 0.3
        weights = np.ones(6)
        s, b, _ = fn(mono, metric, weights=weights, n_iterations=500, inlier_threshold=0.1)
        np.testing.assert_allclose(s, 1.5, atol=0.05)
        np.testing.assert_allclose(b, 0.3, atol=0.1)


@pytest.mark.regression
class TestRotationDelta:
    def test_identical_quaternions_zero(self):
        fn = _alignment()._rotation_delta
        q = np.array([1, 0, 0, 0], dtype=np.float64)
        np.testing.assert_allclose(fn(q, q), 0.0, atol=1e-10)

    def test_opposite_sign_zero(self):
        fn = _alignment()._rotation_delta
        q = np.array([1, 0, 0, 0], dtype=np.float64)
        np.testing.assert_allclose(fn(q, -q), 0.0, atol=1e-10)

    def test_90_degree_delta(self):
        fn = _alignment()._rotation_delta
        q1 = np.array([1, 0, 0, 0], dtype=np.float64)
        q2 = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], dtype=np.float64)
        np.testing.assert_allclose(fn(q1, q2), 90.0, atol=1.0)


@pytest.mark.regression
class TestComputeConfidence:
    def test_no_sparse_points_returns_zeros(self):
        fn = _alignment().compute_confidence
        conf = fn(np.ones((10, 10), np.float32), np.ones((10, 10), np.float32),
                  np.array([]), np.zeros((0, 2)))
        assert conf.shape == (10, 10) and np.all(conf == 0)

    def test_output_shape_and_range(self):
        fn = _alignment().compute_confidence
        conf = fn(np.ones((20, 20), np.float32), np.ones((20, 20), np.float32) * 2.0,
                  np.array([2.0, 2.1, 1.9]),
                  np.array([[5, 5], [10, 10], [15, 15]], dtype=np.float64))
        assert conf.shape == (20, 20)
        assert np.all(conf >= 0) and np.all(conf <= 1)


# ===========================================================================
# Point Cloud tests (IncrementalPlyWriter)
# ===========================================================================

@pytest.mark.regression
class TestIncrementalPlyWriter:
    def _writer_cls(self):
        try:
            return _pointcloud().IncrementalPlyWriter
        except Exception:
            pytest.skip("Cannot import IncrementalPlyWriter (open3d dependency)")

    def test_basic_write(self, tmp_path):
        Writer = self._writer_cls()
        path = tmp_path / "inc.ply"
        w = Writer(path)
        w.add_frame(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                     np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]], dtype=np.float64))
        assert w.finalize() == 2 and path.exists()

    def test_uint8_colors(self, tmp_path):
        Writer = self._writer_cls()
        w = Writer(tmp_path / "u8.ply")
        w.add_frame(np.array([[1, 2, 3]], dtype=np.float32),
                     np.array([[255, 128, 0]], dtype=np.uint8))
        assert w.finalize() == 1

    def test_empty_frame_no_count(self, tmp_path):
        Writer = self._writer_cls()
        w = Writer(tmp_path / "e.ply")
        w.add_frame(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float64))
        w.add_frame(np.array([[1, 1, 1]], np.float32), np.array([[100, 100, 100]], np.uint8))
        assert w.finalize() == 1

    def test_total_points_property(self, tmp_path):
        Writer = self._writer_cls()
        w = Writer(tmp_path / "prop.ply")
        w.add_frame(np.ones((10, 3), np.float32), np.ones((10, 3), np.float64) * 0.5)
        assert w.total_points == 10


# ===========================================================================
# COLMAP Runner tests
# ===========================================================================

@pytest.mark.regression
class TestColmapRunnerHelpers:
    def test_find_vocab_tree_not_found(self, tmp_path):
        assert _colmap_runner()._find_vocab_tree(tmp_path) is None

    def test_find_vocab_tree_in_workspace(self, tmp_path):
        vt = tmp_path / "vocab_tree.bin"; vt.write_bytes(b"fake")
        assert _colmap_runner()._find_vocab_tree(tmp_path) == vt

    def test_find_vocab_tree_in_parent(self, tmp_path):
        sub = tmp_path / "ws"; sub.mkdir()
        vt = tmp_path / "vocab_tree.bin"; vt.write_bytes(b"fake")
        assert _colmap_runner()._find_vocab_tree(sub) == vt

    def test_rotation_matrix_to_quaternion_identity(self):
        q = _colmap_runner()._rotation_matrix_to_quaternion(np.eye(3))
        np.testing.assert_allclose(np.abs(np.array(q)), [1, 0, 0, 0], atol=1e-10)

    def test_rotation_matrix_to_quaternion_roundtrip(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        q = _colmap_runner()._rotation_matrix_to_quaternion(R)
        w, x, y, z = q
        R_back = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
        np.testing.assert_allclose(R_back, R, atol=1e-10)

    def test_select_best_model_single(self, tmp_path):
        sparse = tmp_path / "sparse"; m0 = sparse / "0"; m0.mkdir(parents=True)
        (m0 / "images.bin").write_bytes(b"x" * 100)
        assert _colmap_runner()._select_best_model(sparse) == m0

    def test_select_best_model_no_subdirs(self, tmp_path):
        sparse = tmp_path / "sparse"; sparse.mkdir(parents=True)
        assert _colmap_runner()._select_best_model(sparse) == sparse

    def test_hloc_available_returns_bool(self):
        assert isinstance(_colmap_runner()._hloc_available(), bool)

    def test_generate_sequential_pairs(self, tmp_path):
        img_dir = tmp_path / "images"; img_dir.mkdir()
        for i in range(5):
            (img_dir / f"frame_{i:04d}.png").write_bytes(b"fake")
        pairs_path = tmp_path / "pairs.txt"
        _colmap_runner()._generate_sequential_pairs(img_dir, pairs_path, overlap=2)
        lines = pairs_path.read_text().strip().split("\n")
        assert len(lines) > 0
        for line in lines:
            assert len(line.strip().split()) == 2


@pytest.mark.regression
class TestQuatToRot:
    """Tests for _quat_to_rot in flame_fitter module."""

    def test_identity(self):
        R = _flame_fitter()._quat_to_rot(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        w = np.sqrt(2) / 2
        R = _flame_fitter()._quat_to_rot(w, 0, 0, w)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# ===========================================================================
# Landmarks tests
# ===========================================================================

@pytest.mark.regression
class TestLandmarkQuality:
    def test_large_face_high_confidence(self):
        fn = _landmarks()._compute_landmark_quality
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[0] = [100, 100]; lm[1] = [540, 380]
        for i in range(2, 478):
            lm[i] = [320, 240]
        assert 0.5 < fn(lm, 0.9, 480, 640) <= 1.0

    def test_very_small_face_low_quality(self):
        fn = _landmarks()._compute_landmark_quality
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[0] = [300, 230]; lm[1] = [320, 250]
        for i in range(2, 478):
            lm[i] = [310, 240]
        assert fn(lm, 0.9, 480, 640) < 0.9

    def test_zero_confidence(self):
        fn = _landmarks()._compute_landmark_quality
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[:, 0] = np.linspace(100, 500, 478)
        lm[:, 1] = np.linspace(100, 400, 478)
        q = fn(lm, 0.0, 480, 640)
        assert 0.0 <= q <= 1.0

    def test_output_clipped_0_1(self):
        fn = _landmarks()._compute_landmark_quality
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[:, 0] = np.linspace(50, 590, 478)
        lm[:, 1] = np.linspace(50, 430, 478)
        assert 0.0 <= fn(lm, 1.0, 480, 640) <= 1.0


# ===========================================================================
# FLAME Fitter tests
# ===========================================================================

@pytest.mark.regression
class TestFlameFitterHelpers:
    def test_normalize_frame_stem_basic(self):
        fn = _flame_fitter()._normalize_frame_stem
        assert fn("frame_000001.png") == "frame_000001"
        assert fn("FRAME_001.JPG") == "frame_001"
        assert fn("frames_srgb/frame_001.png") == "frame_001"

    def test_normalize_frame_stem_path(self):
        assert _flame_fitter()._normalize_frame_stem("some/nested/dir/image.tiff") == "image"

    def test_estimate_face_yaw_frontal(self):
        fn = _flame_fitter()._estimate_face_yaw_from_landmarks
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[1] = [320, 240]; lm[234] = [200, 240]; lm[454] = [440, 240]
        assert fn(lm, 640) < 15.0

    def test_estimate_face_yaw_profile(self):
        fn = _flame_fitter()._estimate_face_yaw_from_landmarks
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[1] = [450, 240]; lm[234] = [200, 240]; lm[454] = [440, 240]
        assert fn(lm, 640) > 10.0

    def test_estimate_face_yaw_too_few(self):
        assert _flame_fitter()._estimate_face_yaw_from_landmarks(np.zeros((100, 2)), 640) == 90.0

    def test_select_frontal_frame_empty(self):
        assert _flame_fitter()._select_frontal_frame([]) is None

    def test_select_frontal_frame_picks_most_frontal(self):
        fn = _flame_fitter()._select_frontal_frame
        lm_front = np.zeros((478, 2), dtype=np.float32)
        lm_front[1] = [320, 240]; lm_front[234] = [200, 240]; lm_front[454] = [440, 240]
        lm_side = np.zeros((478, 2), dtype=np.float32)
        lm_side[1] = [450, 240]; lm_side[234] = [200, 240]; lm_side[454] = [440, 240]
        data = [
            {"landmarks_2d": lm_front, "image_width": 640},
            {"landmarks_2d": lm_side, "image_width": 640},
        ]
        assert fn(data) == 0

    def test_build_landmark_weights(self):
        torch = pytest.importorskip("torch")
        m = _flame_fitter()
        indices = [33, 0, 1, 500]
        weights = m._build_landmark_weights(indices, torch.device("cpu"))
        assert weights.shape == (4,)
        assert float(weights[0]) == m._LANDMARK_REGION_WEIGHTS["eye"]

    def test_estimate_scale_from_ipd_normal(self):
        fn = _flame_fitter()._estimate_scale_from_ipd
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[468] = [300, 240]; lm[473] = [363, 240]
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        scale = fn(lm, K, depth_at_face=0.5)
        assert isinstance(scale, float) and scale > 0

    def test_estimate_scale_from_ipd_too_few(self):
        assert _flame_fitter()._estimate_scale_from_ipd(
            np.zeros((100, 2), np.float32), np.eye(3) * 500) == 1.0

    def test_match_landmarks_to_cameras_basic(self):
        fn = _flame_fitter()._match_landmarks_to_cameras
        with tempfile.TemporaryDirectory() as tmpdir:
            lm_file = Path(tmpdir) / "frame_001.json"
            lm_file.write_text(json.dumps({"detected": True, "image": "frame_001.png"}))
            colmap_images = {
                "frame_001.png": {"camera_id": 1, "rotation_matrix": np.eye(3).tolist(),
                                  "translation": [[0], [0], [0]]},
            }
            matches = fn([lm_file], colmap_images)
            assert len(matches) == 1 and matches[0]["image_name"] == "frame_001.png"


# ===========================================================================
# Face Segmentation tests
# ===========================================================================

@pytest.mark.regression
class TestMaskQuality:
    def test_all_zero_mask(self):
        assert _face_seg().compute_mask_quality(np.zeros((100, 100), np.uint8)) == 0.0

    def test_all_ones_mask(self):
        assert _face_seg().compute_mask_quality(np.full((100, 100), 255, np.uint8)) == 0.0

    def test_good_circular_mask(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 255, -1)
        assert _face_seg().compute_mask_quality(mask) > 0.5

    def test_tiny_mask_low_quality(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[98:102, 98:102] = 255
        assert _face_seg().compute_mask_quality(mask) < 0.5

    def test_scattered_islands_lower_quality(self):
        fn = _face_seg().compute_mask_quality
        mask = np.zeros((200, 200), dtype=np.uint8)
        for x, y in [(30, 30), (170, 30), (30, 170), (170, 170), (100, 100)]:
            cv2.circle(mask, (x, y), 10, 255, -1)
        mask2 = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask2, (100, 100), 50, 255, -1)
        assert fn(mask) < fn(mask2)


@pytest.mark.regression
class TestKeepLargestComponent:
    def test_single_component_unchanged(self):
        fn = _face_seg()._keep_largest_component
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)
        assert np.count_nonzero(fn(mask)) == np.count_nonzero(mask)

    def test_removes_small_island(self):
        fn = _face_seg()._keep_largest_component
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)
        cv2.circle(mask, (90, 90), 5, 255, -1)
        assert np.count_nonzero(fn(mask)) < np.count_nonzero(mask)

    def test_empty_mask(self):
        assert np.count_nonzero(_face_seg()._keep_largest_component(
            np.zeros((50, 50), np.uint8))) == 0


# ===========================================================================
# COLMAP Priors tests
# ===========================================================================

@pytest.mark.regression
class TestColmapPriors:
    def test_identity_to_quat(self):
        q = _colmap_priors().rotation_matrix_to_colmap_quat(np.eye(3))
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_unit_quaternion_output(self):
        fn = _colmap_priors().rotation_matrix_to_colmap_quat
        rng = np.random.default_rng(42)
        for _ in range(10):
            A = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            np.testing.assert_allclose(np.linalg.norm(fn(Q)), 1.0, atol=1e-10)

    def test_canonical_form_w_positive(self):
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        q = _colmap_priors().rotation_matrix_to_colmap_quat(R)
        assert q[0] >= 0

    def test_generate_priors_file(self, tmp_path):
        rots = np.stack([np.eye(3)] * 3)
        names = ["f001.jpg", "f002.jpg", "f003.jpg"]
        result = _colmap_priors().generate_colmap_priors(rots, names, tmp_path / "priors.txt")
        content = result.read_text()
        assert "f001.jpg" in content and "f003.jpg" in content
        for line in content.strip().split("\n"):
            if not line.startswith("#"):
                assert len(line.strip().split()) == 15

    def test_generate_priors_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Mismatch"):
            _colmap_priors().generate_colmap_priors(
                np.stack([np.eye(3)] * 2), ["f001.jpg"], tmp_path / "p.txt")

    def test_rotation_roundtrip_via_scipy(self):
        from scipy.spatial.transform import Rotation
        R_orig = Rotation.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        q = _colmap_priors().rotation_matrix_to_colmap_quat(R_orig)
        R_back = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        np.testing.assert_allclose(R_back, R_orig, atol=1e-10)
