"""Snapshot regression tests for Face3D pipeline stage outputs.

Uses Syrupy to snapshot structured summaries of pipeline data (COLMAP models,
Gaussian parameters, depth maps, sensor data, frame metadata, color correction,
landmarks, FLAME parameters, segmentation masks, pipeline config).

A custom serializer rounds numpy floats and converts arrays to stat dicts
(with distribution percentiles) so that snapshots are deterministic and
human-readable.

All fixtures use fixed seeds, so snapshots should be perfectly stable.
Run ``pytest --snapshot-update`` to regenerate after intentional changes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from syrupy.extensions.amber import AmberSnapshotExtension
from syrupy.extensions.amber.serializer import AmberDataSerializer

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Custom serializer: rounds floats and summarises numpy arrays
# ---------------------------------------------------------------------------

def _round_value(v: Any, decimals: int = 6) -> Any:
    """Round a single value if it is a float or numpy float."""
    if isinstance(v, (np.floating, float)):
        return round(float(v), decimals)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _summarise_ndarray(arr: np.ndarray, decimals: int = 6) -> dict:
    """Convert a numpy array to a JSON-friendly stats dict with distribution fingerprint."""
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
        flat = arr.ravel().astype(np.float64)
        summary["min"] = round(float(np.min(flat)), decimals)
        summary["max"] = round(float(np.max(flat)), decimals)
        summary["mean"] = round(float(np.mean(flat)), decimals)
        summary["std"] = round(float(np.std(flat)), decimals)
        summary["median"] = round(float(np.median(flat)), decimals)
        summary["p10"] = round(float(np.percentile(flat, 10)), decimals)
        summary["p25"] = round(float(np.percentile(flat, 25)), decimals)
        summary["p75"] = round(float(np.percentile(flat, 75)), decimals)
        summary["p90"] = round(float(np.percentile(flat, 90)), decimals)
    return summary


def _prepare_for_snapshot(obj: Any, decimals: int = 6) -> Any:
    """Recursively prepare an object for snapshot serialization."""
    if isinstance(obj, np.ndarray):
        return _summarise_ndarray(obj, decimals)
    if isinstance(obj, dict):
        return {k: _prepare_for_snapshot(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_prepare_for_snapshot(v, decimals) for v in obj]
    return _round_value(obj, decimals)


class Face3DDataSerializer(AmberDataSerializer):
    """Serializer that rounds numpy floats and summarises arrays."""

    @classmethod
    def _normalize(cls, data: Any) -> Any:
        return super()._normalize(_prepare_for_snapshot(data))


class Face3DSnapshotExtension(AmberSnapshotExtension):
    serializer_class = Face3DDataSerializer


@pytest.fixture
def snapshot_face3d(snapshot):
    return snapshot.use_extension(Face3DSnapshotExtension)


# ---------------------------------------------------------------------------
# Helper: extract a stable summary from data structures
# ---------------------------------------------------------------------------

def _colmap_cameras_summary(cameras: dict) -> dict:
    """Summarise COLMAP cameras dict for snapshot comparison."""
    result = {"num_cameras": len(cameras)}
    for cam_id, cam in sorted(cameras.items()):
        result[f"cam_{cam_id}"] = {
            "model": cam.model,
            "width": int(cam.width),
            "height": int(cam.height),
            "params": [round(float(p), 6) for p in cam.params],
        }
    return result


def _colmap_images_summary(images: dict) -> dict:
    """Summarise COLMAP images dict for snapshot comparison."""
    result = {"num_images": len(images)}
    for img_id, img in sorted(images.items()):
        result[f"img_{img_id}"] = {
            "name": img.name,
            "camera_id": int(img.camera_id),
            "qvec": [round(float(q), 6) for q in img.qvec],
            "tvec": [round(float(t), 6) for t in img.tvec],
            "num_points2d": len(img.xys),
        }
    return result


def _colmap_points_summary(points: dict) -> dict:
    """Summarise COLMAP points3D dict for snapshot comparison."""
    all_xyz = np.array([pt.xyz for pt in points.values()])
    all_errors = np.array([pt.error for pt in points.values()])
    return {
        "num_points": len(points),
        "xyz_stats": _summarise_ndarray(all_xyz),
        "error_stats": _summarise_ndarray(all_errors),
    }


def _gaussian_summary(model: Any) -> dict:
    """Summarise a GaussianModel for snapshot comparison."""
    pos = model.positions.detach().cpu().numpy()
    scales = model.scales.detach().cpu().numpy()
    opacities = model.opacities.detach().cpu().numpy()
    rotations = model.rotations.detach().cpu().numpy()
    colors_sh = model.colors_sh.detach().cpu().numpy()

    return {
        "num_gaussians": int(model.num_gaussians),
        "positions": _summarise_ndarray(pos),
        "scales": _summarise_ndarray(scales),
        "opacities": _summarise_ndarray(opacities),
        "rotations": _summarise_ndarray(rotations),
        "colors_sh": _summarise_ndarray(colors_sh),
        "rotation_norms": {
            "min": round(float(np.linalg.norm(rotations, axis=1).min()), 6),
            "max": round(float(np.linalg.norm(rotations, axis=1).max()), 6),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestColmapSnapshots:
    """Snapshot tests for COLMAP binary model I/O."""

    def test_cameras_snapshot(self, sample_colmap_model, snapshot_face3d):
        from utils.colmap_io import read_cameras_binary

        cameras = read_cameras_binary(sample_colmap_model / "cameras.bin")
        summary = _colmap_cameras_summary(cameras)
        assert summary == snapshot_face3d

    def test_images_snapshot(self, sample_colmap_model, snapshot_face3d):
        from utils.colmap_io import read_images_binary

        images = read_images_binary(sample_colmap_model / "images.bin")
        summary = _colmap_images_summary(images)
        assert summary == snapshot_face3d

    def test_points3d_snapshot(self, sample_colmap_model, snapshot_face3d):
        from utils.colmap_io import read_points3d_binary

        points = read_points3d_binary(sample_colmap_model / "points3D.bin")
        summary = _colmap_points_summary(points)
        assert summary == snapshot_face3d

    def test_camera_intrinsics_matrix(self, sample_colmap_model, snapshot_face3d):
        from utils.colmap_io import read_cameras_binary, get_intrinsics_matrix

        cameras = read_cameras_binary(sample_colmap_model / "cameras.bin")
        cam = list(cameras.values())[0]
        K = get_intrinsics_matrix(cam)
        summary = {
            "K": [[round(float(v), 6) for v in row] for row in K],
        }
        assert summary == snapshot_face3d


@pytest.mark.regression
class TestGaussianSnapshots:
    """Snapshot tests for Gaussian splat model parameters."""

    def test_gaussian_stats(self, sample_gaussians, snapshot_face3d):
        summary = _gaussian_summary(sample_gaussians)
        assert summary == snapshot_face3d


@pytest.mark.regression
class TestDepthMapSnapshots:
    """Snapshot tests for depth map statistics."""

    def test_depth_map_stats(self, sample_depth_maps, snapshot_face3d):
        summaries = {}
        for i, path in enumerate(sample_depth_maps):
            depth = np.load(str(path))
            summaries[f"depth_{i}"] = _summarise_ndarray(depth)
        assert summaries == snapshot_face3d


@pytest.mark.regression
class TestSensorSnapshots:
    """Snapshot tests for IMU / sensor data array shapes and value ranges."""

    def test_imu_data_stats(self, sample_imu_data, snapshot_face3d):
        summary = _prepare_for_snapshot(sample_imu_data)
        assert summary == snapshot_face3d


@pytest.mark.regression
class TestFrameSnapshots:
    """Snapshot tests for frame metadata."""

    def test_frame_metadata(self, sample_frames, snapshot_face3d):
        summaries = {
            "num_frames": len(sample_frames),
            "frames": [],
        }
        for path in sample_frames:
            img = cv2.imread(str(path))
            summaries["frames"].append({
                "name": path.name,
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
                "channels": int(img.shape[2]),
                "dtype": str(img.dtype),
                "mean_intensity": round(float(img.mean()), 6),
            })
        assert summaries == snapshot_face3d


@pytest.mark.regression
class TestColorCorrectionSnapshots:
    """Snapshot tests for S-Log3 and sRGB color correction transfer functions."""

    def test_slog3_to_linear_curve(self, snapshot_face3d):
        from capture.color_correction import _slog3_to_linear

        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        linear = _slog3_to_linear(x.copy())
        summary = _summarise_ndarray(linear)
        assert summary == snapshot_face3d

    def test_linear_to_srgb_curve(self, snapshot_face3d):
        from capture.color_correction import _linear_to_srgb_curve

        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        srgb = _linear_to_srgb_curve(x)
        summary = _summarise_ndarray(srgb)
        assert summary == snapshot_face3d

    def test_slog3_to_srgb_full_pipeline(self, snapshot_face3d):
        from capture.color_correction import _slog3_to_linear, _linear_to_srgb_curve

        # Synthetic 64x64 frame with gradient values
        rng = np.random.default_rng(42)
        frame = rng.uniform(0.1, 0.9, size=(64, 64, 3)).astype(np.float32)

        linear = _slog3_to_linear(frame.copy())
        # Normalize to [0,1] for sRGB
        max_val = linear.max()
        if max_val > 0:
            linear_norm = np.clip(linear / max_val, 0.0, 1.0).astype(np.float32)
        else:
            linear_norm = linear
        srgb = _linear_to_srgb_curve(linear_norm)

        summary = {
            "input": _summarise_ndarray(frame),
            "linear": _summarise_ndarray(linear),
            "srgb": _summarise_ndarray(srgb),
        }
        assert summary == snapshot_face3d


@pytest.mark.regression
class TestLandmarkSnapshots:
    """Snapshot tests for landmark detection output structure."""

    def test_landmark_output_structure(self, snapshot_face3d):
        """Snapshot a synthetic landmark detection result matching the real structure."""
        rng = np.random.default_rng(478)
        num_landmarks = 478

        landmarks_2d = rng.uniform(50, 590, size=(num_landmarks, 2)).astype(np.float32)
        landmarks_3d = np.column_stack([
            landmarks_2d,
            rng.uniform(-50, 50, size=(num_landmarks,)).astype(np.float32),
        ])

        result = {
            "num_landmarks": num_landmarks,
            "landmarks_2d": _summarise_ndarray(landmarks_2d),
            "landmarks_3d": _summarise_ndarray(landmarks_3d),
            "confidence": round(float(rng.uniform(0.8, 1.0)), 6),
            "quality_score": round(float(rng.uniform(0.6, 1.0)), 6),
        }
        assert result == snapshot_face3d


@pytest.mark.regression
class TestFLAMEParamSnapshots:
    """Snapshot tests for FLAME parameter shapes and synthetic fit results."""

    def test_flame_param_shapes(self, snapshot_face3d):
        """Snapshot FLAME parameter shapes matching the fitter output structure."""
        rng = np.random.default_rng(5023)

        # Mimic FLAMEFitter.fit() output
        result = {
            "shape_params": _summarise_ndarray(
                rng.normal(0, 0.01, size=(1, 100)).astype(np.float32)
            ),
            "expression_params": _summarise_ndarray(
                rng.normal(0, 0.01, size=(1, 50)).astype(np.float32)
            ),
            "jaw_pose": _summarise_ndarray(
                rng.normal(0, 0.01, size=(1, 3)).astype(np.float32)
            ),
            "neck_pose": _summarise_ndarray(
                rng.normal(0, 0.01, size=(1, 3)).astype(np.float32)
            ),
            "global_rotation": _summarise_ndarray(
                rng.normal(0, 0.01, size=(1, 3)).astype(np.float32)
            ),
            "translation": _summarise_ndarray(
                rng.normal(0, 0.01, size=(1, 3)).astype(np.float32)
            ),
            "vertices_shape": [1, 5023, 3],
            "faces_shape": [9976, 3],
        }
        assert result == snapshot_face3d


@pytest.mark.regression
class TestSegmentationMaskSnapshots:
    """Snapshot tests for segmentation mask properties."""

    def test_synthetic_mask_stats(self, snapshot_face3d):
        """Snapshot a synthetic face mask (ellipse on black background)."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.ellipse(mask, (320, 240), (120, 160), 0, 0, 360, 255, -1)

        fg_pixels = int(np.count_nonzero(mask))
        total_pixels = mask.shape[0] * mask.shape[1]

        summary = {
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "value_range": [int(mask.min()), int(mask.max())],
            "fg_pixels": fg_pixels,
            "total_pixels": total_pixels,
            "fg_ratio": round(fg_pixels / total_pixels, 6),
        }
        assert summary == snapshot_face3d


@pytest.mark.regression
class TestPipelineConfigSnapshots:
    """Snapshot tests for pipeline configuration hyperparameters."""

    def test_key_hyperparameters(self, snapshot_face3d):
        """Snapshot critical pipeline config values to catch config drift."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "pipeline.yaml"
        if not config_path.exists():
            pytest.skip("pipeline.yaml not found")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        summary = {
            "capture.max_frames": config["capture"]["max_frames"],
            "capture.filtering.blur_threshold": config["capture"]["filtering"]["blur_threshold"],
            "capture.filtering.quick_mode": config["capture"]["filtering"]["quick_mode"],
            "reconstruction.use_da3_unified": config["reconstruction"]["use_da3_unified"],
            "reconstruction.da3.process_res": config["reconstruction"]["da3"]["process_res"],
            "reconstruction.da3.conf_threshold": config["reconstruction"]["da3"]["conf_threshold"],
            "reconstruction.flame.num_shape_coeffs": config["reconstruction"]["flame"]["num_shape_coeffs"],
            "reconstruction.flame.num_expression_coeffs": config["reconstruction"]["flame"]["num_expression_coeffs"],
            "splatting.training.iterations": config["splatting"]["training"]["iterations"],
            "splatting.training.max_num_gaussians": config["splatting"]["training"]["max_num_gaussians"],
            "photos.multi_lens.main_200mp_weight": config["photos"]["multi_lens"]["main_200mp_weight"],
        }
        assert summary == snapshot_face3d


@pytest.mark.regression
class TestDA3ConfidenceMapSnapshots:
    """Snapshot tests for DA3 confidence map statistics."""

    def test_confidence_map_stats(self, snapshot_face3d):
        """Snapshot synthetic confidence map distribution."""
        rng = np.random.default_rng(336)
        # Simulate DA3 confidence: mostly high values with some low-confidence edges
        conf = rng.beta(5, 1, size=(480, 640)).astype(np.float32)
        summary = _summarise_ndarray(conf)
        assert summary == snapshot_face3d


@pytest.mark.regression
class TestEdgeCaseSnapshots:
    """Snapshot tests for edge cases: empty, single-element, extreme values."""

    def test_empty_array(self, snapshot_face3d):
        """Empty array should produce shape/dtype but no stats."""
        arr = np.array([], dtype=np.float32)
        summary = _summarise_ndarray(arr)
        assert summary == snapshot_face3d

    def test_single_element_array(self, snapshot_face3d):
        """Single-element array: all stats should equal that element."""
        arr = np.array([3.14159], dtype=np.float64)
        summary = _summarise_ndarray(arr)
        assert summary == snapshot_face3d

    def test_very_large_values(self, snapshot_face3d):
        """Arrays with very large values (1e10) should serialize correctly."""
        rng = np.random.default_rng(999)
        arr = rng.uniform(1e9, 1e10, size=(10, 3)).astype(np.float64)
        summary = _summarise_ndarray(arr)
        assert summary == snapshot_face3d

    def test_negative_depth_values(self, snapshot_face3d):
        """Negative depth values (invalid) -- stats should capture the negatives."""
        rng = np.random.default_rng(111)
        depth = rng.uniform(-2.0, 5.0, size=(120, 160)).astype(np.float32)
        summary = _summarise_ndarray(depth)
        # Also flag the issue
        summary["has_negatives"] = bool(np.any(depth < 0))
        summary["negative_fraction"] = round(float(np.mean(depth < 0)), 6)
        assert summary == snapshot_face3d
