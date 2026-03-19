"""Unit tests for capture and data processing modules.

Tests cover:
- photo_processor: focal length snapping, EXIF extraction, auto-rotation, lens matching
- multi_res_processor: linear-to-sRGB conversion, image resizing, face crop extraction
- data_organizer: lens classification, resolution-based classification, sensor analysis,
  manifest structure, sensor-video matching

All tests are fast — no real files, no rawpy, no GPU. Synthetic data only.
"""

import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ═══════════════════════════════════════════════════════════════════════════
# Photo Processor Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestSnapFocalLength35mm:
    """Tests for _snap_focal_length_35mm in photo_processor."""

    def test_exact_13mm_ultrawide(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(13.0) == 13

    def test_exact_23mm_wide(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(23.0) == 23

    def test_exact_70mm_telephoto(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(70.0) == 70

    def test_exact_200mm_supertelephoto(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(200.0) == 200

    def test_snap_near_13mm(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(11.0) == 13
        assert _snap_focal_length_35mm(15.0) == 13

    def test_snap_near_23mm(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(20.0) == 23
        assert _snap_focal_length_35mm(25.0) == 23

    def test_snap_near_70mm(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(60.0) == 70
        assert _snap_focal_length_35mm(80.0) == 70

    def test_snap_near_200mm(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(180.0) == 200
        assert _snap_focal_length_35mm(250.0) == 200

    def test_midpoint_between_13_and_23(self):
        """At midpoint (18), should snap to nearest — 23 is closer."""
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(18.0) == 13  # |18-13|=5, |18-23|=5, min picks first

    def test_very_small_focal_length(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(1.0) == 13

    def test_very_large_focal_length(self):
        from capture.photo_processor import _snap_focal_length_35mm
        assert _snap_focal_length_35mm(500.0) == 200


@pytest.mark.regression
class TestAutoRotate:
    """Tests for _auto_rotate in photo_processor."""

    def _make_image(self):
        """Create a small asymmetric test image to verify rotation."""
        return np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ], dtype=np.uint8)  # 3x2x3

    def test_orientation_1_no_change(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 1)
        np.testing.assert_array_equal(result, img)

    def test_orientation_2_flip_lr(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 2)
        np.testing.assert_array_equal(result, np.fliplr(img))

    def test_orientation_3_rotate_180(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 3)
        np.testing.assert_array_equal(result, np.rot90(img, 2))

    def test_orientation_4_flip_ud(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 4)
        np.testing.assert_array_equal(result, np.flipud(img))

    def test_orientation_6_rotate_90cw(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 6)
        np.testing.assert_array_equal(result, np.rot90(img, 3))

    def test_orientation_8_rotate_90ccw(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 8)
        np.testing.assert_array_equal(result, np.rot90(img, 1))

    def test_invalid_orientation_returns_unchanged(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 99)
        np.testing.assert_array_equal(result, img)

    def test_orientation_5(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 5)
        np.testing.assert_array_equal(result, np.rot90(np.fliplr(img), 1))

    def test_orientation_7(self):
        from capture.photo_processor import _auto_rotate
        img = self._make_image()
        result = _auto_rotate(img, 7)
        np.testing.assert_array_equal(result, np.rot90(np.fliplr(img), 3))


@pytest.mark.regression
class TestMatchPhotoLensToVideo:
    """Tests for match_photo_lens_to_video in photo_processor."""

    def test_same_lens_wide(self):
        from capture.photo_processor import match_photo_lens_to_video
        exif = {"focal_length_35mm": 23.0, "focal_length": 6.3}
        video_meta = {"detected_lens": "wide"}
        lens, same = match_photo_lens_to_video(exif, video_meta)
        assert lens == "wide"
        assert same is True

    def test_different_lens(self):
        from capture.photo_processor import match_photo_lens_to_video
        exif = {"focal_length_35mm": 70.0}
        video_meta = {"detected_lens": "wide"}
        lens, same = match_photo_lens_to_video(exif, video_meta)
        assert lens == "telephoto"
        assert same is False

    def test_no_video_metadata_assumes_same(self):
        from capture.photo_processor import match_photo_lens_to_video
        exif = {"focal_length_35mm": 23.0}
        lens, same = match_photo_lens_to_video(exif, None)
        assert lens == "wide"
        assert same is True

    def test_no_focal_length_returns_unknown(self):
        from capture.photo_processor import match_photo_lens_to_video
        exif = {"focal_length_35mm": None, "focal_length": None}
        lens, same = match_photo_lens_to_video(exif, None)
        assert lens == "unknown"
        assert same is False

    def test_raw_focal_length_with_crop_factor(self):
        """When only physical focal length available, uses 6.7x crop factor."""
        from capture.photo_processor import match_photo_lens_to_video
        # 6.3mm * 6.7 ~= 42.2mm -> snaps to 23mm (closest)
        exif = {"focal_length_35mm": None, "focal_length": 6.3}
        lens, same = match_photo_lens_to_video(exif, None)
        # 6.3 * 6.7 = 42.21, snaps to 23 (distance 19.2) vs 70 (distance 27.8)
        assert lens == "wide"

    def test_ultrawide_lens_detection(self):
        from capture.photo_processor import match_photo_lens_to_video
        exif = {"focal_length_35mm": 13.0}
        lens, same = match_photo_lens_to_video(exif, None)
        assert lens == "ultrawide"

    def test_supertelephoto_lens_detection(self):
        from capture.photo_processor import match_photo_lens_to_video
        exif = {"focal_length_35mm": 200.0}
        lens, same = match_photo_lens_to_video(exif, None)
        assert lens == "supertelephoto"


@pytest.mark.regression
class TestS25UltraLensConstants:
    """Verify lens constant dictionaries are consistent."""

    def test_all_lenses_have_physical_focal(self):
        from capture.photo_processor import _S25_ULTRA_LENSES, _S25_ULTRA_PHYSICAL_FOCAL
        for focal_35 in _S25_ULTRA_LENSES:
            assert focal_35 in _S25_ULTRA_PHYSICAL_FOCAL, \
                f"Missing physical focal for {focal_35}mm"

    def test_known_focal_lengths_sorted(self):
        from capture.photo_processor import _KNOWN_FOCAL_LENGTHS_35MM
        assert _KNOWN_FOCAL_LENGTHS_35MM == sorted(_KNOWN_FOCAL_LENGTHS_35MM)

    def test_four_lenses(self):
        from capture.photo_processor import _S25_ULTRA_LENSES
        assert len(_S25_ULTRA_LENSES) == 4


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Res Processor Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestLinearToSrgb16bit:
    """Tests for _linear_to_srgb_16bit in multi_res_processor."""

    def test_zero_stays_zero(self):
        from capture.multi_res_processor import _linear_to_srgb_16bit
        inp = np.zeros((2, 2, 3), dtype=np.uint16)
        result = _linear_to_srgb_16bit(inp)
        np.testing.assert_array_equal(result, 0)

    def test_max_stays_near_max(self):
        """Linear 1.0 maps to sRGB 1.0, but float rounding may lose 1 LSB."""
        from capture.multi_res_processor import _linear_to_srgb_16bit
        inp = np.full((2, 2, 3), 65535, dtype=np.uint16)
        result = _linear_to_srgb_16bit(inp)
        assert np.all(result >= 65534)  # Allow +-1 for float precision

    def test_midpoint_gamma_expansion(self):
        """Linear 0.5 should map to ~sRGB 0.735 (188/256 * 256 ~ 48059)."""
        from capture.multi_res_processor import _linear_to_srgb_16bit
        # 0.5 in 16-bit = 32767
        inp = np.full((1, 1, 3), 32767, dtype=np.uint16)
        result = _linear_to_srgb_16bit(inp)
        # sRGB(0.5) = 1.055 * 0.5^(1/2.4) - 0.055 ≈ 0.7354
        expected_float = 1.055 * (0.5 ** (1.0 / 2.4)) - 0.055
        expected_16bit = int(expected_float * 65535)
        # Allow tolerance of +-100 due to rounding
        assert abs(int(result[0, 0, 0]) - expected_16bit) < 100

    def test_output_dtype_is_uint16(self):
        from capture.multi_res_processor import _linear_to_srgb_16bit
        inp = np.ones((4, 4, 3), dtype=np.uint16) * 30000
        result = _linear_to_srgb_16bit(inp)
        assert result.dtype == np.uint16

    def test_output_shape_preserved(self):
        from capture.multi_res_processor import _linear_to_srgb_16bit
        inp = np.ones((10, 20, 3), dtype=np.uint16) * 1000
        result = _linear_to_srgb_16bit(inp)
        assert result.shape == (10, 20, 3)

    def test_monotonicity(self):
        """sRGB transform should be monotonically increasing."""
        from capture.multi_res_processor import _linear_to_srgb_16bit
        values = np.linspace(0, 65535, 100, dtype=np.float64).astype(np.uint16)
        inp = values.reshape(-1, 1, 1).repeat(3, axis=2)
        result = _linear_to_srgb_16bit(inp)
        channel = result[:, 0, 0].astype(np.int32)
        diffs = np.diff(channel)
        assert np.all(diffs >= 0), "sRGB curve is not monotonically increasing"

    def test_low_linear_region(self):
        """Values below the 0.0031308 threshold use the linear segment."""
        from capture.multi_res_processor import _linear_to_srgb_16bit
        # 0.001 * 65535 ≈ 65
        inp = np.full((1, 1, 3), 65, dtype=np.uint16)
        result = _linear_to_srgb_16bit(inp)
        linear_val = 65.0 / 65535.0
        expected = 12.92 * linear_val
        expected_16 = int(expected * 65535)
        assert abs(int(result[0, 0, 0]) - expected_16) < 10


@pytest.mark.regression
class TestResizeImage:
    """Tests for _resize_image in multi_res_processor."""

    def test_none_target_returns_same(self):
        from capture.multi_res_processor import _resize_image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _resize_image(img, None)
        assert result is img  # Same object, no copy

    def test_scale_factor_quarter(self):
        from capture.multi_res_processor import _resize_image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _resize_image(img, 0.25)
        assert result.shape == (120, 160, 3)

    def test_scale_factor_half(self):
        from capture.multi_res_processor import _resize_image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _resize_image(img, 0.5)
        assert result.shape == (240, 320, 3)

    def test_exact_target_size(self):
        from capture.multi_res_processor import _resize_image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _resize_image(img, (1920, 1440))
        assert result.shape == (1440, 1920, 3)

    def test_scale_factor_2x_upscale(self):
        from capture.multi_res_processor import _resize_image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _resize_image(img, 2.0)
        assert result.shape == (200, 200, 3)

    def test_preserves_dtype_uint8(self):
        from capture.multi_res_processor import _resize_image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = _resize_image(img, 0.5)
        assert result.dtype == np.uint8

    def test_preserves_dtype_uint16(self):
        from capture.multi_res_processor import _resize_image
        img = np.ones((480, 640, 3), dtype=np.uint16) * 30000
        result = _resize_image(img, 0.5)
        assert result.dtype == np.uint16

    def test_200mp_to_quarter(self):
        """Simulate 200MP -> quarter res dimension calculation."""
        from capture.multi_res_processor import _resize_image
        # Use small proxy (can't allocate 200MP in test)
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        result = _resize_image(img, 0.25)
        assert result.shape[0] == 30
        assert result.shape[1] == 40


@pytest.mark.regression
class TestExtractFaceCrop:
    """Tests for _extract_face_crop in multi_res_processor."""

    def test_basic_crop(self):
        from capture.multi_res_processor import _extract_face_crop
        full = np.random.default_rng(42).integers(0, 255, (1200, 1600, 3), dtype=np.uint8)
        # Face bbox at quarter res: x=100, y=80, w=60, h=80
        crop, info = _extract_face_crop(full, (100, 80, 60, 80), 0.25, padding_factor=1.0)
        # At full res: x=400, y=320, w=240, h=320
        assert info["x1"] == 400
        assert info["y1"] == 320
        assert info["crop_width"] == 240
        assert info["crop_height"] == 320

    def test_padding_expands_crop(self):
        from capture.multi_res_processor import _extract_face_crop
        full = np.zeros((1200, 1600, 3), dtype=np.uint8)
        crop_no_pad, info_no = _extract_face_crop(full, (100, 100, 100, 100), 0.25, padding_factor=1.0)
        crop_pad, info_pad = _extract_face_crop(full, (100, 100, 100, 100), 0.25, padding_factor=1.5)
        assert info_pad["crop_width"] >= info_no["crop_width"]
        assert info_pad["crop_height"] >= info_no["crop_height"]

    def test_clamps_to_image_bounds(self):
        from capture.multi_res_processor import _extract_face_crop
        full = np.zeros((400, 400, 3), dtype=np.uint8)
        # Face near top-left corner at quarter res
        crop, info = _extract_face_crop(full, (0, 0, 50, 50), 0.25, padding_factor=2.0)
        assert info["x1"] >= 0
        assert info["y1"] >= 0
        assert info["x2"] <= 400
        assert info["y2"] <= 400

    def test_crop_info_contains_full_dims(self):
        from capture.multi_res_processor import _extract_face_crop
        full = np.zeros((1200, 1600, 3), dtype=np.uint8)
        _, info = _extract_face_crop(full, (50, 50, 50, 50), 0.25)
        assert info["full_width"] == 1600
        assert info["full_height"] == 1200

    def test_crop_shape_matches_info(self):
        from capture.multi_res_processor import _extract_face_crop
        full = np.random.default_rng(99).integers(0, 255, (800, 1000, 3), dtype=np.uint8)
        crop, info = _extract_face_crop(full, (50, 40, 80, 100), 0.25, padding_factor=1.5)
        assert crop.shape[0] == info["crop_height"]
        assert crop.shape[1] == info["crop_width"]


@pytest.mark.regression
class TestDefaultTiers:
    """Test the default tier configuration constants."""

    def test_full_tier_is_none(self):
        from capture.multi_res_processor import _DEFAULT_TIERS
        assert _DEFAULT_TIERS["full"] is None

    def test_quarter_tier_is_025(self):
        from capture.multi_res_processor import _DEFAULT_TIERS
        assert _DEFAULT_TIERS["quarter"] == 0.25

    def test_display_tier_is_tuple(self):
        from capture.multi_res_processor import _DEFAULT_TIERS
        assert _DEFAULT_TIERS["display"] == (1920, 1440)


# ═══════════════════════════════════════════════════════════════════════════
# Data Organizer Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestClassifyLens:
    """Tests for _classify_lens in data_organizer."""

    def test_ultrawide_13mm(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(13.0) == "ultrawide_13mm"

    def test_ultrawide_range(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(10.0) == "ultrawide_13mm"
        assert _classify_lens(18.0) == "ultrawide_13mm"

    def test_wide_23mm(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(23.0) == "wide_23mm"

    def test_wide_range(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(19.0) == "wide_23mm"
        assert _classify_lens(28.0) == "wide_23mm"

    def test_telephoto_70mm(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(70.0) == "telephoto_70mm"

    def test_telephoto_range(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(55.0) == "telephoto_70mm"
        assert _classify_lens(85.0) == "telephoto_70mm"

    def test_supertelephoto_200mm(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(200.0) == "supertelephoto_200mm"

    def test_supertelephoto_range(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(150.0) == "supertelephoto_200mm"
        assert _classify_lens(250.0) == "supertelephoto_200mm"

    def test_none_returns_unknown(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(None) == "unknown"

    def test_gap_between_ultrawide_and_wide(self):
        """18.5mm falls in the gap — not in any range."""
        from capture.data_organizer import _classify_lens
        assert _classify_lens(18.5) == "unknown"

    def test_gap_between_wide_and_telephoto(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(40.0) == "unknown"

    def test_gap_between_telephoto_and_supertelephoto(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(100.0) == "unknown"

    def test_below_range(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(5.0) == "unknown"

    def test_above_range(self):
        from capture.data_organizer import _classify_lens
        assert _classify_lens(300.0) == "unknown"


@pytest.mark.regression
class TestClassifyLensByResolution:
    """Tests for _classify_lens_by_resolution in data_organizer."""

    def test_200mp_main_sensor(self):
        from capture.data_organizer import _classify_lens_by_resolution
        assert _classify_lens_by_resolution(16320, 12240) == "main_200mp"

    def test_200mp_rotated(self):
        from capture.data_organizer import _classify_lens_by_resolution
        assert _classify_lens_by_resolution(12240, 16320) == "main_200mp"

    def test_wide_lens_photo(self):
        from capture.data_organizer import _classify_lens_by_resolution
        # 5712x4284 = ~24.5MP
        assert _classify_lens_by_resolution(5712, 4284) == "wide_23mm"

    def test_video_resolution_returns_none(self):
        """Video frames (1920x1080) are too small for photo classification."""
        from capture.data_organizer import _classify_lens_by_resolution
        assert _classify_lens_by_resolution(1920, 1080) is None

    def test_4080x3060_is_wide(self):
        """Quarter-res 200MP (4080x3060 ≈ 12.5MP) is below 20MP threshold."""
        from capture.data_organizer import _classify_lens_by_resolution
        # 4080*3060 = 12,484,800 < 20M
        assert _classify_lens_by_resolution(4080, 3060) is None

    def test_boundary_150mp(self):
        """Exactly at the 150MP boundary."""
        from capture.data_organizer import _classify_lens_by_resolution
        # Need w*h > 150,000,000
        # 15000 * 10001 = 150,015,000 > 150M
        assert _classify_lens_by_resolution(15000, 10001) == "main_200mp"

    def test_boundary_20mp(self):
        from capture.data_organizer import _classify_lens_by_resolution
        # Need w*h > 20,000,000
        # 5000 * 4001 = 20,005,000 > 20M
        assert _classify_lens_by_resolution(5000, 4001) == "wide_23mm"


@pytest.mark.regression
class TestLensConstants:
    """Verify data_organizer lens constant dictionaries are consistent."""

    def test_physical_focal_keys_match_lens_names(self):
        from capture.data_organizer import _LENS_CLASSIFICATION, _PHYSICAL_FOCAL
        lens_names = set(_LENS_CLASSIFICATION.values())
        focal_names = set(_PHYSICAL_FOCAL.keys())
        assert lens_names == focal_names

    def test_sensor_width_keys_match(self):
        from capture.data_organizer import _LENS_CLASSIFICATION, _SENSOR_WIDTH_MM
        lens_names = set(_LENS_CLASSIFICATION.values())
        sensor_names = set(_SENSOR_WIDTH_MM.keys())
        assert lens_names == sensor_names

    def test_lens_ranges_do_not_overlap(self):
        from capture.data_organizer import _LENS_CLASSIFICATION
        ranges = list(_LENS_CLASSIFICATION.keys())
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                lo_i, hi_i = ranges[i]
                lo_j, hi_j = ranges[j]
                assert hi_i < lo_j or hi_j < lo_i, \
                    f"Ranges ({lo_i},{hi_i}) and ({lo_j},{hi_j}) overlap"


@pytest.mark.regression
class TestSensorLogFiles:
    """Tests for sensor log file constants."""

    def test_known_sensor_files(self):
        from capture.data_organizer import _SENSOR_LOG_FILES
        assert "Accelerometer.csv" in _SENSOR_LOG_FILES
        assert "Gyroscope.csv" in _SENSOR_LOG_FILES
        assert "Orientation.csv" in _SENSOR_LOG_FILES
        assert "Barometer.csv" in _SENSOR_LOG_FILES

    def test_sensor_keys_are_lowercase(self):
        from capture.data_organizer import _SENSOR_LOG_FILES
        for key in _SENSOR_LOG_FILES.values():
            assert key == key.lower()


@pytest.mark.regression
class TestAnalyzeSensorZip:
    """Tests for _analyze_sensor_zip in data_organizer."""

    def _make_sensor_zip(self, tmp_path, sensors=None):
        """Create a synthetic Sensor Logger ZIP."""
        if sensors is None:
            sensors = {
                "Accelerometer.csv": self._make_csv(
                    "time,x,y,z",
                    start_ns=1700000000000000000,
                    n_samples=100,
                    dt_ns=10000000,  # 100 Hz
                ),
                "Gyroscope.csv": self._make_csv(
                    "time,x,y,z",
                    start_ns=1700000000000000000,
                    n_samples=100,
                    dt_ns=10000000,
                ),
            }

        zip_path = tmp_path / "sensor_log.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in sensors.items():
                zf.writestr(name, content)
        return zip_path

    def _make_csv(self, header, start_ns, n_samples, dt_ns):
        lines = [header]
        rng = np.random.default_rng(42)
        for i in range(n_samples):
            t = start_ns + i * dt_ns
            vals = rng.normal(0, 1, 3)
            lines.append(f"{t},{vals[0]:.6f},{vals[1]:.6f},{vals[2]:.6f}")
        return "\n".join(lines)

    def test_detects_sensors(self, tmp_path):
        from capture.data_organizer import _analyze_sensor_zip
        zip_path = self._make_sensor_zip(tmp_path)
        result = _analyze_sensor_zip(zip_path)
        assert "accelerometer" in result["sensors"]
        assert "gyroscope" in result["sensors"]

    def test_sample_count(self, tmp_path):
        from capture.data_organizer import _analyze_sensor_zip
        zip_path = self._make_sensor_zip(tmp_path)
        result = _analyze_sensor_zip(zip_path)
        assert result["sensors"]["accelerometer"]["samples"] == 100

    def test_sample_rate_approximate(self, tmp_path):
        from capture.data_organizer import _analyze_sensor_zip
        zip_path = self._make_sensor_zip(tmp_path)
        result = _analyze_sensor_zip(zip_path)
        hz = result["sensors"]["accelerometer"]["hz"]
        # 100 samples over ~1 second at 10ms intervals = ~100 Hz
        assert 90 < hz < 110

    def test_empty_zip(self, tmp_path):
        from capture.data_organizer import _analyze_sensor_zip
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass
        result = _analyze_sensor_zip(zip_path)
        assert len(result["sensors"]) == 0

    def test_gps_flag(self, tmp_path):
        from capture.data_organizer import _analyze_sensor_zip
        sensors = {
            "Location.csv": self._make_csv(
                "time,latitude,longitude,altitude",
                start_ns=1700000000000000000,
                n_samples=10,
                dt_ns=1000000000,
            ),
        }
        zip_path = self._make_sensor_zip(tmp_path, sensors)
        result = _analyze_sensor_zip(zip_path)
        assert result["has_gps"] is True


@pytest.mark.regression
class TestMatchSensorToVideo:
    """Tests for _match_sensor_to_video in data_organizer."""

    def test_no_sensors_returns_none(self):
        from capture.data_organizer import _match_sensor_to_video
        video, photo = _match_sensor_to_video([], {"duration": 10.0})
        assert video is None
        assert photo is None

    def test_single_sensor_matches(self):
        from capture.data_organizer import _match_sensor_to_video
        sa = [{
            "path": "/sensor1.zip",
            "sensors": {
                "gyroscope": {
                    "time_range": [1700000000000000000, 1700000010000000000],
                    "hz": 100,
                    "samples": 1000,
                },
            },
        }]
        video, photo = _match_sensor_to_video(sa, {"duration": 10.0})
        assert video == "/sensor1.zip"

    def test_no_video_duration_returns_first(self):
        from capture.data_organizer import _match_sensor_to_video
        sa = [
            {"path": "/a.zip", "sensors": {"gyroscope": {"time_range": [0, 10], "hz": 100, "samples": 1000}}},
            {"path": "/b.zip", "sensors": {"gyroscope": {"time_range": [0, 20], "hz": 100, "samples": 2000}}},
        ]
        video, photo = _match_sensor_to_video(sa, {"duration": 0.0})
        assert video == "/a.zip"
        assert photo == "/b.zip"

    def test_best_duration_match_wins(self):
        from capture.data_organizer import _match_sensor_to_video
        # Video is 10s. Sensor A is 10s, Sensor B is 60s.
        sa = [
            {
                "path": "/short.zip",
                "sensors": {
                    "gyroscope": {
                        "time_range": [1700000000000000000, 1700000010000000000],  # 10s
                        "hz": 100,
                        "samples": 1000,
                    },
                },
            },
            {
                "path": "/long.zip",
                "sensors": {
                    "gyroscope": {
                        "time_range": [1700000000000000000, 1700000060000000000],  # 60s
                        "hz": 100,
                        "samples": 6000,
                    },
                },
            },
        ]
        video, photo = _match_sensor_to_video(sa, {"duration": 10.0})
        assert video == "/short.zip"
        assert photo == "/long.zip"

    def test_no_gyroscope_skipped(self):
        from capture.data_organizer import _match_sensor_to_video
        sa = [{
            "path": "/no_gyro.zip",
            "sensors": {
                "accelerometer": {"time_range": [0, 10], "hz": 100, "samples": 1000},
            },
        }]
        video, photo = _match_sensor_to_video(sa, {"duration": 10.0})
        assert video is None


@pytest.mark.regression
class TestOrganizeCaptureSession:
    """Tests for organize_capture_session manifest structure."""

    def test_empty_inputs_produce_valid_manifest(self, tmp_path):
        from capture.data_organizer import organize_capture_session
        output = tmp_path / "session_out"
        manifest = organize_capture_session([], [], [], output)

        assert "video" in manifest
        assert "photos" in manifest
        assert "sensors" in manifest
        assert "summary" in manifest
        assert manifest["summary"]["num_videos"] == 0
        assert manifest["summary"]["num_photos_total"] == 0

    def test_manifest_json_written(self, tmp_path):
        from capture.data_organizer import organize_capture_session
        output = tmp_path / "session_out"
        organize_capture_session([], [], [], output)
        manifest_path = output / "session_manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "summary" in data

    def test_nonexistent_files_handled_gracefully(self, tmp_path):
        from capture.data_organizer import organize_capture_session
        output = tmp_path / "session_out"
        manifest = organize_capture_session(
            [tmp_path / "no_such.mp4"],
            [tmp_path / "no_such.jpg"],
            [tmp_path / "no_such.zip"],
            output,
        )
        assert manifest["summary"]["num_videos"] == 0
        assert manifest["summary"]["num_photos_total"] == 0


@pytest.mark.regression
class TestComputeMultiLensCameraModels:
    """Tests for compute_multi_lens_camera_models in data_organizer."""

    def test_video_camera_model(self):
        from capture.data_organizer import compute_multi_lens_camera_models
        manifest = {
            "video": {
                "resolution": [1920, 1080],
                "lens": "wide_23mm",
            },
            "photos": {},
        }
        with patch("utils.camera.focal_length_pixels", return_value=1500.0) as mock_fl:
            with patch.dict("sys.modules", {}):
                models = compute_multi_lens_camera_models(manifest)
        assert "video" in models
        assert models["video"]["cx"] == 960.0
        assert models["video"]["cy"] == 540.0
        assert models["video"]["width"] == 1920

    def test_empty_video_no_model(self):
        from capture.data_organizer import compute_multi_lens_camera_models
        manifest = {"video": {"resolution": [0, 0]}, "photos": {}}
        with patch("utils.camera.focal_length_pixels", return_value=100.0):
            models = compute_multi_lens_camera_models(manifest)
        assert "video" not in models

    def test_photo_camera_models(self):
        from capture.data_organizer import compute_multi_lens_camera_models
        manifest = {
            "video": {"resolution": [0, 0]},
            "photos": {
                "wide_23mm": [{"resolution": [5712, 4284], "focal_length": 6.3}],
            },
        }
        with patch("utils.camera.focal_length_pixels", return_value=4500.0):
            models = compute_multi_lens_camera_models(manifest)
        assert "photo_wide_23mm" in models
        assert models["photo_wide_23mm"]["cx"] == 5712 / 2.0

    def test_200mp_main_sensor_model(self):
        from capture.data_organizer import compute_multi_lens_camera_models
        manifest = {
            "video": {"resolution": [0, 0]},
            "photos": {
                "main_200mp": [{"resolution": [16320, 12240]}],
            },
        }
        with patch("utils.camera.focal_length_pixels", return_value=12000.0):
            models = compute_multi_lens_camera_models(manifest)
        assert "photo_main_200mp" in models
        assert models["photo_main_200mp"]["focal_mm"] == 6.3


@pytest.mark.regression
class TestPhysicalFocalConstants:
    """Verify physical focal length values are reasonable."""

    def test_ultrawide_is_shortest(self):
        from capture.data_organizer import _PHYSICAL_FOCAL
        assert _PHYSICAL_FOCAL["ultrawide_13mm"] < _PHYSICAL_FOCAL["wide_23mm"]

    def test_supertelephoto_is_longest(self):
        from capture.data_organizer import _PHYSICAL_FOCAL
        assert _PHYSICAL_FOCAL["supertelephoto_200mm"] > _PHYSICAL_FOCAL["telephoto_70mm"]

    def test_all_positive(self):
        from capture.data_organizer import _PHYSICAL_FOCAL
        for name, fl in _PHYSICAL_FOCAL.items():
            assert fl > 0, f"{name} has non-positive focal length"

    def test_sensor_widths_positive(self):
        from capture.data_organizer import _SENSOR_WIDTH_MM
        for name, sw in _SENSOR_WIDTH_MM.items():
            assert sw > 0, f"{name} has non-positive sensor width"
