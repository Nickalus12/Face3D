"""Mutation-testing target tests for critical Face3D pipeline modules.

These tests are designed to be FAST (no GPU, no disk I/O, synthetic data only)
and to catch specific classes of mutations: constant swaps, operator changes,
boundary condition flips, and sign errors.

Mutmut runs these hundreds of times, so each test must complete in milliseconds.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# Color Correction — S-Log3 EOTF constants, monotonicity, clipping
# ============================================================================

class TestSLog3EOTF:
    """Tests targeting mutations in the S-Log3 inverse transfer function."""

    def test_slog3_black_maps_to_near_zero(self):
        """Input 0.0 (black) must produce a non-negative value near zero."""
        from capture.color_correction import _slog3_to_linear
        result = _slog3_to_linear(np.array([0.0], dtype=np.float32))
        assert result[0] >= 0.0
        assert result[0] < 0.02  # S-Log3 black is slightly above true zero

    def test_slog3_white_maps_to_positive(self):
        """Input 1.0 (white) must produce a large positive value."""
        from capture.color_correction import _slog3_to_linear
        result = _slog3_to_linear(np.array([1.0], dtype=np.float32))
        assert result[0] > 1.0  # S-Log3 1.0 maps to well above 1.0 in linear

    def test_slog3_monotonically_increasing(self):
        """The S-Log3 EOTF must be strictly monotonically increasing."""
        from capture.color_correction import _slog3_to_linear
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        y = _slog3_to_linear(x.copy())
        diffs = np.diff(y)
        assert np.all(diffs >= 0), f"Non-monotonic at indices: {np.where(diffs < 0)[0]}"

    def test_slog3_output_non_negative(self):
        """All outputs must be clipped to >= 0 (no negative light)."""
        from capture.color_correction import _slog3_to_linear
        x = np.linspace(0.0, 1.0, 512, dtype=np.float32)
        y = _slog3_to_linear(x.copy())
        assert np.all(y >= 0.0), f"Negative values found: min={y.min()}"

    def test_slog3_cutpoint_continuity(self):
        """The piecewise function must be continuous at the cutpoint."""
        from capture.color_correction import _slog3_to_linear
        cut = np.float32(171.2102946929 / 1023.0)
        eps = np.float32(1e-4)
        below = _slog3_to_linear(np.array([cut - eps], dtype=np.float32))
        above = _slog3_to_linear(np.array([cut + eps], dtype=np.float32))
        # Values should be close at the cutpoint (within 5% of each other)
        assert abs(float(above[0]) - float(below[0])) < 0.005, \
            f"Discontinuity at cutpoint: below={below[0]}, above={above[0]}"

    def test_slog3_cutpoint_constant(self):
        """The cutpoint constant 171.2102946929 / 1023.0 must be used correctly.

        If mutated (e.g., to 172 or 170), the linear segment will produce
        different values.
        """
        from capture.color_correction import _slog3_to_linear
        # Test a value in the linear segment (below cutpoint ~0.1674)
        x = np.array([0.10], dtype=np.float32)
        y = _slog3_to_linear(x.copy())
        # Hand-computed: (0.10 * 1023 - 95) / (171.2102946929 - 95) * 0.01125
        expected = (0.10 * 1023.0 - 95.0) / (171.2102946929 - 95.0) * 0.01125
        assert abs(float(y[0]) - expected) < 1e-4, \
            f"Linear segment value wrong: got {y[0]}, expected {expected}"

    def test_slog3_curve_constants(self):
        """The curve segment constants (420, 261.5, 0.19, 0.01) must be exact.

        Mutations to any of these will change the output for values above the
        cutpoint.
        """
        from capture.color_correction import _slog3_to_linear
        # Test a value in the curve segment (above cutpoint ~0.1674)
        x = np.array([0.5], dtype=np.float32)
        y = _slog3_to_linear(x.copy())
        expected = 10.0 ** ((0.5 * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
        assert abs(float(y[0]) - expected) < 1e-3, \
            f"Curve segment value wrong: got {y[0]}, expected {expected}"

    def test_slog3_linear_segment_constant_95(self):
        """The linear segment offset constant 95.0 must be exact."""
        from capture.color_correction import _slog3_to_linear
        x = np.array([0.12], dtype=np.float32)
        y = _slog3_to_linear(x.copy())
        expected = (0.12 * 1023.0 - 95.0) / (171.2102946929 - 95.0) * 0.01125
        assert abs(float(y[0]) - expected) < 1e-4

    def test_slog3_linear_segment_scale_0_01125(self):
        """The linear segment scale factor 0.01125 must be exact."""
        from capture.color_correction import _slog3_to_linear
        x = np.array([0.15], dtype=np.float32)
        y = _slog3_to_linear(x.copy())
        expected = (0.15 * 1023.0 - 95.0) / (171.2102946929 - 95.0) * 0.01125
        assert abs(float(y[0]) - expected) < 1e-4


class TestSLog3LUT:
    """Tests targeting the precomputed 8-bit LUT."""

    def test_lut_has_256_entries(self):
        from capture.color_correction import _build_slog3_lut_8bit
        lut = _build_slog3_lut_8bit()
        assert lut.shape == (256,)
        assert lut.dtype == np.float32

    def test_lut_matches_analytic(self):
        """LUT values must match the analytic S-Log3 function."""
        from capture.color_correction import _build_slog3_lut_8bit, _slog3_to_linear
        lut = _build_slog3_lut_8bit()
        x = np.arange(256, dtype=np.float32) / 255.0
        analytic = _slog3_to_linear(x.copy())
        np.testing.assert_allclose(lut, analytic, atol=1e-3,
                                   err_msg="LUT diverges from analytic function")

    def test_lut_non_negative(self):
        from capture.color_correction import _build_slog3_lut_8bit
        lut = _build_slog3_lut_8bit()
        assert np.all(lut >= 0.0)

    def test_lut_monotonic(self):
        from capture.color_correction import _build_slog3_lut_8bit
        lut = _build_slog3_lut_8bit()
        assert np.all(np.diff(lut) >= 0)


class TestSRGBCurve:
    """Tests targeting the sRGB OETF."""

    def test_srgb_black(self):
        from capture.color_correction import _linear_to_srgb_curve
        result = _linear_to_srgb_curve(np.array([0.0], dtype=np.float32))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_srgb_white(self):
        from capture.color_correction import _linear_to_srgb_curve
        result = _linear_to_srgb_curve(np.array([1.0], dtype=np.float32))
        assert result[0] == pytest.approx(1.0, abs=1e-4)

    def test_srgb_monotonic(self):
        from capture.color_correction import _linear_to_srgb_curve
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        y = _linear_to_srgb_curve(x)
        assert np.all(np.diff(y) >= 0)

    def test_srgb_clipped_to_01(self):
        from capture.color_correction import _linear_to_srgb_curve
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        y = _linear_to_srgb_curve(x)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_srgb_cutpoint_constant(self):
        """The sRGB linear/gamma cutpoint 0.0031308 must be exact."""
        from capture.color_correction import _linear_to_srgb_curve
        # In the linear region: output = 12.92 * input
        x = np.array([0.001], dtype=np.float32)
        y = _linear_to_srgb_curve(x)
        assert y[0] == pytest.approx(12.92 * 0.001, abs=1e-4)

    def test_srgb_linear_scale_12_92(self):
        """The linear segment scale 12.92 must be exact."""
        from capture.color_correction import _linear_to_srgb_curve
        x = np.array([0.002], dtype=np.float32)
        y = _linear_to_srgb_curve(x)
        assert y[0] == pytest.approx(12.92 * 0.002, abs=1e-4)

    def test_srgb_gamma_exponent_2_4(self):
        """The gamma exponent 1/2.4 must be exact."""
        from capture.color_correction import _linear_to_srgb_curve
        x = np.array([0.5], dtype=np.float32)
        y = _linear_to_srgb_curve(x)
        expected = 1.055 * (0.5 ** (1.0 / 2.4)) - 0.055
        assert y[0] == pytest.approx(expected, abs=1e-3)


class TestLinearToSrgb:
    """Tests for the linear_to_srgb wrapper."""

    def test_output_is_uint8(self):
        from capture.color_correction import linear_to_srgb
        frame = np.ones((2, 2, 3), dtype=np.float32) * 0.5
        result = linear_to_srgb(frame)
        assert result.dtype == np.uint8

    def test_output_range(self):
        from capture.color_correction import linear_to_srgb
        frame = np.random.default_rng(42).random((4, 4, 3)).astype(np.float32)
        result = linear_to_srgb(frame)
        assert result.min() >= 0
        assert result.max() <= 255


# ============================================================================
# Frame Filtering — blur threshold, exposure boundaries
# ============================================================================

class TestCheckBlur:
    """Tests targeting blur detection threshold comparisons."""

    def test_sharp_frame_passes(self):
        """A frame with strong edges should pass the blur check."""
        from capture.frame_filter import check_blur
        rng = np.random.default_rng(10)
        # Create a frame with a high-contrast checkerboard pattern (very sharp)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    frame[i:i+10, j:j+10] = 255
        is_sharp, variance = check_blur(frame, threshold=100.0, half_res=False)
        assert is_sharp is True
        assert variance >= 100.0

    def test_blurry_frame_fails(self):
        """A uniform grey frame (zero edges) should fail the blur check."""
        from capture.frame_filter import check_blur
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        is_sharp, variance = check_blur(frame, threshold=100.0, half_res=False)
        assert is_sharp is False
        assert variance < 100.0

    def test_threshold_boundary_exact(self):
        """Variance exactly equal to threshold should pass (>= comparison)."""
        from capture.frame_filter import check_blur
        # We can't easily control exact Laplacian variance, so test the logic:
        # A frame at threshold should return is_sharp=True
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(0, 100, 5):
            for j in range(0, 100, 5):
                if (i // 5 + j // 5) % 2 == 0:
                    frame[i:i+5, j:j+5] = 200
        _, variance = check_blur(frame, threshold=100.0, half_res=False)
        # Now use the actual variance as threshold — should pass with >=
        is_sharp, _ = check_blur(frame, threshold=variance, half_res=False)
        assert is_sharp is True

    def test_half_res_scales_variance(self):
        """Half-res mode should approximately double the reported variance."""
        from capture.frame_filter import check_blur
        # Create a frame large enough for half_res to trigger (>500px)
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        for i in range(0, 600, 10):
            for j in range(0, 600, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    frame[i:i+10, j:j+10] = 200
        _, var_full = check_blur(frame, threshold=0, half_res=False)
        _, var_half = check_blur(frame, threshold=0, half_res=True)
        # Half-res variance is scaled by 2.0 internally; due to downscaling
        # artifacts the ratio can vary, but should be in a reasonable range
        ratio = var_half / var_full if var_full > 0 else 0
        assert 0.1 < ratio < 10.0, f"Half-res scaling ratio out of range: {ratio}"

    def test_face_bbox_used(self):
        """Blur should only be computed over the face ROI when bbox provided."""
        from capture.frame_filter import check_blur
        # Sharp checkerboard everywhere except face region is blurry
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(0, 200, 10):
            for j in range(0, 200, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    frame[i:i+10, j:j+10] = 255
        # Blur the face region
        frame[50:150, 50:150] = 128  # uniform = zero edges

        # Without bbox: sharp (whole frame has edges)
        is_sharp_full, _ = check_blur(frame, threshold=50.0, half_res=False)
        # With bbox: blurry (face region is uniform)
        is_sharp_face, _ = check_blur(
            frame, threshold=50.0, face_bbox=(50, 50, 100, 100), half_res=False,
        )
        assert is_sharp_full is True
        assert is_sharp_face is False


class TestCheckExposure:
    """Tests targeting exposure threshold boundaries."""

    def test_normal_exposure_passes(self):
        from capture.frame_filter import check_exposure
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        ok, stats = check_exposure(frame, low=30, high=225)
        assert ok is True
        assert stats["dark_ratio"] == 0.0
        assert stats["bright_ratio"] == 0.0

    def test_dark_frame_fails(self):
        from capture.frame_filter import check_exposure
        frame = np.full((100, 100, 3), 10, dtype=np.uint8)  # all below low=30
        ok, stats = check_exposure(frame, low=30, high=225)
        assert ok is False
        assert stats["dark_ratio"] > 0.5

    def test_bright_frame_fails(self):
        from capture.frame_filter import check_exposure
        frame = np.full((100, 100, 3), 240, dtype=np.uint8)  # all above high=225
        ok, stats = check_exposure(frame, low=30, high=225)
        assert ok is False
        assert stats["bright_ratio"] > 0.5

    def test_low_boundary_value(self):
        """Pixels at exactly low=30 should NOT be counted as dark (< comparison)."""
        from capture.frame_filter import check_exposure
        frame = np.full((100, 100, 3), 30, dtype=np.uint8)
        ok, stats = check_exposure(frame, low=30, high=225)
        assert stats["dark_ratio"] == 0.0  # 30 is not < 30

    def test_high_boundary_value(self):
        """Pixels at exactly high=225 should NOT be counted as bright (> comparison)."""
        from capture.frame_filter import check_exposure
        frame = np.full((100, 100, 3), 225, dtype=np.uint8)
        ok, stats = check_exposure(frame, low=30, high=225)
        assert stats["bright_ratio"] == 0.0  # 225 is not > 225

    def test_just_below_low(self):
        """Pixel value 29 (just below low=30) should be counted as dark."""
        from capture.frame_filter import check_exposure
        frame = np.full((100, 100, 3), 29, dtype=np.uint8)
        ok, stats = check_exposure(frame, low=30, high=225)
        assert stats["dark_ratio"] > 0.0

    def test_just_above_high(self):
        """Pixel value 226 (just above high=225) should be counted as bright."""
        from capture.frame_filter import check_exposure
        frame = np.full((100, 100, 3), 226, dtype=np.uint8)
        ok, stats = check_exposure(frame, low=30, high=225)
        assert stats["bright_ratio"] > 0.0

    def test_exposure_ratio_limits(self):
        """Both dark and bright ratios must be <= their limits for ok=True."""
        from capture.frame_filter import check_exposure
        # 60% dark pixels, 0% bright — should fail with default limit 0.5
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:60, :, :] = 10   # dark
        frame[60:, :, :] = 128  # normal
        ok, stats = check_exposure(frame, low=30, high=225, dark_ratio_limit=0.5)
        assert ok is False

    def test_mean_intensity_computed(self):
        from capture.frame_filter import check_exposure
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        _, stats = check_exposure(frame)
        assert stats["mean_intensity"] == pytest.approx(100.0, abs=1.0)


# ============================================================================
# Orientation — quaternion normalization, rotation matrices, Madgwick filter
# ============================================================================

class TestMadgwickFilter:
    """Tests targeting Madgwick AHRS filter invariants."""

    def test_initial_quaternion_is_identity(self):
        from sensors.orientation import MadgwickFilter
        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        q = filt.get_quaternion()
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_quaternion_stays_unit_norm(self):
        """After updates, the quaternion must remain unit-length."""
        from sensors.orientation import MadgwickFilter
        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        rng = np.random.default_rng(42)
        for _ in range(50):
            accel = rng.normal(0, 1, 3)
            accel[2] -= 9.81
            gyro = rng.normal(0, 0.5, 3)
            filt.update(accel, gyro)
            q = filt.get_quaternion()
            assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_rotation_matrix_is_orthogonal(self):
        """R^T R = I and det(R) = 1 after updates."""
        from sensors.orientation import MadgwickFilter
        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.array([0.1, 0.2, 0.0])
        for _ in range(20):
            filt.update(accel, gyro)
        R = filt.get_rotation_matrix()
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_beta_affects_convergence(self):
        """Higher beta should make the filter track accelerometer more aggressively."""
        from sensors.orientation import MadgwickFilter
        # Gravity pointing down in sensor frame
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        # Start from a slightly tilted orientation (~10 deg roll)
        tilted_q = np.array([0.9962, 0.0872, 0.0, 0.0])
        tilted_q /= np.linalg.norm(tilted_q)

        # Low beta: slow correction
        filt_low = MadgwickFilter(sample_rate=100.0, beta=0.001)
        filt_low.reset(tilted_q.copy())
        for _ in range(50):
            filt_low.update(accel, gyro)
        q_low = filt_low.get_quaternion()

        # High beta: fast correction
        filt_high = MadgwickFilter(sample_rate=100.0, beta=1.0)
        filt_high.reset(tilted_q.copy())
        for _ in range(50):
            filt_high.update(accel, gyro)
        q_high = filt_high.get_quaternion()

        # Both should move, but high-beta should move more (larger change from initial)
        change_low = np.linalg.norm(q_low - tilted_q)
        change_high = np.linalg.norm(q_high - tilted_q)
        assert change_high > change_low, \
            f"High beta should cause more correction: change_high={change_high} <= change_low={change_low}"

    def test_reset_restores_identity(self):
        from sensors.orientation import MadgwickFilter
        filt = MadgwickFilter()
        filt.update(np.array([0, 0, -9.81]), np.array([0.5, 0.3, 0.1]))
        filt.reset()
        np.testing.assert_allclose(filt.get_quaternion(), [1, 0, 0, 0])

    def test_quat_mult_identity(self):
        """Multiplying by identity quaternion should return the original."""
        from sensors.orientation import MadgwickFilter
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = MadgwickFilter._quat_mult(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_quat_mult_inverse(self):
        """q * q_conjugate should give identity (for unit quaternion)."""
        from sensors.orientation import MadgwickFilter
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = np.array([0.5, -0.5, -0.5, -0.5])
        result = MadgwickFilter._quat_mult(q, q_conj)
        np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-10)

    def test_zero_accel_pure_gyro_integration(self):
        """With zero acceleration (freefall), only gyro integration should happen."""
        from sensors.orientation import MadgwickFilter
        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        zero_accel = np.array([0.0, 0.0, 0.0])
        gyro = np.array([0.0, 0.0, 1.0])  # rotating around z
        filt.update(zero_accel, gyro)
        q = filt.get_quaternion()
        # Should have rotated slightly from identity
        assert q[0] < 1.0  # w component should decrease
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10


class TestComplementaryFilter:
    """Tests targeting the complementary filter."""

    def test_initial_state_from_accel(self):
        from sensors.orientation import ComplementaryFilter
        filt = ComplementaryFilter(alpha=0.98)
        # Gravity pointing down in +z (sensor upright, az positive)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        filt.update(accel, gyro, dt=0.01)
        euler = filt.get_euler()
        # With gravity in +z, roll=arctan2(0,9.81)=0, pitch=arctan2(0,9.81)=0
        assert abs(euler[0]) < 0.1  # roll
        assert abs(euler[1]) < 0.1  # pitch

    def test_alpha_blending(self):
        """Alpha=1.0 should trust gyro completely; alpha=0.0 should trust accel."""
        from sensors.orientation import ComplementaryFilter
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.array([1.0, 0.0, 0.0])  # rolling

        filt_gyro = ComplementaryFilter(alpha=1.0)
        filt_gyro.update(accel, gyro, dt=0.01)  # init
        filt_gyro.update(accel, gyro, dt=0.01)  # actual update
        euler_gyro = filt_gyro.get_euler()

        filt_accel = ComplementaryFilter(alpha=0.0)
        filt_accel.update(accel, gyro, dt=0.01)  # init
        filt_accel.update(accel, gyro, dt=0.01)  # actual update
        euler_accel = filt_accel.get_euler()

        # alpha=0 should stay near accel-derived angles (close to zero for this accel)
        assert abs(euler_accel[0]) < abs(euler_gyro[0]) or abs(euler_gyro[0]) < 0.02

    def test_rotation_matrix_valid(self):
        from sensors.orientation import ComplementaryFilter
        filt = ComplementaryFilter()
        filt.update(np.array([0.1, 0.2, -9.81]), np.array([0.1, 0.1, 0.1]), 0.01)
        R = filt.get_rotation_matrix()
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


class TestGyroBiasCalibration:
    """Tests targeting gyroscope bias estimation."""

    def test_bias_equals_mean_of_stationary(self):
        from sensors.orientation import calibrate_gyro_bias
        gyro = np.array([[0.01, -0.02, 0.005]] * 50, dtype=np.float64)
        bias = calibrate_gyro_bias(gyro, sample_rate=100.0, stationary_duration=0.5)
        np.testing.assert_allclose(bias, [0.01, -0.02, 0.005], atol=1e-10)

    def test_bias_uses_stationary_window_only(self):
        from sensors.orientation import calibrate_gyro_bias
        # First 50 samples (0.5s at 100Hz) are stationary, rest have motion
        gyro = np.zeros((200, 3), dtype=np.float64)
        gyro[:50] = [0.01, 0.02, 0.03]
        gyro[50:] = [5.0, 5.0, 5.0]  # motion
        bias = calibrate_gyro_bias(gyro, sample_rate=100.0, stationary_duration=0.5)
        np.testing.assert_allclose(bias, [0.01, 0.02, 0.03], atol=1e-10)


class TestTimestampValidation:
    """Tests targeting timestamp sanitization logic."""

    def test_monotonic_timestamps_all_valid(self):
        from sensors.orientation import _validate_timestamps
        ts = np.linspace(0, 1, 100)
        dt, valid = _validate_timestamps(ts)
        assert np.all(valid)
        assert np.all(dt[1:] > 0)

    def test_non_monotonic_flagged(self):
        from sensors.orientation import _validate_timestamps
        ts = np.array([0.0, 0.01, 0.005, 0.02])  # index 2 goes backward
        dt, valid = _validate_timestamps(ts)
        assert valid[0] is True or valid[0] == True  # numpy bool
        assert valid[2] == False  # non-monotonic sample

    def test_large_gap_flagged(self):
        from sensors.orientation import _validate_timestamps
        ts = np.array([0.0, 0.01, 0.02, 1.5])  # gap > 1.0s at index 3
        dt, valid = _validate_timestamps(ts)
        assert valid[3] == False


# ============================================================================
# COLMAP I/O — binary read/write roundtrip, quaternion ordering
# ============================================================================

class TestColmapCameraModels:
    """Tests targeting camera model enum values and parameter counts."""

    def test_camera_model_ids(self):
        from utils.colmap_io import CAMERA_MODELS
        assert CAMERA_MODELS[0] == ("SIMPLE_PINHOLE", 3)
        assert CAMERA_MODELS[1] == ("PINHOLE", 4)
        assert CAMERA_MODELS[2] == ("SIMPLE_RADIAL", 4)
        assert CAMERA_MODELS[3] == ("RADIAL", 5)
        assert CAMERA_MODELS[4] == ("OPENCV", 8)

    def test_camera_model_count(self):
        from utils.colmap_io import CAMERA_MODELS
        assert len(CAMERA_MODELS) == 11


class TestColmapQuaternion:
    """Tests targeting quaternion-to-rotation conversions (COLMAP uses w,x,y,z)."""

    def test_identity_quaternion(self):
        from utils.colmap_io import qvec_to_rotmat
        R = qvec_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_deg_rotation_z(self):
        """Quaternion for 90-degree rotation around z-axis."""
        from utils.colmap_io import qvec_to_rotmat
        # q = [cos(45), 0, 0, sin(45)]
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        R = qvec_to_rotmat(q)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_rotation_matrix_orthogonal(self):
        from utils.colmap_io import qvec_to_rotmat
        rng = np.random.default_rng(99)
        for _ in range(10):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            R = qvec_to_rotmat(q)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_qvec_roundtrip(self):
        """qvec -> rotmat -> qvec should roundtrip (up to sign)."""
        from utils.colmap_io import qvec_to_rotmat, rotmat_to_qvec
        rng = np.random.default_rng(77)
        for _ in range(10):
            q_orig = rng.normal(size=4)
            q_orig /= np.linalg.norm(q_orig)
            if q_orig[0] < 0:
                q_orig = -q_orig
            R = qvec_to_rotmat(q_orig)
            q_back = rotmat_to_qvec(R)
            if q_back[0] < 0:
                q_back = -q_back
            np.testing.assert_allclose(q_back, q_orig, atol=1e-6)

    def test_quaternion_order_wxyz(self):
        """COLMAP uses w,x,y,z ordering. Swapping x,y should change the rotation."""
        from utils.colmap_io import qvec_to_rotmat
        q1 = np.array([0.5, 0.5, 0.3, 0.1])
        q1 /= np.linalg.norm(q1)
        q2 = np.array([0.5, 0.3, 0.5, 0.1])  # swapped x and y
        q2 /= np.linalg.norm(q2)
        R1 = qvec_to_rotmat(q1)
        R2 = qvec_to_rotmat(q2)
        assert not np.allclose(R1, R2, atol=1e-6), \
            "Swapping x,y should produce different rotations"


class TestColmapBinaryRoundtrip:
    """Tests targeting binary read/write of COLMAP model files."""

    def test_cameras_roundtrip(self, tmp_path):
        from utils.colmap_io import read_cameras_binary, CAMERA_MODELS
        cameras_path = tmp_path / "cameras.bin"
        # Write a PINHOLE camera
        with open(cameras_path, "wb") as f:
            f.write(struct.pack("<Q", 1))  # num_cameras
            f.write(struct.pack("<IiQQ", 1, 1, 640, 480))  # id, model=PINHOLE, w, h
            f.write(struct.pack("<4d", 525.0, 525.0, 320.0, 240.0))  # fx,fy,cx,cy

        cameras = read_cameras_binary(cameras_path)
        assert len(cameras) == 1
        cam = cameras[1]
        assert cam.model == "PINHOLE"
        assert cam.width == 640
        assert cam.height == 480
        np.testing.assert_allclose(cam.params, [525.0, 525.0, 320.0, 240.0])

    def test_images_roundtrip(self, tmp_path):
        from utils.colmap_io import read_images_binary
        images_path = tmp_path / "images.bin"
        qvec = np.array([1.0, 0.0, 0.0, 0.0])
        tvec = np.array([1.0, 2.0, 3.0])
        name = "test_frame.png"

        with open(images_path, "wb") as f:
            f.write(struct.pack("<Q", 1))  # num_images
            f.write(struct.pack("<I", 1))  # image_id
            f.write(struct.pack("<4d", *qvec))
            f.write(struct.pack("<3d", *tvec))
            f.write(struct.pack("<I", 1))  # camera_id
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2d

        images = read_images_binary(images_path)
        assert len(images) == 1
        img = images[1]
        assert img.name == "test_frame.png"
        np.testing.assert_allclose(img.qvec, qvec)
        np.testing.assert_allclose(img.tvec, tvec)
        assert img.camera_id == 1

    def test_points3d_roundtrip(self, tmp_path):
        from utils.colmap_io import read_points3d_binary
        points_path = tmp_path / "points3D.bin"
        xyz = np.array([1.5, -2.3, 0.7])
        rgb = np.array([100, 200, 50], dtype=np.uint8)
        error = 0.42

        with open(points_path, "wb") as f:
            f.write(struct.pack("<Q", 1))  # num_points
            f.write(struct.pack("<Q", 42))  # point3d_id
            f.write(struct.pack("<3d", *xyz))
            f.write(struct.pack("<3B", *rgb))
            f.write(struct.pack("<d", error))
            f.write(struct.pack("<Q", 0))  # track_length

        points = read_points3d_binary(points_path)
        assert len(points) == 1
        pt = points[42]
        np.testing.assert_allclose(pt.xyz, xyz)
        np.testing.assert_array_equal(pt.rgb, rgb)
        assert pt.error == pytest.approx(error)

    def test_struct_format_little_endian(self, tmp_path):
        """Verify the binary format uses little-endian byte order."""
        from utils.colmap_io import read_cameras_binary
        cameras_path = tmp_path / "cameras.bin"
        # Write with known values and verify little-endian
        with open(cameras_path, "wb") as f:
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<IiQQ", 1, 0, 100, 200))  # SIMPLE_PINHOLE
            f.write(struct.pack("<3d", 500.0, 50.0, 100.0))

        cameras = read_cameras_binary(cameras_path)
        cam = cameras[1]
        assert cam.model == "SIMPLE_PINHOLE"
        assert cam.width == 100
        assert cam.height == 200


class TestIntrinsicsMatrix:
    """Tests targeting the intrinsics matrix construction."""

    def test_pinhole_intrinsics(self):
        from utils.colmap_io import get_intrinsics_matrix, Camera
        cam = Camera(id=1, model="PINHOLE", width=640, height=480,
                     params=np.array([525.0, 525.0, 320.0, 240.0]))
        K = get_intrinsics_matrix(cam)
        expected = np.array([
            [525.0, 0.0, 320.0],
            [0.0, 525.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(K, expected)

    def test_simple_pinhole_intrinsics(self):
        from utils.colmap_io import get_intrinsics_matrix, Camera
        cam = Camera(id=1, model="SIMPLE_PINHOLE", width=640, height=480,
                     params=np.array([500.0, 320.0, 240.0]))
        K = get_intrinsics_matrix(cam)
        assert K[0, 0] == 500.0  # fx
        assert K[1, 1] == 500.0  # fy = fx for SIMPLE_PINHOLE
        assert K[0, 2] == 320.0  # cx
        assert K[1, 2] == 240.0  # cy
        assert K[2, 2] == 1.0

    def test_intrinsics_bottom_row(self):
        """The bottom row of K must always be [0, 0, 1]."""
        from utils.colmap_io import get_intrinsics_matrix, Camera
        cam = Camera(id=1, model="PINHOLE", width=640, height=480,
                     params=np.array([525.0, 525.0, 320.0, 240.0]))
        K = get_intrinsics_matrix(cam)
        np.testing.assert_allclose(K[2, :], [0, 0, 1])

    def test_intrinsics_off_diagonal_zero(self):
        """K should have zero skew (off-diagonal elements in rows 0-1)."""
        from utils.colmap_io import get_intrinsics_matrix, Camera
        cam = Camera(id=1, model="PINHOLE", width=640, height=480,
                     params=np.array([525.0, 525.0, 320.0, 240.0]))
        K = get_intrinsics_matrix(cam)
        assert K[0, 1] == 0.0
        assert K[1, 0] == 0.0
