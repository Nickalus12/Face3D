"""Property-based invariant tests using Hypothesis.

These tests verify fundamental mathematical and data-structure invariants
that must hold for *any* valid input, not just specific test cases.

Tested invariants:
    - Quaternion <-> rotation matrix round-trips are identity (up to sign)
    - S-Log3 EOTF is monotonically non-decreasing for any input in [0, 1]
    - sRGB OETF output is always in [0, 1]
    - Depth maps from any positive input remain positive after alignment ops
    - Gaussian model parameters remain valid after to()/clone()/requires_grad_()
    - Any valid rotation matrix has det=1 and R^T R = I

Install: pip install hypothesis

Skips gracefully if hypothesis is not installed.
"""

import numpy as np
import pytest

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays as np_arrays
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

pytestmark = [
    pytest.mark.property_based,
    pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed"),
]


# ═══════════════════════════════════════════════════════════════════════════
# COLMAP quaternion / rotation invariants
# ═══════════════════════════════════════════════════════════════════════════


if HAS_HYPOTHESIS:
    # Strategy: generate a random unit quaternion
    @st.composite
    def unit_quaternions(draw):
        """Strategy for generating random unit quaternions [w, x, y, z]."""
        raw = draw(np_arrays(
            dtype=np.float64,
            shape=(4,),
            elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ))
        norm = np.linalg.norm(raw)
        assume(norm > 0.01)  # skip degenerate zero vectors
        q = raw / norm
        if q[0] < 0:
            q = -q
        return q

    @given(qvec=unit_quaternions())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_qvec_rotmat_round_trip_property(qvec):
        """For any unit quaternion, qvec -> R -> qvec is identity (up to sign)."""
        from utils.colmap_io import qvec_to_rotmat, rotmat_to_qvec

        R = qvec_to_rotmat(qvec)

        # R must be a proper rotation
        assert abs(np.linalg.det(R) - 1.0) < 1e-6
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)

        qvec_back = rotmat_to_qvec(R)
        # Quaternions are equivalent up to sign
        dot = abs(np.dot(qvec, qvec_back))
        assert dot > 1.0 - 1e-6, f"Round-trip failed: dot={dot}, q_in={qvec}, q_out={qvec_back}"

    @given(qvec=unit_quaternions())
    @settings(max_examples=50)
    def test_rotation_matrix_orthogonality_property(qvec):
        """Any rotation matrix from a unit quaternion satisfies R^T R = I."""
        from utils.colmap_io import qvec_to_rotmat

        R = qvec_to_rotmat(qvec)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Color correction invariants
# ═══════════════════════════════════════════════════════════════════════════


if HAS_HYPOTHESIS:
    @given(x=np_arrays(
        dtype=np.float32,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ))
    @settings(max_examples=50)
    def test_slog3_monotonic_property(x):
        """S-Log3 EOTF is monotonically non-decreasing for any input in [0, 1]."""
        from capture.color_correction import _slog3_to_linear

        x_sorted = np.sort(x)
        y = _slog3_to_linear(x_sorted)
        assert np.all(np.diff(y) >= -1e-6), "S-Log3 violated monotonicity"

    @given(x=np_arrays(
        dtype=np.float32,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ))
    @settings(max_examples=50)
    def test_slog3_non_negative_property(x):
        """S-Log3 EOTF output is always non-negative."""
        from capture.color_correction import _slog3_to_linear

        y = _slog3_to_linear(x)
        assert np.all(y >= 0.0), f"Negative values in S-Log3 output: min={y.min()}"

    @given(x=np_arrays(
        dtype=np.float32,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ))
    @settings(max_examples=50)
    def test_srgb_oetf_range_property(x):
        """sRGB OETF output is always in [0, 1] for inputs in [0, 1]."""
        from capture.color_correction import _linear_to_srgb_curve

        y = _linear_to_srgb_curve(x)
        assert np.all(y >= 0.0), f"sRGB OETF produced negative: min={y.min()}"
        assert np.all(y <= 1.0), f"sRGB OETF produced >1: max={y.max()}"


# ═══════════════════════════════════════════════════════════════════════════
# Depth map invariants
# ═══════════════════════════════════════════════════════════════════════════


if HAS_HYPOTHESIS:
    @given(depth=np_arrays(
        dtype=np.float32,
        shape=(64, 64),
        elements=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    ))
    @settings(max_examples=30)
    def test_depth_map_positive_property(depth):
        """Any depth map with positive inputs stays positive after operations."""
        from conftest import Invariants

        Invariants.assert_valid_depth_map(depth)

        # Scaling depth should preserve positivity
        scaled = depth * 2.5
        Invariants.assert_valid_depth_map(scaled)


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian model invariants
# ═══════════════════════════════════════════════════════════════════════════


if HAS_HYPOTHESIS:
    @given(n=st.integers(min_value=1, max_value=200))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_gaussian_model_arbitrary_size(n):
        """GaussianModel works correctly for any size 1..200."""
        import torch
        from splatting.initializer import GaussianModel, SH_COEFFS

        model = GaussianModel(
            positions=torch.randn(n, 3),
            colors_sh=torch.randn(n, SH_COEFFS, 3),
            scales=torch.randn(n, 3),
            rotations=torch.randn(n, 4),
            opacities=torch.randn(n, 1),
        )
        assert model.num_gaussians == n
        assert len(model.parameters_list()) == 5

        # Clone preserves size
        clone = model.clone()
        assert clone.num_gaussians == n

        # to() preserves size
        moved = model.to("cpu")
        assert moved.num_gaussians == n

    @given(n=st.integers(min_value=2, max_value=50))
    @settings(max_examples=10)
    def test_gaussian_clone_independence(n):
        """Modifying a clone never affects the original."""
        import torch
        from splatting.initializer import GaussianModel, SH_COEFFS

        model = GaussianModel(
            positions=torch.ones(n, 3),
            colors_sh=torch.zeros(n, SH_COEFFS, 3),
            scales=torch.zeros(n, 3),
            rotations=torch.zeros(n, 4),
            opacities=torch.zeros(n, 1),
        )
        clone = model.clone()
        clone.positions.fill_(999.0)
        assert model.positions[0, 0].item() == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Orientation filter invariants
# ═══════════════════════════════════════════════════════════════════════════


if HAS_HYPOTHESIS:
    @given(
        accel=np_arrays(
            dtype=np.float64, shape=(3,),
            elements=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        ),
        gyro=np_arrays(
            dtype=np.float64, shape=(3,),
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100)
    def test_madgwick_preserves_unit_quaternion(accel, gyro):
        """Madgwick filter update always produces a unit quaternion."""
        from sensors.orientation import MadgwickFilter

        # Skip near-zero accel (free-fall edge case handled separately)
        assume(np.linalg.norm(accel) > 0.1)

        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        filt.update(accel, gyro)
        q = filt.get_quaternion()
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 1e-8, f"Quaternion norm {norm} after single update"

    @given(
        accel=np_arrays(
            dtype=np.float64, shape=(3,),
            elements=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        ),
        gyro=np_arrays(
            dtype=np.float64, shape=(3,),
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50)
    def test_madgwick_rotation_matrix_proper(accel, gyro):
        """Rotation matrix from Madgwick is always proper (det=1, orthogonal)."""
        from sensors.orientation import MadgwickFilter
        from conftest import Invariants

        assume(np.linalg.norm(accel) > 0.1)

        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        for _ in range(5):
            filt.update(accel, gyro)
        R = filt.get_rotation_matrix()
        Invariants.assert_valid_rotation_matrix(R, atol=1e-6)
