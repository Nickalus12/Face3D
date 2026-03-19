"""Numba JIT-accelerated kernels for CPU-bound hot loops.

Provides accelerated versions of color correction, sensor fusion,
and depth/geometry operations. All functions have pure-numpy fallbacks
so numba is an optional dependency.

Usage:
    from utils.numba_kernels import slog3_to_linear, linear_to_srgb, ...

When numba is installed, functions are compiled to native machine code
on first call (cached to disk for instant startup on subsequent runs).
Without numba, the same functions run as plain numpy — identical results,
just slower.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Numba availability detection with graceful fallback
# ---------------------------------------------------------------------------

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def prange(*args):
        """Fallback prange that delegates to range."""
        return range(*args)

    def njit(*args, **kwargs):
        """Fallback njit decorator that returns the function unchanged."""
        def wrapper(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper


# ============================================================================
# Color correction kernels
# ============================================================================

@njit(cache=True)
def _slog3_to_linear_pixel(x: float) -> float:
    """S-Log3 EOTF for a single pixel value in [0, 1]."""
    cut = 171.2102946929 / 1023.0  # ~0.16736
    x_scaled = x * 1023.0
    if x < cut:
        val = (x_scaled - 95.0) / (171.2102946929 - 95.0) * 0.01125
    else:
        val = (10.0 ** ((x_scaled - 420.0) / 261.5)) * 0.19 - 0.01
    if val < 0.0:
        val = 0.0
    return val


@njit(cache=True, parallel=True)
def slog3_to_linear(img: np.ndarray) -> np.ndarray:
    """Vectorized S-Log3 EOTF on a float32 image array.

    Args:
        img: HxWxC or HxW float32 array with values in [0, 1].

    Returns:
        Same-shape float32 array in linear light.
    """
    flat = img.ravel()
    out = np.empty(flat.shape[0], dtype=np.float32)
    for i in prange(flat.shape[0]):
        v = float(flat[i])
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out[i] = np.float32(_slog3_to_linear_pixel(v))
    return out.reshape(img.shape)


@njit(cache=True)
def _linear_to_srgb_pixel(x: float) -> float:
    """sRGB OETF for a single linear pixel value in [0, 1]."""
    if x <= 0.0031308:
        val = 12.92 * x
    else:
        val = 1.055 * (x ** (1.0 / 2.4)) - 0.055
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


@njit(cache=True, parallel=True)
def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Vectorized sRGB OETF (linear -> sRGB gamma).

    Args:
        img: HxWxC or HxW float32 array in linear light, [0, 1].

    Returns:
        Same-shape float32 array in sRGB gamma space, [0, 1].
    """
    flat = img.ravel()
    out = np.empty(flat.shape[0], dtype=np.float32)
    for i in prange(flat.shape[0]):
        v = float(flat[i])
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out[i] = np.float32(_linear_to_srgb_pixel(v))
    return out.reshape(img.shape)


@njit(cache=True, parallel=True)
def apply_lut(img_uint8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 256-entry uint8 LUT to an image.

    Args:
        img_uint8: HxWxC uint8 image.
        lut: (256,) uint8 lookup table.

    Returns:
        HxWxC uint8 image with LUT applied per channel.
    """
    flat = img_uint8.ravel()
    out = np.empty(flat.shape[0], dtype=np.uint8)
    for i in prange(flat.shape[0]):
        out[i] = lut[flat[i]]
    return out.reshape(img_uint8.shape)


# ============================================================================
# Sensor / orientation kernels
# ============================================================================

@njit(cache=True)
def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q1 * q2, both as [w, x, y, z].

    Args:
        q1: (4,) quaternion.
        q2: (4,) quaternion.

    Returns:
        (4,) quaternion product.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


@njit(cache=True)
def quaternion_multiply_batch(q1_batch: np.ndarray, q2_batch: np.ndarray) -> np.ndarray:
    """Batch quaternion multiplication.

    Args:
        q1_batch: (N, 4) quaternions [w, x, y, z].
        q2_batch: (N, 4) quaternions [w, x, y, z].

    Returns:
        (N, 4) quaternion products.
    """
    n = q1_batch.shape[0]
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        out[i] = quaternion_multiply(q1_batch[i], q2_batch[i])
    return out


@njit(cache=True)
def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a single quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Args:
        q: (4,) quaternion [w, x, y, z].

    Returns:
        (3, 3) rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[0, 1] = 2.0 * (x * y - w * z)
    R[0, 2] = 2.0 * (x * z + w * y)
    R[1, 0] = 2.0 * (x * y + w * z)
    R[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[1, 2] = 2.0 * (y * z - w * x)
    R[2, 0] = 2.0 * (x * z - w * y)
    R[2, 1] = 2.0 * (y * z + w * x)
    R[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


@njit(cache=True)
def quaternion_to_rotation_matrix_batch(q_batch: np.ndarray) -> np.ndarray:
    """Batch quaternion to rotation matrix conversion.

    Args:
        q_batch: (N, 4) quaternions [w, x, y, z].

    Returns:
        (N, 3, 3) rotation matrices.
    """
    n = q_batch.shape[0]
    out = np.empty((n, 3, 3), dtype=np.float64)
    for i in range(n):
        out[i] = quaternion_to_rotation_matrix(q_batch[i])
    return out


@njit(cache=True)
def madgwick_update(
    q: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    beta: float,
    dt: float,
) -> np.ndarray:
    """Single Madgwick AHRS filter step.

    Fuses accelerometer and gyroscope readings to update the orientation
    quaternion. This is the inner-loop hot function called once per IMU
    sample (typically 100-200 Hz).

    Args:
        q: (4,) current orientation quaternion [w, x, y, z].
        gyro: (3,) gyroscope reading [gx, gy, gz] in rad/s.
        accel: (3,) accelerometer reading [ax, ay, az] in m/s^2.
        beta: Filter gain (gyro/accel trade-off).
        dt: Time step in seconds.

    Returns:
        (4,) updated orientation quaternion [w, x, y, z], normalized.
    """
    # Normalize accelerometer
    ax, ay, az = accel[0], accel[1], accel[2]
    a_norm = np.sqrt(ax * ax + ay * ay + az * az)

    if a_norm < 1e-10:
        # Pure gyroscope integration (free-fall / no gravity reference)
        gx, gy, gz = gyro[0], gyro[1], gyro[2]
        g_quat = np.array([0.0, gx, gy, gz], dtype=np.float64)
        q_dot = np.empty(4, dtype=np.float64)
        # 0.5 * q * g_quat
        pw, px, py, pz = q[0], q[1], q[2], q[3]
        rw, rx, ry, rz = g_quat[0], g_quat[1], g_quat[2], g_quat[3]
        q_dot[0] = 0.5 * (pw * rw - px * rx - py * ry - pz * rz)
        q_dot[1] = 0.5 * (pw * rx + px * rw + py * rz - pz * ry)
        q_dot[2] = 0.5 * (pw * ry - px * rz + py * rw + pz * rx)
        q_dot[3] = 0.5 * (pw * rz + px * ry - py * rx + pz * rw)
        q_out = q + q_dot * dt
        n = np.sqrt(q_out[0] ** 2 + q_out[1] ** 2 + q_out[2] ** 2 + q_out[3] ** 2)
        q_out /= n
        return q_out

    ax /= a_norm
    ay /= a_norm
    az /= a_norm

    w, x, y, z = q[0], q[1], q[2], q[3]

    # Objective function: rotate [0,0,1] by q, compare to accel
    f0 = 2.0 * (x * z - w * y) - ax
    f1 = 2.0 * (w * x + y * z) - ay
    f2 = 1.0 - 2.0 * (x * x + y * y) - az

    # Jacobian transpose * f
    step = np.empty(4, dtype=np.float64)
    step[0] = -2.0 * y * f0 + 2.0 * x * f1
    step[1] = 2.0 * z * f0 + 2.0 * w * f1 - 4.0 * x * f2
    step[2] = -2.0 * w * f0 + 2.0 * z * f1 - 4.0 * y * f2
    step[3] = 2.0 * x * f0 + 2.0 * y * f1

    step_norm = np.sqrt(step[0] ** 2 + step[1] ** 2 + step[2] ** 2 + step[3] ** 2)
    if step_norm > 1e-12:
        step /= step_norm

    # Quaternion derivative from gyroscope
    gx, gy, gz = gyro[0], gyro[1], gyro[2]
    q_dot_gyro = np.empty(4, dtype=np.float64)
    q_dot_gyro[0] = 0.5 * (- x * gx - y * gy - z * gz)
    q_dot_gyro[1] = 0.5 * (w * gx + y * gz - z * gy)
    q_dot_gyro[2] = 0.5 * (w * gy - x * gz + z * gx)
    q_dot_gyro[3] = 0.5 * (w * gz + x * gy - y * gx)

    # Fuse
    q_out = q + (q_dot_gyro - beta * step) * dt
    n = np.sqrt(q_out[0] ** 2 + q_out[1] ** 2 + q_out[2] ** 2 + q_out[3] ** 2)
    q_out /= n
    return q_out


@njit(cache=True)
def madgwick_filter_full(
    accel_data: np.ndarray,
    gyro_data: np.ndarray,
    dt_arr: np.ndarray,
    valid_mask: np.ndarray,
    beta: float,
    sample_rate: float,
) -> np.ndarray:
    """Run the full Madgwick filter over an IMU sequence.

    This replaces the Python-level for-loop in orientation.py with a
    single compiled function, eliminating per-sample Python overhead.

    Args:
        accel_data: (N, 3) accelerometer readings.
        gyro_data: (N, 3) gyroscope readings (bias-corrected).
        dt_arr: (N,) per-sample time deltas.
        valid_mask: (N,) boolean mask of valid samples.
        beta: Filter gain.
        sample_rate: IMU sampling rate (used for first sample dt).

    Returns:
        (N, 4) quaternion array [w, x, y, z] per sample.
    """
    n = accel_data.shape[0]
    quats = np.empty((n, 4), dtype=np.float64)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for i in range(n):
        if not valid_mask[i] and i > 0:
            quats[i] = quats[i - 1]
            continue

        dt = dt_arr[i] if i > 0 else 1.0 / sample_rate
        if dt <= 0.0:
            dt = 1.0 / sample_rate

        q = madgwick_update(q, gyro_data[i], accel_data[i], beta, dt)
        quats[i] = q

    return quats


# ============================================================================
# Depth / geometry kernels
# ============================================================================

@njit(cache=True, parallel=True)
def unproject_depth_to_points(
    depth: np.ndarray,
    K_inv_row0: np.ndarray,
    K_inv_row1: np.ndarray,
    K_inv_row2: np.ndarray,
    mask: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    """Unproject masked depth pixels to 3D camera-space points.

    Uses the inverse intrinsic matrix rows to compute ray directions,
    then scales by depth. This is the inner loop of depth unprojection.

    Args:
        depth: (H, W) depth map.
        K_inv_row0: (3,) first row of K^-1.
        K_inv_row1: (3,) second row of K^-1.
        K_inv_row2: (3,) third row of K^-1.
        mask: (M,) boolean mask over the flattened pixel grid.
        ys: (N,) y-coordinates of valid pixels (after mask).
        xs: (N,) x-coordinates of valid pixels (after mask).

    Returns:
        (N, 3) float64 points in camera coordinates.
    """
    n = ys.shape[0]
    points = np.empty((n, 3), dtype=np.float64)
    for i in prange(n):
        px = float(xs[i])
        py = float(ys[i])
        d = float(depth[ys[i], xs[i]])
        # K_inv @ [px, py, 1]^T * d
        points[i, 0] = (K_inv_row0[0] * px + K_inv_row0[1] * py + K_inv_row0[2]) * d
        points[i, 1] = (K_inv_row1[0] * px + K_inv_row1[1] * py + K_inv_row1[2]) * d
        points[i, 2] = (K_inv_row2[0] * px + K_inv_row2[1] * py + K_inv_row2[2]) * d
    return points


@njit(cache=True, parallel=True)
def confidence_filter(conf_map: np.ndarray, threshold: float) -> np.ndarray:
    """Fast confidence thresholding on a 2D confidence map.

    Args:
        conf_map: (H, W) float32 confidence map.
        threshold: Minimum confidence value to keep.

    Returns:
        (H, W) boolean mask where conf_map > threshold.
    """
    H = conf_map.shape[0]
    W = conf_map.shape[1]
    out = np.empty((H, W), dtype=np.bool_)
    for i in prange(H):
        for j in range(W):
            out[i, j] = conf_map[i, j] > threshold
    return out


@njit(cache=True)
def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z].

    Handles all four cases (trace > 0, and each diagonal max).
    Matches the COLMAP quaternion convention.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (4,) normalized quaternion [w, x, y, z].
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    q /= n
    return q
