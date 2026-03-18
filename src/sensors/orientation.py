"""IMU-based orientation estimation using Madgwick AHRS and complementary filters.

Includes gyroscope bias calibration from stationary initial samples and
timestamp validation for robust handling of Samsung metadata quirks.
"""

import logging
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

logger = logging.getLogger(__name__)


class MadgwickFilter:
    """Madgwick AHRS (Attitude and Heading Reference System) filter.

    Fuses 6-DOF IMU data (accelerometer + gyroscope) into orientation
    quaternions using gradient-descent-based complementary filtering.

    Reference:
        S. Madgwick, "An efficient orientation filter for inertial and
        inertial/magnetic sensor arrays", 2010.
    """

    def __init__(self, sample_rate: float = 100.0, beta: float = 0.1) -> None:
        """Initialise the filter.

        Args:
            sample_rate: IMU sampling frequency in Hz.
            beta: Filter gain (trade-off between gyro trust and accel correction).
                  Higher values track the accelerometer more aggressively.
        """
        self._sample_rate = sample_rate
        self._beta = beta
        # Internal quaternion stored as [w, x, y, z]
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def reset(self, quaternion: Optional[np.ndarray] = None) -> None:
        """Reset the filter state.

        Args:
            quaternion: Optional initial orientation as [w, x, y, z].
                        Defaults to identity.
        """
        if quaternion is not None:
            q = np.asarray(quaternion, dtype=np.float64).copy()
            q /= np.linalg.norm(q)
            self._q = q
        else:
            self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, accel: np.ndarray, gyro: np.ndarray) -> None:
        """Process one IMU sample and advance the orientation estimate.

        Args:
            accel: Accelerometer reading [ax, ay, az] in m/s^2.
            gyro: Gyroscope reading [gx, gy, gz] in rad/s.
        """
        q = self._q
        dt = 1.0 / self._sample_rate

        # Normalise accelerometer (skip correction if zero / free-fall)
        a = np.asarray(accel, dtype=np.float64)
        a_norm = np.linalg.norm(a)
        if a_norm < 1e-10:
            # Pure gyroscope integration when no valid gravity reference
            g = np.asarray(gyro, dtype=np.float64)
            q_dot = 0.5 * self._quat_mult(q, np.array([0.0, g[0], g[1], g[2]]))
            q = q + q_dot * dt
            q /= np.linalg.norm(q)
            self._q = q
            return

        a = a / a_norm
        g = np.asarray(gyro, dtype=np.float64)

        w, x, y, z = q

        # Objective function: rotate gravity reference [0,0,1] by q and compare to a
        # f = q* [0,0,0,1] q - [0, ax, ay, az]
        f = np.array([
            2.0 * (x * z - w * y) - a[0],
            2.0 * (w * x + y * z) - a[1],
            1.0 - 2.0 * (x * x + y * y) - a[2],
        ], dtype=np.float64)

        # Jacobian J^T
        j_t = np.array([
            [-2.0 * y, 2.0 * x, 0.0],
            [2.0 * z, 2.0 * w, -4.0 * x],
            [-2.0 * w, 2.0 * z, -4.0 * y],
            [2.0 * x, 2.0 * y, 0.0],
        ], dtype=np.float64)

        # Gradient descent corrective step
        step = j_t @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-12:
            step /= step_norm

        # Quaternion derivative from gyroscope
        q_dot_gyro = 0.5 * self._quat_mult(q, np.array([0.0, g[0], g[1], g[2]]))

        # Fuse
        q = q + (q_dot_gyro - self._beta * step) * dt
        q /= np.linalg.norm(q)
        self._q = q

    def get_quaternion(self) -> np.ndarray:
        """Return the current orientation as a quaternion [w, x, y, z]."""
        return self._q.copy()

    def get_rotation_matrix(self) -> np.ndarray:
        """Return the current orientation as a 3x3 rotation matrix."""
        w, x, y, z = self._q
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    @staticmethod
    def _quat_mult(p: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Hamilton quaternion product p * r, both as [w, x, y, z]."""
        pw, px, py, pz = p
        rw, rx, ry, rz = r
        return np.array([
            pw*rw - px*rx - py*ry - pz*rz,
            pw*rx + px*rw + py*rz - pz*ry,
            pw*ry - px*rz + py*rw + pz*rx,
            pw*rz + px*ry - py*rx + pz*rw,
        ], dtype=np.float64)


class ComplementaryFilter:
    """Simple complementary filter for orientation estimation.

    Blends high-frequency gyroscope integration with low-frequency
    accelerometer-derived tilt using a tuneable *alpha* parameter.
    Faster and often more stable than Madgwick for slow face-capture
    orbits where the motion is gentle and predictable.

    The orientation is stored as Euler angles [roll, pitch, yaw] in
    radians and converted to rotation matrices / quaternions on demand.
    """

    def __init__(self, alpha: float = 0.98) -> None:
        """Initialise the complementary filter.

        Args:
            alpha: Blending factor in [0, 1].  Higher values trust the
                   gyroscope more; lower values track the accelerometer.
        """
        self._alpha = alpha
        # Orientation as [roll, pitch, yaw] in radians
        self._orientation = np.zeros(3, dtype=np.float64)
        self._initialised = False

    def reset(self) -> None:
        """Reset the filter to its initial state."""
        self._orientation = np.zeros(3, dtype=np.float64)
        self._initialised = False

    @staticmethod
    def _accel_to_rp(accel: np.ndarray) -> tuple[float, float]:
        """Derive roll and pitch from a gravity-dominated accelerometer reading."""
        ax, ay, az = accel
        roll = float(np.arctan2(ay, az))
        pitch = float(np.arctan2(-ax, np.sqrt(ay * ay + az * az)))
        return roll, pitch

    def update(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """Process one IMU sample.

        Args:
            accel: Accelerometer [ax, ay, az] in m/s^2.
            gyro: Gyroscope [gx, gy, gz] in rad/s.
            dt: Time step in seconds since the last sample.
        """
        accel = np.asarray(accel, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)

        accel_roll, accel_pitch = self._accel_to_rp(accel)

        if not self._initialised:
            self._orientation[0] = accel_roll
            self._orientation[1] = accel_pitch
            self._orientation[2] = 0.0
            self._initialised = True
            return

        # Gyro integration (high-frequency path)
        gyro_orientation = self._orientation + gyro * dt

        # Complementary blend
        a = self._alpha
        self._orientation[0] = a * gyro_orientation[0] + (1 - a) * accel_roll
        self._orientation[1] = a * gyro_orientation[1] + (1 - a) * accel_pitch
        self._orientation[2] = gyro_orientation[2]  # yaw from gyro only

    def get_euler(self) -> np.ndarray:
        """Return the current orientation as [roll, pitch, yaw] in radians."""
        return self._orientation.copy()

    def get_rotation_matrix(self) -> np.ndarray:
        """Return the current orientation as a 3x3 rotation matrix."""
        r, p, y = self._orientation
        return Rotation.from_euler("xyz", [r, p, y]).as_matrix()

    def get_quaternion(self) -> np.ndarray:
        """Return the current orientation as a quaternion [w, x, y, z]."""
        r, p, y = self._orientation
        q_xyzw = Rotation.from_euler("xyz", [r, p, y]).as_quat()  # [x,y,z,w]
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def calibrate_gyro_bias(
    gyro_data: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    sample_rate: float = 100.0,
    stationary_duration: float = 0.5,
) -> np.ndarray:
    """Estimate gyroscope bias from an initial stationary period.

    During the first *stationary_duration* seconds the device is assumed
    to be still.  The bias is the mean gyroscope reading over that window.

    Args:
        gyro_data: (N, 3) gyroscope readings in rad/s.
        timestamps: Optional (N,) timestamps in seconds.  When provided
            the stationary window is determined from actual times;
            otherwise *sample_rate* is used to estimate sample count.
        sample_rate: Fallback sampling rate in Hz (used when *timestamps*
            is ``None``).
        stationary_duration: Duration in seconds of the assumed-stationary
            period at the start of the recording.

    Returns:
        Gyroscope bias vector (3,) in rad/s.
    """
    if timestamps is not None and len(timestamps) > 1:
        mask = timestamps - timestamps[0] <= stationary_duration
        n_cal = int(np.sum(mask))
    else:
        n_cal = max(1, int(sample_rate * stationary_duration))

    n_cal = min(n_cal, len(gyro_data))
    bias = gyro_data[:n_cal].mean(axis=0)
    logger.info(
        "Gyro bias calibrated from %d samples: [%.6f, %.6f, %.6f] rad/s",
        n_cal, *bias,
    )
    return bias


def _validate_timestamps(timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and sanitise an IMU timestamp array.

    Handles non-monotonic timestamps (which can occur with Samsung
    metadata) by computing per-sample ``dt`` values, rejecting samples
    where ``dt <= 0`` or ``dt > 1.0 s``, and returning a boolean mask
    of valid samples.

    Args:
        timestamps: (N,) array of timestamps in seconds.

    Returns:
        Tuple of (dt, valid_mask) where *dt* is (N,) with ``dt[0] = 0``
        and *valid_mask* is a boolean (N,) array.  Invalid samples have
        ``dt`` set to ``0.0`` and ``valid_mask`` set to ``False``.
    """
    n = len(timestamps)
    dt = np.zeros(n, dtype=np.float64)
    valid = np.ones(n, dtype=bool)

    if n < 2:
        return dt, valid

    raw_dt = np.diff(timestamps)
    dt[1:] = raw_dt

    # Flag non-monotonic or excessively large gaps
    bad = (dt <= 0) | (dt > 1.0)
    bad[0] = False  # first sample is always valid (dt=0)
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        logger.warning(
            "Timestamp validation: rejecting %d / %d samples "
            "(non-monotonic or dt > 1.0s)",
            n_bad, n,
        )
    valid[bad] = False
    dt[bad] = 0.0

    return dt, valid


def compute_rotations(
    imu_data: dict,
    sample_rate: float = 100.0,
    beta: float = 0.1,
    use_complementary: bool = False,
    complementary_alpha: float = 0.98,
    auto_calibrate_gyro: bool = True,
    stationary_duration: float = 0.5,
) -> np.ndarray:
    """Process a full IMU sequence through an orientation filter.

    Supports both the Madgwick AHRS filter and a simpler complementary
    filter (selected via *use_complementary*).  When *auto_calibrate_gyro*
    is enabled, the first *stationary_duration* seconds of gyroscope data
    are used to estimate and subtract sensor bias before integration.

    Timestamps in ``imu_data`` (key ``timestamps``) are validated for
    monotonicity; non-monotonic or excessively gapped samples are skipped.

    Args:
        imu_data: Dictionary with ``accel_xyz`` (N, 3), ``gyro_xyz`` (N, 3),
            and optionally ``timestamps`` (N,).
        sample_rate: IMU sampling rate in Hz (used as fallback when
            timestamps are absent).
        beta: Madgwick filter gain (ignored when *use_complementary*).
        use_complementary: If ``True``, use the complementary filter
            instead of Madgwick.
        complementary_alpha: Alpha parameter for the complementary filter.
        auto_calibrate_gyro: If ``True``, estimate and subtract gyroscope
            bias from the initial stationary period.
        stationary_duration: Duration in seconds of the assumed-stationary
            calibration window.

    Returns:
        Array of shape (N, 3, 3) containing per-sample rotation matrices.
    """
    accel = np.asarray(imu_data["accel_xyz"], dtype=np.float64)
    gyro = np.asarray(imu_data["gyro_xyz"], dtype=np.float64)
    n_samples = len(accel)

    timestamps = imu_data.get("timestamps")
    has_timestamps = timestamps is not None and len(timestamps) == n_samples
    if has_timestamps:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        dt_arr, valid_mask = _validate_timestamps(timestamps)
    else:
        dt_arr = np.full(n_samples, 1.0 / sample_rate, dtype=np.float64)
        dt_arr[0] = 0.0
        valid_mask = np.ones(n_samples, dtype=bool)

    # Gyroscope bias calibration
    if auto_calibrate_gyro and n_samples > 0:
        bias = calibrate_gyro_bias(
            gyro,
            timestamps=timestamps if has_timestamps else None,
            sample_rate=sample_rate,
            stationary_duration=stationary_duration,
        )
        gyro = gyro - bias

    rotations = np.empty((n_samples, 3, 3), dtype=np.float64)

    if use_complementary:
        filt = ComplementaryFilter(alpha=complementary_alpha)
        for i in range(n_samples):
            if not valid_mask[i] and i > 0:
                # Skip invalid timestamp samples; carry forward last rotation
                rotations[i] = rotations[i - 1]
                continue
            dt = float(dt_arr[i]) if i > 0 else 1.0 / sample_rate
            filt.update(accel[i], gyro[i], dt)
            rotations[i] = filt.get_rotation_matrix()

        logger.info(
            "Computed %d orientation estimates (complementary, alpha=%.3f)",
            n_samples, complementary_alpha,
        )
    else:
        filt = MadgwickFilter(sample_rate=sample_rate, beta=beta)
        for i in range(n_samples):
            if not valid_mask[i] and i > 0:
                rotations[i] = rotations[i - 1]
                continue
            filt.update(accel[i], gyro[i])
            rotations[i] = filt.get_rotation_matrix()

        logger.info(
            "Computed %d orientation estimates (Madgwick, rate=%.1f Hz, beta=%.3f)",
            n_samples, sample_rate, beta,
        )

    return rotations


def estimate_gravity(accel_data: np.ndarray, cutoff_hz: float = 1.0, sample_rate: float = 100.0) -> np.ndarray:
    """Estimate the gravity direction from accelerometer data using a low-pass filter.

    A first-order Butterworth low-pass filter removes high-frequency motion
    components, leaving predominantly the gravity signal.  The result is
    normalised to a unit vector.

    Args:
        accel_data: Accelerometer readings of shape (N, 3) in m/s^2.
        cutoff_hz: Low-pass cutoff frequency in Hz.
        sample_rate: Accelerometer sampling rate in Hz.

    Returns:
        Unit gravity vector (3,) in the sensor frame.
    """
    from scipy.signal import butter, filtfilt

    if len(accel_data) < 12:
        # Not enough samples for filtfilt; just average
        gravity = accel_data.mean(axis=0)
    else:
        nyq = sample_rate / 2.0
        cutoff_norm = min(cutoff_hz / nyq, 0.99)
        b, a = butter(1, cutoff_norm, btype="low")
        filtered = filtfilt(b, a, accel_data, axis=0)
        gravity = filtered.mean(axis=0)

    norm = np.linalg.norm(gravity)
    if norm < 1e-10:
        logger.warning("Gravity estimate is near-zero; returning default [0, 0, -1]")
        return np.array([0.0, 0.0, -1.0])

    gravity /= norm
    logger.info("Estimated gravity direction: [%.4f, %.4f, %.4f]", *gravity)
    return gravity


def interpolate_rotations_to_frames(
    rotations: np.ndarray,
    imu_timestamps: np.ndarray,
    frame_timestamps: np.ndarray,
) -> np.ndarray:
    """Interpolate rotation matrices from IMU rate to video frame rate using SLERP.

    Args:
        rotations: (N, 3, 3) rotation matrices at IMU sample times.
        imu_timestamps: (N,) timestamps corresponding to *rotations*.
        frame_timestamps: (M,) target frame timestamps.

    Returns:
        (M, 3, 3) rotation matrices interpolated to *frame_timestamps*.
    """
    n_imu = len(imu_timestamps)
    if n_imu < 2:
        raise ValueError("Need at least 2 rotation samples for SLERP interpolation")

    # Build scipy Rotation objects
    rots = Rotation.from_matrix(rotations)

    # Clamp frame timestamps to the IMU time range for extrapolation safety
    t_min, t_max = imu_timestamps[0], imu_timestamps[-1]
    clamped = np.clip(frame_timestamps, t_min, t_max)

    # scipy Slerp requires strictly increasing times; deduplicate if needed
    unique_mask = np.concatenate(([True], np.diff(imu_timestamps) > 1e-12))
    t_unique = imu_timestamps[unique_mask]
    rots_unique = Rotation.from_matrix(rotations[unique_mask])

    if len(t_unique) < 2:
        # All timestamps identical; replicate the single rotation
        single_mat = rotations[0]
        return np.tile(single_mat, (len(frame_timestamps), 1, 1))

    slerp = Slerp(t_unique, rots_unique)
    interpolated = slerp(clamped)

    result = interpolated.as_matrix()
    logger.info(
        "SLERP-interpolated %d IMU rotations -> %d frame rotations",
        n_imu, len(frame_timestamps),
    )
    return result
