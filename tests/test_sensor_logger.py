"""Tests for sensor data processing: IMU parsing, orientation estimation,
Sensor Logger ZIP parsing, and video-sensor alignment.

Tested invariants:
    - Quaternions from orientation filters are always unit-length
    - Rotation matrices are always proper rotations (det=1, orthogonal)
    - Timestamps are monotonically non-decreasing after validation
    - Gyro bias calibration produces a finite 3-vector
    - SLERP interpolation preserves quaternion unit-length property
    - Gravity estimate is a unit vector
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# IMU parsing (src.sensors.imu_parser)
# ═══════════════════════════════════════════════════════════════════════════


class TestIMUParser:
    """Tests for IMU sidecar parsing."""

    def test_parse_imu_from_json_sidecar(self, tmp_path):
        """Parse a JSON sidecar file with Samsung-style IMU data."""
        from sensors.imu_parser import parse_imu_from_sidecar

        samples = [
            {"t": i * 0.01, "ax": 0.1, "ay": -0.2, "az": -9.8,
             "gx": 0.01, "gy": -0.02, "gz": 0.005}
            for i in range(100)
        ]
        sidecar = tmp_path / "imu_data.json"
        sidecar.write_text(json.dumps({"samples": samples}), encoding="utf-8")

        result = parse_imu_from_sidecar(sidecar)
        assert result["timestamps"].shape == (100,)
        assert result["accel_xyz"].shape == (100, 3)
        assert result["gyro_xyz"].shape == (100, 3)
        # Timestamps should be monotonic
        assert np.all(np.diff(result["timestamps"]) >= 0)

    def test_parse_imu_from_csv_sidecar(self, tmp_path):
        """Parse a CSV sidecar with standard IMU columns."""
        from sensors.imu_parser import parse_imu_from_sidecar

        csv_path = tmp_path / "imu_data.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
            for i in range(50):
                writer.writerow([i * 0.01, 0.0, 0.0, -9.81, 0.0, 0.5, 0.0])

        result = parse_imu_from_sidecar(csv_path)
        assert result["timestamps"].shape == (50,)
        assert result["accel_xyz"].shape == (50, 3)

    def test_parse_imu_missing_file(self, tmp_path):
        """FileNotFoundError for missing sidecar."""
        from sensors.imu_parser import parse_imu_from_sidecar

        with pytest.raises(FileNotFoundError):
            parse_imu_from_sidecar(tmp_path / "missing.json")

    def test_align_timestamps(self, sample_imu_data):
        """Interpolate IMU data to video frame timestamps."""
        from sensors.imu_parser import align_timestamps

        frame_ts = np.linspace(0, 0.9, 10)
        aligned = align_timestamps(sample_imu_data, frame_ts)
        assert aligned["timestamps"].shape == (10,)
        assert aligned["accel_xyz"].shape == (10, 3)
        assert aligned["gyro_xyz"].shape == (10, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Orientation estimation (src.sensors.orientation)
# ═══════════════════════════════════════════════════════════════════════════


class TestOrientationEstimation:
    """Tests for Madgwick filter, complementary filter, and utilities."""

    def test_madgwick_filter_identity(self):
        """Madgwick filter with zero input stays at identity orientation."""
        from sensors.orientation import MadgwickFilter

        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        q = filt.get_quaternion()
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_madgwick_quaternion_stays_normalized(self, sample_imu_data):
        """Quaternion output is unit-length after every update."""
        from sensors.orientation import MadgwickFilter

        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        accel = sample_imu_data["accel_xyz"]
        gyro = sample_imu_data["gyro_xyz"]

        for i in range(len(accel)):
            filt.update(accel[i], gyro[i])
            q = filt.get_quaternion()
            norm = np.linalg.norm(q)
            assert abs(norm - 1.0) < 1e-10, f"Quaternion norm {norm} at step {i}"

    def test_madgwick_rotation_matrix_valid(self, sample_imu_data):
        """Rotation matrix from Madgwick is always a proper rotation."""
        from sensors.orientation import MadgwickFilter
        from helpers import Invariants

        filt = MadgwickFilter(sample_rate=100.0, beta=0.1)
        accel = sample_imu_data["accel_xyz"]
        gyro = sample_imu_data["gyro_xyz"]

        for i in range(len(accel)):
            filt.update(accel[i], gyro[i])

        R = filt.get_rotation_matrix()
        Invariants.assert_valid_rotation_matrix(R)

    def test_complementary_filter_identity(self):
        """Complementary filter starts near zero Euler angles for gravity-aligned input."""
        from sensors.orientation import ComplementaryFilter

        filt = ComplementaryFilter(alpha=0.98)
        # Pure gravity in +z (sensor at rest, z-up convention)
        filt.update(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt=0.01)
        euler = filt.get_euler()
        assert abs(euler[0]) < 0.1, f"Roll should be near 0, got {euler[0]}"
        assert abs(euler[1]) < 0.1, f"Pitch should be near 0, got {euler[1]}"

    def test_compute_rotations_madgwick(self, sample_imu_data):
        """compute_rotations (Madgwick) produces valid rotation matrices."""
        from sensors.orientation import compute_rotations
        from helpers import Invariants

        rotations = compute_rotations(sample_imu_data, sample_rate=100.0)
        assert rotations.shape == (100, 3, 3)

        # Check first and last rotation matrices
        Invariants.assert_valid_rotation_matrix(rotations[0])
        Invariants.assert_valid_rotation_matrix(rotations[-1])

    def test_compute_rotations_complementary(self, sample_imu_data):
        """compute_rotations (complementary) produces valid rotation matrices."""
        from sensors.orientation import compute_rotations
        from helpers import Invariants

        rotations = compute_rotations(
            sample_imu_data, sample_rate=100.0, use_complementary=True,
        )
        assert rotations.shape == (100, 3, 3)
        Invariants.assert_valid_rotation_matrix(rotations[50])

    def test_gyro_bias_calibration(self, sample_imu_data):
        """Gyro bias calibration produces a finite 3-vector."""
        from sensors.orientation import calibrate_gyro_bias

        bias = calibrate_gyro_bias(
            sample_imu_data["gyro_xyz"],
            timestamps=sample_imu_data["timestamps"],
        )
        assert bias.shape == (3,)
        assert np.all(np.isfinite(bias))

    def test_gravity_estimate_unit_vector(self, sample_imu_data):
        """Estimated gravity is a unit vector."""
        from sensors.orientation import estimate_gravity

        gravity = estimate_gravity(sample_imu_data["accel_xyz"])
        assert gravity.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(gravity), 1.0, atol=1e-10)

    def test_interpolate_rotations_slerp(self, sample_imu_data):
        """SLERP interpolation preserves rotation matrix validity."""
        from sensors.orientation import compute_rotations, interpolate_rotations_to_frames
        from helpers import Invariants

        rotations = compute_rotations(sample_imu_data, sample_rate=100.0)
        imu_ts = sample_imu_data["timestamps"]
        frame_ts = np.linspace(0.0, 0.9, 10)

        interp = interpolate_rotations_to_frames(rotations, imu_ts, frame_ts)
        assert interp.shape == (10, 3, 3)

        for i in range(10):
            Invariants.assert_valid_rotation_matrix(interp[i], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Sensor Logger ZIP parsing (src.sensors.sensor_logger)
# ═══════════════════════════════════════════════════════════════════════════


class TestSensorLoggerParser:
    """Tests for Sensor Logger ZIP archive parsing."""

    def _create_sensor_zip(self, tmp_path, num_samples=50):
        """Helper: create a fake Sensor Logger ZIP with CSV files."""
        zip_path = tmp_path / "sensor_data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Accelerometer.csv
            accel_buf = io.StringIO()
            writer = csv.DictWriter(accel_buf, fieldnames=["time", "seconds_elapsed", "x", "y", "z"])
            writer.writeheader()
            for i in range(num_samples):
                writer.writerow({
                    "time": str(1700000000000000000 + i * 10000000),
                    "seconds_elapsed": f"{i * 0.01:.4f}",
                    "x": "0.05", "y": "-0.03", "z": "-9.81",
                })
            zf.writestr("Accelerometer.csv", accel_buf.getvalue())

            # Gyroscope.csv
            gyro_buf = io.StringIO()
            writer = csv.DictWriter(gyro_buf, fieldnames=["time", "seconds_elapsed", "x", "y", "z"])
            writer.writeheader()
            for i in range(num_samples):
                writer.writerow({
                    "time": str(1700000000000000000 + i * 10000000),
                    "seconds_elapsed": f"{i * 0.01:.4f}",
                    "x": "0.001", "y": "0.5", "z": "-0.002",
                })
            zf.writestr("Gyroscope.csv", gyro_buf.getvalue())

            # Orientation.csv (pre-fused quaternions)
            orient_buf = io.StringIO()
            writer = csv.DictWriter(orient_buf,
                fieldnames=["time", "seconds_elapsed", "qw", "qx", "qy", "qz", "roll", "pitch", "yaw"])
            writer.writeheader()
            for i in range(num_samples):
                angle = i * 0.01 * 0.5  # slow rotation
                writer.writerow({
                    "time": str(1700000000000000000 + i * 10000000),
                    "seconds_elapsed": f"{i * 0.01:.4f}",
                    "qw": f"{np.cos(angle / 2):.6f}",
                    "qx": "0.0",
                    "qy": f"{np.sin(angle / 2):.6f}",
                    "qz": "0.0",
                    "roll": "0.0", "pitch": f"{np.degrees(angle):.2f}", "yaw": "0.0",
                })
            zf.writestr("Orientation.csv", orient_buf.getvalue())

            # Metadata.csv
            meta_buf = io.StringIO()
            writer = csv.DictWriter(meta_buf, fieldnames=["device name", "recording time"])
            writer.writeheader()
            writer.writerow({"device name": "Galaxy S25 Ultra", "recording time": "2025-03-15 14:30:00"})
            zf.writestr("Metadata.csv", meta_buf.getvalue())

        return zip_path

    def test_parse_sensor_zip(self, tmp_path):
        """Parse a synthetic Sensor Logger ZIP file."""
        from sensors.sensor_logger import parse_sensor_logger_zip

        zip_path = self._create_sensor_zip(tmp_path)
        npz_path, summary = parse_sensor_logger_zip(zip_path, tmp_path / "output")

        assert npz_path.exists()
        assert summary["total_samples"] > 0
        assert any(name == "accelerometer" for name, _ in summary["sensors"])
        assert any(name == "orientation" for name, _ in summary["sensors"])

        # Verify NPZ contents
        data = np.load(str(npz_path), allow_pickle=True)
        assert "accel_xyz" in data
        assert "gyro_xyz" in data
        assert "orientation_quats" in data
        # Orientation quaternions should be roughly unit-length
        quats = data["orientation_quats"]
        norms = np.linalg.norm(quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_parse_sensor_zip_missing_file(self, tmp_path):
        """FileNotFoundError for missing ZIP."""
        from sensors.sensor_logger import parse_sensor_logger_zip

        with pytest.raises(FileNotFoundError):
            parse_sensor_logger_zip(tmp_path / "missing.zip", tmp_path / "output")

    def test_orientation_quaternion_validity(self, tmp_path):
        """Orientation quaternions from parsed ZIP are normalized."""
        from sensors.sensor_logger import parse_sensor_logger_zip
        from helpers import Invariants

        zip_path = self._create_sensor_zip(tmp_path, num_samples=100)
        npz_path, _ = parse_sensor_logger_zip(zip_path, tmp_path / "output")

        data = np.load(str(npz_path), allow_pickle=True)
        quats = data["orientation_quats"]
        Invariants.assert_valid_quaternions(quats, atol=1e-3)

    def test_generate_rotation_priors(self, tmp_path):
        """Generate COLMAP rotation priors from sensor data."""
        from sensors.sensor_logger import (
            parse_sensor_logger_zip,
            generate_rotation_priors_from_sensor_logger,
        )

        zip_path = self._create_sensor_zip(tmp_path, num_samples=100)
        npz_path, _ = parse_sensor_logger_zip(zip_path, tmp_path / "output")

        # Simulate aligned sensor data with frame_quaternions
        data = np.load(str(npz_path), allow_pickle=True)
        frame_quats = data["orientation_quats"][:10]  # 10 frames
        sensor_data = {"frame_quaternions": frame_quats}

        frame_names = [f"frame_{i:06d}.png" for i in range(10)]
        priors_path = tmp_path / "image_priors.txt"

        result = generate_rotation_priors_from_sensor_logger(
            sensor_data, frame_names, priors_path,
        )
        assert result.exists()

        # Verify priors file format
        lines = result.read_text(encoding="utf-8").strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 10

        for line in data_lines:
            parts = line.split()
            assert len(parts) == 15  # name + 4 quat + 3 trans + 4 q_std + 3 t_std


# ═══════════════════════════════════════════════════════════════════════════
# Timestamp validation
# ═══════════════════════════════════════════════════════════════════════════


class TestTimestampValidation:
    """Tests for timestamp sanitization (non-monotonic, gaps, etc.)."""

    def test_validate_monotonic_timestamps(self):
        """Monotonic timestamps should all be valid."""
        from sensors.orientation import _validate_timestamps

        ts = np.linspace(0, 1, 100)
        dt, valid = _validate_timestamps(ts)
        assert np.all(valid)
        assert dt[0] == 0.0
        assert np.all(dt[1:] > 0)

    def test_validate_non_monotonic_timestamps(self):
        """Non-monotonic timestamps should be flagged invalid."""
        from sensors.orientation import _validate_timestamps

        ts = np.array([0.0, 0.01, 0.02, 0.015, 0.03])  # index 3 goes backward
        dt, valid = _validate_timestamps(ts)
        assert not valid[3], "Backward timestamp should be flagged invalid"
        assert valid[0] and valid[1] and valid[2] and valid[4]

    def test_validate_large_gap_timestamps(self):
        """Gaps > 1 second should be flagged invalid."""
        from sensors.orientation import _validate_timestamps

        ts = np.array([0.0, 0.01, 0.02, 5.0, 5.01])  # gap at index 3
        dt, valid = _validate_timestamps(ts)
        assert not valid[3], "Large gap should be flagged invalid"


# ═══════════════════════════════════════════════════════════════════════════
# Camera trajectory analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestCameraTrajectory:
    """Tests for camera trajectory estimation from sensor data."""

    def test_estimate_camera_trajectory(self):
        """Trajectory analysis returns expected fields."""
        from sensors.sensor_logger import estimate_camera_trajectory

        # Generate a 90-degree rotation sequence
        n = 50
        angles = np.linspace(0, np.pi / 2, n)
        quats = np.column_stack([
            np.cos(angles / 2),
            np.zeros(n),
            np.sin(angles / 2),
            np.zeros(n),
        ])
        timestamps = np.linspace(0, 1, n)

        result = estimate_camera_trajectory(quats, timestamps)
        assert "total_rotation_deg" in result
        assert "net_rotation_deg" in result
        assert "coverage_assessment" in result
        # Should detect ~90 degrees of rotation
        assert result["net_rotation_deg"] > 80
        assert result["net_rotation_deg"] < 100

    def test_trajectory_insufficient_data(self):
        """Trajectory with <2 samples returns 'insufficient data'."""
        from sensors.sensor_logger import estimate_camera_trajectory

        result = estimate_camera_trajectory(
            np.array([[1, 0, 0, 0]]),
            np.array([0.0]),
        )
        assert result["coverage_assessment"] == "insufficient data"
