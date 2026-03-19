"""Parse Sensor Logger app (https://www.tszheichoi.com/sensorlogger) ZIP exports.

The Sensor Logger app for Android records synchronized sensor data as CSV files
inside a ZIP archive. This module extracts and aligns that data to video frames,
producing high-quality rotation priors for COLMAP from Samsung's pre-fused
orientation quaternions (which are far superior to Madgwick-filtered IMU data).

Supported CSVs:
    Accelerometer.csv, Gyroscope.csv, Orientation.csv, Magnetometer.csv,
    Compass.csv, Barometer.csv, Location.csv, Light.csv, GameOrientation.csv,
    Metadata.csv
"""

import csv
import io
import logging
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict]:
    """Read a CSV file from the ZIP and return rows as list of dicts."""
    # Sensor Logger may nest files at root or inside a folder
    matching = [n for n in zf.namelist() if n.endswith(name)]
    if not matching:
        return []
    with zf.open(matching[0]) as f:
        text = io.TextIOWrapper(f, encoding="utf-8")
        reader = csv.DictReader(text)
        return list(reader)


def _extract_timestamps(rows: list[dict]) -> np.ndarray:
    """Extract nanosecond epoch timestamps from the 'time' column."""
    return np.array([int(r["time"]) for r in rows], dtype=np.int64)


def _extract_seconds_elapsed(rows: list[dict]) -> np.ndarray:
    """Extract seconds_elapsed as float64 array."""
    return np.array([float(r["seconds_elapsed"]) for r in rows], dtype=np.float64)


def _safe_float_col(rows: list[dict], col: str) -> np.ndarray:
    """Extract a float column, returning NaN for missing/empty values."""
    vals = []
    for r in rows:
        v = r.get(col, "")
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            vals.append(float("nan"))
    return np.array(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Parse Sensor Logger ZIP
# ---------------------------------------------------------------------------

def parse_sensor_logger_zip(
    zip_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> tuple[Path, dict]:
    """Extract and parse all sensor CSVs from a Sensor Logger ZIP archive.

    Saves a comprehensive NPZ file with all available sensor streams and
    returns the path to it along with a summary dict.

    Args:
        zip_path: Path to the Sensor Logger .zip file.
        output_dir: Directory where the output .npz will be saved.

    Returns:
        Tuple of (npz_path, summary_dict).
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Sensor Logger ZIP not found: {zip_path}")

    npz_data: dict[str, np.ndarray] = {}
    summary: dict = {"zip_path": str(zip_path), "sensors": []}

    with zipfile.ZipFile(zip_path, "r") as zf:
        # -- Metadata --
        meta_rows = _read_csv_from_zip(zf, "Metadata.csv")
        if meta_rows:
            # Metadata.csv typically has key-value rows or a single info row
            meta = {}
            for row in meta_rows:
                # Some formats: columns are the keys themselves
                meta.update(row)
            device_name = meta.get("device name", meta.get("device", "unknown"))
            recording_time = meta.get("recording time", meta.get("time", ""))
            npz_data["device_name"] = np.array(str(device_name))
            npz_data["recording_time"] = np.array(str(recording_time))
            summary["device_name"] = str(device_name)
            summary["recording_time"] = str(recording_time)
            logger.info("Sensor Logger metadata: device=%s, time=%s", device_name, recording_time)
        else:
            npz_data["device_name"] = np.array("unknown")
            npz_data["recording_time"] = np.array("")

        # -- Accelerometer --
        accel_rows = _read_csv_from_zip(zf, "Accelerometer.csv")
        if accel_rows:
            npz_data["accel_timestamps"] = _extract_seconds_elapsed(accel_rows)
            npz_data["accel_epoch_ns"] = _extract_timestamps(accel_rows)
            npz_data["accel_xyz"] = np.column_stack([
                _safe_float_col(accel_rows, "x"),
                _safe_float_col(accel_rows, "y"),
                _safe_float_col(accel_rows, "z"),
            ])
            summary["sensors"].append(("accelerometer", len(accel_rows)))
            logger.info("Accelerometer: %d samples", len(accel_rows))

        # -- Gyroscope --
        gyro_rows = _read_csv_from_zip(zf, "Gyroscope.csv")
        if gyro_rows:
            npz_data["gyro_timestamps"] = _extract_seconds_elapsed(gyro_rows)
            npz_data["gyro_epoch_ns"] = _extract_timestamps(gyro_rows)
            npz_data["gyro_xyz"] = np.column_stack([
                _safe_float_col(gyro_rows, "x"),
                _safe_float_col(gyro_rows, "y"),
                _safe_float_col(gyro_rows, "z"),
            ])
            summary["sensors"].append(("gyroscope", len(gyro_rows)))
            logger.info("Gyroscope: %d samples", len(gyro_rows))

        # -- Orientation (pre-fused quaternions!) --
        orient_rows = _read_csv_from_zip(zf, "Orientation.csv")
        if orient_rows:
            npz_data["orientation_timestamps"] = _extract_seconds_elapsed(orient_rows)
            npz_data["orientation_epoch_ns"] = _extract_timestamps(orient_rows)
            # Sensor Logger outputs qx, qy, qz, qw — store as wxyz (COLMAP convention order)
            qx = _safe_float_col(orient_rows, "qx")
            qy = _safe_float_col(orient_rows, "qy")
            qz = _safe_float_col(orient_rows, "qz")
            qw = _safe_float_col(orient_rows, "qw")
            npz_data["orientation_quats"] = np.column_stack([qw, qx, qy, qz])  # wxyz
            npz_data["orientation_euler"] = np.column_stack([
                _safe_float_col(orient_rows, "roll"),
                _safe_float_col(orient_rows, "pitch"),
                _safe_float_col(orient_rows, "yaw"),
            ])
            summary["sensors"].append(("orientation", len(orient_rows)))
            logger.info("Orientation: %d samples (pre-fused quaternions)", len(orient_rows))

        # -- GameOrientation (game rotation vector, no magnetometer) --
        game_rows = _read_csv_from_zip(zf, "GameOrientation.csv")
        if game_rows:
            npz_data["game_orientation_timestamps"] = _extract_seconds_elapsed(game_rows)
            gqx = _safe_float_col(game_rows, "qx")
            gqy = _safe_float_col(game_rows, "qy")
            gqz = _safe_float_col(game_rows, "qz")
            gqw = _safe_float_col(game_rows, "qw")
            npz_data["game_orientation_quats"] = np.column_stack([gqw, gqx, gqy, gqz])
            summary["sensors"].append(("game_orientation", len(game_rows)))
            logger.info("GameOrientation: %d samples", len(game_rows))

        # -- Magnetometer --
        mag_rows = _read_csv_from_zip(zf, "Magnetometer.csv")
        if mag_rows:
            npz_data["mag_timestamps"] = _extract_seconds_elapsed(mag_rows)
            npz_data["mag_xyz"] = np.column_stack([
                _safe_float_col(mag_rows, "x"),
                _safe_float_col(mag_rows, "y"),
                _safe_float_col(mag_rows, "z"),
            ])
            summary["sensors"].append(("magnetometer", len(mag_rows)))
            logger.info("Magnetometer: %d samples", len(mag_rows))

        # -- Compass --
        compass_rows = _read_csv_from_zip(zf, "Compass.csv")
        if compass_rows:
            npz_data["compass_timestamps"] = _extract_seconds_elapsed(compass_rows)
            npz_data["compass_heading"] = _safe_float_col(compass_rows, "heading")
            summary["sensors"].append(("compass", len(compass_rows)))
            logger.info("Compass: %d samples", len(compass_rows))

        # -- Barometer --
        baro_rows = _read_csv_from_zip(zf, "Barometer.csv")
        if baro_rows:
            npz_data["baro_timestamps"] = _extract_seconds_elapsed(baro_rows)
            npz_data["baro_pressure"] = _safe_float_col(baro_rows, "pressure")
            summary["sensors"].append(("barometer", len(baro_rows)))
            logger.info("Barometer: %d samples", len(baro_rows))

        # -- Light --
        light_rows = _read_csv_from_zip(zf, "Light.csv")
        if light_rows:
            npz_data["light_timestamps"] = _extract_seconds_elapsed(light_rows)
            npz_data["light_lux"] = _safe_float_col(light_rows, "lux")
            summary["sensors"].append(("light", len(light_rows)))
            logger.info("Light: %d samples", len(light_rows))

        # -- Location (GPS) --
        loc_rows = _read_csv_from_zip(zf, "Location.csv")
        if loc_rows:
            npz_data["gps_timestamps"] = _extract_seconds_elapsed(loc_rows)
            npz_data["gps_lat"] = _safe_float_col(loc_rows, "latitude")
            npz_data["gps_lon"] = _safe_float_col(loc_rows, "longitude")
            npz_data["gps_alt"] = _safe_float_col(loc_rows, "altitude")
            summary["sensors"].append(("gps", len(loc_rows)))
            logger.info("GPS/Location: %d samples", len(loc_rows))

        # -- Recording start epoch (from first accelerometer timestamp or metadata) --
        if "accel_epoch_ns" in npz_data and len(npz_data["accel_epoch_ns"]) > 0:
            npz_data["recording_start_epoch_ns"] = np.array(npz_data["accel_epoch_ns"][0])
        elif "orientation_epoch_ns" in npz_data and len(npz_data["orientation_epoch_ns"]) > 0:
            npz_data["recording_start_epoch_ns"] = np.array(npz_data["orientation_epoch_ns"][0])
        else:
            npz_data["recording_start_epoch_ns"] = np.array(0, dtype=np.int64)

    # Save NPZ
    npz_path = output_dir / "sensor_logger_data.npz"
    np.savez_compressed(str(npz_path), **npz_data)

    total_samples = sum(count for _, count in summary["sensors"])
    logger.info(
        "Saved Sensor Logger data: %d sensors, %d total samples -> %s",
        len(summary["sensors"]),
        total_samples,
        npz_path,
    )
    summary["npz_path"] = str(npz_path)
    summary["total_samples"] = total_samples

    return npz_path, summary


# ---------------------------------------------------------------------------
# 2. Align sensor data to video frames
# ---------------------------------------------------------------------------

def align_sensor_to_video(
    sensor_npz_path: Union[str, Path],
    video_path: Union[str, Path],
    video_start_frame_time: Optional[float] = None,
) -> dict:
    """Interpolate all sensor streams to video frame timestamps.

    The sensor logger and video are started independently. This function
    computes frame timestamps from the video FPS and interpolates sensor
    data to match.

    When ``video_start_frame_time`` is provided (seconds_elapsed in sensor
    time), it is used as the offset. Otherwise we assume the sensor recording
    started approximately when the video started (offset = 0).

    Args:
        sensor_npz_path: Path to the sensor_logger_data.npz file.
        video_path: Path to the video file (used to extract FPS and duration).
        video_start_frame_time: Optional offset in seconds_elapsed to align
            sensor time to video time. If None, offset is assumed 0.

    Returns:
        Dict with per-frame interpolated sensor data:
            frame_quaternions (Nx4, wxyz), frame_accel (Nx3), frame_gyro (Nx3),
            frame_light (N,), frame_heading (N,), frame_timestamps (N,),
            num_frames (int).
    """
    import json
    import subprocess

    sensor_npz_path = Path(sensor_npz_path)
    video_path = Path(video_path)

    # Load sensor data
    data = np.load(str(sensor_npz_path), allow_pickle=True)

    # Get video info via ffprobe
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    # Extract FPS and duration from video stream
    fps = None
    duration = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            # Parse r_frame_rate (e.g. "30000/1001" or "30/1")
            r_fps = stream.get("r_frame_rate", "30/1")
            num, den = r_fps.split("/")
            fps = float(num) / float(den)
            duration = float(stream.get("duration", probe.get("format", {}).get("duration", 0)))
            break

    if fps is None or fps <= 0:
        fps = 30.0
        logger.warning("Could not detect video FPS, defaulting to %.1f", fps)
    if duration is None or duration <= 0:
        duration = float(probe.get("format", {}).get("duration", 0))

    num_frames = int(duration * fps)
    if num_frames <= 0:
        raise ValueError(f"Could not determine frame count from video: {video_path}")

    # Compute video frame timestamps in sensor time (seconds_elapsed)
    offset = video_start_frame_time if video_start_frame_time is not None else 0.0
    frame_timestamps = offset + np.arange(num_frames, dtype=np.float64) / fps

    logger.info(
        "Video: %.1f FPS, %.1f s, %d frames (sensor offset=%.3f s)",
        fps, duration, num_frames, offset,
    )

    result_dict: dict = {
        "frame_timestamps": frame_timestamps,
        "num_frames": num_frames,
        "fps": fps,
        "duration": duration,
    }

    # -- Interpolate orientation quaternions --
    if "orientation_timestamps" in data and "orientation_quats" in data:
        orient_t = data["orientation_timestamps"]
        orient_q = data["orientation_quats"]  # Nx4, wxyz

        if len(orient_t) >= 2:
            # SLERP interpolation for quaternions
            from scipy.spatial.transform import Rotation, Slerp

            # Ensure monotonic timestamps (drop duplicates)
            unique_mask = np.concatenate(([True], np.diff(orient_t) > 1e-9))
            t_unique = orient_t[unique_mask]
            q_unique = orient_q[unique_mask]

            if len(t_unique) >= 2:
                # scipy Rotation expects xyzw, our data is wxyz
                q_xyzw = np.column_stack([q_unique[:, 1], q_unique[:, 2], q_unique[:, 3], q_unique[:, 0]])
                rots = Rotation.from_quat(q_xyzw)

                # Clamp frame times to sensor range
                t_clamped = np.clip(frame_timestamps, t_unique[0], t_unique[-1])
                slerp = Slerp(t_unique, rots)
                interp_rots = slerp(t_clamped)

                # Convert back to wxyz
                q_interp_xyzw = interp_rots.as_quat()  # xyzw
                frame_quats = np.column_stack([
                    q_interp_xyzw[:, 3],  # w
                    q_interp_xyzw[:, 0],  # x
                    q_interp_xyzw[:, 1],  # y
                    q_interp_xyzw[:, 2],  # z
                ])
                result_dict["frame_quaternions"] = frame_quats
                logger.info("Interpolated orientation: %d sensor -> %d frames (SLERP)", len(orient_t), num_frames)
            else:
                logger.warning("Not enough unique orientation timestamps for SLERP")
        else:
            logger.warning("Orientation data has fewer than 2 samples, skipping interpolation")

    # -- Interpolate accelerometer --
    if "accel_timestamps" in data and "accel_xyz" in data:
        accel_t = data["accel_timestamps"]
        accel_xyz = data["accel_xyz"]
        if len(accel_t) >= 2:
            t_clamped = np.clip(frame_timestamps, accel_t[0], accel_t[-1])
            frame_accel = np.column_stack([
                np.interp(t_clamped, accel_t, accel_xyz[:, i])
                for i in range(3)
            ])
            result_dict["frame_accel"] = frame_accel
            logger.info("Interpolated accelerometer: %d sensor -> %d frames", len(accel_t), num_frames)

    # -- Interpolate gyroscope --
    if "gyro_timestamps" in data and "gyro_xyz" in data:
        gyro_t = data["gyro_timestamps"]
        gyro_xyz = data["gyro_xyz"]
        if len(gyro_t) >= 2:
            t_clamped = np.clip(frame_timestamps, gyro_t[0], gyro_t[-1])
            frame_gyro = np.column_stack([
                np.interp(t_clamped, gyro_t, gyro_xyz[:, i])
                for i in range(3)
            ])
            result_dict["frame_gyro"] = frame_gyro
            logger.info("Interpolated gyroscope: %d sensor -> %d frames", len(gyro_t), num_frames)

    # -- Interpolate light sensor --
    if "light_timestamps" in data and "light_lux" in data:
        light_t = data["light_timestamps"]
        light_lux = data["light_lux"]
        if len(light_t) >= 2:
            t_clamped = np.clip(frame_timestamps, light_t[0], light_t[-1])
            frame_light = np.interp(t_clamped, light_t, light_lux)
            result_dict["frame_light"] = frame_light
            logger.info("Interpolated light: %d sensor -> %d frames", len(light_t), num_frames)

    # -- Interpolate compass heading --
    if "compass_timestamps" in data and "compass_heading" in data:
        compass_t = data["compass_timestamps"]
        heading = data["compass_heading"]
        if len(compass_t) >= 2:
            # Heading is circular (0-360), use sin/cos interpolation
            heading_rad = np.deg2rad(heading)
            t_clamped = np.clip(frame_timestamps, compass_t[0], compass_t[-1])
            sin_interp = np.interp(t_clamped, compass_t, np.sin(heading_rad))
            cos_interp = np.interp(t_clamped, compass_t, np.cos(heading_rad))
            frame_heading = np.rad2deg(np.arctan2(sin_interp, cos_interp)) % 360.0
            result_dict["frame_heading"] = frame_heading
            logger.info("Interpolated compass: %d sensor -> %d frames", len(compass_t), num_frames)

    return result_dict


# ---------------------------------------------------------------------------
# 3. Generate COLMAP rotation priors from pre-fused orientations
# ---------------------------------------------------------------------------

def _android_enu_quat_to_colmap(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert an Android ENU orientation quaternion to COLMAP camera frame.

    Android's rotation vector sensor provides orientation in the ENU
    (East-North-Up) frame:
        - X points East
        - Y points North
        - Z points Up

    The quaternion represents the rotation from the world (ENU) frame to
    the device frame (where X=right, Y=up, Z=out-of-screen).

    COLMAP's camera coordinate system:
        - X points right
        - Y points down
        - Z points forward (into the scene)

    The device-to-camera transform flips Y and Z:
        R_cam_from_device = diag(1, -1, -1)

    So: R_colmap = R_cam_from_device @ R_device_from_world
                 = diag(1,-1,-1) @ R_android

    Args:
        q_wxyz: (N, 4) or (4,) quaternions in wxyz order from Android.

    Returns:
        (N, 4) or (4,) quaternions in wxyz order for COLMAP.
    """
    single = q_wxyz.ndim == 1
    if single:
        q_wxyz = q_wxyz.reshape(1, 4)

    # Convert wxyz -> scipy xyzw for Rotation
    q_xyzw = np.column_stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]])
    rots = Rotation.from_quat(q_xyzw)

    # Device-to-camera transform: flip Y and Z
    R_cam_from_device = np.diag([1.0, -1.0, -1.0])

    # Apply: R_colmap = R_cam_from_device @ R_android
    colmap_matrices = np.array([R_cam_from_device @ r.as_matrix() for r in rots])
    colmap_rots = Rotation.from_matrix(colmap_matrices)

    # Convert back to wxyz with canonical form (w >= 0)
    q_out_xyzw = colmap_rots.as_quat(canonical=True)
    q_out_wxyz = np.column_stack([q_out_xyzw[:, 3], q_out_xyzw[:, 0], q_out_xyzw[:, 1], q_out_xyzw[:, 2]])

    if single:
        return q_out_wxyz[0]
    return q_out_wxyz


def generate_rotation_priors_from_sensor_logger(
    sensor_data: dict,
    frame_names: Sequence[str],
    output_path: Union[str, Path],
    q_std: float = 0.005,
) -> Path:
    """Write COLMAP image_priors.txt from Sensor Logger pre-fused quaternions.

    These priors are significantly more accurate than Madgwick-filtered IMU
    because Samsung's sensor fusion algorithm combines accelerometer, gyroscope,
    and magnetometer data with a proprietary Kalman filter.

    Args:
        sensor_data: Dict from ``align_sensor_to_video``, must contain
            ``frame_quaternions`` (Nx4, wxyz in Android ENU frame).
        frame_names: Sequence of image filenames matching the frame order.
        output_path: Path for the output priors file.
        q_std: Quaternion standard deviation (tighter = more trust in sensor).
            Default 0.005 is tighter than Madgwick's 0.01 because pre-fused
            orientations are more reliable.

    Returns:
        Resolved Path to the written file.
    """
    output_path = Path(output_path)

    if "frame_quaternions" not in sensor_data:
        raise ValueError("sensor_data must contain 'frame_quaternions' from align_sensor_to_video")

    frame_quats = sensor_data["frame_quaternions"]  # Nx4, wxyz, Android ENU
    n_frames = len(frame_names)

    if len(frame_quats) < n_frames:
        logger.warning(
            "Fewer quaternions (%d) than frame names (%d); truncating frame list",
            len(frame_quats), n_frames,
        )
        n_frames = len(frame_quats)
    elif len(frame_quats) > n_frames:
        frame_quats = frame_quats[:n_frames]

    # Convert Android ENU -> COLMAP camera frame
    colmap_quats = _android_enu_quat_to_colmap(frame_quats)

    # Translation uncertainty: effectively unconstrained
    t_std = 1e6

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# COLMAP image pose priors from Sensor Logger pre-fused orientation\n"
            "# IMAGE_NAME QW QX QY QZ TX TY TZ "
            "QW_STD QX_STD QY_STD QZ_STD TX_STD TY_STD TZ_STD\n"
        )
        for i in range(n_frames):
            q = colmap_quats[i]
            line = (
                f"{frame_names[i]} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                f"0.0 0.0 0.0 "
                f"{q_std} {q_std} {q_std} {q_std} "
                f"{t_std:.1f} {t_std:.1f} {t_std:.1f}\n"
            )
            fh.write(line)

    logger.info(
        "Wrote COLMAP priors for %d frames from Sensor Logger data to %s "
        "(q_std=%.4f, much tighter than Madgwick)",
        n_frames, output_path, q_std,
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# 4. Estimate camera trajectory from orientation data
# ---------------------------------------------------------------------------

def estimate_camera_trajectory(
    orientation_quats: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """Analyse the camera trajectory from orientation quaternions.

    Computes the total angular coverage, rotation rate, and identifies the
    primary rotation axis. This validates that the capture covered enough
    viewing angles for good 3D reconstruction.

    Args:
        orientation_quats: (N, 4) quaternions in wxyz order.
        timestamps: (N,) timestamps in seconds.

    Returns:
        Dict with trajectory analysis:
            total_rotation_deg, mean_rotation_rate_deg_s,
            max_rotation_rate_deg_s, primary_axis, coverage_assessment.
    """
    if len(orientation_quats) < 2:
        return {
            "total_rotation_deg": 0.0,
            "mean_rotation_rate_deg_s": 0.0,
            "max_rotation_rate_deg_s": 0.0,
            "primary_axis": "unknown",
            "coverage_assessment": "insufficient data",
        }

    # Convert wxyz -> scipy xyzw
    q_xyzw = np.column_stack([
        orientation_quats[:, 1],
        orientation_quats[:, 2],
        orientation_quats[:, 3],
        orientation_quats[:, 0],
    ])
    rots = Rotation.from_quat(q_xyzw)

    # Compute relative rotations between consecutive frames
    total_angle = 0.0
    angles = []
    axis_accum = np.zeros(3)

    for i in range(1, len(rots)):
        rel_rot = rots[i - 1].inv() * rots[i]
        rotvec = rel_rot.as_rotvec()
        angle = np.linalg.norm(rotvec)
        angles.append(angle)
        total_angle += angle
        if angle > 1e-6:
            axis_accum += np.abs(rotvec / angle) * angle

    total_deg = np.degrees(total_angle)

    # Also compute the total rotation from first to last frame
    total_rot = rots[0].inv() * rots[-1]
    net_rotation_deg = np.degrees(np.linalg.norm(total_rot.as_rotvec()))

    # Rotation rates
    dt = np.diff(timestamps)
    dt = np.clip(dt, 1e-6, None)  # avoid division by zero
    rates_deg = np.degrees(np.array(angles)) / dt

    # Primary axis
    axis_labels = ["X (pitch)", "Y (yaw)", "Z (roll)"]
    if np.sum(axis_accum) > 1e-6:
        axis_norm = axis_accum / np.sum(axis_accum)
        primary_idx = int(np.argmax(axis_norm))
        primary_axis = axis_labels[primary_idx]
    else:
        primary_axis = "unknown"

    # Coverage assessment
    if net_rotation_deg >= 270:
        assessment = f"Excellent: camera orbited ~{net_rotation_deg:.0f} degrees"
    elif net_rotation_deg >= 180:
        assessment = f"Good: camera orbited ~{net_rotation_deg:.0f} degrees"
    elif net_rotation_deg >= 90:
        assessment = f"Fair: camera orbited ~{net_rotation_deg:.0f} degrees (more coverage recommended)"
    else:
        assessment = f"Limited: only ~{net_rotation_deg:.0f} degrees covered (need more angles)"

    result = {
        "total_rotation_deg": round(total_deg, 1),
        "net_rotation_deg": round(net_rotation_deg, 1),
        "mean_rotation_rate_deg_s": round(float(np.mean(rates_deg)), 1),
        "max_rotation_rate_deg_s": round(float(np.max(rates_deg)), 1),
        "primary_axis": primary_axis,
        "coverage_assessment": assessment,
    }

    logger.info(
        "Camera trajectory: total=%.1f deg, net=%.1f deg, mean rate=%.1f deg/s — %s",
        total_deg, net_rotation_deg, np.mean(rates_deg), assessment,
    )
    return result


# ---------------------------------------------------------------------------
# 5. Compute lighting profile from light sensor
# ---------------------------------------------------------------------------

def compute_lighting_profile(
    light_lux: np.ndarray,
    timestamps: np.ndarray,
    frame_timestamps: np.ndarray,
    outlier_factor: float = 2.0,
) -> dict:
    """Compute per-frame lighting adjustment factors from the ambient light sensor.

    Identifies frames captured under significantly different lighting
    conditions. Returns adjustment factors that can be used to normalize
    appearance or weight the appearance embedding.

    Args:
        light_lux: (N,) ambient light readings in lux.
        timestamps: (N,) sensor timestamps in seconds.
        frame_timestamps: (M,) frame timestamps in seconds.
        outlier_factor: Frames with lux > median * factor or < median / factor
            are flagged as outliers.

    Returns:
        Dict with:
            frame_lux (M,): interpolated lux values per frame
            frame_adjustment_factors (M,): multiplicative adjustment factors
                (1.0 = median lighting, >1.0 = darker than median, <1.0 = brighter)
            outlier_mask (M,): boolean mask of frames with unusual lighting
            median_lux: median illumination across all frames
            lux_range: (min, max) lux across frames
    """
    if len(light_lux) < 2 or len(timestamps) < 2:
        n_frames = len(frame_timestamps)
        return {
            "frame_lux": np.ones(n_frames, dtype=np.float64),
            "frame_adjustment_factors": np.ones(n_frames, dtype=np.float64),
            "outlier_mask": np.zeros(n_frames, dtype=bool),
            "median_lux": 1.0,
            "lux_range": (1.0, 1.0),
        }

    # Remove NaN values
    valid = ~np.isnan(light_lux) & ~np.isnan(timestamps)
    if np.sum(valid) < 2:
        n_frames = len(frame_timestamps)
        return {
            "frame_lux": np.ones(n_frames, dtype=np.float64),
            "frame_adjustment_factors": np.ones(n_frames, dtype=np.float64),
            "outlier_mask": np.zeros(n_frames, dtype=bool),
            "median_lux": 1.0,
            "lux_range": (1.0, 1.0),
        }

    t_valid = timestamps[valid]
    lux_valid = light_lux[valid]

    # Interpolate to frame times
    t_clamped = np.clip(frame_timestamps, t_valid[0], t_valid[-1])
    frame_lux = np.interp(t_clamped, t_valid, lux_valid)

    # Compute adjustment factors relative to median
    median_lux = float(np.median(frame_lux))
    if median_lux < 1e-6:
        median_lux = 1.0

    # Factor: ratio of median to actual (darker frames get > 1.0 factor)
    adjustment = np.where(frame_lux > 1e-6, median_lux / frame_lux, 1.0)

    # Outlier detection
    outlier_mask = (frame_lux > median_lux * outlier_factor) | (frame_lux < median_lux / outlier_factor)
    n_outliers = int(np.sum(outlier_mask))

    if n_outliers > 0:
        logger.warning(
            "Lighting profile: %d/%d frames have unusual lighting (>%.0fx change from median %.0f lux)",
            n_outliers, len(frame_timestamps), outlier_factor, median_lux,
        )
    else:
        logger.info(
            "Lighting profile: consistent lighting (median=%.0f lux, range=%.0f-%.0f lux)",
            median_lux, float(np.min(frame_lux)), float(np.max(frame_lux)),
        )

    return {
        "frame_lux": frame_lux,
        "frame_adjustment_factors": adjustment,
        "outlier_mask": outlier_mask,
        "median_lux": median_lux,
        "lux_range": (float(np.min(frame_lux)), float(np.max(frame_lux))),
    }
