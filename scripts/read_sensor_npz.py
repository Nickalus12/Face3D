"""Helper script to read sensor NPZ data and output JSON for the Tauri app.

Usage:
    python read_sensor_npz.py summary <npz_path>
    python read_sensor_npz.py data <npz_path> <sensor_type>

sensor_type: barometer, light, accel_magnitude, gyro_magnitude, orientation
"""

import json
import sys
from pathlib import Path

import numpy as np


def downsample(timestamps: np.ndarray, values: np.ndarray, max_points: int = 500) -> list:
    """Downsample to max_points evenly spaced samples. Returns [[t, v], ...]."""
    n = len(timestamps)
    if n <= max_points:
        indices = np.arange(n)
    else:
        indices = np.linspace(0, n - 1, max_points, dtype=int)

    result = []
    for i in indices:
        t = float(timestamps[i])
        v = float(values[i])
        if not (np.isnan(t) or np.isnan(v)):
            result.append([t, v])
    return result


def get_summary(npz_path: str) -> dict:
    """Return summary info about the sensor NPZ file."""
    data = np.load(npz_path, allow_pickle=True)

    device_name = str(data["device_name"]) if "device_name" in data else "unknown"
    recording_time = str(data["recording_time"]) if "recording_time" in data else ""

    sensors = []
    duration = 0.0

    sensor_map = {
        "accelerometer": ("accel_timestamps", "accel_xyz"),
        "gyroscope": ("gyro_timestamps", "gyro_xyz"),
        "orientation": ("orientation_timestamps", "orientation_quats"),
        "barometer": ("baro_timestamps", "baro_pressure"),
        "light": ("light_timestamps", "light_lux"),
        "magnetometer": ("mag_timestamps", "mag_xyz"),
        "compass": ("compass_timestamps", "compass_heading"),
        "gps": ("gps_timestamps", "gps_lat"),
        "game_orientation": ("game_orientation_timestamps", "game_orientation_quats"),
    }

    for sensor_name, (ts_key, data_key) in sensor_map.items():
        if ts_key in data and data_key in data:
            ts = data[ts_key]
            count = len(ts)
            if count > 0:
                sensor_duration = float(ts[-1] - ts[0])
                duration = max(duration, sensor_duration)
                sensors.append({
                    "name": sensor_name,
                    "samples": count,
                    "duration": round(sensor_duration, 2),
                })

    return {
        "device_name": device_name,
        "recording_time": recording_time,
        "duration": round(duration, 2),
        "sensors": sensors,
    }


def get_data(npz_path: str, sensor_type: str) -> list:
    """Return [[timestamp, value], ...] for a specific sensor type, downsampled."""
    data = np.load(npz_path, allow_pickle=True)

    if sensor_type == "barometer":
        if "baro_timestamps" not in data or "baro_pressure" not in data:
            return []
        return downsample(data["baro_timestamps"], data["baro_pressure"])

    elif sensor_type == "light":
        if "light_timestamps" not in data or "light_lux" not in data:
            return []
        return downsample(data["light_timestamps"], data["light_lux"])

    elif sensor_type == "accel_magnitude":
        if "accel_timestamps" not in data or "accel_xyz" not in data:
            return []
        xyz = data["accel_xyz"]
        magnitude = np.sqrt(np.sum(xyz ** 2, axis=1))
        return downsample(data["accel_timestamps"], magnitude)

    elif sensor_type == "gyro_magnitude":
        if "gyro_timestamps" not in data or "gyro_xyz" not in data:
            return []
        xyz = data["gyro_xyz"]
        magnitude = np.sqrt(np.sum(xyz ** 2, axis=1))
        # Convert rad/s to deg/s
        magnitude = np.degrees(magnitude)
        return downsample(data["gyro_timestamps"], magnitude)

    elif sensor_type == "orientation":
        # Return quaternion data as [[t, qw, qx, qy, qz], ...]
        if "orientation_timestamps" not in data or "orientation_quats" not in data:
            return []
        ts = data["orientation_timestamps"]
        quats = data["orientation_quats"]  # Nx4, wxyz
        n = len(ts)
        max_pts = 500
        if n <= max_pts:
            indices = np.arange(n)
        else:
            indices = np.linspace(0, n - 1, max_pts, dtype=int)
        result = []
        for i in indices:
            t = float(ts[i])
            q = quats[i]
            if not np.any(np.isnan(q)):
                result.append([t, float(q[0]), float(q[1]), float(q[2]), float(q[3])])
        return result

    else:
        return []


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: read_sensor_npz.py <summary|data> <npz_path> [sensor_type]"}))
        sys.exit(1)

    command = sys.argv[1]
    npz_path = sys.argv[2]

    if not Path(npz_path).exists():
        print(json.dumps({"error": f"File not found: {npz_path}"}))
        sys.exit(0)

    try:
        if command == "summary":
            result = get_summary(npz_path)
            print(json.dumps(result))
        elif command == "data":
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Missing sensor_type argument"}))
                sys.exit(1)
            sensor_type = sys.argv[3]
            result = get_data(npz_path, sensor_type)
            print(json.dumps(result))
        else:
            print(json.dumps({"error": f"Unknown command: {command}"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
