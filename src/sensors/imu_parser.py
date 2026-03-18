"""Parse IMU/sensor data from Samsung Galaxy S25 Ultra video metadata and sidecar files."""

import csv
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _run_ffprobe(video_path: Path) -> dict:
    """Run ffprobe and return parsed JSON output with stream and format info."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _find_motion_stream(probe_info: dict) -> Optional[int]:
    """Find the stream index of the motion metadata track in the container."""
    for stream in probe_info.get("streams", []):
        codec_tag = stream.get("codec_tag_string", "").lower()
        codec_type = stream.get("codec_type", "").lower()
        tags = stream.get("tags", {})
        handler_name = tags.get("handler_name", "").lower()

        # Samsung encodes motion data as a data/metadata stream with recognisable handler names
        is_motion = any(
            kw in handler_name
            for kw in ("motion", "sensor", "gyro", "accel", "imu", "meta")
        )
        is_data_stream = codec_type in ("data", "subtitle", "unknown")

        if is_data_stream and (is_motion or "gpmd" in codec_tag or "mett" in codec_tag):
            return int(stream["index"])

    return None


def _extract_raw_motion_bytes(video_path: Path, stream_index: int) -> bytes:
    """Extract raw bytes from the motion metadata stream using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v", "quiet",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
        "-f", "rawvideo",
        "-codec", "copy",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


def _parse_samsung_motion_bytes(raw: bytes, duration_s: float) -> dict:
    """Parse Samsung-style motion metadata binary payload.

    Samsung Galaxy devices (S21+, S25 Ultra, etc.) embed a packed binary stream
    of accelerometer and gyroscope samples.  The typical layout per sample is
    6 x int16 (little-endian): ax, ay, az, gx, gy, gz.

    If the binary does not divide evenly into 12-byte records we attempt 24-byte
    records (6 x float32) as a fallback.
    """
    if len(raw) == 0:
        raise ValueError("Motion metadata stream is empty")

    RECORD_INT16 = 12  # 6 x int16
    RECORD_FLOAT32 = 24  # 6 x float32

    if len(raw) % RECORD_INT16 == 0:
        n_samples = len(raw) // RECORD_INT16
        data = np.frombuffer(raw, dtype="<i2").reshape(n_samples, 6).astype(np.float64)
        # Samsung int16 scale factors: accelerometer in milli-g, gyroscope in milli-dps
        accel = data[:, :3] * 9.80665 / 1000.0  # -> m/s^2
        gyro = data[:, 3:] * (np.pi / 180.0) / 1000.0  # -> rad/s
    elif len(raw) % RECORD_FLOAT32 == 0:
        n_samples = len(raw) // RECORD_FLOAT32
        data = np.frombuffer(raw, dtype="<f4").reshape(n_samples, 6).astype(np.float64)
        accel = data[:, :3]
        gyro = data[:, 3:]
    else:
        raise ValueError(
            f"Cannot interpret motion stream of {len(raw)} bytes as packed IMU records"
        )

    logger.info("Parsed %d IMU samples from motion stream", n_samples)

    timestamps = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    return {
        "timestamps": timestamps,
        "accel_xyz": accel,
        "gyro_xyz": gyro,
    }


def parse_imu_from_video(video_path: Union[str, Path], output_path: Union[str, Path]) -> dict:
    """Extract IMU data from a Samsung MP4 video's embedded motion metadata track.

    Samsung Galaxy devices record accelerometer and gyroscope data in a dedicated
    metadata stream inside the MP4 container.  This function locates that stream
    via ffprobe, extracts the raw bytes with ffmpeg, parses the binary payload,
    and saves the result as a compressed NumPy archive (.npz).

    Args:
        video_path: Path to the source MP4 video file.
        output_path: Path where the .npz output will be written.

    Returns:
        Dictionary with keys ``timestamps``, ``accel_xyz``, ``gyro_xyz`` (numpy arrays).

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If no motion metadata stream is found.
        subprocess.CalledProcessError: If ffprobe/ffmpeg invocations fail.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info("Probing video for motion metadata: %s", video_path)
    probe = _run_ffprobe(video_path)

    stream_idx = _find_motion_stream(probe)
    if stream_idx is None:
        raise RuntimeError(
            f"No motion/IMU metadata stream found in {video_path}. "
            "Ensure the video was recorded with Samsung motion metadata enabled."
        )

    logger.info("Found motion metadata on stream index %d", stream_idx)

    duration_s = float(probe.get("format", {}).get("duration", 0.0))
    if duration_s <= 0:
        # Fallback: try the video stream duration
        for s in probe["streams"]:
            if s.get("codec_type") == "video":
                duration_s = float(s.get("duration", 0.0))
                break
    if duration_s <= 0:
        raise RuntimeError("Could not determine video duration")

    raw_bytes = _extract_raw_motion_bytes(video_path, stream_idx)
    imu_data = _parse_samsung_motion_bytes(raw_bytes, duration_s)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        timestamps=imu_data["timestamps"],
        accel_xyz=imu_data["accel_xyz"],
        gyro_xyz=imu_data["gyro_xyz"],
    )
    logger.info("Saved IMU data to %s (%d samples)", output_path, len(imu_data["timestamps"]))

    return imu_data


def parse_imu_from_sidecar(sidecar_path: Union[str, Path]) -> dict:
    """Parse IMU data from a sidecar JSON or CSV file.

    Supports two formats:

    **JSON** (Samsung Sensors app style)::

        {
          "samples": [
            {"t": 0.0, "ax": ..., "ay": ..., "az": ..., "gx": ..., "gy": ..., "gz": ...},
            ...
          ]
        }

    Also accepts a flat list of sample objects at the top level.

    **CSV** with columns: ``timestamp, ax, ay, az, gx, gy, gz``

    Accelerometer values are expected in m/s^2 and gyroscope in rad/s.

    Args:
        sidecar_path: Path to the sidecar JSON or CSV file.

    Returns:
        Dictionary with keys ``timestamps``, ``accel_xyz``, ``gyro_xyz``.
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar file not found: {sidecar_path}")

    suffix = sidecar_path.suffix.lower()

    if suffix == ".json":
        with open(sidecar_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        samples = raw if isinstance(raw, list) else raw.get("samples", raw.get("data", []))
        if not samples:
            raise ValueError("No sample data found in JSON sidecar")

        # Detect key naming variants
        first = samples[0]
        t_key = next((k for k in ("t", "time", "timestamp", "ts") if k in first), None)
        if t_key is None:
            raise ValueError("Cannot find timestamp key in JSON samples")

        ax_key = next((k for k in ("ax", "accel_x", "accelerometer_x") if k in first), "ax")
        gx_key = next((k for k in ("gx", "gyro_x", "gyroscope_x") if k in first), "gx")

        # Infer the suffix pattern from detected keys
        a_suffix = ax_key[1:]  # e.g. "x" or "_x" or "cel_x"
        a_prefix = ax_key[: -len(a_suffix)]  # e.g. "a" or "accel" or "accelerometer"
        g_suffix = gx_key[1:]
        g_prefix = gx_key[: -len(g_suffix)]

        timestamps = np.array([s[t_key] for s in samples], dtype=np.float64)
        accel = np.column_stack([
            np.array([s[f"{a_prefix}{c}"] for s in samples], dtype=np.float64)
            for c in (a_suffix, a_suffix.replace("x", "y"), a_suffix.replace("x", "z"))
        ])
        gyro = np.column_stack([
            np.array([s[f"{g_prefix}{c}"] for s in samples], dtype=np.float64)
            for c in (g_suffix, g_suffix.replace("x", "y"), g_suffix.replace("x", "z"))
        ])

    elif suffix == ".csv":
        timestamps_list = []
        accel_list = []
        gyro_list = []

        with open(sidecar_path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = [h.strip().lower() for h in next(reader)]

            # Find column indices
            t_idx = next(
                (i for i, h in enumerate(header) if h in ("t", "time", "timestamp", "ts")),
                0,
            )
            col_map = {h: i for i, h in enumerate(header)}

            ax_key = next((k for k in ("ax", "accel_x", "accelerometer_x") if k in col_map), None)
            if ax_key is None:
                raise ValueError(f"Cannot find accelerometer columns in CSV header: {header}")
            ay_key = ax_key.replace("x", "y")
            az_key = ax_key.replace("x", "z")
            gx_key = next((k for k in ("gx", "gyro_x", "gyroscope_x") if k in col_map), None)
            if gx_key is None:
                raise ValueError(f"Cannot find gyroscope columns in CSV header: {header}")
            gy_key = gx_key.replace("x", "y")
            gz_key = gx_key.replace("x", "z")

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                timestamps_list.append(float(row[t_idx]))
                accel_list.append([
                    float(row[col_map[ax_key]]),
                    float(row[col_map[ay_key]]),
                    float(row[col_map[az_key]]),
                ])
                gyro_list.append([
                    float(row[col_map[gx_key]]),
                    float(row[col_map[gy_key]]),
                    float(row[col_map[gz_key]]),
                ])

        timestamps = np.array(timestamps_list, dtype=np.float64)
        accel = np.array(accel_list, dtype=np.float64)
        gyro = np.array(gyro_list, dtype=np.float64)

    else:
        raise ValueError(f"Unsupported sidecar format: {suffix} (expected .json or .csv)")

    logger.info("Parsed %d IMU samples from sidecar %s", len(timestamps), sidecar_path.name)

    return {
        "timestamps": timestamps,
        "accel_xyz": accel,
        "gyro_xyz": gyro,
    }


def align_timestamps(
    imu_data: dict,
    frame_timestamps: np.ndarray,
) -> dict:
    """Interpolate IMU data (100-500 Hz) to align with video frame timestamps.

    Uses linear interpolation for accelerometer and gyroscope channels so that
    each video frame has a corresponding IMU sample.

    Args:
        imu_data: Dictionary with ``timestamps``, ``accel_xyz``, ``gyro_xyz``.
        frame_timestamps: 1-D array of video frame timestamps (seconds).

    Returns:
        New dictionary with the same keys, resampled to *frame_timestamps*.
    """
    imu_t = imu_data["timestamps"]
    accel = imu_data["accel_xyz"]
    gyro = imu_data["gyro_xyz"]

    if len(imu_t) < 2:
        raise ValueError("Need at least 2 IMU samples for interpolation")

    aligned_accel = np.column_stack([
        np.interp(frame_timestamps, imu_t, accel[:, axis])
        for axis in range(3)
    ])
    aligned_gyro = np.column_stack([
        np.interp(frame_timestamps, imu_t, gyro[:, axis])
        for axis in range(3)
    ])

    logger.info(
        "Aligned %d IMU samples -> %d frame-level samples",
        len(imu_t),
        len(frame_timestamps),
    )

    return {
        "timestamps": frame_timestamps.copy(),
        "accel_xyz": aligned_accel,
        "gyro_xyz": aligned_gyro,
    }
