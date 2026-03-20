"""Score video frames by camera stability using gyroscope data.

Frames captured during fast camera rotation are blurrier than frames
captured during slow, steady motion. The gyroscope gives angular
velocity at ~460Hz -- we match each video frame timestamp to the nearest
gyroscope readings and compute a stability score.

Score = 1.0 / (1.0 + angular_velocity_magnitude)

High score = camera was still/slow = sharp frame
Low score  = camera was rotating fast = blurry frame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_frame_stability_scores(
    gyro_timestamps: np.ndarray,
    gyro_xyz: np.ndarray,
    frame_timestamps: np.ndarray,
    window_ms: float = 50.0,
) -> np.ndarray:
    """Compute per-frame stability scores from gyroscope angular velocity.

    For each frame timestamp, averages the gyroscope angular velocity
    magnitude over a window centered on that timestamp. Converts to a
    stability score in [0, 1] via 1/(1+mag).

    Args:
        gyro_timestamps: (N,) seconds_elapsed timestamps for gyroscope.
        gyro_xyz: (N, 3) angular velocity in rad/s.
        frame_timestamps: (M,) frame timestamps in seconds_elapsed.
        window_ms: Window size in milliseconds to average gyro around
            each frame timestamp. Default 50ms covers ~23 gyro samples
            at 460Hz, enough to smooth jitter.

    Returns:
        (M,) array of stability scores in [0, 1].
        1.0 = camera was perfectly still, 0.0 = rotating very fast.
    """
    if len(gyro_timestamps) == 0 or len(gyro_xyz) == 0:
        logger.warning("No gyroscope data available, returning all-1.0 scores")
        return np.ones(len(frame_timestamps), dtype=np.float64)

    window_s = window_ms / 1000.0
    half_w = window_s / 2.0
    n_frames = len(frame_timestamps)
    scores = np.ones(n_frames, dtype=np.float64)

    # Precompute angular velocity magnitudes for efficiency
    gyro_mag = np.linalg.norm(gyro_xyz, axis=1)

    for i, ft in enumerate(frame_timestamps):
        mask = (gyro_timestamps >= ft - half_w) & (gyro_timestamps <= ft + half_w)
        if mask.sum() > 0:
            mean_mag = float(gyro_mag[mask].mean())
            scores[i] = 1.0 / (1.0 + mean_mag)
        # else: leave as 1.0 (no data = assume stable)

    return scores


def load_gyro_and_score_frames(
    sensor_npz_path: Union[str, Path],
    frame_timestamps: np.ndarray,
    window_ms: float = 50.0,
) -> np.ndarray:
    """Load gyroscope data from sensor NPZ and compute stability scores.

    Convenience wrapper that loads gyro data from the sensor logger NPZ
    file and calls compute_frame_stability_scores.

    Args:
        sensor_npz_path: Path to sensor_logger_data.npz.
        frame_timestamps: (M,) frame timestamps in seconds_elapsed.
        window_ms: Window size in ms for gyro averaging.

    Returns:
        (M,) stability scores in [0, 1].
    """
    sensor_npz_path = Path(sensor_npz_path)
    data = np.load(str(sensor_npz_path), allow_pickle=True)

    if "gyro_timestamps" not in data or "gyro_xyz" not in data:
        logger.warning("No gyroscope data in %s, returning all-1.0 scores", sensor_npz_path)
        return np.ones(len(frame_timestamps), dtype=np.float64)

    return compute_frame_stability_scores(
        gyro_timestamps=data["gyro_timestamps"],
        gyro_xyz=data["gyro_xyz"],
        frame_timestamps=frame_timestamps,
        window_ms=window_ms,
    )


def score_extracted_frames(
    sensor_npz_path: Union[str, Path],
    num_extracted_frames: int,
    video_duration: float,
    video_fps: float = 60.0,
    window_ms: float = 50.0,
) -> np.ndarray:
    """Score extracted frames by stability, inferring their timestamps.

    When frames are extracted uniformly (every Nth video frame), this
    reconstructs approximate timestamps from the frame count, video
    duration, and FPS, then scores each frame.

    Args:
        sensor_npz_path: Path to sensor_logger_data.npz.
        num_extracted_frames: Number of extracted frames.
        video_duration: Video duration in seconds.
        video_fps: Video frame rate (used to compute frame step).
        window_ms: Window size in ms for gyro averaging.

    Returns:
        (num_extracted_frames,) stability scores in [0, 1].
    """
    # Reconstruct approximate frame timestamps
    # Uniform extraction: effective_fps = num_frames / duration
    # frame_step = round(video_fps / effective_fps)
    # Frame indices: 0, frame_step, 2*frame_step, ...
    effective_fps = num_extracted_frames / max(video_duration, 0.1)
    frame_step = max(1, round(video_fps / effective_fps))
    frame_indices = np.arange(num_extracted_frames) * frame_step
    frame_timestamps = frame_indices / video_fps

    return load_gyro_and_score_frames(
        sensor_npz_path=sensor_npz_path,
        frame_timestamps=frame_timestamps,
        window_ms=window_ms,
    )
