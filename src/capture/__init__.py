"""Capture module for Samsung Galaxy S25 Ultra Pro Video ingestion.

Handles frame extraction from H.265/HEVC video, LOG-to-linear color
correction, quality-based frame filtering, multi-source data organization,
and multi-resolution photo processing for 3D face reconstruction.
"""

from capture.frame_extractor import extract_frames
from capture.color_correction import correct_log_to_linear, batch_color_correct
from capture.frame_filter import filter_frames
from capture.photo_processor import process_photos
from capture.data_organizer import (
    organize_capture_session,
    compute_multi_lens_camera_models,
    estimate_photo_timestamps,
)
from capture.multi_res_processor import (
    process_200mp_photos,
    register_multi_lens_photos,
)

__all__ = [
    "extract_frames",
    "correct_log_to_linear",
    "batch_color_correct",
    "filter_frames",
    "process_photos",
    "organize_capture_session",
    "compute_multi_lens_camera_models",
    "estimate_photo_timestamps",
    "process_200mp_photos",
    "register_multi_lens_photos",
]
