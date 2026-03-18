"""Capture module for Samsung Galaxy S25 Ultra Pro Video ingestion.

Handles frame extraction from H.265/HEVC video, LOG-to-linear color
correction, and quality-based frame filtering for 3D face reconstruction.
"""

from capture.frame_extractor import extract_frames
from capture.color_correction import correct_log_to_linear, batch_color_correct
from capture.frame_filter import filter_frames
from capture.photo_processor import process_photos

__all__ = [
    "extract_frames",
    "correct_log_to_linear",
    "batch_color_correct",
    "filter_frames",
    "process_photos",
]
