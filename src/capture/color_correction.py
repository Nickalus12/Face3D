"""LOG-to-linear color correction for Samsung Galaxy S25 Ultra Pro Video.

Converts S-Log3 (and similar LOG curves) to linear light, then optionally
to sRGB for COLMAP feature matching. All operations use float32 precision
to preserve dynamic range.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _slog3_to_linear(x: np.ndarray) -> np.ndarray:
    """Apply the inverse S-Log3 transfer function.

    S-Log3 is defined piecewise. Values are expected in [0, 1] range.
    Reference: Sony S-Log3 white paper.
    """
    x = np.clip(x, 0.0, 1.0).astype(np.float64)
    out = np.empty_like(x)

    # Cutpoint in S-Log3 encoded domain
    cut = 171.2102946929 / 1023.0  # ~0.16736

    linear_region = x < cut
    # Linear segment below the cut
    out[linear_region] = (x[linear_region] * 1023.0 - 95.0) / (
        171.2102946929 - 95.0
    ) * 0.01125000

    # Curve segment above the cut
    curve = ~linear_region
    out[curve] = (
        10.0 ** ((x[curve] * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01
    )

    return np.clip(out, 0.0, None).astype(np.float32)


def _linear_to_srgb_curve(linear: np.ndarray) -> np.ndarray:
    """Apply the sRGB OETF (linear -> sRGB gamma)."""
    linear = np.clip(linear, 0.0, 1.0).astype(np.float32)
    out = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def correct_log_to_linear(
    frame_path: str | Path,
    output_path: str | Path,
    log_type: str = "slog3",
) -> Path:
    """Convert a LOG-encoded frame to linear light and save it.

    Reads the image, applies the inverse LOG transfer function to produce
    scene-linear values, then writes the result as a 16-bit PNG to
    preserve the expanded dynamic range.

    Args:
        frame_path: Path to the input LOG-encoded frame (PNG/JPEG/TIFF).
        output_path: Destination path for the linear-light output (PNG).
        log_type: LOG curve type. Currently supports ``"slog3"``.

    Returns:
        Path to the saved linear-light frame.

    Raises:
        ValueError: If *log_type* is not supported.
        FileNotFoundError: If the input frame does not exist.
    """
    frame_path = Path(frame_path)
    output_path = Path(output_path)

    if not frame_path.is_file():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    supported = {"slog3"}
    if log_type not in supported:
        raise ValueError(
            f"Unsupported log_type '{log_type}'. Supported: {supported}"
        )

    img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {frame_path}")

    # Normalize to [0, 1] float
    if img.dtype == np.uint16:
        img_float = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)

    if log_type == "slog3":
        linear = _slog3_to_linear(img_float)
    else:
        linear = img_float  # fallback, shouldn't reach here

    # Normalize linear output to [0, 1] for 16-bit encoding
    max_val = linear.max()
    if max_val > 0:
        linear_norm = linear / max_val
    else:
        linear_norm = linear

    out_16 = (np.clip(linear_norm, 0.0, 1.0) * 65535.0).astype(np.uint16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out_16)
    logger.debug("Linear frame saved: %s", output_path)

    return output_path


def linear_to_srgb(linear_frame: np.ndarray) -> np.ndarray:
    """Convert a linear-light frame (float32, [0..1]) to sRGB uint8.

    Applies the standard sRGB OETF and quantizes to 8-bit for use with
    COLMAP and other tools that expect gamma-encoded input.

    Args:
        linear_frame: HxWxC float32 array in linear light, range [0, 1].

    Returns:
        HxWxC uint8 array in sRGB gamma space.
    """
    if linear_frame.dtype != np.float32:
        linear_frame = linear_frame.astype(np.float32)

    srgb = _linear_to_srgb_curve(linear_frame)
    return (srgb * 255.0).astype(np.uint8)


def batch_color_correct(
    frames_dir: str | Path,
    output_srgb_dir: str | Path,
    output_linear_dir: str | Path | None = None,
    log_type: str = "slog3",
) -> list[Path]:
    """Process all frames in a directory: LOG -> linear -> sRGB.

    For each PNG/JPG frame in *frames_dir*, converts from LOG to linear
    light, then applies sRGB gamma. Optionally saves the intermediate
    linear frames as 16-bit PNGs.

    Args:
        frames_dir: Directory containing LOG-encoded input frames.
        output_srgb_dir: Directory for sRGB output frames (8-bit PNG).
        output_linear_dir: Optional directory for linear output frames
            (16-bit PNG). If ``None``, linear intermediates are not saved
            to disk.
        log_type: LOG curve identifier (default ``"slog3"``).

    Returns:
        List of paths to the sRGB output frames.
    """
    frames_dir = Path(frames_dir)
    output_srgb_dir = Path(output_srgb_dir)
    output_srgb_dir.mkdir(parents=True, exist_ok=True)

    if output_linear_dir is not None:
        output_linear_dir = Path(output_linear_dir)
        output_linear_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    frame_files = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    if not frame_files:
        logger.warning("No image files found in %s", frames_dir)
        return []

    logger.info(
        "Batch color correction: %d frames, log_type=%s", len(frame_files), log_type
    )

    srgb_paths: list[Path] = []

    for i, frame_path in enumerate(frame_files):
        img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Skipping unreadable file: %s", frame_path)
            continue

        # Normalize to float [0, 1]
        if img.dtype == np.uint16:
            img_float = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img_float = img.astype(np.float32) / 255.0
        else:
            img_float = img.astype(np.float32)

        # LOG -> linear
        if log_type == "slog3":
            linear = _slog3_to_linear(img_float)
        else:
            raise ValueError(f"Unsupported log_type: {log_type}")

        # Save linear intermediate if requested
        if output_linear_dir is not None:
            linear_max = linear.max()
            if linear_max > 0:
                linear_norm = linear / linear_max
            else:
                linear_norm = linear
            out_16 = (np.clip(linear_norm, 0.0, 1.0) * 65535.0).astype(np.uint16)
            linear_path = output_linear_dir / frame_path.with_suffix(".png").name
            cv2.imwrite(str(linear_path), out_16)

        # Normalize linear for sRGB conversion
        linear_max = linear.max()
        if linear_max > 0:
            linear_for_srgb = np.clip(linear / linear_max, 0.0, 1.0).astype(
                np.float32
            )
        else:
            linear_for_srgb = linear

        srgb = linear_to_srgb(linear_for_srgb)
        srgb_path = output_srgb_dir / frame_path.with_suffix(".png").name
        cv2.imwrite(str(srgb_path), srgb)
        srgb_paths.append(srgb_path)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d / %d frames", i + 1, len(frame_files))

    logger.info("Batch color correction complete: %d sRGB frames", len(srgb_paths))
    return srgb_paths
