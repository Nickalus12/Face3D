"""LOG-to-linear color correction for Samsung Galaxy S25 Ultra Pro Video.

Converts S-Log3 (and similar LOG curves) to linear light, then optionally
to sRGB for COLMAP feature matching. All operations use float32 precision
to preserve dynamic range.

Optimised for speed: vectorised S-Log3 EOTF using precomputed LUT for
8-bit inputs, in-place operations where possible, and LOG profile
detection to skip correction entirely for non-LOG footage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from utils.numba_kernels import HAS_NUMBA
from utils.timing import timed

logger = logging.getLogger(__name__)

# Precomputed 8-bit LUT for S-Log3 -> linear mapping (256 entries).
# Each entry maps a uint8 value (0-255) to its linear-light float32 value.
# Built once at import time to avoid per-frame recomputation.
_SLOG3_LUT_8BIT: np.ndarray | None = None
# Cached combined S-Log3 -> sRGB uint8 LUT (avoids rebuilding per frame)
_SLOG3_TO_SRGB_LUT_U8: np.ndarray | None = None


def _build_slog3_lut_8bit() -> np.ndarray:
    """Build a uint8 -> float32 LUT for the inverse S-Log3 transfer."""
    x = np.arange(256, dtype=np.float64) / 255.0
    cut = 171.2102946929 / 1023.0

    out = np.empty(256, dtype=np.float32)
    linear_mask = x < cut
    out[linear_mask] = ((x[linear_mask] * 1023.0 - 95.0) /
                        (171.2102946929 - 95.0) * 0.01125000).astype(np.float32)
    curve_mask = ~linear_mask
    out[curve_mask] = (
        10.0 ** ((x[curve_mask] * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
    ).astype(np.float32)
    out = np.clip(out, 0.0, None)
    return out


def _get_slog3_lut_8bit() -> np.ndarray:
    """Return the cached 8-bit S-Log3 LUT, building it on first call."""
    global _SLOG3_LUT_8BIT
    if _SLOG3_LUT_8BIT is None:
        _SLOG3_LUT_8BIT = _build_slog3_lut_8bit()
    return _SLOG3_LUT_8BIT


def _slog3_to_linear(x: np.ndarray) -> np.ndarray:
    """Apply the inverse S-Log3 transfer function.

    S-Log3 is defined piecewise. Values are expected in [0, 1] range.
    Reference: Sony S-Log3 white paper.

    Uses Numba JIT when available for ~2-5x speedup on large images.
    Falls back to vectorised numpy otherwise.
    """
    x = np.clip(x, 0.0, 1.0, out=x if x.flags.writeable else None).astype(np.float32)

    if HAS_NUMBA:
        from utils.numba_kernels import slog3_to_linear
        return slog3_to_linear(x)

    # Numpy fallback — vectorised piecewise
    cut = np.float32(171.2102946929 / 1023.0)  # ~0.16736
    x_scaled = x * np.float32(1023.0)
    linear_val = (x_scaled - np.float32(95.0)) / np.float32(171.2102946929 - 95.0) * np.float32(0.01125)
    curve_val = np.float32(10.0) ** ((x_scaled - np.float32(420.0)) / np.float32(261.5)) * np.float32(0.19) - np.float32(0.01)

    out = np.where(x < cut, linear_val, curve_val)
    return np.clip(out, 0.0, None).astype(np.float32)


def _linear_to_srgb_curve(linear: np.ndarray) -> np.ndarray:
    """Apply the sRGB OETF (linear -> sRGB gamma).

    Uses Numba JIT when available for parallel per-pixel computation.
    """
    linear = np.clip(linear, 0.0, 1.0).astype(np.float32)

    if HAS_NUMBA:
        from utils.numba_kernels import linear_to_srgb
        return linear_to_srgb(linear)

    # Numpy fallback
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


def _get_slog3_to_srgb_lut_u8() -> np.ndarray:
    """Return the cached combined S-Log3 -> sRGB uint8 LUT, building on first call.

    This avoids recomputing the sRGB OETF on the 256-entry LUT every frame,
    which was previously done inside _slog3_to_srgb_lut_8bit().
    """
    global _SLOG3_TO_SRGB_LUT_U8
    if _SLOG3_TO_SRGB_LUT_U8 is None:
        lut = _get_slog3_lut_8bit()  # uint8 -> float32 linear
        srgb_lut = _linear_to_srgb_curve(lut / max(lut.max(), 1e-8))
        _SLOG3_TO_SRGB_LUT_U8 = np.clip(srgb_lut * 255.0, 0, 255).astype(np.uint8)
        logger.debug("Built combined S-Log3->sRGB LUT (cached for all frames)")
    return _SLOG3_TO_SRGB_LUT_U8


def _slog3_to_srgb_lut_8bit(img_uint8: np.ndarray) -> np.ndarray:
    """Fast S-Log3 -> sRGB conversion for 8-bit images via precomputed LUT.

    Applies the full S-Log3 EOTF + sRGB OETF in a single LUT lookup per
    channel, avoiding per-pixel float arithmetic entirely.

    The combined LUT is cached at module level so it is built once and
    reused across all frames in a batch.

    Args:
        img_uint8: HxWxC uint8 image in S-Log3 encoding.

    Returns:
        HxWxC uint8 image in sRGB gamma space.
    """
    srgb_lut_u8 = _get_slog3_to_srgb_lut_u8()

    # cv2.LUT applies per-channel LUT lookup — extremely fast
    return cv2.LUT(img_uint8, srgb_lut_u8)


@timed
def batch_color_correct(
    frames_dir: str | Path,
    output_srgb_dir: str | Path,
    output_linear_dir: str | Path | None = None,
    log_type: str = "slog3",
    is_log: bool = True,
    max_workers: int | None = None,
) -> list[Path]:
    """Process all frames in a directory: LOG -> linear -> sRGB.

    For each PNG/JPG frame in *frames_dir*, converts from LOG to linear
    light, then applies sRGB gamma. Optionally saves the intermediate
    linear frames as 16-bit PNGs.

    When *is_log* is ``False``, frames are simply copied to the output
    directory without any colour transform (the video was not recorded
    with a LOG profile).  Use
    :func:`~src.capture.frame_extractor.detect_log_profile` to determine
    this automatically.

    For 8-bit uint8 inputs with ``log_type="slog3"``, a precomputed LUT
    is used for the entire S-Log3 -> sRGB pipeline, giving ~5-10x speedup
    over per-pixel float arithmetic.

    Args:
        frames_dir: Directory containing LOG-encoded input frames.
        output_srgb_dir: Directory for sRGB output frames (8-bit PNG).
        output_linear_dir: Optional directory for linear output frames
            (16-bit PNG). If ``None``, linear intermediates are not saved
            to disk.
        log_type: LOG curve identifier (default ``"slog3"``).
        is_log: Whether the input frames are actually LOG-encoded.
            When ``False``, frames are copied directly.
        max_workers: Number of parallel worker processes for color
            correction. Defaults to ``cpu_count - 1``.

    Returns:
        List of paths to the sRGB output frames.
    """
    import shutil

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

    # --- Fast path: not LOG, just copy frames ---
    if not is_log:
        logger.info(
            "Batch color correction: SKIPPED (not LOG) — copying %d frames as-is",
            len(frame_files),
        )
        srgb_paths: list[Path] = []
        for frame_path in frame_files:
            dst = output_srgb_dir / frame_path.with_suffix(".png").name
            if frame_path.suffix.lower() == ".png":
                shutil.copy2(frame_path, dst)
            else:
                # Convert non-PNG to PNG
                img = cv2.imread(str(frame_path))
                if img is not None:
                    cv2.imwrite(str(dst), img)
                else:
                    continue
            srgb_paths.append(dst)
        logger.info("Copied %d frames to %s", len(srgb_paths), output_srgb_dir)
        return srgb_paths

    logger.info(
        "Batch color correction: %d frames, log_type=%s", len(frame_files), log_type
    )

    # Build per-frame work items for parallel processing
    from utils.parallel import parallel_map

    def _process_single_frame(frame_path: Path) -> Path | None:
        """Process a single frame (LOG -> sRGB). Runs in a worker process."""
        srgb_path = output_srgb_dir / frame_path.with_suffix(".png").name

        # Skip if output already exists (supports resumable processing)
        if srgb_path.exists():
            return srgb_path

        img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        # --- Fast 8-bit LUT path for slog3 ---
        if log_type == "slog3" and img.dtype == np.uint8 and output_linear_dir is None:
            srgb = _slog3_to_srgb_lut_8bit(img)
            cv2.imwrite(str(srgb_path), srgb)
            return srgb_path

        # --- General float path (16-bit inputs or when linear output needed) ---
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
        cv2.imwrite(str(srgb_path), srgb)
        return srgb_path

    results = parallel_map(
        _process_single_frame,
        frame_files,
        max_workers=max_workers,
        desc="Color correction",
        use_threads=False,
    )

    srgb_paths = [p for p in results if p is not None]

    logger.info("Batch color correction complete: %d sRGB frames", len(srgb_paths))
    return srgb_paths
