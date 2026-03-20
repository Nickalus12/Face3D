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


def _gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """Apply Gray World assumption white balance correction.

    Adjusts each channel so the average color is neutral gray.
    Works well for scenes with diverse colors (outdoor, well-lit indoor).
    """
    img_float = img.astype(np.float32)
    avg_b = img_float[:, :, 0].mean()
    avg_g = img_float[:, :, 1].mean()
    avg_r = img_float[:, :, 2].mean()
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    if avg_b > 0 and avg_g > 0 and avg_r > 0:
        img_float[:, :, 0] *= avg_all / avg_b
        img_float[:, :, 1] *= avg_all / avg_g
        img_float[:, :, 2] *= avg_all / avg_r

    return np.clip(img_float, 0, 255).astype(np.uint8)


def _white_patch_white_balance(
    img: np.ndarray,
    percentile: float = 95.0,
) -> np.ndarray:
    """Apply White Patch (Max-RGB) white balance correction.

    Assumes the brightest pixels in each channel represent white.
    Uses a percentile (default 95th) instead of the absolute max to
    avoid being thrown off by specular highlights or sensor noise.

    Works better than Gray World for scenes dominated by a single color
    (e.g. skin in face close-ups).
    """
    img_float = img.astype(np.float32)
    max_b = np.percentile(img_float[:, :, 0], percentile)
    max_g = np.percentile(img_float[:, :, 1], percentile)
    max_r = np.percentile(img_float[:, :, 2], percentile)
    max_val = max(max_b, max_g, max_r, 1.0)

    if max_b > 0:
        img_float[:, :, 0] *= max_val / max_b
    if max_g > 0:
        img_float[:, :, 1] *= max_val / max_g
    if max_r > 0:
        img_float[:, :, 2] *= max_val / max_r

    return np.clip(img_float, 0, 255).astype(np.uint8)


def _pick_best_white_balance(img: np.ndarray) -> np.ndarray:
    """Apply both Gray World and White Patch WB, return the more neutral result.

    For 3D reconstruction, we want minimal color cast so that feature
    descriptors and photometric loss see consistent colors. We score each
    result by how close the mean A and B channels in LAB are to neutral
    (128). The result closer to neutral wins.
    """
    gw = _gray_world_white_balance(img)
    wp = _white_patch_white_balance(img)

    def _neutrality_score(frame: np.ndarray) -> float:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # Distance of mean A,B from neutral (128)
        mean_a = lab[:, :, 1].mean()
        mean_b = lab[:, :, 2].mean()
        return abs(mean_a - 128.0) + abs(mean_b - 128.0)

    score_gw = _neutrality_score(gw)
    score_wp = _neutrality_score(wp)

    return gw if score_gw <= score_wp else wp


def _adaptive_clahe_clip_limit(
    l_channel: np.ndarray,
    base_clip: float = 2.5,
    min_clip: float = 1.0,
    max_clip: float = 5.0,
) -> float:
    """Compute an adaptive CLAHE clip limit based on the image histogram.

    Dark images (low mean L) get a higher clip limit because they need
    more local contrast boost. Bright, well-exposed images get a lower
    clip limit to avoid over-enhancement and haloing.

    The clip limit scales linearly between min_clip and max_clip based
    on how far the mean luminance is from the ideal midpoint (128).
    """
    mean_l = float(l_channel.mean())
    std_l = float(l_channel.std())

    # Darker images need more contrast boost
    # Mean L in [0, 255]; ideal is ~128
    darkness_factor = max(0.0, (128.0 - mean_l) / 128.0)  # 0..1, higher = darker

    # Low-contrast images also benefit from more CLAHE
    # std_l for a well-exposed image is typically 40-60
    contrast_factor = max(0.0, (50.0 - std_l) / 50.0)  # 0..1, higher = flatter

    # Combine: weight darkness more (0.7) vs contrast (0.3)
    boost = 0.7 * darkness_factor + 0.3 * contrast_factor
    clip = base_clip + boost * (max_clip - base_clip)
    return float(np.clip(clip, min_clip, max_clip))


def _gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction via a precomputed uint8 LUT.

    gamma < 1.0 brightens (useful for dark footage): output = input^gamma,
    so dark pixel 60/255 -> (0.235)^0.6 = 0.416 -> brighter.
    gamma > 1.0 darkens.
    """
    table = np.array(
        [(i / 255.0) ** gamma * 255.0 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, table)


def _unsharp_mask(
    img: np.ndarray,
    sigma: float = 1.0,
    strength: float = 0.5,
) -> np.ndarray:
    """Apply mild unsharp mask to improve feature detection quality.

    Uses a Gaussian blur subtraction approach. The strength is kept low
    (0.3-0.7) to sharpen edges for SIFT/feature matching without
    amplifying noise or creating ringing artifacts.

    Args:
        img: BGR uint8 image.
        sigma: Gaussian blur sigma (controls radius of sharpening).
        strength: Sharpening amount. 0 = no effect, 1 = strong.

    Returns:
        Sharpened BGR uint8 image.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    # sharpened = original * (1 + strength) - blurred * strength
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def auto_enhance_frames(
    frames_dir: str | Path,
    output_dir: str | Path,
    target_brightness: float = 128.0,
    apply_white_balance: bool = True,
    apply_clahe: bool = True,
    apply_denoise: bool = True,
    apply_gamma: bool = True,
    apply_sharpen: bool = True,
    clahe_clip_limit: float = 2.5,
    clahe_grid_size: int = 8,
    sharpen_sigma: float = 1.0,
    sharpen_strength: float = 0.4,
) -> list[Path]:
    """Comprehensive auto-enhancement for 3D reconstruction preprocessing.

    Applies up to six correction passes (all optional, adaptive):

    1. White balance -- picks the better of Gray World and White Patch
       per-frame to minimize color cast regardless of scene content.
    2. Gamma correction -- for consistently dark footage, applies a
       proper gamma curve before CLAHE to lift shadows without clipping.
    3. Adaptive CLAHE -- clip limit auto-adjusts based on per-frame
       histogram: dark/flat images get more contrast boost, bright
       images get less. Operates on L-channel in LAB to preserve color.
    4. Brightness normalization -- global gain to hit target brightness.
    5. Mild denoising -- fastNlMeans only when footage is dark (noise
       is amplified by the above corrections).
    6. Unsharp mask -- mild edge sharpening to improve SIFT/feature
       detection quality for COLMAP and DA3.

    All enhancements are tuned to improve feature detection (more
    COLMAP keypoints, better DA3 matching) and 2DGS training quality
    (consistent color/exposure reduces photometric loss noise).

    Args:
        frames_dir: Input frames directory.
        output_dir: Output directory for enhanced frames.
        target_brightness: Target mean L-channel brightness (0-255).
        apply_white_balance: Pick best of Gray World / White Patch WB.
        apply_clahe: Apply adaptive CLAHE local contrast enhancement.
        apply_denoise: Apply mild denoising for dark/noisy frames.
        apply_gamma: Apply gamma correction for dark footage.
        apply_sharpen: Apply mild unsharp mask for feature detection.
        clahe_clip_limit: Base CLAHE contrast limit (auto-adjusted per frame).
        clahe_grid_size: CLAHE tile grid size (8 is standard).
        sharpen_sigma: Unsharp mask Gaussian sigma.
        sharpen_strength: Unsharp mask strength (0-1, 0.3-0.5 recommended).

    Returns:
        List of enhanced frame paths.
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(
        list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
    )
    if not frame_files:
        return []

    # --- Analyze overall exposure and color from sample frames ---
    sample_brightnesses: list[float] = []
    sample_a_means: list[float] = []
    sample_b_means: list[float] = []
    sample_count = min(len(frame_files), 20)
    for fp in frame_files[:sample_count]:
        img = cv2.imread(str(fp))
        if img is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            sample_brightnesses.append(float(lab[:, :, 0].mean()))
            sample_a_means.append(float(lab[:, :, 1].mean()))
            sample_b_means.append(float(lab[:, :, 2].mean()))

    if not sample_brightnesses:
        return []

    mean_l = float(np.mean(sample_brightnesses))
    std_l_across = float(np.std(sample_brightnesses))
    logger.info(
        "Auto-enhance: %d frames, mean L=%.0f (std=%.1f), target=%.0f",
        len(frame_files), mean_l, std_l_across, target_brightness,
    )

    # --- Determine gamma correction need ---
    # Only apply gamma for consistently dark footage (mean L < 80).
    # Gamma < 1 brightens. Scale: L=40 -> gamma=0.55, L=80 -> gamma=1.0
    gamma_value = 1.0
    if apply_gamma and mean_l < 80.0:
        gamma_value = max(0.45, mean_l / 80.0)
        logger.info(
            "Gamma correction enabled: gamma=%.2f (mean L=%.0f is dark)",
            gamma_value, mean_l,
        )

    # --- Global brightness gain ---
    # Computed after gamma (gamma brightens first, gain handles the rest)
    effective_mean = mean_l
    if gamma_value < 1.0:
        # Estimate post-gamma brightness: L_new ~ L^gamma * 255
        effective_mean = ((mean_l / 255.0) ** gamma_value) * 255.0
    global_gain = min(target_brightness / max(effective_mean, 1.0), 2.5)
    needs_brightness = global_gain > 1.05 or global_gain < 0.95

    results: list[Path] = []
    from tqdm import tqdm

    for fp in tqdm(frame_files, desc="Enhancing"):
        dst = output_dir / fp.name
        if dst.exists():
            results.append(dst)
            continue

        img = cv2.imread(str(fp))
        if img is None:
            continue

        # Step 1: White balance (best of Gray World vs White Patch)
        if apply_white_balance:
            img = _pick_best_white_balance(img)

        # Step 2: Gamma correction for dark footage
        if apply_gamma and gamma_value < 1.0:
            img = _gamma_correction(img, gamma_value)

        # Step 3: Adaptive CLAHE + brightness in LAB space
        if apply_clahe or needs_brightness:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            if apply_clahe:
                # Adaptive clip limit per frame
                clip = _adaptive_clahe_clip_limit(
                    lab[:, :, 0], base_clip=clahe_clip_limit,
                )
                clahe_obj = cv2.createCLAHE(
                    clipLimit=clip,
                    tileGridSize=(clahe_grid_size, clahe_grid_size),
                )
                lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])

            if needs_brightness:
                l_float = lab[:, :, 0].astype(np.float32) * global_gain
                lab[:, :, 0] = np.clip(l_float, 0, 255).astype(np.uint8)

            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Step 4: Mild denoising (only for dark footage where noise is amplified)
        if apply_denoise and mean_l < 90:
            img = cv2.fastNlMeansDenoisingColored(
                img, None, h=5, hColor=5,
                templateWindowSize=7, searchWindowSize=21,
            )

        # Step 5: Mild unsharp mask for feature detection
        if apply_sharpen:
            img = _unsharp_mask(img, sigma=sharpen_sigma, strength=sharpen_strength)

        cv2.imwrite(str(dst), img)
        results.append(dst)

    # --- Report improvement ---
    if results:
        sample_final: list[float] = []
        for fp in results[:sample_count]:
            final_img = cv2.imread(str(fp))
            if final_img is not None:
                final_lab = cv2.cvtColor(final_img, cv2.COLOR_BGR2LAB)
                sample_final.append(float(final_lab[:, :, 0].mean()))
        if sample_final:
            final_mean = float(np.mean(sample_final))
            final_std = float(np.std(sample_final))
            logger.info(
                "Auto-enhance complete: %d frames, L-channel %.0f -> %.0f "
                "(std %.1f -> %.1f)",
                len(results), mean_l, final_mean, std_l_across, final_std,
            )

    return results


def normalize_color_across_frames(
    frames_dir: str | Path,
    output_dir: str | Path | None = None,
    reference_method: str = "median",
) -> list[Path]:
    """Normalize color temperature and brightness across all frames.

    For 3D Gaussian Splatting training, color consistency across views
    is critical. Inconsistent white balance or exposure between frames
    causes the photometric loss to fight against itself, producing
    floaters and color artifacts.

    This function computes a reference color profile (mean or median of
    all frames' LAB channel statistics) and shifts each frame to match.
    Only the chrominance (A, B channels) and brightness (L channel) are
    adjusted -- spatial detail is untouched.

    Args:
        frames_dir: Directory containing frames to normalize.
        output_dir: Output directory. If None, frames are modified in-place.
        reference_method: How to compute the reference profile.
            ``"median"`` (default) is robust to outlier frames.
            ``"mean"`` uses the arithmetic mean.

    Returns:
        List of normalized frame paths.
    """
    frames_dir = Path(frames_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = frames_dir  # In-place

    frame_files = sorted(
        list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
    )
    if not frame_files:
        return []

    # --- Pass 1: Collect per-frame LAB statistics ---
    logger.info("Color normalization pass 1: analyzing %d frames...", len(frame_files))
    stats_l: list[float] = []
    stats_a: list[float] = []
    stats_b: list[float] = []

    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is None:
            stats_l.append(128.0)
            stats_a.append(128.0)
            stats_b.append(128.0)
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        stats_l.append(float(lab[:, :, 0].mean()))
        stats_a.append(float(lab[:, :, 1].mean()))
        stats_b.append(float(lab[:, :, 2].mean()))

    # Compute reference
    agg_fn = np.median if reference_method == "median" else np.mean
    ref_l = float(agg_fn(stats_l))
    ref_a = float(agg_fn(stats_a))
    ref_b = float(agg_fn(stats_b))

    l_spread = float(np.std(stats_l))
    a_spread = float(np.std(stats_a))
    b_spread = float(np.std(stats_b))

    logger.info(
        "Color normalization reference: L=%.1f A=%.1f B=%.1f "
        "(spread: L=%.1f A=%.1f B=%.1f)",
        ref_l, ref_a, ref_b, l_spread, a_spread, b_spread,
    )

    # Skip if already consistent (all spreads < 3)
    if l_spread < 3.0 and a_spread < 2.0 and b_spread < 2.0:
        logger.info(
            "Frames already color-consistent (spread L=%.1f, A=%.1f, B=%.1f). "
            "Skipping normalization.",
            l_spread, a_spread, b_spread,
        )
        return list(frame_files)

    # --- Pass 2: Shift each frame toward the reference ---
    logger.info("Color normalization pass 2: adjusting %d frames...", len(frame_files))
    from tqdm import tqdm

    results: list[Path] = []
    for i, fp in enumerate(tqdm(frame_files, desc="Normalizing color")):
        dst = output_dir / fp.name

        img = cv2.imread(str(fp))
        if img is None:
            results.append(dst)
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute per-frame shift needed
        shift_l = ref_l - stats_l[i]
        shift_a = ref_a - stats_a[i]
        shift_b = ref_b - stats_b[i]

        # Apply gentle blending: 70% shift (avoid over-correction that
        # destroys legitimate per-view lighting variation needed for
        # multi-view stereo).
        blend = 0.7
        lab[:, :, 0] += shift_l * blend
        lab[:, :, 1] += shift_a * blend
        lab[:, :, 2] += shift_b * blend

        lab = np.clip(lab, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        cv2.imwrite(str(dst), img)
        results.append(dst)

    # Report final consistency
    final_l_vals: list[float] = []
    for fp in results[:20]:
        check = cv2.imread(str(fp))
        if check is not None:
            check_lab = cv2.cvtColor(check, cv2.COLOR_BGR2LAB)
            final_l_vals.append(float(check_lab[:, :, 0].mean()))
    if final_l_vals:
        logger.info(
            "Color normalization complete: %d frames, L spread %.1f -> %.1f",
            len(results), l_spread, float(np.std(final_l_vals)),
        )

    return results
