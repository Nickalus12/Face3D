"""Expert RAW photo processing for high-quality texture views.

Processes Samsung Expert RAW photos (DNG/JPG/PNG) and integrates them into
the Face3D reconstruction pipeline as additional high-weight training views.

Supports:
- DNG files via rawpy with DHT demosaicing and camera white balance
- JPG/PNG files with EXIF-based auto-rotation and color space normalization
- Lens matching against S25 Ultra video metadata
- COLMAP image_registrator for adding photos to existing reconstruction
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Samsung S25 Ultra lens focal lengths (35mm equiv)
_S25_ULTRA_LENSES = {
    13: "ultrawide",
    23: "wide",
    70: "telephoto",
    200: "supertelephoto",
}

# Physical focal lengths (mm) corresponding to 35mm equivalents
_S25_ULTRA_PHYSICAL_FOCAL = {
    13: 1.95,   # ultrawide
    23: 6.3,    # wide (main)
    70: 6.3,    # telephoto (periscope)
    200: 6.3,   # supertelephoto (periscope)
}

_KNOWN_FOCAL_LENGTHS_35MM = sorted(_S25_ULTRA_LENSES.keys())


def _snap_focal_length_35mm(focal_mm: float) -> int:
    """Snap a 35mm-equivalent focal length to the nearest known S25 Ultra lens."""
    return min(_KNOWN_FOCAL_LENGTHS_35MM, key=lambda k: abs(k - focal_mm))


def _extract_exif_pillow(photo_path: Path) -> dict:
    """Extract EXIF metadata from a photo using Pillow.

    Returns a dict with keys: focal_length, focal_length_35mm, exposure_time,
    iso, make, model, orientation, width, height.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(photo_path)
        exif_raw = img._getexif() or {}

        # Build tag name -> value mapping
        exif = {}
        for tag_id, value in exif_raw.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            exif[tag_name] = value

        metadata = {
            "width": img.width,
            "height": img.height,
            "focal_length": None,
            "focal_length_35mm": None,
            "exposure_time": None,
            "iso": None,
            "make": exif.get("Make", ""),
            "model": exif.get("Model", ""),
            "orientation": exif.get("Orientation", 1),
        }

        # Focal length (raw, in mm)
        fl = exif.get("FocalLength")
        if fl is not None:
            if hasattr(fl, "numerator"):
                metadata["focal_length"] = float(fl.numerator) / float(fl.denominator)
            else:
                metadata["focal_length"] = float(fl)

        # Focal length in 35mm equivalent
        fl35 = exif.get("FocalLengthIn35mmFilm")
        if fl35 is not None:
            metadata["focal_length_35mm"] = float(fl35)

        # Exposure time
        et = exif.get("ExposureTime")
        if et is not None:
            if hasattr(et, "numerator"):
                metadata["exposure_time"] = float(et.numerator) / float(et.denominator)
            else:
                metadata["exposure_time"] = float(et)

        # ISO
        iso = exif.get("ISOSpeedRatings")
        if iso is not None:
            if isinstance(iso, (list, tuple)):
                metadata["iso"] = int(iso[0])
            else:
                metadata["iso"] = int(iso)

        img.close()
        return metadata

    except Exception as e:
        logger.warning("Failed to extract EXIF from %s: %s", photo_path, e)
        return {
            "width": 0, "height": 0,
            "focal_length": None, "focal_length_35mm": None,
            "exposure_time": None, "iso": None,
            "make": "", "model": "",
            "orientation": 1,
        }


def _auto_rotate(image: np.ndarray, orientation: int) -> np.ndarray:
    """Auto-rotate an image based on EXIF orientation tag.

    Args:
        image: HxWxC numpy array.
        orientation: EXIF orientation value (1-8).

    Returns:
        Correctly oriented image.
    """
    if orientation == 1:
        return image
    elif orientation == 2:
        return np.fliplr(image)
    elif orientation == 3:
        return np.rot90(image, 2)
    elif orientation == 4:
        return np.flipud(image)
    elif orientation == 5:
        return np.rot90(np.fliplr(image), 1)
    elif orientation == 6:
        return np.rot90(image, 3)  # 90 CW
    elif orientation == 7:
        return np.rot90(np.fliplr(image), 3)
    elif orientation == 8:
        return np.rot90(image, 1)  # 90 CCW
    return image


def _process_dng(photo_path: Path) -> tuple[np.ndarray, int]:
    """Process a DNG RAW file with rawpy.

    Uses DHT demosaicing for highest quality, camera white balance,
    and linear output space.

    Returns:
        Tuple of (image as uint16 numpy array in RGB, bit depth 16).
    """
    import rawpy

    with rawpy.imread(str(photo_path)) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            no_auto_bright=True,
            gamma=(1, 1),  # Linear output
        )

    # Apply sRGB gamma curve for consistency with video frames
    rgb_float = rgb.astype(np.float32) / 65535.0
    # Linear to sRGB
    srgb = np.where(
        rgb_float <= 0.0031308,
        12.92 * rgb_float,
        1.055 * np.power(np.clip(rgb_float, 0.0031308, None), 1.0 / 2.4) - 0.055,
    )
    srgb = np.clip(srgb, 0.0, 1.0)
    # Output as 16-bit
    output = (srgb * 65535.0).astype(np.uint16)

    return output, 16


def process_photos(
    photo_paths: list[str | Path],
    output_dir: str | Path,
    target_colorspace: str = "srgb",
) -> tuple[list[Path], list[dict]]:
    """Process Expert RAW photos for integration into the Face3D pipeline.

    Accepts JPG, DNG, and PNG photos. DNG files are processed with rawpy
    for maximum quality. All outputs are saved as PNG with consistent color
    space.

    Args:
        photo_paths: List of paths to input photos.
        output_dir: Directory for processed output images.
        target_colorspace: Target color space ("srgb").

    Returns:
        Tuple of (list of processed photo paths, list of per-photo metadata dicts).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_paths: list[Path] = []
    all_metadata: list[dict] = []

    for i, photo_path in enumerate(photo_paths):
        photo_path = Path(photo_path)
        if not photo_path.is_file():
            logger.warning("Photo not found, skipping: %s", photo_path)
            continue

        suffix = photo_path.suffix.lower()
        logger.info("Processing photo %d/%d: %s", i + 1, len(photo_paths), photo_path.name)

        # Extract EXIF before processing
        exif = _extract_exif_pillow(photo_path)

        try:
            if suffix == ".dng":
                # RAW processing with rawpy
                image, bps = _process_dng(photo_path)
                out_name = f"photo_{i:04d}.png"
            elif suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                # Read with OpenCV
                if suffix in (".tif", ".tiff"):
                    image = cv2.imread(str(photo_path), cv2.IMREAD_UNCHANGED)
                else:
                    image = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)

                if image is None:
                    logger.warning("Failed to read photo: %s", photo_path)
                    continue

                # Convert BGR to RGB for auto-rotation, then back
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bps = 16 if image.dtype == np.uint16 else 8
                out_name = f"photo_{i:04d}.png"
            else:
                logger.warning("Unsupported photo format: %s", suffix)
                continue

            # Auto-rotate based on EXIF orientation
            orientation = exif.get("orientation", 1)
            if orientation and orientation != 1:
                image = _auto_rotate(image, orientation)
                logger.debug("Auto-rotated photo %s (orientation=%d)", photo_path.name, orientation)
                # Update dimensions after rotation
                exif["width"] = image.shape[1]
                exif["height"] = image.shape[0]

            # Convert RGB back to BGR for cv2.imwrite
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Save as PNG
            out_path = output_dir / out_name
            if bps == 16:
                cv2.imwrite(str(out_path), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                cv2.imwrite(str(out_path), image_bgr)

            processed_paths.append(out_path)

            # Build metadata entry
            meta = {
                "source_file": str(photo_path),
                "output_file": str(out_path),
                "output_name": out_name,
                "bit_depth": bps,
                "is_raw": suffix == ".dng",
                "is_photo": True,
                **exif,
            }
            all_metadata.append(meta)

            logger.info(
                "  -> %s (%dx%d, %dbit, focal=%s, focal35=%s, iso=%s)",
                out_name, exif.get("width", 0), exif.get("height", 0),
                bps, exif.get("focal_length"), exif.get("focal_length_35mm"),
                exif.get("iso"),
            )

        except Exception as e:
            logger.error("Failed to process photo %s: %s", photo_path, e)
            continue

    # Save metadata JSON
    meta_path = output_dir / "photo_metadata.json"
    meta_path.write_text(json.dumps(all_metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Processed %d photos, metadata saved to %s", len(processed_paths), meta_path)

    return processed_paths, all_metadata


def match_photo_lens_to_video(
    photo_exif: dict,
    video_lens_metadata: dict | None = None,
) -> tuple[str, bool]:
    """Compare photo and video focal lengths to determine lens match.

    Args:
        photo_exif: EXIF metadata dict from a processed photo.
        video_lens_metadata: Optional dict with video lens info (e.g. from
            lens_metadata.json produced by frame_extractor).

    Returns:
        Tuple of (detected_lens_name, is_same_lens_as_video).
    """
    # Determine photo lens from 35mm focal length
    photo_focal_35 = photo_exif.get("focal_length_35mm")
    if photo_focal_35 is None:
        photo_focal_raw = photo_exif.get("focal_length")
        if photo_focal_raw is not None:
            # Rough crop factor for S25 Ultra main sensor (~6.7x)
            photo_focal_35 = photo_focal_raw * 6.7
        else:
            logger.warning("No focal length in photo EXIF, cannot match lens")
            return "unknown", False

    snapped = _snap_focal_length_35mm(photo_focal_35)
    photo_lens = _S25_ULTRA_LENSES.get(snapped, "unknown")

    # Compare to video lens
    if video_lens_metadata:
        video_lens = video_lens_metadata.get("detected_lens", "unknown")
        is_same = photo_lens == video_lens
        if not is_same:
            logger.warning(
                "Photo lens (%s, %.0fmm) differs from video lens (%s). "
                "This requires a separate camera model in COLMAP.",
                photo_lens, photo_focal_35, video_lens,
            )
        return photo_lens, is_same

    return photo_lens, True  # Assume same if no video metadata


def register_photos_into_reconstruction(
    photo_dir: str | Path,
    colmap_model_dir: str | Path,
    colmap_binary: str | Path,
    database_path: str | Path | None = None,
) -> Path:
    """Register Expert RAW photos into an existing COLMAP reconstruction.

    Uses COLMAP's image_registrator to add new images to the existing model
    after extracting features and matching against the existing database.

    Args:
        photo_dir: Directory containing processed photo images.
        colmap_model_dir: Path to existing COLMAP sparse model (cameras.bin, etc.).
        colmap_binary: Path to COLMAP binary/bat file.
        database_path: Path to COLMAP database. If None, uses colmap_model_dir/../database.db.

    Returns:
        Path to the updated model directory.
    """
    photo_dir = Path(photo_dir)
    colmap_model_dir = Path(colmap_model_dir)
    colmap_binary = Path(colmap_binary)

    if database_path is None:
        database_path = colmap_model_dir.parent.parent.parent / "database.db"
    database_path = Path(database_path)

    # Output model directory
    updated_model_dir = colmap_model_dir.parent / "0_with_photos"
    updated_model_dir.mkdir(parents=True, exist_ok=True)

    colmap_bin = str(colmap_binary)

    logger.info("Registering photos from %s into COLMAP model at %s", photo_dir, colmap_model_dir)

    # Step 1: Extract features for photos
    logger.info("Extracting SIFT features for photos...")
    cmd_extract = [
        colmap_bin, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(photo_dir),
        "--ImageReader.single_camera", "0",  # Photos may have different intrinsics
        "--SiftExtraction.max_num_features", "8192",
    ]
    try:
        subprocess.run(cmd_extract, capture_output=True, text=True, check=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("COLMAP feature extraction failed: %s", e)
        # Copy original model as fallback
        _copy_model(colmap_model_dir, updated_model_dir)
        return updated_model_dir

    # Step 2: Match against existing database
    logger.info("Matching photos against existing reconstruction...")
    cmd_match = [
        colmap_bin, "exhaustive_matcher",
        "--database_path", str(database_path),
    ]
    try:
        subprocess.run(cmd_match, capture_output=True, text=True, check=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("COLMAP matching failed: %s", e)
        _copy_model(colmap_model_dir, updated_model_dir)
        return updated_model_dir

    # Step 3: Register new images into existing model
    logger.info("Running image_registrator...")
    cmd_register = [
        colmap_bin, "image_registrator",
        "--database_path", str(database_path),
        "--input_path", str(colmap_model_dir),
        "--output_path", str(updated_model_dir),
    ]
    try:
        subprocess.run(cmd_register, capture_output=True, text=True, check=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("COLMAP image_registrator failed: %s", e)
        _copy_model(colmap_model_dir, updated_model_dir)
        return updated_model_dir

    # Step 4: Bundle adjustment to refine
    logger.info("Running bundle adjustment...")
    refined_dir = colmap_model_dir.parent / "0_refined"
    refined_dir.mkdir(parents=True, exist_ok=True)

    cmd_ba = [
        colmap_bin, "bundle_adjuster",
        "--input_path", str(updated_model_dir),
        "--output_path", str(refined_dir),
    ]
    try:
        subprocess.run(cmd_ba, capture_output=True, text=True, check=True, timeout=600)
        # Use refined model
        _copy_model(refined_dir, updated_model_dir)
        logger.info("Bundle adjustment complete, model updated at %s", updated_model_dir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("Bundle adjustment failed (using pre-BA model): %s", e)

    return updated_model_dir


def _copy_model(src_dir: Path, dst_dir: Path) -> None:
    """Copy COLMAP model files from src to dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["cameras.bin", "images.bin", "points3D.bin",
                   "cameras.txt", "images.txt", "points3D.txt"]:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)
