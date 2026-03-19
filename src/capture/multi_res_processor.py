"""Multi-resolution photo processor for 200MP and multi-lens captures.

Handles the 16320x12240 (200MP) Expert RAW photos from the Samsung Galaxy
S25 Ultra main sensor.  These images are too large for direct GPU processing
(would exceed 16GB VRAM), so this module creates multiple resolution tiers
and extracts tight face crops at full resolution for texture baking.

Also handles registering multi-lens photos into a COLMAP reconstruction
with per-lens camera models.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default multi-resolution output tiers
_DEFAULT_TIERS = {
    "full": None,           # Original resolution (16320x12240 for 200MP)
    "quarter": 0.25,        # 4080x3060 — for DA3/COLMAP registration
    "display": (1920, 1440),  # Fixed size for preview
}


def _load_dng_rawpy(dng_path: Path) -> np.ndarray:
    """Load a DNG file with rawpy using DHT demosaicing.

    Returns a 16-bit RGB numpy array in linear light.
    """
    import rawpy

    with rawpy.imread(str(dng_path)) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            no_auto_bright=True,
            gamma=(1, 1),  # Linear output
        )

    return rgb


def _linear_to_srgb_16bit(linear: np.ndarray) -> np.ndarray:
    """Convert linear 16-bit RGB to sRGB gamma 16-bit."""
    f = linear.astype(np.float32) / 65535.0
    srgb = np.where(
        f <= 0.0031308,
        12.92 * f,
        1.055 * np.power(np.clip(f, 0.0031308, None), 1.0 / 2.4) - 0.055,
    )
    return (np.clip(srgb, 0.0, 1.0) * 65535.0).astype(np.uint16)


def _resize_image(
    image: np.ndarray,
    target: float | tuple[int, int] | None,
) -> np.ndarray:
    """Resize an image by scale factor or to exact (width, height).

    Args:
        image: HxWxC numpy array.
        target: Scale factor (0.25 = quarter), (w, h) tuple, or None (no-op).

    Returns:
        Resized image.
    """
    if target is None:
        return image

    h, w = image.shape[:2]

    if isinstance(target, (float, int)) and not isinstance(target, tuple):
        new_w = int(w * target)
        new_h = int(h * target)
    else:
        new_w, new_h = target

    # Use INTER_AREA for downscaling (best quality), INTER_LANCZOS4 for upscaling
    if new_w < w or new_h < h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4

    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _detect_face_bbox_mediapipe(image_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect face bounding box using MediaPipe on a (possibly downscaled) image.

    Returns (x, y, w, h) in the input image's coordinate space, or None.
    """
    try:
        import mediapipe as mp

        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.5,
        ) as detector:
            results = detector.process(image_rgb)

        if not results.detections:
            return None

        # Use first (highest confidence) detection
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        ih, iw = image_rgb.shape[:2]

        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)

        return (x, y, w, h)

    except Exception as e:
        logger.warning("MediaPipe face detection failed: %s", e)
        return None


def _extract_face_crop(
    full_image: np.ndarray,
    face_bbox_on_quarter: tuple[int, int, int, int],
    quarter_scale: float,
    padding_factor: float = 1.5,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract a face crop from the full-resolution image.

    The face bounding box is detected on the quarter-res image and then
    scaled up to full-res coordinates with padding.

    Args:
        full_image: Full-resolution HxWxC image.
        face_bbox_on_quarter: (x, y, w, h) from quarter-res detection.
        quarter_scale: Scale factor used for the quarter-res image.
        padding_factor: How much to expand the bbox (1.5 = 50% padding).

    Returns:
        Tuple of (face_crop, crop_info_dict).
    """
    full_h, full_w = full_image.shape[:2]
    scale = 1.0 / quarter_scale

    # Scale bbox to full res
    qx, qy, qw, qh = face_bbox_on_quarter
    fx = int(qx * scale)
    fy = int(qy * scale)
    fw = int(qw * scale)
    fh = int(qh * scale)

    # Add padding
    pad_w = int(fw * (padding_factor - 1.0) / 2.0)
    pad_h = int(fh * (padding_factor - 1.0) / 2.0)

    x1 = max(0, fx - pad_w)
    y1 = max(0, fy - pad_h)
    x2 = min(full_w, fx + fw + pad_w)
    y2 = min(full_h, fy + fh + pad_h)

    crop = full_image[y1:y2, x1:x2]
    crop_info = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "crop_width": x2 - x1, "crop_height": y2 - y1,
        "full_width": full_w, "full_height": full_h,
    }

    return crop, crop_info


def process_200mp_photos(
    photo_paths: list[str | Path],
    output_dir: str | Path,
    target_sizes: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Process 200MP Expert RAW photos into multiple resolution tiers.

    Creates:
    - Full res (16320x12240) as 16-bit TIFF for texture baking only
    - Quarter res (4080x3060) for DA3/COLMAP registration
    - Display res (1920x1440) for quick preview
    - Face crops at full resolution (the texture gold)

    Args:
        photo_paths: List of paths to 200MP DNG or JPG files.
        output_dir: Base output directory.  Sub-directories ``full/``,
            ``quarter/``, ``display/``, ``face_crops/`` are created.
        target_sizes: Override default resolution tiers. Dict mapping
            tier name to scale factor or (w, h) tuple.

    Returns:
        Dict with keys ``full``, ``quarter``, ``display``, ``face_crops``,
        each containing a list of per-photo info dicts with ``path``,
        ``resolution``, etc.
    """
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = None

    output_dir = Path(output_dir)
    tiers = target_sizes or _DEFAULT_TIERS

    # Create output subdirs
    tier_dirs: dict[str, Path] = {}
    for tier_name in tiers:
        d = output_dir / tier_name
        d.mkdir(parents=True, exist_ok=True)
        tier_dirs[tier_name] = d

    face_crops_dir = output_dir / "face_crops"
    face_crops_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict[str, Any]]] = {
        tier_name: [] for tier_name in tiers
    }
    results["face_crops"] = []

    for i, pp in enumerate(photo_paths):
        pp = Path(pp)
        if not pp.exists():
            logger.warning("Photo not found: %s", pp)
            continue

        suffix = pp.suffix.lower()
        stem = pp.stem
        logger.info(
            "Processing 200MP photo %d/%d: %s", i + 1, len(photo_paths), pp.name,
        )

        try:
            # Load full-res image
            if suffix == ".dng":
                image_linear = _load_dng_rawpy(pp)
                image_srgb = _linear_to_srgb_16bit(image_linear)
                is_16bit = True
            elif suffix in (".jpg", ".jpeg"):
                image_srgb = cv2.imread(str(pp), cv2.IMREAD_COLOR)
                if image_srgb is None:
                    logger.warning("Failed to read: %s", pp)
                    continue
                image_srgb = cv2.cvtColor(image_srgb, cv2.COLOR_BGR2RGB)
                is_16bit = False
            elif suffix in (".tif", ".tiff"):
                image_srgb = cv2.imread(str(pp), cv2.IMREAD_UNCHANGED)
                if image_srgb is None:
                    logger.warning("Failed to read: %s", pp)
                    continue
                image_srgb = cv2.cvtColor(image_srgb, cv2.COLOR_BGR2RGB)
                is_16bit = image_srgb.dtype == np.uint16
            else:
                logger.warning("Unsupported format: %s", suffix)
                continue

            full_h, full_w = image_srgb.shape[:2]
            logger.info("  Loaded: %dx%d, %dbit", full_w, full_h, 16 if is_16bit else 8)

            # Generate each resolution tier
            quarter_image = None
            quarter_scale = 0.25

            for tier_name, tier_target in tiers.items():
                if tier_target is None:
                    # Full res -> save as 16-bit TIFF
                    out_path = tier_dirs[tier_name] / f"{stem}.tiff"
                    image_bgr = cv2.cvtColor(image_srgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_path), image_bgr)
                    results[tier_name].append({
                        "path": str(out_path),
                        "source": str(pp),
                        "resolution": [full_w, full_h],
                        "bit_depth": 16 if is_16bit else 8,
                    })
                    logger.info("  -> %s: %s", tier_name, out_path.name)
                else:
                    resized = _resize_image(image_srgb, tier_target)
                    rh, rw = resized.shape[:2]

                    # Track quarter-res for face detection
                    if isinstance(tier_target, float) and tier_target <= 0.25:
                        quarter_image = resized
                        quarter_scale = tier_target

                    # Save as PNG (8-bit for quarter/display)
                    out_path = tier_dirs[tier_name] / f"{stem}.png"
                    if is_16bit and isinstance(tier_target, float) and tier_target >= 0.5:
                        # Keep 16-bit for higher res tiers
                        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(
                            str(out_path), resized_bgr,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1],
                        )
                    else:
                        # Convert to 8-bit for smaller tiers
                        if resized.dtype == np.uint16:
                            resized_8 = (resized.astype(np.float32) / 256.0).astype(np.uint8)
                        else:
                            resized_8 = resized
                        resized_bgr = cv2.cvtColor(resized_8, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(out_path), resized_bgr)

                    results[tier_name].append({
                        "path": str(out_path),
                        "source": str(pp),
                        "resolution": [rw, rh],
                    })
                    logger.info("  -> %s: %dx%d %s", tier_name, rw, rh, out_path.name)

            # ---- Face crop extraction ----
            if quarter_image is None:
                # Generate quarter-res for face detection
                quarter_image = _resize_image(image_srgb, quarter_scale)

            face_bbox = _detect_face_bbox_mediapipe(quarter_image)
            if face_bbox is not None:
                crop, crop_info = _extract_face_crop(
                    image_srgb, face_bbox, quarter_scale, padding_factor=1.5,
                )
                crop_path = face_crops_dir / f"{stem}_face.tiff"
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_bgr)

                results["face_crops"].append({
                    "path": str(crop_path),
                    "source": str(pp),
                    "resolution": [crop_info["crop_width"], crop_info["crop_height"]],
                    "crop_info": crop_info,
                })
                logger.info(
                    "  -> face_crop: %dx%d %s",
                    crop_info["crop_width"], crop_info["crop_height"],
                    crop_path.name,
                )
            else:
                logger.warning("  No face detected in %s, skipping face crop", pp.name)

            # Free memory — 200MP images are ~600MB in memory
            del image_srgb
            if suffix == ".dng":
                del image_linear

        except Exception as e:
            logger.error("Failed to process %s: %s", pp.name, e, exc_info=True)
            continue

    # Save processing summary
    summary_path = output_dir / "multi_res_summary.json"
    summary = {
        tier: [{"path": r["path"], "resolution": r["resolution"]} for r in rlist]
        for tier, rlist in results.items()
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Multi-res processing complete: %d photos -> %s",
        len(photo_paths), output_dir,
    )

    for tier_name, rlist in results.items():
        if rlist:
            logger.info("  %s: %d files", tier_name, len(rlist))

    return results


def register_multi_lens_photos(
    wide_photos_dir: str | Path,
    main_photos_dir: str | Path,
    colmap_model_dir: str | Path,
    colmap_binary: str | Path,
    database_path: str | Path | None = None,
    camera_models: dict[str, dict] | None = None,
) -> Path:
    """Register multi-lens photos into a COLMAP reconstruction.

    Wide-lens photos (same lens as video) are registered first since they
    share intrinsics.  Then 200MP main sensor photos (quarter-res versions)
    are registered with a separate camera model.

    Args:
        wide_photos_dir: Directory containing processed wide-lens photos.
        main_photos_dir: Directory containing quarter-res 200MP photos.
        colmap_model_dir: Path to existing COLMAP sparse model.
        colmap_binary: Path to COLMAP binary/bat file.
        database_path: Path to COLMAP database. Auto-detected if None.
        camera_models: Camera model dicts from ``compute_multi_lens_camera_models``.
            Used to set initial intrinsics for each lens group.

    Returns:
        Path to the updated COLMAP model directory.
    """
    wide_photos_dir = Path(wide_photos_dir)
    main_photos_dir = Path(main_photos_dir)
    colmap_model_dir = Path(colmap_model_dir)
    colmap_binary = Path(colmap_binary)

    if database_path is None:
        database_path = colmap_model_dir.parent.parent.parent / "database.db"
    database_path = Path(database_path)

    colmap_bin = str(colmap_binary)
    output_model = colmap_model_dir.parent / "0_multi_lens"
    output_model.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Register wide-lens photos (same camera as video) ----
    intermediate_model = colmap_model_dir
    if wide_photos_dir.exists() and any(wide_photos_dir.iterdir()):
        logger.info("Registering wide-lens photos from %s...", wide_photos_dir)
        try:
            intermediate_model = _register_lens_group(
                photo_dir=wide_photos_dir,
                model_dir=colmap_model_dir,
                colmap_bin=colmap_bin,
                database_path=database_path,
                output_name="0_wide",
                single_camera_with_existing=True,
                camera_model_params=camera_models.get("photo_wide_23mm") if camera_models else None,
            )
        except Exception as e:
            logger.warning("Wide-lens registration failed: %s", e)
            intermediate_model = colmap_model_dir

    # ---- Step 2: Register 200MP quarter-res photos (different camera) ----
    if main_photos_dir.exists() and any(main_photos_dir.iterdir()):
        logger.info("Registering 200MP photos (quarter-res) from %s...", main_photos_dir)
        try:
            output_model = _register_lens_group(
                photo_dir=main_photos_dir,
                model_dir=intermediate_model,
                colmap_bin=colmap_bin,
                database_path=database_path,
                output_name="0_multi_lens",
                single_camera_with_existing=False,  # Different camera model
                camera_model_params=camera_models.get("photo_main_200mp") if camera_models else None,
            )
        except Exception as e:
            logger.warning("200MP registration failed: %s", e)
            output_model = intermediate_model
    else:
        # Copy intermediate to final
        _copy_model(intermediate_model, output_model)

    logger.info("Multi-lens registration complete: %s", output_model)
    return output_model


def _register_lens_group(
    photo_dir: Path,
    model_dir: Path,
    colmap_bin: str,
    database_path: Path,
    output_name: str,
    single_camera_with_existing: bool,
    camera_model_params: dict | None = None,
) -> Path:
    """Register a group of photos from one lens into a COLMAP model.

    Args:
        photo_dir: Directory with photos to register.
        model_dir: Input COLMAP model.
        colmap_bin: COLMAP binary path.
        database_path: COLMAP database path.
        output_name: Name for the output model subdirectory.
        single_camera_with_existing: If True, share camera with existing model.
        camera_model_params: Optional initial camera intrinsics.

    Returns:
        Path to output model directory.
    """
    output_dir = model_dir.parent / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature extraction
    cmd_extract = [
        colmap_bin, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(photo_dir),
        "--ImageReader.single_camera", "1" if single_camera_with_existing else "0",
        "--SiftExtraction.max_num_features", "8192",
    ]

    # Set initial camera params if available
    if camera_model_params and not single_camera_with_existing:
        cmd_extract.extend([
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.camera_params",
            f"{camera_model_params['fx']},{camera_model_params['fy']},"
            f"{camera_model_params['cx']},{camera_model_params['cy']}",
        ])

    try:
        subprocess.run(
            cmd_extract, capture_output=True, text=True, check=True, timeout=300,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Feature extraction failed for %s: %s", photo_dir, e)
        _copy_model(model_dir, output_dir)
        return output_dir

    # Exhaustive matching
    try:
        subprocess.run(
            [colmap_bin, "exhaustive_matcher",
             "--database_path", str(database_path)],
            capture_output=True, text=True, check=True, timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Matching failed: %s", e)
        _copy_model(model_dir, output_dir)
        return output_dir

    # Image registrator
    try:
        subprocess.run(
            [colmap_bin, "image_registrator",
             "--database_path", str(database_path),
             "--input_path", str(model_dir),
             "--output_path", str(output_dir)],
            capture_output=True, text=True, check=True, timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Image registration failed: %s", e)
        _copy_model(model_dir, output_dir)
        return output_dir

    # Bundle adjustment
    refined_dir = model_dir.parent / f"{output_name}_refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [colmap_bin, "bundle_adjuster",
             "--input_path", str(output_dir),
             "--output_path", str(refined_dir)],
            capture_output=True, text=True, check=True, timeout=600,
        )
        _copy_model(refined_dir, output_dir)
        logger.info("Bundle adjustment complete for %s", output_name)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("Bundle adjustment failed for %s (using pre-BA model): %s", output_name, e)

    return output_dir


def _copy_model(src_dir: Path, dst_dir: Path) -> None:
    """Copy COLMAP model files from src to dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "cameras.bin", "images.bin", "points3D.bin",
        "cameras.txt", "images.txt", "points3D.txt",
    ]:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)
