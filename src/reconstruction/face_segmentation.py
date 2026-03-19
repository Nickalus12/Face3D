"""Face region segmentation using MediaPipe Selfie Segmentation.

Produces binary face/background masks for filtering reconstruction inputs
and isolating face regions from background clutter.

Supports optional SAM2 backend for higher-quality masks (especially around
hair and ears), with automatic fallback to MediaPipe when SAM2 is not
installed.  All masks can be refined with OpenCV GrabCut for cleaner
boundaries at no extra dependency cost.
"""

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Handle MediaPipe API changes across versions
if hasattr(mp, "solutions") and hasattr(mp.solutions, "selfie_segmentation"):
    _selfie_seg_module = mp.solutions.selfie_segmentation
else:
    from mediapipe.python.solutions import selfie_segmentation as _selfie_seg_module

# ---------------------------------------------------------------------------
# SAM2 availability check
# ---------------------------------------------------------------------------
_SAM2_AVAILABLE = False
_sam2_build = None
_sam2_predictor_cls = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _sam2_build = build_sam2
    _sam2_predictor_cls = SAM2ImagePredictor
    _SAM2_AVAILABLE = True
    logger.info("SAM2 is available — will use for high-quality segmentation")
except ImportError:
    try:
        from segment_anything_2.build_sam import build_sam2
        from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor
        _sam2_build = build_sam2
        _sam2_predictor_cls = SAM2ImagePredictor
        _SAM2_AVAILABLE = True
        logger.info("SAM2 (segment_anything_2) is available")
    except ImportError:
        logger.debug("SAM2 not installed — will use MediaPipe for segmentation")


# ---------------------------------------------------------------------------
# SAM2 segmentation
# ---------------------------------------------------------------------------

def try_sam2_segmentation(
    image: np.ndarray,
    predictor: object | None = None,
    point_coords: np.ndarray | None = None,
) -> np.ndarray | None:
    """Attempt face segmentation using SAM2 for higher-quality masks.

    SAM2 produces much cleaner boundaries around hair, ears, and jawline
    compared to MediaPipe Selfie Segmentation.

    Args:
        image: BGR image as numpy array (H, W, 3).
        predictor: Pre-initialized SAM2ImagePredictor instance.  When
            ``None`` the function returns ``None`` (caller should
            initialise once and pass it in for batch usage).
        point_coords: Optional (N, 2) array of prompt points in pixel
            coordinates (x, y).  When ``None`` the image centre is used
            as a single foreground prompt.

    Returns:
        Binary mask as uint8 (H, W) with 255=foreground, 0=background,
        or ``None`` if SAM2 is unavailable or inference fails.
    """
    if not _SAM2_AVAILABLE or predictor is None:
        return None

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        h, w = image.shape[:2]
        if point_coords is None:
            # Default prompt: centre of image (likely the face)
            point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
        point_labels = np.ones(len(point_coords), dtype=np.int32)  # all foreground

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Pick the highest-scoring mask
        best_idx = int(np.argmax(scores))
        mask = (masks[best_idx] > 0).astype(np.uint8) * 255

        return mask
    except Exception as e:
        logger.warning("SAM2 segmentation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# GrabCut mask refinement
# ---------------------------------------------------------------------------

def refine_mask_with_grabcut(
    image: np.ndarray,
    mask: np.ndarray,
    iter_count: int = 5,
) -> np.ndarray:
    """Refine a segmentation mask using OpenCV GrabCut.

    GrabCut uses colour information to improve mask boundaries, producing
    smoother edges especially around hair.  This is a *free* quality
    improvement since OpenCV is already a dependency.

    Args:
        image: BGR image as numpy array (H, W, 3).
        mask: Binary mask (H, W) with 255=foreground, 0=background.
        iter_count: Number of GrabCut iterations (default 5).

    Returns:
        Refined binary mask as uint8 (H, W), 255=foreground.
    """
    h, w = mask.shape[:2]

    # Convert binary mask to GrabCut label mask
    gc_mask = np.where(mask > 127, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

    # Mark definite foreground in the interior (eroded region)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    interior = cv2.erode(mask, kernel, iterations=2)
    gc_mask[interior > 127] = cv2.GC_FGD

    # Mark definite background far from the mask
    exterior = cv2.dilate(mask, kernel, iterations=3)
    gc_mask[exterior == 0] = cv2.GC_BGD

    # GrabCut needs at least some definite FG and BG
    if not (np.any(gc_mask == cv2.GC_FGD) and np.any(gc_mask == cv2.GC_BGD)):
        logger.debug("GrabCut: insufficient FG/BG labels, returning original mask")
        return mask

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image, gc_mask, None,
            bgd_model, fgd_model,
            iterCount=iter_count,
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as e:
        logger.debug("GrabCut failed (%s), returning original mask", e)
        return mask

    # Combine probable + definite foreground
    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    return refined


# ---------------------------------------------------------------------------
# Mask quality scoring
# ---------------------------------------------------------------------------

def compute_mask_quality(mask: np.ndarray) -> float:
    """Compute a quality score for a segmentation mask.

    The score is in [0, 1] and checks:
      - Mask is not all-zero or all-one (degenerate).
      - Foreground covers a reasonable area ratio (10-90% of the frame).
      - A single dominant connected component (no scattered islands).
      - Smooth boundary (low perimeter-to-area ratio).

    Args:
        mask: Binary mask (H, W) with 255=foreground, 0=background.

    Returns:
        Quality score in [0.0, 1.0].
    """
    h, w = mask.shape[:2]
    total_pixels = h * w
    fg_pixels = int(np.count_nonzero(mask))

    # Degenerate masks
    if fg_pixels == 0 or fg_pixels == total_pixels:
        return 0.0

    score = 1.0

    # --- Area ratio check (ideal: 10%-90%) ---
    area_ratio = fg_pixels / total_pixels
    if area_ratio < 0.05:
        score *= 0.2
    elif area_ratio < 0.10:
        score *= 0.6
    elif area_ratio > 0.95:
        score *= 0.3
    elif area_ratio > 0.90:
        score *= 0.7

    # --- Connected components check ---
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    num_fg_components = num_labels - 1  # exclude background label 0

    if num_fg_components == 0:
        return 0.0
    elif num_fg_components == 1:
        pass  # ideal
    elif num_fg_components <= 3:
        # Check if the largest component dominates
        fg_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_ratio = float(fg_areas.max()) / fg_pixels
        if largest_ratio > 0.9:
            score *= 0.95  # minor islands, mostly fine
        else:
            score *= 0.7
    else:
        # Many scattered islands
        fg_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_ratio = float(fg_areas.max()) / fg_pixels
        score *= max(0.3, largest_ratio * 0.8)

    # --- Boundary smoothness (compactness = 4*pi*area / perimeter^2) ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, closed=True)
        area = cv2.contourArea(largest_contour)
        if perimeter > 0 and area > 0:
            compactness = (4 * np.pi * area) / (perimeter * perimeter)
            # compactness of 1.0 = perfect circle; faces are ~0.4-0.7
            # Penalise very jagged masks (compactness < 0.1)
            if compactness < 0.05:
                score *= 0.5
            elif compactness < 0.1:
                score *= 0.75

    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Main segmenter class
# ---------------------------------------------------------------------------

class FaceSegmenter:
    """Segments face/person regions from images."""

    def __init__(self, model_type: str = "mediapipe"):
        """Initialize the segmentation model.

        Args:
            model_type: Segmentation backend. Supports 'mediapipe' and
                'sam2' (falls back to mediapipe if SAM2 is not installed).
        """
        self._model_type = model_type
        self._sam2_predictor = None

        if model_type == "sam2" and _SAM2_AVAILABLE:
            logger.info("Using SAM2 backend for face segmentation")
            # SAM2 predictor will be initialised lazily or by the caller
            self._model_type = "sam2"
            self._segmenter = None
        elif model_type == "sam2" and not _SAM2_AVAILABLE:
            logger.warning("SAM2 requested but not installed — falling back to MediaPipe")
            self._model_type = "mediapipe"

        if self._model_type == "mediapipe":
            self._segmenter = _selfie_seg_module.SelfieSegmentation(
                model_selection=1,  # 1 = general model (landscape), 0 = closer-range
            )

    # Maximum dimension for MediaPipe input.  Larger images are
    # downscaled before inference and the mask is upscaled back.
    # MediaPipe resizes internally to a fixed tensor anyway, so
    # feeding 200MP images only wastes time on the colorspace
    # conversion and data transfer.
    _MAX_INPUT_DIM = 1080

    def segment(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Produce a binary face/person mask for a single image.

        For images larger than ``_MAX_INPUT_DIM`` pixels on their longest
        side, the image is downscaled before MediaPipe inference and the
        resulting mask is upscaled back to the original resolution using
        nearest-neighbor interpolation.  This avoids the massive overhead
        of processing 200MP photos through MediaPipe when it internally
        resizes to a much smaller tensor anyway.

        Args:
            image: BGR image as numpy array (H, W, 3).
            threshold: Confidence threshold for the binary mask (0-1).

        Returns:
            Binary mask as uint8 array (H, W) with 255 for face/person, 0 for background.
        """
        h, w = image.shape[:2]

        # Downscale large images for MediaPipe efficiency
        scale = 1.0
        process_image = image
        if max(h, w) > self._MAX_INPUT_DIM:
            scale = self._MAX_INPUT_DIM / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            process_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        mask = None

        # Try SAM2 first if available (SAM2 benefits from higher res, pass original)
        if _SAM2_AVAILABLE and self._sam2_predictor is not None:
            mask = try_sam2_segmentation(image, predictor=self._sam2_predictor)

        # Fallback to MediaPipe (use downscaled image)
        if mask is None and self._segmenter is not None:
            rgb = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
            results = self._segmenter.process(rgb)
            raw_mask = results.segmentation_mask
            mask = (raw_mask > threshold).astype(np.uint8) * 255

            # Upscale mask back to original resolution if we downscaled
            if scale < 1.0:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if mask is None:
            # Last resort: return empty mask
            logger.warning("No segmentation backend produced a mask")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Morphological cleanup: close small holes, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Optional: keep only the largest connected component
        mask = _keep_largest_component(mask)

        return mask

    def segment_batch(
        self,
        frames_dir: Path,
        output_dir: Path,
        threshold: float = 0.5,
        refine_grabcut: bool = True,
        max_workers: int | None = None,
    ) -> list[Path]:
        """Segment all images in a directory and save masks as PNG.

        Image I/O (reading from disk) is parallelised with a
        :class:`~concurrent.futures.ThreadPoolExecutor` while MediaPipe
        segmentation runs sequentially on the main thread (MediaPipe is
        not thread/process-safe).  High-resolution images are
        automatically downscaled before inference (see :meth:`segment`).

        Args:
            frames_dir: Directory containing input images.
            output_dir: Directory where mask PNG files are written.
            threshold: Confidence threshold for binary masks.
            refine_grabcut: If ``True``, apply GrabCut refinement to each
                mask for cleaner boundaries (default ``True``).
            max_workers: Number of threads for parallel image I/O.
                Defaults to an I/O-optimised count.

        Returns:
            List of paths to generated mask files.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.parallel import get_optimal_workers

        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_paths = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in image_extensions
        )

        if not image_paths:
            logger.warning("No images found in %s", frames_dir)
            return []

        logger.info("Segmenting %d images ...", len(image_paths))

        # --- Pre-load images in parallel (I/O-bound) ---
        if max_workers is None:
            max_workers = get_optimal_workers("io")

        loaded_images: dict[int, np.ndarray | None] = {}

        def _read_image(idx_path: tuple[int, Path]) -> tuple[int, np.ndarray | None]:
            idx, path = idx_path
            return idx, cv2.imread(str(path))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_read_image, (i, p)): i
                for i, p in enumerate(image_paths)
            }
            for fut in as_completed(futures):
                idx, img = fut.result()
                loaded_images[idx] = img

        # --- Sequential MediaPipe segmentation ---
        output_paths: list[Path] = []
        quality_scores: list[float] = []
        low_quality_count = 0

        for i, img_path in enumerate(tqdm(image_paths, desc="Segmentation")):
            image = loaded_images.get(i)
            if image is None:
                logger.warning("Could not read image: %s", img_path)
                continue

            mask = self.segment(image, threshold=threshold)

            # GrabCut refinement
            if refine_grabcut:
                mask = refine_mask_with_grabcut(image, mask)

            # Compute and log mask quality
            quality = compute_mask_quality(mask)
            quality_scores.append(quality)
            if quality < 0.5:
                low_quality_count += 1
                logger.warning(
                    "Low mask quality (%.2f) for %s", quality, img_path.name,
                )

            out_path = output_dir / (img_path.stem + "_mask.png")
            cv2.imwrite(str(out_path), mask)
            output_paths.append(out_path)

        # Free loaded images
        loaded_images.clear()

        # Summary statistics
        if quality_scores:
            mean_quality = sum(quality_scores) / len(quality_scores)
            logger.info(
                "Saved %d masks to %s (mean quality: %.3f, low quality: %d)",
                len(output_paths), output_dir, mean_quality, low_quality_count,
            )
            if low_quality_count > 0:
                logger.warning(
                    "%d / %d masks have quality < 0.5 — review these frames",
                    low_quality_count, len(quality_scores),
                )
        else:
            logger.info("Saved %d masks to %s", len(output_paths), output_dir)

        return output_paths

    def close(self) -> None:
        if hasattr(self, "_segmenter") and self._segmenter is not None and hasattr(self._segmenter, "close"):
            self._segmenter.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask

    # Label 0 is background; find the largest foreground component
    # stats[:, cv2.CC_STAT_AREA] gives area per label
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    largest_label = 1 + np.argmax(areas)

    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result
