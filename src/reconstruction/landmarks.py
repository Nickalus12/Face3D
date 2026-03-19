"""Facial landmark detection using MediaPipe Face Mesh.

Detects 478 facial landmarks per frame and provides utilities to map
MediaPipe landmarks to FLAME mesh vertices via barycentric coordinates.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Handle MediaPipe API changes across versions
if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
    _face_mesh_module = mp.solutions.face_mesh
else:
    from mediapipe.python.solutions import face_mesh as _face_mesh_module


def _compute_landmark_quality(
    landmarks_2d: np.ndarray,
    confidence: float,
    image_height: int,
    image_width: int,
) -> float:
    """Compute a quality score for a landmark detection.

    The score combines detection confidence with face size relative to the
    frame. Larger faces relative to the image tend to give better landmarks.

    Args:
        landmarks_2d: (N, 2) landmark pixel coordinates.
        confidence: MediaPipe detection confidence in [0, 1].
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Quality score in [0, 1], higher is better.
    """
    # Face bounding box relative to image
    x_min, y_min = landmarks_2d.min(axis=0)
    x_max, y_max = landmarks_2d.max(axis=0)
    face_width = x_max - x_min
    face_height = y_max - y_min
    face_area = face_width * face_height
    image_area = image_height * image_width

    # Face size ratio: what fraction of the image is the face
    # Ideal for face capture: 0.1 to 0.5 of image area
    size_ratio = face_area / max(image_area, 1)

    # Score the size: penalize very small faces (<5% of image) and very large (>80%)
    if size_ratio < 0.05:
        size_score = size_ratio / 0.05  # Linear ramp 0->1 from 0% to 5%
    elif size_ratio > 0.8:
        size_score = max(0.0, 1.0 - (size_ratio - 0.8) / 0.2)
    else:
        size_score = 1.0

    # Combine confidence and size score (weighted)
    quality = 0.6 * confidence + 0.4 * size_score
    return float(np.clip(quality, 0.0, 1.0))


class FaceLandmarkDetector:
    """Detects 478 facial landmarks using MediaPipe Face Mesh."""

    def __init__(self, max_faces: int = 1, min_detection_confidence: float = 0.5):
        """Initialize MediaPipe Face Mesh.

        Args:
            max_faces: Maximum number of faces to detect.
            min_detection_confidence: Minimum confidence for initial detection.
        """
        self._max_faces = max_faces
        self._min_detection_confidence = min_detection_confidence
        self._face_mesh = _face_mesh_module.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    # Maximum dimension for MediaPipe input.  Larger images are
    # downscaled before inference and coordinates are mapped back to
    # the original resolution.  MediaPipe resizes internally anyway,
    # so this avoids wasting time on 200MP photos.
    _MAX_INPUT_DIM = 1920

    def detect(self, image: np.ndarray) -> Optional[dict]:
        """Detect facial landmarks in a single image.

        For images larger than ``_MAX_INPUT_DIM`` pixels on their longest
        side, the image is downscaled before MediaPipe inference and the
        resulting landmark coordinates are mapped back to the original
        resolution.  This dramatically improves throughput on high-res
        photos (e.g. 200MP Samsung Expert RAW) with no loss of accuracy
        since MediaPipe internally resizes to a fixed input tensor.

        Args:
            image: BGR image as a numpy array (H, W, 3).

        Returns:
            Dict with keys 'landmarks_2d' (478, 2), 'landmarks_3d' (478, 3),
            'confidence' (float), and 'quality_score' (float in [0,1]),
            or None if no face is detected.
        """
        h, w = image.shape[:2]

        # Downscale if image is very large (MediaPipe resizes internally anyway)
        scale = 1.0
        process_image = image
        if max(h, w) > self._MAX_INPUT_DIM:
            scale = self._MAX_INPUT_DIM / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            process_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ph, pw = process_image.shape[:2]

        try:
            rgb = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
        except Exception as e:
            logger.warning("MediaPipe face mesh failed: %s", e)
            return None

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        num_landmarks = len(face.landmark)

        landmarks_2d = np.empty((num_landmarks, 2), dtype=np.float32)
        landmarks_3d = np.empty((num_landmarks, 3), dtype=np.float32)

        visibility_sum = 0.0
        for i, lm in enumerate(face.landmark):
            # Map normalized coords to original resolution
            landmarks_2d[i, 0] = lm.x * w
            landmarks_2d[i, 1] = lm.y * h
            landmarks_3d[i, 0] = lm.x * w
            landmarks_3d[i, 1] = lm.y * h
            landmarks_3d[i, 2] = lm.z * w  # z is scaled relative to image width
            visibility_sum += getattr(lm, "visibility", 1.0)

        confidence = visibility_sum / max(num_landmarks, 1)

        # Compute quality score based on confidence and face size relative to frame
        quality_score = _compute_landmark_quality(landmarks_2d, confidence, h, w)

        return {
            "landmarks_2d": landmarks_2d,
            "landmarks_3d": landmarks_3d,
            "confidence": float(confidence),
            "quality_score": float(quality_score),
        }

    def detect_batch(
        self,
        frames_dir: Path,
        output_dir: Path,
        max_workers: int | None = None,
    ) -> list[Path]:
        """Detect landmarks for every image in a directory and save as JSON.

        Image I/O (reading from disk) is parallelised with a
        :class:`~concurrent.futures.ThreadPoolExecutor` while MediaPipe
        detection runs sequentially on the main thread (MediaPipe/TFLite
        is not thread-safe).  High-resolution images are automatically
        downscaled before inference for speed (see :meth:`detect`).

        Args:
            frames_dir: Directory containing input images.
            output_dir: Directory where per-frame JSON files are written.
            max_workers: Number of threads for parallel image I/O.
                Defaults to an I/O-optimised count.

        Returns:
            List of paths to generated JSON files.
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

        logger.info("Detecting landmarks for %d images ...", len(image_paths))

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

        # --- Sequential MediaPipe detection ---
        output_paths: list[Path] = []
        detected_count = 0
        failed_count = 0

        for i, img_path in enumerate(tqdm(image_paths, desc="Landmarks")):
            image = loaded_images.get(i)
            out_path = output_dir / (img_path.stem + ".json")

            if image is None:
                logger.warning("Could not read image: %s", img_path)
                payload = {"detected": False, "image": img_path.name}
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                output_paths.append(out_path)
                failed_count += 1
                continue

            try:
                result = self.detect(image)
            except Exception as e:
                logger.warning("Landmark detection failed for %s: %s", img_path.name, e)
                result = None

            if result is None:
                payload = {"detected": False, "image": img_path.name}
                failed_count += 1
            else:
                payload = {
                    "detected": True,
                    "image": img_path.name,
                    "image_width": image.shape[1],
                    "image_height": image.shape[0],
                    "landmarks_2d": result["landmarks_2d"].tolist(),
                    "landmarks_3d": result["landmarks_3d"].tolist(),
                    "confidence": result["confidence"],
                    "quality_score": result["quality_score"],
                }
                detected_count += 1

            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            output_paths.append(out_path)

        # Free loaded images
        loaded_images.clear()

        logger.info(
            "Saved %d landmark files to %s (%d detected, %d failed)",
            len(output_paths), output_dir, detected_count, failed_count,
        )
        return output_paths

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_mediapipe_to_flame_mapping(
    embedding_path: Path,
) -> dict:
    """Load the MediaPipe-to-FLAME landmark embedding.

    The .npz file contains arrays that map a subset of MediaPipe landmark
    indices to positions on the FLAME mesh surface via barycentric
    interpolation on triangle faces.

    Args:
        embedding_path: Path to mediapipe_landmark_embedding.npz.

    Returns:
        Dict with:
            'lmk_faces_idx': (K,) int array — FLAME face indices per landmark.
            'lmk_bary_coords': (K, 3) float array — barycentric weights.
            'landmark_indices': (K,) int array — which MediaPipe landmark
                indices are covered (0-based).
    """
    embedding_path = Path(embedding_path)
    data = np.load(str(embedding_path), allow_pickle=True)

    lmk_faces_idx = data["lmk_face_idx"].astype(np.int64).flatten()
    lmk_bary_coords = data["lmk_b_coords"].astype(np.float64)

    # Some embeddings include an explicit list of MediaPipe indices that were
    # selected; fall back to sequential if absent.
    if "landmark_indices" in data:
        landmark_indices = data["landmark_indices"].astype(np.int64).flatten()
    else:
        landmark_indices = np.arange(len(lmk_faces_idx), dtype=np.int64)

    logger.info(
        "Loaded MediaPipe→FLAME embedding: %d landmarks, from %s",
        len(lmk_faces_idx),
        embedding_path.name,
    )

    return {
        "lmk_faces_idx": lmk_faces_idx,
        "lmk_bary_coords": lmk_bary_coords,
        "landmark_indices": landmark_indices,
    }
