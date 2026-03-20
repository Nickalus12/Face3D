"""InsightFace-based face detection and landmark extraction.

Replaces MediaPipe for higher accuracy, especially on side profiles and
extreme poses. Provides 106 2D landmarks, 68 3D landmarks, 512-dim face
embeddings, and face bounding boxes with high confidence scores.

Uses the buffalo_l model bundle (best quality) with CUDA acceleration.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load InsightFace (heavy import)
_app = None


def _get_app():
    """Lazy-initialize InsightFace FaceAnalysis."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace initialized (buffalo_l, CUDA)")
    return _app


def detect_single_face(image: np.ndarray) -> Optional[dict]:
    """Detect the largest face in an image and return landmarks + metadata.

    Args:
        image: BGR image array (OpenCV format).

    Returns:
        Dict with keys: bbox, det_score, landmarks_2d_5, landmarks_2d_106,
        landmarks_3d_68, embedding, age, gender. Or None if no face found.
    """
    app = _get_app()
    faces = app.get(image)
    if not faces:
        return None

    # Pick the largest face (by bounding box area)
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    result = {
        "bbox": face.bbox.tolist(),
        "det_score": float(face.det_score),
        "landmarks_2d_5": face.kps.tolist() if hasattr(face, "kps") and face.kps is not None else None,
        "landmarks_2d_106": face.landmark_2d_106.tolist() if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None else None,
        "landmarks_3d_68": face.landmark_3d_68.tolist() if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None else None,
        "embedding": face.embedding.tolist() if hasattr(face, "embedding") and face.embedding is not None else None,
        "age": int(face.age) if hasattr(face, "age") else None,
        "gender": "M" if getattr(face, "gender", 0) == 1 else "F",
    }
    return result


def detect_batch(
    image_paths: list[Path],
    output_dir: Path,
    max_image_size: int = 1920,
) -> dict:
    """Run InsightFace detection on a batch of images.

    Saves per-image JSON files with landmarks, embeddings, and metadata.
    Also generates landmark overlay preview images.

    Args:
        image_paths: List of image file paths.
        output_dir: Directory to save landmark JSON files.
        max_image_size: Downscale images larger than this before detection.

    Returns:
        Summary dict with counts and quality stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir.parent / "previews" if output_dir.parent.exists() else None

    detected = 0
    failed = 0
    scores = []
    embeddings = []

    from tqdm import tqdm
    for img_path in tqdm(image_paths, desc="InsightFace landmarks"):
        stem = img_path.stem
        out_file = output_dir / f"{stem}.json"

        # Skip if already processed
        if out_file.exists():
            try:
                data = json.loads(out_file.read_text())
                if data.get("detected") and data.get("det_score", 0) > 0:
                    detected += 1
                    scores.append(data["det_score"])
                    continue
            except Exception:
                pass

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Cannot read image: %s", img_path)
            failed += 1
            _save_empty_result(out_file, img_path, "unreadable")
            continue

        # Downscale if needed
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_image_size:
            scale = max_image_size / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # Detect
        result = detect_single_face(image)
        if result is None:
            failed += 1
            _save_empty_result(out_file, img_path, "no_face")
            continue

        # Scale landmarks back to original resolution if downscaled
        if scale != 1.0:
            inv_scale = 1.0 / scale
            if result["landmarks_2d_106"]:
                result["landmarks_2d_106"] = (np.array(result["landmarks_2d_106"]) * inv_scale).tolist()
            if result["landmarks_2d_5"]:
                result["landmarks_2d_5"] = (np.array(result["landmarks_2d_5"]) * inv_scale).tolist()
            if result["landmarks_3d_68"]:
                pts = np.array(result["landmarks_3d_68"])
                pts[:, :2] *= inv_scale  # Only scale x, y — z is depth
                result["landmarks_3d_68"] = pts.tolist()
            if result["bbox"]:
                result["bbox"] = (np.array(result["bbox"]) * inv_scale).tolist()

        # Build MediaPipe-compatible landmarks_2d field for FLAME fitter compatibility
        # The FLAME fitter reads "landmarks_2d" — provide the 68 3D points as 2D
        compat_2d = None
        if result.get("landmarks_3d_68"):
            compat_2d = [[pt[0], pt[1]] for pt in result["landmarks_3d_68"]]

        # Save result
        output_data = {
            "detected": True,
            "detector": "insightface_buffalo_l",
            "image": str(img_path.name),
            "image_width": w,
            "image_height": h,
            "landmarks_2d": compat_2d,  # Compat field for FLAME fitter
            **result,
        }
        out_file.write_text(json.dumps(output_data, indent=2))

        detected += 1
        scores.append(result["det_score"])
        if result.get("embedding"):
            embeddings.append(result["embedding"])

    # Save embeddings for ML prior (vector search)
    if embeddings:
        emb_path = output_dir / "face_embeddings.npz"
        np.savez_compressed(emb_path, embeddings=np.array(embeddings, dtype=np.float32))
        logger.info("Saved %d face embeddings to %s", len(embeddings), emb_path)

    summary = {
        "total": len(image_paths),
        "detected": detected,
        "failed": failed,
        "mean_score": float(np.mean(scores)) if scores else 0,
        "min_score": float(np.min(scores)) if scores else 0,
        "detector": "insightface_buffalo_l",
    }
    logger.info(
        "InsightFace: %d/%d detected (mean score %.3f), %d failed",
        detected, len(image_paths), summary["mean_score"], failed,
    )
    return summary


def generate_previews(
    image_paths: list[Path],
    landmark_dir: Path,
    preview_dir: Path,
    num_previews: int = 10,
) -> list[Path]:
    """Generate landmark overlay preview images.

    Picks evenly-spaced frames and draws landmarks on them.
    """
    preview_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(image_paths) // num_previews)
    picks = list(range(0, len(image_paths), step))[:num_previews]

    saved = []
    for idx in picks:
        img_path = image_paths[idx]
        lm_path = landmark_dir / f"{img_path.stem}.json"
        if not lm_path.exists():
            continue

        data = json.loads(lm_path.read_text())
        if not data.get("detected"):
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Draw 106 2D landmarks
        pts_106 = data.get("landmarks_2d_106")
        if pts_106:
            for pt in pts_106:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        # Draw bounding box
        bbox = data.get("bbox")
        if bbox:
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 200, 255), 2,
            )

        # Draw detection score
        score = data.get("det_score", 0)
        cv2.putText(
            image, f"InsightFace {score:.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        out_path = preview_dir / f"landmarks_{img_path.stem}.png"
        cv2.imwrite(str(out_path), image)
        saved.append(out_path)

    logger.info("Generated %d landmark preview images", len(saved))
    return saved


def _save_empty_result(out_file: Path, img_path: Path, reason: str):
    """Save an empty detection result."""
    out_file.write_text(json.dumps({
        "detected": False,
        "detector": "insightface_buffalo_l",
        "image": str(img_path.name),
        "reason": reason,
    }, indent=2))


def get_landmarks_for_flame(landmark_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load 68-point 3D landmarks in FLAME-compatible format.

    Returns:
        (landmarks_list, frame_names) — list of (68, 2) arrays and frame names.
    """
    landmarks = []
    names = []

    for lm_file in sorted(landmark_dir.glob("*.json")):
        data = json.loads(lm_file.read_text())
        if not data.get("detected"):
            continue

        # Use 3D 68-point landmarks (x, y only for 2D projection matching)
        pts_3d = data.get("landmarks_3d_68")
        if pts_3d:
            pts = np.array(pts_3d, dtype=np.float32)[:, :2]  # (68, 2)
            landmarks.append(pts)
            names.append(lm_file.stem)

    return landmarks, names
