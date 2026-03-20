"""Single photo analyzer -- extracts all available information from one face photo.

Every analysis result is stored in the DB, accumulating a face identity profile
over time. Each photo makes the system smarter for future reconstructions.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# InsightFace app singleton (lazy-loaded, ~1s cold start)
_insight_app = None


def _get_insight_app():
    """Lazy-load InsightFace app on first use."""
    global _insight_app
    if _insight_app is None:
        import insightface
        _insight_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition", "landmark_2d_106", "landmark_3d_68", "genderage"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _insight_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace model loaded")
    return _insight_app


# 3D model points for solvePnP head pose estimation (68-landmark subset)
# Standard 3D face model coordinates (nose tip centered)
_MODEL_POINTS_68 = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (30)
    (0.0, -330.0, -65.0),     # Chin (8)
    (-225.0, 170.0, -135.0),  # Left eye left corner (36)
    (225.0, 170.0, -135.0),   # Right eye right corner (45)
    (-150.0, -150.0, -125.0), # Left mouth corner (48)
    (150.0, -150.0, -125.0),  # Right mouth corner (54)
], dtype=np.float64)

# Indices into 68-landmark array for the 6 points above
_POSE_LANDMARK_IDS = [30, 8, 36, 45, 48, 54]


@dataclass
class PhotoAnalysis:
    """Complete analysis results from a single photo."""
    # Face detection
    detected: bool = False
    detection_score: float = 0.0
    bbox: list[float] = field(default_factory=list)

    # Identity
    embedding: list[float] = field(default_factory=list)
    age: int = 0
    gender: str = ""
    identity_match: Optional[str] = None
    identity_confidence: float = 0.0

    # Landmarks
    landmarks_2d_106: list[list[float]] = field(default_factory=list)
    landmarks_3d_68: list[list[float]] = field(default_factory=list)

    # Geometry
    head_pose: dict = field(default_factory=dict)
    face_area_ratio: float = 0.0

    # Quality
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    noise_level: float = 0.0
    color_temperature: str = "neutral"

    # Depth (monocular) -- not stored in JSON, only as separate file
    depth_map: Optional[np.ndarray] = field(default=None, repr=False)

    # Segmentation -- not stored in JSON, only as separate file
    face_mask: Optional[np.ndarray] = field(default=None, repr=False)

    # Viewpoint
    estimated_angle: str = "unknown"

    # Recommendations
    quality_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)

    # Timing
    analysis_time_ms: float = 0.0

    def to_json(self) -> str:
        """Serialize to JSON, excluding numpy arrays."""
        d = asdict(self)
        d.pop("depth_map", None)
        d.pop("face_mask", None)
        return json.dumps(d)


def _estimate_head_pose(landmarks_3d_68: np.ndarray, img_h: int, img_w: int) -> dict:
    """Estimate yaw/pitch/roll from 68 3D landmarks using solvePnP."""
    if len(landmarks_3d_68) < 68:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    image_points = np.array([
        landmarks_3d_68[idx][:2] for idx in _POSE_LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS_68, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return {
        "yaw": float(round(angles[1], 1)),
        "pitch": float(round(angles[0], 1)),
        "roll": float(round(angles[2], 1)),
    }


def _classify_viewpoint(yaw: float) -> str:
    """Classify head angle into a viewpoint category."""
    abs_yaw = abs(yaw)
    if abs_yaw < 15:
        return "front-facing"
    elif abs_yaw < 45:
        direction = "left" if yaw < 0 else "right"
        return f"three-quarter {direction}"
    elif abs_yaw < 75:
        direction = "left" if yaw < 0 else "right"
        return f"side profile {direction}"
    else:
        direction = "left" if yaw < 0 else "right"
        return f"near-profile {direction}"


def _analyze_color(img_bgr: np.ndarray) -> tuple[float, float, float, str]:
    """Analyze brightness, contrast, noise, and color temperature."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Noise estimation via Laplacian of a small region
    h, w = gray.shape
    center_crop = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    noise_level = float(np.std(cv2.Laplacian(center_crop, cv2.CV_64F)))

    # Color temperature from mean BGR
    b_mean = float(np.mean(img_bgr[:, :, 0]))
    r_mean = float(np.mean(img_bgr[:, :, 2]))
    if r_mean > b_mean * 1.15:
        color_temp = "warm"
    elif b_mean > r_mean * 1.15:
        color_temp = "cool"
    else:
        color_temp = "neutral"

    return brightness, contrast, noise_level, color_temp


def _compute_sharpness(img_bgr: np.ndarray, bbox: list[float]) -> float:
    """Compute Laplacian variance in the face region."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_crop = img_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _score_quality(
    sharpness: float,
    brightness: float,
    contrast: float,
    noise_level: float,
    detection_score: float,
    face_area_ratio: float,
) -> tuple[float, list[str]]:
    """Compute 0-100 quality score and generate recommendations."""
    score = 0.0
    recs = []

    # Sharpness (0-30 points)
    if sharpness > 200:
        score += 30
    elif sharpness > 100:
        score += 20
    elif sharpness > 50:
        score += 10
        recs.append("Image slightly soft -- consider better focus or stabilization")
    else:
        recs.append("Image is blurry -- use tripod or better lighting")

    # Brightness (0-20 points)
    if 80 < brightness < 180:
        score += 20
    elif 50 < brightness < 220:
        score += 12
        if brightness < 80:
            recs.append("Lighting too dark -- increase ambient light or use flash")
        else:
            recs.append("Slightly overexposed -- reduce light or lower exposure")
    else:
        if brightness <= 50:
            recs.append("Very dark image -- significantly increase lighting")
        else:
            recs.append("Very overexposed -- significantly reduce exposure")

    # Contrast (0-15 points)
    if 30 < contrast < 80:
        score += 15
    elif 20 < contrast < 100:
        score += 8
    else:
        recs.append("Poor contrast -- check lighting setup")

    # Detection confidence (0-15 points)
    score += min(15, detection_score * 15)

    # Face size (0-10 points)
    if face_area_ratio > 0.05:
        score += 10
    elif face_area_ratio > 0.02:
        score += 5
        recs.append("Face is small in frame -- move closer or zoom in")
    else:
        recs.append("Face too small -- get closer for better detail")

    # Noise (0-10 points)
    if noise_level < 20:
        score += 10
    elif noise_level < 40:
        score += 5
    else:
        recs.append("High noise level -- use lower ISO or better lighting")

    score = min(100, max(0, score))

    if not recs:
        recs.append("Good quality for reconstruction")

    return round(score, 1), recs


def _run_face_segmentation(img_bgr: np.ndarray, bbox: list[float]) -> Optional[np.ndarray]:
    """Simple GrabCut segmentation from face bounding box."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Expand bbox by 30% for GrabCut
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - int(bw * 0.15))
    y1 = max(0, y1 - int(bh * 0.15))
    x2 = min(w, x2 + int(bw * 0.15))
    y2 = min(h, y2 + int(bh * 0.15))

    rect = (x1, y1, x2 - x1, y2 - y1)
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return result_mask
    except cv2.error:
        return None


def analyze_photo(
    image_path: str | Path,
    run_depth: bool = False,
    run_segmentation: bool = True,
    stored_embeddings: Optional[list[tuple[str, list[float]]]] = None,
) -> PhotoAnalysis:
    """Run full analysis on a single photo.

    Args:
        image_path: Path to the image file.
        run_depth: If True, run DA3 monocular depth (slow, ~2s).
        run_segmentation: If True, run GrabCut face segmentation.
        stored_embeddings: List of (name, embedding) tuples for identity matching.

    Returns:
        PhotoAnalysis with all extracted information.
    """
    t0 = time.perf_counter()
    result = PhotoAnalysis()
    image_path = Path(image_path)

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        result.recommendations = [f"File not found: {image_path}"]
        return result

    # Load image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.error(f"Failed to read image: {image_path}")
        result.recommendations = ["Failed to read image -- unsupported format?"]
        return result

    img_h, img_w = img_bgr.shape[:2]

    # Downscale if very large (>4K) for faster InsightFace
    scale = 1.0
    if max(img_h, img_w) > 2048:
        scale = 2048.0 / max(img_h, img_w)
        img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr

    # Run InsightFace
    app = _get_insight_app()
    faces = app.get(img_small)

    if not faces:
        result.detected = False
        result.recommendations = ["No face detected -- ensure face is visible and well-lit"]
        result.analysis_time_ms = (time.perf_counter() - t0) * 1000
        return result

    # Use the largest face (by bbox area)
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    result.detected = True
    result.detection_score = round(float(face.det_score), 3)

    # Bbox -- scale back to original resolution
    bbox_scaled = [float(v / scale) for v in face.bbox]
    result.bbox = [round(v, 1) for v in bbox_scaled]

    # Embedding (512-dim)
    if face.embedding is not None:
        result.embedding = [round(float(v), 6) for v in face.embedding]

    # Age & Gender
    if hasattr(face, "age"):
        result.age = int(face.age)
    if hasattr(face, "gender"):
        result.gender = "M" if face.gender == 1 else "F"

    # 2D Landmarks (106-point)
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106 / scale  # scale back
        result.landmarks_2d_106 = [[round(float(x), 1), round(float(y), 1)] for x, y in lm106]

    # 3D Landmarks (68-point)
    if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
        lm68 = face.landmark_3d_68.copy()
        lm68[:, :2] /= scale  # scale back xy
        result.landmarks_3d_68 = [[round(float(x), 1), round(float(y), 1), round(float(z), 1)] for x, y, z in lm68]

    # Head pose -- use InsightFace built-in if available, else solvePnP
    if hasattr(face, "pose") and face.pose is not None:
        pose = face.pose
        result.head_pose = {
            "yaw": round(float(pose[0]), 1),
            "pitch": round(float(pose[1]), 1),
            "roll": round(float(pose[2]), 1),
        }
    elif result.landmarks_3d_68:
        lm68_arr = np.array(result.landmarks_3d_68)
        result.head_pose = _estimate_head_pose(lm68_arr, img_h, img_w)
    else:
        result.head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    yaw = result.head_pose.get("yaw", 0.0)
    result.estimated_angle = _classify_viewpoint(yaw)

    # Face area ratio
    bx1, by1, bx2, by2 = result.bbox
    face_area = (bx2 - bx1) * (by2 - by1)
    result.face_area_ratio = round(face_area / (img_h * img_w), 4)

    # Sharpness
    result.sharpness = round(_compute_sharpness(img_bgr, result.bbox), 1)

    # Color analysis
    brightness, contrast, noise_level, color_temp = _analyze_color(img_bgr)
    result.brightness = round(brightness, 1)
    result.contrast = round(contrast, 1)
    result.noise_level = round(noise_level, 1)
    result.color_temperature = color_temp

    # Quality score
    result.quality_score, result.recommendations = _score_quality(
        result.sharpness, result.brightness, result.contrast,
        result.noise_level, result.detection_score, result.face_area_ratio,
    )

    # Identity matching
    if stored_embeddings and result.embedding:
        emb = np.array(result.embedding)
        best_name = None
        best_sim = 0.0
        for name, stored_emb in stored_embeddings:
            stored = np.array(stored_emb)
            sim = float(np.dot(emb, stored) / (np.linalg.norm(emb) * np.linalg.norm(stored) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim > 0.6:
            result.identity_match = best_name
            result.identity_confidence = round(best_sim, 3)

    # Face segmentation (optional)
    if run_segmentation and result.bbox:
        result.face_mask = _run_face_segmentation(img_bgr, result.bbox)

    # Depth estimation (optional, slow)
    if run_depth:
        try:
            from depth.da3_unified import run_da3_depth_only
            result.depth_map = run_da3_depth_only(str(image_path))
        except Exception as e:
            logger.warning(f"Depth estimation failed: {e}")

    result.analysis_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


def analyze_and_print_json(image_path: str, run_depth: bool = False) -> None:
    """Analyze a photo and print JSON to stdout for Tauri IPC."""
    result = analyze_photo(image_path, run_depth=run_depth, run_segmentation=False)
    print(result.to_json())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python -m analysis.single_photo <image_path> [--depth]"}))
        sys.exit(1)

    img_path = sys.argv[1]
    do_depth = "--depth" in sys.argv

    logging.basicConfig(level=logging.INFO)
    analyze_and_print_json(img_path, run_depth=do_depth)
