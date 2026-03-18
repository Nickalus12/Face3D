"""Multi-view FLAME model fitting via differentiable landmark reprojection.

Optimizes FLAME shape, expression, and pose parameters to minimize the
reprojection error of facial landmarks across multiple calibrated views.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Per-landmark weight groups (MediaPipe 468-style indices) ─────────────────
# These assume the landmark_indices mapping selects a subset; weights are
# applied by *position* in the mapped subset.  The helper below builds a
# weight tensor given the full 478-landmark index list so that we can
# classify each selected landmark into a semantic region.

_LANDMARK_REGION_WEIGHTS = {
    "contour": 0.5,     # jawline / face boundary
    "forehead": 0.3,    # forehead landmarks (less stable)
    "nose": 1.5,        # nose bridge & tip
    "eye": 2.0,         # eye contour & iris
    "mouth": 1.5,       # lips & mouth corners
    "default": 1.0,
}

# MediaPipe landmark index ranges (approximate, 468 canonical landmarks).
# Ranges are inclusive on both ends.
_MP_EYE_INDICES = set(range(33, 42)) | set(range(133, 134)) | set(range(153, 160)) | \
                  set(range(246, 250)) | set(range(263, 272)) | set(range(362, 363)) | \
                  set(range(373, 382)) | set(range(466, 468)) | \
                  {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                   130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30, 247,
                   263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
                   359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258, 257, 259, 260, 467}
_MP_MOUTH_INDICES = {0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                     61, 185, 40, 39, 37, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                     308, 324, 318, 402, 317, 14, 87, 178, 88, 95}
_MP_NOSE_INDICES = {1, 2, 98, 327, 168, 5, 4, 195, 197, 6, 122, 351, 196, 3, 236, 456,
                    198, 174, 399, 131, 49, 279, 360, 278, 438, 218, 115, 344, 219, 439}
_MP_CONTOUR_INDICES = set(range(0, 17)) | {234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
                                             152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
                                             10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                                             288, 397, 365, 379, 378, 400, 377}
_MP_FOREHEAD_INDICES = {10, 338, 297, 332, 284, 251, 389, 356, 67, 109, 103, 54, 21, 162,
                        127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152}


def _build_landmark_weights(landmark_indices: list | np.ndarray, device: torch.device) -> torch.Tensor:
    """Return a (K,) weight tensor for the selected landmark subset."""
    weights = []
    for idx in landmark_indices:
        idx = int(idx)
        if idx in _MP_EYE_INDICES:
            weights.append(_LANDMARK_REGION_WEIGHTS["eye"])
        elif idx in _MP_MOUTH_INDICES:
            weights.append(_LANDMARK_REGION_WEIGHTS["mouth"])
        elif idx in _MP_NOSE_INDICES:
            weights.append(_LANDMARK_REGION_WEIGHTS["nose"])
        elif idx in _MP_CONTOUR_INDICES:
            weights.append(_LANDMARK_REGION_WEIGHTS["contour"])
        elif idx in _MP_FOREHEAD_INDICES:
            weights.append(_LANDMARK_REGION_WEIGHTS["forehead"])
        else:
            weights.append(_LANDMARK_REGION_WEIGHTS["default"])
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FLAMEFitter:
    """Fits FLAME parameters to multi-view 2D landmark observations."""

    def __init__(self, flame_model, landmark_embedding: dict, device: str = "cuda"):
        """Set up the fitter.

        Args:
            flame_model: An instance of FLAMEModel.
            landmark_embedding: Dict from load_mediapipe_to_flame_mapping with
                'lmk_faces_idx', 'lmk_bary_coords', 'landmark_indices'.
            device: Torch device string.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.flame = flame_model.to(self.device)
        self.flame.eval()

        self.lmk_faces_idx = landmark_embedding["lmk_faces_idx"]
        self.lmk_bary_coords = landmark_embedding["lmk_bary_coords"]
        self.landmark_indices = landmark_embedding["landmark_indices"]

        # Pre-build per-landmark weights
        self._landmark_weights = _build_landmark_weights(self.landmark_indices, self.device)

        # Mixed-precision scaler (only used on CUDA)
        self._use_amp = self.device.type == "cuda"

    def fit(
        self,
        landmarks_2d_per_frame: list[np.ndarray],
        cameras: list[dict],
        num_shape_coeffs: int = 100,
        num_expr_coeffs: int = 50,
        num_iterations: int = 1000,
        lr: float = 0.01,
        lambda_temporal: float = 0.1,
    ) -> dict:
        """Optimize FLAME parameters against multi-view 2D landmarks.

        Args:
            landmarks_2d_per_frame: List of (478, 2) arrays — one per frame.
                Only the subset indicated by self.landmark_indices is used.
            cameras: List of camera dicts, each with keys:
                'K' (3x3 intrinsic), 'R' (3x3 rotation), 't' (3x1 translation),
                'dist' (optional distortion coeffs), 'width', 'height'.
            num_shape_coeffs: Number of shape PCs to optimize.
            num_expr_coeffs: Number of expression PCs to optimize.
            num_iterations: Total Adam iterations across all stages.
            lr: Base learning rate (used for stage 1; later stages use lower).
            lambda_temporal: Weight for temporal smoothness regularization
                when fitting multiple frames. Set to 0.0 to disable.

        Returns:
            Dict with optimized parameters and final loss.
        """
        num_frames = len(landmarks_2d_per_frame)
        device = self.device
        lmk_idx = self.landmark_indices

        # Prepare target landmarks: select the mapped subset
        targets = []
        for lm2d in landmarks_2d_per_frame:
            subset = lm2d[lmk_idx]  # (K, 2)
            targets.append(torch.tensor(subset, dtype=torch.float32, device=device))
        targets = torch.stack(targets)  # (F, K, 2)

        # Prepare camera matrices
        cam_K = []
        cam_R = []
        cam_t = []
        for cam in cameras:
            cam_K.append(torch.tensor(cam["K"], dtype=torch.float32, device=device).reshape(3, 3))
            cam_R.append(torch.tensor(cam["R"], dtype=torch.float32, device=device).reshape(3, 3))
            cam_t.append(torch.tensor(cam["t"], dtype=torch.float32, device=device).reshape(3, 1))

        # ── Optimizable parameters ─────────────────────────────────────
        shape_params = torch.zeros(1, num_shape_coeffs, device=device, requires_grad=True)
        expr_params = torch.zeros(1, num_expr_coeffs, device=device, requires_grad=True)
        jaw_pose = torch.zeros(1, 3, device=device, requires_grad=True)
        neck_pose = torch.zeros(1, 3, device=device, requires_grad=True)
        global_rot = torch.zeros(1, 3, device=device, requires_grad=True)
        translation = torch.zeros(1, 3, device=device, requires_grad=True)

        all_params = [shape_params, expr_params, jaw_pose, neck_pose, global_rot, translation]

        # For temporal smoothness: per-frame expression params (only when > 1 frame)
        per_frame_expr = None
        if num_frames > 1 and lambda_temporal > 0.0:
            per_frame_expr = [
                torch.zeros(1, num_expr_coeffs, device=device, requires_grad=True)
                for _ in range(num_frames)
            ]

        # ── Staged optimization with different LRs ──────────────────────
        # Stage 1 (0-20%): Rigid alignment only
        # Stage 2 (20-50%): + shape + neck pose
        # Stage 3 (50-100%): + expression + jaw pose (full)
        stage_boundary_1 = int(num_iterations * 0.2)
        stage_boundary_2 = int(num_iterations * 0.5)

        stage_configs = [
            {
                "end_iter": stage_boundary_1,
                "params": [global_rot, translation],
                "lr": lr,  # 0.01
                "label": "rigid",
            },
            {
                "end_iter": stage_boundary_2,
                "params": [global_rot, translation, shape_params, neck_pose],
                "lr": lr * 0.5,  # 0.005
                "label": "rigid+shape",
            },
            {
                "end_iter": num_iterations,
                "params": all_params + (per_frame_expr if per_frame_expr else []),
                "lr": lr * 0.1,  # 0.001
                "label": "full",
            },
        ]

        best_loss = float("inf")
        best_state = None
        initial_loss = None
        loss_history = []

        iter_count = 0
        for stage in stage_configs:
            stage_params = stage["params"]
            stage_lr = stage["lr"]
            stage_end = stage["end_iter"]
            stage_iters_total = stage_end - iter_count

            optimizer = torch.optim.Adam(stage_params, lr=stage_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(stage_iters_total, 1), eta_min=stage_lr * 0.1
            )

            logger.info(
                "Stage '%s': iters %d-%d, lr=%.4f, %d params groups",
                stage["label"], iter_count, stage_end, stage_lr, len(stage_params),
            )

            while iter_count < stage_end:
                optimizer.zero_grad()

                # Forward FLAME with optional mixed precision
                if self._use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        result = self.flame(
                            shape_params=shape_params,
                            expression_params=expr_params,
                            jaw_pose=jaw_pose,
                            neck_pose=neck_pose,
                            global_rotation=global_rot,
                            translation=translation,
                        )
                        vertices = result["vertices"]  # (1, V, 3)
                        lmk_3d = self.flame.get_landmarks(
                            vertices, self.lmk_faces_idx, self.lmk_bary_coords
                        )  # (1, K, 3)
                else:
                    result = self.flame(
                        shape_params=shape_params,
                        expression_params=expr_params,
                        jaw_pose=jaw_pose,
                        neck_pose=neck_pose,
                        global_rotation=global_rot,
                        translation=translation,
                    )
                    vertices = result["vertices"]
                    lmk_3d = self.flame.get_landmarks(
                        vertices, self.lmk_faces_idx, self.lmk_bary_coords
                    )

                # Projection loss across all frames (float32 precision)
                lmk_3d_f32 = lmk_3d.float()
                proj_loss = torch.tensor(0.0, device=device)
                for fi in range(num_frames):
                    proj_loss = proj_loss + self._projection_loss(
                        lmk_3d_f32[0], targets[fi], cam_K[fi], cam_R[fi], cam_t[fi]
                    )
                proj_loss = proj_loss / num_frames

                # Regularization
                reg_loss = self._regularization_loss(shape_params, expr_params)

                # Temporal smoothness (only when fitting a sequence with per-frame exprs)
                temporal_loss = torch.tensor(0.0, device=device)
                if per_frame_expr is not None and num_frames > 1:
                    for fi in range(1, num_frames):
                        temporal_loss = temporal_loss + torch.sum(
                            (per_frame_expr[fi] - per_frame_expr[fi - 1]) ** 2
                        )
                    temporal_loss = lambda_temporal * temporal_loss / (num_frames - 1)

                total_loss = proj_loss + reg_loss + temporal_loss
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                current_loss = total_loss.item()
                loss_history.append(current_loss)

                # Track initial loss for convergence check
                if initial_loss is None:
                    initial_loss = current_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_state = {
                        "shape_params": shape_params.detach().cpu().clone(),
                        "expression_params": expr_params.detach().cpu().clone(),
                        "jaw_pose": jaw_pose.detach().cpu().clone(),
                        "neck_pose": neck_pose.detach().cpu().clone(),
                        "global_rotation": global_rot.detach().cpu().clone(),
                        "translation": translation.detach().cpu().clone(),
                    }

                if iter_count % 100 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "Iter %4d [%s] | proj=%.6f reg=%.6f temp=%.6f total=%.6f lr=%.5f",
                        iter_count, stage["label"],
                        proj_loss.item(), reg_loss.item(), temporal_loss.item(),
                        total_loss.item(), current_lr,
                    )

                iter_count += 1

        logger.info("Fitting complete — best loss: %.6f", best_loss)

        # Convergence checks
        if initial_loss is not None and best_loss >= initial_loss:
            logger.warning(
                "FLAME fitting did not converge: initial_loss=%.6f, best_loss=%.6f",
                initial_loss, best_loss,
            )

        if best_loss > 100.0:
            logger.warning(
                "FLAME fitting final loss %.4f > 100 pixels mean error — result may be poor",
                best_loss,
            )

        # Final forward pass with best parameters
        with torch.no_grad():
            final_result = self.flame(
                shape_params=best_state["shape_params"].to(device),
                expression_params=best_state["expression_params"].to(device),
                jaw_pose=best_state["jaw_pose"].to(device),
                neck_pose=best_state["neck_pose"].to(device),
                global_rotation=best_state["global_rotation"].to(device),
                translation=best_state["translation"].to(device),
            )

        best_state["vertices"] = final_result["vertices"].cpu().numpy()
        best_state["faces"] = final_result["faces"].cpu().numpy()
        best_state["joints"] = final_result["joints"].cpu().numpy()
        best_state["loss"] = best_loss
        best_state["loss_history"] = loss_history
        best_state["converged"] = (initial_loss is not None and best_loss < initial_loss)

        # Convert tensors to numpy for serialization
        for key in ["shape_params", "expression_params", "jaw_pose", "neck_pose",
                     "global_rotation", "translation"]:
            best_state[key] = best_state[key].numpy()

        return best_state

    def _projection_loss(
        self,
        landmarks_3d: torch.Tensor,
        landmarks_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Project 3D landmarks through a camera and compute weighted Huber loss.

        Uses smooth L1 (Huber) loss instead of plain L2 to be robust against
        outlier landmarks (occluded, hair-covered, etc.).  Per-landmark weights
        emphasise eyes and mouth over jawline/forehead.

        Args:
            landmarks_3d: (K, 3) 3D landmark positions in world space.
            landmarks_2d: (K, 2) target 2D landmark positions in pixels.
            K: (3, 3) camera intrinsic matrix.
            R: (3, 3) camera rotation (world-to-camera).
            t: (3, 1) camera translation.

        Returns:
            Scalar weighted Huber reprojection loss.
        """
        # World to camera: p_cam = R @ p_world + t
        pts_cam = (R @ landmarks_3d.T + t).T  # (K, 3)

        # Perspective projection
        depth = pts_cam[:, 2:3].clamp(min=1e-6)
        pts_norm = pts_cam[:, :2] / depth  # (K, 2)

        # Apply intrinsics: u = fx * x + cx, v = fy * y + cy
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_2d = torch.stack([
            fx * pts_norm[:, 0] + cx,
            fy * pts_norm[:, 1] + cy,
        ], dim=1)  # (K, 2)

        # Weighted Huber (smooth L1) loss per landmark
        per_lmk_loss = F.smooth_l1_loss(
            pts_2d, landmarks_2d, beta=2.0, reduction="none"
        )  # (K, 2)
        per_lmk_loss = per_lmk_loss.sum(dim=1)  # (K,)

        # Apply per-landmark weights
        weighted_loss = per_lmk_loss * self._landmark_weights
        loss = weighted_loss.mean()
        return loss

    def _regularization_loss(
        self,
        shape_params: torch.Tensor,
        expression_params: torch.Tensor,
        shape_weight: float = 1e-4,
        expr_weight: float = 1e-3,
    ) -> torch.Tensor:
        """L2 regularization on FLAME parameters.

        Args:
            shape_params: (1, N_shape) shape coefficients.
            expression_params: (1, N_expr) expression coefficients.
            shape_weight: Regularization weight for shape.
            expr_weight: Regularization weight for expression.

        Returns:
            Scalar regularization loss.
        """
        shape_reg = shape_weight * torch.sum(shape_params ** 2)
        expr_reg = expr_weight * torch.sum(expression_params ** 2)
        return shape_reg + expr_reg


def fit_flame_to_sequence(
    frames_dir: Path,
    landmarks_dir: Path,
    colmap_model_dir: Path,
    flame_model_path: Path,
    embedding_path: Path,
    output_dir: Path,
) -> dict:
    """End-to-end FLAME fitting to a multi-view capture sequence.

    Loads landmarks, cameras, and the FLAME model, runs optimization,
    and saves the fitted parameters and mesh.

    Args:
        frames_dir: Directory containing input images.
        landmarks_dir: Directory containing per-frame landmark JSON files.
        colmap_model_dir: Path to COLMAP sparse model (cameras.bin, images.bin, points3D.bin).
        flame_model_path: Path to FLAME .pkl file.
        embedding_path: Path to mediapipe_landmark_embedding.npz.
        output_dir: Directory for fitted output.

    Returns:
        Dict of fitted FLAME parameters.
    """
    import struct

    frames_dir = Path(frames_dir)
    landmarks_dir = Path(landmarks_dir)
    colmap_model_dir = Path(colmap_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cameras from COLMAP ───────────────────────────────────────
    cameras, images = _load_colmap_model(colmap_model_dir)

    # ── Load landmarks ─────────────────────────────────────────────────
    landmark_files = sorted(landmarks_dir.glob("*.json"))
    landmarks_2d_per_frame = []
    camera_list = []

    for lm_file in landmark_files:
        try:
            with open(lm_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to read landmark file %s: %s, skipping", lm_file, e)
            continue

        if not data.get("detected", False):
            continue

        # Handle null/empty landmark data
        lm2d_raw = data.get("landmarks_2d")
        if lm2d_raw is None or len(lm2d_raw) == 0:
            logger.warning("Landmark file %s has no landmark data, skipping", lm_file.name)
            continue

        lm2d = np.array(lm2d_raw, dtype=np.float32)
        if lm2d.ndim != 2 or lm2d.shape[1] < 2:
            logger.warning("Landmark file %s has invalid shape %s, skipping", lm_file.name, lm2d.shape)
            continue

        landmarks_2d_per_frame.append(lm2d)

        # Match to COLMAP image by filename
        img_name = data["image"]
        if img_name in images:
            cam_info = images[img_name]
            cam_id = cam_info["camera_id"]
            cam = cameras[cam_id]

            # Build camera dict
            R = cam_info["rotation_matrix"]
            t = cam_info["translation"]
            K = np.array([
                [cam["fx"], 0, cam["cx"]],
                [0, cam["fy"], cam["cy"]],
                [0, 0, 1],
            ], dtype=np.float64)

            camera_list.append({
                "K": K.tolist(),
                "R": R.tolist(),
                "t": t.tolist(),
                "width": cam["width"],
                "height": cam["height"],
            })
        else:
            logger.warning("Image %s not found in COLMAP model — skipping", img_name)

    if not landmarks_2d_per_frame:
        logger.error("No valid landmark detections found in %s", landmarks_dir)
        raise RuntimeError("No valid landmark detections found")

    # Trim to matched pairs
    min_len = min(len(landmarks_2d_per_frame), len(camera_list))
    landmarks_2d_per_frame = landmarks_2d_per_frame[:min_len]
    camera_list = camera_list[:min_len]

    logger.info("Fitting FLAME to %d views ...", len(camera_list))

    # ── Load FLAME model and embedding ─────────────────────────────────
    from reconstruction.flame_model import FLAMEModel
    from reconstruction.landmarks import load_mediapipe_to_flame_mapping

    flame = FLAMEModel(flame_model_path)
    embedding = load_mediapipe_to_flame_mapping(embedding_path)

    fitter = FLAMEFitter(flame, embedding)
    result = fitter.fit(landmarks_2d_per_frame, camera_list)

    # ── Save outputs ───────────────────────────────────────────────────
    # Save parameters
    params_path = output_dir / "flame_params.npz"
    np.savez(
        str(params_path),
        shape_params=result["shape_params"],
        expression_params=result["expression_params"],
        jaw_pose=result["jaw_pose"],
        neck_pose=result["neck_pose"],
        global_rotation=result["global_rotation"],
        translation=result["translation"],
    )
    logger.info("Saved FLAME parameters to %s", params_path)

    # Save mesh as OBJ
    mesh_path = output_dir / "fitted_mesh.obj"
    _save_obj(mesh_path, result["vertices"][0], result["faces"])
    logger.info("Saved fitted mesh to %s", mesh_path)

    # Save convergence data (loss vs iteration)
    if "loss_history" in result:
        convergence_path = output_dir / "convergence.json"
        convergence_data = {
            "loss_history": result["loss_history"],
            "final_loss": result["loss"],
            "converged": result.get("converged", True),
            "num_iterations": len(result["loss_history"]),
            "num_views": len(camera_list),
        }
        with open(convergence_path, "w", encoding="utf-8") as fh:
            json.dump(convergence_data, fh)
        logger.info("Saved convergence data to %s", convergence_path)

    return result


# ── COLMAP binary model reader ──────────────────────────────────────────────


def _load_colmap_model(model_dir: Path) -> tuple[dict, dict]:
    """Load cameras and images from a COLMAP binary model.

    Returns:
        Tuple of (cameras_dict, images_dict).
        cameras_dict: cam_id -> {width, height, fx, fy, cx, cy, ...}
        images_dict: image_name -> {camera_id, rotation_matrix, translation}
    """
    model_dir = Path(model_dir)
    cameras = _read_cameras_binary(model_dir / "cameras.bin")
    images = _read_images_binary(model_dir / "images.bin")
    return cameras, images


def _read_cameras_binary(path: Path) -> dict:
    """Read cameras.bin from a COLMAP model."""
    import struct

    cameras = {}
    with open(path, "rb") as fh:
        num_cameras = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", fh.read(4))[0]
            model_id = struct.unpack("<i", fh.read(4))[0]
            width = struct.unpack("<Q", fh.read(8))[0]
            height = struct.unpack("<Q", fh.read(8))[0]

            # Number of parameters depends on camera model
            num_params_map = {
                0: 3,   # SIMPLE_PINHOLE: f, cx, cy
                1: 4,   # PINHOLE: fx, fy, cx, cy
                2: 4,   # SIMPLE_RADIAL: f, cx, cy, k1
                3: 5,   # RADIAL: f, cx, cy, k1, k2
                4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                5: 12,  # OPENCV_FISHEYE
                6: 5,   # FULL_OPENCV (first 5)
            }
            num_params = num_params_map.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", fh.read(8 * num_params))

            if model_id == 0:  # SIMPLE_PINHOLE
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1:  # PINHOLE
                fx, fy, cx, cy = params[:4]
            elif model_id in (2, 3):  # SIMPLE_RADIAL, RADIAL
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:  # OPENCV and others
                fx, fy, cx, cy = params[:4]

            cameras[cam_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "params": list(params),
            }

    return cameras


def _read_images_binary(path: Path) -> dict:
    """Read images.bin from a COLMAP model."""
    import struct

    images = {}
    with open(path, "rb") as fh:
        num_images = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", fh.read(4))[0]

            # Quaternion (w, x, y, z)
            qw, qx, qy, qz = struct.unpack("<4d", fh.read(32))
            # Translation
            tx, ty, tz = struct.unpack("<3d", fh.read(24))
            # Camera ID
            camera_id = struct.unpack("<I", fh.read(4))[0]

            # Image name (null-terminated string)
            name_chars = []
            while True:
                ch = fh.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            name = "".join(name_chars)

            # 2D points (skip)
            num_points2d = struct.unpack("<Q", fh.read(8))[0]
            # Each point: x, y (2 doubles) + point3d_id (1 long long)
            fh.read(num_points2d * 24)

            # Convert quaternion to rotation matrix
            R = _quat_to_rot(qw, qx, qy, qz)

            images[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "rotation_matrix": R.tolist(),
                "translation": [[tx], [ty], [tz]],
                "qvec": [qw, qx, qy, qz],
            }

    return images


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z,   2*y*z - 2*w*x],
        [2*x*z - 2*w*y,       2*y*z + 2*w*x,       1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)
    return R


def _save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a mesh to Wavefront OBJ format."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# FLAME fitted mesh\n")
        for v in vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            # OBJ is 1-indexed
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
