"""Multi-view FLAME model fitting via differentiable landmark reprojection.

Optimizes FLAME shape, expression, and pose parameters to minimize the
reprojection error of facial landmarks across multiple calibrated views.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.timing import timed

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

# MediaPipe indices for left/right pupils (iris center landmarks)
_MP_LEFT_PUPIL = 468   # Left iris center
_MP_RIGHT_PUPIL = 473  # Right iris center
# Average adult inter-pupillary distance in meters
_AVERAGE_IPD_METERS = 0.063


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


def _estimate_face_yaw_from_landmarks(landmarks_2d: np.ndarray, image_width: int) -> float:
    """Estimate face yaw angle from 2D landmarks.

    Uses the nose tip (index 1) and left/right face contour landmarks to
    estimate how frontal the face is.  Returns absolute yaw in degrees
    (0 = frontal, 90 = profile).

    Args:
        landmarks_2d: (478, 2) or (N, 2) landmark array.
        image_width: Width of the source image in pixels.

    Returns:
        Estimated absolute yaw angle in degrees.
    """
    if landmarks_2d.shape[0] < 400:
        return 90.0  # Not enough landmarks to estimate

    nose_tip = landmarks_2d[1]
    # Left and right ear-region landmarks
    left_ear = landmarks_2d[234]
    right_ear = landmarks_2d[454]

    face_center_x = (left_ear[0] + right_ear[0]) / 2.0
    face_width = abs(right_ear[0] - left_ear[0])

    if face_width < 1.0:
        return 90.0

    # Asymmetry ratio: how far the nose is from center relative to face width
    asymmetry = (nose_tip[0] - face_center_x) / face_width
    # Map to approximate yaw: asymmetry of 0 = frontal, 0.5 = ~90 degrees
    yaw_deg = abs(asymmetry) * 180.0
    return min(yaw_deg, 90.0)


def _estimate_scale_from_ipd(
    landmarks_2d: np.ndarray,
    camera_K: np.ndarray,
    depth_at_face: float = 0.5,
) -> float:
    """Estimate world-scale correction factor from inter-pupillary distance.

    The FLAME model is in meters.  Given the detected IPD in pixels and the
    camera intrinsics, we can estimate the depth-to-metric scale so that the
    reconstructed IPD matches the average human IPD of ~6.3 cm.

    Args:
        landmarks_2d: (478, 2) landmark array with pupil landmarks.
        camera_K: (3, 3) camera intrinsic matrix.
        depth_at_face: Estimated depth to the face in the reconstruction
            coordinate system (used as initial guess).

    Returns:
        Scale factor to multiply translation by, or 1.0 if estimation fails.
    """
    if landmarks_2d.shape[0] <= max(_MP_LEFT_PUPIL, _MP_RIGHT_PUPIL):
        return 1.0

    left_pupil = landmarks_2d[_MP_LEFT_PUPIL]
    right_pupil = landmarks_2d[_MP_RIGHT_PUPIL]
    ipd_pixels = np.linalg.norm(left_pupil - right_pupil)

    if ipd_pixels < 5.0:
        return 1.0  # Too small to be reliable

    fx = camera_K[0, 0]
    if fx < 1.0:
        return 1.0

    # IPD in world units at the given depth: ipd_world = ipd_pixels * depth / fx
    ipd_world = ipd_pixels * depth_at_face / fx

    if ipd_world < 1e-6:
        return 1.0

    scale = _AVERAGE_IPD_METERS / ipd_world
    return float(scale)


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
        frame_confidence_weights: Optional[list[float]] = None,
    ) -> dict:
        """Optimize FLAME parameters against multi-view 2D landmarks.

        Args:
            landmarks_2d_per_frame: List of (478, 2) arrays -- one per frame.
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
            frame_confidence_weights: Optional per-frame confidence weights
                in [0, 1].  Higher weight means more influence on the fit.
                If None, all frames are weighted equally.

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

        # Per-frame confidence weights
        if frame_confidence_weights is not None:
            conf_weights = torch.tensor(
                frame_confidence_weights, dtype=torch.float32, device=device
            )
            # Normalize so they sum to num_frames (preserves loss scale)
            conf_weights = conf_weights * num_frames / conf_weights.sum().clamp(min=1e-6)
        else:
            conf_weights = torch.ones(num_frames, dtype=torch.float32, device=device)

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
        # For early stopping on plateau
        _plateau_window = 100
        _plateau_threshold = 0.001  # 0.1% improvement

        iter_count = 0
        early_stopped = False
        for stage in stage_configs:
            if early_stopped:
                break
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
                    frame_loss = self._projection_loss(
                        lmk_3d_f32[0], targets[fi], cam_K[fi], cam_R[fi], cam_t[fi]
                    )
                    proj_loss = proj_loss + frame_loss * conf_weights[fi]
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
                    # Compute trend indicator
                    trend = ""
                    if len(loss_history) >= _plateau_window + 1:
                        prev = loss_history[-_plateau_window - 1]
                        if prev > 0:
                            pct_change = (prev - current_loss) / prev * 100
                            if pct_change > 1.0:
                                trend = " [decreasing]"
                            elif pct_change > 0.0:
                                trend = " [slow]"
                            else:
                                trend = " [plateau]"
                    logger.info(
                        "Iter %4d [%s] | proj=%.6f reg=%.6f temp=%.6f total=%.6f lr=%.5f%s",
                        iter_count, stage["label"],
                        proj_loss.item(), reg_loss.item(), temporal_loss.item(),
                        total_loss.item(), current_lr, trend,
                    )

                # Early stopping: check for plateau after we have enough history
                if (iter_count > 0 and iter_count % _plateau_window == 0
                        and len(loss_history) >= _plateau_window + 1
                        and stage["label"] == "full"):
                    window_start_loss = loss_history[-_plateau_window - 1]
                    if window_start_loss > 0:
                        improvement = (window_start_loss - current_loss) / window_start_loss
                        if improvement < _plateau_threshold:
                            logger.info(
                                "Early stopping at iter %d: improvement %.4f%% < %.1f%% "
                                "over last %d iters",
                                iter_count, improvement * 100,
                                _plateau_threshold * 100, _plateau_window,
                            )
                            early_stopped = True
                            break

                iter_count += 1

        logger.info("Fitting complete -- best loss: %.6f (after %d iters)", best_loss, iter_count)

        # Convergence checks
        if initial_loss is not None and best_loss >= initial_loss:
            logger.warning(
                "FLAME fitting did not converge: initial_loss=%.6f, best_loss=%.6f",
                initial_loss, best_loss,
            )

        if best_loss > 100.0:
            logger.warning(
                "FLAME fitting final loss %.4f > 100 pixels mean error -- result may be poor",
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
        best_state["early_stopped"] = early_stopped

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


# ── Joint FLAME optimization (GaussianSwap-inspired) ────────────────────────


def joint_optimize_flame(
    per_frame_params: list[dict],
    landmarks_per_frame: list[np.ndarray],
    cameras: list[dict],
    flame_model,
    landmark_embedding: dict,
    num_keyframes: int = 10,
    joint_iterations: int = 200,
    temporal_weight: float = 0.1,
    device: str = "cuda",
) -> list[dict]:
    """Joint FLAME optimization across keyframes for temporal consistency.

    Approach (inspired by GaussianSwap, ICLR 2026):
    1. Select keyframes (uniformly sampled + high-confidence frames)
    2. Share shape parameters across all frames (identity is constant)
    3. Optimize shared shape + per-frame expression/pose jointly
    4. Add temporal smoothness loss between adjacent frames
    5. Propagate optimized shape back to all frames

    Loss:
    L = L_landmark + lambda_shape * L_shape_reg + lambda_temporal * L_temporal

    Where L_temporal = sum ||expr_t - expr_{t-1}||^2 + ||pose_t - pose_{t-1}||^2

    Args:
        per_frame_params: Output from per-frame fitting (list of dicts with
            'shape_params', 'expression_params', 'jaw_pose', 'neck_pose',
            'global_rotation', 'translation', 'loss').
        landmarks_per_frame: List of (478, 2) landmark arrays per frame.
        cameras: List of camera dicts with 'K', 'R', 't'.
        flame_model: An instance of FLAMEModel.
        landmark_embedding: Dict with 'lmk_faces_idx', 'lmk_bary_coords',
            'landmark_indices'.
        num_keyframes: Number of keyframes to sample for joint optimization.
        joint_iterations: Number of Adam iterations for joint stage.
        temporal_weight: Weight for temporal smoothness loss term.
        device: Torch device string.

    Returns:
        Updated per_frame_params list with temporally consistent parameters.
        The shared shape is propagated to all frames.
    """
    num_frames = len(per_frame_params)
    if num_frames < 3:
        logger.info("Joint optimization skipped: only %d frames (need >= 3)", num_frames)
        return per_frame_params

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    lmk_idx = landmark_embedding["landmark_indices"]
    lmk_faces_idx = landmark_embedding["lmk_faces_idx"]
    lmk_bary_coords = landmark_embedding["lmk_bary_coords"]
    lmk_weights = _build_landmark_weights(lmk_idx, dev)

    # ── 1. Select keyframes ──────────────────────────────────────────
    num_kf = min(num_keyframes, num_frames)

    # Uniform sampling indices
    uniform_indices = set(np.linspace(0, num_frames - 1, num_kf, dtype=int).tolist())

    # Add high-confidence frames (lowest per-frame loss)
    frame_losses = [p.get("loss", float("inf")) for p in per_frame_params]
    sorted_by_loss = np.argsort(frame_losses)
    # Add top-confidence frames until we reach num_kf
    for idx in sorted_by_loss:
        if len(uniform_indices) >= num_kf:
            break
        uniform_indices.add(int(idx))

    keyframe_indices = sorted(uniform_indices)
    logger.info(
        "Joint optimization: %d keyframes selected from %d total frames",
        len(keyframe_indices), num_frames,
    )

    # ── 2. Initialize optimizable parameters ─────────────────────────
    # Shared shape: average of per-frame shapes (identity should be constant)
    shape_arrays = [p["shape_params"].flatten() for p in per_frame_params]
    avg_shape = np.mean(shape_arrays, axis=0)
    len(avg_shape)

    shared_shape = torch.tensor(
        avg_shape, dtype=torch.float32, device=dev
    ).unsqueeze(0).requires_grad_(True)  # (1, N_shape)

    # Per-keyframe expression, jaw_pose, neck_pose, global_rot, translation
    kf_expr = []
    kf_jaw = []
    kf_neck = []
    kf_rot = []
    kf_trans = []
    for ki in keyframe_indices:
        p = per_frame_params[ki]
        kf_expr.append(torch.tensor(
            p["expression_params"].flatten(), dtype=torch.float32, device=dev
        ).unsqueeze(0).requires_grad_(True))
        kf_jaw.append(torch.tensor(
            p["jaw_pose"].flatten(), dtype=torch.float32, device=dev
        ).unsqueeze(0).requires_grad_(True))
        kf_neck.append(torch.tensor(
            p["neck_pose"].flatten(), dtype=torch.float32, device=dev
        ).unsqueeze(0).requires_grad_(True))
        kf_rot.append(torch.tensor(
            p["global_rotation"].flatten(), dtype=torch.float32, device=dev
        ).unsqueeze(0).requires_grad_(True))
        kf_trans.append(torch.tensor(
            p["translation"].flatten(), dtype=torch.float32, device=dev
        ).unsqueeze(0).requires_grad_(True))

    # Prepare target landmarks and cameras for keyframes
    kf_targets = []
    kf_K = []
    kf_R = []
    kf_t = []
    for ki in keyframe_indices:
        lm2d = landmarks_per_frame[ki][lmk_idx]
        kf_targets.append(torch.tensor(lm2d, dtype=torch.float32, device=dev))
        cam = cameras[ki]
        kf_K.append(torch.tensor(cam["K"], dtype=torch.float32, device=dev).reshape(3, 3))
        kf_R.append(torch.tensor(cam["R"], dtype=torch.float32, device=dev).reshape(3, 3))
        kf_t.append(torch.tensor(cam["t"], dtype=torch.float32, device=dev).reshape(3, 1))

    # ── 3. Build optimizer ───────────────────────────────────────────
    all_optim_params = [shared_shape]
    for i in range(len(keyframe_indices)):
        all_optim_params.extend([kf_expr[i], kf_jaw[i], kf_neck[i], kf_rot[i], kf_trans[i]])

    optimizer = torch.optim.Adam(all_optim_params, lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=joint_iterations, eta_min=0.0005
    )

    flame = flame_model.to(dev)
    flame.eval()

    # Record pre-joint loss for comparison
    pre_joint_losses = [per_frame_params[ki].get("loss", float("inf")) for ki in keyframe_indices]

    # ── 4. Joint optimization loop ───────────────────────────────────
    best_loss = float("inf")
    best_state = None
    loss_history = []

    n_kf = len(keyframe_indices)

    for it in range(joint_iterations):
        optimizer.zero_grad()

        total_proj_loss = torch.tensor(0.0, device=dev)

        for i in range(n_kf):
            result = flame(
                shape_params=shared_shape,
                expression_params=kf_expr[i],
                jaw_pose=kf_jaw[i],
                neck_pose=kf_neck[i],
                global_rotation=kf_rot[i],
                translation=kf_trans[i],
            )
            vertices = result["vertices"]
            lmk_3d = flame.get_landmarks(vertices, lmk_faces_idx, lmk_bary_coords)
            lmk_3d = lmk_3d.float()[0]  # (K, 3)

            # Projection loss
            pts_cam = (kf_R[i] @ lmk_3d.T + kf_t[i]).T
            depth = pts_cam[:, 2:3].clamp(min=1e-6)
            pts_norm = pts_cam[:, :2] / depth
            fx, fy = kf_K[i][0, 0], kf_K[i][1, 1]
            cx, cy = kf_K[i][0, 2], kf_K[i][1, 2]
            pts_2d = torch.stack([
                fx * pts_norm[:, 0] + cx,
                fy * pts_norm[:, 1] + cy,
            ], dim=1)

            per_lmk = F.smooth_l1_loss(pts_2d, kf_targets[i], beta=2.0, reduction="none").sum(dim=1)
            total_proj_loss = total_proj_loss + (per_lmk * lmk_weights).mean()

        total_proj_loss = total_proj_loss / n_kf

        # Shape regularization
        shape_reg = 1e-4 * torch.sum(shared_shape ** 2)

        # Expression regularization
        expr_reg = torch.tensor(0.0, device=dev)
        for i in range(n_kf):
            expr_reg = expr_reg + torch.sum(kf_expr[i] ** 2)
        expr_reg = 1e-3 * expr_reg / n_kf

        # Temporal smoothness loss (adjacent keyframes)
        temporal_loss = torch.tensor(0.0, device=dev)
        if n_kf > 1 and temporal_weight > 0:
            for i in range(1, n_kf):
                temporal_loss = temporal_loss + torch.sum((kf_expr[i] - kf_expr[i - 1]) ** 2)
                temporal_loss = temporal_loss + torch.sum((kf_jaw[i] - kf_jaw[i - 1]) ** 2)
                temporal_loss = temporal_loss + torch.sum((kf_rot[i] - kf_rot[i - 1]) ** 2)
            temporal_loss = temporal_weight * temporal_loss / (n_kf - 1)

        total_loss = total_proj_loss + shape_reg + expr_reg + temporal_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = total_loss.item()
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                "shared_shape": shared_shape.detach().cpu().clone(),
                "kf_expr": [e.detach().cpu().clone() for e in kf_expr],
                "kf_jaw": [j.detach().cpu().clone() for j in kf_jaw],
                "kf_neck": [n.detach().cpu().clone() for n in kf_neck],
                "kf_rot": [r.detach().cpu().clone() for r in kf_rot],
                "kf_trans": [t.detach().cpu().clone() for t in kf_trans],
            }

        if it % 50 == 0:
            logger.info(
                "Joint iter %3d | proj=%.6f shape_reg=%.6f expr_reg=%.6f temp=%.6f total=%.6f",
                it, total_proj_loss.item(), shape_reg.item(), expr_reg.item(),
                temporal_loss.item(), current_loss,
            )

    # ── 5. Propagate results back to all frames ──────────────────────
    if best_state is None:
        logger.warning("Joint optimization produced no improvement, keeping per-frame results")
        return per_frame_params

    optimized_shape = best_state["shared_shape"].numpy()

    # Update keyframe params
    for i, ki in enumerate(keyframe_indices):
        per_frame_params[ki]["shape_params"] = optimized_shape.copy()
        per_frame_params[ki]["expression_params"] = best_state["kf_expr"][i].numpy()
        per_frame_params[ki]["jaw_pose"] = best_state["kf_jaw"][i].numpy()
        per_frame_params[ki]["neck_pose"] = best_state["kf_neck"][i].numpy()
        per_frame_params[ki]["global_rotation"] = best_state["kf_rot"][i].numpy()
        per_frame_params[ki]["translation"] = best_state["kf_trans"][i].numpy()

    # Propagate shared shape to non-keyframes
    for fi in range(num_frames):
        if fi not in keyframe_indices:
            per_frame_params[fi]["shape_params"] = optimized_shape.copy()

    # Interpolate expression/pose for non-keyframes between keyframes
    for fi in range(num_frames):
        if fi in keyframe_indices:
            continue
        # Find bracketing keyframes
        left_ki = None
        right_ki = None
        for ki in keyframe_indices:
            if ki <= fi:
                left_ki = ki
            if ki >= fi and right_ki is None:
                right_ki = ki
        if left_ki is None:
            left_ki = keyframe_indices[0]
        if right_ki is None:
            right_ki = keyframe_indices[-1]
        if left_ki == right_ki:
            # Frame is before first or after last keyframe; copy nearest
            continue
        # Linear interpolation factor
        t_interp = (fi - left_ki) / max(right_ki - left_ki, 1)
        keyframe_indices.index(left_ki)
        keyframe_indices.index(right_ki)
        for param_key in ["expression_params", "jaw_pose"]:
            left_val = per_frame_params[left_ki][param_key]
            right_val = per_frame_params[right_ki][param_key]
            per_frame_params[fi][param_key] = (
                left_val * (1.0 - t_interp) + right_val * t_interp
            )

    # Log improvement
    post_joint_losses = []
    for ki in keyframe_indices:
        post_joint_losses.append(per_frame_params[ki].get("loss", float("inf")))
    pre_mean = np.mean([v for v in pre_joint_losses if np.isfinite(v)]) if pre_joint_losses else 0
    logger.info(
        "Joint optimization complete: %d iterations, best_loss=%.6f, "
        "pre-joint avg keyframe loss=%.4f, shape consistency enforced across %d frames",
        joint_iterations, best_loss, pre_mean, num_frames,
    )

    return per_frame_params


# ── Frame matching utilities ─────────────────────────────────────────────────


def _normalize_frame_stem(name: str) -> str:
    """Extract a canonical stem from a frame filename for matching.

    Strips directory paths, extensions, and normalizes common naming patterns
    so that e.g. 'frames_srgb/frame_000001.png' matches 'frame_000001.json'.

    Args:
        name: Filename or path string.

    Returns:
        Lowercased stem string for comparison.
    """
    stem = Path(name).stem.lower()
    return stem


def _match_landmarks_to_cameras(
    landmark_files: list[Path],
    colmap_images: dict,
) -> list[dict]:
    """Match landmark JSON files to COLMAP camera entries.

    The matching is done by normalised filename stem to handle differences
    in path prefixes or extensions between the landmark JSONs and the COLMAP
    images.bin entries.

    Args:
        landmark_files: Sorted list of landmark JSON paths.
        colmap_images: Dict from _read_images_binary (image_name -> info).

    Returns:
        List of dicts, each with keys:
            'landmark_file': Path to the landmark JSON.
            'image_name': Matched COLMAP image name.
            'colmap_info': The COLMAP image dict.
    """
    # Build a lookup from normalised stem to COLMAP image name + info
    stem_to_colmap = {}
    for img_name, img_info in colmap_images.items():
        stem = _normalize_frame_stem(img_name)
        stem_to_colmap[stem] = (img_name, img_info)

    matched = []
    for lm_file in landmark_files:
        # Try matching by landmark file stem first
        lm_stem = _normalize_frame_stem(lm_file.name)
        if lm_stem in stem_to_colmap:
            cname, cinfo = stem_to_colmap[lm_stem]
            matched.append({
                "landmark_file": lm_file,
                "image_name": cname,
                "colmap_info": cinfo,
            })
            continue

        # Try reading the JSON to get the stored image name
        try:
            with open(lm_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            img_name_from_json = data.get("image", "")
            json_stem = _normalize_frame_stem(img_name_from_json)
            if json_stem in stem_to_colmap:
                cname, cinfo = stem_to_colmap[json_stem]
                matched.append({
                    "landmark_file": lm_file,
                    "image_name": cname,
                    "colmap_info": cinfo,
                })
        except (json.JSONDecodeError, IOError):
            continue

    return matched


def _select_frontal_frame(
    landmark_data_list: list[dict],
) -> Optional[int]:
    """Pick the frame index with the most frontal face view.

    Args:
        landmark_data_list: List of dicts each with 'landmarks_2d' (np array)
            and 'image_width' (int).

    Returns:
        Index into landmark_data_list of the most frontal frame, or None
        if the list is empty.
    """
    if not landmark_data_list:
        return None

    best_idx = 0
    best_yaw = 90.0
    for i, entry in enumerate(landmark_data_list):
        yaw = _estimate_face_yaw_from_landmarks(
            entry["landmarks_2d"], entry["image_width"]
        )
        if yaw < best_yaw:
            best_yaw = yaw
            best_idx = i

    logger.info("Most frontal frame: index %d (estimated yaw %.1f deg)", best_idx, best_yaw)
    return best_idx


def _render_flame_overlay(
    vertices_3d: np.ndarray,
    faces: np.ndarray,
    camera: dict,
    image_path: Path,
    output_path: Path,
) -> bool:
    """Render FLAME mesh wireframe projected onto an image and save as PNG.

    Args:
        vertices_3d: (V, 3) FLAME mesh vertices in world coordinates.
        faces: (F, 3) triangle indices.
        camera: Camera dict with 'K', 'R', 't', 'width', 'height'.
        image_path: Path to the source image to overlay on.
        output_path: Path to save the overlay PNG.

    Returns:
        True if the overlay was saved successfully.
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.debug("Cannot read image for overlay: %s", image_path)
            return False

        K = np.array(camera["K"], dtype=np.float64).reshape(3, 3)
        R = np.array(camera["R"], dtype=np.float64).reshape(3, 3)
        t = np.array(camera["t"], dtype=np.float64).reshape(3, 1)

        # Project vertices: p_cam = R @ p_world + t
        pts_cam = (R @ vertices_3d.T + t).T  # (V, 3)
        depth = pts_cam[:, 2:3]
        depth = np.clip(depth, 1e-6, None)
        pts_norm = pts_cam[:, :2] / depth

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_2d = np.column_stack([
            fx * pts_norm[:, 0] + cx,
            fy * pts_norm[:, 1] + cy,
        ]).astype(np.int32)

        # Draw wireframe edges from face triangles
        overlay = image.copy()
        # Only draw a subset of edges to keep it readable
        edge_set = set()
        for f in faces:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                edge = (min(a, b), max(a, b))
                if edge not in edge_set:
                    edge_set.add(edge)

        h, w = image.shape[:2]
        for a, b in edge_set:
            p1 = tuple(pts_2d[a])
            p2 = tuple(pts_2d[b])
            # Skip edges that project outside the image
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                    0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(overlay, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)

        # Blend with original
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        logger.info("Saved FLAME overlay to %s", output_path)
        return True

    except Exception as e:
        logger.warning("Failed to render FLAME overlay: %s", e)
        return False


@timed
def fit_flame_to_sequence(
    frames_dir: Path,
    landmarks_dir: Path,
    colmap_model_dir: Path,
    flame_model_path: Path,
    embedding_path: Path,
    output_dir: Path,
    fitting_config: Optional[dict] = None,
) -> dict:
    """End-to-end FLAME fitting to a multi-view capture sequence.

    Loads landmarks, cameras, and the FLAME model, runs optimization,
    and saves the fitted parameters and mesh.

    The function is robust to partial data: it normalizes filenames for
    matching and requires a minimum of 3 frames with both landmark
    detections and camera parameters.

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
    frames_dir = Path(frames_dir)
    landmarks_dir = Path(landmarks_dir)
    colmap_model_dir = Path(colmap_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    MIN_MATCHED_FRAMES = 3

    # ── Load cameras from COLMAP ───────────────────────────────────────
    cameras_bin_path = colmap_model_dir / "cameras.bin"
    images_bin_path = colmap_model_dir / "images.bin"

    if not cameras_bin_path.exists() or not images_bin_path.exists():
        raise RuntimeError(
            f"COLMAP model not found at {colmap_model_dir} "
            f"(cameras.bin exists={cameras_bin_path.exists()}, "
            f"images.bin exists={images_bin_path.exists()})"
        )

    cameras, images = _load_colmap_model(colmap_model_dir)
    logger.info(
        "Loaded COLMAP model: %d cameras, %d images",
        len(cameras), len(images),
    )

    # ── Load and match landmarks to cameras ────────────────────────────
    landmark_files = sorted(landmarks_dir.glob("*.json"))
    if not landmark_files:
        raise RuntimeError(f"No landmark JSON files found in {landmarks_dir}")

    logger.info("Found %d landmark files in %s", len(landmark_files), landmarks_dir)

    # First pass: build matched pairs using stem-based matching
    matches = _match_landmarks_to_cameras(landmark_files, images)
    logger.info(
        "Stem-based matching: %d/%d landmark files matched to COLMAP images",
        len(matches), len(landmark_files),
    )

    # Second pass: load landmark data for matched frames only, filter bad ones
    landmarks_2d_per_frame = []
    camera_list = []
    frame_confidence_weights = []
    frame_image_names = []  # Track for visualization
    landmark_data_for_frontal = []  # For frontal frame selection

    for match in matches:
        lm_file = match["landmark_file"]
        colmap_info = match["colmap_info"]

        try:
            with open(lm_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to read landmark file %s: %s, skipping", lm_file, e)
            continue

        # Skip frames without detections
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

        # Build camera dict from COLMAP data
        cam_id = colmap_info["camera_id"]
        if cam_id not in cameras:
            logger.warning("Camera ID %d not found in cameras.bin, skipping %s", cam_id, lm_file.name)
            continue

        cam = cameras[cam_id]
        R = colmap_info["rotation_matrix"]
        t = colmap_info["translation"]
        K = np.array([
            [cam["fx"], 0, cam["cx"]],
            [0, cam["fy"], cam["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)

        cam_dict = {
            "K": K.tolist(),
            "R": R if isinstance(R, list) else R.tolist(),
            "t": t if isinstance(t, list) else t.tolist(),
            "width": cam["width"],
            "height": cam["height"],
        }

        # Get confidence/quality weight for this frame
        quality = data.get("quality_score", data.get("confidence", 1.0))

        landmarks_2d_per_frame.append(lm2d)
        camera_list.append(cam_dict)
        frame_confidence_weights.append(float(quality))
        frame_image_names.append(data.get("image", lm_file.stem + ".png"))

        landmark_data_for_frontal.append({
            "landmarks_2d": lm2d,
            "image_width": data.get("image_width", cam["width"]),
        })

    # ── Check minimum frame count ──────────────────────────────────────
    num_matched = len(landmarks_2d_per_frame)
    logger.info(
        "Found %d frames with both landmarks and camera params out of %d landmark files "
        "and %d COLMAP images",
        num_matched, len(landmark_files), len(images),
    )

    if num_matched < MIN_MATCHED_FRAMES:
        raise RuntimeError(
            f"Only {num_matched} frames have both landmarks and camera params "
            f"(minimum {MIN_MATCHED_FRAMES} required). "
            f"Check that landmark detection (stage 9) and camera estimation "
            f"(stage 6/COLMAP) produced results for overlapping frames."
        )

    # ── Estimate scale from inter-pupillary distance ───────────────────
    # Pick the most frontal frame for scale estimation
    frontal_idx = _select_frontal_frame(landmark_data_for_frontal)
    if frontal_idx is not None:
        frontal_lm = landmarks_2d_per_frame[frontal_idx]
        frontal_K = np.array(camera_list[frontal_idx]["K"], dtype=np.float64).reshape(3, 3)
        scale_factor = _estimate_scale_from_ipd(frontal_lm, frontal_K, depth_at_face=0.5)
        if 0.1 < scale_factor < 10.0 and abs(scale_factor - 1.0) > 0.05:
            logger.info(
                "IPD-based scale estimation: factor=%.3f (will adjust FLAME translation)",
                scale_factor,
            )
        else:
            scale_factor = 1.0
            logger.info("IPD-based scale estimation: factor=%.3f (within tolerance, not adjusting)", scale_factor)
    else:
        scale_factor = 1.0

    # ── Load FLAME model and embedding ─────────────────────────────────
    from reconstruction.flame_model import FLAMEModel
    from reconstruction.landmarks import load_mediapipe_to_flame_mapping

    flame = FLAMEModel(flame_model_path)
    embedding = load_mediapipe_to_flame_mapping(embedding_path)

    fitter = FLAMEFitter(flame, embedding)
    result = fitter.fit(
        landmarks_2d_per_frame,
        camera_list,
        frame_confidence_weights=frame_confidence_weights,
    )

    # ── Joint FLAME optimization (GaussianSwap-inspired) ──────────────
    # Parse fitting config for joint optimization settings
    _fitting_cfg = fitting_config or {}
    _joint_enabled = _fitting_cfg.get("joint_optimization", True)

    if _joint_enabled and num_matched >= 3:
        _joint_keyframes = _fitting_cfg.get("joint_keyframes", 10)
        _joint_iters = _fitting_cfg.get("joint_iterations", 200)
        _temporal_weight = _fitting_cfg.get("temporal_smoothness_weight", 0.1)

        logger.info(
            "Running joint FLAME optimization: keyframes=%d, iterations=%d, "
            "temporal_weight=%.3f",
            _joint_keyframes, _joint_iters, _temporal_weight,
        )

        # Build per-frame params list from the single-fit result
        # The existing fit produces one shared result; replicate it per frame
        # so joint optimization can refine per-frame expression/pose
        per_frame_params = []
        for fi in range(num_matched):
            per_frame_params.append({
                "shape_params": result["shape_params"].copy(),
                "expression_params": result["expression_params"].copy(),
                "jaw_pose": result["jaw_pose"].copy(),
                "neck_pose": result["neck_pose"].copy(),
                "global_rotation": result["global_rotation"].copy(),
                "translation": result["translation"].copy(),
                "loss": result["loss"],
            })

        per_frame_params = joint_optimize_flame(
            per_frame_params=per_frame_params,
            landmarks_per_frame=landmarks_2d_per_frame,
            cameras=camera_list,
            flame_model=flame,
            landmark_embedding=embedding,
            num_keyframes=_joint_keyframes,
            joint_iterations=_joint_iters,
            temporal_weight=_temporal_weight,
        )

        # Use the first keyframe's result as the canonical output
        # (shape is shared; expression/pose from the most frontal frame)
        canonical_idx = frontal_idx if frontal_idx is not None else 0
        result["shape_params"] = per_frame_params[canonical_idx]["shape_params"]
        result["expression_params"] = per_frame_params[canonical_idx]["expression_params"]
        result["jaw_pose"] = per_frame_params[canonical_idx]["jaw_pose"]
        result["neck_pose"] = per_frame_params[canonical_idx]["neck_pose"]
        result["global_rotation"] = per_frame_params[canonical_idx]["global_rotation"]
        result["translation"] = per_frame_params[canonical_idx]["translation"]
        result["joint_optimized"] = True
        result["per_frame_params"] = per_frame_params

        # Regenerate vertices with the joint-optimized parameters
        with torch.no_grad():
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            final_result = flame.to(dev)(
                shape_params=torch.tensor(result["shape_params"], dtype=torch.float32, device=dev),
                expression_params=torch.tensor(result["expression_params"], dtype=torch.float32, device=dev),
                jaw_pose=torch.tensor(result["jaw_pose"], dtype=torch.float32, device=dev),
                neck_pose=torch.tensor(result["neck_pose"], dtype=torch.float32, device=dev),
                global_rotation=torch.tensor(result["global_rotation"], dtype=torch.float32, device=dev),
                translation=torch.tensor(result["translation"], dtype=torch.float32, device=dev),
            )
            result["vertices"] = final_result["vertices"].cpu().numpy()
            result["faces"] = final_result["faces"].cpu().numpy()

        logger.info("Joint optimization applied — shape consistent across %d frames", num_matched)
    elif _joint_enabled and num_matched < 3:
        logger.info("Joint optimization skipped: only %d matched frames", num_matched)
        result["joint_optimized"] = False
    else:
        result["joint_optimized"] = False

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

    # Save per-frame parameters if joint optimization produced them
    if result.get("joint_optimized") and "per_frame_params" in result:
        per_frame_dir = output_dir / "per_frame"
        per_frame_dir.mkdir(parents=True, exist_ok=True)
        for fi, pfp in enumerate(result["per_frame_params"]):
            np.savez(
                str(per_frame_dir / f"frame_{fi:04d}.npz"),
                shape_params=pfp["shape_params"],
                expression_params=pfp["expression_params"],
                jaw_pose=pfp["jaw_pose"],
                neck_pose=pfp["neck_pose"],
                global_rotation=pfp["global_rotation"],
                translation=pfp["translation"],
            )
        logger.info(
            "Saved %d per-frame FLAME parameters to %s",
            len(result["per_frame_params"]), per_frame_dir,
        )

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
            "early_stopped": result.get("early_stopped", False),
            "num_iterations": len(result["loss_history"]),
            "num_views": len(camera_list),
            "scale_factor": scale_factor,
            "joint_optimized": result.get("joint_optimized", False),
        }
        with open(convergence_path, "w", encoding="utf-8") as fh:
            json.dump(convergence_data, fh)
        logger.info("Saved convergence data to %s", convergence_path)

    # ── Save intermediate visualizations ──────────────────────────────
    # Pick up to 3 frames spread across the sequence for overlay verification
    vis_dir = output_dir / "verification"
    num_vis = min(3, num_matched)
    if num_vis > 0:
        vis_indices = np.linspace(0, num_matched - 1, num_vis, dtype=int)
        if frontal_idx is not None and frontal_idx not in vis_indices:
            vis_indices = np.unique(np.append(vis_indices, frontal_idx))
            vis_indices.sort()

        vertices_3d = result["vertices"][0]  # (V, 3)
        faces = result["faces"]

        for vi in vis_indices:
            img_name = frame_image_names[vi]
            img_path = frames_dir / img_name
            if not img_path.exists():
                # Try common extensions
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = frames_dir / (Path(img_name).stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break

            if img_path.exists():
                overlay_name = f"overlay_{Path(img_name).stem}.png"
                _render_flame_overlay(
                    vertices_3d, faces,
                    camera_list[vi], img_path,
                    vis_dir / overlay_name,
                )

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
