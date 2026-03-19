"""FLAME parametric face model wrapper.

Loads the FLAME .pkl model and implements the forward pass with Linear Blend
Skinning (LBS) to produce a posed, shaped, and expression-deformed face mesh.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _safe_load_flame_pickle(model_path: Path) -> dict:
    """Safely load a FLAME .pkl file with fallback strategies for compatibility.

    Handles numpy version mismatches and pickle protocol differences.

    Args:
        model_path: Path to the FLAME .pkl file.

    Returns:
        Dict of FLAME model data.

    Raises:
        RuntimeError: If all loading strategies fail.
    """
    strategies = [
        ("latin1 encoding", {"encoding": "latin1"}),
        ("bytes encoding", {"encoding": "bytes"}),
        ("ASCII encoding", {"encoding": "ASCII"}),
    ]

    last_error = None
    for desc, kwargs in strategies:
        try:
            with open(model_path, "rb") as fh:
                data = pickle.load(fh, **kwargs)

            # Verify essential keys and convert numpy arrays explicitly
            essential_keys = ["v_template", "f", "shapedirs", "weights", "J_regressor", "kintree_table"]
            for key in essential_keys:
                if key not in data:
                    raise KeyError(f"Missing essential key '{key}' in FLAME pickle")
                if hasattr(data[key], "toarray"):
                    # Convert scipy sparse to dense numpy
                    data[key] = np.array(data[key].toarray())
                elif isinstance(data[key], np.ndarray):
                    # Ensure standard numpy array (handles memmap, etc.)
                    data[key] = np.array(data[key])

            logger.info("FLAME pickle loaded successfully with %s", desc)
            return data

        except (pickle.UnpicklingError, ModuleNotFoundError, ImportError,
                UnicodeDecodeError, KeyError) as e:
            last_error = e
            logger.debug("FLAME pickle load failed with %s: %s", desc, e)
            continue
        except Exception as e:
            # Catch numpy version issues, protocol errors, etc.
            err_str = str(e).lower()
            if "unsupported pickle protocol" in err_str or "numpy" in err_str:
                last_error = e
                logger.debug("FLAME pickle load failed with %s (numpy/protocol): %s", desc, e)
                continue
            raise

    raise RuntimeError(
        f"Failed to load FLAME model from {model_path} after trying all strategies. "
        f"Last error: {last_error}"
    )


def _to_tensor(arr, dtype=torch.float32) -> torch.Tensor:
    """Convert a numpy array (or scipy sparse matrix) to a dense torch tensor."""
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return torch.tensor(np.asarray(arr), dtype=dtype)


class FLAMEModel(nn.Module):
    """FLAME: Faces Learned with an Articulated Model and Expressions.

    Implements the full FLAME forward pass:
        1. Shape blend shapes  (300 PCs)
        2. Expression blend shapes (100 PCs)
        3. Pose blend shapes
        4. Linear Blend Skinning with a kinematic tree
    """

    NUM_SHAPE_PCS = 300
    NUM_EXPR_PCS = 100
    NUM_VERTICES = 5023
    NUM_JOINTS = 5  # global, neck, jaw, left-eye, right-eye

    def __init__(self, model_path: Path, device: str = "cuda"):
        """Load a FLAME .pkl model file.

        Args:
            model_path: Path to a FLAME pickle (e.g. flame2023_Open.pkl).
            device: Torch device string.
        """
        super().__init__()
        model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info("Loading FLAME model from %s ...", model_path)
        flame_data = _safe_load_flame_pickle(model_path)

        # Template mesh -------------------------------------------------
        self.register_buffer(
            "v_template",
            _to_tensor(flame_data["v_template"]),  # (5023, 3)
        )
        self.register_buffer(
            "faces",
            _to_tensor(flame_data["f"], dtype=torch.long),  # (F, 3)
        )

        # Blend shapes --------------------------------------------------
        # shapedirs: (5023, 3, 300)  — shape PCs
        shapedirs = _to_tensor(flame_data["shapedirs"])
        self.register_buffer("shapedirs", shapedirs)

        # Expression blend shapes come from the same file, stored after
        # shape PCs in some versions, or in a separate key.
        if "expressiondir" in flame_data:
            exprdirs = _to_tensor(flame_data["expressiondir"])
        elif shapedirs.shape[2] > self.NUM_SHAPE_PCS:
            # Combined array: first 300 are shape, rest are expression
            exprdirs = shapedirs[:, :, self.NUM_SHAPE_PCS:]
            shapedirs = shapedirs[:, :, :self.NUM_SHAPE_PCS]
            self.shapedirs = shapedirs
        else:
            # Fallback: zero expression dirs
            logger.warning("No expression blend shapes found — using zeros")
            exprdirs = torch.zeros(
                self.NUM_VERTICES, 3, self.NUM_EXPR_PCS,
                dtype=torch.float32,
            )
        self.register_buffer("exprdirs", exprdirs)

        # Pose blend shapes: (5023, 3, num_pose_basis)
        posedirs = _to_tensor(flame_data["posedirs"])
        # posedirs are stored as (5023*3, num_pose_basis) in some versions
        if posedirs.dim() == 2:
            num_pose_basis = posedirs.shape[1]
            posedirs = posedirs.reshape(self.NUM_VERTICES, 3, num_pose_basis)
        self.register_buffer("posedirs", posedirs)

        # Skinning weights: (5023, 5)
        self.register_buffer("lbs_weights", _to_tensor(flame_data["weights"]))

        # Joint regressor: (5, 5023)
        J_regressor = _to_tensor(flame_data["J_regressor"])
        self.register_buffer("J_regressor", J_regressor)

        # Kinematic tree: parent indices for each joint
        self.register_buffer(
            "kintree_table",
            _to_tensor(flame_data["kintree_table"], dtype=torch.long),
        )
        parents = flame_data["kintree_table"][0].astype(np.int64)
        parents[0] = -1  # root has no parent
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))

        self.to(self.device)
        logger.info(
            "FLAME loaded: %d vertices, %d faces, %d shape PCs, %d expression PCs",
            self.v_template.shape[0],
            self.faces.shape[0],
            self.shapedirs.shape[2],
            self.exprdirs.shape[2],
        )

    def forward(
        self,
        shape_params: torch.Tensor,
        expression_params: torch.Tensor,
        jaw_pose: torch.Tensor,
        neck_pose: Optional[torch.Tensor] = None,
        global_rotation: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
    ) -> dict:
        """FLAME forward pass.

        All pose inputs are axis-angle vectors of shape (batch, 3).
        shape_params:      (batch, n_shape)  — up to 300 coefficients
        expression_params: (batch, n_expr)   — up to 100 coefficients
        jaw_pose:          (batch, 3)
        neck_pose:         (batch, 3)  or None (defaults to zero)
        global_rotation:   (batch, 3)  or None (defaults to zero)
        translation:       (batch, 3)  or None (defaults to zero)

        Returns:
            Dict with 'vertices' (batch, 5023, 3), 'faces' (F, 3), 'joints' (batch, 5, 3).
        """
        batch_size = shape_params.shape[0]
        device = shape_params.device

        # Defaults
        if neck_pose is None:
            neck_pose = torch.zeros(batch_size, 3, device=device)
        if global_rotation is None:
            global_rotation = torch.zeros(batch_size, 3, device=device)
        if translation is None:
            translation = torch.zeros(batch_size, 3, device=device)

        # ── 1. Shape blend shapes ──────────────────────────────────────
        n_shape = shape_params.shape[1]
        # shapedirs: (V, 3, 300) → use first n_shape PCs
        shape_offsets = torch.einsum(
            "vcs,bs->bvc", self.shapedirs[:, :, :n_shape], shape_params
        )  # (batch, V, 3)

        # ── 2. Expression blend shapes ─────────────────────────────────
        n_expr = expression_params.shape[1]
        expr_offsets = torch.einsum(
            "vce,be->bvc", self.exprdirs[:, :, :n_expr], expression_params
        )  # (batch, V, 3)

        # Shaped vertices (before posing)
        v_shaped = self.v_template.unsqueeze(0) + shape_offsets + expr_offsets  # (B, V, 3)

        # ── 3. Joint locations ─────────────────────────────────────────
        joints = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)  # (B, J, 3)

        # ── 4. Build full pose vector and rotation matrices ────────────
        # FLAME has 5 joints: root/global, neck, jaw, left_eye, right_eye
        # We set eyes to zero for now.
        eye_pose = torch.zeros(batch_size, 6, device=device)  # 2 eyes × 3
        full_pose = torch.cat(
            [global_rotation, neck_pose, jaw_pose, eye_pose], dim=1
        )  # (B, 15)
        full_pose = full_pose.reshape(batch_size, self.NUM_JOINTS, 3)

        rot_mats = _batch_rodrigues(full_pose.reshape(-1, 3)).reshape(
            batch_size, self.NUM_JOINTS, 3, 3
        )  # (B, J, 3, 3)

        # ── 5. Pose blend shapes ──────────────────────────────────────
        # Pose feature: (R_j - I) flattened for each joint except root
        ident = torch.eye(3, device=device, dtype=torch.float32)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)  # (B, (J-1)*9)
        num_pose_basis = self.posedirs.shape[2]
        pose_feature_trunc = pose_feature[:, :num_pose_basis]
        pose_offsets = torch.einsum(
            "vcp,bp->bvc", self.posedirs, pose_feature_trunc
        )  # (B, V, 3)

        v_posed = v_shaped + pose_offsets

        # ── 6. Linear Blend Skinning ───────────────────────────────────
        vertices = _lbs(
            v_posed, rot_mats, joints, self.parents, self.lbs_weights
        )  # (B, V, 3)

        # ── 7. Apply global translation ────────────────────────────────
        vertices = vertices + translation.unsqueeze(1)

        # ── NaN check ─────────────────────────────────────────────────
        if torch.isnan(vertices).any():
            nan_count = torch.isnan(vertices).sum().item()
            logger.warning(
                "FLAME forward pass produced %d NaN values in vertices, clamping to zero",
                nan_count,
            )
            vertices = torch.where(torch.isnan(vertices), torch.zeros_like(vertices), vertices)

        # Recompute joint positions from final vertices
        joints_final = torch.einsum("jv,bvc->bjc", self.J_regressor, vertices)

        if torch.isnan(joints_final).any():
            logger.warning("FLAME forward pass produced NaN values in joints, clamping to zero")
            joints_final = torch.where(
                torch.isnan(joints_final), torch.zeros_like(joints_final), joints_final
            )

        return {
            "vertices": vertices,
            "faces": self.faces,
            "joints": joints_final,
        }

    def get_landmarks(
        self,
        vertices: torch.Tensor,
        lmk_faces_idx: np.ndarray,
        lmk_bary_coords: np.ndarray,
    ) -> torch.Tensor:
        """Extract landmark positions from mesh vertices using barycentric coords.

        Args:
            vertices: (batch, 5023, 3) mesh vertices.
            lmk_faces_idx: (K,) face indices into self.faces.
            lmk_bary_coords: (K, 3) barycentric weights.

        Returns:
            (batch, K, 3) landmark positions.
        """
        device = vertices.device
        faces_idx = torch.tensor(lmk_faces_idx, dtype=torch.long, device=device)
        bary = torch.tensor(lmk_bary_coords, dtype=torch.float32, device=device)

        # Get triangle vertex indices: (K, 3)
        tri_verts = self.faces[faces_idx]  # (K, 3) — vertex indices

        vertices.shape[0]
        # Gather triangle vertex positions: (B, K, 3, 3)
        v0 = vertices[:, tri_verts[:, 0]]  # (B, K, 3)
        v1 = vertices[:, tri_verts[:, 1]]
        v2 = vertices[:, tri_verts[:, 2]]

        # Barycentric interpolation: (B, K, 3)
        landmarks = (
            bary[None, :, 0:1] * v0
            + bary[None, :, 1:2] * v1
            + bary[None, :, 2:3] * v2
        )

        return landmarks


def load_flame_masks(masks_path: Path) -> dict[str, np.ndarray]:
    """Load FLAME vertex region masks.

    Args:
        masks_path: Path to FLAME_masks.pkl.

    Returns:
        Dict mapping region names (e.g. 'face', 'eye_region', 'lips',
        'nose', 'scalp', 'boundary', 'neck', 'forehead', 'left_eyeball',
        'right_eyeball') to 1-D int arrays of vertex indices.
    """
    masks_path = Path(masks_path)
    try:
        with open(masks_path, "rb") as fh:
            raw = pickle.load(fh, encoding="latin1")
    except Exception:
        logger.debug("FLAME masks pickle load with latin1 failed, trying bytes encoding")
        with open(masks_path, "rb") as fh:
            raw = pickle.load(fh, encoding="bytes")

    masks: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            masks[key] = value.astype(np.int64).flatten()
        elif hasattr(value, "toarray"):
            masks[key] = np.array(value.toarray()).astype(np.int64).flatten()
        else:
            masks[key] = np.asarray(value, dtype=np.int64).flatten()

    logger.info(
        "Loaded FLAME masks: %s",
        {k: len(v) for k, v in masks.items()},
    )
    return masks


# ── Internal LBS utilities ──────────────────────────────────────────────────


def _batch_rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices (Rodrigues' formula).

    Args:
        rot_vecs: (N, 3) axis-angle rotation vectors.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)  # (N, 1)
    axis = rot_vecs / angle  # (N, 3)

    cos_a = torch.cos(angle).unsqueeze(2)  # (N, 1, 1)
    sin_a = torch.sin(angle).unsqueeze(2)  # (N, 1, 1)

    # Skew-symmetric matrix K
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros(batch_size, device=device, dtype=dtype)

    K = torch.stack([
        zeros, -kz, ky,
        kz, zeros, -kx,
        -ky, kx, zeros,
    ], dim=1).reshape(batch_size, 3, 3)

    ident = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)

    # Rodrigues: R = I + sin(θ) K + (1 - cos(θ)) K²
    rot_mat = ident + sin_a * K + (1.0 - cos_a) * torch.bmm(K, K)

    return rot_mat


def _lbs(
    v_posed: torch.Tensor,
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
) -> torch.Tensor:
    """Linear Blend Skinning.

    Args:
        v_posed: (B, V, 3) vertices after shape/expression/pose blend shapes.
        rot_mats: (B, J, 3, 3) per-joint rotation matrices.
        joints: (B, J, 3) rest-pose joint locations.
        parents: (J,) parent joint index for each joint (-1 for root).
        lbs_weights: (V, J) skinning weights.

    Returns:
        (B, V, 3) skinned vertices.
    """
    batch_size = v_posed.shape[0]
    num_joints = rot_mats.shape[1]
    device = v_posed.device

    # Build world-space transforms via forward kinematics
    # Each transform is a 4x4 matrix
    transforms = torch.zeros(batch_size, num_joints, 4, 4, device=device)

    for j in range(num_joints):
        # Local transform: rotation at joint j, translation = joint position
        local = torch.zeros(batch_size, 4, 4, device=device)
        local[:, :3, :3] = rot_mats[:, j]
        local[:, :3, 3] = joints[:, j]
        local[:, 3, 3] = 1.0

        parent_idx = parents[j].item()
        if parent_idx < 0:
            # Root joint — world transform is the local transform
            transforms[:, j] = local
        else:
            # Child joint — chain with parent's world transform
            # First express joint j in parent's local frame
            rel_joint = joints[:, j] - joints[:, parent_idx]
            local_rel = torch.zeros(batch_size, 4, 4, device=device)
            local_rel[:, :3, :3] = rot_mats[:, j]
            local_rel[:, :3, 3] = rel_joint
            local_rel[:, 3, 3] = 1.0
            transforms[:, j] = torch.bmm(
                transforms[:, parent_idx], local_rel
            )

    # Subtract the rest-pose joint locations so that the transform moves
    # vertices *from* their rest position.
    # T_j' = T_j * [I | -j_rest; 0 1]
    joint_homo = torch.zeros(batch_size, num_joints, 4, 1, device=device)
    joint_homo[:, :, :3, 0] = joints
    joint_homo[:, :, 3, 0] = 1.0

    posed_joints = torch.matmul(transforms, joint_homo)  # (B, J, 4, 1)

    rel_transforms = transforms.clone()
    rel_transforms[:, :, :3, 3] = (
        posed_joints[:, :, :3, 0] - joints
    )

    # Subtract rest-pose: pad = [I | -j; 0 1], T' = T * pad
    init_bone = torch.zeros(batch_size, num_joints, 4, 4, device=device)
    init_bone[:, :, :3, :3] = torch.eye(3, device=device)
    init_bone[:, :, :3, 3] = -joints
    init_bone[:, :, 3, 3] = 1.0

    rel_transforms = torch.matmul(transforms, init_bone)  # (B, J, 4, 4)

    # Blend transforms: (B, V, 4, 4) = sum_j w_vj * T'_j
    W = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)  # (B, V, J)
    T_blend = torch.einsum("bvj,bjmn->bvmn", W, rel_transforms)  # (B, V, 4, 4)

    # Apply to homogeneous vertices
    v_homo = torch.ones(batch_size, v_posed.shape[1], 4, device=device)
    v_homo[:, :, :3] = v_posed

    v_skinned = torch.einsum("bvmn,bvn->bvm", T_blend, v_homo)[:, :, :3]

    return v_skinned
