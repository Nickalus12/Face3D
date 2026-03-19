"""FLAME-driven animation of Gaussian Splatting face models.

Given a trained Gaussian model with FLAME binding metadata (triangle indices
and barycentric coordinates), this module enables facial animation by:
1. Changing FLAME expression/pose parameters to deform the mesh.
2. Moving each bound Gaussian to follow its parent triangle.
3. Rendering the animated Gaussians from camera viewpoints.
4. Encoding frame sequences to video via ffmpeg.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from splatting.camera_utils import Camera, generate_turntable_cameras
from splatting.initializer import GaussianModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: normal vector to quaternion
# ---------------------------------------------------------------------------

def _normal_to_quaternion_torch(normals: torch.Tensor) -> torch.Tensor:
    """Convert unit normal vectors to quaternions aligning z-axis to the normal.

    Args:
        normals: (N, 3) unit normal vectors on GPU/CPU.

    Returns:
        (N, 4) quaternions in (w, x, y, z) format.
    """
    device = normals.device
    n = normals.shape[0]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)

    cross = torch.cross(
        z_axis.unsqueeze(0).expand(n, -1), normals, dim=1
    )  # (N, 3)
    dot = normals[:, 2]  # dot with [0,0,1] = z component

    quats = torch.zeros(n, 4, device=device)

    # Anti-parallel: normal ~ [0, 0, -1]
    anti = dot < -0.9999
    quats[anti, 0] = 0.0
    quats[anti, 1] = 1.0

    # Parallel: normal ~ [0, 0, 1]
    para = dot > 0.9999
    quats[para, 0] = 1.0

    # General case
    gen = ~anti & ~para
    if gen.any():
        quats[gen, 0] = 1.0 + dot[gen]
        quats[gen, 1:] = cross[gen]
        norms = quats[gen].norm(dim=1, keepdim=True).clamp(min=1e-10)
        quats[gen] = quats[gen] / norms

    return quats


def _compute_triangle_normals_torch(
    vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Compute per-triangle unit normals.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle vertex indices (long).

    Returns:
        (F, 3) unit normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    lengths = normals.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return normals / lengths


# ---------------------------------------------------------------------------
# Expression presets
# ---------------------------------------------------------------------------

def load_expression_presets(config_path: Path) -> dict:
    """Load predefined expression parameter sets from a YAML file.

    Args:
        config_path: Path to expressions.yaml.

    Returns:
        Dict mapping expression names to parameter dicts with keys
        'expression', 'jaw_pose', 'neck_pose'.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Expression presets not found at %s, using built-in defaults", config_path)
        return _builtin_presets()

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    presets = {}
    for name, params in data.items():
        presets[name] = {
            "expression": np.array(params.get("expression", [0.0] * 10), dtype=np.float32),
            "jaw_pose": np.array(params.get("jaw_pose", [0.0, 0.0, 0.0]), dtype=np.float32),
            "neck_pose": np.array(params.get("neck_pose", [0.0, 0.0, 0.0]), dtype=np.float32),
        }

    logger.info("Loaded %d expression presets from %s", len(presets), config_path)
    return presets


def _builtin_presets() -> dict:
    """Fallback built-in expression presets using FLAME expression PCs.

    FLAME expression blend shapes (first 10 most impactful):
        0: jaw opening / general mouth
        1: lip stretch / smile
        2: lip funneler
        3: lip puckerer
        4: lip tightener
        5: upper lip raiser
        6: lower lip depressor
        7: chin raiser
        8: brow raise
        9: brow furrow
    """
    return {
        "neutral": {
            "expression": np.zeros(10, dtype=np.float32),
            "jaw_pose": np.zeros(3, dtype=np.float32),
            "neck_pose": np.zeros(3, dtype=np.float32),
        },
        "smile": {
            "expression": np.array(
                [0.0, 1.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.2, 0.3, 0.0],
                dtype=np.float32,
            ),
            "jaw_pose": np.array([0.1, 0.0, 0.0], dtype=np.float32),
            "neck_pose": np.zeros(3, dtype=np.float32),
        },
        "surprise": {
            "expression": np.array(
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 1.5, 0.0],
                dtype=np.float32,
            ),
            "jaw_pose": np.array([0.35, 0.0, 0.0], dtype=np.float32),
            "neck_pose": np.zeros(3, dtype=np.float32),
        },
        "blink": {
            # Eye closure is driven by expression PCs that affect the orbital
            # region. In FLAME 2023 the first few PCs that noticeably close
            # the eyelids tend to be around index 5-8 range.
            "expression": np.array(
                [0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.8, 0.0, 0.0],
                dtype=np.float32,
            ),
            "jaw_pose": np.zeros(3, dtype=np.float32),
            "neck_pose": np.zeros(3, dtype=np.float32),
        },
        "frown": {
            "expression": np.array(
                [0.0, -0.8, 0.0, 0.0, 0.5, 0.0, 0.4, 0.0, -0.5, 0.8],
                dtype=np.float32,
            ),
            "jaw_pose": np.zeros(3, dtype=np.float32),
            "neck_pose": np.zeros(3, dtype=np.float32),
        },
    }


# ---------------------------------------------------------------------------
# FaceAnimator
# ---------------------------------------------------------------------------


class FaceAnimator:
    """Animate FLAME-bound Gaussian splat faces via parametric deformation.

    Each Gaussian that is bound to a FLAME triangle (triangle_index != -1)
    moves to follow the deformed triangle position via barycentric
    interpolation. Free Gaussians (triangle_index == -1) remain static.

    Args:
        flame_model_path: Path to the FLAME .pkl model.
        flame_params: Dict or .npz path with fitted shape/expression params.
        binding_metadata: Dict or .npz path with triangle_indices, bary_coords,
            faces, num_bound, num_free.
        gaussians: The trained GaussianModel to animate.
        device: Torch device string.
    """

    def __init__(
        self,
        flame_model_path: Path,
        flame_params: dict | Path,
        binding_metadata: dict | Path,
        gaussians: GaussianModel,
        device: str = "cuda",
    ):
        from reconstruction.flame_model import FLAMEModel

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load FLAME model
        logger.info("Loading FLAME model for animation from %s", flame_model_path)
        self.flame = FLAMEModel(Path(flame_model_path), device=str(self.device))
        self.flame.eval()

        # Load fitted parameters
        if isinstance(flame_params, (str, Path)):
            params = np.load(str(flame_params))
            self.shape_params = torch.from_numpy(
                params["shape_params"].flatten().astype(np.float32)
            ).to(self.device)
            self.expression_params_base = torch.from_numpy(
                params.get("expression_params", np.zeros(100, dtype=np.float32)).flatten().astype(np.float32)
            ).to(self.device)
        else:
            self.shape_params = torch.from_numpy(
                flame_params.get("shape_params", np.zeros(300, dtype=np.float32)).flatten().astype(np.float32)
            ).to(self.device)
            self.expression_params_base = torch.from_numpy(
                flame_params.get("expression_params", np.zeros(100, dtype=np.float32)).flatten().astype(np.float32)
            ).to(self.device)

        # Load binding metadata
        if isinstance(binding_metadata, (str, Path)):
            meta = np.load(str(binding_metadata), allow_pickle=True)
            self.triangle_indices = torch.from_numpy(
                meta["triangle_indices"].astype(np.int64)
            ).to(self.device)
            self.bary_coords = torch.from_numpy(
                meta["bary_coords"].astype(np.float32)
            ).to(self.device)
            self.faces = torch.from_numpy(
                meta["faces"].astype(np.int64)
            ).to(self.device)
            self.num_bound = int(meta["num_bound"])
            self.num_free = int(meta["num_free"])
        else:
            self.triangle_indices = torch.from_numpy(
                binding_metadata["triangle_indices"].astype(np.int64)
            ).to(self.device)
            self.bary_coords = torch.from_numpy(
                binding_metadata["bary_coords"].astype(np.float32)
            ).to(self.device)
            self.faces = torch.from_numpy(
                binding_metadata["faces"].astype(np.int64)
            ).to(self.device)
            self.num_bound = int(binding_metadata["num_bound"])
            self.num_free = int(binding_metadata["num_free"])

        # Store the original Gaussians
        self.gaussians = gaussians.to(self.device)

        # Precompute the mask for bound Gaussians
        self.bound_mask = self.triangle_indices >= 0  # (N,)
        self.bound_indices = self.bound_mask.nonzero(as_tuple=True)[0]

        # Get the rest-pose mesh vertices for reference
        with torch.no_grad():
            rest_result = self.flame.forward(
                shape_params=self.shape_params.unsqueeze(0),
                expression_params=self.expression_params_base.unsqueeze(0),
                jaw_pose=torch.zeros(1, 3, device=self.device),
            )
            self.rest_vertices = rest_result["vertices"][0]  # (V, 3)

        logger.info(
            "FaceAnimator initialized: %d bound, %d free Gaussians, "
            "%d FLAME vertices, %d triangles",
            self.num_bound, self.num_free,
            self.rest_vertices.shape[0], self.faces.shape[0],
        )

    def animate(
        self,
        expression_params: Optional[torch.Tensor] = None,
        jaw_pose: Optional[torch.Tensor] = None,
        neck_pose: Optional[torch.Tensor] = None,
        global_rotation: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
    ) -> GaussianModel:
        """Deform the FLAME mesh and move bound Gaussians accordingly.

        Args:
            expression_params: (N_expr,) new expression coefficients.
                If None, uses the base fitted expression.
            jaw_pose: (3,) axis-angle jaw rotation. Defaults to zero.
            neck_pose: (3,) axis-angle neck rotation. Defaults to zero.
            global_rotation: (3,) axis-angle global rotation. Defaults to zero.
            translation: (3,) global translation. Defaults to zero.

        Returns:
            A new GaussianModel with updated positions and rotations
            for bound Gaussians; free Gaussians keep their original values.
        """
        device = self.device

        # Prepare FLAME inputs (batch=1)
        shape = self.shape_params.unsqueeze(0)

        if expression_params is not None:
            if not isinstance(expression_params, torch.Tensor):
                expression_params = torch.tensor(expression_params, dtype=torch.float32)
            expr = expression_params.to(device).unsqueeze(0)
        else:
            expr = self.expression_params_base.unsqueeze(0)

        if jaw_pose is not None:
            if not isinstance(jaw_pose, torch.Tensor):
                jaw_pose = torch.tensor(jaw_pose, dtype=torch.float32)
            jaw = jaw_pose.to(device).unsqueeze(0)
        else:
            jaw = torch.zeros(1, 3, device=device)

        if neck_pose is not None:
            if not isinstance(neck_pose, torch.Tensor):
                neck_pose = torch.tensor(neck_pose, dtype=torch.float32)
            neck = neck_pose.to(device).unsqueeze(0)
        else:
            neck = torch.zeros(1, 3, device=device)

        if global_rotation is not None:
            if not isinstance(global_rotation, torch.Tensor):
                global_rotation = torch.tensor(global_rotation, dtype=torch.float32)
            glob_rot = global_rotation.to(device).unsqueeze(0)
        else:
            glob_rot = None

        trans = None
        if translation is not None:
            if not isinstance(translation, torch.Tensor):
                translation = torch.tensor(translation, dtype=torch.float32)
            trans = translation.to(device).unsqueeze(0)

        # Run FLAME forward pass
        with torch.no_grad():
            result = self.flame.forward(
                shape_params=shape,
                expression_params=expr,
                jaw_pose=jaw,
                neck_pose=neck,
                global_rotation=glob_rot,
                translation=trans,
            )
            deformed_verts = result["vertices"][0]  # (V, 3)

        # Clone the original Gaussians
        animated = self.gaussians.clone().to(device)

        # Update bound Gaussians via barycentric interpolation
        if self.bound_indices.numel() > 0:
            # Get triangle indices for bound Gaussians
            bound_tri_idx = self.triangle_indices[self.bound_indices]  # (M,)
            bound_bary = self.bary_coords[self.bound_indices]  # (M, 3)

            # Look up deformed triangle vertex positions
            tri_vert_indices = self.faces[bound_tri_idx]  # (M, 3) — vertex indices
            v0 = deformed_verts[tri_vert_indices[:, 0]]  # (M, 3)
            v1 = deformed_verts[tri_vert_indices[:, 1]]
            v2 = deformed_verts[tri_vert_indices[:, 2]]

            # Barycentric interpolation for new positions
            new_positions = (
                bound_bary[:, 0:1] * v0
                + bound_bary[:, 1:2] * v1
                + bound_bary[:, 2:3] * v2
            )  # (M, 3)

            # Compute new triangle normals for rotation update
            tri_normals = _compute_triangle_normals_torch(deformed_verts, self.faces)
            bound_normals = tri_normals[bound_tri_idx]  # (M, 3)
            new_rotations = _normal_to_quaternion_torch(bound_normals)  # (M, 4)

            # Write updated values
            animated.positions[self.bound_indices] = new_positions
            animated.rotations[self.bound_indices] = new_rotations

        return animated

    def _interpolate_params(
        self,
        start: dict,
        end: dict,
        num_frames: int,
    ) -> list[dict]:
        """Linearly interpolate between two parameter sets.

        Args:
            start: Dict with 'expression', 'jaw_pose', 'neck_pose' arrays.
            end: Dict with same keys.
            num_frames: Number of interpolation steps.

        Returns:
            List of num_frames parameter dicts.
        """
        frames = []
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            frame = {}
            for key in ("expression", "jaw_pose", "neck_pose"):
                s = np.asarray(start.get(key, np.zeros(3, dtype=np.float32)), dtype=np.float32)
                e = np.asarray(end.get(key, np.zeros(3, dtype=np.float32)), dtype=np.float32)
                # Pad to same length
                max_len = max(len(s), len(e))
                s_padded = np.zeros(max_len, dtype=np.float32)
                e_padded = np.zeros(max_len, dtype=np.float32)
                s_padded[:len(s)] = s
                e_padded[:len(e)] = e
                frame[key] = (1 - t) * s_padded + t * e_padded
            frames.append(frame)
        return frames

    def _render_frame(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        bg_color: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Render a single frame from the given camera.

        Args:
            gaussians: GaussianModel to render (on device).
            camera: Camera to render from.
            bg_color: Background color (3,). Defaults to white.

        Returns:
            (H, W, 3) uint8 RGB image.
        """
        from gsplat import rasterization

        device = gaussians.device
        if bg_color is None:
            bg_color = torch.ones(3, device=device)

        viewmat = camera.get_viewmat(device)
        K = camera.get_K(device)

        scales = torch.exp(gaussians.scales)
        opacities = torch.sigmoid(gaussians.opacities.squeeze(-1))

        with torch.no_grad():
            renders, alphas, _ = rasterization(
                means=gaussians.positions,
                quats=gaussians.rotations,
                scales=scales,
                opacities=opacities,
                colors=gaussians.colors_sh,
                viewmats=viewmat[None],
                Ks=K[None],
                width=camera.width,
                height=camera.height,
                sh_degree=3,
                packed=True,
            )

        rgb = renders[0].cpu().numpy()  # (H, W, 3)
        alpha = alphas[0, :, :, 0].cpu().numpy()

        # Composite onto white background
        rgb_bg = rgb * alpha[:, :, None] + (1.0 - alpha[:, :, None])
        img = (rgb_bg * 255).clip(0, 255).astype(np.uint8)
        return img

    def generate_expression_sequence(
        self,
        expressions: list[dict],
        output_dir: Path,
        cameras: Optional[list[Camera]] = None,
        fps: int = 30,
    ) -> Path:
        """Render an animation sequence from a list of expression parameter dicts.

        Args:
            expressions: List of dicts, each with optional keys 'expression',
                'jaw_pose', 'neck_pose', 'global_rotation', 'translation'.
            output_dir: Directory for rendered frames and video.
            cameras: Camera(s) to render from. If None, generates a front-facing
                camera from the Gaussian positions.
            fps: Frames per second for video encoding.

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if cameras is None:
            cameras = self._make_front_cameras()

        num_frames = len(expressions)
        logger.info(
            "Generating animation: %d frames, %d camera(s), fps=%d",
            num_frames, len(cameras), fps,
        )

        for cam_idx, cam in enumerate(cameras):
            cam_dir = output_dir / f"cam_{cam_idx:02d}" if len(cameras) > 1 else output_dir
            cam_dir.mkdir(parents=True, exist_ok=True)

            for frame_idx, expr_params in enumerate(tqdm(
                expressions, desc=f"Rendering cam {cam_idx}"
            )):
                animated = self.animate(
                    expression_params=expr_params.get("expression"),
                    jaw_pose=expr_params.get("jaw_pose"),
                    neck_pose=expr_params.get("neck_pose"),
                    global_rotation=expr_params.get("global_rotation"),
                    translation=expr_params.get("translation"),
                )

                img = self._render_frame(animated, cam)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(cam_dir / f"frame_{frame_idx:04d}.png"), img_bgr)

                if frame_idx % 20 == 0:
                    torch.cuda.empty_cache()

            # Encode to video
            self._encode_video(cam_dir, fps)

        return output_dir

    def generate_predefined_animations(
        self,
        output_dir: Path,
        fps: int = 30,
        presets_path: Optional[Path] = None,
    ) -> Path:
        """Generate demo animations: smile, surprise, and head turn.

        Args:
            output_dir: Base output directory for animation videos.
            fps: Frames per second.
            presets_path: Path to expressions.yaml. Falls back to built-ins.

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        presets = load_expression_presets(presets_path) if presets_path else _builtin_presets()
        neutral = presets.get("neutral", _builtin_presets()["neutral"])

        # --- 1. Neutral -> Smile -> Neutral (2 seconds) ---
        smile = presets.get("smile", _builtin_presets()["smile"])
        smile_frames = (
            self._interpolate_params(neutral, smile, fps // 2)       # 0.5s ramp up
            + self._interpolate_params(smile, smile, fps)            # 1.0s hold
            + self._interpolate_params(smile, neutral, fps // 2)     # 0.5s ramp down
        )
        logger.info("Generating smile animation (%d frames)...", len(smile_frames))
        self.generate_expression_sequence(
            smile_frames, output_dir / "smile", fps=fps,
        )

        # --- 2. Neutral -> Surprise -> Neutral (2 seconds) ---
        surprise = presets.get("surprise", _builtin_presets()["surprise"])
        surprise_frames = (
            self._interpolate_params(neutral, surprise, fps // 2)
            + self._interpolate_params(surprise, surprise, fps)
            + self._interpolate_params(surprise, neutral, fps // 2)
        )
        logger.info("Generating surprise animation (%d frames)...", len(surprise_frames))
        self.generate_expression_sequence(
            surprise_frames, output_dir / "surprise", fps=fps,
        )

        # --- 3. Slow head turn left-right (3 seconds) ---
        turn_frames = []
        total_turn_frames = fps * 3
        for i in range(total_turn_frames):
            t = i / max(total_turn_frames - 1, 1)
            # Sinusoidal head rotation: left -> center -> right -> center
            angle = 0.4 * math.sin(2 * math.pi * t)  # ~23 degrees max
            turn_frames.append({
                "expression": neutral["expression"],
                "jaw_pose": neutral["jaw_pose"],
                "neck_pose": np.array([0.0, angle, 0.0], dtype=np.float32),
            })
        logger.info("Generating head turn animation (%d frames)...", len(turn_frames))
        self.generate_expression_sequence(
            turn_frames, output_dir / "head_turn", fps=fps,
        )

        logger.info("All predefined animations saved to %s", output_dir)
        return output_dir

    def export_animated_mesh(
        self,
        expressions: list[dict],
        output_path: Path,
    ) -> Path:
        """Export a mesh sequence as individual .obj files per frame.

        Also attempts to export as a single animated .glb if trimesh is available.

        Args:
            expressions: List of expression parameter dicts.
            output_path: Output directory for mesh frames.

        Returns:
            Path to the output directory.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting %d mesh frames to %s", len(expressions), output_path)

        mesh_frames = []
        for frame_idx, expr_params in enumerate(tqdm(expressions, desc="Mesh export")):
            # Get deformed FLAME vertices
            device = self.device
            shape = self.shape_params.unsqueeze(0)

            expr = expr_params.get("expression")
            if expr is not None:
                if not isinstance(expr, torch.Tensor):
                    expr = torch.tensor(expr, dtype=torch.float32)
                expr = expr.to(device).unsqueeze(0)
            else:
                expr = self.expression_params_base.unsqueeze(0)

            jaw = expr_params.get("jaw_pose")
            if jaw is not None:
                if not isinstance(jaw, torch.Tensor):
                    jaw = torch.tensor(jaw, dtype=torch.float32)
                jaw = jaw.to(device).unsqueeze(0)
            else:
                jaw = torch.zeros(1, 3, device=device)

            neck = expr_params.get("neck_pose")
            if neck is not None:
                if not isinstance(neck, torch.Tensor):
                    neck = torch.tensor(neck, dtype=torch.float32)
                neck = neck.to(device).unsqueeze(0)
            else:
                neck = torch.zeros(1, 3, device=device)

            with torch.no_grad():
                result = self.flame.forward(
                    shape_params=shape,
                    expression_params=expr,
                    jaw_pose=jaw,
                    neck_pose=neck,
                )
                verts = result["vertices"][0].cpu().numpy()  # (V, 3)

            faces_np = self.flame.faces.cpu().numpy()

            # Save individual .obj
            obj_path = output_path / f"frame_{frame_idx:04d}.obj"
            _write_obj(verts, faces_np, obj_path)
            mesh_frames.append((verts, faces_np))

        # Try exporting as animated glTF via trimesh
        try:
            import trimesh

            glb_path = output_path / "animated_sequence.glb"
            scene = trimesh.Scene()
            # Export the first frame mesh as the base geometry
            mesh = trimesh.Trimesh(
                vertices=mesh_frames[0][0],
                faces=mesh_frames[0][1],
                process=False,
            )
            scene.add_geometry(mesh)
            scene.export(str(glb_path))
            logger.info("Exported base mesh as glTF to %s", glb_path)
        except ImportError:
            logger.info("trimesh not available; skipping glTF export")
        except Exception as e:
            logger.warning("glTF export failed: %s", e)

        logger.info("Exported %d mesh frames to %s", len(expressions), output_path)
        return output_path

    def _make_front_cameras(
        self,
        image_width: int = 800,
        image_height: int = 800,
    ) -> list[Camera]:
        """Generate a single front-facing camera based on Gaussian positions."""
        positions = self.gaussians.positions.detach()
        center = positions.mean(dim=0).cpu().numpy()
        extent = (positions.max(dim=0).values - positions.min(dim=0).values).cpu().numpy()
        radius = float(np.linalg.norm(extent)) * 0.7

        # Front-facing camera (angle = 0)
        return generate_turntable_cameras(
            center=center,
            radius=radius,
            num_views=1,
            height=0.0,
            fov=0.7854,
            image_width=image_width,
            image_height=image_height,
        )

    @staticmethod
    def _encode_video(frames_dir: Path, fps: int = 30) -> Optional[Path]:
        """Encode PNG frames to MP4 video using ffmpeg.

        Args:
            frames_dir: Directory containing frame_XXXX.png files.
            fps: Frames per second.

        Returns:
            Path to video file, or None if ffmpeg is unavailable.
        """
        video_path = frames_dir / "animation.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", str(fps),
                    "-i", str(frames_dir / "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(video_path),
                ],
                capture_output=True,
                check=True,
            )
            logger.info("Encoded animation video to %s", video_path)
            return video_path
        except FileNotFoundError:
            logger.info("ffmpeg not available; frames saved to %s", frames_dir)
            return None
        except subprocess.CalledProcessError as e:
            logger.warning("ffmpeg encoding failed: %s", e.stderr[:200] if e.stderr else e)
            return None


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------


def _write_obj(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """Write a simple .obj mesh file.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle vertex indices (0-based).
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ uses 1-based indexing
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
