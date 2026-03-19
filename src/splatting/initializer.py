"""Initialize Gaussian splat parameters from various face reconstruction sources."""

from __future__ import annotations

import logging
import pickle
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement

# Lazy open3d import — 2+ second import time, only needed for mesh I/O
def _get_o3d():
    import open3d as o3d
    return o3d

logger = logging.getLogger(__name__)

# Default SH degree for initialization.
# NOTE: The trainer uses sh_degree_max=1 (4 coefficients) for faces.
# We allocate only what's needed; the trainer will grow shN if needed.
DEFAULT_SH_DEGREE = 1
DEFAULT_SH_COEFFS = (DEFAULT_SH_DEGREE + 1) ** 2  # 4

# Legacy constant kept for backward compatibility with saved models
SH_DEGREE = 3
SH_COEFFS = (SH_DEGREE + 1) ** 2  # 16


@dataclass
class GaussianModel:
    """Container for all learnable Gaussian parameters.

    All tensors live on the same device and require grad when used for training.
    """

    positions: torch.Tensor          # (N, 3)  xyz
    colors_sh: torch.Tensor          # (N, SH_COEFFS, 3)  spherical harmonic coefficients
    scales: torch.Tensor             # (N, 3)  log-scale
    rotations: torch.Tensor          # (N, 4)  quaternions (w, x, y, z)
    opacities: torch.Tensor          # (N, 1)  logit-space opacity

    # Bookkeeping (not saved)
    _xyz_gradient_accum: torch.Tensor | None = field(default=None, repr=False)
    _xyz_gradient_count: torch.Tensor | None = field(default=None, repr=False)
    _max_radii2D: torch.Tensor | None = field(default=None, repr=False)

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def to(self, device: torch.device | str) -> GaussianModel:
        """Move all tensors to the given device."""
        self.positions = self.positions.to(device)
        self.colors_sh = self.colors_sh.to(device)
        self.scales = self.scales.to(device)
        self.rotations = self.rotations.to(device)
        self.opacities = self.opacities.to(device)
        if self._xyz_gradient_accum is not None:
            self._xyz_gradient_accum = self._xyz_gradient_accum.to(device)
        if self._xyz_gradient_count is not None:
            self._xyz_gradient_count = self._xyz_gradient_count.to(device)
        if self._max_radii2D is not None:
            self._max_radii2D = self._max_radii2D.to(device)
        return self

    def requires_grad_(self, requires: bool = True) -> GaussianModel:
        """Set requires_grad on all learnable parameters."""
        self.positions.requires_grad_(requires)
        self.colors_sh.requires_grad_(requires)
        self.scales.requires_grad_(requires)
        self.rotations.requires_grad_(requires)
        self.opacities.requires_grad_(requires)
        return self

    def parameters_list(self) -> list[torch.Tensor]:
        """Return the five learnable parameter tensors."""
        return [self.positions, self.colors_sh, self.scales, self.rotations, self.opacities]

    def reset_gradient_tracking(self) -> None:
        """Reset the accumulators used for adaptive density control."""
        n = self.num_gaussians
        device = self.device
        self._xyz_gradient_accum = torch.zeros(n, 1, device=device)
        self._xyz_gradient_count = torch.zeros(n, 1, device=device, dtype=torch.int32)
        self._max_radii2D = torch.zeros(n, device=device)

    def clone(self) -> GaussianModel:
        """Deep-copy all tensors."""
        return GaussianModel(
            positions=self.positions.detach().clone(),
            colors_sh=self.colors_sh.detach().clone(),
            scales=self.scales.detach().clone(),
            rotations=self.rotations.detach().clone(),
            opacities=self.opacities.detach().clone(),
        )


def _rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0,1] to the DC (0th) spherical harmonic coefficient.

    The SH basis value for degree-0 is 0.28209479...  (1 / (2*sqrt(pi))).
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _estimate_initial_scales(points: np.ndarray, k: int = 4) -> np.ndarray:
    """Estimate per-point scale from local point density using KNN.

    Uses scipy's cKDTree for batch queries (100x faster than open3d's
    per-point loop for 300K points).

    Args:
        points: (N, 3) point positions.
        k: Number of nearest neighbours (excluding self).

    Returns:
        (N, 3) log-scale values.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    # Batch query: k+1 to include self, then discard self (index 0)
    dists, _ = tree.query(points, k=k + 1, workers=-1)  # (N, k+1)
    # dists[:, 0] is self (distance 0), take mean of k actual neighbors
    mean_dists = np.mean(dists[:, 1:], axis=1).astype(np.float32)  # (N,)
    mean_dists = np.maximum(mean_dists, 1e-7)

    log_scales = np.log(mean_dists)
    return np.stack([log_scales, log_scales, log_scales], axis=-1)


def initialize_from_pointcloud(
    pointcloud_path: Path,
    output_path: Path,
) -> GaussianModel:
    """Initialize Gaussians from a .ply point cloud.

    Loads positions and vertex colors, estimates scales from local density,
    and writes the result as a 3DGS-compatible .ply.

    Args:
        pointcloud_path: Path to input .ply (e.g. from COLMAP dense or depth fusion).
        output_path: Where to save the initialized 3DGS .ply.

    Returns:
        A GaussianModel with all parameters on CPU.
    """
    logger.info("Loading point cloud from %s", pointcloud_path)
    o3d = _get_o3d()
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    n = len(points)
    logger.info("Loaded %d points", n)

    # Colors
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)  # already [0,1]
    else:
        logger.warning("Point cloud has no colors; defaulting to grey")
        colors = np.full((n, 3), 0.5, dtype=np.float32)

    # SH coefficients: allocate only what the trainer needs (default SH1 = 4 coeffs)
    # The trainer's _build_splats() splits into sh0 (1) + shN (rest).
    # Allocating SH3 (16) wastes 43MB VRAM on zeros for 300K Gaussians.
    sh = np.zeros((n, DEFAULT_SH_COEFFS, 3), dtype=np.float32)
    sh[:, 0, :] = _rgb_to_sh0(colors)

    # Scales from KNN density
    logger.info("Estimating initial scales via KNN (k=4)...")
    scales = _estimate_initial_scales(points, k=4)

    # Identity quaternions (w=1, x=y=z=0)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    # Opacity: inverse sigmoid of 0.8
    opacity_value = np.log(0.8 / (1.0 - 0.8))  # logit(0.8) ~ 1.386
    opacities = np.full((n, 1), opacity_value, dtype=np.float32)

    model = GaussianModel(
        positions=torch.from_numpy(points),
        colors_sh=torch.from_numpy(sh),
        scales=torch.from_numpy(scales),
        rotations=torch.from_numpy(rotations),
        opacities=torch.from_numpy(opacities),
    )

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_gaussians_ply(model, output_path)
    logger.info("Saved initialized Gaussians (%d) to %s", n, output_path)

    return model


def initialize_from_flame_mesh(
    mesh_path: Path,
    texture_image: Path | None = None,
    num_samples: int = 200_000,
) -> GaussianModel:
    """Initialize Gaussians by sampling points on a FLAME mesh surface.

    Better for faces because the sampled points follow face topology,
    giving denser coverage on face regions vs. background.

    Args:
        mesh_path: Path to .obj or .ply FLAME mesh.
        texture_image: Optional texture image for color initialization.
        num_samples: Number of surface samples to generate.

    Returns:
        A GaussianModel on CPU.
    """
    logger.info("Loading FLAME mesh from %s", mesh_path)
    o3d = _get_o3d()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {mesh_path} has no triangles")

    # Sample points uniformly on the mesh surface
    logger.info("Sampling %d points on mesh surface...", num_samples)
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    points = np.asarray(pcd.points, dtype=np.float32)
    n = len(points)

    # Colors: from texture if available, else from vertex colors, else skin tone
    if texture_image is not None and texture_image.exists():
        import cv2

        tex = cv2.imread(str(texture_image))
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Approximate: use vertex colors from the sampled point cloud
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)
        else:
            # Sample center of texture as a rough average skin color
            h, w = tex.shape[:2]
            center_color = tex[h // 2, w // 2]
            colors = np.tile(center_color, (n, 1)).astype(np.float32)
    elif mesh.has_vertex_colors():
        # The sampled point cloud inherits vertex colors via interpolation
        colors = np.asarray(pcd.colors, dtype=np.float32)
    else:
        # Default skin tone
        skin_rgb = np.array([0.76, 0.60, 0.50], dtype=np.float32)
        colors = np.tile(skin_rgb, (n, 1))

    # SH DC term (allocate only DEFAULT_SH_COEFFS, not full SH3)
    sh = np.zeros((n, DEFAULT_SH_COEFFS, 3), dtype=np.float32)
    sh[:, 0, :] = _rgb_to_sh0(colors)

    # Scales
    logger.info("Estimating initial scales...")
    scales = _estimate_initial_scales(points, k=4)

    # Identity rotations
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    # Opacity
    opacity_value = np.log(0.8 / 0.2)
    opacities = np.full((n, 1), opacity_value, dtype=np.float32)

    model = GaussianModel(
        positions=torch.from_numpy(points),
        colors_sh=torch.from_numpy(sh),
        scales=torch.from_numpy(scales),
        rotations=torch.from_numpy(rotations),
        opacities=torch.from_numpy(opacities),
    )

    logger.info("Initialized %d Gaussians from FLAME mesh", n)
    return model


# ---------------------------------------------------------------------------
# FLAME-bound Gaussian initialization (GaussianAvatars-style)
# ---------------------------------------------------------------------------


def _compute_triangle_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-triangle unit normals.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle vertex indices.

    Returns:
        (F, 3) unit normals for each triangle.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.clip(lengths, 1e-10, None)
    return normals / lengths


def _compute_triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute the area of each triangle.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle vertex indices.

    Returns:
        (F,) area of each triangle.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _sample_barycentric(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random barycentric coordinates using the square-root method.

    Args:
        num_samples: Number of barycentric coordinate triples to generate.
        rng: Numpy random generator.

    Returns:
        (num_samples, 3) array of barycentric coordinates (u, v, w) that sum to 1.
    """
    r1 = rng.random(num_samples, dtype=np.float32)
    r2 = rng.random(num_samples, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    return np.stack([u, v, w], axis=-1)


def _normal_to_quaternion(normals: np.ndarray) -> np.ndarray:
    """Convert unit normal vectors to quaternions that align the z-axis to the normal.

    Uses the rotation from [0, 0, 1] to each normal vector.

    Args:
        normals: (N, 3) unit normal vectors.

    Returns:
        (N, 4) quaternions in (w, x, y, z) format.
    """
    n = len(normals)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Cross product gives rotation axis, dot product gives cos(angle)
    cross = np.cross(np.tile(z_axis, (n, 1)), normals)  # (N, 3)
    dot = normals[:, 2]  # z-component = dot with [0,0,1]

    quats = np.zeros((n, 4), dtype=np.float32)

    # Handle the degenerate case where normal is nearly [0, 0, -1]
    anti_parallel = dot < -0.9999
    quats[anti_parallel] = [0.0, 1.0, 0.0, 0.0]  # 180-degree rotation around x

    # Handle nearly parallel (normal ~ [0, 0, 1])
    parallel = dot > 0.9999
    quats[parallel] = [1.0, 0.0, 0.0, 0.0]  # identity

    # General case
    general = ~anti_parallel & ~parallel
    if np.any(general):
        c = cross[general]
        d = dot[general]
        # q = [1 + dot, cross] then normalize (half-angle formula)
        quats[general, 0] = 1.0 + d
        quats[general, 1:] = c
        norms = np.linalg.norm(quats[general], axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        quats[general] /= norms

    return quats


def _apply_flame_params(
    v_template: np.ndarray,
    shapedirs: np.ndarray,
    shape_params: np.ndarray,
    exprdirs: Optional[np.ndarray] = None,
    expression_params: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply FLAME shape and expression parameters to the template mesh.

    Performs only the blend shape deformation (no posing/LBS) to get
    a shaped+expressed mesh in the canonical pose.

    Args:
        v_template: (V, 3) template vertex positions.
        shapedirs: (V, 3, N_shape) shape blend shape basis.
        shape_params: (N_shape,) shape coefficients.
        exprdirs: (V, 3, N_expr) expression blend shape basis, or None.
        expression_params: (N_expr,) expression coefficients, or None.

    Returns:
        (V, 3) deformed vertex positions.
    """
    # Shape deformation: v_template + shapedirs @ shape_params
    n_shape = min(shape_params.shape[0], shapedirs.shape[2])
    vertices = v_template.copy()
    vertices += np.einsum("vcs,s->vc", shapedirs[:, :, :n_shape], shape_params[:n_shape])

    # Expression deformation
    if exprdirs is not None and expression_params is not None:
        n_expr = min(expression_params.shape[0], exprdirs.shape[2])
        vertices += np.einsum("vce,e->vc", exprdirs[:, :, :n_expr], expression_params[:n_expr])

    return vertices


def initialize_from_flame_binding(
    flame_model_path: Path,
    flame_params_path: Optional[Path] = None,
    num_gaussians_per_triangle: int = 3,
    color_image: Optional[Path] = None,
    cameras: Optional[list] = None,
) -> tuple[GaussianModel, dict]:
    """Initialize Gaussians bound to FLAME mesh triangles (GaussianAvatars-style).

    Each Gaussian is parameterized by barycentric coordinates relative to a
    FLAME triangle, enabling animation by deforming the FLAME mesh. Additional
    "free" Gaussians are placed around the head for hair and other non-FLAME
    regions.

    Args:
        flame_model_path: Path to FLAME .pkl model file.
        flame_params_path: Optional path to fitted FLAME parameters (.npz from
            stage 10). If None, uses the template mesh.
        num_gaussians_per_triangle: Number of Gaussians to place per triangle.
        color_image: Optional reference image for color initialization (unused
            currently; reserved for texture projection).
        cameras: Optional camera list for color projection (unused currently).

    Returns:
        Tuple of (GaussianModel, metadata_dict).
        metadata_dict contains:
            - 'triangle_indices': (N,) int array, triangle index per Gaussian
              (-1 for free Gaussians).
            - 'bary_coords': (N, 3) float array, barycentric coordinates
              (zeros for free Gaussians).
            - 'faces': (F, 3) int array, FLAME triangle vertex indices.
            - 'num_bound': int, number of triangle-bound Gaussians.
            - 'num_free': int, number of free Gaussians.
    """
    flame_model_path = Path(flame_model_path)
    rng = np.random.default_rng(42)

    # ── 1. Load FLAME model ───────────────────────────────────────────
    logger.info("Loading FLAME model from %s ...", flame_model_path)
    with open(flame_model_path, "rb") as fh:
        flame_data = pickle.load(fh, encoding="latin1")

    v_template = np.asarray(flame_data["v_template"], dtype=np.float32)  # (5023, 3)
    faces = np.asarray(flame_data["f"], dtype=np.int64)                  # (F, 3)

    # Load blend shape bases (needed if applying params)
    shapedirs = np.asarray(
        flame_data["shapedirs"].toarray() if hasattr(flame_data["shapedirs"], "toarray")
        else flame_data["shapedirs"],
        dtype=np.float32,
    )

    # Handle combined shape+expression dirs
    exprdirs = None
    if "expressiondir" in flame_data:
        exprdirs = np.asarray(
            flame_data["expressiondir"].toarray() if hasattr(flame_data["expressiondir"], "toarray")
            else flame_data["expressiondir"],
            dtype=np.float32,
        )
    elif shapedirs.shape[2] > 300:
        exprdirs = shapedirs[:, :, 300:]
        shapedirs = shapedirs[:, :, :300]

    num_faces = len(faces)
    logger.info("FLAME mesh: %d vertices, %d triangles", len(v_template), num_faces)

    # ── 2. Apply fitted parameters if available ───────────────────────
    if flame_params_path is not None:
        flame_params_path = Path(flame_params_path)
        if flame_params_path.exists():
            logger.info("Applying fitted FLAME parameters from %s", flame_params_path)
            params = np.load(str(flame_params_path))
            shape_params = params.get("shape_params", np.zeros(300, dtype=np.float32)).flatten()
            expression_params = params.get("expression_params", None)
            if expression_params is not None:
                expression_params = expression_params.flatten()

            vertices = _apply_flame_params(
                v_template, shapedirs, shape_params,
                exprdirs, expression_params,
            )
            logger.info("Applied shape (%d) + expression (%d) blend shapes",
                         len(shape_params),
                         len(expression_params) if expression_params is not None else 0)
        else:
            logger.warning("FLAME params file not found at %s, using template mesh",
                           flame_params_path)
            vertices = v_template.copy()
    else:
        vertices = v_template.copy()

    # ── 3. Compute per-triangle geometry ──────────────────────────────
    tri_normals = _compute_triangle_normals(vertices, faces)    # (F, 3)
    tri_areas = _compute_triangle_areas(vertices, faces)        # (F,)

    # Triangle centroids for reference
    tri_v0 = vertices[faces[:, 0]]
    tri_v1 = vertices[faces[:, 1]]
    tri_v2 = vertices[faces[:, 2]]

    # ── 4. Place bound Gaussians on each triangle ─────────────────────
    num_bound = num_faces * num_gaussians_per_triangle
    logger.info("Placing %d bound Gaussians (%d per triangle x %d triangles)",
                num_bound, num_gaussians_per_triangle, num_faces)

    # Sample barycentric coordinates for all bound Gaussians
    bary_all = _sample_barycentric(num_bound, rng)  # (num_bound, 3)

    # Triangle index for each bound Gaussian
    tri_indices_bound = np.repeat(np.arange(num_faces, dtype=np.int64), num_gaussians_per_triangle)

    # Compute 3D positions from barycentric coordinates
    # pos = u * v0 + v * v1 + w * v2
    bound_v0 = tri_v0[tri_indices_bound]  # (num_bound, 3)
    bound_v1 = tri_v1[tri_indices_bound]
    bound_v2 = tri_v2[tri_indices_bound]
    bound_positions = (
        bary_all[:, 0:1] * bound_v0
        + bary_all[:, 1:2] * bound_v1
        + bary_all[:, 2:3] * bound_v2
    )  # (num_bound, 3)

    # Scales proportional to sqrt(triangle area) — smaller triangles get smaller Gaussians
    bound_tri_areas = tri_areas[tri_indices_bound]
    scale_from_area = np.sqrt(bound_tri_areas).astype(np.float32)
    # Clamp minimum scale
    scale_from_area = np.clip(scale_from_area, 1e-6, None)
    bound_log_scales = np.log(scale_from_area)
    # Anisotropic: flatten slightly along the normal (z-scale smaller)
    bound_scales = np.stack([
        bound_log_scales,
        bound_log_scales,
        bound_log_scales - 0.5,  # thinner along normal direction
    ], axis=-1).astype(np.float32)  # (num_bound, 3)

    # Rotations: align z-axis with triangle normal
    bound_normals = tri_normals[tri_indices_bound]
    bound_rotations = _normal_to_quaternion(bound_normals)  # (num_bound, 4)

    # Colors: interpolate vertex colors (default skin tone since FLAME has no vertex colors)
    skin_rgb = np.array([0.76, 0.60, 0.50], dtype=np.float32)
    bound_colors = np.tile(skin_rgb, (num_bound, 1))

    # Opacity: logit(0.8) ~ 1.386
    opacity_val = np.log(0.8 / 0.2).astype(np.float32)
    bound_opacities = np.full((num_bound, 1), opacity_val, dtype=np.float32)

    # ── 5. Generate free Gaussians for hair, ears, neck ───────────────
    # Determine how many free Gaussians to add (~10K total)
    num_free_hair = 7000
    num_free_ears_neck = 3000
    num_free = num_free_hair + num_free_ears_neck

    logger.info("Generating %d free Gaussians (hair: %d, ears/neck: %d)",
                num_free, num_free_hair, num_free_ears_neck)

    # --- Hair shell: offset along normals from scalp/top-of-head region ---
    # Use triangles in the upper part of the mesh (above the centroid y-coordinate)
    mesh_centroid = vertices.mean(axis=0)
    # Identify "top" triangles — triangles whose centroid is above the mesh centroid
    tri_centroids = (tri_v0 + tri_v1 + tri_v2) / 3.0
    # In FLAME, y-axis typically points up; use the top ~30% of triangles
    y_values = tri_centroids[:, 1]
    y_threshold = np.percentile(y_values, 70)
    top_mask = y_values >= y_threshold
    top_tri_indices = np.where(top_mask)[0]

    if len(top_tri_indices) == 0:
        # Fallback: use all triangles
        top_tri_indices = np.arange(num_faces)

    # Sample triangles proportional to area for hair region
    top_areas = tri_areas[top_tri_indices]
    top_probs = top_areas / top_areas.sum()
    hair_tri_picks = rng.choice(top_tri_indices, size=num_free_hair, p=top_probs)

    # Sample barycentric coords on these triangles
    hair_bary = _sample_barycentric(num_free_hair, rng)
    hair_base_pos = (
        hair_bary[:, 0:1] * vertices[faces[hair_tri_picks, 0]]
        + hair_bary[:, 1:2] * vertices[faces[hair_tri_picks, 1]]
        + hair_bary[:, 2:3] * vertices[faces[hair_tri_picks, 2]]
    )

    # Offset along triangle normals by 2-5 cm
    hair_offsets = rng.uniform(0.02, 0.05, size=(num_free_hair, 1)).astype(np.float32)
    hair_normals = tri_normals[hair_tri_picks]
    # Add some tangential jitter for volume
    tangent_jitter = rng.normal(0, 0.01, size=(num_free_hair, 3)).astype(np.float32)
    hair_positions = hair_base_pos + hair_offsets * hair_normals + tangent_jitter

    # --- Ears/neck region: offset from lower/side triangles ---
    # Lower region (below centroid) and side regions
    y_threshold_low = np.percentile(y_values, 30)
    low_mask = y_values <= y_threshold_low
    # Side regions: triangles with significant x-offset from center
    x_offset = np.abs(tri_centroids[:, 0] - mesh_centroid[0])
    x_threshold = np.percentile(x_offset, 70)
    side_mask = x_offset >= x_threshold
    ear_neck_mask = low_mask | side_mask
    ear_neck_indices = np.where(ear_neck_mask)[0]

    if len(ear_neck_indices) == 0:
        ear_neck_indices = np.arange(num_faces)

    ear_neck_areas = tri_areas[ear_neck_indices]
    ear_neck_probs = ear_neck_areas / ear_neck_areas.sum()
    ear_neck_picks = rng.choice(ear_neck_indices, size=num_free_ears_neck, p=ear_neck_probs)

    ear_neck_bary = _sample_barycentric(num_free_ears_neck, rng)
    ear_neck_base = (
        ear_neck_bary[:, 0:1] * vertices[faces[ear_neck_picks, 0]]
        + ear_neck_bary[:, 1:2] * vertices[faces[ear_neck_picks, 1]]
        + ear_neck_bary[:, 2:3] * vertices[faces[ear_neck_picks, 2]]
    )

    # Offset by 2-4 cm along normals
    ear_neck_offsets = rng.uniform(0.02, 0.04, size=(num_free_ears_neck, 1)).astype(np.float32)
    ear_neck_normals_dir = tri_normals[ear_neck_picks]
    ear_neck_jitter = rng.normal(0, 0.008, size=(num_free_ears_neck, 3)).astype(np.float32)
    ear_neck_positions = ear_neck_base + ear_neck_offsets * ear_neck_normals_dir + ear_neck_jitter

    # Combine free Gaussians
    free_positions = np.concatenate([hair_positions, ear_neck_positions], axis=0)

    # Free Gaussian scales — larger than bound (they represent diffuse regions)
    free_scale_val = np.log(0.01).astype(np.float32)  # ~1cm base scale
    # Hair gets slightly larger, ears/neck slightly smaller
    hair_scale = np.full((num_free_hair, 3), free_scale_val + 0.3, dtype=np.float32)
    ear_neck_scale = np.full((num_free_ears_neck, 3), free_scale_val, dtype=np.float32)
    free_scales = np.concatenate([hair_scale, ear_neck_scale], axis=0)

    # Free rotations: identity (these move freely, no alignment needed)
    free_rotations = np.zeros((num_free, 4), dtype=np.float32)
    free_rotations[:, 0] = 1.0

    # Free colors: dark brown for hair, skin tone for ears/neck
    hair_rgb = np.array([0.15, 0.10, 0.07], dtype=np.float32)
    ear_neck_rgb = np.array([0.72, 0.56, 0.46], dtype=np.float32)
    free_colors = np.concatenate([
        np.tile(hair_rgb, (num_free_hair, 1)),
        np.tile(ear_neck_rgb, (num_free_ears_neck, 1)),
    ], axis=0)

    # Free opacity: slightly lower than bound (0.6)
    free_opacity_val = np.log(0.6 / 0.4).astype(np.float32)
    free_opacities = np.full((num_free, 1), free_opacity_val, dtype=np.float32)

    # ── 6. Combine bound + free Gaussians ─────────────────────────────
    total_n = num_bound + num_free
    all_positions = np.concatenate([bound_positions, free_positions], axis=0).astype(np.float32)
    all_scales = np.concatenate([bound_scales, free_scales], axis=0).astype(np.float32)
    all_rotations = np.concatenate([bound_rotations, free_rotations], axis=0).astype(np.float32)
    all_colors = np.concatenate([bound_colors, free_colors], axis=0).astype(np.float32)
    all_opacities = np.concatenate([bound_opacities, free_opacities], axis=0).astype(np.float32)

    # SH coefficients: DC term from colors, rest zero (DEFAULT_SH_COEFFS, not full SH3)
    all_sh = np.zeros((total_n, DEFAULT_SH_COEFFS, 3), dtype=np.float32)
    all_sh[:, 0, :] = _rgb_to_sh0(all_colors)

    # Build GaussianModel
    model = GaussianModel(
        positions=torch.from_numpy(all_positions),
        colors_sh=torch.from_numpy(all_sh),
        scales=torch.from_numpy(all_scales),
        rotations=torch.from_numpy(all_rotations),
        opacities=torch.from_numpy(all_opacities),
    )

    # Build metadata for animation support
    # Triangle indices: bound Gaussians have their triangle index, free have -1
    all_tri_indices = np.full(total_n, -1, dtype=np.int64)
    all_tri_indices[:num_bound] = tri_indices_bound

    # Barycentric coords: bound Gaussians have their coords, free have zeros
    all_bary_coords = np.zeros((total_n, 3), dtype=np.float32)
    all_bary_coords[:num_bound] = bary_all

    metadata = {
        "triangle_indices": all_tri_indices,
        "bary_coords": all_bary_coords,
        "faces": faces,
        "num_bound": num_bound,
        "num_free": num_free,
    }

    logger.info(
        "FLAME-bound initialization complete: %d total Gaussians "
        "(%d bound + %d free)",
        total_n, num_bound, num_free,
    )

    return model, metadata


def initialize_from_colmap_sparse(
    colmap_model_dir: Path,
) -> GaussianModel:
    """Fallback initializer using COLMAP sparse reconstruction points.

    Reads the binary points3D file from a COLMAP model directory.

    Args:
        colmap_model_dir: Path to COLMAP model directory containing
            points3D.bin (or points3D.txt).

    Returns:
        A GaussianModel on CPU.
    """
    points3d_bin = colmap_model_dir / "points3D.bin"
    points3d_txt = colmap_model_dir / "points3D.txt"

    if points3d_bin.exists():
        points, colors = _read_colmap_points3d_binary(points3d_bin)
    elif points3d_txt.exists():
        points, colors = _read_colmap_points3d_text(points3d_txt)
    else:
        raise FileNotFoundError(
            f"No points3D.bin or points3D.txt found in {colmap_model_dir}"
        )

    n = len(points)
    logger.info("Loaded %d sparse COLMAP points", n)

    # Normalize colors to [0,1]
    colors = colors.astype(np.float32) / 255.0

    sh = np.zeros((n, DEFAULT_SH_COEFFS, 3), dtype=np.float32)
    sh[:, 0, :] = _rgb_to_sh0(colors)

    scales = _estimate_initial_scales(points, k=min(4, max(1, n - 1)))

    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    opacity_value = np.log(0.8 / 0.2)
    opacities = np.full((n, 1), opacity_value, dtype=np.float32)

    model = GaussianModel(
        positions=torch.from_numpy(points),
        colors_sh=torch.from_numpy(sh),
        scales=torch.from_numpy(scales),
        rotations=torch.from_numpy(rotations),
        opacities=torch.from_numpy(opacities),
    )

    logger.info("Initialized %d Gaussians from COLMAP sparse points", n)
    return model


# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

def _read_colmap_points3d_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.bin and return positions and RGB arrays."""
    points_list = []
    colors_list = []

    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            # Skip track entries (image_id + point2d_idx per entry)
            f.read(track_length * 8)
            points_list.append(xyz)
            colors_list.append(rgb)

    return (
        np.array(points_list, dtype=np.float32),
        np.array(colors_list, dtype=np.uint8),
    )


def _read_colmap_points3d_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.txt and return positions and RGB arrays."""
    points_list = []
    colors_list = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            rgb = [int(parts[4]), int(parts[5]), int(parts[6])]
            points_list.append(xyz)
            colors_list.append(rgb)

    return (
        np.array(points_list, dtype=np.float32),
        np.array(colors_list, dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# PLY save helper
# ---------------------------------------------------------------------------

def _save_gaussians_ply(model: GaussianModel, path: Path) -> None:
    """Save a GaussianModel as a 3DGS-compatible .ply file."""
    n = model.num_gaussians
    positions = model.positions.detach().cpu().numpy()
    sh = model.colors_sh.detach().cpu().numpy()  # (N, C, 3)
    scales = model.scales.detach().cpu().numpy()
    rotations = model.rotations.detach().cpu().numpy()
    opacities = model.opacities.detach().cpu().numpy()

    # Build structured array
    # 3DGS PLY format: x y z nx ny nz f_dc_0..2 f_rest_0..N opacity scale_0..2 rot_0..3
    actual_sh_coeffs = sh.shape[1]  # May be 4 (SH1) or 16 (SH3)
    num_sh_rest = actual_sh_coeffs - 1
    dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ]
    # DC SH (3 values)
    for i in range(3):
        dtype_list.append((f"f_dc_{i}", "f4"))
    # Rest SH (15 * 3 = 45 values)
    for i in range(num_sh_rest * 3):
        dtype_list.append((f"f_rest_{i}", "f4"))
    dtype_list.append(("opacity", "f4"))
    for i in range(3):
        dtype_list.append((f"scale_{i}", "f4"))
    for i in range(4):
        dtype_list.append((f"rot_{i}", "f4"))

    arr = np.zeros(n, dtype=dtype_list)
    arr["x"] = positions[:, 0]
    arr["y"] = positions[:, 1]
    arr["z"] = positions[:, 2]
    arr["nx"] = 0.0
    arr["ny"] = 0.0
    arr["nz"] = 0.0

    # DC coefficients: sh[:, 0, :] stored as f_dc_0, f_dc_1, f_dc_2
    for i in range(3):
        arr[f"f_dc_{i}"] = sh[:, 0, i]

    # Rest coefficients: sh[:, 1:, :] flattened
    sh_rest = sh[:, 1:, :].reshape(n, -1)  # (N, 45)
    for i in range(num_sh_rest * 3):
        arr[f"f_rest_{i}"] = sh_rest[:, i]

    arr["opacity"] = opacities[:, 0]
    for i in range(3):
        arr[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = rotations[:, i]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))
