"""Export trained Gaussian models to various formats and render novel views.

Includes SuGaR-inspired mesh extraction (Poisson reconstruction from flat
Gaussians) and a density-field marching cubes alternative.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from splatting.camera_utils import Camera, generate_turntable_cameras
from splatting.initializer import GaussianModel, SH_COEFFS, _save_gaussians_ply

logger = logging.getLogger(__name__)

# SH basis constant for degree-0 (DC term): 1 / (2 * sqrt(pi))
_SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Original exports (kept for backward compatibility)
# ---------------------------------------------------------------------------


def export_gaussians_ply(gaussians: GaussianModel, output_path: Path) -> None:
    """Save trained Gaussians as a standard 3DGS .ply file.

    Compatible with common Gaussian splatting viewers (e.g. antimatter15,
    playcanvas SuperSplat, etc.).

    Args:
        gaussians: Trained GaussianModel (CPU or GPU).
        output_path: Destination .ply file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_gaussians_ply(gaussians, output_path)
    logger.info(
        "Exported %d Gaussians to %s",
        gaussians.num_gaussians, output_path,
    )


def export_mesh_from_gaussians(
    gaussians: GaussianModel,
    output_path: Path,
    voxel_size: float = 0.005,
    depth_threshold: float = 3.0,
    num_views: int = 120,
    image_size: int = 512,
) -> None:
    """Extract a triangle mesh from Gaussians via TSDF fusion.

    Renders the Gaussian scene from multiple viewpoints and integrates
    the resulting RGB-D images into a TSDF volume, then extracts a mesh.

    Args:
        gaussians: Trained GaussianModel.
        output_path: Where to save the mesh (.ply or .obj).
        voxel_size: TSDF voxel size in meters.
        depth_threshold: Maximum depth for truncation.
        num_views: Number of viewpoints for rendering.
        image_size: Resolution of each rendered view.
    """
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians = gaussians.to(device)

    # Compute scene center and extent from Gaussian positions
    positions = gaussians.positions.detach()
    center = positions.mean(dim=0).cpu().numpy()
    extent = (positions.max(dim=0).values - positions.min(dim=0).values).cpu().numpy()
    radius = float(np.linalg.norm(extent)) * 0.8
    fov = 0.8

    # Generate views around the scene
    cameras = generate_turntable_cameras(
        center=center,
        radius=radius,
        num_views=num_views,
        height=0.0,
        fov=fov,
        image_width=image_size,
        image_height=image_size,
    )

    # Add views at different heights for better coverage
    for h_offset in [-0.3 * radius, 0.3 * radius]:
        cams_h = generate_turntable_cameras(
            center=center,
            radius=radius,
            num_views=num_views // 3,
            height=h_offset,
            fov=fov,
            image_width=image_size,
            image_height=image_size,
        )
        cameras.extend(cams_h)

    # Prepare Gaussian rendering parameters
    scales = torch.exp(gaussians.scales)
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1))
    bg = torch.zeros(3, device=device)

    # Camera intrinsics for Open3D
    fx = image_size / (2.0 * math.tan(fov / 2.0))
    fy = fx
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_size, image_size, fx, fy, image_size / 2.0, image_size / 2.0
    )

    # TSDF volume
    sdf_trunc = voxel_size * 5.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    logger.info("Rendering %d views for TSDF integration...", len(cameras))

    for cam in tqdm(cameras, desc="TSDF integration"):
        viewmat = cam.get_viewmat(device)
        K = cam.get_K(device)

        with torch.no_grad():
            renders, alphas, _ = rasterization(
                means=gaussians.positions,
                quats=gaussians.rotations,
                scales=scales,
                opacities=opacities,
                colors=gaussians.colors_sh,
                viewmats=viewmat[None],
                Ks=K[None],
                width=image_size,
                height=image_size,
                sh_degree=3,
                packed=True,
                render_mode="RGB+ED",
            )

        rgb_np = (renders[0, :, :, :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        depth_np = renders[0, :, :, 3].cpu().numpy().astype(np.float32)

        # Mask out low-confidence and far regions
        alpha_np = alphas[0, :, :, 0].cpu().numpy()
        depth_np[alpha_np < 0.5] = 0.0
        depth_np[depth_np > depth_threshold] = 0.0

        # Convert depth to uint16 (millimeters) for Open3D
        depth_mm = (depth_np * 1000.0).astype(np.uint16)

        color_o3d = o3d.geometry.Image(rgb_np)
        depth_o3d = o3d.geometry.Image(depth_mm)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=depth_threshold,
            convert_rgb_to_intensity=False,
        )

        # Extrinsic: Open3D expects camera-to-world, so invert viewmat
        extrinsic = viewmat.cpu().numpy().astype(np.float64)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic))

        torch.cuda.empty_cache()

    # Extract mesh
    logger.info("Extracting triangle mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    logger.info(
        "Exported mesh with %d vertices and %d triangles to %s",
        len(mesh.vertices), len(mesh.triangles), output_path,
    )


# ---------------------------------------------------------------------------
# SuGaR-inspired mesh extraction
# ---------------------------------------------------------------------------


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) quaternions (w, x, y, z) to (N, 3, 3) rotation matrices."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _sh_dc_to_rgb(sh_dc: torch.Tensor) -> np.ndarray:
    """Convert SH DC coefficients (N, 3) to linear RGB [0, 1] as numpy array."""
    rgb = (sh_dc * _SH_C0 + 0.5).clamp(0.0, 1.0)
    return rgb.cpu().numpy().astype(np.float64)


def compute_gaussian_normals(gaussians: GaussianModel) -> torch.Tensor:
    """Extract surface normal directions from each Gaussian's rotation and scale.

    For each Gaussian the covariance is R @ diag(s^2) @ R^T. The surface
    normal is the eigenvector corresponding to the *smallest* eigenvalue,
    which is the column of R associated with the smallest scale value.

    Args:
        gaussians: A GaussianModel with scales (log-space) and rotations.

    Returns:
        (N, 3) unit normals on CPU.
    """
    device = gaussians.device
    scales = torch.exp(gaussians.scales)  # (N, 3)
    quats = gaussians.rotations          # (N, 4)

    # Normalize quaternions
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Build rotation matrices: columns are the local axes
    R = _quat_to_rotmat(quats)  # (N, 3, 3)

    # The smallest-scale axis index per Gaussian
    min_axis = scales.argmin(dim=-1)  # (N,)

    # Gather the column of R corresponding to the smallest scale
    # R[:, :, axis] gives the direction of that axis in world frame
    idx = min_axis.unsqueeze(1).unsqueeze(2).expand(-1, 3, 1)  # (N, 3, 1)
    normals = torch.gather(R, 2, idx).squeeze(2)  # (N, 3)

    # Normalize (should already be unit, but numerics)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return normals.detach().cpu()


def extract_mesh_sugar(
    gaussians: GaussianModel,
    output_path: Path,
    resolution: int = 256,
    density_threshold: float = 0.5,
    poisson_depth: int = 8,
    decimate_target: int = 100_000,
) -> o3d.geometry.TriangleMesh:
    """Extract a clean mesh from Gaussians using SuGaR-inspired Poisson reconstruction.

    The key insight from SuGaR (Guedon & Lepetit, CVPR 2024) is that
    well-trained Gaussians tend to become flat (one scale axis much
    smaller than the other two) and align with scene surfaces.  We
    identify these flat Gaussians, treat their positions as oriented
    surface samples, and run Poisson reconstruction to obtain a
    watertight mesh.

    Steps:
        1. Filter to "flat" Gaussians (smallest scale / mean of other two < 0.1).
        2. Extract positions as surface points, normals from smallest-scale axis.
        3. Sample colors from the SH DC term.
        4. Run Open3D Poisson surface reconstruction.
        5. Remove low-density vertices and small connected components.
        6. Optionally decimate to *decimate_target* triangles.
        7. Transfer vertex colors from nearest Gaussian.
        8. Save as .ply and .obj.

    Args:
        gaussians: Trained GaussianModel.
        output_path: Base path for the output mesh (saved as .ply and .obj).
        resolution: Unused (kept for API symmetry with density_field method).
        density_threshold: Percentile (0-1) below which Poisson density
            vertices are removed.  Lower keeps more geometry.
        poisson_depth: Octree depth for Poisson reconstruction (higher = finer).
        decimate_target: Target triangle count for mesh simplification.
            Set to 0 to skip decimation.

    Returns:
        The extracted Open3D TriangleMesh.
    """
    device = gaussians.device
    n = gaussians.num_gaussians
    logger.info("SuGaR mesh extraction: %d total Gaussians", n)

    # --- 1. Identify flat (surface-like) Gaussians ---
    scales = torch.exp(gaussians.scales).detach()  # (N, 3)
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1)).detach()  # (N,)

    # Sort scales per Gaussian to find the smallest axis
    sorted_scales, _ = scales.sort(dim=-1)  # ascending
    smallest = sorted_scales[:, 0]
    other_mean = sorted_scales[:, 1:].mean(dim=-1)

    flatness_ratio = smallest / other_mean.clamp(min=1e-8)

    # Keep Gaussians that are flat AND reasonably opaque
    flat_mask = (flatness_ratio < 0.1) & (opacities > 0.05)
    flat_indices = flat_mask.nonzero(as_tuple=True)[0]

    n_flat = len(flat_indices)
    logger.info(
        "Found %d flat Gaussians (%.1f%% of total)",
        n_flat, 100.0 * n_flat / max(n, 1),
    )

    if n_flat < 100:
        # Not enough flat Gaussians -- relax the threshold progressively
        for threshold in [0.2, 0.3, 0.5]:
            flat_mask = (flatness_ratio < threshold) & (opacities > 0.02)
            flat_indices = flat_mask.nonzero(as_tuple=True)[0]
            n_flat = len(flat_indices)
            logger.info(
                "Relaxed flatness to %.1f: %d Gaussians", threshold, n_flat
            )
            if n_flat >= 100:
                break

    if n_flat < 50:
        raise RuntimeError(
            f"Only {n_flat} flat Gaussians found; not enough for Poisson "
            "reconstruction. The Gaussians may not be well-trained or the "
            "scene may lack clear surfaces."
        )

    # --- 2. Extract positions and normals for flat Gaussians ---
    flat_positions = gaussians.positions[flat_indices].detach().cpu()  # (M, 3)

    # Compute normals for ALL Gaussians then index
    all_normals = compute_gaussian_normals(gaussians)  # (N, 3) on CPU
    flat_normals = all_normals[flat_indices.cpu()]  # (M, 3)

    # --- 3. Sample colors from SH DC coefficients ---
    sh_dc = gaussians.colors_sh[flat_indices, 0, :].detach()  # (M, 3)
    flat_colors = _sh_dc_to_rgb(sh_dc)  # (M, 3) numpy float64

    # --- 4. Build oriented point cloud and run Poisson reconstruction ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flat_positions.numpy().astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(flat_normals.numpy().astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(flat_colors)

    # Orient normals consistently (important for Poisson)
    pcd.orient_normals_consistent_tangent_plane(k=15)

    logger.info(
        "Running Poisson reconstruction (depth=%d) on %d oriented points...",
        poisson_depth, n_flat,
    )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False,
    )
    logger.info(
        "Poisson output: %d vertices, %d triangles",
        len(mesh.vertices), len(mesh.triangles),
    )

    # --- 5. Remove low-density vertices ---
    densities = np.asarray(densities)
    if len(densities) > 0 and density_threshold > 0:
        threshold_val = np.quantile(densities, density_threshold)
        vertices_to_remove = densities < threshold_val
        mesh.remove_vertices_by_mask(vertices_to_remove)
        logger.info(
            "After density filtering (%.0f%% percentile): %d vertices, %d triangles",
            density_threshold * 100, len(mesh.vertices), len(mesh.triangles),
        )

    # Remove small connected components
    if len(mesh.triangles) > 0:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if len(cluster_n_triangles) > 1:
            # Keep only the largest cluster
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
            logger.info(
                "After removing small components: %d vertices, %d triangles",
                len(mesh.vertices), len(mesh.triangles),
            )

    # --- 6. Decimate ---
    if decimate_target > 0 and len(mesh.triangles) > decimate_target:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=decimate_target)
        logger.info(
            "Decimated to %d vertices, %d triangles",
            len(mesh.vertices), len(mesh.triangles),
        )

    # --- 7. Vertex coloring via nearest-Gaussian lookup ---
    mesh_verts = np.asarray(mesh.vertices).astype(np.float32)
    if len(mesh_verts) > 0:
        # Use all Gaussians (not just flat ones) for color lookup to get
        # the best color at each vertex location
        all_positions = gaussians.positions.detach().cpu().numpy().astype(np.float32)
        all_sh_dc = gaussians.colors_sh[:, 0, :].detach()  # (N, 3)
        all_colors = _sh_dc_to_rgb(all_sh_dc)  # (N, 3) float64

        # Build KD-tree on all Gaussian positions
        gauss_pcd = o3d.geometry.PointCloud()
        gauss_pcd.points = o3d.utility.Vector3dVector(all_positions.astype(np.float64))
        kdtree = o3d.geometry.KDTreeFlann(gauss_pcd)

        vertex_colors = np.zeros((len(mesh_verts), 3), dtype=np.float64)
        for i, v in enumerate(mesh_verts):
            _, idx, _ = kdtree.search_knn_vector_3d(v.astype(np.float64), 1)
            vertex_colors[i] = all_colors[idx[0]]

        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Recompute normals for clean shading
    mesh.compute_vertex_normals()

    # --- 8. Save ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as .ply
    ply_path = output_path.with_suffix(".ply")
    o3d.io.write_triangle_mesh(str(ply_path), mesh)
    logger.info("Saved SuGaR mesh (PLY) to %s", ply_path)

    # Save as .obj
    obj_path = output_path.with_suffix(".obj")
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    logger.info("Saved SuGaR mesh (OBJ) to %s", obj_path)

    return mesh


def extract_mesh_density_field(
    gaussians: GaussianModel,
    output_path: Path,
    grid_resolution: int = 256,
    iso_value: float = 0.5,
) -> o3d.geometry.TriangleMesh:
    """Extract a mesh by evaluating a Gaussian density field on a 3D grid.

    For each grid point, compute the sum of weighted Gaussian contributions:
        density(x) = sum_i [ opacity_i * exp(-0.5 * (x - mu_i)^T Sigma_i^{-1} (x - mu_i)) ]
    Then run marching cubes at *iso_value* to extract the isosurface.

    This is simpler than the Poisson approach but can be slower for large
    grids and many Gaussians.  The computation runs on GPU in batched
    chunks to fit within VRAM.

    Args:
        gaussians: Trained GaussianModel.
        output_path: Where to save the mesh (.ply / .obj).
        grid_resolution: Number of voxels along each axis.
        iso_value: Density threshold for the isosurface.

    Returns:
        The extracted Open3D TriangleMesh.
    """
    from skimage.measure import marching_cubes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians = gaussians.to(device)

    positions = gaussians.positions.detach()           # (N, 3)
    scales = torch.exp(gaussians.scales.detach())      # (N, 3)
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1).detach())  # (N,)
    quats = gaussians.rotations.detach()               # (N, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    N = positions.shape[0]

    # Compute bounding box with padding
    pos_min = positions.min(dim=0).values
    pos_max = positions.max(dim=0).values
    extent = pos_max - pos_min
    padding = extent * 0.1
    grid_min = pos_min - padding
    grid_max = pos_max + padding
    grid_extent = grid_max - grid_min

    logger.info(
        "Density field: %d^3 grid, %d Gaussians, bounding box size %.4f",
        grid_resolution, N, grid_extent.norm().item(),
    )

    # Build rotation matrices and inverse covariance matrices on GPU
    R = _quat_to_rotmat(quats)                     # (N, 3, 3)
    S = torch.diag_embed(scales)                   # (N, 3, 3)
    # Covariance: Sigma = R @ S^2 @ R^T
    # Inverse covariance: Sigma^{-1} = R @ S^{-2} @ R^T
    S_inv_sq = torch.diag_embed(1.0 / (scales ** 2).clamp(min=1e-12))  # (N, 3, 3)
    Sigma_inv = R @ S_inv_sq @ R.transpose(1, 2)   # (N, 3, 3)

    # Create the 3D grid coordinates
    lin = torch.linspace(0, 1, grid_resolution, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid_points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (R^3, 3)
    # Scale to world coordinates
    grid_points = grid_points * grid_extent[None, :] + grid_min[None, :]

    total_points = grid_points.shape[0]
    density = torch.zeros(total_points, device=device)

    # Process in chunks to manage VRAM (RTX 3080 16GB)
    # Chunk the grid points; for each chunk evaluate all Gaussians
    point_chunk_size = max(1, min(4096, total_points))
    gauss_chunk_size = max(1, min(8192, N))

    logger.info("Evaluating density field in chunks...")
    for p_start in tqdm(range(0, total_points, point_chunk_size), desc="Density field"):
        p_end = min(p_start + point_chunk_size, total_points)
        pts = grid_points[p_start:p_end]  # (P, 3)
        chunk_density = torch.zeros(p_end - p_start, device=device)

        for g_start in range(0, N, gauss_chunk_size):
            g_end = min(g_start + gauss_chunk_size, N)

            mu = positions[g_start:g_end]           # (G, 3)
            S_inv_chunk = Sigma_inv[g_start:g_end]  # (G, 3, 3)
            opa = opacities[g_start:g_end]           # (G,)

            diff = pts[:, None, :] - mu[None, :, :]  # (P, G, 3)
            # Mahalanobis: diff^T @ Sigma_inv @ diff
            # (P, G, 3) @ (G, 3, 3) -> (P, G, 3)
            tmp = torch.einsum("pgj,gjk->pgk", diff, S_inv_chunk)
            mahal = (tmp * diff).sum(dim=-1)  # (P, G)

            gauss_val = opa[None, :] * torch.exp(-0.5 * mahal)  # (P, G)
            chunk_density += gauss_val.sum(dim=1)  # (P,)

        density[p_start:p_end] = chunk_density

    # Reshape to 3D grid
    density_grid = density.cpu().numpy().reshape(grid_resolution, grid_resolution, grid_resolution)

    logger.info(
        "Density stats: min=%.4f, max=%.4f, mean=%.4f",
        density_grid.min(), density_grid.max(), density_grid.mean(),
    )

    # Adaptive iso_value if needed
    if density_grid.max() < iso_value:
        iso_value = density_grid.max() * 0.5
        logger.warning("Max density below iso_value; using adaptive iso=%.4f", iso_value)

    if density_grid.max() < 1e-6:
        raise RuntimeError("Density field is essentially zero everywhere")

    # --- Marching cubes ---
    try:
        verts, faces, normals_mc, _ = marching_cubes(
            density_grid, level=iso_value, spacing=(
                grid_extent[0].item() / grid_resolution,
                grid_extent[1].item() / grid_resolution,
                grid_extent[2].item() / grid_resolution,
            ),
        )
    except ValueError as e:
        raise RuntimeError(f"Marching cubes failed: {e}") from e

    # Offset vertices to world coordinates
    verts = verts + grid_min.cpu().numpy()

    logger.info(
        "Marching cubes: %d vertices, %d faces", len(verts), len(faces),
    )

    # Build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    # --- Color vertices by nearest Gaussian ---
    all_positions_np = positions.detach().cpu().numpy().astype(np.float64)
    all_sh_dc = gaussians.colors_sh[:, 0, :].detach()
    all_colors = _sh_dc_to_rgb(all_sh_dc)

    gauss_pcd = o3d.geometry.PointCloud()
    gauss_pcd.points = o3d.utility.Vector3dVector(all_positions_np)
    kdtree = o3d.geometry.KDTreeFlann(gauss_pcd)

    mesh_verts = np.asarray(mesh.vertices)
    vertex_colors = np.zeros((len(mesh_verts), 3), dtype=np.float64)
    for i, v in enumerate(mesh_verts):
        _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
        vertex_colors[i] = all_colors[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # --- Save ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ply_path = output_path.with_suffix(".ply")
    o3d.io.write_triangle_mesh(str(ply_path), mesh)
    logger.info("Saved density-field mesh (PLY) to %s", ply_path)

    obj_path = output_path.with_suffix(".obj")
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    logger.info("Saved density-field mesh (OBJ) to %s", obj_path)

    return mesh


# ---------------------------------------------------------------------------
# Animation / FLAME binding export
# ---------------------------------------------------------------------------


def export_for_animation(
    gaussians: GaussianModel,
    mesh: o3d.geometry.TriangleMesh,
    flame_binding_data: dict | None,
    output_path: Path,
) -> None:
    """Export mesh with FLAME vertex correspondence for downstream animation.

    If FLAME binding data is available (triangle indices + barycentric
    coords), this function saves the mesh alongside a JSON metadata file
    that maps mesh vertices to FLAME vertices, enabling animation by
    driving the FLAME model and having the mesh follow.

    Args:
        gaussians: Trained GaussianModel (for reference).
        mesh: The extracted triangle mesh (e.g. from ``extract_mesh_sugar``).
        flame_binding_data: Dictionary with keys:
            - ``"triangle_indices"``: (V,) int array of FLAME triangle indices.
            - ``"barycentric_coords"``: (V, 3) float array of barycentric weights.
            - ``"flame_vertices"``: (5023, 3) FLAME template vertex positions.
            May be ``None`` if no FLAME fitting was performed.
        output_path: Base path for output files.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the mesh itself
    mesh_ply = output_path.with_suffix(".ply")
    o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
    logger.info("Saved animation mesh to %s", mesh_ply)

    if flame_binding_data is None:
        logger.warning(
            "No FLAME binding data provided; mesh saved without "
            "animation correspondence."
        )
        return

    # Build binding metadata
    binding = {}

    tri_indices = flame_binding_data.get("triangle_indices")
    bary_coords = flame_binding_data.get("barycentric_coords")
    flame_verts = flame_binding_data.get("flame_vertices")

    if tri_indices is not None:
        if isinstance(tri_indices, torch.Tensor):
            tri_indices = tri_indices.detach().cpu().numpy()
        binding["triangle_indices"] = tri_indices.tolist()

    if bary_coords is not None:
        if isinstance(bary_coords, torch.Tensor):
            bary_coords = bary_coords.detach().cpu().numpy()
        binding["barycentric_coords"] = bary_coords.tolist()

    if flame_verts is not None:
        if isinstance(flame_verts, torch.Tensor):
            flame_verts = flame_verts.detach().cpu().numpy()
        binding["flame_vertices_shape"] = list(flame_verts.shape)
        # Save FLAME vertices as a separate numpy file
        flame_npy = output_path.with_name(output_path.stem + "_flame_verts.npy")
        np.save(str(flame_npy), flame_verts.astype(np.float32))
        binding["flame_vertices_file"] = flame_npy.name
        logger.info("Saved FLAME vertices to %s", flame_npy)

    binding["num_mesh_vertices"] = len(mesh.vertices)
    binding["num_mesh_triangles"] = len(mesh.triangles)

    meta_path = output_path.with_name(output_path.stem + "_binding.json")
    with open(meta_path, "w") as f:
        json.dump(binding, f, indent=2)
    logger.info("Saved FLAME binding metadata to %s", meta_path)


# ---------------------------------------------------------------------------
# gsplat compression export
# ---------------------------------------------------------------------------


def export_compressed(
    gaussians: GaussianModel,
    output_path: Path,
) -> None:
    """Export Gaussians in a compressed format using gsplat's PngCompression.

    Achieves substantial file-size reduction (e.g. 236 MB -> 16.5 MB for
    1M Gaussians) by quantizing parameters and encoding them as PNG
    images with K-means SH clustering.

    Falls back to ``gsplat.export_splats`` for compressed PLY if
    PngCompression is unavailable, and finally to a standard PLY export.

    Args:
        gaussians: Trained GaussianModel.
        output_path: Output path.  For PngCompression this is used as a
            directory; for PLY formats it is the file path.
    """
    output_path = Path(output_path)

    # Prepare the splats dict that gsplat compression expects
    device = gaussians.device
    scales = torch.exp(gaussians.scales.detach())
    quats = gaussians.rotations.detach()
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1).detach())
    sh_all = gaussians.colors_sh.detach()  # (N, 16, 3)

    splats = {
        "means": gaussians.positions.detach(),
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh_all[:, :1, :],          # (N, 1, 3)
        "shN": sh_all[:, 1:, :],          # (N, 15, 3)
    }

    # --- Try PngCompression first ---
    try:
        from gsplat.compression import PngCompression

        compress_dir = output_path.with_suffix("") if output_path.suffix else output_path
        compress_dir = Path(str(compress_dir) + "_compressed")
        compress_dir.mkdir(parents=True, exist_ok=True)

        compressor = PngCompression()
        compressor.compress(compress_dir, splats)

        logger.info(
            "Exported compressed Gaussians via PngCompression to %s",
            compress_dir,
        )
        return
    except ImportError:
        logger.info("gsplat.compression.PngCompression not available; trying export_splats")
    except Exception as e:
        logger.warning("PngCompression failed (%s); trying export_splats", e)

    # --- Try export_splats for compressed PLY ---
    try:
        from gsplat import export_splats

        ply_path = output_path.with_suffix(".ply")
        ply_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            export_splats(
                means=splats["means"],
                scales=splats["scales"],
                quats=splats["quats"],
                opacities=splats["opacities"],
                sh0=splats["sh0"],
                shN=splats["shN"],
                format="ply_compressed",
                save_to=str(ply_path),
            )
        logger.info("Exported compressed PLY via export_splats to %s", ply_path)
        return
    except (ImportError, AttributeError):
        logger.info("gsplat.export_splats not available; falling back to standard PLY")
    except Exception as e:
        logger.warning("export_splats failed (%s); falling back to standard PLY", e)

    # --- Fallback: standard PLY ---
    ply_path = output_path.with_suffix(".ply")
    export_gaussians_ply(gaussians, ply_path)
    logger.info("Exported standard (uncompressed) PLY to %s", ply_path)


# ---------------------------------------------------------------------------
# Novel view rendering
# ---------------------------------------------------------------------------


def render_novel_views(
    gaussians: GaussianModel,
    cameras: list[Camera] | None = None,
    output_dir: Path | str = "novel_views",
    num_views: int = 60,
) -> Path:
    """Render a turntable sequence of novel views around the face.

    If cameras is None, generates turntable cameras automatically from
    the Gaussian positions.

    Args:
        gaussians: Trained GaussianModel.
        cameras: Optional pre-defined camera list. If None, generates turntable.
        output_dir: Directory to save rendered frames.
        num_views: Number of views (used only when cameras is None).

    Returns:
        Path to the output directory containing rendered frames.
    """
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians = gaussians.to(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cameras is None:
        positions = gaussians.positions.detach()
        center = positions.mean(dim=0).cpu().numpy()
        extent = (positions.max(dim=0).values - positions.min(dim=0).values).cpu().numpy()
        radius = float(np.linalg.norm(extent)) * 0.7
        cameras = generate_turntable_cameras(
            center=center,
            radius=radius,
            num_views=num_views,
            fov=0.7854,
            image_width=800,
            image_height=800,
        )

    scales = torch.exp(gaussians.scales)
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1))
    bg = torch.ones(3, device=device)  # white background for face renders

    logger.info("Rendering %d novel views...", len(cameras))

    for i, cam in enumerate(tqdm(cameras, desc="Rendering novel views")):
        viewmat = cam.get_viewmat(device)
        K = cam.get_K(device)

        with torch.no_grad():
            renders, alphas, _ = rasterization(
                means=gaussians.positions,
                quats=gaussians.rotations,
                scales=scales,
                opacities=opacities,
                colors=gaussians.colors_sh,
                viewmats=viewmat[None],
                Ks=K[None],
                width=cam.width,
                height=cam.height,
                sh_degree=3,
                packed=True,
            )

        rgb = renders[0].cpu().numpy()  # (H, W, 3)
        alpha = alphas[0, :, :, 0].cpu().numpy()  # (H, W)

        # Composite onto white background
        rgb_bg = rgb * alpha[:, :, None] + (1.0 - alpha[:, :, None])
        img = (rgb_bg * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        frame_path = output_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), img_bgr)

        if i % 20 == 0:
            torch.cuda.empty_cache()

    # Optionally create video if ffmpeg is available
    video_path = output_dir / "turntable.mp4"
    try:
        import subprocess

        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", "30",
                "-i", str(output_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", str(video_path),
            ],
            capture_output=True,
            check=True,
        )
        logger.info("Created turntable video at %s", video_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.info("ffmpeg not available; frames saved to %s", output_dir)

    return output_dir


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    gaussians: GaussianModel,
    cameras: list[Camera],
    images_dir: Path,
) -> dict[str, float]:
    """Compute PSNR, SSIM, and LPIPS on held-out test views.

    Args:
        gaussians: Trained GaussianModel.
        cameras: List of test Camera objects.
        images_dir: Directory containing ground-truth images.

    Returns:
        Dictionary with average PSNR, SSIM, and LPIPS values.
    """
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians = gaussians.to(device)

    # Try to load LPIPS
    lpips_fn = None
    try:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
    except ImportError:
        logger.warning("lpips not installed; LPIPS metric will be skipped")

    scales = torch.exp(gaussians.scales)
    opacities = torch.sigmoid(gaussians.opacities.squeeze(-1))

    psnr_values = []
    ssim_values = []
    lpips_values = []

    logger.info("Computing metrics on %d test views...", len(cameras))

    for cam in tqdm(cameras, desc="Computing metrics"):
        # Load ground truth
        if cam.image_path is None:
            continue
        img_path = images_dir / cam.image_path
        if not img_path.exists():
            # Try common extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = images_dir / (Path(cam.image_path).stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
        if not img_path.exists():
            continue

        gt_img = cv2.imread(str(img_path))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if gt_img.shape[0] != cam.height or gt_img.shape[1] != cam.width:
            gt_img = cv2.resize(gt_img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
        gt_tensor = torch.from_numpy(gt_img).to(device)  # (H, W, 3)

        viewmat = cam.get_viewmat(device)
        K = cam.get_K(device)

        with torch.no_grad():
            renders, alphas, _ = rasterization(
                means=gaussians.positions,
                quats=gaussians.rotations,
                scales=scales,
                opacities=opacities,
                colors=gaussians.colors_sh,
                viewmats=viewmat[None],
                Ks=K[None],
                width=cam.width,
                height=cam.height,
                sh_degree=3,
                packed=True,
            )

        rendered = renders[0]  # (H, W, 3)

        # PSNR
        mse = torch.mean((rendered - gt_tensor) ** 2).item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))
        psnr_values.append(psnr)

        # SSIM
        ssim_val = _compute_ssim(rendered, gt_tensor)
        ssim_values.append(ssim_val)

        # LPIPS
        if lpips_fn is not None:
            # LPIPS expects (B, C, H, W) in [0, 1]
            pred_lpips = rendered.permute(2, 0, 1).unsqueeze(0)
            gt_lpips = gt_tensor.permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                lpips_val = lpips_fn(pred_lpips, gt_lpips).item()
            lpips_values.append(lpips_val)

        torch.cuda.empty_cache()

    results = {
        "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
        "ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
    }
    if lpips_values:
        results["lpips"] = float(np.mean(lpips_values))

    logger.info(
        "Metrics: PSNR=%.2f, SSIM=%.4f%s",
        results["psnr"],
        results["ssim"],
        f", LPIPS={results.get('lpips', 'N/A')}" if "lpips" in results else "",
    )

    return results


def _compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM between two (H, W, 3) tensors. Returns scalar."""
    import torch.nn.functional as F

    x = img1.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    y = img2.permute(2, 0, 1).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel_size = 11
    sigma = 1.5
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(3, -1, -1, -1)

    pad = kernel_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad, groups=3)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=3)

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=3) - mu_x ** 2
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=3) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=3) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    return ssim_map.mean().item()
