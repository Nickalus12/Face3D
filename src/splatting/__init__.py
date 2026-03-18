"""3D Gaussian Splatting module for face reconstruction.

Provides initialization, training, and export of Gaussian splat
representations optimized for face geometry and appearance.
"""

from splatting.initializer import (
    GaussianModel,
    initialize_from_pointcloud,
    initialize_from_flame_mesh,
    initialize_from_flame_binding,
    initialize_from_colmap_sparse,
)
from splatting.trainer import GaussianTrainer, TrainingConfig
from splatting.exporter import (
    export_gaussians_ply,
    export_mesh_from_gaussians,
    extract_mesh_sugar,
    extract_mesh_density_field,
    compute_gaussian_normals,
    export_for_animation,
    export_compressed,
    render_novel_views,
    compute_metrics,
)
from splatting.camera_utils import Camera, load_cameras_from_colmap, generate_turntable_cameras
from splatting.texture_baker import bake_texture, create_spherical_uv, create_material_file


def initialize_gaussians(
    source_path: str,
    output_path: str,
    method: str = "pointcloud",
    **kwargs,
) -> GaussianModel:
    """Initialize Gaussians from a given source.

    Args:
        source_path: Path to the source data (point cloud, mesh, or COLMAP dir).
        output_path: Where to save the initialized Gaussian .ply.
        method: One of 'pointcloud', 'flame', 'colmap_sparse'.
        **kwargs: Forwarded to the chosen initializer.

    Returns:
        Initialized GaussianModel ready for training.
    """
    from pathlib import Path

    if method == "pointcloud":
        return initialize_from_pointcloud(Path(source_path), Path(output_path), **kwargs)
    elif method == "flame":
        return initialize_from_flame_mesh(Path(source_path), **kwargs)
    elif method == "flame_binding":
        model, _metadata = initialize_from_flame_binding(Path(source_path), **kwargs)
        return model
    elif method == "colmap_sparse":
        return initialize_from_colmap_sparse(Path(source_path), **kwargs)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def train_gaussians(
    gaussians: GaussianModel,
    colmap_dir: str,
    images_dir: str,
    output_dir: str,
    depth_dir: str | None = None,
    masks_dir: str | None = None,
    **config_overrides,
) -> GaussianModel:
    """Train Gaussian splats end-to-end.

    Args:
        gaussians: Initialized GaussianModel.
        colmap_dir: Path to COLMAP model directory for camera poses.
        images_dir: Directory of training images.
        output_dir: Directory for checkpoints and outputs.
        depth_dir: Optional depth maps directory for depth supervision.
        masks_dir: Optional masks directory for masked loss.
        **config_overrides: Override any TrainingConfig field.

    Returns:
        Trained GaussianModel.
    """
    from pathlib import Path

    cameras = load_cameras_from_colmap(Path(colmap_dir))
    config = TrainingConfig(**config_overrides)
    trainer = GaussianTrainer(config)
    trained = trainer.train(
        gaussians,
        cameras,
        Path(images_dir),
        depth_dir=Path(depth_dir) if depth_dir else None,
        masks_dir=Path(masks_dir) if masks_dir else None,
        output_dir=Path(output_dir),
    )
    return trained


def export_model(
    gaussians: GaussianModel,
    output_dir: str,
    export_ply: bool = True,
    export_mesh: bool = True,
    export_compress: bool = True,
    render_video: bool = False,
    bake_tex: bool = False,
    cameras: list | None = None,
    flame_binding_data: dict | None = None,
    images_dir: str | None = None,
    texture_resolution: int = 4096,
    flame_texture_path: str | None = None,
) -> dict[str, str]:
    """Export a trained Gaussian model in various formats.

    Mesh extraction uses SuGaR-inspired Poisson reconstruction by default,
    falling back to TSDF fusion if that fails.

    Args:
        gaussians: Trained GaussianModel.
        output_dir: Output directory.
        export_ply: Save as standard 3DGS .ply.
        export_mesh: Extract and save mesh (SuGaR then TSDF fallback).
        export_compress: Export compressed Gaussians via gsplat.
        render_video: Render a turntable video.
        bake_tex: Bake a UV texture atlas onto the extracted mesh.
        cameras: Optional camera list for video rendering and texture baking.
        flame_binding_data: Optional FLAME binding for animation export.
        images_dir: Directory of source images (required for texture baking).
        texture_resolution: Texture atlas resolution (default 4096).
        flame_texture_path: Path to FLAME_texture.npz for built-in UVs.

    Returns:
        Dictionary mapping format names to output file paths.
    """
    import logging
    from pathlib import Path

    _logger = logging.getLogger(__name__)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    if export_ply:
        ply_path = out / "gaussians.ply"
        export_gaussians_ply(gaussians, ply_path)
        results["ply"] = str(ply_path)

    mesh_file_path = None
    if export_mesh:
        mesh_path = out / "mesh"
        mesh = None
        try:
            mesh = extract_mesh_sugar(gaussians, mesh_path)
            mesh_file_path = mesh_path.with_suffix(".obj")
            results["mesh"] = str(mesh_path.with_suffix(".ply"))
            _logger.info("SuGaR mesh extraction succeeded")
        except Exception as e:
            _logger.warning("SuGaR mesh extraction failed (%s); falling back to TSDF", e)
            try:
                tsdf_path = out / "mesh.ply"
                export_mesh_from_gaussians(gaussians, tsdf_path)
                mesh_file_path = tsdf_path
                results["mesh"] = str(tsdf_path)
            except Exception as e2:
                _logger.error("TSDF mesh extraction also failed: %s", e2)

        # Export animation binding if mesh and FLAME data available
        if mesh is not None and flame_binding_data is not None:
            anim_path = out / "mesh_animated"
            try:
                export_for_animation(gaussians, mesh, flame_binding_data, anim_path)
                results["animation"] = str(anim_path.with_suffix(".ply"))
            except Exception as e:
                _logger.warning("Animation export failed: %s", e)

    # Texture baking
    if bake_tex and mesh_file_path is not None and cameras and images_dir:
        try:
            from splatting.texture_baker import bake_texture

            tex_path = bake_texture(
                mesh_path=mesh_file_path,
                cameras=cameras,
                images_dir=Path(images_dir),
                output_dir=out,
                texture_resolution=texture_resolution,
                prefer_photos=True,
                flame_texture_path=flame_texture_path,
            )
            if tex_path is not None:
                results["textured_mesh"] = str(tex_path)
                _logger.info("Texture baking succeeded")
        except Exception as e:
            _logger.warning("Texture baking failed: %s", e)

    if export_compress:
        compress_path = out / "gaussians_compressed"
        try:
            export_compressed(gaussians, compress_path)
            results["compressed"] = str(compress_path)
        except Exception as e:
            _logger.warning("Compressed export failed: %s", e)

    if render_video:
        video_dir = out / "turntable"
        render_novel_views(gaussians, cameras, video_dir)
        results["video"] = str(video_dir)

    return results


__all__ = [
    "GaussianModel",
    "TrainingConfig",
    "GaussianTrainer",
    "Camera",
    "initialize_gaussians",
    "initialize_from_pointcloud",
    "initialize_from_flame_mesh",
    "initialize_from_flame_binding",
    "initialize_from_colmap_sparse",
    "train_gaussians",
    "export_model",
    "export_gaussians_ply",
    "export_mesh_from_gaussians",
    "extract_mesh_sugar",
    "extract_mesh_density_field",
    "compute_gaussian_normals",
    "export_for_animation",
    "export_compressed",
    "render_novel_views",
    "compute_metrics",
    "load_cameras_from_colmap",
    "generate_turntable_cameras",
    "bake_texture",
    "create_spherical_uv",
    "create_material_file",
]
