"""Integration layer connecting the ML prior system to the Face3D pipeline.

This module provides the two main entry points:
    1. post_scan_update() — called after a successful pipeline run to add
       scan data to the replay buffer and retrain the prior network.
    2. get_prior_initialization() — called before Gaussian initialization
       to predict better starting values from the prior network + physics.

Data flow:
    Pipeline completes scan
        -> post_scan_update(session_dir, buffer, prior)
            -> buffer.add_scan(...)
            -> prior.train_on_buffer(buffer)  [if >= 5 scans]
            -> prior.save(weights_path)

    Pipeline starts new scan
        -> get_prior_initialization(prior, refiner, flame_params)
            -> prior.predict_initialization(shape, expr)
            -> refiner.refine(vertices, offsets)
            -> return initialization dict for GaussianModel
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_BUFFER_DIR = "data/ml/replay_buffer"
DEFAULT_WEIGHTS_PATH = "data/ml/prior_weights.pth"
MIN_SCANS_FOR_TRAINING = 5


def post_scan_update(
    session_dir: Path,
    buffer,  # ScanReplayBuffer
    prior,   # FacePriorNetwork
    weights_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Called after a successful pipeline run to update the learning system.

    Steps:
        1. Collect scan data from session output directory
        2. Add scan data to the replay buffer
        3. If buffer has >= MIN_SCANS_FOR_TRAINING scans, retrain prior network
        4. Save updated prior weights

    Args:
        session_dir: Path to the session's processed data directory, e.g.
            data/processed/{session}/ containing frames/, flame/, colmap/, etc.
        buffer: ScanReplayBuffer instance.
        prior: FacePriorNetwork instance.
        weights_path: Where to save prior weights. Defaults to
            data/ml/prior_weights.pth relative to session_dir's root.

    Returns:
        Dict with 'scan_added' (bool), 'prior_trained' (bool),
        'buffer_count' (int), 'training_result' (dict or None).
    """
    session_dir = Path(session_dir)
    session_id = session_dir.name

    result = {
        "scan_added": False,
        "prior_trained": False,
        "buffer_count": buffer.scan_count(),
        "training_result": None,
    }

    # Resolve default weights path
    if weights_path is None:
        # Walk up from session_dir to find project data root
        project_root = _find_project_root(session_dir)
        weights_path = project_root / DEFAULT_WEIGHTS_PATH

    # ── 1. Collect scan data ──────────────────────────────────────────
    try:
        frames_dir = _find_frames_dir(session_dir)
        flame_params = _load_flame_params(session_dir)
        gaussian_model = _load_gaussian_model(session_dir)
        cameras = _load_cameras(session_dir)
        training_metrics = _load_training_metrics(session_dir)
    except FileNotFoundError as e:
        logger.warning("Could not collect scan data for '%s': %s", session_id, e)
        return result

    # ── 2. Add to replay buffer ───────────────────────────────────────
    try:
        buffer.add_scan(
            session_id=session_id,
            frames_dir=frames_dir,
            flame_params=flame_params,
            gaussian_model=gaussian_model,
            cameras=cameras,
            training_metrics=training_metrics,
        )
        result["scan_added"] = True
        result["buffer_count"] = buffer.scan_count()
        logger.info("Added scan '%s' to replay buffer (%d total)", session_id, result["buffer_count"])
    except Exception as e:
        logger.error("Failed to add scan '%s' to buffer: %s", session_id, e)
        return result

    # ── 3. Retrain prior if enough scans ──────────────────────────────
    if result["buffer_count"] >= MIN_SCANS_FOR_TRAINING:
        try:
            training_result = prior.train_on_buffer(buffer, epochs=100, lr=1e-3)
            result["prior_trained"] = True
            result["training_result"] = training_result
            logger.info("Prior network retrained: loss=%.6f", training_result.get("final_loss", -1))
        except Exception as e:
            logger.error("Prior network training failed: %s", e)

        # ── 4. Save weights ───────────────────────────────────────────
        if result["prior_trained"]:
            try:
                prior.save(weights_path)
            except Exception as e:
                logger.error("Failed to save prior weights: %s", e)

    return result


def get_prior_initialization(
    prior,    # FacePriorNetwork
    refiner,  # TaichiMeshRefiner
    flame_params: dict,
    flame_vertices: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Get physics-refined Gaussian initialization from the prior network.

    Steps:
        1. Run prior network to predict per-region offsets/scales/opacities
        2. If flame_vertices provided, apply Taichi physics refinement
        3. Return initialization dict suitable for GaussianModel creation

    Args:
        prior: Trained FacePriorNetwork instance.
        refiner: TaichiMeshRefiner instance (with FLAME topology).
        flame_params: Dict with 'shape_params' and 'expression_params' arrays.
        flame_vertices: Optional (5023, 3) base FLAME vertices for physics
            refinement. If None, physics refinement is skipped.

    Returns:
        Dict with:
            'offsets': (num_regions, 3) or (n_vertices, 3) position offsets
            'scales': (num_regions, 3) log-scale factors per region
            'opacities': (num_regions,) opacity values [0, 1]
            'refined_vertices': (n_vertices, 3) if flame_vertices provided
            'normals': (n_vertices, 3) surface normals if vertices provided
    """
    shape_params = np.asarray(flame_params.get("shape_params", np.zeros(100)), dtype=np.float32)
    expr_params = np.asarray(
        flame_params.get("expression_params", np.zeros(50)), dtype=np.float32
    )

    # 1. Predict from prior network
    prediction = prior.predict_initialization(shape_params, expr_params)

    result = {
        "offsets": prediction["offsets"],
        "scales": prediction["scales"],
        "opacities": prediction["opacities"],
    }

    # 2. Physics refinement (if vertices available)
    if flame_vertices is not None:
        flame_vertices = np.asarray(flame_vertices, dtype=np.float32)

        try:
            refined = refiner.refine(
                vertices=flame_vertices,
                offsets=prediction["offsets"],
                iterations=50,
                lambda_smooth=0.3,
            )
            result["refined_vertices"] = refined
            result["normals"] = refiner.compute_surface_normals(refined)
            logger.info("Physics refinement applied to %d vertices", len(refined))
        except Exception as e:
            logger.warning("Physics refinement failed, using raw predictions: %s", e)
            # Fall back to raw offset application
            if flame_vertices is not None:
                expanded = refiner._expand_region_offsets(prediction["offsets"])
                result["refined_vertices"] = flame_vertices + expanded
                result["normals"] = refiner.compute_surface_normals(flame_vertices + expanded)

    return result


# ---------------------------------------------------------------------------
# Private helpers for collecting scan data from session directories
# ---------------------------------------------------------------------------

def _find_project_root(session_dir: Path) -> Path:
    """Walk up from session_dir to find the project root (contains src/)."""
    current = session_dir
    for _ in range(10):
        if (current / "src").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback: assume session_dir is under data/processed/{session}
    return session_dir.parent.parent.parent


def _find_frames_dir(session_dir: Path) -> Path:
    """Find the frames directory within a session."""
    candidates = [
        session_dir / "frames",
        session_dir / "frames_filtered",
        session_dir / "frames_corrected",
    ]
    for d in candidates:
        if d.is_dir() and any(d.iterdir()):
            return d
    raise FileNotFoundError(f"No frames directory found in {session_dir}")


def _load_flame_params(session_dir: Path) -> dict:
    """Load FLAME parameters from session output."""
    candidates = [
        session_dir / "flame" / "flame_params.npz",
        session_dir / "flame" / "flame_fit_result.npz",
        session_dir / "flame_params.npz",
    ]
    for p in candidates:
        if p.exists():
            data = np.load(str(p))
            return {k: data[k] for k in data.files}
    raise FileNotFoundError(f"No FLAME params found in {session_dir}")


def _load_gaussian_model(session_dir: Path) -> Any:
    """Load a trained Gaussian model (or its stats) from session output.

    Returns a lightweight object with positions, scales, opacities attributes
    extracted from the PLY file, not the full GaussianModel.
    """
    from plyfile import PlyData

    candidates = [
        session_dir / "gaussians.ply",
        session_dir.parent.parent / "output" / session_dir.name / "gaussians.ply",
    ]
    # Also check parent output directory patterns
    session_name = session_dir.name
    output_dir = session_dir.parent.parent / "output" / session_name
    if output_dir.is_dir():
        candidates.extend(output_dir.glob("*.ply"))

    for p in candidates:
        p = Path(p)
        if p.exists():
            try:
                ply = PlyData.read(str(p))
                vertex = ply["vertex"]

                positions = np.column_stack([
                    vertex["x"], vertex["y"], vertex["z"]
                ]).astype(np.float32)

                scales = np.column_stack([
                    vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]
                ]).astype(np.float32)

                opacities = np.array(vertex["opacity"]).reshape(-1, 1).astype(np.float32)

                # Return a simple namespace object
                class _LightGaussians:
                    pass

                model = _LightGaussians()
                model.positions = positions
                model.scales = scales
                model.opacities = opacities
                return model
            except Exception as e:
                logger.debug("Could not load PLY from %s: %s", p, e)
                continue

    raise FileNotFoundError(f"No Gaussian model PLY found for {session_dir}")


def _load_cameras(session_dir: Path) -> list[dict]:
    """Load camera parameters from session data."""
    # Try JSON cameras first
    cam_json = session_dir / "cameras.json"
    if cam_json.exists():
        with open(cam_json) as f:
            return json.load(f)

    # Try COLMAP cameras
    try:
        from utils.colmap_io import read_cameras_binary, read_images_binary

        colmap_dir = session_dir / "colmap" / "sparse" / "0"
        if not colmap_dir.is_dir():
            colmap_dir = session_dir / "colmap" / "sparse"

        if colmap_dir.is_dir():
            cameras_bin = colmap_dir / "cameras.bin"
            images_bin = colmap_dir / "images.bin"
            if cameras_bin.exists() and images_bin.exists():
                cameras = read_cameras_binary(cameras_bin)
                images = read_images_binary(images_bin)
                cam_list = []
                for img_id, img in images.items():
                    cam = cameras[img.camera_id]
                    K = np.eye(3, dtype=np.float32)
                    K[0, 0] = cam.params[0]  # fx
                    K[1, 1] = cam.params[1]  # fy
                    K[0, 2] = cam.params[2]  # cx
                    K[1, 2] = cam.params[3]  # cy
                    cam_list.append({
                        "K": K,
                        "R": img.qvec2rotmat().astype(np.float32),
                        "t": img.tvec.astype(np.float32),
                        "width": cam.width,
                        "height": cam.height,
                    })
                return cam_list
    except Exception as e:
        logger.debug("Could not load COLMAP cameras: %s", e)

    # Return empty list if no cameras found (non-fatal)
    logger.debug("No camera data found for %s", session_dir)
    return []


def _load_training_metrics(session_dir: Path) -> dict:
    """Load training metrics from session output."""
    candidates = [
        session_dir / "training_metrics.json",
        session_dir / "train_log.json",
        session_dir.parent.parent / "output" / session_dir.name / "training_metrics.json",
    ]
    for p in candidates:
        p = Path(p)
        if p.exists():
            with open(p) as f:
                return json.load(f)

    # Return defaults if no metrics file found
    return {
        "final_loss": -1.0,
        "iterations": 0,
        "convergence_rate": 0.0,
        "num_gaussians": 0,
    }
