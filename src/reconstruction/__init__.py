"""Reconstruction module for 3D face reconstruction using FLAME parametric model.

Provides COLMAP-based multi-view reconstruction, facial landmark detection,
FLAME model fitting, and face segmentation.
"""

from reconstruction.colmap_runner import run_colmap, run_colmap_dense
from reconstruction.landmarks import FaceLandmarkDetector, load_mediapipe_to_flame_mapping
from reconstruction.flame_model import FLAMEModel, load_flame_masks
from reconstruction.flame_fitter import FLAMEFitter, fit_flame_to_sequence
from reconstruction.face_segmentation import FaceSegmenter

# Convenience aliases matching the expected public API
detect_landmarks = FaceLandmarkDetector
fit_flame = fit_flame_to_sequence
segment_face = FaceSegmenter

__all__ = [
    "run_colmap",
    "run_colmap_dense",
    "detect_landmarks",
    "fit_flame",
    "segment_face",
    "refine_mesh",
    "FaceLandmarkDetector",
    "load_mediapipe_to_flame_mapping",
    "FLAMEModel",
    "load_flame_masks",
    "FLAMEFitter",
    "fit_flame_to_sequence",
    "FaceSegmenter",
]


def refine_mesh(
    vertices,
    faces,
    flame_masks=None,
    iterations: int = 3,
    lambda_smooth: float = 0.5,
    lambda_detail: float = 0.1,
):
    """Refine a fitted FLAME mesh via Laplacian smoothing with region-aware weights.

    Args:
        vertices: (N, 3) numpy array of mesh vertices.
        faces: (F, 3) numpy array of triangle indices.
        flame_masks: Optional dict of region name -> vertex indices from load_flame_masks.
        iterations: Number of smoothing iterations.
        lambda_smooth: Global smoothing weight.
        lambda_detail: Detail preservation weight for sensitive regions (eyes, lips).

    Returns:
        Refined vertices as (N, 3) numpy array.
    """
    import numpy as np

    refined = vertices.copy().astype(np.float64)
    num_verts = len(refined)

    # Build adjacency from faces
    adjacency = [set() for _ in range(num_verts)]
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[f[i]].add(f[j])

    # Per-vertex smoothing weight — reduce smoothing on detail-sensitive regions
    weights = np.full(num_verts, lambda_smooth, dtype=np.float64)
    if flame_masks is not None:
        detail_regions = {"eye_region", "lips", "eyeballs", "nose"}
        for region, indices in flame_masks.items():
            if region in detail_regions:
                weights[indices] = lambda_detail

    for _ in range(iterations):
        new_verts = refined.copy()
        for vi in range(num_verts):
            neighbours = adjacency[vi]
            if not neighbours:
                continue
            neighbour_mean = np.mean(refined[list(neighbours)], axis=0)
            new_verts[vi] = refined[vi] + weights[vi] * (neighbour_mean - refined[vi])
        refined = new_verts

    return refined
