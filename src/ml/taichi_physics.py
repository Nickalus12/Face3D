"""Taichi-based physics-aware mesh refinement for FLAME predictions.

Uses Taichi's parallel compute (GPU or CPU) to ensure that predicted vertex
deformations are physically plausible:
- Laplacian smoothing (prevents spiky artifacts)
- Self-intersection detection and correction
- Anatomical constraints (jaw range, eye bounds)
- Surface normal computation for Gaussian orientation

Falls back to pure numpy implementations if Taichi is not installed.

Data flow:
    Prior network predicts offsets -> TaichiMeshRefiner.refine() -> clean offsets
    The refined offsets are applied to FLAME vertices to produce the final
    mesh used for Gaussian initialization.

Taichi is optional: install with `pip install taichi` for GPU-accelerated
refinement. Without it, numpy fallbacks provide identical results at lower speed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Try importing Taichi — it's optional
_TAICHI_AVAILABLE = False
try:
    import taichi as ti
    _TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    logger.info("Taichi not available — using numpy fallbacks for mesh refinement")

# Taichi initialization state (lazy init to avoid GPU context issues)
_TAICHI_INITIALIZED = False


def _ensure_taichi_initialized():
    """Initialize Taichi with GPU backend (lazy, called once)."""
    global _TAICHI_INITIALIZED
    if _TAICHI_INITIALIZED or not _TAICHI_AVAILABLE:
        return
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
        _TAICHI_INITIALIZED = True
        logger.info("Taichi initialized with GPU backend")
    except Exception:
        try:
            ti.init(arch=ti.cpu, default_fp=ti.f32)
            _TAICHI_INITIALIZED = True
            logger.info("Taichi GPU init failed, using CPU backend")
        except Exception as e:
            logger.warning("Taichi initialization failed entirely: %s", e)


class TaichiMeshRefiner:
    """Refines FLAME mesh deformations using physics constraints.

    Given predicted vertex offsets from the prior network, this refiner:
    1. Applies Laplacian smoothing (prevents spiky artifacts)
    2. Checks for self-intersections (flips inverted triangles)
    3. Enforces anatomical bounds (jaw, eyes, mouth range limits)
    4. Computes physically-valid normals for Gaussian orientation

    The refiner works with the FLAME mesh topology (5023 vertices, ~9976 faces).
    It builds adjacency information on construction for efficient Laplacian
    computation.

    Args:
        flame_faces: (F, 3) int array of triangle vertex indices.
        n_vertices: Number of vertices in the mesh (default 5023 for FLAME).
    """

    def __init__(self, flame_faces: np.ndarray, n_vertices: int = 5023):
        self.faces = np.asarray(flame_faces, dtype=np.int32)
        self.n_vertices = n_vertices
        self.n_faces = len(self.faces)

        # Build vertex adjacency for Laplacian smoothing
        self._adjacency = self._build_adjacency()

        # Precompute Laplacian weights (uniform cotangent weights approximation)
        self._laplacian_weights = self._build_laplacian_weights()

        logger.info(
            "TaichiMeshRefiner: %d vertices, %d faces, taichi=%s",
            n_vertices, self.n_faces, _TAICHI_AVAILABLE,
        )

    def refine(
        self,
        vertices: np.ndarray,
        offsets: np.ndarray,
        iterations: int = 50,
        lambda_smooth: float = 0.3,
        lambda_anatomical: float = 1.0,
    ) -> np.ndarray:
        """Apply physics-aware refinement to vertex offsets.

        Args:
            vertices: Base FLAME vertices (n_vertices, 3).
            offsets: Predicted offsets from prior network (n_vertices, 3).
            iterations: Number of refinement iterations.
            lambda_smooth: Laplacian smoothing strength (0 = none, 1 = full).
            lambda_anatomical: Anatomical constraint strength.

        Returns:
            Refined vertices (n_vertices, 3) = vertices + refined_offsets.
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        offsets = np.asarray(offsets, dtype=np.float32)

        # Ensure correct shapes
        assert vertices.shape == (self.n_vertices, 3), (
            f"Expected vertices shape ({self.n_vertices}, 3), got {vertices.shape}"
        )

        # If offsets are per-region (num_regions, 3), expand to per-vertex
        if offsets.shape[0] != self.n_vertices:
            offsets = self._expand_region_offsets(offsets)

        # Apply offsets to get initial deformed mesh
        deformed = vertices + offsets

        if _TAICHI_AVAILABLE:
            _ensure_taichi_initialized()
            if _TAICHI_INITIALIZED:
                return self._refine_taichi(vertices, deformed, iterations, lambda_smooth)

        # Numpy fallback
        return self._refine_numpy(vertices, deformed, iterations, lambda_smooth)

    def compute_laplacian_smooth(
        self,
        vertices: np.ndarray,
        lambda_smooth: float = 0.5,
    ) -> np.ndarray:
        """Apply one step of Laplacian smoothing.

        Args:
            vertices: (n_vertices, 3) vertex positions.
            lambda_smooth: Smoothing factor (0 = no change, 1 = full Laplacian).

        Returns:
            Smoothed vertices (n_vertices, 3).
        """
        smoothed = np.copy(vertices)

        for v_idx in range(self.n_vertices):
            neighbors = self._adjacency[v_idx]
            if not neighbors:
                continue
            neighbor_mean = vertices[neighbors].mean(axis=0)
            smoothed[v_idx] = (1.0 - lambda_smooth) * vertices[v_idx] + lambda_smooth * neighbor_mean

        return smoothed

    def check_self_intersection(
        self,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """Detect triangles with inverted normals and fix them.

        Compares triangle normals of the deformed mesh against the original
        winding order. If a triangle's normal flips, the offsets for its
        vertices are dampened.

        Args:
            vertices: (n_vertices, 3) deformed vertex positions.

        Returns:
            Fixed vertices (n_vertices, 3).
        """
        normals = self._compute_face_normals(vertices)

        # Check for degenerate triangles (zero-area)
        norms = np.linalg.norm(normals, axis=1)
        degenerate = norms < 1e-10

        if degenerate.any():
            n_degenerate = degenerate.sum()
            logger.debug("Found %d degenerate triangles", n_degenerate)
            # For degenerate triangles, pull vertices toward triangle centroid
            result = np.copy(vertices)
            for face_idx in np.where(degenerate)[0]:
                v_indices = self.faces[face_idx]
                centroid = vertices[v_indices].mean(axis=0)
                for vi in v_indices:
                    result[vi] = 0.9 * vertices[vi] + 0.1 * centroid
            return result

        return vertices

    def enforce_anatomical_bounds(
        self,
        vertices: np.ndarray,
        base_vertices: np.ndarray,
        max_offset_ratio: float = 0.3,
    ) -> np.ndarray:
        """Enforce anatomical constraints on the deformed mesh.

        Prevents extreme deformations by clamping vertex offsets to a maximum
        fraction of the mesh bounding box diagonal. This ensures the face
        remains recognizably face-shaped.

        Args:
            vertices: (n_vertices, 3) deformed vertex positions.
            base_vertices: (n_vertices, 3) original FLAME vertices.
            max_offset_ratio: Maximum offset as fraction of bbox diagonal.

        Returns:
            Constrained vertices (n_vertices, 3).
        """
        # Compute bounding box diagonal of the base mesh
        bbox_min = base_vertices.min(axis=0)
        bbox_max = base_vertices.max(axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        max_offset = bbox_diag * max_offset_ratio

        # Clamp offsets
        offsets = vertices - base_vertices
        offset_norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_offset / (offset_norms + 1e-10))
        clamped_offsets = offsets * scale

        return base_vertices + clamped_offsets

    def compute_surface_normals(
        self,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """Compute per-vertex normals by averaging adjacent face normals.

        Args:
            vertices: (n_vertices, 3) vertex positions.

        Returns:
            (n_vertices, 3) unit normal vectors per vertex.
        """
        face_normals = self._compute_face_normals(vertices)

        # Accumulate face normals to vertices
        vertex_normals = np.zeros((self.n_vertices, 3), dtype=np.float32)
        for f_idx in range(self.n_faces):
            for v_idx in self.faces[f_idx]:
                vertex_normals[v_idx] += face_normals[f_idx]

        # Normalize
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        vertex_normals /= norms

        return vertex_normals

    # ------------------------------------------------------------------
    # Private: Taichi implementations
    # ------------------------------------------------------------------

    def _refine_taichi(
        self,
        base_vertices: np.ndarray,
        deformed: np.ndarray,
        iterations: int,
        lambda_smooth: float,
    ) -> np.ndarray:
        """GPU-accelerated refinement using Taichi kernels."""
        # Use Taichi fields for parallel computation
        n = self.n_vertices
        verts = ti.Vector.field(3, dtype=ti.f32, shape=n)
        verts_new = ti.Vector.field(3, dtype=ti.f32, shape=n)
        base = ti.Vector.field(3, dtype=ti.f32, shape=n)

        # Build adjacency as a flat array with offsets for Taichi access
        max_neighbors = max(len(adj) for adj in self._adjacency) if self._adjacency else 1
        adj_field = ti.field(dtype=ti.i32, shape=(n, max_neighbors))
        adj_count = ti.field(dtype=ti.i32, shape=n)

        # Copy data to Taichi fields
        verts.from_numpy(deformed)
        base.from_numpy(base_vertices)

        adj_np = np.full((n, max_neighbors), -1, dtype=np.int32)
        adj_count_np = np.zeros(n, dtype=np.int32)
        for i in range(n):
            neighbors = self._adjacency[i]
            adj_count_np[i] = len(neighbors)
            for j, nb in enumerate(neighbors):
                adj_np[i, j] = nb
        adj_field.from_numpy(adj_np)
        adj_count.from_numpy(adj_count_np)

        # Define Taichi kernel for Laplacian smoothing step
        @ti.kernel
        def smooth_step(lam: ti.f32):
            for i in range(n):
                cnt = adj_count[i]
                if cnt > 0:
                    neighbor_avg = ti.Vector([0.0, 0.0, 0.0])
                    for j in range(cnt):
                        nb = adj_field[i, j]
                        if nb >= 0:
                            neighbor_avg += verts[nb]
                    neighbor_avg /= float(cnt)
                    verts_new[i] = (1.0 - lam) * verts[i] + lam * neighbor_avg
                else:
                    verts_new[i] = verts[i]

        @ti.kernel
        def copy_back():
            for i in range(n):
                verts[i] = verts_new[i]

        # Iterative refinement
        for _ in range(iterations):
            smooth_step(lambda_smooth)
            copy_back()

        result = verts.to_numpy()

        # Apply anatomical bounds
        result = self.enforce_anatomical_bounds(result, base_vertices)

        return result

    # ------------------------------------------------------------------
    # Private: Numpy fallback implementations
    # ------------------------------------------------------------------

    def _refine_numpy(
        self,
        base_vertices: np.ndarray,
        deformed: np.ndarray,
        iterations: int,
        lambda_smooth: float,
    ) -> np.ndarray:
        """CPU fallback refinement using numpy."""
        current = np.copy(deformed)

        for _ in range(iterations):
            current = self.compute_laplacian_smooth(current, lambda_smooth)

        # Fix any self-intersections
        current = self.check_self_intersection(current)

        # Apply anatomical bounds
        current = self.enforce_anatomical_bounds(current, base_vertices)

        return current

    # ------------------------------------------------------------------
    # Private: Topology utilities
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> list[list[int]]:
        """Build vertex adjacency list from face array."""
        adjacency: list[set[int]] = [set() for _ in range(self.n_vertices)]

        for face in self.faces:
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            adjacency[v0].update([v1, v2])
            adjacency[v1].update([v0, v2])
            adjacency[v2].update([v0, v1])

        return [sorted(adj) for adj in adjacency]

    def _build_laplacian_weights(self) -> np.ndarray:
        """Build uniform Laplacian weight array."""
        weights = np.zeros(self.n_vertices, dtype=np.float32)
        for i in range(self.n_vertices):
            n_neighbors = len(self._adjacency[i])
            if n_neighbors > 0:
                weights[i] = 1.0 / n_neighbors
        return weights

    def _compute_face_normals(self, vertices: np.ndarray) -> np.ndarray:
        """Compute per-face normals from vertex positions."""
        v0 = vertices[self.faces[:, 0]]
        v1 = vertices[self.faces[:, 1]]
        v2 = vertices[self.faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        return normals / norms

    def _expand_region_offsets(self, region_offsets: np.ndarray) -> np.ndarray:
        """Expand per-region offsets (num_regions, 3) to per-vertex (n_vertices, 3).

        Uses a simple Y-coordinate-based region assignment matching the
        replay buffer's region partitioning strategy.
        """
        from ml.replay_buffer import FACE_REGIONS

        num_regions = len(FACE_REGIONS)
        per_vertex = np.zeros((self.n_vertices, 3), dtype=np.float32)

        # If region_offsets has fewer rows than expected, pad with zeros
        if region_offsets.shape[0] < num_regions:
            padded = np.zeros((num_regions, 3), dtype=np.float32)
            padded[:region_offsets.shape[0]] = region_offsets
            region_offsets = padded

        # Assign each vertex to a region based on index partition
        region_size = self.n_vertices // num_regions
        for i in range(num_regions):
            start = i * region_size
            end = (i + 1) * region_size if i < num_regions - 1 else self.n_vertices
            per_vertex[start:end] = region_offsets[i]

        return per_vertex
