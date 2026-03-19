"""UV texture baking from multi-view images onto an extracted mesh.

Projects high-quality source images onto a mesh surface to create a
UV-mapped texture atlas.  Supports xatlas UV unwrapping (preferred),
FLAME built-in UVs, and a spherical-projection fallback suitable for
roughly convex face geometry.

Visibility is resolved via Open3D RaycastingScene so that only
unoccluded surface points contribute colour, and the best view for
each texel is selected by a weighted score that accounts for surface
angle, pixel resolution, and an optional photo-vs-video preference.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from splatting.camera_utils import Camera

logger = logging.getLogger(__name__)

# Maximum number of cameras to use for texture baking (sorted by coverage)
_MAX_BAKE_CAMERAS = 20

# Timeout in seconds for the raycasting-based visibility pass per face batch
_RAYCAST_TIMEOUT_SEC = 120.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def bake_texture(
    mesh_path: str | Path,
    cameras: list[Camera],
    images_dir: str | Path,
    output_dir: str | Path,
    texture_resolution: int = 2048,
    prefer_photos: bool = True,
    flame_texture_path: str | Path | None = None,
    max_cameras: int = _MAX_BAKE_CAMERAS,
) -> Path | None:
    """Bake a UV texture atlas onto *mesh_path* from multi-view images.

    Algorithm
    ---------
    1. Load the mesh (OBJ or PLY via trimesh).
    2. UV-unwrap: try xatlas, then FLAME built-in UVs, then spherical
       fallback.
    3. For every texel in the UV map, find the 3D surface point, project
       it into each camera, perform a visibility check, score views, and
       sample the winning pixel colour.
    4. Blend border regions with a weighted average, inpaint any remaining
       holes, and write ``texture.png`` + ``mesh_textured.obj``.

    Parameters
    ----------
    mesh_path : path
        Input mesh (OBJ / PLY) from SuGaR or TSDF extraction.
    cameras : list[Camera]
        COLMAP-derived cameras with extrinsic and intrinsic data.
    images_dir : path
        Directory containing the source images referenced by each camera's
        ``image_path`` attribute.
    output_dir : path
        Destination for ``texture.png``, ``mesh_textured.obj``, and
        ``mesh_textured.mtl``.
    texture_resolution : int
        Width and height of the output texture atlas (default 4096).
    prefer_photos : bool
        If True, Expert RAW / photo frames are weighted 3x higher than
        ordinary video frames when selecting the best view.
    flame_texture_path : path, optional
        Path to ``FLAME_texture.npz`` for built-in FLAME UV coordinates.

    Returns
    -------
    Path or None
        Path to the saved ``mesh_textured.obj``, or None if the mesh does
        not exist or has no faces.
    """
    mesh_path = Path(mesh_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load mesh
    # ------------------------------------------------------------------
    if not mesh_path.exists():
        logger.warning("Mesh file %s not found; skipping texture baking", mesh_path)
        return None

    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        logger.warning("Mesh has no faces; skipping texture baking")
        return None

    logger.info(
        "Loaded mesh: %d vertices, %d faces from %s",
        len(mesh.vertices), len(mesh.faces), mesh_path,
    )

    # ------------------------------------------------------------------
    # 2. UV unwrap
    # ------------------------------------------------------------------
    uv_coords = _unwrap_mesh(mesh, flame_texture_path)
    logger.info("UV unwrap complete: %d UV coordinates", len(uv_coords))

    # ------------------------------------------------------------------
    # 3. Select best cameras (limit to max_cameras for performance)
    # ------------------------------------------------------------------
    cameras = _select_best_cameras(cameras, mesh, max_cameras)
    logger.info("Using %d cameras for texture baking", len(cameras))

    # ------------------------------------------------------------------
    # 4. Build raycasting scene for visibility tests
    # ------------------------------------------------------------------
    scene: o3d.t.geometry.RaycastingScene | None = None
    try:
        mesh_o3d = o3d.t.geometry.TriangleMesh()
        mesh_o3d.vertex.positions = o3d.core.Tensor(
            np.asarray(mesh.vertices, dtype=np.float32)
        )
        mesh_o3d.triangle.indices = o3d.core.Tensor(
            np.asarray(mesh.faces, dtype=np.int32)
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_o3d)

        # Quick sanity check — cast a single ray to see if it returns in time
        _test_start = time.monotonic()
        _test_origin = np.asarray(mesh.vertices, dtype=np.float32).mean(axis=0)
        _test_ray = o3d.core.Tensor(
            [[*_test_origin, 0.0, 0.0, 1.0]], dtype=o3d.core.Dtype.Float32
        )
        scene.cast_rays(_test_ray)
        _test_elapsed = time.monotonic() - _test_start
        if _test_elapsed > 5.0:
            logger.warning(
                "Raycasting sanity check took %.1fs; disabling visibility "
                "testing to avoid hanging",
                _test_elapsed,
            )
            scene = None
    except Exception as e:
        logger.warning("Failed to build raycasting scene (%s); skipping visibility", e)
        scene = None

    # Precompute face normals
    mesh.fix_normals()
    if mesh.face_normals is None or len(mesh.face_normals) == 0:
        face_normals = _compute_face_normals(mesh)
    else:
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Load source images (lazy, keyed by camera uid)
    # ------------------------------------------------------------------
    cam_images: dict[int, np.ndarray] = {}

    def _get_image(cam: Camera) -> np.ndarray | None:
        if cam.uid in cam_images:
            return cam_images[cam.uid]
        if cam.image_path is None:
            return None
        img_path = images_dir / cam.image_path
        if not img_path.exists():
            # Try common extensions
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = images_dir / (Path(cam.image_path).stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
        if not img_path.exists():
            return None
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cam_images[cam.uid] = img
        return img

    # ------------------------------------------------------------------
    # 6. Rasterise the UV map
    # ------------------------------------------------------------------
    tex_h = tex_w = texture_resolution
    texture = np.zeros((tex_h, tex_w, 3), dtype=np.float64)
    weight_map = np.zeros((tex_h, tex_w), dtype=np.float64)

    # Build per-face UV triangles (indices into uv_coords match face vertex order)
    # uv_coords is (F*3, 2) – three UV coords per face, laid out sequentially
    face_uvs = uv_coords.reshape(-1, 3, 2)  # (F, 3, 2)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    n_faces = len(faces)
    logger.info(
        "Baking %dx%d texture from %d cameras across %d faces...",
        tex_w, tex_h, len(cameras), n_faces,
    )

    bake_start = time.monotonic()
    raycast_total_time = 0.0
    raycast_disabled_mid_run = False

    # Rasterise each triangle into the texture
    for fi in range(n_faces):
        # Progress logging every 10% of faces
        if fi > 0 and fi % max(1, n_faces // 10) == 0:
            elapsed = time.monotonic() - bake_start
            pct = 100.0 * fi / n_faces
            logger.info(
                "Texture baking progress: %d/%d faces (%.0f%%) — %.1fs elapsed",
                fi, n_faces, pct, elapsed,
            )

        tri_uv = face_uvs[fi]  # (3, 2) in [0, 1]
        tri_verts = vertices[faces[fi]]  # (3, 3) world coords
        normal = face_normals[fi]

        # Bounding box of the triangle in texel space
        uv_px = tri_uv.copy()
        uv_px[:, 0] *= tex_w - 1
        uv_px[:, 1] *= tex_h - 1

        u_min = max(int(np.floor(uv_px[:, 0].min())), 0)
        u_max = min(int(np.ceil(uv_px[:, 0].max())), tex_w - 1)
        v_min = max(int(np.floor(uv_px[:, 1].min())), 0)
        v_max = min(int(np.ceil(uv_px[:, 1].max())), tex_h - 1)

        if u_min > u_max or v_min > v_max:
            continue

        # Vectorised texel sampling within the bounding box
        us = np.arange(u_min, u_max + 1, dtype=np.float64)
        vs = np.arange(v_min, v_max + 1, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)  # (rows, cols)
        texel_uv = np.stack([uu.ravel(), vv.ravel()], axis=-1)  # (N, 2)

        # Barycentric coordinates inside the triangle
        bary = _barycentric_2d(uv_px, texel_uv)  # (N, 3)
        inside = np.all(bary >= -1e-4, axis=1)
        if not inside.any():
            continue

        bary = bary[inside]
        texel_uv_inside = texel_uv[inside]

        # 3D positions of these texels
        pts_3d = bary @ tri_verts  # (M, 3)

        # Disable raycasting mid-run if cumulative raycast time is excessive
        active_scene = scene
        if scene is not None and not raycast_disabled_mid_run:
            if raycast_total_time > _RAYCAST_TIMEOUT_SEC:
                logger.warning(
                    "Raycasting cumulative time (%.1fs) exceeded timeout (%.0fs); "
                    "falling back to nearest-camera projection without occlusion",
                    raycast_total_time, _RAYCAST_TIMEOUT_SEC,
                )
                active_scene = None
                raycast_disabled_mid_run = True

        # Track raycast time for this batch
        rc_t0 = time.monotonic()

        # Select best view for each texel
        colors, weights = _sample_best_views(
            pts_3d, normal, cameras, _get_image, active_scene, prefer_photos,
        )

        raycast_total_time += time.monotonic() - rc_t0

        # Write into texture (weighted accumulation for blending at edges)
        for k in range(len(texel_uv_inside)):
            tu = int(round(texel_uv_inside[k, 0]))
            tv = int(round(texel_uv_inside[k, 1]))
            if 0 <= tu < tex_w and 0 <= tv < tex_h:
                w = weights[k]
                if w > 0:
                    # UV convention: row = tex_h - 1 - v (flip Y for image)
                    row = tex_h - 1 - tv
                    texture[row, tu] += colors[k] * w
                    weight_map[row, tu] += w

    bake_elapsed = time.monotonic() - bake_start
    logger.info(
        "Texture baking rasterisation complete: %.1fs total (%.1fs raycasting)",
        bake_elapsed, raycast_total_time,
    )

    # Normalise accumulated colours
    valid = weight_map > 0
    texture[valid] /= weight_map[valid, np.newaxis]
    texture = np.clip(texture, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 7. Inpaint holes
    # ------------------------------------------------------------------
    mask = (~valid).astype(np.uint8) * 255
    texture_bgr = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
    texture_bgr = cv2.inpaint(texture_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    texture = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    texture_path = output_dir / "texture.png"
    cv2.imwrite(str(texture_path), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
    logger.info("Saved texture atlas to %s", texture_path)

    mtl_path = output_dir / "mesh_textured.mtl"
    create_material_file(texture_path, mtl_path)

    obj_path = output_dir / "mesh_textured.obj"
    _write_textured_obj(mesh, uv_coords, obj_path, mtl_path)
    logger.info("Saved textured mesh to %s", obj_path)

    return obj_path


# ---------------------------------------------------------------------------
# Camera selection
# ---------------------------------------------------------------------------


def _select_best_cameras(
    cameras: list[Camera],
    mesh: trimesh.Trimesh,
    max_cameras: int = _MAX_BAKE_CAMERAS,
) -> list[Camera]:
    """Select the best *max_cameras* cameras for texture baking.

    Cameras are scored by how well they cover the mesh surface: a
    combination of the fraction of mesh vertices visible in the image
    and the average cosine angle between the camera view direction and
    mesh centroid.  This avoids using all 100+ cameras (which makes the
    per-texel loop extremely slow) while keeping good angular coverage.

    If *max_cameras* >= len(cameras), all cameras are returned as-is.
    """
    if len(cameras) <= max_cameras:
        return cameras

    centroid = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    verts_arr = np.asarray(mesh.vertices)
    extent = verts_arr.max(axis=0) - verts_arr.min(axis=0)
    mesh_radius = float(np.linalg.norm(extent)) * 0.5

    scored: list[tuple[float, int]] = []

    for idx, cam in enumerate(cameras):
        R = cam.R.astype(np.float64)
        T = cam.T.astype(np.float64)
        cam_pos = -R.T @ T

        # Distance and direction to mesh centroid
        to_mesh = centroid - cam_pos
        dist = float(np.linalg.norm(to_mesh))
        if dist < 1e-8:
            continue
        view_dir = to_mesh / dist

        # Camera forward direction (third row of R = z-axis in camera frame)
        cam_fwd = R[2, :]
        cos_angle = float(np.dot(cam_fwd, view_dir))

        # Score: prefer cameras that face the mesh and are at a moderate distance
        # Closer cameras get slightly higher resolution scores
        resolution_score = min(mesh_radius / (dist + 1e-3), 2.0)
        score = max(cos_angle, 0.0) * resolution_score

        scored.append((score, idx))

    # Sort descending by score, then pick the top max_cameras with diverse
    # azimuth coverage
    scored.sort(key=lambda x: -x[0])

    # Greedy selection: pick top scorer, then prefer cameras with different
    # azimuth angles to ensure coverage
    selected_indices: list[int] = []
    selected_azimuths: list[float] = []

    for score, idx in scored:
        if len(selected_indices) >= max_cameras:
            break
        cam = cameras[idx]
        R = cam.R.astype(np.float64)
        T = cam.T.astype(np.float64)
        cam_pos = -R.T @ T
        to_mesh = centroid - cam_pos
        azimuth = math.atan2(float(to_mesh[0]), float(to_mesh[2]))

        # Accept if it's far enough in azimuth from already-selected cameras
        # (at least 10 degrees apart) or if we haven't filled half the slots yet
        min_az_dist = min(
            (abs(azimuth - a) for a in selected_azimuths),
            default=math.pi,
        )
        # Wrap-around
        min_az_dist = min(min_az_dist, 2 * math.pi - min_az_dist)

        if min_az_dist > math.radians(10) or len(selected_indices) < max_cameras // 2:
            selected_indices.append(idx)
            selected_azimuths.append(azimuth)

    # If greedy didn't fill all slots, add remaining top-scored cameras
    if len(selected_indices) < max_cameras:
        remaining = [idx for _, idx in scored if idx not in set(selected_indices)]
        for idx in remaining[: max_cameras - len(selected_indices)]:
            selected_indices.append(idx)

    logger.info(
        "Selected %d/%d cameras for texture baking (top by view coverage)",
        len(selected_indices), len(cameras),
    )
    return [cameras[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# UV unwrapping strategies
# ---------------------------------------------------------------------------


def _unwrap_mesh(
    mesh: trimesh.Trimesh,
    flame_texture_path: str | Path | None = None,
) -> np.ndarray:
    """Return (F*3, 2) UV coordinates for the mesh.

    Strategy priority:
      1. xatlas (via trimesh or standalone)
      2. FLAME built-in UVs from FLAME_texture.npz
      3. Spherical projection fallback
    """
    # --- 1. xatlas ---
    try:
        import xatlas as _xatlas  # noqa: F401

        logger.info("UV unwrapping with xatlas...")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        vmapping, new_faces, uvs = _xatlas.parametrize(vertices, faces)
        # Build per-face UV array: (F, 3, 2) -> (F*3, 2)
        per_face_uv = uvs[new_faces.ravel()]
        logger.info("xatlas produced %d UV coords", len(per_face_uv))
        # Reindex mesh vertices to match xatlas output
        mesh.vertices = vertices[vmapping]
        mesh.faces = new_faces
        return per_face_uv
    except ImportError:
        logger.info("xatlas not available; trying FLAME UVs")
    except Exception as e:
        logger.warning("xatlas unwrap failed (%s); trying FLAME UVs", e)

    # --- 2. FLAME built-in UVs ---
    if flame_texture_path is not None:
        flame_texture_path = Path(flame_texture_path)
        if flame_texture_path.exists():
            try:
                return _load_flame_uvs(mesh, flame_texture_path)
            except Exception as e:
                logger.warning("FLAME UV loading failed (%s); using spherical fallback", e)

    # --- 3. Spherical fallback ---
    logger.info("Using spherical UV mapping fallback")
    return create_spherical_uv(np.asarray(mesh.vertices), mesh.faces)


def _load_flame_uvs(
    mesh: trimesh.Trimesh,
    flame_texture_path: Path,
) -> np.ndarray:
    """Load UV coordinates from FLAME_texture.npz.

    The file typically contains 'vt' (per-vertex UVs) and 'ft' (UV face
    indices).  Returns (F*3, 2) per-face-vertex UV array.
    """
    data = np.load(str(flame_texture_path), allow_pickle=True)

    # Identify UV arrays (different FLAME versions use different key names)
    vt = None
    ft = None
    for key in ("vt", "uv_coords", "uvs"):
        if key in data:
            vt = data[key]
            break
    for key in ("ft", "uv_faces", "face_uvs"):
        if key in data:
            ft = data[key]
            break

    if vt is None:
        raise ValueError("No UV vertex data found in FLAME_texture.npz")

    n_mesh_faces = len(mesh.faces)

    if ft is not None and len(ft) >= n_mesh_faces:
        # Use the face-based UV indexing
        ft = ft[:n_mesh_faces]
        per_face_uv = vt[ft.ravel()].astype(np.float64)
    elif len(vt) >= len(mesh.vertices):
        # Per-vertex UVs: index via mesh faces
        per_face_uv = vt[mesh.faces.ravel()].astype(np.float64)
    else:
        raise ValueError(
            f"FLAME UV count ({len(vt)}) does not match mesh vertices ({len(mesh.vertices)})"
        )

    logger.info("Loaded FLAME UVs: %d face-vertex UV pairs", len(per_face_uv))
    return per_face_uv.reshape(-1, 2)


def create_spherical_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Simple spherical UV mapping for roughly convex face geometry.

    Projects vertices onto a unit sphere centred at *center* (default:
    centroid) and maps (theta, phi) to (u, v).

    Parameters
    ----------
    vertices : (V, 3)
    faces : (F, 3)
    center : (3,) optional

    Returns
    -------
    (F*3, 2) per-face-vertex UV coordinates in [0, 1].
    """
    if center is None:
        center = vertices.mean(axis=0)

    dirs = vertices - center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    dirs = dirs / norms

    # Spherical coordinates
    theta = np.arctan2(dirs[:, 0], dirs[:, 2])  # azimuth
    phi = np.arcsin(np.clip(dirs[:, 1], -1, 1))  # elevation

    u = (theta / (2.0 * np.pi)) + 0.5  # [0, 1]
    v = (phi / np.pi) + 0.5            # [0, 1]

    per_vertex_uv = np.stack([u, v], axis=-1)  # (V, 2)
    per_face_uv = per_vertex_uv[faces.ravel()]  # (F*3, 2)
    return per_face_uv.astype(np.float64)


# ---------------------------------------------------------------------------
# View selection and sampling
# ---------------------------------------------------------------------------


def select_best_view(
    point_3d: np.ndarray,
    normal: np.ndarray,
    cameras: list[Camera],
    get_image_fn,
    scene: o3d.t.geometry.RaycastingScene | None = None,
    prefer_photos: bool = True,
) -> tuple[np.ndarray | None, float]:
    """Find the best camera view for a single 3D surface point.

    Returns (rgb_colour, weight) or (None, 0.0) if no view is valid.

    Score = dot(normal, view_dir) * resolution_factor * photo_bonus
    """
    best_color = None
    best_score = 0.0

    point = point_3d.astype(np.float64)
    normal_n = normal / (np.linalg.norm(normal) + 1e-8)

    for cam in cameras:
        img = get_image_fn(cam)
        if img is None:
            continue

        # Camera position in world frame:  R^T @ (-T)
        R = cam.R.astype(np.float64)
        T = cam.T.astype(np.float64)
        cam_pos = -R.T @ T

        view_dir = cam_pos - point
        dist = np.linalg.norm(view_dir)
        if dist < 1e-8:
            continue
        view_dir /= dist

        # Angle between surface normal and view direction
        cos_angle = float(np.dot(normal_n, view_dir))
        if cos_angle < 0.05:
            # Surface faces away from camera
            continue

        # Project point into image
        p_cam = R @ point + T
        if p_cam[2] <= 0:
            continue

        K = np.array([
            [cam.fx, 0.0, cam.width / 2.0],
            [0.0, cam.fy, cam.height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        p_img = K @ p_cam
        px = p_img[0] / p_img[2]
        py = p_img[1] / p_img[2]

        # Check within image bounds (with margin)
        margin = 2.0
        if px < margin or px >= cam.width - margin or py < margin or py >= cam.height - margin:
            continue

        # Visibility check via raycasting
        if scene is not None:
            ray_origin = point + normal_n * 1e-4  # slight offset to avoid self-hit
            ray_dir = view_dir
            ray = o3d.core.Tensor(
                [[*ray_origin, *ray_dir]], dtype=o3d.core.Dtype.Float32
            )
            result = scene.cast_rays(ray)
            t_hit = result["t_hit"].numpy()[0]
            # If ray hits something before reaching the camera, the point is occluded
            if t_hit < dist * 0.98:
                continue

        # Resolution factor: higher when camera is closer
        # Normalise so typical face-capture distance (~0.5m) gives ~1.0
        resolution_factor = min(1.0 / (dist + 0.01), 5.0)

        # Photo bonus
        photo_bonus = 1.0
        if prefer_photos and cam.image_path is not None:
            name_lower = cam.image_path.lower()
            if any(tag in name_lower for tag in ("photo", "raw", "expert", "dng")):
                photo_bonus = 3.0

        score = cos_angle * resolution_factor * photo_bonus

        if score > best_score:
            best_score = score
            # Sample pixel colour (bilinear)
            best_color = _sample_pixel_bilinear(img, px, py)

    return best_color, best_score


def _sample_best_views(
    pts_3d: np.ndarray,
    normal: np.ndarray,
    cameras: list[Camera],
    get_image_fn,
    scene: o3d.t.geometry.RaycastingScene | None,
    prefer_photos: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample colours for a batch of 3D points sharing the same face normal.

    Returns (colors (M, 3), weights (M,)).
    """
    n = len(pts_3d)
    colors = np.zeros((n, 3), dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)

    # Batch project all points into all cameras to find candidates
    normal_n = normal / (np.linalg.norm(normal) + 1e-8)

    for cam in cameras:
        img = get_image_fn(cam)
        if img is None:
            continue

        R = cam.R.astype(np.float64)
        T = cam.T.astype(np.float64)
        cam_pos = -R.T @ T

        # View direction for all points
        view_dirs = cam_pos - pts_3d  # (M, 3)
        dists = np.linalg.norm(view_dirs, axis=1)  # (M,)
        valid = dists > 1e-8
        if not valid.any():
            continue
        view_dirs[valid] /= dists[valid, np.newaxis]

        # Angle check
        cos_angles = view_dirs @ normal_n  # (M,)
        angle_ok = cos_angles > 0.05
        active = valid & angle_ok
        if not active.any():
            continue

        # Project to image
        p_cam = (R @ pts_3d.T).T + T  # (M, 3)
        depth_ok = p_cam[:, 2] > 0
        active &= depth_ok
        if not active.any():
            continue

        K = np.array([
            [cam.fx, 0.0, cam.width / 2.0],
            [0.0, cam.fy, cam.height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        p_img = (K @ p_cam.T).T  # (M, 3)
        px = p_img[:, 0] / p_img[:, 2]
        py = p_img[:, 1] / p_img[:, 2]

        margin = 2.0
        in_bounds = (
            (px >= margin)
            & (px < cam.width - margin)
            & (py >= margin)
            & (py < cam.height - margin)
        )
        active &= in_bounds
        if not active.any():
            continue

        # Visibility (batch raycasting)
        active_idx = np.where(active)[0]
        if scene is not None and len(active_idx) > 0:
            origins = pts_3d[active_idx] + normal_n * 1e-4
            dirs = view_dirs[active_idx]
            rays = np.hstack([origins, dirs]).astype(np.float32)
            ray_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
            result = scene.cast_rays(ray_tensor)
            t_hits = result["t_hit"].numpy()
            cam_dists = dists[active_idx]
            occluded = t_hits < cam_dists * 0.98
            occluded_set = set(active_idx[occluded])
            active_idx = np.array([i for i in active_idx if i not in occluded_set])
            if len(active_idx) == 0:
                continue

        # Score
        resolution_factors = np.minimum(1.0 / (dists[active_idx] + 0.01), 5.0)
        photo_bonus = 1.0
        if prefer_photos and cam.image_path is not None:
            name_lower = cam.image_path.lower()
            if any(tag in name_lower for tag in ("photo", "raw", "expert", "dng")):
                photo_bonus = 3.0

        scores = cos_angles[active_idx] * resolution_factors * photo_bonus

        # Sample pixels
        for j, idx in enumerate(active_idx):
            sc = scores[j]
            if sc > weights[idx]:
                c = _sample_pixel_bilinear(img, px[idx], py[idx])
                if c is not None:
                    colors[idx] = c
                    weights[idx] = sc

    return colors, weights


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _barycentric_2d(
    tri: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Compute barycentric coordinates for *points* w.r.t. 2D *tri*.

    Parameters
    ----------
    tri : (3, 2) triangle vertex positions in 2D.
    points : (N, 2) query points.

    Returns
    -------
    (N, 3) barycentric weights.
    """
    v0 = tri[2] - tri[0]
    v1 = tri[1] - tri[0]
    v2 = points - tri[0]

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    dot02 = v2 @ v0
    dot12 = v2 @ v1

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return np.full((len(points), 3), -1.0)

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v

    return np.stack([w, v, u], axis=-1)


def _compute_face_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute per-face normals from vertices and faces."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return (normals / norms).astype(np.float32)


def _sample_pixel_bilinear(
    img: np.ndarray,
    px: float,
    py: float,
) -> np.ndarray | None:
    """Sample a pixel from *img* using bilinear interpolation.

    Returns (3,) float64 RGB in [0, 255] or None if out of bounds.
    """
    h, w = img.shape[:2]
    x = float(px)
    y = float(py)

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return None

    fx = x - x0
    fy = y - y0

    c00 = img[y0, x0].astype(np.float64)
    c10 = img[y0, x1].astype(np.float64)
    c01 = img[y1, x0].astype(np.float64)
    c11 = img[y1, x1].astype(np.float64)

    return (
        c00 * (1 - fx) * (1 - fy)
        + c10 * fx * (1 - fy)
        + c01 * (1 - fx) * fy
        + c11 * fx * fy
    )


# ---------------------------------------------------------------------------
# OBJ / MTL writers
# ---------------------------------------------------------------------------


def create_material_file(
    texture_path: str | Path,
    output_mtl_path: str | Path,
) -> Path:
    """Write a Wavefront .mtl file referencing *texture_path*.

    Parameters
    ----------
    texture_path : path
        The texture image file (e.g. ``texture.png``).
    output_mtl_path : path
        Destination for the .mtl file.

    Returns
    -------
    Path to the written .mtl file.
    """
    texture_path = Path(texture_path)
    output_mtl_path = Path(output_mtl_path)
    output_mtl_path.parent.mkdir(parents=True, exist_ok=True)

    mtl_content = (
        "# Face3D texture material\n"
        "newmtl face_texture\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.000 0.000 0.000\n"
        "Ns 10.0\n"
        "d 1.0\n"
        "illum 1\n"
        f"map_Kd {texture_path.name}\n"
    )

    output_mtl_path.write_text(mtl_content)
    logger.info("Saved material file to %s", output_mtl_path)
    return output_mtl_path


def _write_textured_obj(
    mesh: trimesh.Trimesh,
    uv_coords: np.ndarray,
    obj_path: Path,
    mtl_path: Path,
) -> None:
    """Write a Wavefront OBJ with UV texture coordinates and material reference.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    uv_coords : (F*3, 2)  per-face-vertex UVs
    obj_path : output path
    mtl_path : companion .mtl file path
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    face_uvs = uv_coords.reshape(-1, 3, 2)  # (F, 3, 2)

    with open(obj_path, "w") as f:
        f.write("# Face3D textured mesh\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl face_texture\n\n")

        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        f.write("\n")

        # Texture coordinates
        for fi in range(len(faces)):
            for vi in range(3):
                u, v_coord = face_uvs[fi, vi]
                f.write(f"vt {u:.8f} {v_coord:.8f}\n")

        f.write("\n")

        # Vertex normals (if available)
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
            for n in mesh.vertex_normals:
                f.write(f"vn {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")
            f.write("\n")

            # Faces: v/vt/vn (1-indexed)
            for fi in range(len(faces)):
                vt_base = fi * 3 + 1  # 1-indexed
                v0 = faces[fi, 0] + 1
                v1 = faces[fi, 1] + 1
                v2 = faces[fi, 2] + 1
                f.write(
                    f"f {v0}/{vt_base}/{v0} "
                    f"{v1}/{vt_base + 1}/{v1} "
                    f"{v2}/{vt_base + 2}/{v2}\n"
                )
        else:
            # Faces: v/vt (1-indexed)
            for fi in range(len(faces)):
                vt_base = fi * 3 + 1
                v0 = faces[fi, 0] + 1
                v1 = faces[fi, 1] + 1
                v2 = faces[fi, 2] + 1
                f.write(
                    f"f {v0}/{vt_base} "
                    f"{v1}/{vt_base + 1} "
                    f"{v2}/{vt_base + 2}\n"
                )

    logger.info("Wrote textured OBJ with %d faces to %s", len(faces), obj_path)
