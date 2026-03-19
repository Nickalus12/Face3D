"""COLMAP orchestration for Structure-from-Motion and dense reconstruction.

Drives the COLMAP binary through subprocess calls with face-capture-optimized
parameter presets.  Optionally uses SuperPoint+SuperGlue (hloc) for learned
feature matching when available.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_COLMAP_BINARY = "D:/Projects/3D/COLMAP/COLMAP-3.9.1-windows-cuda/COLMAP.bat"

# Number of threads for CPU-bound stages (defaults to CPU count)
_NUM_THREADS = str(os.cpu_count() or 4)


def _hloc_available() -> bool:
    """Check whether hloc (Hierarchical Localization) is importable."""
    try:
        import hloc  # noqa: F401
        return True
    except ImportError:
        return False


def _find_vocab_tree(workspace_dir: Path) -> Optional[Path]:
    """Look for a vocab tree binary in common locations."""
    candidates = [
        workspace_dir / "vocab_tree.bin",
        workspace_dir.parent / "vocab_tree.bin",
        Path("D:/Projects/3D/COLMAP/vocab_tree_flickr100K_words32K.bin"),
        Path("D:/Projects/3D/COLMAP/vocab_tree_flickr100K_words256K.bin"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_colmap(
    images_dir: Path,
    workspace_dir: Path,
    colmap_binary: str = _DEFAULT_COLMAP_BINARY,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    max_image_size: int = 3840,
    max_num_features: int = 8192,
    image_priors_path: Optional[Path] = None,
    use_learned_features: bool = True,
) -> Path:
    """Run the full COLMAP sparse SfM pipeline.

    Executes feature_extractor, sequential_matcher, and mapper in order.
    Uses settings tuned for close-range face captures (high feature count,
    OPENCV camera model for lens distortion, relaxed matching thresholds).

    When ``use_learned_features`` is True and hloc is installed, SuperPoint
    features with SuperGlue matching are used instead of SIFT.  If hloc is
    not available, the pipeline falls back to SIFT with an increased feature
    count for better coverage on textureless skin.

    Args:
        images_dir: Directory containing input images.
        workspace_dir: Working directory for COLMAP outputs.
        colmap_binary: Path to the COLMAP executable / batch file.
        camera_model: Camera model string (OPENCV, PINHOLE, etc.).
        single_camera: Whether all images come from one camera.
        max_image_size: Maximum image dimension for feature extraction.
        max_num_features: Maximum number of features to detect per image.
        image_priors_path: Optional path to a JSON file with per-image
            rotation priors from IMU data. Expected format is a dict mapping
            image filenames to 3x3 rotation matrices (row-major list of 9 floats).
        use_learned_features: If True, attempt to use SuperPoint+SuperGlue
            via hloc before falling back to SIFT.

    Returns:
        Path to the sparse reconstruction directory.
    """
    images_dir = Path(images_dir)
    workspace_dir = Path(workspace_dir)
    database_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # ── Try learned features (SuperPoint + SuperGlue) via hloc ─────────
    used_hloc = False
    if use_learned_features and _hloc_available():
        used_hloc = _run_hloc_features(
            images_dir, workspace_dir, database_path, camera_model, single_camera
        )

    if not used_hloc:
        # Fall back to SIFT -- bump feature count for textureless face skin
        sift_max_features = max(max_num_features, 16384)
        if sift_max_features > max_num_features:
            logger.info(
                "hloc not available; using SIFT with increased features (%d -> %d)",
                max_num_features, sift_max_features,
            )

        # ── Feature extraction ──────────────────────────────────────────
        logger.info("Running COLMAP feature extraction ...")
        feature_cmd = [
            colmap_binary, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--SiftExtraction.max_image_size", str(max_image_size),
            "--SiftExtraction.max_num_features", str(sift_max_features),
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
            "--SiftExtraction.first_octave", "-1",
            "--SiftExtraction.num_octaves", "4",
            "--SiftExtraction.peak_threshold", "0.004",
            "--SiftExtraction.edge_threshold", "16",
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.num_threads", _NUM_THREADS,
        ]
        _run_cmd(feature_cmd, "feature_extractor")

        # ── Sequential matching ─────────────────────────────────────────
        logger.info("Running COLMAP sequential matching ...")
        match_cmd = [
            colmap_binary, "sequential_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "1",
            "--SiftMatching.max_ratio", "0.85",
            "--SiftMatching.max_distance", "0.7",
            "--SiftMatching.max_num_matches", "32768",
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.num_threads", _NUM_THREADS,
            "--SequentialMatching.overlap", "20",
            "--SequentialMatching.quadratic_overlap", "1",
        ]

        # Only enable loop detection when a vocab tree is available
        vocab_tree = _find_vocab_tree(workspace_dir)
        if vocab_tree is not None:
            match_cmd.extend([
                "--SequentialMatching.loop_detection", "1",
                "--SequentialMatching.vocab_tree_path", str(vocab_tree),
            ])
            logger.info("Loop detection enabled with vocab tree: %s", vocab_tree)
        else:
            match_cmd.extend(["--SequentialMatching.loop_detection", "0"])
            logger.info(
                "No vocab tree found — loop detection disabled, "
                "using sequential overlap only"
            )

        _run_cmd(match_cmd, "sequential_matcher")

    # ── Optionally write image rotation priors ──────────────────────────
    if image_priors_path is not None:
        _write_image_priors(image_priors_path, database_path, colmap_binary)

    # ── Mapper ──────────────────────────────────────────────────────────
    logger.info("Running COLMAP mapper ...")
    mapper_cmd = [
        colmap_binary, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_global_max_num_iterations", "50",
        "--Mapper.ba_global_max_refinements", "5",
        "--Mapper.ba_local_max_num_iterations", "40",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "1",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.min_num_matches", "15",
        "--Mapper.init_min_num_inliers", "50",
        "--Mapper.multiple_models", "0",
        "--Mapper.extract_colors", "1",
        "--Mapper.num_threads", _NUM_THREADS,
    ]
    _run_cmd(mapper_cmd, "mapper")

    # ── Single model enforcement ────────────────────────────────────────
    model_dir = _select_best_model(sparse_dir)
    logger.info("Sparse reconstruction complete: %s", model_dir)
    return model_dir


def _run_hloc_features(
    images_dir: Path,
    workspace_dir: Path,
    database_path: Path,
    camera_model: str,
    single_camera: bool,
) -> bool:
    """Run SuperPoint extraction + SuperGlue matching via hloc.

    Returns True if successful, False if hloc pipeline fails (caller should
    fall back to SIFT).
    """
    try:
        from hloc import extract_features, match_features, reconstruction
        logger.info("Using hloc SuperPoint+SuperGlue for feature extraction and matching")

        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superglue"]

        hloc_dir = workspace_dir / "hloc"
        hloc_dir.mkdir(parents=True, exist_ok=True)

        extract_features.main(
            feature_conf, images_dir, export_dir=hloc_dir
        )
        # Generate pairs from sequential overlap (face capture orbits)
        pairs_path = hloc_dir / "pairs.txt"
        _generate_sequential_pairs(images_dir, pairs_path, overlap=20)

        match_path = match_features.main(
            matcher_conf, pairs_path, feature_conf["output"],
            export_dir=hloc_dir,
        )

        # Import features into the COLMAP database
        reconstruction.create_empty_db(database_path)
        reconstruction.import_features(database_path, hloc_dir, images_dir)
        reconstruction.import_matches(database_path, match_path, hloc_dir, images_dir)

        logger.info("hloc feature extraction and matching complete")
        return True
    except Exception as exc:
        logger.warning("hloc pipeline failed (%s) — falling back to SIFT", exc)
        # Clean up partial database so SIFT can start fresh
        if database_path.exists():
            database_path.unlink()
        return False


def _generate_sequential_pairs(
    images_dir: Path, pairs_path: Path, overlap: int = 20
) -> None:
    """Write sequential image pairs file for hloc matching."""
    image_names = sorted(
        p.name for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    )
    pairs = []
    n = len(image_names)
    for i in range(n):
        for j in range(i + 1, min(i + overlap + 1, n)):
            pairs.append(f"{image_names[i]} {image_names[j]}")
        # Loop closure: connect last images to first images
        if i >= n - overlap:
            for j in range(0, min(overlap, i)):
                pair = f"{image_names[i]} {image_names[j]}"
                if pair not in pairs:
                    pairs.append(pair)
    with open(pairs_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(pairs) + "\n")
    logger.info("Generated %d sequential pairs (overlap=%d, with loop closure)", len(pairs), overlap)


def _select_best_model(sparse_dir: Path) -> Path:
    """Select the best (largest) model from the mapper output.

    COLMAP mapper may produce multiple sub-models (0/, 1/, ...).  This
    function checks how many were produced, warns if there are multiple,
    and returns the largest one by number of registered images.
    """
    model_dirs = sorted(
        [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    if not model_dirs:
        # Mapper wrote directly into sparse_dir (no numbered sub-folders)
        return sparse_dir

    if len(model_dirs) == 1:
        return model_dirs[0]

    # Multiple models -- pick the largest by images.bin file size as a proxy
    # for number of registered images.
    logger.warning(
        "COLMAP produced %d separate models. Ideally there should be 1. "
        "Consider adjusting matching overlap or adding more images for "
        "complete loop closure.",
        len(model_dirs),
    )

    best_dir = model_dirs[0]
    best_size = 0
    for md in model_dirs:
        images_bin = md / "images.bin"
        if images_bin.exists():
            sz = images_bin.stat().st_size
            logger.info("  Model %s: images.bin = %d bytes", md.name, sz)
            if sz > best_size:
                best_size = sz
                best_dir = md
        else:
            logger.info("  Model %s: no images.bin found", md.name)

    logger.info("Selected model %s as the largest reconstruction", best_dir.name)

    # Try to merge other models into the best one
    _try_merge_models(sparse_dir, best_dir, model_dirs)

    return best_dir


def _try_merge_models(
    sparse_dir: Path, best_dir: Path, model_dirs: list[Path]
) -> None:
    """Attempt to merge secondary models into the primary one.

    This is a best-effort operation.  If pycolmap is available, use its
    model merging; otherwise log a suggestion for the user.
    """
    secondary = [d for d in model_dirs if d != best_dir]
    if not secondary:
        return

    try:
        import pycolmap
        best_model = pycolmap.Reconstruction(str(best_dir))
        merged_count = 0
        for sd in secondary:
            other = pycolmap.Reconstruction(str(sd))
            try:
                best_model.merge(other)
                merged_count += 1
                logger.info("Merged model %s into model %s", sd.name, best_dir.name)
            except Exception as exc:
                logger.warning("Could not merge model %s: %s", sd.name, exc)
        if merged_count > 0:
            best_model.write(str(best_dir))
            logger.info("Wrote merged reconstruction to %s", best_dir)
    except ImportError:
        logger.info(
            "pycolmap not installed — cannot auto-merge models. "
            "Install pycolmap or manually merge with `colmap model_merger`."
        )


def run_colmap_dense(
    workspace_dir: Path,
    colmap_binary: str = _DEFAULT_COLMAP_BINARY,
) -> Path:
    """Run COLMAP dense reconstruction pipeline.

    Executes image_undistorter, patch_match_stereo, and stereo_fusion.

    Args:
        workspace_dir: Working directory (must already contain sparse/0).
        colmap_binary: Path to the COLMAP executable.

    Returns:
        Path to the fused point-cloud PLY file.
    """
    workspace_dir = Path(workspace_dir)
    sparse_dir = workspace_dir / "sparse" / "0"
    dense_dir = workspace_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    # ── Undistort images ────────────────────────────────────────────────
    logger.info("Running image undistorter ...")
    undistort_cmd = [
        colmap_binary, "image_undistorter",
        "--image_path", str(workspace_dir / "images"),
        "--input_path", str(sparse_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", "3840",
        "--num_threads", _NUM_THREADS,
    ]
    _run_cmd(undistort_cmd, "image_undistorter")

    # ── Patch-match stereo ──────────────────────────────────────────────
    logger.info("Running patch-match stereo ...")
    stereo_cmd = [
        colmap_binary, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "1",
        "--PatchMatchStereo.max_image_size", "3840",
        "--PatchMatchStereo.window_radius", "5",
        "--PatchMatchStereo.window_step", "1",
        "--PatchMatchStereo.num_samples", "15",
        "--PatchMatchStereo.num_iterations", "5",
        "--PatchMatchStereo.filter", "1",
        "--PatchMatchStereo.filter_min_ncc", "0.1",
        "--PatchMatchStereo.filter_min_triangulation_angle", "1.0",
        "--PatchMatchStereo.filter_min_num_consistent", "2",
        "--PatchMatchStereo.filter_geom_consistency_max_cost", "1.0",
        "--PatchMatchStereo.gpu_index", "0",
    ]
    _run_cmd(stereo_cmd, "patch_match_stereo")

    # ── Stereo fusion ───────────────────────────────────────────────────
    logger.info("Running stereo fusion ...")
    fused_path = dense_dir / "fused.ply"
    fusion_cmd = [
        colmap_binary, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_path),
        "--StereoFusion.min_num_pixels", "3",
        "--StereoFusion.max_num_pixels", "10000",
        "--StereoFusion.max_traversal_depth", "100",
        "--StereoFusion.max_reproj_error", "2.0",
        "--StereoFusion.max_depth_error", "0.01",
        "--StereoFusion.max_normal_error", "10.0",
        "--StereoFusion.num_threads", _NUM_THREADS,
    ]
    _run_cmd(fusion_cmd, "stereo_fusion")

    logger.info("Dense reconstruction complete: %s", fused_path)
    return fused_path


# ── Private helpers ─────────────────────────────────────────────────────────


def _run_cmd(cmd: list[str], stage_name: str) -> None:
    """Execute a COLMAP sub-command and stream its output."""
    logger.debug("Command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in result.stdout.splitlines():
            logger.debug("[%s] %s", stage_name, line)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "COLMAP %s failed (exit %d):\n%s",
            stage_name,
            exc.returncode,
            exc.stdout,
        )
        raise RuntimeError(f"COLMAP {stage_name} failed (exit {exc.returncode})") from exc
    except FileNotFoundError:
        raise FileNotFoundError(
            f"COLMAP binary not found at '{cmd[0]}'. "
            "Set the colmap_binary parameter to the correct path."
        )


def _write_image_priors(
    priors_path: Path,
    database_path: Path,
    colmap_binary: str,
) -> None:
    """Inject IMU rotation priors into the COLMAP database.

    Reads a JSON file mapping image filenames to 3x3 rotation matrices
    (row-major, 9 floats) and writes a prior_rotation to each image row
    via a temporary text model that is imported back.
    """
    priors_path = Path(priors_path)
    if not priors_path.exists():
        logger.warning("Image priors file not found: %s — skipping", priors_path)
        return

    # Try loading as JSON first (legacy format), then COLMAP text format
    priors: dict = {}
    with open(priors_path, "r", encoding="utf-8") as fh:
        content = fh.read().strip()

    if not content:
        return

    try:
        priors = json.loads(content)
    except json.JSONDecodeError:
        # Parse COLMAP image_priors.txt format:
        # IMAGE_NAME QW QX QY QZ TX TY TZ QW_STD QX_STD QY_STD QZ_STD TX_STD TY_STD TZ_STD
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                image_name = parts[0]
                # Store quaternion as rotation values (qw, qx, qy, qz)
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Convert quaternion to rotation matrix for the database
                import numpy as np
                R = np.array([
                    [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
                    [2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
                    [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy],
                ])
                priors[image_name] = R.flatten().tolist()
        logger.info("Parsed %d priors from COLMAP text format", len(priors))

    if not priors:
        return

    logger.info("Applying %d rotation priors to database ...", len(priors))

    # COLMAP does not expose a direct CLI for per-image rotation priors, so
    # we manipulate the SQLite database directly.
    import sqlite3

    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    for image_name, rot_values in priors.items():
        if len(rot_values) != 9:
            logger.warning("Skipping prior for %s — expected 9 values, got %d", image_name, len(rot_values))
            continue

        # Convert 3x3 row-major rotation matrix to quaternion (w, x, y, z)
        import numpy as np

        R = np.array(rot_values, dtype=np.float64).reshape(3, 3)
        qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)

        cursor.execute(
            "UPDATE images SET prior_qw=?, prior_qx=?, prior_qy=?, prior_qz=? WHERE name=?",
            (float(qw), float(qx), float(qy), float(qz), image_name),
        )

    conn.commit()
    conn.close()
    logger.info("Rotation priors written to database.")


def _rotation_matrix_to_quaternion(R) -> tuple:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z)."""
    import numpy as np

    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return tuple(q)
