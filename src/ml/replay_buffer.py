"""Experience replay buffer for progressive face scan learning.

After each successful pipeline run, the buffer captures representative data:
- Subsampled frames (resized to 512px max)
- FLAME parameters (shape, expression, pose per frame)
- Gaussian model statistics (aggregated per region, not full model)
- Camera parameters (intrinsics, extrinsics)
- Training metadata (final loss, iterations, convergence rate)

This data trains a Prior Network that predicts better Gaussian initializations
for future scans. The buffer uses JSON metadata + numpy .npz files (no pickle)
for safety and portability.

Data layout on disk:
    buffer_dir/
        index.json              # Scan metadata index
        scans/
            {session_id}/
                metadata.json   # Training metrics, camera count, frame count
                frames/         # Subsampled 512px JPEGs
                flame_params.npz  # shape, expression, pose arrays
                gaussian_stats.npz  # Per-region aggregated statistics
                cameras.npz     # Intrinsics + extrinsics
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# FLAME vertex indices for face regions (approximate partitioning of 5023 vertices).
# These are derived from FLAME topology — forehead/eyes/nose/mouth/cheeks/chin
# occupy specific vertex index ranges. Used for aggregating Gaussian statistics.
FACE_REGION_VERTEX_RANGES: dict[str, tuple[int, int]] = {
    "forehead": (0, 500),
    "eyes": (500, 1200),
    "nose": (1200, 1700),
    "mouth": (1700, 2300),
    "cheeks": (2300, 3200),
    "chin": (3200, 3700),
    "ears": (3700, 4200),
    "neck": (4200, 4700),
    "hair": (4700, 5023),
}

# All region names in canonical order
FACE_REGIONS = list(FACE_REGION_VERTEX_RANGES.keys())


class ScanReplayBuffer:
    """Stores representative data from completed scans for progressive learning.

    Thread-safe for single-writer usage (one pipeline run at a time).
    The index file is re-read on each mutation to handle external edits.

    Args:
        buffer_dir: Root directory for the replay buffer.
        max_scans: Maximum number of scans to retain (FIFO eviction).
        frames_per_scan: Number of representative frames to keep per scan.
    """

    def __init__(
        self,
        buffer_dir: Path,
        max_scans: int = 100,
        frames_per_scan: int = 20,
    ):
        self.buffer_dir = Path(buffer_dir)
        self.max_scans = max_scans
        self.frames_per_scan = frames_per_scan

        self.scans_dir = self.buffer_dir / "scans"
        self.index_path = self.buffer_dir / "index.json"

        # Ensure directories exist
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.scans_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load index
        if not self.index_path.exists():
            self._write_index({"scans": [], "version": 1})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_scan(
        self,
        session_id: str,
        frames_dir: Path,
        flame_params: dict,
        gaussian_model: Any,
        cameras: list[dict],
        training_metrics: dict,
    ) -> None:
        """Add a completed scan to the replay buffer.

        Args:
            session_id: Unique session identifier.
            frames_dir: Directory containing extracted frames (PNG/JPG).
            flame_params: Dict with keys 'shape_params' (N_shape,),
                'expression_params' (N_expr,), 'jaw_pose' (3,), etc.
                Arrays can be numpy or torch tensors.
            gaussian_model: A GaussianModel instance (from splatting.initializer).
                Only aggregated statistics are stored, not the full model.
            cameras: List of camera dicts, each with 'K' (3x3), 'R' (3x3),
                't' (3,), 'width', 'height'.
            training_metrics: Dict with 'final_loss', 'iterations',
                'convergence_rate', 'num_gaussians', etc.
        """
        logger.info("Adding scan '%s' to replay buffer", session_id)

        scan_dir = self.scans_dir / session_id
        scan_dir.mkdir(parents=True, exist_ok=True)

        # 1. Subsample and resize frames
        self._save_representative_frames(frames_dir, scan_dir / "frames")

        # 2. Save FLAME parameters
        self._save_flame_params(flame_params, scan_dir / "flame_params.npz")

        # 3. Save Gaussian statistics (aggregated, not full model)
        self._save_gaussian_stats(gaussian_model, scan_dir / "gaussian_stats.npz")

        # 4. Save camera parameters
        self._save_cameras(cameras, scan_dir / "cameras.npz")

        # 5. Save training metadata
        metadata = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "num_frames": len(list((scan_dir / "frames").glob("*.jpg"))),
            "num_cameras": len(cameras),
            "training_metrics": _sanitize_for_json(training_metrics),
        }
        with open(scan_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 6. Update index (and evict old scans if over capacity)
        self._update_index(session_id, metadata)

        logger.info(
            "Scan '%s' added to buffer (%d frames, %d cameras)",
            session_id,
            metadata["num_frames"],
            len(cameras),
        )

    def get_replay_batch(self, batch_size: int = 16) -> dict:
        """Sample a batch of frames + params from previous scans for replay.

        Returns a dict with:
            'frames': (batch_size, 512, 512, 3) uint8 array
            'shape_params': (batch_size, N_shape) float32
            'expression_params': (batch_size, N_expr) float32
            'gaussian_stats': dict of per-region arrays

        Falls back to smaller batch if buffer has fewer frames.
        """
        index = self._read_index()
        scan_ids = [s["session_id"] for s in index["scans"]]

        if not scan_ids:
            raise ValueError("Replay buffer is empty — no scans available")

        rng = np.random.default_rng()
        frames_list = []
        shape_list = []
        expr_list = []

        collected = 0
        max_attempts = batch_size * 5

        for _ in range(max_attempts):
            if collected >= batch_size:
                break

            # Pick a random scan
            sid = rng.choice(scan_ids)
            scan_dir = self.scans_dir / sid

            # Pick a random frame
            frame_files = sorted((scan_dir / "frames").glob("*.jpg"))
            if not frame_files:
                continue

            frame_path = rng.choice(frame_files)
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            # Resize to 512x512 (center crop + resize)
            img = _resize_square(img, 512)
            frames_list.append(img)

            # Load FLAME params for this scan
            params_path = scan_dir / "flame_params.npz"
            if params_path.exists():
                params = np.load(str(params_path))
                shape_list.append(params.get("shape_params", np.zeros(100, dtype=np.float32)))
                expr_list.append(params.get("expression_params", np.zeros(50, dtype=np.float32)))
            else:
                shape_list.append(np.zeros(100, dtype=np.float32))
                expr_list.append(np.zeros(50, dtype=np.float32))

            collected += 1

        if collected == 0:
            raise ValueError("Could not sample any frames from replay buffer")

        return {
            "frames": np.stack(frames_list),
            "shape_params": np.stack(shape_list),
            "expression_params": np.stack(expr_list),
        }

    def get_prior_training_data(self) -> dict:
        """Get all accumulated data for training the prior network.

        Returns a dict with:
            'shape_params': (N_scans, N_shape) float32
            'expression_params': (N_scans, N_expr) float32
            'gaussian_stats': dict mapping region_name -> {
                'mean_offset': (N_scans, 3),
                'mean_scale': (N_scans, 3),
                'mean_opacity': (N_scans,),
            }
            'training_metrics': list of dicts
        """
        index = self._read_index()
        scan_ids = [s["session_id"] for s in index["scans"]]

        shape_all = []
        expr_all = []
        region_stats: dict[str, dict[str, list]] = {
            r: {"mean_offset": [], "mean_scale": [], "mean_opacity": []}
            for r in FACE_REGIONS
        }
        metrics_all = []

        for sid in scan_ids:
            scan_dir = self.scans_dir / sid

            # FLAME params
            params_path = scan_dir / "flame_params.npz"
            if params_path.exists():
                params = np.load(str(params_path))
                shape_all.append(
                    params.get("shape_params", np.zeros(100, dtype=np.float32))
                )
                expr_all.append(
                    params.get("expression_params", np.zeros(50, dtype=np.float32))
                )
            else:
                shape_all.append(np.zeros(100, dtype=np.float32))
                expr_all.append(np.zeros(50, dtype=np.float32))

            # Gaussian stats
            stats_path = scan_dir / "gaussian_stats.npz"
            if stats_path.exists():
                stats = np.load(str(stats_path))
                for region in FACE_REGIONS:
                    key_offset = f"{region}_mean_offset"
                    key_scale = f"{region}_mean_scale"
                    key_opacity = f"{region}_mean_opacity"
                    region_stats[region]["mean_offset"].append(
                        stats.get(key_offset, np.zeros(3, dtype=np.float32))
                    )
                    region_stats[region]["mean_scale"].append(
                        stats.get(key_scale, np.zeros(3, dtype=np.float32))
                    )
                    region_stats[region]["mean_opacity"].append(
                        stats.get(key_opacity, np.float32(0.5))
                    )
            else:
                for region in FACE_REGIONS:
                    region_stats[region]["mean_offset"].append(np.zeros(3, dtype=np.float32))
                    region_stats[region]["mean_scale"].append(np.zeros(3, dtype=np.float32))
                    region_stats[region]["mean_opacity"].append(np.float32(0.5))

            # Training metrics
            meta_path = scan_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                metrics_all.append(meta.get("training_metrics", {}))

        # Stack arrays
        gaussian_stats_stacked = {}
        for region in FACE_REGIONS:
            gaussian_stats_stacked[region] = {
                "mean_offset": np.stack(region_stats[region]["mean_offset"]),
                "mean_scale": np.stack(region_stats[region]["mean_scale"]),
                "mean_opacity": np.array(region_stats[region]["mean_opacity"]),
            }

        return {
            "shape_params": np.stack(shape_all) if shape_all else np.zeros((0, 100), dtype=np.float32),
            "expression_params": np.stack(expr_all) if expr_all else np.zeros((0, 50), dtype=np.float32),
            "gaussian_stats": gaussian_stats_stacked,
            "training_metrics": metrics_all,
        }

    def scan_count(self) -> int:
        """Number of scans in the buffer."""
        index = self._read_index()
        return len(index["scans"])

    def summary(self) -> str:
        """Pretty-print buffer contents."""
        index = self._read_index()
        scans = index["scans"]
        lines = [
            f"Replay Buffer: {len(scans)}/{self.max_scans} scans",
            f"  Location: {self.buffer_dir}",
        ]
        for s in scans:
            ts = s.get("timestamp", "unknown")
            nf = s.get("num_frames", "?")
            nc = s.get("num_cameras", "?")
            metrics = s.get("training_metrics", {})
            loss = metrics.get("final_loss", "?")
            lines.append(f"  - {s['session_id']} ({ts}): {nf} frames, {nc} cams, loss={loss}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_representative_frames(self, frames_dir: Path, output_dir: Path) -> None:
        """Subsample frames and resize to 512px max dimension."""
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = Path(frames_dir)

        # Collect all image files
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        all_frames = sorted(
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in exts
        )

        if not all_frames:
            logger.warning("No frames found in %s", frames_dir)
            return

        # Subsample: keep every Nth frame to get frames_per_scan
        n = len(all_frames)
        step = max(1, n // self.frames_per_scan)
        selected = all_frames[::step][:self.frames_per_scan]

        for i, frame_path in enumerate(selected):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            # Resize so max dimension is 512px
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > 512:
                scale = 512.0 / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out_path = output_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        logger.info("Saved %d representative frames to %s", len(selected), output_dir)

    def _save_flame_params(self, flame_params: dict, output_path: Path) -> None:
        """Save FLAME parameters as .npz (converting tensors to numpy)."""
        arrays = {}
        for key, value in flame_params.items():
            if value is None:
                continue
            arr = _to_numpy(value)
            if arr is not None:
                arrays[key] = arr.astype(np.float32)
        np.savez_compressed(str(output_path), **arrays)

    def _save_gaussian_stats(self, gaussian_model: Any, output_path: Path) -> None:
        """Save aggregated per-region Gaussian statistics.

        Instead of storing the full model (which can be 100MB+), we compute
        summary statistics per face region: mean position offset from centroid,
        mean scale, mean opacity.
        """
        stats = {}

        try:
            positions = _to_numpy(gaussian_model.positions)  # (N, 3)
            scales = _to_numpy(gaussian_model.scales)        # (N, 3)
            opacities = _to_numpy(gaussian_model.opacities)  # (N, 1)
        except (AttributeError, TypeError):
            logger.warning("Could not extract Gaussian model parameters, saving empty stats")
            np.savez_compressed(str(output_path))
            return

        n_gaussians = len(positions)
        centroid = positions.mean(axis=0)

        # Partition Gaussians into face regions based on proximity to
        # FLAME vertex ranges. This is approximate — we use position-based
        # assignment since we don't have the binding metadata here.
        #
        # Simple approach: divide the Gaussian positions into regions by
        # their Y-coordinate (top-to-bottom) and X-coordinate (left-right).
        # This gives a rough region assignment.
        offsets = positions - centroid  # (N, 3)
        y_normalized = offsets[:, 1]  # up-down
        offsets[:, 0]  # left-right

        # Sort by Y to divide into vertical bands
        y_sorted_idx = np.argsort(y_normalized)
        region_size = n_gaussians // len(FACE_REGIONS)

        for i, region in enumerate(FACE_REGIONS):
            start = i * region_size
            end = (i + 1) * region_size if i < len(FACE_REGIONS) - 1 else n_gaussians
            region_idx = y_sorted_idx[start:end]

            if len(region_idx) == 0:
                stats[f"{region}_mean_offset"] = np.zeros(3, dtype=np.float32)
                stats[f"{region}_mean_scale"] = np.zeros(3, dtype=np.float32)
                stats[f"{region}_mean_opacity"] = np.float32(0.5)
                continue

            stats[f"{region}_mean_offset"] = offsets[region_idx].mean(axis=0).astype(np.float32)
            stats[f"{region}_mean_scale"] = scales[region_idx].mean(axis=0).astype(np.float32)
            # Opacities are in logit space — convert to probability for stats
            opacity_probs = 1.0 / (1.0 + np.exp(-opacities[region_idx].flatten()))
            stats[f"{region}_mean_opacity"] = np.float32(opacity_probs.mean())

        stats["centroid"] = centroid.astype(np.float32)
        stats["num_gaussians"] = np.int32(n_gaussians)

        np.savez_compressed(str(output_path), **stats)

    def _save_cameras(self, cameras: list[dict], output_path: Path) -> None:
        """Save camera parameters as .npz."""
        if not cameras:
            np.savez_compressed(str(output_path))
            return

        intrinsics = []
        extrinsics_r = []
        extrinsics_t = []
        widths = []
        heights = []

        for cam in cameras:
            K = _to_numpy(cam.get("K", np.eye(3)))
            R = _to_numpy(cam.get("R", np.eye(3)))
            t = _to_numpy(cam.get("t", np.zeros(3)))
            intrinsics.append(K.astype(np.float32))
            extrinsics_r.append(R.astype(np.float32))
            extrinsics_t.append(t.flatten().astype(np.float32))
            widths.append(cam.get("width", 0))
            heights.append(cam.get("height", 0))

        np.savez_compressed(
            str(output_path),
            intrinsics=np.stack(intrinsics),
            extrinsics_r=np.stack(extrinsics_r),
            extrinsics_t=np.stack(extrinsics_t),
            widths=np.array(widths, dtype=np.int32),
            heights=np.array(heights, dtype=np.int32),
        )

    def _read_index(self) -> dict:
        """Read the index file from disk."""
        if not self.index_path.exists():
            return {"scans": [], "version": 1}
        with open(self.index_path) as f:
            return json.load(f)

    def _write_index(self, index: dict) -> None:
        """Write the index file to disk atomically."""
        tmp_path = self.index_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(index, f, indent=2)
        # Atomic rename (works on Windows with same-volume paths)
        tmp_path.replace(self.index_path)

    def _update_index(self, session_id: str, metadata: dict) -> None:
        """Add a scan to the index and evict old scans if over capacity."""
        index = self._read_index()

        # Remove existing entry for this session (update in place)
        index["scans"] = [s for s in index["scans"] if s["session_id"] != session_id]

        # Add new entry
        index["scans"].append(metadata)

        # Evict oldest scans if over capacity (FIFO)
        while len(index["scans"]) > self.max_scans:
            evicted = index["scans"].pop(0)
            evicted_dir = self.scans_dir / evicted["session_id"]
            if evicted_dir.exists():
                shutil.rmtree(evicted_dir, ignore_errors=True)
                logger.info("Evicted scan '%s' from buffer (over capacity)", evicted["session_id"])

        self._write_index(index)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _to_numpy(value: Any) -> Optional[np.ndarray]:
    """Convert a value to numpy array, handling torch tensors."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    # Handle torch tensors without importing torch at module level
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except (ValueError, TypeError):
        return None


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):  # torch scalar
        return obj.item()
    return obj


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize an image to size x size with center crop."""
    h, w = img.shape[:2]
    # Center crop to square
    if h > w:
        start = (h - w) // 2
        img = img[start:start + w]
    elif w > h:
        start = (w - h) // 2
        img = img[:, start:start + h]
    # Resize
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
