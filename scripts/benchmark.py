#!/usr/bin/env python3
"""Face3D Pipeline Benchmark Suite

Standalone benchmark runner that profiles each pipeline stage with synthetic
data and reports per-stage timing, GPU memory peaks, and quality metrics.

Modes:
    --quick     30-second test with minimal data (5 frames, 50 Gaussians)
    --full      5-minute full benchmark (20 frames, 500 Gaussians)
    --report    Generate HTML benchmark report from saved JSON

Usage:
    python scripts/benchmark.py --quick
    python scripts/benchmark.py --full
    python scripts/benchmark.py --report

Output:
    data/output/benchmarks/benchmark_<timestamp>.json
    data/output/benchmarks/benchmark_<timestamp>.html  (with --report)

Metrics per stage:
    - Wall-clock time (seconds)
    - GPU memory peak (MB, if CUDA available)
    - Output validity (pass/fail)
    - Quality metrics where applicable (PSNR, SSIM for renders)
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import struct
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_info() -> dict:
    """Return GPU info dict or empty dict if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": props.name,
                "total_memory_mb": props.total_mem // (1024 * 1024),
                "compute_capability": f"{props.major}.{props.minor}",
                "cuda_version": torch.version.cuda,
            }
    except ImportError:
        pass
    return {}


def _peak_gpu_memory_mb() -> float:
    """Return peak GPU memory usage in MB since last reset, or 0."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def _reset_gpu_memory_stats():
    """Reset GPU memory tracking."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextmanager
def timed_stage(name: str, results: dict):
    """Context manager that records timing and GPU memory for a stage."""
    _reset_gpu_memory_stats()
    log.info("=" * 60)
    log.info("STAGE: %s", name)
    log.info("=" * 60)
    t0 = time.perf_counter()
    stage_result = {"status": "running", "time_s": 0.0, "gpu_peak_mb": 0.0}
    results[name] = stage_result
    try:
        yield stage_result
        elapsed = time.perf_counter() - t0
        stage_result["time_s"] = round(elapsed, 3)
        stage_result["gpu_peak_mb"] = round(_peak_gpu_memory_mb(), 1)
        stage_result["status"] = "pass"
        log.info("  PASS: %.3fs (GPU peak: %.1f MB)", elapsed, stage_result["gpu_peak_mb"])
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        stage_result["time_s"] = round(elapsed, 3)
        stage_result["gpu_peak_mb"] = round(_peak_gpu_memory_mb(), 1)
        stage_result["status"] = "fail"
        stage_result["error"] = str(exc)
        log.error("  FAIL: %s (%.3fs)", exc, elapsed)


def _create_synthetic_frames(out_dir: Path, count: int = 5) -> list[Path]:
    """Generate synthetic test frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    paths = []
    for i in range(count):
        frame = rng.integers(100, 160, size=(480, 640, 3), dtype=np.uint8)
        cv2.ellipse(frame, (320, 240), (110, 140), 0, 0, 360, (130, 170, 210), -1)
        cv2.circle(frame, (285, 210), 12, (40, 40, 40), -1)
        cv2.circle(frame, (355, 210), 12, (40, 40, 40), -1)
        path = out_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def _create_synthetic_colmap(model_dir: Path, num_cameras: int = 5, num_points: int = 100):
    """Create a synthetic COLMAP binary model."""
    import math
    from utils.colmap_io import rotmat_to_qvec

    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)

    with open(model_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", num_cameras))
        for cam_id in range(1, num_cameras + 1):
            f.write(struct.pack("<IiQQ", cam_id, 1, 640, 480))
            f.write(struct.pack("<4d", 525.0, 525.0, 320.0, 240.0))

    with open(model_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", num_cameras))
        for img_id in range(1, num_cameras + 1):
            angle = math.pi * (img_id - 1) / max(num_cameras - 1, 1)
            cam_pos = np.array([2.0 * math.cos(angle), 0.0, 2.0 * math.sin(angle)])
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right) + 1e-12
            up = np.cross(right, forward)
            R = np.stack([right, -up, forward], axis=0)
            tvec = -R @ cam_pos
            qvec = rotmat_to_qvec(R)
            if qvec[0] < 0:
                qvec = -qvec

            f.write(struct.pack("<I", img_id))
            f.write(struct.pack("<4d", *qvec))
            f.write(struct.pack("<3d", *tvec))
            f.write(struct.pack("<I", img_id))
            name = f"frame_{img_id - 1:06d}.png"
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 5))
            for _ in range(5):
                f.write(struct.pack("<2d", rng.uniform(0, 640), rng.uniform(0, 480)))
                f.write(struct.pack("<q", int(rng.integers(1, num_points + 1))))

    with open(model_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", num_points))
        for pt_id in range(1, num_points + 1):
            xyz = rng.normal(0, 0.12, size=3)
            rgb = rng.integers(100, 220, size=3, dtype=np.uint8)
            f.write(struct.pack("<Q", pt_id))
            f.write(struct.pack("<3d", *xyz))
            f.write(struct.pack("<3B", *rgb))
            f.write(struct.pack("<d", rng.uniform(0.1, 2.0)))
            f.write(struct.pack("<Q", 2))
            for _ in range(2):
                f.write(struct.pack("<II",
                    int(rng.integers(1, num_cameras + 1)),
                    int(rng.integers(0, 5)),
                ))


# ---------------------------------------------------------------------------
# Benchmark stages
# ---------------------------------------------------------------------------

def run_benchmark(mode: str = "quick") -> dict:
    """Run the full benchmark suite.

    Args:
        mode: "quick" (5 frames, ~30s) or "full" (20 frames, ~5min).

    Returns:
        Dict with all benchmark results.
    """
    is_quick = mode == "quick"
    num_frames = 5 if is_quick else 20
    num_points = 100 if is_quick else 500

    import tempfile
    work_dir = Path(tempfile.mkdtemp(prefix="face3d_bench_"))
    log.info("Benchmark work directory: %s", work_dir)

    results = {
        "metadata": {
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": _gpu_info(),
            "num_frames": num_frames,
            "num_points": num_points,
        },
        "stages": {},
    }
    stages = results["stages"]

    # --- Create synthetic data ---
    frames_dir = work_dir / "frames"
    colmap_dir = work_dir / "colmap" / "sparse" / "0"
    srgb_dir = work_dir / "frames_srgb"

    with timed_stage("create_synthetic_data", stages):
        _create_synthetic_frames(frames_dir, count=num_frames)
        _create_synthetic_colmap(colmap_dir, num_cameras=num_frames, num_points=num_points)

    # --- Stage 2: Color correction ---
    with timed_stage("color_correction", stages) as ctx:
        from capture.color_correction import batch_color_correct
        result = batch_color_correct(frames_dir, srgb_dir, log_type="slog3", is_log=True)
        ctx["output_count"] = len(result)

    # --- Stage 3: Frame filtering ---
    with timed_stage("frame_filtering", stages) as ctx:
        from capture.frame_filter import filter_frames
        report_path = work_dir / "filter_report.json"
        selected = filter_frames(frames_dir, report_path, blur_threshold=5.0, require_face=False)
        ctx["selected_count"] = len(selected)
        ctx["total_count"] = num_frames

    # --- Stage 4-5: Orientation estimation ---
    with timed_stage("orientation_estimation", stages) as ctx:
        from sensors.orientation import compute_rotations

        rng = np.random.default_rng(55)
        n_imu = 1000 if not is_quick else 200
        imu_data = {
            "timestamps": np.linspace(0.0, 10.0, n_imu),
            "accel_xyz": np.column_stack([
                np.zeros(n_imu), np.zeros(n_imu), np.full(n_imu, -9.81)
            ]) + rng.normal(0, 0.05, (n_imu, 3)),
            "gyro_xyz": np.column_stack([
                np.zeros(n_imu), np.full(n_imu, 0.5), np.zeros(n_imu)
            ]) + rng.normal(0, 0.01, (n_imu, 3)),
        }
        rotations = compute_rotations(imu_data, sample_rate=n_imu / 10.0)
        ctx["num_rotations"] = len(rotations)

    # --- Stage 6: COLMAP I/O ---
    with timed_stage("colmap_io_read", stages) as ctx:
        from utils.colmap_io import read_cameras_binary, read_images_binary, read_points3d_binary

        cameras = read_cameras_binary(colmap_dir / "cameras.bin")
        images = read_images_binary(colmap_dir / "images.bin")
        points = read_points3d_binary(colmap_dir / "points3D.bin")
        ctx["cameras"] = len(cameras)
        ctx["images"] = len(images)
        ctx["points"] = len(points)

    # --- Stage 12: Gaussian initialization ---
    with timed_stage("gaussian_initialization", stages) as ctx:
        from splatting.initializer import initialize_from_colmap_sparse

        model = initialize_from_colmap_sparse(colmap_dir)
        ctx["num_gaussians"] = model.num_gaussians

    # --- Stage 14: PLY export ---
    with timed_stage("ply_export", stages) as ctx:
        from splatting.initializer import _save_gaussians_ply

        ply_path = work_dir / "output.ply"
        _save_gaussians_ply(model, ply_path)
        ctx["ply_size_kb"] = round(ply_path.stat().st_size / 1024, 1)

    # --- Summary ---
    total_time = sum(s.get("time_s", 0) for s in stages.values())
    results["summary"] = {
        "total_time_s": round(total_time, 3),
        "stages_passed": sum(1 for s in stages.values() if s.get("status") == "pass"),
        "stages_failed": sum(1 for s in stages.values() if s.get("status") == "fail"),
        "peak_gpu_mb": max((s.get("gpu_peak_mb", 0) for s in stages.values()), default=0),
    }

    # Save results
    output_dir = PROJECT_ROOT / "data" / "output" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("Results saved to %s", json_path)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("BENCHMARK SUMMARY (%s mode)", mode)
    log.info("=" * 60)
    for name, stage in stages.items():
        status = stage.get("status", "?")
        time_s = stage.get("time_s", 0)
        gpu_mb = stage.get("gpu_peak_mb", 0)
        icon = "PASS" if status == "pass" else "FAIL"
        log.info("  %-30s %s  %7.3fs  GPU: %6.1f MB", name, icon, time_s, gpu_mb)
    log.info("  %-30s      %7.3fs  GPU peak: %.1f MB",
             "TOTAL", total_time, results["summary"]["peak_gpu_mb"])

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    return results


def generate_report(json_path: Path) -> Path:
    """Generate a simple HTML report from benchmark JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    stages = data.get("stages", {})
    summary = data.get("summary", {})

    rows = ""
    for name, stage in stages.items():
        status_class = "pass" if stage.get("status") == "pass" else "fail"
        rows += f"""
        <tr class="{status_class}">
            <td>{name}</td>
            <td>{stage.get('status', '?')}</td>
            <td>{stage.get('time_s', 0):.3f}s</td>
            <td>{stage.get('gpu_peak_mb', 0):.1f} MB</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Face3D Benchmark Report</title>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 2em; background: #1a1a2e; color: #e0e0e0; }}
    h1 {{ color: #16c784; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
    th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
    th {{ background: #16213e; color: #16c784; }}
    tr.pass {{ background: #0d2818; }}
    tr.fail {{ background: #2d0a0a; }}
    .meta {{ color: #888; font-size: 0.9em; }}
    .summary {{ font-size: 1.2em; margin: 1em 0; padding: 1em; background: #16213e; border-radius: 8px; }}
</style></head><body>
    <h1>Face3D Pipeline Benchmark Report</h1>
    <div class="meta">
        <p>Mode: {meta.get('mode', '?')} | Date: {meta.get('timestamp', '?')}</p>
        <p>Platform: {meta.get('platform', '?')} | Python: {meta.get('python', '?')}</p>
        <p>GPU: {meta.get('gpu', {}).get('name', 'N/A')}</p>
    </div>
    <div class="summary">
        Total: {summary.get('total_time_s', 0):.3f}s |
        Passed: {summary.get('stages_passed', 0)} |
        Failed: {summary.get('stages_failed', 0)} |
        Peak GPU: {summary.get('peak_gpu_mb', 0):.1f} MB
    </div>
    <table>
        <tr><th>Stage</th><th>Status</th><th>Time</th><th>GPU Peak</th></tr>
        {rows}
    </table>
</body></html>"""

    html_path = json_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    log.info("HTML report: %s", html_path)
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Face3D Pipeline Benchmark Suite")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Quick benchmark (~30s)")
    group.add_argument("--full", action="store_true", help="Full benchmark (~5min)")
    group.add_argument("--report", type=str, metavar="JSON_PATH",
                       help="Generate HTML report from benchmark JSON")
    args = parser.parse_args()

    if args.report:
        json_path = Path(args.report)
        if not json_path.exists():
            log.error("JSON file not found: %s", json_path)
            sys.exit(1)
        generate_report(json_path)
    else:
        mode = "full" if args.full else "quick"
        results = run_benchmark(mode=mode)
        if results["summary"]["stages_failed"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
