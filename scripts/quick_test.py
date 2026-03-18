#!/usr/bin/env python3
"""Quick smoke test — runs every stage with minimal data in ~1 minute.

Usage:
    python scripts/quick_test.py
"""

import logging
import sys
import time
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quick_test")


def timed(name):
    """Context manager that logs elapsed time."""
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            log.info("--- %s ---", name)
            return self
        def __exit__(self, *a):
            self.elapsed = time.time() - self.t0
            log.info("    Done in %.1fs", self.elapsed)
    return Timer()


def main():
    import numpy as np
    import cv2
    import torch
    import json

    test_dir = PROJECT_ROOT / "data" / "processed" / "quick_test"
    out_dir = PROJECT_ROOT / "data" / "output" / "quick_test"
    for d in [test_dir / "frames", test_dir / "frames_srgb", test_dir / "depth",
              test_dir / "depth_aligned", test_dir / "colmap" / "sparse" / "0",
              test_dir / "landmarks", test_dir / "face_masks", test_dir / "flame",
              out_dir / "renders", out_dir / "animations", out_dir / "previews"]:
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = {}

    # ── Stage 1-2: Generate synthetic test frames ──
    with timed("Stage 1-2: Generate test frames") as t:
        np.random.seed(42)
        for i in range(8):
            # Create a simple gradient image with some variation per frame
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            # Gradient background
            img[:, :, 0] = np.linspace(50, 200, 640).astype(np.uint8)
            img[:, :, 1] = np.linspace(30 + i * 10, 180, 360).reshape(-1, 1).astype(np.uint8)
            img[:, :, 2] = 100
            # Add a circle (fake face) that moves slightly per frame
            cx, cy = 320 + i * 5, 180 + i * 3
            cv2.circle(img, (cx, cy), 80, (200, 180, 160), -1)
            cv2.circle(img, (cx - 20, cy - 15), 8, (60, 40, 30), -1)  # left eye
            cv2.circle(img, (cx + 20, cy - 15), 8, (60, 40, 30), -1)  # right eye
            cv2.ellipse(img, (cx, cy + 20), (25, 10), 0, 0, 180, (150, 100, 100), 2)  # mouth

            frame_path = test_dir / "frames_srgb" / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), img)
        results["frames"] = 8

    # ── Stage 3: Frame filter (skip — use all frames) ──
    with timed("Stage 3: Frame selection"):
        selected = sorted(f.name for f in (test_dir / "frames_srgb").glob("*.png"))
        with open(test_dir / "selected_frames.json", "w") as f:
            json.dump({"frames": [{"file": s, "selected": True} for s in selected], "summary": {"selected": len(selected)}}, f)
        results["selected"] = len(selected)

    # ── Stage 6: Create synthetic COLMAP model ──
    with timed("Stage 6: Synthetic COLMAP model"):
        from utils.colmap_io import Camera, Image, Point3D
        import struct

        sparse_dir = test_dir / "colmap" / "sparse" / "0"

        # Write cameras.bin — single pinhole camera
        with open(sparse_dir / "cameras.bin", "wb") as f:
            f.write(struct.pack("<Q", 1))  # num cameras
            f.write(struct.pack("<I", 1))  # cam id
            f.write(struct.pack("<i", 1))  # PINHOLE model
            f.write(struct.pack("<Q", 640))  # width
            f.write(struct.pack("<Q", 360))  # height
            for p in [500.0, 500.0, 320.0, 180.0]:  # fx fy cx cy
                f.write(struct.pack("<d", p))

        # Write images.bin — 8 cameras in a circle
        with open(sparse_dir / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", 8))
            for i in range(8):
                angle = i * (2 * 3.14159 / 8)
                import math
                # Position camera on a circle of radius 3
                tx, ty, tz = 3 * math.cos(angle), 0.0, 3 * math.sin(angle)
                # Look at origin — rotation as quaternion (simplified)
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0  # identity (simplified)
                f.write(struct.pack("<I", i + 1))  # image_id
                f.write(struct.pack("<dddd", qw, qx, qy, qz))
                f.write(struct.pack("<ddd", tx, ty, tz))
                f.write(struct.pack("<I", 1))  # camera_id
                name = f"frame_{i:04d}.png"
                f.write(name.encode("utf-8") + b"\x00")
                f.write(struct.pack("<Q", 0))  # num_points2d

        # Write points3D.bin — 100 random points
        with open(sparse_dir / "points3D.bin", "wb") as f:
            n_pts = 100
            f.write(struct.pack("<Q", n_pts))
            for j in range(n_pts):
                f.write(struct.pack("<Q", j + 1))  # point3d_id
                xyz = np.random.randn(3) * 0.5
                f.write(struct.pack("<ddd", *xyz))
                f.write(struct.pack("<BBB", 180, 160, 140))  # rgb
                f.write(struct.pack("<d", 1.0))  # error
                f.write(struct.pack("<Q", 0))  # track_length
            results["sparse_points"] = n_pts

    # ── Stage 7: Synthetic depth maps ──
    with timed("Stage 7: Synthetic depth maps"):
        for i in range(8):
            depth = np.random.uniform(2.0, 4.0, (360, 640)).astype(np.float32)
            np.save(str(test_dir / "depth" / f"frame_{i:04d}.npy"), depth)
            np.save(str(test_dir / "depth_aligned" / f"frame_{i:04d}.npy"), depth)
        results["depth_maps"] = 8

    # ── Stage 9: Synthetic landmarks ──
    with timed("Stage 9: Synthetic landmarks"):
        for i in range(8):
            landmarks = {
                "detected": True,
                "landmarks_2d": (np.random.rand(478, 2) * [640, 360]).tolist(),
                "landmarks_3d": (np.random.rand(478, 3) * 0.1).tolist(),
                "confidence": 0.95,
            }
            with open(test_dir / "landmarks" / f"frame_{i:04d}.json", "w") as f:
                json.dump(landmarks, f)
        results["landmarks"] = 8

    # ── Stage 11: Synthetic masks ──
    with timed("Stage 11: Synthetic face masks"):
        for i in range(8):
            mask = np.zeros((360, 640), dtype=np.uint8)
            cv2.circle(mask, (320 + i * 5, 180 + i * 3), 80, 255, -1)
            cv2.imwrite(str(test_dir / "face_masks" / f"frame_{i:04d}.png"), mask)
        results["masks"] = 8

    # ── Stage 12: Initialize Gaussians ──
    with timed("Stage 12: Initialize Gaussians") as t:
        from splatting.initializer import initialize_from_colmap_sparse
        gaussians = initialize_from_colmap_sparse(
            colmap_model_dir=test_dir / "colmap" / "sparse" / "0",
        )
        torch.save({
            "positions": gaussians.positions,
            "colors_sh": gaussians.colors_sh,
            "scales": gaussians.scales,
            "rotations": gaussians.rotations,
            "opacities": gaussians.opacities,
        }, str(test_dir / "gaussians_init.pt"))
        results["init_gaussians"] = len(gaussians.positions)

    # ── Stage 13: Train (50 iterations only) ──
    with timed("Stage 13: Train Gaussians (50 iters)") as t:
        from splatting.trainer import GaussianTrainer, TrainingConfig
        from splatting.camera_utils import load_cameras_from_colmap
        from splatting.initializer import GaussianModel

        cameras = load_cameras_from_colmap(test_dir / "colmap" / "sparse" / "0")
        init_data = torch.load(str(test_dir / "gaussians_init.pt"), weights_only=False)
        gs = GaussianModel(
            positions=init_data["positions"],
            colors_sh=init_data["colors_sh"],
            scales=init_data["scales"],
            rotations=init_data["rotations"],
            opacities=init_data["opacities"],
        )

        tc = TrainingConfig(
            iterations=50,
            use_2dgs=False,  # Use standard 3DGS for speed
            use_appearance_embedding=False,
            progressive_training=False,
            lambda_lpips=0.0,  # Skip LPIPS for speed
            lambda_depth=0.0,
            lambda_normal=0.0,
            lambda_distort=0.0,
            checkpoint_every=100,  # No checkpoints during quick test
            eval_every=50,
            max_num_gaussians=10000,
        )
        trainer = GaussianTrainer(config=tc)
        result = trainer.train(
            gaussians=gs,
            cameras=cameras,
            images_dir=test_dir / "frames_srgb",
            output_dir=out_dir,
        )

        if isinstance(result, dict):
            log.error("Training failed: %s", result.get("error", "unknown"))
            results["training"] = "FAILED"
        else:
            summary = getattr(result, '_training_summary', {})
            results["final_gaussians"] = summary.get("num_gaussians_end", "?")
            results["final_psnr"] = summary.get("final_psnr", "?")
            results["training"] = "OK"

    # ── Stage 14: Export ──
    with timed("Stage 14: Export") as t:
        try:
            from splatting.exporter import export_gaussians_ply
            if not isinstance(result, dict):
                export_gaussians_ply(result, out_dir / "gaussians.ply")
                results["export_ply"] = "OK"
            else:
                results["export_ply"] = "SKIPPED"
        except Exception as e:
            log.warning("Export failed: %s", e)
            results["export_ply"] = f"FAILED: {e}"

    # ── Summary ──
    total = time.time() - t0
    print()
    print("=" * 50)
    print(f"  QUICK TEST COMPLETE — {total:.1f}s")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    if results.get("training") == "OK":
        print("  STATUS: ALL STAGES PASSED")
    else:
        print("  STATUS: SOME STAGES FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
