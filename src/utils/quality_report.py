"""Quality report generator and comparison visualization for Face3D pipeline.

Generates self-contained HTML reports with inline SVG charts and base64-encoded
images. Also produces comparison grids (ground truth vs rendered vs depth vs
error map) and computes per-stage timing breakdowns.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

log = logging.getLogger("face3d.quality_report")


# ---------------------------------------------------------------------------
# Stage name mapping
# ---------------------------------------------------------------------------

STAGE_NAMES = {
    1: "Extract frames",
    2: "Color correction",
    3: "Filter frames",
    4: "Parse sensors",
    5: "Compute priors",
    6: "COLMAP / DA3",
    7: "Depth estimation",
    8: "Align depth",
    9: "Landmarks",
    10: "FLAME fitting",
    11: "Segmentation",
    12: "Init Gaussians",
    13: "Train Gaussians",
    14: "Export",
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def compute_stage_timing(metrics_jsonl_path: str | Path) -> dict[str, Any]:
    """Parse metrics.jsonl and return per-stage timing plus bottleneck info.

    Returns
    -------
    dict with keys:
        stages : dict[str, float]  — {stage_name: elapsed_seconds}
        bottleneck : str           — name of the slowest stage
        bottleneck_seconds : float
        total_seconds : float
    """
    metrics_jsonl_path = Path(metrics_jsonl_path)
    stages: dict[str, float] = {}
    if not metrics_jsonl_path.exists():
        return {"stages": stages, "bottleneck": "", "bottleneck_seconds": 0.0, "total_seconds": 0.0}

    for line in metrics_jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            stage_num = entry.get("stage")
            elapsed = entry.get("elapsed_s", 0.0)
            name = STAGE_NAMES.get(stage_num, f"Stage {stage_num}")
            stages[name] = elapsed
        except (json.JSONDecodeError, KeyError):
            continue

    total = sum(stages.values())
    if stages:
        bottleneck = max(stages, key=stages.get)
        bottleneck_s = stages[bottleneck]
    else:
        bottleneck, bottleneck_s = "", 0.0

    return {
        "stages": stages,
        "bottleneck": bottleneck,
        "bottleneck_seconds": bottleneck_s,
        "total_seconds": total,
    }


# ---------------------------------------------------------------------------
# Comparison grid
# ---------------------------------------------------------------------------

def _load_image(path: Path, max_size: int | None = None) -> np.ndarray:
    """Load an image as RGB numpy array, optionally downsizing."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max_size is not None:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _make_error_heatmap(gt: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    """Absolute difference between gt and rendered, mapped to turbo colormap."""
    diff = np.mean(np.abs(gt.astype(np.float32) - rendered.astype(np.float32)), axis=-1)
    diff_norm = np.clip(diff / 255.0, 0, 1)
    # Apply turbo-like colormap via cv2
    diff_u8 = (diff_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def generate_comparison_grid(
    images_dir: str | Path,
    renders_dir: str | Path,
    output_path: str | Path,
    num_comparisons: int = 5,
) -> Path:
    """Create side-by-side comparison grids: GT | Rendered | Depth | Error Map.

    Parameters
    ----------
    images_dir : path to ground truth frames (frames_srgb)
    renders_dir : path to rendered frames from Gaussians (renders/)
    output_path : where to save the montage PNG
    num_comparisons : number of comparison rows

    Returns
    -------
    Path to the saved montage.
    """
    images_dir = Path(images_dir)
    renders_dir = Path(renders_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather ground truth frames
    gt_frames = sorted(
        list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    )
    if not gt_frames:
        log.warning("No ground truth frames found in %s", images_dir)
        return output_path

    # Gather rendered frames
    render_frames = sorted(
        list(renders_dir.glob("frame_*.png")) + list(renders_dir.glob("*.png"))
    )
    if not render_frames:
        log.warning("No render frames found in %s", renders_dir)
        return output_path

    # Pick evenly spaced indices from GT
    n = min(num_comparisons, len(gt_frames), len(render_frames))
    if n == 0:
        log.warning("Not enough frames for comparison")
        return output_path

    gt_indices = np.linspace(0, len(gt_frames) - 1, n, dtype=int)
    render_indices = np.linspace(0, len(render_frames) - 1, n, dtype=int)

    tile_size = 256
    rows = []
    individual_dir = output_path.parent / "comparisons"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for i, (gi, ri) in enumerate(zip(gt_indices, render_indices)):
        gt_img = _load_image(gt_frames[gi], max_size=tile_size)
        rend_img = _load_image(render_frames[ri], max_size=tile_size)

        # Resize to same dimensions
        h = min(gt_img.shape[0], rend_img.shape[0])
        w = min(gt_img.shape[1], rend_img.shape[1])
        gt_img = cv2.resize(gt_img, (w, h))
        rend_img = cv2.resize(rend_img, (w, h))

        # Error heatmap
        error_map = _make_error_heatmap(gt_img, rend_img)

        # Depth placeholder (gray if no depth available)
        depth_vis = np.full_like(gt_img, 128, dtype=np.uint8)
        depth_npy = images_dir.parent / "depth" / (gt_frames[gi].stem + ".npy")
        if depth_npy.exists():
            depth = np.load(str(depth_npy))
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
            depth_color = cv2.applyColorMap(
                cv2.resize(depth_norm, (w, h)), cv2.COLORMAP_TURBO
            )
            depth_vis = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        row = np.concatenate([gt_img, rend_img, depth_vis, error_map], axis=1)
        rows.append(row)

        # Save individual comparison pair
        pair = np.concatenate([gt_img, rend_img], axis=1)
        pair_bgr = cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(individual_dir / f"comparison_{i:02d}.png"), pair_bgr)

    # Add header labels
    label_h = 30
    label_w = rows[0].shape[1]
    label_bar = np.ones((label_h, label_w, 3), dtype=np.uint8) * 40
    col_w = rows[0].shape[1] // 4
    labels = ["Ground Truth", "Rendered", "Depth", "Error Map"]
    for j, lbl in enumerate(labels):
        x = j * col_w + col_w // 2 - len(lbl) * 5
        cv2.putText(
            label_bar, lbl, (max(x, 5), 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    montage = np.concatenate([label_bar] + rows, axis=0)
    montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), montage_bgr)
    log.info("Saved comparison grid to %s (%d comparisons)", output_path, n)
    return output_path


# ---------------------------------------------------------------------------
# HTML report helpers
# ---------------------------------------------------------------------------

def _img_to_base64(path: Path, max_width: int = 600) -> str:
    """Read an image, resize, and return as base64 data URI."""
    if not path.exists():
        return ""
    img = PILImage.open(path)
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _svg_bar_chart(stages: dict[str, float], width: int = 700, bar_height: int = 24) -> str:
    """Generate an inline SVG horizontal bar chart for stage timings."""
    if not stages:
        return "<p>No timing data available.</p>"

    max_val = max(stages.values()) if stages else 1
    margin_left = 160
    chart_w = width - margin_left - 60
    n = len(stages)
    svg_h = n * (bar_height + 6) + 40

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_h}" '
        f'style="font-family:monospace;font-size:12px;">'
    ]

    y = 20
    for name, secs in stages.items():
        bar_w = max(int(secs / max_val * chart_w), 2) if max_val > 0 else 2
        color = "#4a9eff" if secs < max_val * 0.8 else "#ff6b6b"
        lines.append(
            f'<text x="{margin_left - 8}" y="{y + bar_height // 2 + 4}" '
            f'text-anchor="end" fill="#ccc">{name}</text>'
        )
        lines.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_w}" height="{bar_height}" '
            f'rx="3" fill="{color}" />'
        )
        label = f"{secs:.1f}s"
        lines.append(
            f'<text x="{margin_left + bar_w + 6}" y="{y + bar_height // 2 + 4}" '
            f'fill="#eee">{label}</text>'
        )
        y += bar_height + 6

    lines.append("</svg>")
    return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def _collect_output_manifest(out_dir: Path) -> list[dict]:
    """List output files with sizes."""
    manifest = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            manifest.append({
                "path": str(p.relative_to(out_dir)),
                "size": p.stat().st_size,
                "size_human": _format_size(p.stat().st_size),
            })
    return manifest


def _load_training_curves(out_dir: Path) -> dict[str, list]:
    """Try to load training loss/PSNR from checkpoint or log file."""
    curves: dict[str, list] = {"iterations": [], "loss": [], "psnr": []}

    # Check for training_log.json (written by trainer)
    log_path = out_dir / "training_log.json"
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
            curves["iterations"] = data.get("iterations", [])
            curves["loss"] = data.get("loss", [])
            curves["psnr"] = data.get("psnr", [])
            return curves
        except (json.JSONDecodeError, KeyError):
            pass

    # Check for checkpoints with training history
    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        try:
            import torch
            final_ckpt = ckpt_dir / "final.pt"
            if final_ckpt.exists():
                state = torch.load(str(final_ckpt), map_location="cpu", weights_only=False)
                if "training_history" in state:
                    hist = state["training_history"]
                    curves["iterations"] = hist.get("iterations", [])
                    curves["loss"] = hist.get("loss", [])
                    curves["psnr"] = hist.get("psnr", [])
        except Exception:
            pass

    return curves


def _svg_line_chart(
    x_vals: list, y_vals: list, label: str, color: str = "#4a9eff",
    width: int = 500, height: int = 200,
) -> str:
    """Simple SVG line chart."""
    if not x_vals or not y_vals or len(x_vals) != len(y_vals):
        return ""

    margin = {"top": 20, "right": 20, "bottom": 35, "left": 60}
    cw = width - margin["left"] - margin["right"]
    ch = height - margin["top"] - margin["bottom"]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 0.01

    def sx(v):
        return margin["left"] + (v - x_min) / (x_max - x_min) * cw

    def sy(v):
        return margin["top"] + ch - (v - y_min) / (y_max - y_min) * ch

    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_vals, y_vals))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font-family:monospace;font-size:11px;">',
        # Background
        f'<rect width="{width}" height="{height}" fill="#1e1e2e" rx="4"/>',
        # Axes
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" '
        f'y2="{margin["top"] + ch}" stroke="#555" />',
        f'<line x1="{margin["left"]}" y1="{margin["top"] + ch}" '
        f'x2="{margin["left"] + cw}" y2="{margin["top"] + ch}" stroke="#555" />',
        # Data line
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" />',
        # Labels
        f'<text x="{width // 2}" y="{height - 5}" fill="#aaa" text-anchor="middle">Iteration</text>',
        f'<text x="12" y="{height // 2}" fill="#aaa" text-anchor="middle" '
        f'transform="rotate(-90, 12, {height // 2})">{label}</text>',
        # Y axis ticks
        f'<text x="{margin["left"] - 5}" y="{sy(y_min) + 4}" fill="#888" '
        f'text-anchor="end">{y_min:.3f}</text>',
        f'<text x="{margin["left"] - 5}" y="{sy(y_max) + 4}" fill="#888" '
        f'text-anchor="end">{y_max:.3f}</text>',
        # X axis ticks
        f'<text x="{sx(x_min)}" y="{margin["top"] + ch + 15}" fill="#888" '
        f'text-anchor="middle">{int(x_min)}</text>',
        f'<text x="{sx(x_max)}" y="{margin["top"] + ch + 15}" fill="#888" '
        f'text-anchor="middle">{int(x_max)}</text>',
        "</svg>",
    ]
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_quality_report(
    session_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained HTML quality report for a pipeline session.

    Parameters
    ----------
    session_dir : path to the session output directory (data/output/{session})
    output_path : where to write report.html (defaults to session_dir/report.html)

    Returns
    -------
    Path to the generated report file.
    """
    session_dir = Path(session_dir)
    if output_path is None:
        output_path = session_dir / "report.html"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Infer session ID from directory name
    session_id = session_dir.name

    # Locate metrics file
    metrics_path = session_dir / "metrics.jsonl"

    # Parse stage metrics
    stage_metrics: dict[int, dict] = {}
    stage_timings_raw: dict[int, float] = {}
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                sn = entry["stage"]
                stage_metrics[sn] = entry.get("metrics", {})
                stage_timings_raw[sn] = entry.get("elapsed_s", 0.0)
            except (json.JSONDecodeError, KeyError):
                continue

    timing = compute_stage_timing(metrics_path)
    total_time = timing["total_seconds"]

    # Frame stats
    s1 = stage_metrics.get(1, {})
    s3 = stage_metrics.get(3, {})
    s6 = stage_metrics.get(6, {})
    s7 = stage_metrics.get(7, {})
    s9 = stage_metrics.get(9, {})
    s12 = stage_metrics.get(12, {})
    s13 = stage_metrics.get(13, {})

    frames_extracted = s1.get("frames_extracted", "N/A")
    frames_filtered = s3.get("frames_selected", "N/A")
    recon_method = s6.get("method", "unknown")
    registered = s6.get("registered_images", s6.get("poses", "N/A"))
    sparse_points = s6.get("sparse_points", s6.get("points", "N/A"))
    depth_maps = s7.get("depth_maps", "N/A")
    faces_detected = s9.get("faces_detected", "N/A")
    init_gaussians = s12.get("initial_gaussians", "N/A")
    final_psnr = s13.get("final_psnr", "N/A")

    # Mesh stats
    mesh_stats = _get_mesh_stats(session_dir)

    # Output manifest
    manifest = _collect_output_manifest(session_dir)

    # Training curves
    curves = _load_training_curves(session_dir)

    # Comparison grid image
    comparison_b64 = ""
    comparison_path = session_dir / "previews" / "comparison_grid.png"
    if comparison_path.exists():
        comparison_b64 = _img_to_base64(comparison_path, max_width=900)

    # Build timing chart
    timing_chart = _svg_bar_chart(timing["stages"])

    # Build training curve charts
    loss_chart = ""
    psnr_chart = ""
    if curves["iterations"] and curves["loss"]:
        loss_chart = _svg_line_chart(
            curves["iterations"], curves["loss"], "Loss", color="#ff6b6b"
        )
    if curves["iterations"] and curves["psnr"]:
        psnr_chart = _svg_line_chart(
            curves["iterations"], curves["psnr"], "PSNR (dB)", color="#6bffb8"
        )

    # Build manifest table rows
    manifest_rows = ""
    for item in manifest[:50]:  # Cap at 50 entries
        manifest_rows += (
            f'<tr><td>{item["path"]}</td><td style="text-align:right">{item["size_human"]}</td></tr>\n'
        )

    # Render HTML
    html = _REPORT_TEMPLATE.format(
        session_id=session_id,
        total_time=f"{total_time:.1f}s ({total_time / 60:.1f}min)",
        bottleneck=timing["bottleneck"],
        bottleneck_time=f"{timing['bottleneck_seconds']:.1f}s",
        frames_extracted=frames_extracted,
        frames_filtered=frames_filtered,
        registered=registered,
        recon_method=recon_method.upper(),
        sparse_points=sparse_points,
        depth_maps=depth_maps,
        faces_detected=faces_detected,
        init_gaussians=init_gaussians,
        final_psnr=final_psnr,
        mesh_vertices=mesh_stats.get("vertices", "N/A"),
        mesh_triangles=mesh_stats.get("triangles", "N/A"),
        mesh_size=mesh_stats.get("file_size", "N/A"),
        timing_chart=timing_chart,
        loss_chart=loss_chart if loss_chart else "<p>No training loss data available.</p>",
        psnr_chart=psnr_chart if psnr_chart else "<p>No PSNR data available.</p>",
        comparison_img=(
            f'<img src="{comparison_b64}" style="max-width:100%;border-radius:6px;" />'
            if comparison_b64 else "<p>No comparison grid available.</p>"
        ),
        manifest_rows=manifest_rows if manifest_rows else "<tr><td colspan='2'>No output files found.</td></tr>",
    )

    output_path.write_text(html, encoding="utf-8")
    log.info("Quality report saved to %s", output_path)
    return output_path


def _get_mesh_stats(out_dir: Path) -> dict[str, Any]:
    """Extract mesh vertex/triangle counts and file size."""
    stats: dict[str, Any] = {}

    # Try mesh.obj
    mesh_obj = out_dir / "mesh.obj"
    mesh_ply = out_dir / "mesh" / "mesh.ply"
    mesh_path = mesh_ply if mesh_ply.exists() else (mesh_obj if mesh_obj.exists() else None)

    if mesh_path is None:
        return stats

    stats["file_size"] = _format_size(mesh_path.stat().st_size)

    try:
        if mesh_path.suffix == ".obj":
            verts = 0
            faces = 0
            with open(mesh_path) as f:
                for line in f:
                    if line.startswith("v "):
                        verts += 1
                    elif line.startswith("f "):
                        faces += 1
            stats["vertices"] = verts
            stats["triangles"] = faces
        elif mesh_path.suffix == ".ply":
            with open(mesh_path, "rb") as f:
                header = b""
                while True:
                    line = f.readline()
                    header += line
                    if b"end_header" in line:
                        break
            header_text = header.decode("ascii", errors="replace")
            for line in header_text.splitlines():
                if line.startswith("element vertex"):
                    stats["vertices"] = int(line.split()[-1])
                elif line.startswith("element face"):
                    stats["triangles"] = int(line.split()[-1])
    except Exception as e:
        log.debug("Could not parse mesh stats: %s", e)

    return stats


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face3D Quality Report - {session_id}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117; color: #e6edf3; line-height: 1.6;
    padding: 2rem; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ color: #58a6ff; margin-bottom: 0.5rem; font-size: 1.8rem; }}
  h2 {{
    color: #79c0ff; margin: 2rem 0 1rem; font-size: 1.3rem;
    border-bottom: 1px solid #21262d; padding-bottom: 0.4rem;
  }}
  .subtitle {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
  .card {{
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 1rem; text-align: center;
  }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; color: #58a6ff; }}
  .card .label {{ font-size: 0.85rem; color: #8b949e; margin-top: 0.3rem; }}
  .chart-container {{ background: #161b22; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
  table {{
    width: 100%; border-collapse: collapse; background: #161b22;
    border-radius: 8px; overflow: hidden;
  }}
  th {{ background: #21262d; color: #79c0ff; padding: 0.6rem 1rem; text-align: left; font-weight: 600; }}
  td {{ padding: 0.5rem 1rem; border-top: 1px solid #21262d; font-family: monospace; font-size: 0.85rem; }}
  tr:hover td {{ background: #1c2333; }}
  .charts-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
  .charts-row > div {{ flex: 1; min-width: 300px; }}
  .tag {{
    display: inline-block; background: #1f6feb33; color: #58a6ff;
    padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.8rem; margin-right: 0.3rem;
  }}
  .bottleneck {{ color: #ff6b6b; }}
  footer {{ margin-top: 3rem; color: #484f58; font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>

<h1>Face3D Quality Report</h1>
<p class="subtitle">Session: <strong>{session_id}</strong> &middot; Total time: {total_time}
 &middot; Bottleneck: <span class="bottleneck">{bottleneck}</span> ({bottleneck_time})</p>

<h2>Pipeline Summary</h2>
<div class="grid">
  <div class="card"><div class="value">{frames_extracted}</div><div class="label">Frames Extracted</div></div>
  <div class="card"><div class="value">{frames_filtered}</div><div class="label">Frames Filtered</div></div>
  <div class="card"><div class="value">{registered}</div><div class="label">Registered ({recon_method})</div></div>
  <div class="card"><div class="value">{sparse_points}</div><div class="label">Sparse Points</div></div>
  <div class="card"><div class="value">{depth_maps}</div><div class="label">Depth Maps</div></div>
  <div class="card"><div class="value">{faces_detected}</div><div class="label">Faces Detected</div></div>
  <div class="card"><div class="value">{init_gaussians}</div><div class="label">Initial Gaussians</div></div>
  <div class="card"><div class="value">{final_psnr}</div><div class="label">Final PSNR</div></div>
</div>

<h2>Stage Timing</h2>
<div class="chart-container">
{timing_chart}
</div>

<h2>Training Curves</h2>
<div class="charts-row">
  <div class="chart-container">{loss_chart}</div>
  <div class="chart-container">{psnr_chart}</div>
</div>

<h2>Mesh Statistics</h2>
<div class="grid">
  <div class="card"><div class="value">{mesh_vertices}</div><div class="label">Vertices</div></div>
  <div class="card"><div class="value">{mesh_triangles}</div><div class="label">Triangles</div></div>
  <div class="card"><div class="value">{mesh_size}</div><div class="label">File Size</div></div>
</div>

<h2>Comparison: Ground Truth vs Rendered</h2>
<div class="chart-container">
{comparison_img}
</div>

<h2>Output Files</h2>
<table>
  <thead><tr><th>File</th><th style="text-align:right">Size</th></tr></thead>
  <tbody>
{manifest_rows}
  </tbody>
</table>

<footer>
  Generated by Face3D Quality Report &middot; face3d/src/utils/quality_report.py
</footer>

</body>
</html>"""
