"""Visualization utilities for the Face3D pipeline.

Provides depth map coloring, landmark overlays, mask overlays, and turntable
GIF creation. All functions save directly to disk and use only matplotlib,
PIL, cv2, and numpy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
from PIL import Image as PILImage

log = logging.getLogger("face3d.visualization")


# ---------------------------------------------------------------------------
# Landmark region color mapping (MediaPipe 478-point face mesh)
# ---------------------------------------------------------------------------

# Approximate index ranges for MediaPipe face mesh regions.
# These are simplified groupings; exact indices are from the MediaPipe spec.
_LANDMARK_REGIONS = {
    "eyes": list(range(33, 42)) + list(range(133, 174)) + list(range(246, 260))
           + list(range(362, 399)) + list(range(466, 478)),
    "nose": list(range(1, 20)) + list(range(45, 65)) + list(range(275, 296)),
    "mouth": list(range(61, 96)) + list(range(146, 156)) + list(range(178, 200))
            + list(range(291, 320)) + list(range(375, 405)),
    "contour": list(range(10, 33)) + list(range(93, 133)) + list(range(234, 246))
              + list(range(323, 362)) + list(range(454, 466)),
}

_REGION_COLORS = {
    "eyes": (66, 133, 244),      # blue
    "nose": (52, 168, 83),       # green
    "mouth": (234, 67, 53),      # red
    "contour": (160, 160, 160),  # gray
}

# Precompute index-to-color mapping
_INDEX_TO_COLOR: dict[int, tuple[int, int, int]] = {}
for _region, _indices in _LANDMARK_REGIONS.items():
    for _idx in _indices:
        _INDEX_TO_COLOR[_idx] = _REGION_COLORS[_region]

_DEFAULT_COLOR = (200, 200, 200)  # for landmarks not in any region


# ---------------------------------------------------------------------------
# Depth map visualization
# ---------------------------------------------------------------------------

def visualize_depth_map(
    depth: np.ndarray | str | Path,
    output_path: str | Path,
    colormap: str = "turbo",
) -> Path:
    """Normalize a depth map to [0, 1] and save as a colormapped PNG.

    Parameters
    ----------
    depth : 2D numpy array or path to a .npy file
    output_path : where to save the PNG
    colormap : matplotlib colormap name (default: turbo)

    Returns
    -------
    Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(depth, (str, Path)):
        depth = np.load(str(depth))

    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth_norm = (depth - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float64)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(depth_norm.astype(np.float64))  # RGBA float [0,1]
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    # Save
    img = PILImage.fromarray(colored_rgb)
    img.save(str(output_path))
    log.info("Saved depth visualization to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Landmark visualization
# ---------------------------------------------------------------------------

def visualize_landmarks(
    image: np.ndarray | str | Path,
    landmarks_2d: np.ndarray | list | str | Path,
    output_path: str | Path,
) -> Path:
    """Draw 478 MediaPipe landmarks on an image, color-coded by region.

    Parameters
    ----------
    image : BGR image array, or path to image file
    landmarks_2d : (N, 2) array of pixel coordinates, or path to landmarks JSON
    output_path : where to save the annotated PNG

    Returns
    -------
    Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image}")

    vis = image.copy()

    # Load landmarks
    if isinstance(landmarks_2d, (str, Path)):
        import json
        lm_path = Path(landmarks_2d)
        with open(lm_path) as f:
            data = json.load(f)
        # Handle both list-of-lists and dict with "landmarks" key
        if isinstance(data, dict):
            lm_list = data.get("landmarks", data.get("landmarks_2d", []))
        else:
            lm_list = data
        landmarks_2d = np.array(lm_list, dtype=np.float32)

    if isinstance(landmarks_2d, list):
        landmarks_2d = np.array(landmarks_2d, dtype=np.float32)

    if landmarks_2d.ndim != 2 or landmarks_2d.shape[1] < 2:
        log.warning("Invalid landmarks shape: %s", landmarks_2d.shape)
        cv2.imwrite(str(output_path), vis)
        return output_path

    # If landmarks are normalized [0, 1], scale to image dimensions
    h, w = vis.shape[:2]
    if landmarks_2d.max() <= 1.0:
        landmarks_2d[:, 0] *= w
        landmarks_2d[:, 1] *= h

    # Draw each landmark
    for i, (x, y) in enumerate(landmarks_2d[:, :2]):
        color = _INDEX_TO_COLOR.get(i, _DEFAULT_COLOR)
        # BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        cv2.circle(vis, (int(x), int(y)), 1, color_bgr, -1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis)
    log.info("Saved landmark visualization to %s (%d landmarks)", output_path, len(landmarks_2d))
    return output_path


# ---------------------------------------------------------------------------
# Mask overlay
# ---------------------------------------------------------------------------

def visualize_mask_overlay(
    image: np.ndarray | str | Path,
    mask: np.ndarray | str | Path,
    output_path: str | Path,
    alpha: float = 0.5,
) -> Path:
    """Overlay a segmentation mask on an image.

    Face region is tinted green, background is darkened.

    Parameters
    ----------
    image : BGR image array, or path to image file
    mask : binary mask (H, W) with face=255/1, background=0, or path to mask file
    output_path : where to save the overlay PNG
    alpha : transparency of the overlay (0=transparent, 1=opaque)

    Returns
    -------
    Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image}")

    # Load mask
    if isinstance(mask, (str, Path)):
        mask_path = Path(mask)
        if mask_path.suffix == ".npy":
            mask = np.load(str(mask_path))
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    # Ensure mask is binary (0 or 1)
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Resize mask to match image if needed
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create overlay
    overlay = image.copy()

    # Darken background
    bg_mask = mask == 0
    overlay[bg_mask] = (overlay[bg_mask] * 0.3).astype(np.uint8)

    # Green tint on face region
    face_mask = mask == 1
    green_tint = overlay.copy()
    green_tint[face_mask, 1] = np.clip(
        green_tint[face_mask, 1].astype(np.int16) + 60, 0, 255
    ).astype(np.uint8)

    # Blend
    result = cv2.addWeighted(image, 1.0 - alpha, green_tint, alpha, 0)
    # Keep background darkened
    result[bg_mask] = overlay[bg_mask]

    cv2.imwrite(str(output_path), result)
    log.info("Saved mask overlay to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Turntable GIF
# ---------------------------------------------------------------------------

def create_turntable_gif(
    renders_dir: str | Path,
    output_path: str | Path,
    fps: int = 15,
    max_size: int = 512,
) -> Path:
    """Create an animated GIF from turntable render frames.

    Parameters
    ----------
    renders_dir : directory containing frame_*.png render outputs
    output_path : where to save the animated GIF
    fps : frames per second for the GIF
    max_size : maximum dimension (width or height), preserving aspect ratio

    Returns
    -------
    Path to the saved GIF.
    """
    renders_dir = Path(renders_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather render frames in sorted order
    frames = sorted(renders_dir.glob("frame_*.png"))
    if not frames:
        # Fallback: try any PNG
        frames = sorted(renders_dir.glob("*.png"))
    if not frames:
        log.warning("No render frames found in %s", renders_dir)
        return output_path

    pil_frames = []
    for fp in frames:
        img = PILImage.open(fp).convert("RGB")
        # Resize preserving aspect ratio
        w, h = img.size
        scale = max_size / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        pil_frames.append(img)

    if not pil_frames:
        log.warning("No frames loaded for GIF")
        return output_path

    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    log.info("Saved turntable GIF to %s (%d frames, %d fps)", output_path, len(pil_frames), fps)
    return output_path
