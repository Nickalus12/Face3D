"""Monocular depth estimation using Depth Anything V2 or V3.

DA3 is preferred when available — it provides depth, confidence, and camera
poses (extrinsics/intrinsics) in a single forward pass, potentially replacing
COLMAP for pose estimation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Depth Anything V2 model configs
_DA2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

# Max input dimension per DA2 model size (constrained by 16GB VRAM)
_DA2_MAX_INPUT = {"vits": 768, "vitb": 672, "vitl": 518, "vitg": 448}

# DA3 model presets
_DA3_MODELS = {
    "da3-small": "depth-anything/DA3-Small",
    "da3-base": "depth-anything/DA3-Base",
    "da3-large": "depth-anything/DA3-Large",
    "da3-giant": "depth-anything/DA3-Giant",
    "da3metric-large": "depth-anything/DA3Metric-Large",
    "da3mono-large": "depth-anything/DA3Mono-Large",
    "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
}


@dataclass
class DepthResult:
    """Result from depth estimation, supporting both DA2 and DA3 outputs."""
    depth: np.ndarray               # (H, W) float32 depth map
    confidence: Optional[np.ndarray] = None  # (H, W) float32 confidence map (DA3 only)
    extrinsics: Optional[np.ndarray] = None  # (3, 4) float32 world-to-camera (DA3 only)
    intrinsics: Optional[np.ndarray] = None  # (3, 3) float32 camera intrinsics (DA3 only)


class DepthEstimator:
    """Monocular depth estimation with Depth Anything V2 or V3.

    DA3 provides richer output (depth + confidence + camera poses) and is
    recommended when available. Falls back to DA2 if DA3 is not installed.
    """

    def __init__(
        self,
        model_name: str = "auto",
        model_size: str = "vitl",
        device: str = "cuda",
    ):
        """Initialize depth estimator.

        Args:
            model_name: "da3" for Depth Anything 3, "da2" for V2, or "auto"
                        to prefer DA3 and fall back to DA2.
            model_size: For DA2: "vits", "vitb", "vitl", "vitg".
                        For DA3: "da3-large", "da3-giant", etc. Defaults to
                        "da3-large" when model_name is "da3" or "auto".
            device: Torch device.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")

        self._backend = None  # "da2" or "da3"
        self._model = None
        self._model_size = model_size

        if model_name == "auto":
            try:
                self._init_da3(model_size)
            except Exception as e:
                logger.info("DA3 not available (%s), falling back to DA2", e)
                self._init_da2(model_size)
        elif model_name in ("da3", "depth_anything_v3", "depth_anything_3"):
            self._init_da3(model_size)
        else:
            self._init_da2(model_size)

    def _init_da3(self, model_size: str) -> None:
        """Initialize Depth Anything 3."""
        from depth_anything_3.api import DepthAnything3

        # Map simple size names to DA3 presets
        size_map = {"vitl": "da3-large", "vitb": "da3-base", "vits": "da3-small", "vitg": "da3-giant"}
        da3_name = size_map.get(model_size, model_size)
        hf_id = _DA3_MODELS.get(da3_name, da3_name)

        logger.info("Loading Depth Anything 3 (%s) on %s", hf_id, self.device)
        self._model = DepthAnything3.from_pretrained(hf_id)
        self._model = self._model.to(device=self.device)
        self._backend = "da3"
        logger.info("Depth Anything 3 loaded successfully")

    def _init_da2(self, model_size: str) -> None:
        """Initialize Depth Anything V2."""
        if model_size not in _DA2_CONFIGS:
            raise ValueError(f"DA2 model_size must be one of {list(_DA2_CONFIGS.keys())}")

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            import sys
            da2_path = Path(__file__).resolve().parent.parent.parent / "Models" / "depth_anything_v2"
            if da2_path.exists() and str(da2_path) not in sys.path:
                sys.path.insert(0, str(da2_path))
                from depth_anything_v2.dpt import DepthAnythingV2
            else:
                raise ImportError(
                    "depth_anything_v2 not found. Clone it with:\n"
                    "  git clone https://github.com/DepthAnything/Depth-Anything-V2.git Models/depth_anything_v2"
                )

        logger.info("Loading Depth Anything V2 (%s) on %s", model_size, self.device)
        config = _DA2_CONFIGS[model_size]
        model = DepthAnythingV2(**config)

        # Search for checkpoint
        ckpt_name = f"depth_anything_v2_{model_size}.pth"
        project_root = Path(__file__).resolve().parent.parent.parent
        search_paths = [
            project_root / "Models" / "depth_anything_v2" / "checkpoints" / ckpt_name,
            project_root / "checkpoints" / ckpt_name,
            Path("checkpoints") / ckpt_name,
            Path.home() / ".cache" / "depth_anything_v2" / ckpt_name,
        ]

        loaded = False
        for ckpt_path in search_paths:
            if ckpt_path.exists():
                logger.info("Loading checkpoint from %s", ckpt_path)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                loaded = True
                break

        if not loaded:
            try:
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(
                    repo_id=f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}",
                    filename=ckpt_name,
                    cache_dir=Path.home() / ".cache" / "depth_anything_v2",
                )
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            except Exception as e:
                raise RuntimeError(f"Could not load DA2 checkpoint '{ckpt_name}': {e}")

        self._model = model.to(self.device).eval()
        self._backend = "da2"
        self._input_size = _DA2_MAX_INPUT[model_size]
        logger.info("Depth Anything V2 (%s) loaded successfully", model_size)

    @property
    def backend(self) -> str:
        """Return which backend is active: 'da2' or 'da3'."""
        return self._backend

    @property
    def provides_poses(self) -> bool:
        """Whether this estimator also provides camera poses (DA3 only)."""
        return self._backend == "da3"

    def estimate(self, image_path: Union[str, Path]) -> DepthResult:
        """Run depth estimation on a single image.

        Returns:
            DepthResult with depth map and (for DA3) confidence + camera poses.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self._backend == "da3":
            return self._estimate_da3([str(image_path)], single=True)
        else:
            return self._estimate_da2(image_path)

    def estimate_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        batch_size: int = 4,
    ) -> List[Path]:
        """Process multiple images, save depth maps as .npy files.

        For DA3, also saves confidence maps and camera poses.
        Skips already-processed files.
        """
        from tqdm import tqdm

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [Path(p) for p in image_paths]
        output_paths = []

        to_process = []
        for img_path in image_paths:
            out_path = output_dir / f"{img_path.stem}.npy"
            output_paths.append(out_path)
            if not out_path.exists():
                to_process.append((img_path, out_path))

        if not to_process:
            logger.info("All %d depth maps already cached", len(image_paths))
            return output_paths

        logger.info("Processing %d images (%d cached) via %s",
                     len(to_process), len(image_paths) - len(to_process), self._backend.upper())

        if self._backend == "da3":
            # DA3 handles batches natively and provides multi-view consistency
            self._batch_da3(to_process, output_dir)
        else:
            # DA2: process one at a time in batches for VRAM management
            for batch_start in tqdm(
                range(0, len(to_process), batch_size),
                desc="Depth estimation (DA2)",
                total=(len(to_process) + batch_size - 1) // batch_size,
            ):
                batch = to_process[batch_start : batch_start + batch_size]
                for img_path, out_path in batch:
                    try:
                        result = self._estimate_da2(img_path)
                        np.save(str(out_path), result.depth)
                    except Exception as e:
                        logger.error("Failed to process %s: %s", img_path.name, e)

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        logger.info("Depth estimation complete: %d maps saved", len(to_process))
        return output_paths

    def _estimate_da2(self, image_path: Path) -> DepthResult:
        raw_image = cv2.imread(str(image_path))
        if raw_image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        with torch.no_grad():
            depth = self._model.infer_image(raw_image, input_size=self._input_size)

        return DepthResult(depth=depth.astype(np.float32))

    def _estimate_da3(self, image_paths: list[str], single: bool = False) -> DepthResult:
        prediction = self._model.inference(image_paths)

        idx = 0 if single else slice(None)
        result = DepthResult(
            depth=prediction.depth[idx].astype(np.float32),
            confidence=prediction.conf[idx].astype(np.float32) if prediction.conf is not None else None,
            extrinsics=prediction.extrinsics[idx].astype(np.float32) if prediction.extrinsics is not None else None,
            intrinsics=prediction.intrinsics[idx].astype(np.float32) if prediction.intrinsics is not None else None,
        )
        return result

    def _batch_da3(self, to_process: list, output_dir: Path) -> None:
        """Process a batch of images with DA3, saving depth + metadata."""
        from tqdm import tqdm

        # DA3 can process multiple images at once for multi-view consistency
        # Process in chunks to manage VRAM
        chunk_size = 8  # DA3 handles its own batching
        all_img_paths = [str(p[0]) for p in to_process]
        all_out_paths = [p[1] for p in to_process]

        for start in tqdm(range(0, len(all_img_paths), chunk_size), desc="Depth estimation (DA3)"):
            chunk_imgs = all_img_paths[start:start + chunk_size]
            chunk_outs = all_out_paths[start:start + chunk_size]

            try:
                prediction = self._model.inference(chunk_imgs)

                for i, out_path in enumerate(chunk_outs):
                    np.save(str(out_path), prediction.depth[i].astype(np.float32))

                    # Save confidence
                    if prediction.conf is not None:
                        conf_path = output_dir / f"{out_path.stem}_conf.npy"
                        np.save(str(conf_path), prediction.conf[i].astype(np.float32))

                    # Save camera poses
                    if prediction.extrinsics is not None:
                        pose_path = output_dir / f"{out_path.stem}_pose.npz"
                        np.savez(
                            str(pose_path),
                            extrinsics=prediction.extrinsics[i].astype(np.float32),
                            intrinsics=prediction.intrinsics[i].astype(np.float32) if prediction.intrinsics is not None else None,
                        )

            except Exception as e:
                logger.error("DA3 batch failed at chunk %d: %s", start, e)
                # Fall back to per-image processing
                for img_path, out_path in zip(chunk_imgs, chunk_outs):
                    try:
                        result = self._estimate_da3([img_path], single=True)
                        np.save(str(out_path), result.depth)
                    except Exception as e2:
                        logger.error("Failed to process %s: %s", img_path, e2)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def estimate_depth(
    image_path: Union[str, Path],
    model_name: str = "auto",
    model_size: str = "vitl",
    device: str = "cuda",
) -> DepthResult:
    """Convenience function: estimate depth for a single image.

    Uses DA3 if available, otherwise DA2.
    """
    estimator = DepthEstimator(model_name=model_name, model_size=model_size, device=device)
    return estimator.estimate(image_path)
