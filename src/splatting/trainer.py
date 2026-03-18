"""Gaussian Splatting training loop using gsplat's built-in strategies.

Supports both standard 3DGS and 2DGS (2D Gaussian Splatting) rasterization
with face-optimized losses including normal consistency, distortion
regularization, LPIPS perceptual loss, depth supervision, and opacity entropy.

Optimizations (research-backed):
    - packed=False: 25-30% faster per iteration
    - sh_degree_max=1: 15-25% faster for face scenes with controlled lighting
    - radius_clip/near_plane/far_plane: skip out-of-range Gaussians
    - cudnn.benchmark: faster conv operations (SSIM etc.)
    - Reduced densification frequency (refine_every=500)
    - LPIPS and appearance embedding disabled by default (expensive)
    - 3-stage progressive resolution (1/4 -> 1/2 -> full)
    - Staged loss introduction (L1+depth -> +DSSIM -> +normal+distortion)
    - Depth loss with decay
    - LR warmup for position learning rate
    - 3000 iterations default (sufficient with good initialization)
    - SelectiveAdam: visibility-aware optimizer, updates only visible Gaussians (20-40% faster)
    - MCMCStrategy: fixed Gaussian count via MCMC sampling (consistent speed, no growth)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from splatting.camera_utils import Camera
from splatting.initializer import GaussianModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSIM: prefer fused_ssim (CUDA-optimised) with manual fallback
# ---------------------------------------------------------------------------

try:
    from fused_ssim import fused_ssim

    def _ssim_fn(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM using fused CUDA kernel. Inputs: (1, C, H, W)."""
        return fused_ssim(img1, img2)

    _SSIM_BACKEND = "fused_ssim"
except ImportError:

    def _ssim_fn(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Manual SSIM computation. Inputs: (1, C, H, W)."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        kernel_size = 11
        sigma = 1.5
        coords = torch.arange(kernel_size, dtype=torch.float32, device=img1.device) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)  # (11, 11)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 11, 11)
        kernel = kernel.expand(img1.shape[1], -1, -1, -1)  # (C, 1, 11, 11)

        pad = kernel_size // 2
        mu_x = F.conv2d(img1, kernel, padding=pad, groups=img1.shape[1])
        mu_y = F.conv2d(img2, kernel, padding=pad, groups=img2.shape[1])

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=img1.shape[1]) - mu_x_sq
        sigma_y_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=img2.shape[1]) - mu_y_sq
        sigma_xy = F.conv2d(img1 * img2, kernel, padding=pad, groups=img2.shape[1]) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )
        return ssim_map.mean()

    _SSIM_BACKEND = "manual"


logger.info("SSIM backend: %s", _SSIM_BACKEND)

# ---------------------------------------------------------------------------
# 2DGS rasterization: try rasterization_2dgs, fall back to standard
# ---------------------------------------------------------------------------

_USE_2DGS_RASTERIZATION = False
try:
    from gsplat.rendering import rasterization_2dgs as _rasterization_2dgs_fn
    _USE_2DGS_RASTERIZATION = True
    logger.info("2DGS rasterization available (gsplat.rendering.rasterization_2dgs)")
except ImportError:
    _rasterization_2dgs_fn = None
    logger.warning(
        "rasterization_2dgs unavailable (gsplat version mismatch) — "
        "falling back to standard rasterization with rasterize_mode='antialiased'"
    )

# ---------------------------------------------------------------------------
# SelectiveAdam: visibility-aware optimizer (Taming3DGS)
# Only updates parameters for Gaussians visible in the current view.
# Falls back to standard Adam if unavailable (gsplat < 1.4.0).
# ---------------------------------------------------------------------------

_SELECTIVE_ADAM_AVAILABLE = False
try:
    from gsplat.optimizers import SelectiveAdam
    _SELECTIVE_ADAM_AVAILABLE = True
    logger.info("SelectiveAdam available (gsplat.optimizers.SelectiveAdam)")
except ImportError:
    SelectiveAdam = None
    logger.info("SelectiveAdam unavailable — will use standard Adam optimizers")

# ---------------------------------------------------------------------------
# LPIPS: lazy-loaded singleton to avoid import cost until needed
# ---------------------------------------------------------------------------

_lpips_net = None
_lpips_available = True  # Set to False if initialization fails


def _get_lpips_net(device: torch.device):
    """Return a cached LPIPS VGG network (no gradient, eval mode).

    Returns None if LPIPS is unavailable (download failure or OOM).
    """
    global _lpips_net, _lpips_available
    if not _lpips_available:
        return None
    if _lpips_net is None:
        try:
            import lpips
            _lpips_net = lpips.LPIPS(net="vgg").eval().to(device)
            for p in _lpips_net.parameters():
                p.requires_grad_(False)
        except Exception as e:
            logger.warning("LPIPS unavailable, disabling perceptual loss: %s", e)
            _lpips_available = False
            return None
    return _lpips_net


# ---------------------------------------------------------------------------
# Appearance MLP for per-frame colour correction
# ---------------------------------------------------------------------------

class AppearanceMLP(nn.Module):
    """Small MLP that corrects rendered RGB using a per-camera embedding.

    Takes concatenated (RGB + embedding) and outputs a corrected RGB image.
    Operates per-pixel on flattened spatial dims for memory efficiency.
    """

    def __init__(self, appearance_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + appearance_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, rgb: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (H, W, 3)
            embedding: (appearance_dim,)
        Returns:
            corrected_rgb: (H, W, 3)
        """
        H, W, _ = rgb.shape
        # Broadcast embedding to every pixel
        emb = embedding.unsqueeze(0).unsqueeze(0).expand(H, W, -1)  # (H, W, D)
        x = torch.cat([rgb, emb], dim=-1)  # (H, W, 3+D)
        # Flatten, run MLP, reshape
        correction = self.net(x.reshape(-1, x.shape[-1])).reshape(H, W, 3)
        # Residual correction — add to original and clamp
        return (rgb + correction).clamp(0.0, 1.0)


@dataclass
class TrainingConfig:
    """Configuration for Gaussian Splatting training."""

    # Iterations — 3K is sufficient with good initialization and progressive training
    iterations: int = 3_000

    # Strategy selection: "default" uses clone/split/prune, "mcmc" uses MCMC sampling,
    # "auto" selects based on scene type (currently maps to "default")
    strategy: Literal["default", "mcmc", "auto"] = "default"

    # SelectiveAdam: visibility-aware optimizer that only updates visible Gaussians.
    # 20-40% faster optimizer step. Falls back to standard Adam if unavailable.
    use_selective_adam: bool = True

    # Learning rates (per-parameter)
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_means_delay_mult: float = 0.01
    lr_means_max_steps: int = 3_000  # Match iterations for compressed LR schedule
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20

    # LR warmup — linear warmup for position LR over first N iterations
    lr_warmup_iters: int = 100

    # Loss weights
    lambda_dssim: float = 0.2
    depth_weight: float = 0.0
    mask_weight: float = 0.0

    # --- 2DGS-specific settings ---
    use_2dgs: bool = True
    lambda_normal: float = 0.05       # Normal consistency weight
    lambda_distort: float = 0.01      # Distortion regularization weight
    lambda_lpips: float = 0.0         # LPIPS disabled by default — VGG forward pass is expensive
    lambda_depth: float = 0.1         # Depth supervision weight (2DGS median depth)
    lambda_depth_initial: float = 0.1 # Depth weight at start (decays to lambda_depth_final)
    lambda_depth_final: float = 0.01  # Depth weight at 60% of training
    lambda_opacity_entropy: float = 0.001  # Opacity entropy regularization weight

    # Appearance embedding (per-frame exposure/white-balance correction)
    # Disabled by default — small MLP forward+backward is ~5% overhead
    use_appearance_embedding: bool = False
    appearance_dim: int = 32

    # Camera pose refinement (experimental, off by default)
    use_camera_opt: bool = False

    # Progressive training: 3-stage progressive resolution
    #   Stage 1 (0-15%):  1/4 resolution
    #   Stage 2 (15-40%): 1/2 resolution
    #   Stage 3 (40%+):   full resolution
    progressive_training: bool = True
    progressive_stages: int = 3  # 2 = legacy half/full, 3 = quarter/half/full

    # Staged loss introduction (only active when progressive_stages=3):
    #   Stage 1 (0-15%):  L1 + depth supervision only
    #   Stage 2 (15-40%): + D-SSIM
    #   Stage 3 (40%+):   + normal consistency + distortion (full loss)
    staged_loss: bool = True

    # DefaultStrategy parameters — tuned for RTX 3080 16GB
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0005  # Higher threshold = fewer new Gaussians
    grow_scale3d: float = 0.01
    refine_start_iter: int = 500
    refine_stop_iter: int = 0  # 0 = auto (60% of iterations); fewer densification events
    reset_every: int = 3_000
    refine_every: int = 500  # Densify less frequently — fewer Gaussians mid-training
    absgrad: bool = True
    max_num_gaussians: int = 500_000  # Hard cap for 16GB VRAM (~8GB for 500K Gaussians)

    # MCMC-specific parameters (only used when strategy="mcmc")
    # MCMC maintains a fixed Gaussian count via teleportation + sampling — no growth/split
    mcmc_cap_max: int = 200_000  # Max Gaussians (MCMC enforces this internally)
    mcmc_noise_lr: float = 5e5
    mcmc_refine_every: int = 100
    mcmc_refine_start_iter: int = 500
    mcmc_refine_stop_iter: int = 0  # 0 = auto (80% of iterations)
    mcmc_min_opacity: float = 0.005  # Minimum opacity for teleportation

    # SH degree scheduling — SH1 (4 coefficients) is sufficient for face scenes
    # with controlled lighting. SH3 drops throughput from ~12 it/s to ~4.5 it/s.
    sh_degree_max: int = 1
    sh_increase_every: int = 1_000  # increase active SH degree every N iters

    # Mixed precision (disabled by default: gsplat strategies conflict with GradScaler)
    use_amp: bool = False

    # Memory-efficient rasterisation flags
    # packed=False is 25-30% faster per iteration
    packed: bool = False
    sparse_grad: bool = False  # Disabled: conflicts with quat normalization in autograd

    # Rasterization bounds — face scenes are bounded, skip out-of-range Gaussians
    near_plane: float = 0.1
    far_plane: float = 5.0
    radius_clip: float = 2.0  # Skip tiny Gaussians for speed

    # Checkpointing
    checkpoint_every: int = 5_000
    eval_every: int = 1_000

    # Rendering
    background_color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


def _build_splats(gaussians: GaussianModel, device: torch.device) -> torch.nn.ParameterDict:
    """Convert a GaussianModel into the ParameterDict expected by gsplat strategies.

    Keys: 'means', 'scales', 'quats', 'opacities', 'sh0', 'shN'
    """
    gaussians = gaussians.to(device)

    # Split SH into DC (sh0) and higher-order (shN)
    sh_all = gaussians.colors_sh  # (N, SH_COEFFS, 3)
    sh0 = sh_all[:, :1, :]       # (N, 1, 3)
    shN = sh_all[:, 1:, :]       # (N, SH_COEFFS-1, 3)

    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(gaussians.positions.clone()),
        "scales": torch.nn.Parameter(gaussians.scales.clone()),          # log-space
        "quats": torch.nn.Parameter(gaussians.rotations.clone()),
        "opacities": torch.nn.Parameter(gaussians.opacities.squeeze(-1).clone()),  # (N,) logit-space
        "sh0": torch.nn.Parameter(sh0.clone()),                         # (N, 1, 3)
        "shN": torch.nn.Parameter(shN.clone()),                         # (N, 15, 3)
    })
    return splats


def _splats_to_gaussians(splats: torch.nn.ParameterDict) -> GaussianModel:
    """Convert ParameterDict back to GaussianModel for checkpointing / export."""
    sh_all = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # (N, 16, 3)
    return GaussianModel(
        positions=splats["means"].detach(),
        colors_sh=sh_all.detach(),
        scales=splats["scales"].detach(),
        rotations=splats["quats"].detach(),
        opacities=splats["opacities"].detach().unsqueeze(-1),
    )


class GaussianTrainer:
    """Trains 3D Gaussian Splatting models using gsplat rasterization and strategies.

    When ``use_2dgs=True`` (default), uses 2D Gaussian Splatting with surface
    normals, distortion regularization, LPIPS perceptual loss, and opacity
    entropy for high-quality face reconstruction.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        gaussians: GaussianModel,
        cameras: list[Camera],
        images_dir: Path,
    ) -> dict | None:
        """Validate training inputs. Returns an error dict if validation fails, else None."""
        n_points = gaussians.positions.shape[0] if gaussians.positions is not None else 0
        if n_points < 10:
            msg = f"GaussianModel has only {n_points} points (need >= 10)"
            logger.error(msg)
            return {
                "error": msg,
                "iterations": 0,
                "final_loss": float("nan"),
                "final_psnr": 0.0,
                "best_psnr": 0.0,
                "num_gaussians_start": n_points,
                "num_gaussians_end": n_points,
                "peak_vram_mb": 0,
                "total_time_s": 0.0,
                "nan_count": 0,
                "oom_count": 0,
                "gradient_clips": 0,
            }

        if not cameras:
            msg = "cameras list is empty"
            logger.error(msg)
            return {
                "error": msg,
                "iterations": 0,
                "final_loss": float("nan"),
                "final_psnr": 0.0,
                "best_psnr": 0.0,
                "num_gaussians_start": n_points,
                "num_gaussians_end": n_points,
                "peak_vram_mb": 0,
                "total_time_s": 0.0,
                "nan_count": 0,
                "oom_count": 0,
                "gradient_clips": 0,
            }

        if len(cameras) < 2:
            msg = f"Need at least 2 cameras for training, got {len(cameras)}"
            logger.error(msg)
            return {
                "error": msg,
                "iterations": 0,
                "final_loss": float("nan"),
                "final_psnr": 0.0,
                "best_psnr": 0.0,
                "num_gaussians_start": n_points,
                "num_gaussians_end": n_points,
                "peak_vram_mb": 0,
                "total_time_s": 0.0,
                "nan_count": 0,
                "oom_count": 0,
                "gradient_clips": 0,
            }

        if not images_dir.exists():
            msg = f"images_dir does not exist: {images_dir}"
            logger.error(msg)
            return {
                "error": msg,
                "iterations": 0,
                "final_loss": float("nan"),
                "final_psnr": 0.0,
                "best_psnr": 0.0,
                "num_gaussians_start": n_points,
                "num_gaussians_end": n_points,
                "peak_vram_mb": 0,
                "total_time_s": 0.0,
                "nan_count": 0,
                "oom_count": 0,
                "gradient_clips": 0,
            }

        # Check images_dir has images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        has_images = any(
            f.suffix.lower() in image_extensions
            for f in images_dir.iterdir()
            if f.is_file()
        )
        if not has_images:
            msg = f"images_dir has no images (.jpg/.jpeg/.png/.bmp): {images_dir}"
            logger.error(msg)
            return {
                "error": msg,
                "iterations": 0,
                "final_loss": float("nan"),
                "final_psnr": 0.0,
                "best_psnr": 0.0,
                "num_gaussians_start": n_points,
                "num_gaussians_end": n_points,
                "peak_vram_mb": 0,
                "total_time_s": 0.0,
                "nan_count": 0,
                "oom_count": 0,
                "gradient_clips": 0,
            }

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        gaussians: GaussianModel,
        cameras: list[Camera],
        images_dir: Path,
        depth_dir: Path | None = None,
        masks_dir: Path | None = None,
        output_dir: Path | None = None,
        photo_names: set[str] | None = None,
        photo_weight: float = 3.0,
    ) -> GaussianModel | dict:
        """Run the full Gaussian Splatting training loop.

        Args:
            gaussians: Initialized GaussianModel.
            cameras: List of Camera objects with poses.
            images_dir: Directory containing training images.
            depth_dir: Optional directory with depth maps for depth supervision.
            masks_dir: Optional directory with binary masks.
            output_dir: Directory for checkpoints and evaluation outputs.
            photo_names: Optional set of image filenames that are Expert RAW
                photos (as opposed to video frames). These will be sampled
                more frequently during training according to photo_weight.
            photo_weight: Sampling weight multiplier for photo views (default 3.0).
                Video frames have weight 1.0.

        Returns:
            Trained GaussianModel, or an error dict if validation fails.
        """
        cfg = self.config
        t_start = time.time()

        # --- Enable cuDNN benchmark for faster conv operations (SSIM etc.) ---
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # --- Validate inputs ---
        validation_error = self._validate_inputs(gaussians, cameras, images_dir)
        if validation_error is not None:
            return validation_error

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # --- Check if 2DGS fallback is needed ---
        use_2dgs = cfg.use_2dgs
        if use_2dgs and not _USE_2DGS_RASTERIZATION:
            logger.warning(
                "2DGS requested but rasterization_2dgs unavailable — "
                "falling back to standard 3DGS with antialiased rasterization"
            )
            use_2dgs = False

        # --- LPIPS availability check ---
        lpips_disabled_by_fallback = False
        if cfg.lambda_lpips > 0.0:
            lpips_net = _get_lpips_net(self.device)
            if lpips_net is None:
                logger.warning("LPIPS unavailable, disabling perceptual loss")
                lpips_disabled_by_fallback = True

        # --- Build ParameterDict from GaussianModel ---
        splats = _build_splats(gaussians, self.device)
        num_gaussians_start = splats["means"].shape[0]

        # --- Estimate scene scale (used by strategies) ---
        with torch.no_grad():
            scene_scale = splats["means"].std(dim=0).max().item()
        logger.info("Estimated scene scale: %.4f", scene_scale)

        # --- Create strategy ---
        strategy, strategy_state = self._create_strategy(splats, scene_scale)

        # --- Per-parameter optimizers ---
        # When SelectiveAdam is active, each per-key optimizer is a SelectiveAdam
        # instance that accepts a visibility mask in step(). Structure is the same
        # as standard Adam (per-key dict) for gsplat strategy compatibility.
        optimizers = self._create_optimizers(splats)
        _using_selective_adam = _SELECTIVE_ADAM_AVAILABLE and cfg.use_selective_adam

        # Resolve effective strategy name for logging/branching
        _effective_strategy = cfg.strategy if cfg.strategy != "auto" else "default"

        # --- Appearance embedding ---
        appearance_mlp = None
        appearance_embeddings = None
        if cfg.use_appearance_embedding:
            num_cameras = len(cameras)
            appearance_embeddings = nn.Embedding(num_cameras, cfg.appearance_dim).to(self.device)
            appearance_mlp = AppearanceMLP(cfg.appearance_dim).to(self.device)
            optimizers["appearance_embeddings"] = torch.optim.Adam(
                appearance_embeddings.parameters(), lr=1e-3, eps=1e-15,
            )
            optimizers["appearance_mlp"] = torch.optim.Adam(
                appearance_mlp.parameters(), lr=1e-3, eps=1e-15,
            )
            logger.info(
                "Appearance embedding enabled: %d cameras x %d dims",
                num_cameras, cfg.appearance_dim,
            )

        # --- Mixed-precision ---
        use_amp = cfg.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        # --- Preload camera data ---
        cam_data = self._preload_cameras(cameras, images_dir, depth_dir, masks_dir)
        num_cameras = len(cam_data)

        # --- Build weighted sampling distribution for photo views ---
        import random as _random
        _sampling_weights = None
        if photo_names and num_cameras > 0:
            weights = []
            n_photo = 0
            for cd in cam_data:
                # Check if this camera's image_path matches a photo name
                cam_image = cameras[cd["cam_idx"]].image_path if cd["cam_idx"] < len(cameras) else None
                is_photo_view = cam_image is not None and Path(cam_image).name in photo_names
                if is_photo_view:
                    weights.append(photo_weight)
                    n_photo += 1
                else:
                    weights.append(1.0)
            if n_photo > 0:
                _sampling_weights = weights
                logger.info(
                    "Weighted view sampling: %d photo views (weight=%.1f) + %d video views (weight=1.0)",
                    n_photo, photo_weight, num_cameras - n_photo,
                )

        logger.info(
            "Training with %d cameras for %d iterations "
            "(strategy=%s, 2dgs=%s, amp=%s, selective_adam=%s)",
            num_cameras, cfg.iterations, _effective_strategy, use_2dgs, use_amp,
            _using_selective_adam,
        )

        # Active SH degree (start at 0, increase over time)
        active_sh_degree = 0

        # Background colour tensor
        bg = torch.tensor(cfg.background_color, dtype=torch.float32, device=self.device)

        # Progressive training: compute iteration boundaries for resolution stages
        # 3-stage: 0-15% quarter, 15-40% half, 40%+ full
        # 2-stage (legacy): 0-50% half, 50%+ full
        if cfg.progressive_training and cfg.progressive_stages == 3:
            _prog_stage1_end = int(cfg.iterations * 0.15)  # end of 1/4 res
            _prog_stage2_end = int(cfg.iterations * 0.40)  # end of 1/2 res
        elif cfg.progressive_training:
            _prog_stage1_end = 0                            # no 1/4 res stage
            _prog_stage2_end = cfg.iterations // 2          # end of 1/2 res
        else:
            _prog_stage1_end = 0
            _prog_stage2_end = 0
        _prog_logged_stage = 0  # track which resolution stage we've logged

        best_psnr = 0.0
        final_loss = float("nan")
        final_psnr = 0.0

        # --- Robustness tracking ---
        nan_count = 0
        oom_count = 0
        gradient_clips = 0
        consecutive_nan = 0
        last_good_state: dict | None = None  # Saved every 500 iterations
        checkpoint_paths: list[Path] = []  # Track checkpoint files for rotation
        best_checkpoint_psnr = 0.0

        # Reset peak VRAM counter
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        pbar = tqdm(range(1, cfg.iterations + 1), desc="Training Gaussians")

        for iteration in pbar:
            # --- Save last-good state every 500 iterations ---
            if iteration % 500 == 1 or iteration == 1:
                last_good_state = {
                    "splats": {k: v.detach().clone() for k, v in splats.items()},
                    "iteration": iteration,
                }

            # Pick a random training camera (weighted if photo views present)
            if _sampling_weights is not None:
                idx = _random.choices(range(num_cameras), weights=_sampling_weights, k=1)[0]
            else:
                idx = np.random.randint(0, num_cameras)
            cam_info = cam_data[idx]
            viewmat = cam_info["viewmat"]     # (4, 4)
            K = cam_info["K"]                 # (3, 3)
            gt_image = cam_info["image"]      # (H, W, 3)
            gt_depth = cam_info.get("depth")  # (H, W) or None
            mask = cam_info.get("mask")       # (H, W) or None
            w, h = cam_info["width"], cam_info["height"]
            cam_idx = cam_info.get("cam_idx", idx)

            # --- Progressive resolution (3-stage: 1/4 -> 1/2 -> full) ---
            if cfg.progressive_training and iteration <= _prog_stage1_end:
                # Stage 1: quarter resolution
                gt_image_train, gt_depth_train, mask_train, w_train, h_train, K_train = (
                    self._downsample_for_training(gt_image, gt_depth, mask, w, h, K, factor=4)
                )
                if _prog_logged_stage < 1:
                    _prog_logged_stage = 1
                    logger.info(
                        "Progressive training: 1/4 resolution until iteration %d",
                        _prog_stage1_end,
                    )
            elif cfg.progressive_training and iteration <= _prog_stage2_end:
                # Stage 2: half resolution
                gt_image_train, gt_depth_train, mask_train, w_train, h_train, K_train = (
                    self._downsample_for_training(gt_image, gt_depth, mask, w, h, K, factor=2)
                )
                if _prog_logged_stage < 2:
                    _prog_logged_stage = 2
                    logger.info(
                        "Progressive training: switching to 1/2 resolution at iteration %d",
                        iteration,
                    )
            else:
                # Stage 3: full resolution
                gt_image_train = gt_image
                gt_depth_train = gt_depth
                mask_train = mask
                w_train, h_train = w, h
                K_train = K
                if _prog_logged_stage < 3 and cfg.progressive_training:
                    _prog_logged_stage = 3
                    logger.info(
                        "Progressive training: switching to full resolution at iteration %d",
                        iteration,
                    )

            # --- Progressive SH degree ---
            if iteration % cfg.sh_increase_every == 0 and active_sh_degree < cfg.sh_degree_max:
                active_sh_degree += 1
                logger.debug("Increased active SH degree to %d at iter %d", active_sh_degree, iteration)

            # --- Learning-rate scheduling for means ---
            self._update_means_lr(optimizers, iteration)

            # --- Forward pass (with optional AMP) ---
            render_depth = (
                (gt_depth_train is not None and cfg.depth_weight > 0)
                or (gt_depth_train is not None and use_2dgs and cfg.lambda_depth > 0)
            )
            ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else _nullcontext()

            # --- OOM protection: wrap rasterization ---
            oom_retries = 0
            max_oom_retries = 3

            while True:
                try:
                    with ctx:
                        if use_2dgs:
                            rgb, alpha, normals, surf_normals, distort, median_depth, info = self._render_2dgs(
                                splats, viewmat, K_train, w_train, h_train, active_sh_degree, bg,
                            )

                            # Apply appearance correction
                            if appearance_mlp is not None and appearance_embeddings is not None:
                                emb = appearance_embeddings(
                                    torch.tensor(cam_idx, device=self.device)
                                )
                                rgb = appearance_mlp(rgb, emb)

                            loss, loss_dict = self._compute_loss_2dgs(
                                rgb, gt_image_train,
                                normals=normals,
                                surf_normals=surf_normals,
                                distort=distort,
                                median_depth=median_depth,
                                depth_gt=gt_depth_train,
                                mask=mask_train,
                                opacities=torch.sigmoid(splats["opacities"]),
                                lpips_disabled=lpips_disabled_by_fallback,
                                iteration=iteration,
                            )
                        else:
                            rgb, depth, alpha, info = self._render(
                                splats, viewmat, K_train, w_train, h_train, active_sh_degree, bg,
                                render_depth=render_depth,
                            )

                            # Apply appearance correction
                            if appearance_mlp is not None and appearance_embeddings is not None:
                                emb = appearance_embeddings(
                                    torch.tensor(cam_idx, device=self.device)
                                )
                                rgb = appearance_mlp(rgb, emb)

                            loss, loss_dict = self._compute_loss(
                                rgb, gt_image_train,
                                depth_rendered=depth,
                                depth_gt=gt_depth_train,
                                mask=mask_train,
                            )
                    # Rasterization succeeded, break out of retry loop
                    break

                except torch.cuda.OutOfMemoryError:
                    oom_retries += 1
                    oom_count += 1
                    n_gs = splats["means"].shape[0]
                    logger.warning(
                        "OOM during rasterization with %d Gaussians at resolution %dx%d "
                        "(attempt %d/%d)",
                        n_gs, w_train, h_train, oom_retries, max_oom_retries,
                    )
                    torch.cuda.empty_cache()

                    if oom_retries >= max_oom_retries:
                        raise RuntimeError(
                            f"OOM {max_oom_retries} times in same iteration {iteration} "
                            f"with {n_gs} Gaussians at {w_train}x{h_train}. "
                            f"Reduce max_num_gaussians or image resolution."
                        ) from None

                    # Prune 30% of lowest-opacity Gaussians
                    with torch.no_grad():
                        opas = torch.sigmoid(splats["opacities"])
                        keep_k = max(10, int(n_gs * 0.7))
                        _, keep_idx = torch.topk(opas, min(keep_k, n_gs))
                        keep_idx = keep_idx.sort().values
                        for key in splats:
                            splats[key] = torch.nn.Parameter(splats[key][keep_idx].clone())
                        # Rebuild optimizers
                        splat_optimizers = self._create_optimizers(splats)
                        if appearance_mlp is not None:
                            splat_optimizers["appearance_embeddings"] = optimizers["appearance_embeddings"]
                            splat_optimizers["appearance_mlp"] = optimizers["appearance_mlp"]
                        optimizers = splat_optimizers
                        # Re-init strategy state
                        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
                        logger.info(
                            "OOM recovery: pruned to %d Gaussians, retrying",
                            splats["means"].shape[0],
                        )

            # --- NaN/Inf loss detection and recovery ---
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                consecutive_nan += 1

                # Identify which loss component went NaN
                nan_components = [
                    k for k, v in loss_dict.items()
                    if torch.isnan(v) or torch.isinf(v)
                ]
                logger.warning(
                    "NaN/Inf loss at iteration %d (count=%d, consecutive=%d). "
                    "Bad components: %s",
                    iteration, nan_count, consecutive_nan,
                    nan_components if nan_components else "combined total",
                )

                if consecutive_nan >= 3:
                    logger.error(
                        "NaN loss 3 times in a row — aborting training at iteration %d",
                        iteration,
                    )
                    break

                # Restore from last good state
                if last_good_state is not None:
                    logger.info(
                        "Restoring from last good state (iteration %d)",
                        last_good_state["iteration"],
                    )
                    with torch.no_grad():
                        for k in splats:
                            splats[k].data.copy_(last_good_state["splats"][k])

                # Reduce all learning rates by 50%
                for opt in optimizers.values():
                    for pg in opt.param_groups:
                        pg["lr"] *= 0.5
                logger.info("Reduced all learning rates by 50%% after NaN")

                # Zero gradients and skip this iteration
                for opt in optimizers.values():
                    opt.zero_grad(set_to_none=True)
                continue
            else:
                consecutive_nan = 0

            # --- Strategy pre-backward hook (needs info from render) ---
            strategy.step_pre_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=iteration,
                info=info,
            )

            # --- Backward ---
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # --- Gradient explosion protection ---
            total_grad_norm = 0.0
            for key in splats:
                if splats[key].grad is not None:
                    param_norm = splats[key].grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5

            if total_grad_norm > 1000.0:
                gradient_clips += 1
                torch.nn.utils.clip_grad_norm_(splats.parameters(), max_norm=1.0)
                if appearance_mlp is not None:
                    torch.nn.utils.clip_grad_norm_(appearance_mlp.parameters(), max_norm=1.0)
                if appearance_embeddings is not None:
                    torch.nn.utils.clip_grad_norm_(appearance_embeddings.parameters(), max_norm=1.0)
                if gradient_clips <= 10 or gradient_clips % 100 == 0:
                    logger.warning(
                        "Gradient explosion at iter %d (norm=%.1f > 1000) — clipped to 1.0 "
                        "(total clips: %d)",
                        iteration, total_grad_norm, gradient_clips,
                    )

            # --- Strategy post-backward hook (handles densification/pruning) ---
            info["width"] = w_train
            info["height"] = h_train
            info["n_cameras"] = num_cameras

            if _effective_strategy == "mcmc":
                strategy.step_post_backward(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=iteration,
                    info=info,
                    lr=cfg.mcmc_noise_lr,
                )
            else:
                strategy.step_post_backward(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=iteration,
                    info=info,
                    packed=cfg.packed,
                )

            # --- Optimiser step ---
            # SelectiveAdam: pass visibility mask so only visible Gaussians are updated.
            # Visibility is derived from rasterization radii (radii > 0 = visible).
            _splat_keys = {"means", "scales", "quats", "opacities", "sh0", "shN"}

            if _using_selective_adam:
                # Extract visibility from rasterization info
                vis_mask = None
                if "radii" in info:
                    radii = info["radii"]  # (1, N) or (N,)
                    if radii.dim() == 2:
                        vis_mask = radii[0] > 0  # (N,)
                    else:
                        vis_mask = radii > 0  # (N,)

                if scaler is not None:
                    # Unscale all optimizers first, then step with visibility
                    for opt in optimizers.values():
                        scaler.unscale_(opt)
                    for key, opt in optimizers.items():
                        if key in _splat_keys:
                            opt.step(visibility=vis_mask)
                        else:
                            opt.step()
                    scaler.update()
                else:
                    for key, opt in optimizers.items():
                        if key in _splat_keys:
                            opt.step(visibility=vis_mask)
                        else:
                            opt.step()

                for opt in optimizers.values():
                    opt.zero_grad(set_to_none=True)
            else:
                if scaler is not None:
                    for opt in optimizers.values():
                        scaler.step(opt)
                    scaler.update()
                else:
                    for opt in optimizers.values():
                        opt.step()

                for opt in optimizers.values():
                    opt.zero_grad(set_to_none=True)

            # --- Logging ---
            with torch.no_grad():
                psnr_val = -10.0 * math.log10(loss_dict["l1"].item() + 1e-8)
            n_gs = splats["means"].shape[0]
            final_loss = loss.item()
            final_psnr = psnr_val
            postfix = {
                "loss": f"{final_loss:.4f}",
                "psnr": f"{psnr_val:.1f}",
                "n_gs": n_gs,
                "sh": active_sh_degree,
            }
            if use_2dgs and "normal" in loss_dict:
                postfix["nrm"] = f"{loss_dict['normal'].item():.3f}"
            pbar.set_postfix(postfix)

            if iteration % cfg.eval_every == 0:
                best_psnr = max(best_psnr, psnr_val)
                loss_strs = " | ".join(
                    f"{k}: {v.item():.4f}" for k, v in loss_dict.items()
                )
                logger.info(
                    "Iter %d | Loss %.4f | PSNR %.2f | Best %.2f | Gaussians %d | %s",
                    iteration, final_loss, psnr_val, best_psnr, n_gs, loss_strs,
                )

            # --- VRAM safety: enforce Gaussian cap ---
            # MCMC strategy enforces cap_max internally — skip manual pruning
            if _effective_strategy != "mcmc" and n_gs > cfg.max_num_gaussians:
                logger.warning(
                    "Gaussian count %d exceeds cap %d — pruning lowest-opacity Gaussians",
                    n_gs, cfg.max_num_gaussians,
                )
                with torch.no_grad():
                    opas = torch.sigmoid(splats["opacities"])
                    keep_k = int(cfg.max_num_gaussians * 0.9)  # Prune to 90% of cap
                    _, keep_idx = torch.topk(opas, min(keep_k, n_gs))
                    keep_idx = keep_idx.sort().values
                    for key in splats:
                        splats[key] = torch.nn.Parameter(splats[key][keep_idx].clone())
                    # Recreate only splat optimizers (keep appearance optimizers)
                    splat_optimizers = self._create_optimizers(splats)
                    # Merge back appearance optimizers
                    if appearance_mlp is not None:
                        splat_optimizers["appearance_embeddings"] = optimizers["appearance_embeddings"]
                        splat_optimizers["appearance_mlp"] = optimizers["appearance_mlp"]
                    optimizers = splat_optimizers
                    # Reinitialize strategy state for new Gaussian count
                    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
                    n_gs = splats["means"].shape[0]
                    logger.info("Pruned to %d Gaussians, reset strategy state", n_gs)
                    torch.cuda.empty_cache()

            # --- Checkpoint (with rotation and best tracking) ---
            if output_dir and iteration % cfg.checkpoint_every == 0:
                ckpt_path = self._save_checkpoint(
                    splats, iteration, output_dir, optimizers, final_loss, psnr_val,
                )
                checkpoint_paths.append(ckpt_path)
                # Rotate: keep at most 3 checkpoints (excluding best and final)
                while len(checkpoint_paths) > 3:
                    old_path = checkpoint_paths.pop(0)
                    if old_path.exists():
                        old_path.unlink()
                        logger.debug("Deleted old checkpoint: %s", old_path)

            # --- Best checkpoint ---
            if output_dir and psnr_val > best_checkpoint_psnr:
                best_checkpoint_psnr = psnr_val
                self._save_checkpoint(
                    splats, iteration, output_dir, optimizers, final_loss, psnr_val,
                    filename="best.pt",
                )

            # Free intermediate memory periodically
            if iteration % 500 == 0:
                torch.cuda.empty_cache()

        # --- Final save ---
        if output_dir:
            self._save_checkpoint(
                splats, cfg.iterations, output_dir, optimizers, final_loss, final_psnr,
                filename="final.pt",
            )
            self._export_ply(splats, output_dir / "final.ply")
            logger.info("Training complete. Final output saved to %s", output_dir)

        # --- Training summary ---
        total_time = time.time() - t_start
        peak_vram_mb = 0
        if self.device.type == "cuda":
            peak_vram_mb = int(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))

        summary = {
            "iterations": iteration if 'iteration' in dir() else 0,
            "final_loss": final_loss,
            "final_psnr": final_psnr,
            "best_psnr": best_psnr,
            "num_gaussians_start": num_gaussians_start,
            "num_gaussians_end": splats["means"].shape[0],
            "peak_vram_mb": peak_vram_mb,
            "total_time_s": total_time,
            "nan_count": nan_count,
            "oom_count": oom_count,
            "gradient_clips": gradient_clips,
        }
        logger.info("Training summary: %s", summary)

        # Store summary on the returned model for callers that need it
        result_model = _splats_to_gaussians(splats)
        result_model._training_summary = summary
        return result_model

    # ------------------------------------------------------------------
    # Strategy creation
    # ------------------------------------------------------------------

    def _create_strategy(
        self,
        splats: torch.nn.ParameterDict,
        scene_scale: float,
    ) -> tuple:
        """Instantiate the chosen gsplat Strategy and initialise its state."""
        cfg = self.config

        # Resolve "auto" strategy: for face scenes (bounded, known geometry)
        # MCMC is a good fit, but default is more battle-tested, so "auto"
        # currently maps to "default" for safety.
        effective_strategy = cfg.strategy
        if effective_strategy == "auto":
            effective_strategy = "default"
            logger.info("Strategy 'auto' resolved to 'default'")

        if effective_strategy == "mcmc":
            from gsplat import MCMCStrategy

            # Auto refine_stop_iter: 80% of total iterations if set to 0
            mcmc_refine_stop = (
                cfg.mcmc_refine_stop_iter
                if cfg.mcmc_refine_stop_iter > 0
                else int(cfg.iterations * 0.8)
            )

            strategy = MCMCStrategy(
                cap_max=cfg.mcmc_cap_max,
                noise_lr=cfg.mcmc_noise_lr,
                refine_every=cfg.mcmc_refine_every,
                refine_start_iter=cfg.mcmc_refine_start_iter,
                refine_stop_iter=mcmc_refine_stop,
                min_opacity=cfg.mcmc_min_opacity,
            )
            logger.info(
                "Using MCMCStrategy (cap_max=%d, refine_stop=%d, min_opacity=%.4f)",
                cfg.mcmc_cap_max, mcmc_refine_stop, cfg.mcmc_min_opacity,
            )
        else:
            from gsplat import DefaultStrategy

            # 2DGS uses "gradient_2dgs" for densification instead of "means2d"
            # In gsplat 1.4.0, 2DGS gradient_2dgs is a plain tensor (no .absgrad attr),
            # so we must disable absgrad in the strategy when using 2DGS
            use_2dgs = cfg.use_2dgs and _USE_2DGS_RASTERIZATION
            key_for_gradient = "gradient_2dgs" if use_2dgs else "means2d"
            strategy_absgrad = False if use_2dgs else cfg.absgrad

            # Auto refine_stop_iter: 60% of total iterations if set to 0
            refine_stop = cfg.refine_stop_iter if cfg.refine_stop_iter > 0 else int(cfg.iterations * 0.6)

            strategy = DefaultStrategy(
                prune_opa=cfg.prune_opa,
                grow_grad2d=cfg.grow_grad2d,
                grow_scale3d=cfg.grow_scale3d,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=refine_stop,
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=strategy_absgrad,
                key_for_gradient=key_for_gradient,
            )
            logger.info(
                "Using DefaultStrategy (absgrad=%s, key_for_gradient=%s)",
                strategy_absgrad, key_for_gradient,
            )

        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        return strategy, strategy_state

    # ------------------------------------------------------------------
    # Optimiser creation
    # ------------------------------------------------------------------

    def _create_optimizers(
        self,
        splats: torch.nn.ParameterDict,
    ) -> dict[str, torch.optim.Adam]:
        """Create per-parameter optimisers for Gaussian splat parameters.

        When ``use_selective_adam=True`` and SelectiveAdam is available, creates
        per-key SelectiveAdam optimizers. SelectiveAdam only updates visible
        Gaussians per iteration, giving 20-40% faster optimizer steps.

        Per-key structure is preserved for gsplat strategy compatibility --
        strategies (DefaultStrategy, MCMCStrategy) need per-key optimizer access
        to grow/prune/teleport individual Gaussian parameter tensors.

        When SelectiveAdam is unavailable or disabled, falls back to standard
        per-parameter Adam optimizers.
        """
        cfg = self.config

        # --- SelectiveAdam path: per-key SelectiveAdam optimizers ---
        if cfg.use_selective_adam and _SELECTIVE_ADAM_AVAILABLE:
            logger.info("Using SelectiveAdam for splat parameters (visibility-aware)")
            return {
                "means": SelectiveAdam([splats["means"]], lr=cfg.lr_means, eps=1e-15),
                "scales": SelectiveAdam([splats["scales"]], lr=cfg.lr_scales, eps=1e-15),
                "quats": SelectiveAdam([splats["quats"]], lr=cfg.lr_quats, eps=1e-15),
                "opacities": SelectiveAdam([splats["opacities"]], lr=cfg.lr_opacities, eps=1e-15),
                "sh0": SelectiveAdam([splats["sh0"]], lr=cfg.lr_sh0, eps=1e-15),
                "shN": SelectiveAdam([splats["shN"]], lr=cfg.lr_shN, eps=1e-15),
            }

        # --- Standard per-parameter Adam path ---
        if cfg.use_selective_adam and not _SELECTIVE_ADAM_AVAILABLE:
            logger.info(
                "SelectiveAdam requested but unavailable — falling back to standard Adam"
            )
        return {
            "means": torch.optim.Adam([splats["means"]], lr=cfg.lr_means, eps=1e-15),
            "scales": torch.optim.Adam([splats["scales"]], lr=cfg.lr_scales, eps=1e-15),
            "quats": torch.optim.Adam([splats["quats"]], lr=cfg.lr_quats, eps=1e-15),
            "opacities": torch.optim.Adam([splats["opacities"]], lr=cfg.lr_opacities, eps=1e-15),
            "sh0": torch.optim.Adam([splats["sh0"]], lr=cfg.lr_sh0, eps=1e-15),
            "shN": torch.optim.Adam([splats["shN"]], lr=cfg.lr_shN, eps=1e-15),
        }

    # ------------------------------------------------------------------
    # Rendering — 2DGS
    # ------------------------------------------------------------------

    def _render_2dgs(
        self,
        splats: torch.nn.ParameterDict,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        sh_degree: int,
        bg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Render Gaussians using 2DGS rasterization.

        Returns:
            (rgb, alpha, normals, surf_normals, distort, median_depth, meta)
        """
        cfg = self.config

        # Prepare colours: concat sh0 + shN -> (N, SH_COEFFS, 3)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        # 2DGS rasterization: distloss=True requires render_mode with depth
        # In gsplat 1.4.0, backgrounds is NOT auto-expanded for 2DGS, so omit it
        rast_kwargs = dict(
            means=splats["means"],
            quats=F.normalize(splats["quats"], p=2, dim=-1),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat[None],          # (1, 4, 4)
            Ks=K[None],                      # (1, 3, 3)
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=cfg.packed,
            absgrad=True,                    # Per context7: absgrad works with 2DGS
            render_mode="RGB+ED",            # Required by gsplat 1.4.0 when distloss=True
            distloss=True,                   # Enable distortion loss computation
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            radius_clip=cfg.radius_clip,
        )
        colors_out, alphas, normals, surf_normals, distort, median_depth, meta = _rasterization_2dgs_fn(
            **rast_kwargs
        )

        # With render_mode="RGB+ED", colors_out is (1, H, W, 4)
        rgb = colors_out[0, :, :, :3]   # (H, W, 3)
        alpha = alphas[0]               # (H, W, 1)
        normals_out = normals[0]     # (H, W, 3)
        surf_normals_out = surf_normals[0]  # (H, W, 3)
        distort_out = distort[0]     # (H, W, 1)
        median_depth_out = median_depth[0]  # (H, W, 1)

        return rgb, alpha, normals_out, surf_normals_out, distort_out, median_depth_out, meta

    # ------------------------------------------------------------------
    # Rendering — standard 3DGS (fallback)
    # ------------------------------------------------------------------

    def _render(
        self,
        splats: torch.nn.ParameterDict,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        sh_degree: int,
        bg: torch.Tensor,
        render_depth: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict]:
        """Render Gaussians for a single camera using gsplat 3DGS.

        Returns:
            (rgb_image, depth_or_None, alpha, info_dict)
        """
        from gsplat import rasterization

        cfg = self.config

        # Prepare colours: concat sh0 + shN -> (N, SH_COEFFS, 3)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        render_mode = "RGB+ED" if render_depth else "RGB"

        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=F.normalize(splats["quats"], p=2, dim=-1),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat[None],          # (1, 4, 4)
            Ks=K[None],                      # (1, 3, 3)
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=cfg.packed,
            sparse_grad=cfg.sparse_grad,
            absgrad=cfg.absgrad,
            render_mode=render_mode,
            rasterize_mode="antialiased",
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            radius_clip=cfg.radius_clip,
        )

        # renders: (1, H, W, C) where C=3 for RGB, C=4 for RGB+ED
        alpha = alphas[0]  # (H, W, 1)

        if render_depth:
            rgb = renders[0, :, :, :3]    # (H, W, 3)
            depth = renders[0, :, :, 3:]  # (H, W, 1)
        else:
            rgb = renders[0]              # (H, W, 3)
            depth = None

        return rgb, depth, alpha, info

    # ------------------------------------------------------------------
    # Loss computation — 2DGS
    # ------------------------------------------------------------------

    def _compute_loss_2dgs(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        normals: torch.Tensor,
        surf_normals: torch.Tensor,
        distort: torch.Tensor,
        median_depth: torch.Tensor,
        depth_gt: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        opacities: torch.Tensor | None = None,
        lpips_disabled: bool = False,
        iteration: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined 2DGS training loss with face-optimized terms.

        Staged loss introduction (when staged_loss=True):
            - Stage 1 (0-15%): L1 + depth supervision only (fast geometry)
            - Stage 2 (15-40%): + D-SSIM (structural similarity)
            - Stage 3 (40%+): + normal consistency + distortion (full loss)

        Depth loss decay: weight decays from lambda_depth_initial to
        lambda_depth_final by 60% of training.

        Losses:
            - L1 photometric
            - D-SSIM structural similarity
            - Normal consistency (normals vs surf_normals)
            - Distortion regularization
            - LPIPS perceptual similarity
            - Depth supervision (median_depth vs GT)
            - Opacity entropy (push toward 0 or 1)
        """
        cfg = self.config

        # Determine loss stage boundaries
        stage1_end = int(cfg.iterations * 0.15)  # 0-15%: L1 + depth only
        stage2_end = int(cfg.iterations * 0.40)  # 15-40%: + D-SSIM
        use_staged = cfg.staged_loss

        # Apply mask if provided
        if mask is not None:
            mask_3ch = mask.unsqueeze(-1)  # (H, W, 1)
            rendered_masked = rendered * mask_3ch
            target_masked = target * mask_3ch
        else:
            rendered_masked = rendered
            target_masked = target

        # ---- L1 loss (always active) ----
        l1_loss = F.l1_loss(rendered_masked, target_masked)
        loss_dict: dict[str, torch.Tensor] = {"l1": l1_loss}

        # ---- D-SSIM loss (stage 2+, or always if staged_loss=False) ----
        if not use_staged or iteration > stage1_end:
            x = rendered_masked.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            y = target_masked.permute(2, 0, 1).unsqueeze(0)
            ssim_val = _ssim_fn(x, y)
            dssim_loss = (1.0 - ssim_val) / 2.0
            total = (1.0 - cfg.lambda_dssim) * l1_loss + cfg.lambda_dssim * dssim_loss
            loss_dict["dssim"] = dssim_loss
        else:
            total = l1_loss

        # ---- Normal consistency loss (stage 3+, or always if staged_loss=False) ----
        if cfg.lambda_normal > 0.0:
            if not use_staged or iteration > stage2_end:
                # normals and surf_normals are both (H, W, 3)
                normal_consistency = (1.0 - (normals * surf_normals).sum(dim=-1)).mean()
                total = total + cfg.lambda_normal * normal_consistency
                loss_dict["normal"] = normal_consistency

        # ---- Distortion loss (stage 3+, or always if staged_loss=False) ----
        if cfg.lambda_distort > 0.0:
            if not use_staged or iteration > stage2_end:
                dist_loss = distort.mean()
                total = total + cfg.lambda_distort * dist_loss
                loss_dict["distort"] = dist_loss

        # ---- LPIPS perceptual loss (stage 3+, or always if staged_loss=False) ----
        if cfg.lambda_lpips > 0.0 and not lpips_disabled:
            if not use_staged or iteration > stage2_end:
                lpips_net = _get_lpips_net(self.device)
                if lpips_net is not None:
                    # LPIPS expects (B, 3, H, W) in [0, 1]
                    pred_lpips = rendered_masked.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                    gt_lpips = target_masked.permute(2, 0, 1).unsqueeze(0)
                    lpips_val = lpips_net(pred_lpips, gt_lpips).mean()
                    total = total + cfg.lambda_lpips * lpips_val
                    loss_dict["lpips"] = lpips_val

        # ---- Depth supervision with decay (always active) ----
        # Depth weight decays from lambda_depth_initial to lambda_depth_final
        # over the first 60% of iterations
        if depth_gt is not None and cfg.lambda_depth > 0.0:
            # median_depth: (H, W, 1), depth_gt: (H, W)
            depth_pred = median_depth.squeeze(-1)  # (H, W)
            valid_depth = depth_gt > 0
            if valid_depth.any():
                # Compute decaying depth weight
                decay_end = int(cfg.iterations * 0.6)
                if iteration <= decay_end and decay_end > 0:
                    decay_t = iteration / decay_end
                    depth_w = cfg.lambda_depth_initial + (cfg.lambda_depth_final - cfg.lambda_depth_initial) * decay_t
                else:
                    depth_w = cfg.lambda_depth_final
                depth_loss = F.l1_loss(depth_pred[valid_depth], depth_gt[valid_depth])
                total = total + depth_w * depth_loss
                loss_dict["depth"] = depth_loss

        # ---- Opacity entropy loss ----
        if opacities is not None and cfg.lambda_opacity_entropy > 0.0:
            # Push opacities toward 0 or 1 (binary) for cleaner surfaces
            opa = opacities.clamp(1e-6, 1.0 - 1e-6)
            entropy = -(opa * torch.log(opa) + (1.0 - opa) * torch.log(1.0 - opa))
            entropy_loss = entropy.mean()
            total = total + cfg.lambda_opacity_entropy * entropy_loss
            loss_dict["opa_ent"] = entropy_loss

        # ---- Mask loss (unchanged from 3DGS) ----
        if mask is not None and cfg.mask_weight > 0:
            mask_loss = F.binary_cross_entropy(
                rendered.mean(dim=-1).clamp(0, 1),
                mask.float(),
            )
            total = total + cfg.mask_weight * mask_loss
            loss_dict["mask"] = mask_loss

        return total, loss_dict

    # ------------------------------------------------------------------
    # Loss computation — standard 3DGS (fallback)
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        depth_rendered: torch.Tensor | None = None,
        depth_gt: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined training loss for standard 3DGS.

        Loss = (1 - lambda) * L1 + lambda * D-SSIM [+ depth + mask terms]
        """
        cfg = self.config

        # Apply mask if provided
        if mask is not None:
            mask_3ch = mask.unsqueeze(-1)  # (H, W, 1)
            rendered_masked = rendered * mask_3ch
            target_masked = target * mask_3ch
        else:
            rendered_masked = rendered
            target_masked = target

        # L1 loss
        l1_loss = F.l1_loss(rendered_masked, target_masked)

        # D-SSIM loss = (1 - SSIM) / 2
        # SSIM expects (1, C, H, W)
        x = rendered_masked.permute(2, 0, 1).unsqueeze(0)
        y = target_masked.permute(2, 0, 1).unsqueeze(0)
        ssim_val = _ssim_fn(x, y)
        dssim_loss = (1.0 - ssim_val) / 2.0

        total = (1.0 - cfg.lambda_dssim) * l1_loss + cfg.lambda_dssim * dssim_loss

        loss_dict: dict[str, torch.Tensor] = {"l1": l1_loss, "dssim": dssim_loss}

        # Depth supervision
        if depth_rendered is not None and depth_gt is not None and cfg.depth_weight > 0:
            # depth_rendered: (H, W, 1), depth_gt: (H, W)
            depth_pred = depth_rendered.squeeze(-1)
            valid_depth = depth_gt > 0
            if valid_depth.any():
                depth_loss = F.l1_loss(depth_pred[valid_depth], depth_gt[valid_depth])
                total = total + cfg.depth_weight * depth_loss
                loss_dict["depth"] = depth_loss

        # Mask loss: encourage opacity in masked regions
        if mask is not None and cfg.mask_weight > 0:
            mask_loss = F.binary_cross_entropy(
                rendered.mean(dim=-1).clamp(0, 1),
                mask.float(),
            )
            total = total + cfg.mask_weight * mask_loss
            loss_dict["mask"] = mask_loss

        return total, loss_dict

    # ------------------------------------------------------------------
    # Progressive training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample_for_training(
        image: torch.Tensor,
        depth: torch.Tensor | None,
        mask: torch.Tensor | None,
        w: int,
        h: int,
        K: torch.Tensor,
        factor: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int, int, torch.Tensor]:
        """Downsample image, depth, mask, and intrinsics by a given factor for progressive training.

        Args:
            factor: Downsampling factor (2 = half, 4 = quarter). Default 2.
        """
        h_out, w_out = max(1, h // factor), max(1, w // factor)
        scale = 1.0 / factor

        # Downsample image: (H, W, 3) -> (H/f, W/f, 3)
        img_bchw = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        img_down = F.interpolate(img_bchw, size=(h_out, w_out), mode="bilinear", align_corners=False)
        image_out = img_down[0].permute(1, 2, 0)  # (H/f, W/f, 3)

        # Downsample depth
        depth_out = None
        if depth is not None:
            d_bchw = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            d_down = F.interpolate(d_bchw, size=(h_out, w_out), mode="nearest")
            depth_out = d_down[0, 0]  # (H/f, W/f)

        # Downsample mask
        mask_out = None
        if mask is not None:
            m_bchw = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            m_down = F.interpolate(m_bchw, size=(h_out, w_out), mode="nearest")
            mask_out = m_down[0, 0]  # (H/f, W/f)

        # Adjust intrinsics for reduced resolution
        K_out = K.clone()
        K_out[0, :] *= scale  # fx, cx
        K_out[1, :] *= scale  # fy, cy

        return image_out, depth_out, mask_out, w_out, h_out, K_out

    # ------------------------------------------------------------------
    # Learning-rate scheduling
    # ------------------------------------------------------------------

    def _update_means_lr(self, optimizers: dict[str, torch.optim.Adam], iteration: int) -> None:
        """Exponentially decay the means learning rate with linear warmup.

        For the first lr_warmup_iters iterations, linearly ramp from 0 to lr_means.
        After warmup, apply exponential decay from lr_means to lr_means_final.
        """
        cfg = self.config

        # Linear warmup phase
        if cfg.lr_warmup_iters > 0 and iteration <= cfg.lr_warmup_iters:
            lr = cfg.lr_means * (iteration / cfg.lr_warmup_iters)
            optimizers["means"].param_groups[0]["lr"] = lr
            return

        t = min(iteration / cfg.lr_means_max_steps, 1.0)
        lr = math.exp(
            math.log(cfg.lr_means) * (1.0 - t) + math.log(cfg.lr_means_final) * t
        )
        delay_rate = cfg.lr_means_delay_mult + (1.0 - cfg.lr_means_delay_mult) * (
            math.sin(0.5 * math.pi * min(iteration / 1000.0, 1.0))
        )
        lr *= delay_rate
        optimizers["means"].param_groups[0]["lr"] = lr

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _preload_cameras(
        self,
        cameras: list[Camera],
        images_dir: Path,
        depth_dir: Path | None,
        masks_dir: Path | None,
    ) -> list[dict]:
        """Load and cache all training images, depths, and masks on GPU."""
        import cv2

        cam_data = []
        for cam_idx, cam in enumerate(tqdm(cameras, desc="Loading training data")):
            viewmat = cam.get_viewmat(self.device)
            K = cam.get_K(self.device)

            # Load image
            if cam.image_path:
                img_path = images_dir / cam.image_path
                if not img_path.exists():
                    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                        candidate = images_dir / (Path(cam.image_path).stem + ext)
                        if candidate.exists():
                            img_path = candidate
                            break
            else:
                img_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = images_dir / f"{cam.uid:06d}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break

            if img_path is None or not img_path.exists():
                logger.warning("Image not found for camera %d, skipping", cam.uid)
                continue

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            if img.shape[0] != cam.height or img.shape[1] != cam.width:
                img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)

            img_tensor = torch.from_numpy(img).to(self.device)  # (H, W, 3)

            entry: dict = {
                "viewmat": viewmat,
                "K": K,
                "image": img_tensor,
                "width": cam.width,
                "height": cam.height,
                "uid": cam.uid,
                "cam_idx": cam_idx,
            }

            # Depth map
            if depth_dir and cam.image_path:
                depth_name = Path(cam.image_path).stem + ".png"
                depth_path = depth_dir / depth_name
                if depth_path.exists():
                    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if depth_img is not None:
                        depth_img = depth_img.astype(np.float32)
                        if depth_img.max() > 100:
                            depth_img /= 1000.0
                        if depth_img.shape[:2] != (cam.height, cam.width):
                            depth_img = cv2.resize(depth_img, (cam.width, cam.height))
                        entry["depth"] = torch.from_numpy(depth_img).to(self.device)

            # Mask
            if masks_dir and cam.image_path:
                mask_name = Path(cam.image_path).stem + ".png"
                mask_path = masks_dir / mask_name
                if mask_path.exists():
                    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        mask_img = (mask_img > 127).astype(np.float32)
                        if mask_img.shape[:2] != (cam.height, cam.width):
                            mask_img = cv2.resize(mask_img, (cam.width, cam.height))
                        entry["mask"] = torch.from_numpy(mask_img).to(self.device)

            cam_data.append(entry)

        if not cam_data:
            raise RuntimeError("No valid training images found")

        return cam_data

    # ------------------------------------------------------------------
    # Checkpointing & export
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        splats: torch.nn.ParameterDict,
        iteration: int,
        output_dir: Path,
        optimizers: dict[str, torch.optim.Adam] | None = None,
        loss: float | None = None,
        psnr: float | None = None,
        filename: str | None = None,
    ) -> Path:
        """Save a training checkpoint.

        Returns the path to the saved checkpoint file.
        """
        if filename is None:
            filename = f"checkpoint_{iteration:06d}.pt"
        ckpt_path = output_dir / filename
        state = {k: v.detach().cpu() for k, v in splats.items()}
        state["iteration"] = iteration
        if loss is not None:
            state["loss"] = loss
        if psnr is not None:
            state["psnr"] = psnr
        if optimizers is not None:
            state["optimizer_states"] = {
                k: opt.state_dict() for k, opt in optimizers.items()
            }
        torch.save(state, str(ckpt_path))
        logger.info("Saved checkpoint at iteration %d to %s", iteration, ckpt_path)
        return ckpt_path

    def _export_ply(self, splats: torch.nn.ParameterDict, path: Path) -> None:
        """Export final Gaussians as a .ply using gsplat's export_splats."""
        try:
            from gsplat import export_splats

            with torch.no_grad():
                export_splats(
                    means=splats["means"],
                    scales=torch.exp(splats["scales"]),
                    quats=splats["quats"] / splats["quats"].norm(dim=-1, keepdim=True),
                    opacities=torch.sigmoid(splats["opacities"]),
                    sh0=splats["sh0"],
                    shN=splats["shN"],
                    format="ply",
                    save_to=str(path),
                )
            logger.info("Exported PLY via gsplat.export_splats to %s", path)
        except (ImportError, AttributeError):
            # Fallback: save as checkpoint if export_splats is unavailable
            logger.warning("gsplat.export_splats not available; saving raw checkpoint instead")
            self._save_checkpoint(splats, -1, path.parent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _nullcontext:
    """Minimal no-op context manager for when AMP is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
