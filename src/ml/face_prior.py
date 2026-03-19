"""Face prior network for predicting Gaussian initialization from FLAME parameters.

A small MLP (~50K params) that learns to predict per-region Gaussian statistics
(offsets, scales, opacities) from FLAME shape and expression parameters. Trained
on accumulated scan data from the replay buffer.

Architecture:
    Input:  FLAME shape (100-dim) + expression (50-dim) = 150-dim
    Encoder: 3-layer MLP (150 -> 256 -> 256 -> 128) with ReLU
    Heads:
        - offset_head:  128 -> 9*3 = 27  (xyz offset per region)
        - scale_head:   128 -> 9*3 = 27  (xyz log-scale per region)
        - opacity_head: 128 -> 9        (opacity bias per region)

Total params: ~50K (trains in seconds on CPU with 10-20 scans)

Data flow:
    1. Pipeline completes a scan -> replay_buffer.add_scan(...)
    2. If buffer has >= 5 scans -> prior.train_on_buffer(buffer)
    3. Next scan starts -> prior.predict_initialization(shape, expr)
    4. Predicted offsets refined by TaichiMeshRefiner
    5. Refined predictions used as Gaussian initialization
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# Face regions (must match replay_buffer.FACE_REGIONS)
FACE_REGIONS = [
    "forehead", "eyes", "nose", "mouth", "cheeks", "chin", "ears", "neck", "hair"
]
NUM_REGIONS = len(FACE_REGIONS)


class FacePriorNetwork(nn.Module):
    """Small neural network that predicts Gaussian initialization from FLAME params.

    Input: FLAME shape params (100-dim) + expression params (50-dim) = 150-dim
    Output: Per-region Gaussian statistics:
        - offsets: (num_regions, 3) mean offset from FLAME surface per region
        - scales: (num_regions, 3) log-scale factors per region
        - opacities: (num_regions,) opacity bias per region [0, 1]

    The network is intentionally small (~50K params) so it trains fast on
    10-20 scans and generalizes well. It provides a warm start for Gaussian
    initialization rather than perfect predictions.

    Args:
        shape_dim: Dimension of FLAME shape parameters (default 100).
        expr_dim: Dimension of FLAME expression parameters (default 50).
        hidden_dim: Hidden layer width (default 256).
        num_regions: Number of face regions (default 9).
    """

    def __init__(
        self,
        shape_dim: int = 100,
        expr_dim: int = 50,
        hidden_dim: int = 256,
        num_regions: int = NUM_REGIONS,
    ):
        super().__init__()
        self.shape_dim = shape_dim
        self.expr_dim = expr_dim
        self.num_regions = num_regions

        input_dim = shape_dim + expr_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        feat_dim = hidden_dim // 2

        # Per-region prediction heads
        self.offset_head = nn.Linear(feat_dim, num_regions * 3)   # xyz offset
        self.scale_head = nn.Linear(feat_dim, num_regions * 3)    # xyz log-scale
        self.opacity_head = nn.Linear(feat_dim, num_regions)      # opacity bias

        # Initialize heads with small weights for stable start
        for head in [self.offset_head, self.scale_head, self.opacity_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

        param_count = sum(p.numel() for p in self.parameters())
        logger.info("FacePriorNetwork: %d parameters", param_count)

    def forward(
        self,
        shape_params: torch.Tensor,
        expr_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            shape_params: (batch, shape_dim) FLAME shape coefficients.
            expr_params: (batch, expr_dim) FLAME expression coefficients.

        Returns:
            Dict with:
                'offsets': (batch, num_regions, 3) predicted xyz offsets
                'scales': (batch, num_regions, 3) predicted log-scale factors
                'opacities': (batch, num_regions) predicted opacity [0, 1]
        """
        # Pad or truncate to expected dimensions
        shape_params = _pad_or_truncate(shape_params, self.shape_dim)
        expr_params = _pad_or_truncate(expr_params, self.expr_dim)

        x = torch.cat([shape_params, expr_params], dim=-1)
        features = self.encoder(x)

        offsets = self.offset_head(features).view(-1, self.num_regions, 3)
        scales = self.scale_head(features).view(-1, self.num_regions, 3)
        opacities = torch.sigmoid(self.opacity_head(features))

        return {
            "offsets": offsets,
            "scales": scales,
            "opacities": opacities,
        }

    def train_on_buffer(
        self,
        buffer,  # ScanReplayBuffer — avoid circular import
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> dict[str, float]:
        """Train the prior network on accumulated scan data.

        Args:
            buffer: A ScanReplayBuffer instance with training data.
            epochs: Number of training epochs.
            lr: Learning rate.
            device: Device to train on ('cpu' recommended for this small network).

        Returns:
            Dict with 'final_loss', 'epochs_trained'.
        """
        data = buffer.get_prior_training_data()
        n_scans = len(data["shape_params"])

        if n_scans < 2:
            logger.warning(
                "Not enough scans for training (%d < 2), skipping", n_scans
            )
            return {"final_loss": float("inf"), "epochs_trained": 0}

        logger.info("Training prior network on %d scans for %d epochs", n_scans, epochs)

        device = torch.device(device)
        self.to(device)
        self.train()

        # Prepare tensors
        shape_t = torch.from_numpy(data["shape_params"]).float().to(device)
        expr_t = torch.from_numpy(data["expression_params"]).float().to(device)

        # Build target tensors from gaussian_stats
        target_offsets = []
        target_scales = []
        target_opacities = []

        for region in FACE_REGIONS:
            region_data = data["gaussian_stats"].get(region, {})
            target_offsets.append(
                torch.from_numpy(region_data.get("mean_offset", np.zeros((n_scans, 3)))).float()
            )
            target_scales.append(
                torch.from_numpy(region_data.get("mean_scale", np.zeros((n_scans, 3)))).float()
            )
            target_opacities.append(
                torch.from_numpy(region_data.get("mean_opacity", np.full(n_scans, 0.5))).float()
            )

        # Stack: (n_scans, num_regions, 3) and (n_scans, num_regions)
        target_offsets_t = torch.stack(target_offsets, dim=1).to(device)
        target_scales_t = torch.stack(target_scales, dim=1).to(device)
        target_opacities_t = torch.stack(target_opacities, dim=1).to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        final_loss = float("inf")

        for epoch in range(epochs):
            pred = self.forward(shape_t, expr_t)

            loss_offset = nn.functional.mse_loss(pred["offsets"], target_offsets_t)
            loss_scale = nn.functional.mse_loss(pred["scales"], target_scales_t)
            loss_opacity = nn.functional.mse_loss(pred["opacities"], target_opacities_t)

            # Weight offset loss higher (most important for initialization)
            loss = 2.0 * loss_offset + loss_scale + 0.5 * loss_opacity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss = loss.item()

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "  Epoch %d/%d: loss=%.6f (offset=%.6f, scale=%.6f, opacity=%.6f)",
                    epoch + 1, epochs, final_loss,
                    loss_offset.item(), loss_scale.item(), loss_opacity.item(),
                )

        self.eval()
        logger.info("Prior network training complete: final_loss=%.6f", final_loss)

        return {"final_loss": final_loss, "epochs_trained": epochs}

    @torch.no_grad()
    def predict_initialization(
        self,
        shape_params: np.ndarray,
        expr_params: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Predict Gaussian initialization for a new scan.

        Args:
            shape_params: (N_shape,) FLAME shape coefficients.
            expr_params: (N_expr,) FLAME expression coefficients.

        Returns:
            Dict with per-region arrays:
                'offsets': (num_regions, 3) xyz offsets
                'scales': (num_regions, 3) log-scale factors
                'opacities': (num_regions,) opacity values [0, 1]
        """
        self.eval()
        device = next(self.parameters()).device

        shape_t = torch.from_numpy(shape_params).float().unsqueeze(0).to(device)
        expr_t = torch.from_numpy(expr_params).float().unsqueeze(0).to(device)

        pred = self.forward(shape_t, expr_t)

        return {
            "offsets": pred["offsets"][0].cpu().numpy(),
            "scales": pred["scales"][0].cpu().numpy(),
            "opacities": pred["opacities"][0].cpu().numpy(),
        }

    def save(self, path: Path) -> None:
        """Save model weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(path))
        logger.info("Saved prior network weights to %s", path)

    def load(self, path: Path) -> None:
        """Load model weights from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning("Prior network weights not found at %s", path)
            return
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
        self.eval()
        logger.info("Loaded prior network weights from %s", path)


def _pad_or_truncate(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad with zeros or truncate the last dimension to target_dim."""
    current_dim = tensor.shape[-1]
    if current_dim == target_dim:
        return tensor
    if current_dim > target_dim:
        return tensor[..., :target_dim]
    # Pad with zeros
    pad_size = target_dim - current_dim
    pad = torch.zeros(*tensor.shape[:-1], pad_size, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=-1)
