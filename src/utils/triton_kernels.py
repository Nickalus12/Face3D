"""Custom Triton GPU kernels for Face3D pipeline hotspots.

Provides GPU-accelerated implementations of operations that are bottlenecks
in the Gaussian splatting training, depth unprojection, and export stages.
Every kernel has a pure PyTorch fallback so the pipeline runs on any device.

Kernels:
    - confidence_prune: Parallel confidence thresholding for Gaussian pruning
    - sh_to_rgb: Evaluate spherical harmonics (degree 0-3) to RGB
    - unproject_depth: Depth map to 3D point cloud with confidence filtering
    - opacity_entropy: Opacity entropy loss for surface regularization
    - pack_ply_vertices: Pack Gaussian data into PLY binary layout on GPU

Usage:
    from utils.triton_kernels import confidence_prune, sh_to_rgb, unproject_depth

    mask = confidence_prune(confidences, threshold=0.5)
    rgb = sh_to_rgb(sh_coeffs, directions, sh_degree=1)
    points, valid = unproject_depth(depth, K_inv, confidence, threshold=0.3)
"""

from __future__ import annotations

import logging
import struct
import time
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triton availability check
# ---------------------------------------------------------------------------

HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
    logger.info("Triton available (version %s)", triton.__version__)
except ImportError:
    logger.info("Triton not available; all kernels will use PyTorch fallbacks")

# SH basis constants (real spherical harmonics, unnormalized)
_SH_C0 = 0.28209479177387814       # 1 / (2 * sqrt(pi))
_SH_C1 = 0.4886025119029199        # sqrt(3) / (2 * sqrt(pi))
_SH_C2_0 = 1.0925484305920792      # sqrt(15) / (2 * sqrt(pi))
_SH_C2_1 = 0.31539156525252005     # sqrt(5) / (4 * sqrt(pi))
_SH_C2_2 = 0.5462742152960396      # sqrt(15) / (4 * sqrt(pi))
_SH_C3_0 = 0.5900435899266435      # sqrt(70) / (8 * sqrt(pi))  (approx)
_SH_C3_1 = 2.890611442640554       # sqrt(105) / (2 * sqrt(pi)) (approx)
_SH_C3_2 = 0.4570457994644658      # sqrt(42) / (8 * sqrt(pi))  (approx)
_SH_C3_3 = 0.3731763325901154      # sqrt(7) / (4 * sqrt(pi))   (approx)


# ===================================================================
# Kernel 1: Confidence-based Gaussian Pruning
# ===================================================================

if HAS_TRITON:
    @triton.jit
    def _confidence_prune_kernel(
        confidences_ptr,
        threshold,
        mask_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Parallel confidence thresholding for Gaussian pruning.

        For each element, writes True (1) if confidence > threshold, else False (0).
        This replaces the scalar PyTorch comparison with a fused GPU kernel that
        avoids materializing intermediate tensors.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        conf = tl.load(confidences_ptr + offsets, mask=mask, other=0.0)
        result = conf > threshold
        tl.store(mask_ptr + offsets, result, mask=mask)


def confidence_prune(confidences: torch.Tensor, threshold: float) -> torch.Tensor:
    """Prune Gaussians by confidence threshold.

    Returns a boolean mask where True = keep (confidence > threshold).

    Args:
        confidences: (N,) float tensor of confidence values.
        threshold: Minimum confidence to keep.

    Returns:
        (N,) bool tensor.
    """
    if confidences.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=confidences.device)

    if HAS_TRITON and confidences.is_cuda:
        n = confidences.numel()
        confidences_flat = confidences.contiguous().view(-1)
        mask_out = torch.empty(n, dtype=torch.bool, device=confidences.device)
        BLOCK_SIZE = 1024
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _confidence_prune_kernel[grid](
            confidences_flat,
            threshold,
            mask_out,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return mask_out.view(confidences.shape)

    # PyTorch fallback
    return confidences > threshold


# ===================================================================
# Kernel 2: Fast SH (Spherical Harmonics) Evaluation
# ===================================================================

if HAS_TRITON:
    @triton.jit
    def _sh_to_rgb_kernel(
        # Pointers
        sh_ptr,          # (N, C, 3) flattened: sh_ptr[n * C * 3 + c * 3 + ch]
        dir_ptr,         # (N, 3) flattened: dir_ptr[n * 3 + d]
        rgb_ptr,         # (N, 3) output
        # Scalars
        n_points,
        n_coeffs,        # number of SH coefficients (1, 4, 9, or 16)
        # SH constants (passed as runtime values since tl.constexpr is for ints)
        sh_c0: tl.constexpr,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Evaluate spherical harmonics to RGB for a batch of points.

        Supports SH degree 0 (1 coeff) only in the Triton kernel for simplicity
        and speed. Higher degrees use the PyTorch fallback which is already fast
        for the batch sizes in Face3D (300K points).

        For degree 0: rgb = sh[0] * C0 + 0.5
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_points

        # Load DC coefficients for R, G, B
        # sh_ptr layout: [n * n_coeffs * 3 + coeff * 3 + channel]
        base = offsets * n_coeffs * 3  # start of each point's SH data
        sh_r = tl.load(sh_ptr + base + 0, mask=mask, other=0.0)
        sh_g = tl.load(sh_ptr + base + 1, mask=mask, other=0.0)
        sh_b = tl.load(sh_ptr + base + 2, mask=mask, other=0.0)

        # Degree 0: rgb = sh_dc * C0 + 0.5
        c0 = 0.28209479177387814
        r = sh_r * c0 + 0.5
        g = sh_g * c0 + 0.5
        b = sh_b * c0 + 0.5

        # Clamp to [0, 1]
        r = tl.maximum(tl.minimum(r, 1.0), 0.0)
        g = tl.maximum(tl.minimum(g, 1.0), 0.0)
        b = tl.maximum(tl.minimum(b, 1.0), 0.0)

        # Store
        out_base = offsets * 3
        tl.store(rgb_ptr + out_base + 0, r, mask=mask)
        tl.store(rgb_ptr + out_base + 1, g, mask=mask)
        tl.store(rgb_ptr + out_base + 2, b, mask=mask)


def sh_to_rgb(
    sh_coeffs: torch.Tensor,
    directions: torch.Tensor,
    sh_degree: int = 1,
) -> torch.Tensor:
    """Evaluate spherical harmonics to produce RGB colors.

    Args:
        sh_coeffs: (N, C, 3) SH coefficients where C = (degree+1)^2.
        directions: (N, 3) unit viewing directions.
        sh_degree: Active SH degree (0, 1, 2, or 3).

    Returns:
        (N, 3) RGB values clamped to [0, 1].
    """
    n_points = sh_coeffs.shape[0]
    if n_points == 0:
        return torch.zeros(0, 3, dtype=sh_coeffs.dtype, device=sh_coeffs.device)

    n_coeffs = sh_coeffs.shape[1]

    # Triton fast path: degree 0 only (DC term) on CUDA
    if HAS_TRITON and sh_coeffs.is_cuda and sh_degree == 0:
        sh_flat = sh_coeffs.contiguous()
        dir_flat = directions.contiguous()
        rgb_out = torch.empty(n_points, 3, dtype=sh_coeffs.dtype, device=sh_coeffs.device)
        BLOCK_SIZE = 1024
        grid = ((n_points + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _sh_to_rgb_kernel[grid](
            sh_flat, dir_flat, rgb_out,
            n_points, n_coeffs,
            sh_c0=0,  # unused constexpr placeholder
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return rgb_out

    # PyTorch fallback — supports all SH degrees
    result = torch.zeros(n_points, 3, dtype=sh_coeffs.dtype, device=sh_coeffs.device)

    # Degree 0 (DC)
    result += _SH_C0 * sh_coeffs[:, 0, :]

    if sh_degree >= 1 and n_coeffs >= 4:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        # Degree 1: 3 coefficients
        result += _SH_C1 * (-y * sh_coeffs[:, 1, :] + z * sh_coeffs[:, 2, :] - x * sh_coeffs[:, 3, :])

    if sh_degree >= 2 and n_coeffs >= 9:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        result += _SH_C2_0 * xy * sh_coeffs[:, 4, :]
        result += _SH_C2_0 * yz * sh_coeffs[:, 5, :]
        result += _SH_C2_1 * (2.0 * zz - xx - yy) * sh_coeffs[:, 6, :]
        result += _SH_C2_0 * xz * sh_coeffs[:, 7, :]
        result += _SH_C2_2 * (xx - yy) * sh_coeffs[:, 8, :]

    if sh_degree >= 3 and n_coeffs >= 16:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        xx, yy, zz = x * x, y * y, z * z
        result += _SH_C3_0 * y * (3.0 * xx - yy) * sh_coeffs[:, 9, :]
        result += _SH_C3_1 * x * y * z * sh_coeffs[:, 10, :]
        result += _SH_C3_2 * y * (4.0 * zz - xx - yy) * sh_coeffs[:, 11, :]
        result += _SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[:, 12, :]
        result += _SH_C3_2 * x * (4.0 * zz - xx - yy) * sh_coeffs[:, 13, :]
        result += _SH_C3_0 * z * (xx - yy) * sh_coeffs[:, 14, :]
        result += _SH_C3_0 * x * (xx - 3.0 * yy) * sh_coeffs[:, 15, :]

    result += 0.5  # Bias
    return result.clamp(0.0, 1.0)


# ===================================================================
# Kernel 3: Depth Map Unprojection
# ===================================================================

if HAS_TRITON:
    @triton.jit
    def _unproject_depth_kernel(
        depth_ptr,        # (H * W,) flattened depth values
        K_inv_ptr,        # (9,) flattened 3x3 inverse intrinsics
        points_ptr,       # (H * W, 3) output 3D points
        valid_ptr,        # (H * W,) output validity mask
        conf_ptr,         # (H * W,) confidence values (or NULL)
        conf_threshold,   # float threshold
        has_conf,         # int: 1 if confidence is provided, 0 otherwise
        H, W,
        depth_min,
        depth_max,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Unproject depth map pixels to 3D camera-space points.

        Each thread handles one pixel: reads depth, checks validity and
        confidence, computes 3D point using K_inv, and writes output.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        n_elements = H * W
        mask = offsets < n_elements

        # Pixel coordinates
        px = offsets % W
        py = offsets // W

        # Load depth
        d = tl.load(depth_ptr + offsets, mask=mask, other=0.0)

        # Validity check: depth_min < d < depth_max and finite
        valid = (d > depth_min) & (d < depth_max)

        # Confidence check
        if has_conf == 1:
            conf = tl.load(conf_ptr + offsets, mask=mask, other=0.0)
            valid = valid & (conf > conf_threshold)

        # Load K_inv elements (row-major 3x3)
        k00 = tl.load(K_inv_ptr + 0)
        k02 = tl.load(K_inv_ptr + 2)
        k11 = tl.load(K_inv_ptr + 4)
        k12 = tl.load(K_inv_ptr + 5)

        # Unproject: [x_cam, y_cam, z_cam] = d * K_inv @ [px, py, 1]
        # For pinhole: x_cam = d * (px - cx) / fx = d * (k00 * px + k02)
        px_f = px.to(tl.float32)
        py_f = py.to(tl.float32)
        x_cam = d * (k00 * px_f + k02)
        y_cam = d * (k11 * py_f + k12)
        z_cam = d

        # Store points
        out_base = offsets * 3
        tl.store(points_ptr + out_base + 0, x_cam, mask=mask & valid)
        tl.store(points_ptr + out_base + 1, y_cam, mask=mask & valid)
        tl.store(points_ptr + out_base + 2, z_cam, mask=mask & valid)

        # Store zeros for invalid points
        tl.store(points_ptr + out_base + 0, 0.0, mask=mask & ~valid)
        tl.store(points_ptr + out_base + 1, 0.0, mask=mask & ~valid)
        tl.store(points_ptr + out_base + 2, 0.0, mask=mask & ~valid)

        # Store validity mask
        tl.store(valid_ptr + offsets, valid, mask=mask)


def unproject_depth(
    depth: torch.Tensor,
    K_inv: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    conf_threshold: float = 0.3,
    depth_min: float = 0.05,
    depth_max: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unproject a depth map to 3D camera-space points with confidence filtering.

    Args:
        depth: (H, W) depth map in meters.
        K_inv: (3, 3) inverse camera intrinsic matrix.
        confidence: Optional (H, W) confidence map.
        conf_threshold: Minimum confidence to keep a point.
        depth_min: Minimum valid depth.
        depth_max: Maximum valid depth.

    Returns:
        Tuple of:
            points: (H, W, 3) 3D points in camera coordinates (zeros where invalid).
            valid_mask: (H, W) boolean mask of valid points.
    """
    H, W = depth.shape
    device = depth.device

    if H == 0 or W == 0:
        return (
            torch.zeros(H, W, 3, dtype=depth.dtype, device=device),
            torch.zeros(H, W, dtype=torch.bool, device=device),
        )

    if HAS_TRITON and depth.is_cuda:
        depth_flat = depth.contiguous().view(-1)
        K_inv_flat = K_inv.contiguous().view(-1).to(depth.dtype)
        points_out = torch.zeros(H * W, 3, dtype=depth.dtype, device=device)
        valid_out = torch.zeros(H * W, dtype=torch.bool, device=device)

        has_conf = 0
        conf_flat = depth_flat  # dummy, won't be read
        if confidence is not None:
            has_conf = 1
            conf_flat = confidence.contiguous().view(-1)

        n = H * W
        BLOCK_SIZE = 1024
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _unproject_depth_kernel[grid](
            depth_flat, K_inv_flat, points_out, valid_out, conf_flat,
            conf_threshold, has_conf, H, W, depth_min, depth_max,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return points_out.view(H, W, 3), valid_out.view(H, W)

    # PyTorch fallback
    ys = torch.arange(H, device=device, dtype=depth.dtype)
    xs = torch.arange(W, device=device, dtype=depth.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # Validity
    valid = (depth > depth_min) & (depth < depth_max) & torch.isfinite(depth)
    if confidence is not None:
        valid = valid & (confidence > conf_threshold)

    # Unproject using K_inv (pinhole model)
    fx_inv = K_inv[0, 0]
    cx_offset = K_inv[0, 2]
    fy_inv = K_inv[1, 1]
    cy_offset = K_inv[1, 2]

    x_cam = depth * (fx_inv * xx + cx_offset)
    y_cam = depth * (fy_inv * yy + cy_offset)
    z_cam = depth

    points = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    points[~valid] = 0.0

    return points, valid


# ===================================================================
# Kernel 4: Gaussian Opacity Entropy Loss
# ===================================================================

if HAS_TRITON:
    @triton.jit
    def _opacity_entropy_kernel(
        opacities_ptr,
        entropy_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute per-element opacity entropy: -o*log(o) - (1-o)*log(1-o).

        Pushes opacities toward 0 or 1 (binary) for cleaner Gaussian surfaces.
        The result is stored per-element; the caller reduces with .mean().
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        o = tl.load(opacities_ptr + offsets, mask=mask, other=0.5)

        # Clamp to avoid log(0)
        eps = 1e-6
        o_clamped = tl.minimum(tl.maximum(o, eps), 1.0 - eps)
        one_minus_o = 1.0 - o_clamped

        # Entropy: -o * log(o) - (1-o) * log(1-o)
        entropy = -(o_clamped * tl.log(o_clamped) + one_minus_o * tl.log(one_minus_o))

        tl.store(entropy_ptr + offsets, entropy, mask=mask)


def opacity_entropy(opacities: torch.Tensor) -> torch.Tensor:
    """Compute mean opacity entropy loss.

    Encourages opacities to be near 0 or 1 (binary), producing cleaner
    Gaussian surfaces. Used as a regularization term during 2DGS training.

    Args:
        opacities: (N,) or (N, 1) sigmoid-activated opacities in [0, 1].

    Returns:
        Scalar loss tensor (mean entropy).
    """
    opacities_flat = opacities.contiguous().view(-1)
    n = opacities_flat.numel()

    if n == 0:
        return torch.tensor(0.0, dtype=opacities.dtype, device=opacities.device)

    if HAS_TRITON and opacities.is_cuda:
        entropy_out = torch.empty(n, dtype=opacities.dtype, device=opacities.device)
        BLOCK_SIZE = 1024
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _opacity_entropy_kernel[grid](
            opacities_flat, entropy_out, n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return entropy_out.mean()

    # PyTorch fallback
    o = opacities_flat.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(o * torch.log(o) + (1.0 - o) * torch.log(1.0 - o))
    return entropy.mean()


# ===================================================================
# Kernel 5: PLY Vertex Packing
# ===================================================================

if HAS_TRITON:
    @triton.jit
    def _pack_ply_vertices_kernel(
        positions_ptr,   # (N, 3) float32
        colors_ptr,      # (N, 3) float32 in [0, 1]
        normals_ptr,     # (N, 3) float32
        output_ptr,      # (N, 9) float32 packed output
        n_vertices,
        has_normals,     # int: 1 if normals provided
        BLOCK_SIZE: tl.constexpr,
    ):
        """Pack Gaussian positions + colors + normals into a contiguous buffer.

        Output layout per vertex: [x, y, z, nx, ny, nz, r, g, b] as float32.
        Colors are converted from [0,1] to [0,255] and stored as float32
        for uniform access. The CPU side handles final byte conversion.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_vertices

        # Load positions
        pos_base = offsets * 3
        px = tl.load(positions_ptr + pos_base + 0, mask=mask, other=0.0)
        py = tl.load(positions_ptr + pos_base + 1, mask=mask, other=0.0)
        pz = tl.load(positions_ptr + pos_base + 2, mask=mask, other=0.0)

        # Load or default normals
        if has_normals == 1:
            nx = tl.load(normals_ptr + pos_base + 0, mask=mask, other=0.0)
            ny = tl.load(normals_ptr + pos_base + 1, mask=mask, other=0.0)
            nz = tl.load(normals_ptr + pos_base + 2, mask=mask, other=0.0)
        else:
            nx = 0.0
            ny = 0.0
            nz = 0.0

        # Load colors and scale to [0, 255]
        col_base = offsets * 3
        cr = tl.load(colors_ptr + col_base + 0, mask=mask, other=0.0) * 255.0
        cg = tl.load(colors_ptr + col_base + 1, mask=mask, other=0.0) * 255.0
        cb = tl.load(colors_ptr + col_base + 2, mask=mask, other=0.0) * 255.0

        # Clamp colors
        cr = tl.minimum(tl.maximum(cr, 0.0), 255.0)
        cg = tl.minimum(tl.maximum(cg, 0.0), 255.0)
        cb = tl.minimum(tl.maximum(cb, 0.0), 255.0)

        # Store packed: [x, y, z, nx, ny, nz, r, g, b]
        out_base = offsets * 9
        tl.store(output_ptr + out_base + 0, px, mask=mask)
        tl.store(output_ptr + out_base + 1, py, mask=mask)
        tl.store(output_ptr + out_base + 2, pz, mask=mask)
        tl.store(output_ptr + out_base + 3, nx, mask=mask)
        tl.store(output_ptr + out_base + 4, ny, mask=mask)
        tl.store(output_ptr + out_base + 5, nz, mask=mask)
        tl.store(output_ptr + out_base + 6, cr, mask=mask)
        tl.store(output_ptr + out_base + 7, cg, mask=mask)
        tl.store(output_ptr + out_base + 8, cb, mask=mask)


def pack_ply_vertices(
    positions: torch.Tensor,
    colors: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack Gaussian data into a contiguous buffer for PLY binary export.

    This avoids the slow per-vertex Python loop in the streaming PLY writer
    by packing all vertex data in a single GPU kernel call.

    Args:
        positions: (N, 3) float32 xyz positions.
        colors: (N, 3) float32 RGB colors in [0, 1].
        normals: Optional (N, 3) float32 normals. Zeros if not provided.

    Returns:
        (N, 9) float32 tensor: [x, y, z, nx, ny, nz, r*255, g*255, b*255].
    """
    n = positions.shape[0]
    device = positions.device

    if n == 0:
        return torch.zeros(0, 9, dtype=torch.float32, device=device)

    if HAS_TRITON and positions.is_cuda:
        positions_flat = positions.contiguous().view(-1)
        colors_flat = colors.contiguous().view(-1)
        has_normals = 0
        normals_flat = positions_flat  # dummy
        if normals is not None:
            has_normals = 1
            normals_flat = normals.contiguous().view(-1)

        output = torch.empty(n, 9, dtype=torch.float32, device=device)
        output_flat = output.view(-1)

        BLOCK_SIZE = 1024
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _pack_ply_vertices_kernel[grid](
            positions_flat, colors_flat, normals_flat, output_flat,
            n, has_normals,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

    # PyTorch fallback
    if normals is None:
        normals = torch.zeros_like(positions)

    colors_255 = (colors * 255.0).clamp(0.0, 255.0)
    return torch.cat([positions, normals, colors_255], dim=-1)


def pack_ply_to_bytes(packed: torch.Tensor) -> bytes:
    """Convert packed PLY vertex tensor to binary PLY bytes.

    Takes the output of pack_ply_vertices and produces the binary data
    portion of a PLY file (positions as float32, colors as uint8).

    Args:
        packed: (N, 9) tensor from pack_ply_vertices.

    Returns:
        Raw bytes for PLY binary_little_endian format.
    """
    packed_cpu = packed.cpu().numpy()
    n = len(packed_cpu)
    if n == 0:
        return b""

    # Build byte buffer: 3 floats (xyz) + 3 bytes (rgb) = 15 bytes per vertex
    buf = bytearray(n * 15)
    for i in range(n):
        offset = i * 15
        struct.pack_into(
            "<fffBBB", buf, offset,
            packed_cpu[i, 0], packed_cpu[i, 1], packed_cpu[i, 2],  # xyz
            int(packed_cpu[i, 6]), int(packed_cpu[i, 7]), int(packed_cpu[i, 8]),  # rgb
        )
    return bytes(buf)


# ===================================================================
# Benchmarking
# ===================================================================

def benchmark_kernels(
    n_gaussians: int = 300_000,
    depth_h: int = 1080,
    depth_w: int = 1920,
    n_warmup: int = 5,
    n_iters: int = 20,
    device: str = "cuda",
) -> dict:
    """Benchmark Triton kernels vs PyTorch fallbacks.

    Args:
        n_gaussians: Number of Gaussians to test with (default 300K).
        depth_h: Depth map height.
        depth_w: Depth map width.
        n_warmup: Warmup iterations.
        n_iters: Timed iterations.
        device: Device to benchmark on.

    Returns:
        Dictionary with timing results per kernel.
    """
    if not torch.cuda.is_available():
        print("CUDA not available; benchmarks require a GPU.")
        return {}

    results = {}
    dev = torch.device(device)

    print(f"\n{'='*70}")
    print(f"Triton Kernel Benchmarks (Triton available: {HAS_TRITON})")
    print(f"  Gaussians: {n_gaussians:,}")
    print(f"  Depth map: {depth_h}x{depth_w}")
    print(f"  Warmup: {n_warmup}, Iterations: {n_iters}")
    print(f"{'='*70}\n")

    # --- 1. Confidence Prune ---
    print("1. Confidence Prune")
    confs = torch.rand(n_gaussians, device=dev)
    threshold = 0.5

    for _ in range(n_warmup):
        _ = confidence_prune(confs, threshold)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = confidence_prune(confs, threshold)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) / n_iters * 1000

    for _ in range(n_warmup):
        _ = confs > threshold
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = confs > threshold
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / n_iters * 1000

    speedup = pytorch_time / triton_time if triton_time > 0 else 0
    results["confidence_prune"] = {
        "triton_ms": triton_time, "pytorch_ms": pytorch_time, "speedup": speedup,
    }
    print(f"   Triton: {triton_time:.4f} ms | PyTorch: {pytorch_time:.4f} ms | Speedup: {speedup:.2f}x")

    # --- 2. SH to RGB (degree 0) ---
    print("2. SH to RGB (degree 0)")
    sh_coeffs = torch.randn(n_gaussians, 16, 3, device=dev)
    directions = torch.randn(n_gaussians, 3, device=dev)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    for _ in range(n_warmup):
        _ = sh_to_rgb(sh_coeffs, directions, sh_degree=0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = sh_to_rgb(sh_coeffs, directions, sh_degree=0)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) / n_iters * 1000

    # PyTorch fallback for SH degree 0
    for _ in range(n_warmup):
        _ = (sh_coeffs[:, 0, :] * _SH_C0 + 0.5).clamp(0.0, 1.0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = (sh_coeffs[:, 0, :] * _SH_C0 + 0.5).clamp(0.0, 1.0)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / n_iters * 1000

    speedup = pytorch_time / triton_time if triton_time > 0 else 0
    results["sh_to_rgb_deg0"] = {
        "triton_ms": triton_time, "pytorch_ms": pytorch_time, "speedup": speedup,
    }
    print(f"   Triton: {triton_time:.4f} ms | PyTorch: {pytorch_time:.4f} ms | Speedup: {speedup:.2f}x")

    # --- 3. Depth Unprojection ---
    print("3. Depth Unprojection")
    depth = torch.rand(depth_h, depth_w, device=dev) * 5.0 + 0.1
    K = torch.tensor([
        [1000.0, 0.0, depth_w / 2.0],
        [0.0, 1000.0, depth_h / 2.0],
        [0.0, 0.0, 1.0],
    ], device=dev)
    K_inv = torch.inverse(K)
    conf = torch.rand(depth_h, depth_w, device=dev)

    for _ in range(n_warmup):
        _ = unproject_depth(depth, K_inv, conf, conf_threshold=0.3)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = unproject_depth(depth, K_inv, conf, conf_threshold=0.3)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) / n_iters * 1000

    results["unproject_depth"] = {
        "triton_ms": triton_time, "depth_size": f"{depth_h}x{depth_w}",
    }
    print(f"   Triton/PyTorch: {triton_time:.4f} ms ({depth_h}x{depth_w})")

    # --- 4. Opacity Entropy ---
    print("4. Opacity Entropy")
    opas = torch.sigmoid(torch.randn(n_gaussians, device=dev))

    for _ in range(n_warmup):
        _ = opacity_entropy(opas)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = opacity_entropy(opas)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) / n_iters * 1000

    # PyTorch fallback timing
    for _ in range(n_warmup):
        o = opas.clamp(1e-6, 1 - 1e-6)
        _ = -(o * torch.log(o) + (1 - o) * torch.log(1 - o)).mean()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        o = opas.clamp(1e-6, 1 - 1e-6)
        _ = -(o * torch.log(o) + (1 - o) * torch.log(1 - o)).mean()
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / n_iters * 1000

    speedup = pytorch_time / triton_time if triton_time > 0 else 0
    results["opacity_entropy"] = {
        "triton_ms": triton_time, "pytorch_ms": pytorch_time, "speedup": speedup,
    }
    print(f"   Triton: {triton_time:.4f} ms | PyTorch: {pytorch_time:.4f} ms | Speedup: {speedup:.2f}x")

    # --- 5. PLY Vertex Packing ---
    print("5. PLY Vertex Packing")
    positions = torch.randn(n_gaussians, 3, device=dev)
    colors = torch.rand(n_gaussians, 3, device=dev)
    normals = torch.randn(n_gaussians, 3, device=dev)

    for _ in range(n_warmup):
        _ = pack_ply_vertices(positions, colors, normals)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = pack_ply_vertices(positions, colors, normals)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) / n_iters * 1000

    # PyTorch fallback
    for _ in range(n_warmup):
        _ = torch.cat([positions, normals, (colors * 255).clamp(0, 255)], dim=-1)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = torch.cat([positions, normals, (colors * 255).clamp(0, 255)], dim=-1)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / n_iters * 1000

    speedup = pytorch_time / triton_time if triton_time > 0 else 0
    results["pack_ply_vertices"] = {
        "triton_ms": triton_time, "pytorch_ms": pytorch_time, "speedup": speedup,
    }
    print(f"   Triton: {triton_time:.4f} ms | PyTorch: {pytorch_time:.4f} ms | Speedup: {speedup:.2f}x")

    print(f"\n{'='*70}")
    print("Benchmark complete.")
    print(f"{'='*70}\n")

    return results


# ===================================================================
# Module self-test
# ===================================================================

if __name__ == "__main__":
    print(f"Triton available: {HAS_TRITON}")

    if torch.cuda.is_available():
        benchmark_kernels()
    else:
        # CPU-only smoke test with PyTorch fallbacks
        print("\nRunning CPU smoke tests (PyTorch fallbacks only)...")

        # Confidence prune
        c = torch.rand(100)
        m = confidence_prune(c, 0.5)
        assert m.shape == (100,) and m.dtype == torch.bool
        print("  confidence_prune: OK")

        # SH to RGB
        sh = torch.randn(50, 16, 3)
        dirs = torch.randn(50, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        rgb = sh_to_rgb(sh, dirs, sh_degree=1)
        assert rgb.shape == (50, 3) and rgb.min() >= 0 and rgb.max() <= 1
        print("  sh_to_rgb: OK")

        # Depth unprojection
        depth = torch.rand(10, 20) * 5.0 + 0.1
        K_inv = torch.eye(3)
        K_inv[0, 0] = 1.0 / 500.0
        K_inv[1, 1] = 1.0 / 500.0
        K_inv[0, 2] = -10.0 / 500.0
        K_inv[1, 2] = -5.0 / 500.0
        pts, valid = unproject_depth(depth, K_inv)
        assert pts.shape == (10, 20, 3) and valid.shape == (10, 20)
        print("  unproject_depth: OK")

        # Opacity entropy
        opas = torch.sigmoid(torch.randn(100))
        ent = opacity_entropy(opas)
        assert ent.shape == () and ent.item() >= 0
        print("  opacity_entropy: OK")

        # PLY packing
        pos = torch.randn(30, 3)
        col = torch.rand(30, 3)
        packed = pack_ply_vertices(pos, col)
        assert packed.shape == (30, 9)
        print("  pack_ply_vertices: OK")

        print("\nAll CPU smoke tests passed.")
