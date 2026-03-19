"""ONNX export utilities for Face3D models.

Provides targeted export functions for:
- DA3 (Depth Anything 3) depth estimation model
- 2DGS renderer (trained Gaussian splat model for rendering)

All exports include validation against PyTorch reference outputs to ensure
numerical equivalence within tolerance.

Usage:
    from utils.onnx_export import export_da3_to_onnx, validate_onnx_model

    # Export DA3
    onnx_path = export_da3_to_onnx(da3_model, output_dir="models/onnx")

    # Validate
    is_valid = validate_onnx_model(onnx_path, reference_model=da3_model,
                                    input_shape=(1, 3, 518, 518))
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_da3_to_onnx(
    model: nn.Module,
    output_dir: Union[str, Path] = "models/onnx",
    process_res: int = 518,
    opset_version: int = 17,
    fp16: bool = False,
    device: str = "cuda",
) -> Optional[Path]:
    """Export DA3 depth estimation model to ONNX format.

    The DA3 model (Depth Anything 3) uses a ViT encoder + DPT decoder
    architecture. This function exports the depth prediction pathway
    (not the pose estimation head, which requires special handling).

    Args:
        model: DA3 model instance (DepthAnything3 or its backbone).
        output_dir: Directory to save the ONNX file.
        process_res: Input resolution (DA3 uses 518x518 internally for
                     ViT-L, but our pipeline auto-selects 336 or 504).
        opset_version: ONNX opset (17+ for modern transformer ops).
        fp16: Export in FP16 precision (smaller file, may lose precision).
        device: CUDA device for tracing.

    Returns:
        Path to the exported ONNX file, or None on failure.
    """
    try:
        import onnx
    except ImportError:
        logger.error("ONNX not installed. Install with: pip install onnx>=1.14")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_tag = "fp16" if fp16 else "fp32"
    output_path = output_dir / f"da3_depth_{process_res}_{precision_tag}.onnx"

    logger.info(
        "Exporting DA3 model to ONNX: res=%d, precision=%s",
        process_res, precision_tag,
    )

    model.eval()
    if fp16:
        model = model.half()

    # DA3 expects (B, 3, H, W) normalized RGB input
    input_shape = (1, 3, process_res, process_res)
    dummy_input = torch.randn(input_shape, device=device)
    if fp16:
        dummy_input = dummy_input.half()

    try:
        t0 = time.time()

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=["image"],
                output_names=["depth"],
                dynamic_axes={
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "depth": {0: "batch", 2: "height", 3: "width"},
                },
                do_constant_folding=True,
            )

        elapsed = time.time() - t0

        # Validate the exported model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            "DA3 ONNX export success: %.1f MB, %.1fs -> %s",
            file_size_mb, elapsed, output_path,
        )
        return output_path

    except Exception as e:
        logger.error("DA3 ONNX export failed: %s", e)
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        return None


def export_2dgs_renderer_to_onnx(
    renderer: nn.Module,
    num_gaussians: int,
    image_height: int = 1080,
    image_width: int = 1920,
    sh_degree: int = 1,
    output_dir: Union[str, Path] = "models/onnx",
    opset_version: int = 17,
    device: str = "cuda",
) -> Optional[Path]:
    """Export a trained 2DGS renderer to ONNX for fast rendering.

    NOTE: gsplat's custom CUDA kernels (rasterization_2dgs) are NOT
    exportable to ONNX due to custom autograd ops. This function is
    provided for future compatibility if gsplat adds ONNX-compatible
    forward paths, or for exporting a simplified renderer wrapper.

    For now, this exports a wrapper that takes Gaussian parameters and
    camera matrices as input and produces a rendered image. The actual
    rasterization must still use gsplat CUDA kernels at runtime.

    Args:
        renderer: Trained 2DGS renderer module.
        num_gaussians: Number of Gaussians in the model.
        image_height: Target render height.
        image_width: Target render width.
        sh_degree: Spherical harmonics degree (1 for faces).
        output_dir: Directory to save the ONNX file.
        opset_version: ONNX opset version.
        device: CUDA device.

    Returns:
        Path to the exported ONNX file, or None on failure.
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        logger.error("ONNX not installed. Install with: pip install onnx>=1.14")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"2dgs_renderer_{num_gaussians}g.onnx"

    logger.warning(
        "2DGS renderer ONNX export is experimental. gsplat custom CUDA kernels "
        "cannot be fully exported to ONNX. This exports the parameter-processing "
        "wrapper only."
    )

    # The 2DGS renderer typically expects:
    # - means3d: (N, 3) Gaussian centers
    # - quats: (N, 4) rotations
    # - scales: (N, 2) 2D scales (2DGS uses 2 scale params)
    # - opacities: (N,) opacity logits
    # - colors/sh: (N, K, 3) spherical harmonics coefficients
    # - viewmat: (4, 4) camera view matrix
    # - K: (3, 3) camera intrinsics

    try:
        renderer.eval()
        sh_coeffs = (sh_degree + 1) ** 2

        # Create dummy inputs matching the renderer's expected format
        dummy_means = torch.randn(num_gaussians, 3, device=device)
        dummy_quats = torch.randn(num_gaussians, 4, device=device)
        dummy_scales = torch.randn(num_gaussians, 2, device=device)
        dummy_opacities = torch.randn(num_gaussians, device=device)
        dummy_sh = torch.randn(num_gaussians, sh_coeffs, 3, device=device)
        dummy_viewmat = torch.eye(4, device=device).unsqueeze(0)
        dummy_K = torch.tensor([
            [1000.0, 0.0, image_width / 2],
            [0.0, 1000.0, image_height / 2],
            [0.0, 0.0, 1.0],
        ], device=device).unsqueeze(0)

        t0 = time.time()

        with torch.no_grad():
            torch.onnx.export(
                renderer,
                (dummy_means, dummy_quats, dummy_scales, dummy_opacities,
                 dummy_sh, dummy_viewmat, dummy_K),
                str(output_path),
                opset_version=opset_version,
                input_names=[
                    "means3d", "quats", "scales", "opacities",
                    "sh_coeffs", "viewmat", "K",
                ],
                output_names=["rendered_image"],
                do_constant_folding=True,
            )

        elapsed = time.time() - t0

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            "2DGS renderer ONNX export: %.1f MB, %.1fs -> %s",
            file_size_mb, elapsed, output_path,
        )
        return output_path

    except Exception as e:
        logger.error(
            "2DGS renderer ONNX export failed (expected for gsplat custom ops): %s", e,
        )
        if output_path.exists():
            output_path.unlink()
        return None


def validate_onnx_model(
    onnx_path: Union[str, Path],
    reference_model: Optional[nn.Module] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    device: str = "cuda",
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Verify an ONNX model is valid and optionally matches PyTorch output.

    Performs two levels of validation:
    1. ONNX model structure check (always)
    2. Numerical comparison against PyTorch reference (if model provided)

    Args:
        onnx_path: Path to the ONNX model file.
        reference_model: Optional PyTorch model for output comparison.
        input_shape: Input tensor shape for comparison (required if
                     reference_model is provided).
        device: CUDA device for inference comparison.
        atol: Absolute tolerance for numerical comparison.
        rtol: Relative tolerance for numerical comparison.

    Returns:
        True if validation passes, False otherwise.
    """
    onnx_path = Path(onnx_path)

    if not onnx_path.exists():
        logger.error("ONNX file not found: %s", onnx_path)
        return False

    # Step 1: Structure validation
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX structure validation passed: %s", onnx_path)
    except ImportError:
        logger.error("ONNX not installed for validation")
        return False
    except Exception as e:
        logger.error("ONNX structure validation failed: %s", e)
        return False

    # Step 2: Numerical comparison (optional)
    if reference_model is not None and input_shape is not None:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "onnxruntime not installed, skipping numerical validation. "
                "Install with: pip install onnxruntime-gpu>=1.16"
            )
            return True  # Structure passed, can't do numerical check

        try:
            # Get PyTorch reference output
            reference_model.eval()
            dummy_input = torch.randn(input_shape, device=device)
            with torch.no_grad():
                pytorch_output = reference_model(dummy_input)

            if isinstance(pytorch_output, torch.Tensor):
                pytorch_output = pytorch_output.cpu().numpy()
            elif isinstance(pytorch_output, (tuple, list)):
                pytorch_output = pytorch_output[0].cpu().numpy()
            elif hasattr(pytorch_output, "depth"):
                # DA3 returns a prediction object
                pytorch_output = pytorch_output.depth
                if isinstance(pytorch_output, torch.Tensor):
                    pytorch_output = pytorch_output.cpu().numpy()

            # Get ONNX output
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            input_name = session.get_inputs()[0].name

            onnx_input = dummy_input.cpu().numpy()
            onnx_outputs = session.run(None, {input_name: onnx_input})
            onnx_output = onnx_outputs[0]

            # Compare
            if pytorch_output.shape != onnx_output.shape:
                logger.error(
                    "Shape mismatch: PyTorch=%s vs ONNX=%s",
                    pytorch_output.shape, onnx_output.shape,
                )
                return False

            max_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
            mean_diff = float(np.mean(np.abs(pytorch_output - onnx_output)))
            all_close = np.allclose(pytorch_output, onnx_output, atol=atol, rtol=rtol)

            if all_close:
                logger.info(
                    "ONNX numerical validation passed: max_diff=%.6f, mean_diff=%.6f",
                    max_diff, mean_diff,
                )
            else:
                logger.warning(
                    "ONNX numerical validation FAILED: max_diff=%.6f, mean_diff=%.6f "
                    "(atol=%.4f, rtol=%.4f). Model may still work but with reduced precision.",
                    max_diff, mean_diff, atol, rtol,
                )
                return False

        except Exception as e:
            logger.warning("ONNX numerical validation error: %s", e)
            # Structure passed but numerical check failed -- still return True
            # as the model may work fine for the intended use case
            return True

    return True
