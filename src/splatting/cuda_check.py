"""Early CUDA availability check for gsplat (< 1 second).

Call assert_cuda_available() at pipeline startup to fail fast instead of
waiting until training iteration 1 to discover CUDA is broken.
"""

import logging
import sys

logger = logging.getLogger(__name__)


def check_gsplat_cuda() -> tuple[bool, str | None, str]:
    """Verify gsplat CUDA backend is functional.

    Returns:
        (is_available, error_message, backend_name)
    """
    # Step 1: Check PyTorch CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "PyTorch CUDA is not available (torch.cuda.is_available() == False)", "none"
        device_name = torch.cuda.get_device_name(0)
        logger.info("PyTorch CUDA OK: %s", device_name)
    except Exception as e:
        return False, f"PyTorch CUDA check failed: {e}", "none"

    # Step 2: Check gsplat import
    try:
        import gsplat
        logger.info("gsplat version: %s", getattr(gsplat, "__version__", "unknown"))
    except ImportError as e:
        return False, f"gsplat not installed: {e}", "none"

    # Step 3: Check gsplat CUDA backend (_C module)
    try:
        from gsplat.cuda._wrapper import _make_lazy_cuda_func
        # This triggers lazy CUDA compilation if needed
    except Exception as e:
        return False, f"gsplat CUDA wrapper failed: {e}", "none"

    # Step 4: Check 3DGS rasterization
    has_3dgs = False
    try:
        from gsplat import rasterization
        if rasterization is not None:
            has_3dgs = True
    except Exception:
        pass

    # Step 5: Check 2DGS rasterization
    has_2dgs = False
    try:
        from gsplat.rendering import rasterization_2dgs
        if rasterization_2dgs is not None:
            has_2dgs = True
    except Exception:
        pass

    # Step 6: Smoke test — actually call a gsplat CUDA function
    try:
        from gsplat.cuda._wrapper import _make_lazy_cuda_func
        # Try to access the projection function (the one that fails)
        import gsplat.cuda._wrapper as _wrapper
        _C = getattr(_wrapper, "_C", None)
        if _C is None:
            # Try triggering lazy load
            try:
                # Create minimal tensors for a quick test
                means = torch.randn(10, 3, device="cuda")
                quats = torch.randn(10, 4, device="cuda")
                quats = quats / quats.norm(dim=-1, keepdim=True)
                scales = torch.ones(10, 3, device="cuda") * 0.1
                viewmat = torch.eye(4, device="cuda").unsqueeze(0)
                K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                                 dtype=torch.float32, device="cuda").unsqueeze(0)

                # Quick rasterization test (1 pixel render)
                from gsplat import rasterization
                colors = torch.randn(10, 3, device="cuda")
                opacities = torch.sigmoid(torch.randn(10, device="cuda"))

                rgb, alpha, info = rasterization(
                    means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors,
                    viewmats=viewmat, Ks=K,
                    width=64, height=64,
                )
                logger.info("gsplat CUDA smoke test PASSED (rendered 64x64)")
                has_3dgs = True
            except Exception as e:
                return False, f"gsplat CUDA smoke test failed: {e}", "none"

    except Exception as e:
        return False, f"gsplat CUDA verification failed: {e}", "none"

    if not has_3dgs and not has_2dgs:
        return False, "Neither 3DGS nor 2DGS rasterization is functional", "none"

    backend = []
    if has_3dgs:
        backend.append("3DGS")
    if has_2dgs:
        backend.append("2DGS")
    backend_str = "+".join(backend)

    logger.info("gsplat CUDA check PASSED: %s available", backend_str)
    return True, None, backend_str


def assert_cuda_available():
    """Fail fast if CUDA/gsplat is not available. Call at pipeline startup."""
    is_ok, error, backend = check_gsplat_cuda()
    if not is_ok:
        logger.error(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  CUDA / gsplat NOT AVAILABLE — Training will not work!     ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Error: %-50s  ║\n"
            "║                                                            ║\n"
            "║  Fix checklist:                                            ║\n"
            "║  1. Set CUDA_HOME env var before running pipeline          ║\n"
            "║  2. Add cl.exe (VS 2022) to PATH                          ║\n"
            "║  3. Run: nvcc --version (should show CUDA 12.x)           ║\n"
            "║  4. Run: cl.exe (should show MSVC version)                ║\n"
            "║  5. Try: pip install gsplat --force-reinstall              ║\n"
            "╚══════════════════════════════════════════════════════════════╝",
            error[:50] if error else "Unknown",
        )
        sys.exit(1)
    return backend
