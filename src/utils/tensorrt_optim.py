"""TensorRT optimization for accelerating model inference.

Provides optional TensorRT compilation for PyTorch models with:
- Automatic ONNX export and TensorRT engine compilation
- FP16/FP32 precision support (FP16 gives ~2x speedup on RTX 3080)
- Engine caching to avoid recompilation across runs
- torch2trt one-liner fallback for simpler models
- Graceful fallback to PyTorch when TensorRT is unavailable

Usage:
    from utils.tensorrt_optim import TensorRTModel, benchmark_inference

    # Wrap any PyTorch model
    trt_model = TensorRTModel(model, input_shape=(1, 3, 518, 518), fp16=True)
    if trt_model.optimize():
        output = trt_model(input_tensor)
    else:
        output = model(input_tensor)  # fallback

    # Benchmark comparison
    results = benchmark_inference(model, input_tensor, trt_model=trt_model)

All functionality is optional -- if TensorRT, ONNX, or torch2trt are not
installed, the module logs warnings and falls back to standard PyTorch.

Target hardware: NVIDIA RTX 3080 (Ampere, compute capability 8.6).
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default cache directory for compiled TensorRT engines
_DEFAULT_CACHE_DIR = Path(os.environ.get(
    "FACE3D_TRT_CACHE",
    Path.home() / ".cache" / "face3d" / "tensorrt_engines",
))


def _check_tensorrt() -> bool:
    """Check if TensorRT Python bindings are installed."""
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torch2trt() -> bool:
    """Check if torch2trt is installed."""
    try:
        import torch2trt  # noqa: F401
        return True
    except ImportError:
        return False


def _check_onnx() -> bool:
    """Check if ONNX and onnxruntime are installed."""
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _check_onnxruntime_gpu() -> bool:
    """Check if onnxruntime-gpu with TensorRT EP is available."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "TensorrtExecutionProvider" in providers
    except ImportError:
        return False


def _model_hash(model: nn.Module, input_shape: Tuple[int, ...], fp16: bool) -> str:
    """Generate a hash to identify a specific model configuration for caching.

    Uses model class name, parameter count, input shape, and precision
    to create a unique cache key. This avoids recompiling TensorRT engines
    when the same model is used across runs.
    """
    parts = [
        model.__class__.__name__,
        str(sum(p.numel() for p in model.parameters())),
        str(input_shape),
        "fp16" if fp16 else "fp32",
    ]
    key = "|".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class TensorRTModel:
    """Wraps a PyTorch model with TensorRT optimization for inference.

    Attempts compilation via multiple backends in order of preference:
    1. Native TensorRT (via ONNX export + trtexec)
    2. onnxruntime with TensorRT Execution Provider
    3. torch2trt (simpler, fewer model compatibility issues)

    If all fail, the wrapper transparently delegates to the original
    PyTorch model so callers never need to handle the fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        fp16: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ):
        """Initialize the TensorRT model wrapper.

        Args:
            model: PyTorch model (nn.Module) to optimize.
            input_shape: Expected input tensor shape (batch, channels, height, width).
            fp16: Use FP16 (half) precision. ~2x speedup on Ampere GPUs.
            cache_dir: Directory to cache compiled TensorRT engines.
                       Defaults to ~/.cache/face3d/tensorrt_engines/.
            device: Target CUDA device string.
        """
        self.model = model
        self.input_shape = input_shape
        self.fp16 = fp16
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._optimized_model = None
        self._backend = None  # "tensorrt", "onnxrt_trt", "torch2trt", or None
        self._onnx_path = None
        self._engine_path = None

        # Compute cache key
        self._cache_key = _model_hash(model, input_shape, fp16)

    @staticmethod
    def is_available() -> bool:
        """Check if any TensorRT optimization backend is available.

        Returns True if at least one of the following is installed:
        - tensorrt (native)
        - onnxruntime-gpu with TensorRT EP
        - torch2trt
        """
        if not torch.cuda.is_available():
            return False
        return _check_tensorrt() or _check_onnxruntime_gpu() or _check_torch2trt()

    def optimize(self) -> bool:
        """Compile model to TensorRT. Returns True on success.

        Tries backends in order: native TensorRT -> onnxruntime TRT EP -> torch2trt.
        Caches compiled engines so subsequent calls load from disk.
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping TensorRT optimization")
            return False

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Try native TensorRT via ONNX
        if _check_tensorrt() and _check_onnx():
            if self._try_native_tensorrt():
                return True

        # Try onnxruntime with TensorRT EP
        if _check_onnxruntime_gpu() and _check_onnx():
            if self._try_onnxrt_tensorrt():
                return True

        # Try torch2trt (simplest)
        if _check_torch2trt():
            if self._try_torch2trt():
                return True

        logger.warning(
            "TensorRT optimization failed (no compatible backend). "
            "Install tensorrt, onnxruntime-gpu, or torch2trt for acceleration. "
            "Falling back to PyTorch inference."
        )
        return False

    def _try_native_tensorrt(self) -> bool:
        """Attempt native TensorRT compilation via ONNX export."""
        try:
            import tensorrt as trt

            engine_path = self.cache_dir / f"{self._cache_key}.trt"

            # Check cache
            if engine_path.exists():
                logger.info("Loading cached TensorRT engine: %s", engine_path)
                self._optimized_model = _load_trt_engine(engine_path, self.device)
                if self._optimized_model is not None:
                    self._backend = "tensorrt"
                    self._engine_path = engine_path
                    logger.info("TensorRT engine loaded from cache (FP%s)", "16" if self.fp16 else "32")
                    return True

            # Export to ONNX first
            onnx_path = self.cache_dir / f"{self._cache_key}.onnx"
            if not onnx_path.exists():
                logger.info("Exporting model to ONNX: %s", onnx_path)
                success = export_to_onnx(
                    self.model, self.input_shape, onnx_path,
                    device=self.device,
                )
                if not success:
                    return False
            self._onnx_path = onnx_path

            # Build TensorRT engine
            logger.info("Building TensorRT engine (FP%s)...", "16" if self.fp16 else "32")
            t0 = time.time()

            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, trt_logger)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error("ONNX parse error: %s", parser.get_error(i))
                    return False

            config = builder.create_builder_config()
            # Reserve up to 4GB workspace for TRT (RTX 3080 has 16GB)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

            if self.fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 enabled (platform supports fast FP16)")

            # Build the engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error("TensorRT engine build failed")
                return False

            # Save to cache
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            elapsed = time.time() - t0
            logger.info("TensorRT engine built in %.1fs -> %s", elapsed, engine_path)

            # Load the engine
            self._optimized_model = _load_trt_engine(engine_path, self.device)
            if self._optimized_model is not None:
                self._backend = "tensorrt"
                self._engine_path = engine_path
                return True

        except Exception as e:
            logger.warning("Native TensorRT compilation failed: %s", e)

        return False

    def _try_onnxrt_tensorrt(self) -> bool:
        """Attempt optimization via onnxruntime TensorRT Execution Provider."""
        try:
            import onnxruntime as ort

            # Export to ONNX if not already done
            onnx_path = self.cache_dir / f"{self._cache_key}.onnx"
            if not onnx_path.exists():
                logger.info("Exporting model to ONNX for onnxruntime: %s", onnx_path)
                success = export_to_onnx(
                    self.model, self.input_shape, onnx_path,
                    device=self.device,
                )
                if not success:
                    return False
            self._onnx_path = onnx_path

            # Create session with TensorRT EP
            trt_cache = str(self.cache_dir / "ort_trt_cache")
            os.makedirs(trt_cache, exist_ok=True)

            providers = [
                ("TensorrtExecutionProvider", {
                    "device_id": 0,
                    "trt_max_workspace_size": 4 << 30,
                    "trt_fp16_enable": self.fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache,
                }),
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                }),
            ]

            logger.info("Creating onnxruntime session with TensorRT EP...")
            t0 = time.time()
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            elapsed = time.time() - t0

            # Verify TRT EP is actually being used
            active_providers = session.get_providers()
            if "TensorrtExecutionProvider" in active_providers:
                self._optimized_model = _OnnxRuntimeWrapper(session, self.device)
                self._backend = "onnxrt_trt"
                logger.info(
                    "onnxruntime TensorRT EP ready in %.1fs (FP%s)",
                    elapsed, "16" if self.fp16 else "32",
                )
                return True
            else:
                logger.warning(
                    "TensorRT EP requested but not active. Active: %s",
                    active_providers,
                )
                return False

        except Exception as e:
            logger.warning("onnxruntime TensorRT EP failed: %s", e)

        return False

    def _try_torch2trt(self) -> bool:
        """Attempt optimization via torch2trt (simplest approach)."""
        try:
            from torch2trt import torch2trt

            logger.info("Compiling with torch2trt (FP%s)...", "16" if self.fp16 else "32")

            # Create sample input
            dummy = torch.randn(self.input_shape, device=self.device)
            if self.fp16:
                dummy = dummy.half()
                model_fp16 = self.model.half()
            else:
                model_fp16 = self.model

            t0 = time.time()
            model_trt = torch2trt(
                model_fp16,
                [dummy],
                fp16_mode=self.fp16,
                max_workspace_size=4 << 30,
            )
            elapsed = time.time() - t0

            self._optimized_model = model_trt
            self._backend = "torch2trt"
            logger.info("torch2trt compilation done in %.1fs", elapsed)
            return True

        except Exception as e:
            logger.warning("torch2trt compilation failed: %s", e)

        return False

    def __call__(self, *args, **kwargs) -> Any:
        """Run inference through the optimized engine, or fall back to PyTorch.

        Accepts the same arguments as the original model's forward() method.
        """
        if self._optimized_model is not None:
            return self._optimized_model(*args, **kwargs)
        return self.model(*args, **kwargs)

    @property
    def backend(self) -> Optional[str]:
        """Return the active optimization backend name, or None."""
        return self._backend

    @property
    def is_optimized(self) -> bool:
        """Return True if the model is running through an optimized backend."""
        return self._optimized_model is not None

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic information about the optimization state."""
        return {
            "backend": self._backend,
            "is_optimized": self.is_optimized,
            "fp16": self.fp16,
            "input_shape": self.input_shape,
            "cache_key": self._cache_key,
            "cache_dir": str(self.cache_dir),
            "onnx_path": str(self._onnx_path) if self._onnx_path else None,
            "engine_path": str(self._engine_path) if self._engine_path else None,
        }


class _OnnxRuntimeWrapper:
    """Wrapper to make an onnxruntime session callable like a PyTorch model."""

    def __init__(self, session, device: str = "cuda"):
        self.session = session
        self.device = device
        self._input_names = [inp.name for inp in session.get_inputs()]
        self._output_names = [out.name for out in session.get_outputs()]

    def __call__(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Run inference. Accepts torch tensors, returns torch tensors."""
        # Convert input tensors to numpy
        np_inputs = {}
        for i, arg in enumerate(args):
            name = self._input_names[i] if i < len(self._input_names) else f"input_{i}"
            if isinstance(arg, torch.Tensor):
                np_inputs[name] = arg.detach().cpu().numpy()
            else:
                np_inputs[name] = np.asarray(arg)

        # Run inference
        outputs = self.session.run(self._output_names, np_inputs)

        # Convert outputs back to torch tensors
        torch_outputs = []
        for out in outputs:
            torch_outputs.append(torch.from_numpy(out).to(self.device))

        if len(torch_outputs) == 1:
            return torch_outputs[0]
        return tuple(torch_outputs)


def _load_trt_engine(engine_path: Path, device: str = "cuda") -> Optional[Any]:
    """Load a serialized TensorRT engine and create an execution context."""
    try:
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            logger.error("Failed to deserialize TensorRT engine from %s", engine_path)
            return None

        context = engine.create_execution_context()
        return _TRTEngineWrapper(engine, context, device)

    except Exception as e:
        logger.error("Failed to load TensorRT engine: %s", e)
        return None


class _TRTEngineWrapper:
    """Wrapper to make a native TensorRT engine callable like a PyTorch model."""

    def __init__(self, engine, context, device: str = "cuda"):
        self.engine = engine
        self.context = context
        self.device = device

    def __call__(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Run inference through the TensorRT engine.

        Accepts torch tensors on CUDA and returns torch tensors on CUDA.
        Uses CUDA memory directly (zero-copy when possible).
        """
        import tensorrt as trt

        stream = torch.cuda.current_stream()

        # Set up input bindings
        bindings = []
        outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                if i < len(args) and isinstance(args[i], torch.Tensor):
                    tensor = args[i].contiguous()
                    self.context.set_tensor_address(name, tensor.data_ptr())
                    bindings.append(tensor)
            else:
                # Allocate output tensor
                shape = self.context.get_tensor_shape(name)
                dtype_trt = self.engine.get_tensor_dtype(name)
                dtype_map = {
                    trt.float32: torch.float32,
                    trt.float16: torch.float16,
                    trt.int32: torch.int32,
                    trt.int8: torch.int8,
                }
                dtype_torch = dtype_map.get(dtype_trt, torch.float32)
                out_tensor = torch.empty(
                    tuple(shape), dtype=dtype_torch, device=self.device,
                )
                self.context.set_tensor_address(name, out_tensor.data_ptr())
                outputs.append(out_tensor)

        # Execute
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: Union[str, Path],
    opset_version: int = 17,
    device: str = "cuda",
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> bool:
    """Export PyTorch model to ONNX format for TensorRT compilation.

    Args:
        model: PyTorch model (nn.Module).
        input_shape: Input tensor shape (batch, channels, height, width).
        output_path: Path to save the .onnx file.
        opset_version: ONNX opset version (17 recommended for modern ops).
        device: Device to run the trace on.
        dynamic_axes: Optional dynamic axis specification for variable
            batch size / resolution. E.g. {"input": {0: "batch"}}.

    Returns:
        True if export succeeded, False otherwise.
    """
    try:
        import onnx  # noqa: F401

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_input = torch.randn(input_shape, device=device)

        # Default dynamic axes: batch dimension
        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

        logger.info(
            "Exporting to ONNX: shape=%s, opset=%d, path=%s",
            input_shape, opset_version, output_path,
        )

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )

        # Validate the exported model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("ONNX export success: %.1f MB -> %s", file_size_mb, output_path)
        return True

    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# torch2trt one-liner
# ---------------------------------------------------------------------------

def optimize_with_torch2trt(
    model: nn.Module,
    input_data: torch.Tensor,
    fp16: bool = True,
    max_workspace_size: int = 4 << 30,
) -> nn.Module:
    """Use torch2trt for one-line TensorRT optimization.

    This is the simplest approach -- it traces the model with sample input
    and compiles to TensorRT in one call. Works well for models with
    straightforward forward passes (no dynamic control flow).

    Args:
        model: PyTorch model on CUDA.
        input_data: Sample input tensor on CUDA.
        fp16: Enable FP16 mode for ~2x speedup.
        max_workspace_size: TRT workspace in bytes (default 4GB).

    Returns:
        TensorRT-optimized model if successful, original model otherwise.
    """
    try:
        from torch2trt import torch2trt

        logger.info(
            "torch2trt optimization: input=%s, fp16=%s",
            tuple(input_data.shape), fp16,
        )
        t0 = time.time()

        if fp16:
            model = model.half()
            input_data = input_data.half()

        model_trt = torch2trt(
            model,
            [input_data],
            fp16_mode=fp16,
            max_workspace_size=max_workspace_size,
        )

        elapsed = time.time() - t0
        logger.info("torch2trt optimization complete in %.1fs", elapsed)
        return model_trt

    except ImportError:
        logger.warning(
            "torch2trt not installed. Install with: "
            "pip install torch2trt  (requires TensorRT)"
        )
        return model
    except Exception as e:
        logger.warning("torch2trt optimization failed: %s. Using PyTorch inference.", e)
        return model


# ---------------------------------------------------------------------------
# Inference benchmarking
# ---------------------------------------------------------------------------

def benchmark_inference(
    model: nn.Module,
    input_data: torch.Tensor,
    trt_model: Optional[TensorRTModel] = None,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, Any]:
    """Benchmark model inference speed, comparing PyTorch vs TensorRT.

    Runs warmup iterations to stabilize GPU clocks, then measures
    average latency and throughput over the specified iterations.

    Args:
        model: Original PyTorch model.
        input_data: Input tensor on CUDA.
        trt_model: Optional TensorRTModel wrapper to benchmark.
        warmup: Number of warmup iterations (not timed).
        iterations: Number of timed iterations.

    Returns:
        Dict with timing results:
        {
            "pytorch_ms": float,         # Mean PyTorch latency in ms
            "pytorch_fps": float,         # PyTorch throughput (frames/sec)
            "tensorrt_ms": float | None,  # Mean TRT latency (if available)
            "tensorrt_fps": float | None, # TRT throughput (if available)
            "speedup": float | None,      # TRT speedup factor over PyTorch
            "backend": str | None,        # TRT backend used
        }
    """
    results: Dict[str, Any] = {}

    # Benchmark PyTorch
    logger.info("Benchmarking PyTorch inference (%d warmup, %d iterations)...", warmup, iterations)
    pytorch_ms = _time_model(model, input_data, warmup, iterations)
    results["pytorch_ms"] = pytorch_ms
    results["pytorch_fps"] = 1000.0 / pytorch_ms if pytorch_ms > 0 else 0.0
    logger.info("PyTorch: %.2f ms/frame (%.1f FPS)", pytorch_ms, results["pytorch_fps"])

    # Benchmark TensorRT
    if trt_model is not None and trt_model.is_optimized:
        logger.info("Benchmarking TensorRT inference (%s backend)...", trt_model.backend)
        trt_ms = _time_model(trt_model, input_data, warmup, iterations)
        results["tensorrt_ms"] = trt_ms
        results["tensorrt_fps"] = 1000.0 / trt_ms if trt_ms > 0 else 0.0
        results["speedup"] = pytorch_ms / trt_ms if trt_ms > 0 else 0.0
        results["backend"] = trt_model.backend
        logger.info(
            "TensorRT (%s): %.2f ms/frame (%.1f FPS) -- %.2fx speedup",
            trt_model.backend, trt_ms, results["tensorrt_fps"], results["speedup"],
        )
    else:
        results["tensorrt_ms"] = None
        results["tensorrt_fps"] = None
        results["speedup"] = None
        results["backend"] = None
        logger.info("TensorRT model not available for benchmarking")

    return results


def _time_model(
    model: Any,
    input_data: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    """Time a model's inference and return mean latency in milliseconds."""
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(input_data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(input_data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000.0  # Convert to ms


# ---------------------------------------------------------------------------
# Convenience: check what's available
# ---------------------------------------------------------------------------

def get_optimization_status() -> Dict[str, bool]:
    """Return availability of all optimization backends.

    Useful for diagnostics and UI display.
    """
    status = {
        "cuda": torch.cuda.is_available(),
        "tensorrt": _check_tensorrt(),
        "onnx": _check_onnx(),
        "onnxruntime_gpu": _check_onnxruntime_gpu(),
        "torch2trt": _check_torch2trt(),
        "any_trt_backend": False,
    }
    status["any_trt_backend"] = (
        status["tensorrt"] or status["onnxruntime_gpu"] or status["torch2trt"]
    )

    if torch.cuda.is_available():
        status["gpu_name"] = torch.cuda.get_device_name(0)
        status["gpu_compute_capability"] = ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        )
    else:
        status["gpu_name"] = None
        status["gpu_compute_capability"] = None

    return status
