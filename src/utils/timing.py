"""Performance timing and profiling utilities for the Face3D pipeline.

Provides:
    - ``@timed`` decorator for logging function execution time
    - ``StageTimer`` context manager for timing pipeline stages
    - ``PipelineProfiler`` class for accumulating timing across all stages
    - ``gpu_memory_tracker`` context manager for GPU memory monitoring

All timing uses ``time.perf_counter()`` for sub-millisecond precision.
GPU memory tracking gracefully handles environments without CUDA.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU memory helpers (lazy CUDA detection)
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Check CUDA availability without importing torch at module level."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_gpu_memory_stats() -> dict[str, float]:
    """Return current GPU memory stats in GB.  Returns empty dict if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
    except (ImportError, RuntimeError):
        return {}


def _get_cpu_memory_mb() -> float:
    """Return current process RSS in MB.  Returns 0.0 if psutil unavailable."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    except (ImportError, OSError):
        return 0.0


# ---------------------------------------------------------------------------
# @timed decorator
# ---------------------------------------------------------------------------

def timed(fn: Callable | None = None, *, name: str | None = None):
    """Decorator that logs function execution time.

    Usage::

        @timed
        def extract_frames(...):
            ...

        @timed(name="custom label")
        def my_func(...):
            ...

    Overhead is negligible (<1ms per call).
    """
    def decorator(func: Callable) -> Callable:
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                logger.info("%s completed in %.1fs", label, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - t0
                logger.info("%s failed after %.1fs", label, elapsed)
                raise

        return wrapper

    # Support both @timed and @timed(name="...")
    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# StageTimer context manager
# ---------------------------------------------------------------------------

class StageTimer:
    """Context manager for timing pipeline stages.

    Usage::

        with StageTimer("Stage 6: DA3 Unified") as timer:
            run_da3_unified(...)
        print(timer.elapsed)  # seconds as float
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
        self._gpu_start_max: float = 0.0

    def __enter__(self) -> StageTimer:
        # Reset GPU peak counter if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._gpu_start_max = torch.cuda.max_memory_allocated()
        except (ImportError, RuntimeError):
            pass

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.start_time

        gpu_info = ""
        try:
            import torch
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                gpu_info = f", GPU peak {peak_gb:.1f}GB"
        except (ImportError, RuntimeError):
            pass

        if exc_type is not None:
            logger.info("%s failed after %.1fs%s", self.name, self.elapsed, gpu_info)
        else:
            logger.info("%s completed in %.1fs%s", self.name, self.elapsed, gpu_info)

        return None  # don't suppress exceptions


# ---------------------------------------------------------------------------
# gpu_memory_tracker context manager
# ---------------------------------------------------------------------------

@contextmanager
def gpu_memory_tracker(label: str = "operation"):
    """Track GPU memory usage within a block.

    Usage::

        with gpu_memory_tracker("DA3 inference") as mem:
            ...
        # Logs: "DA3 inference: peak GPU 2.1GB, allocated 1.8GB"
        # mem["peak_gb"], mem["allocated_gb"] available after

    Gracefully handles no-GPU environments (returns zeroes).
    """
    result: dict[str, float] = {"peak_gb": 0.0, "allocated_gb": 0.0, "reserved_gb": 0.0}

    has_cuda = False
    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass

    try:
        yield result
    finally:
        if has_cuda:
            try:
                import torch
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                result["peak_gb"] = peak
                result["allocated_gb"] = allocated
                result["reserved_gb"] = reserved
                logger.info(
                    "%s: peak GPU %.1fGB, allocated %.1fGB",
                    label, peak, allocated,
                )
            except (ImportError, RuntimeError):
                pass


# ---------------------------------------------------------------------------
# PipelineProfiler
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """Timing record for a single pipeline stage."""
    name: str
    stage_num: int = -1
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    gpu_peak_gb: float = 0.0
    cpu_peak_mb: float = 0.0
    num_items: int = 0
    throughput: float = 0.0  # items/sec
    skipped: bool = False
    error: str | None = None


class PipelineProfiler:
    """Accumulates timing data across all pipeline stages.

    Usage::

        profiler = PipelineProfiler(session_id="natalie")
        profiler.start_stage("capture", stage_num=1)
        # ... do work ...
        profiler.end_stage("capture", num_items=80)
        profiler.summary()       # Pretty-printed table
        profiler.save_json(path) # Persist to disk
    """

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self.pipeline_start: float = time.perf_counter()
        self.records: dict[str, StageRecord] = {}
        self._active_stage: str | None = None

    def start_stage(self, name: str, stage_num: int = -1) -> None:
        """Mark the beginning of a stage."""
        record = StageRecord(name=name, stage_num=stage_num)
        record.start_time = time.perf_counter()
        record.cpu_peak_mb = _get_cpu_memory_mb()

        # Reset GPU peak memory counter
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass

        self.records[name] = record
        self._active_stage = name

    def end_stage(
        self,
        name: str,
        num_items: int = 0,
        skipped: bool = False,
        error: str | None = None,
    ) -> StageRecord:
        """Mark the end of a stage and compute derived metrics."""
        record = self.records.get(name)
        if record is None:
            # Stage was never started (e.g., skipped entirely)
            record = StageRecord(name=name, skipped=True)
            self.records[name] = record
            return record

        record.end_time = time.perf_counter()
        record.duration = record.end_time - record.start_time
        record.num_items = num_items
        record.skipped = skipped
        record.error = error

        if num_items > 0 and record.duration > 0:
            record.throughput = num_items / record.duration

        # Capture GPU peak memory
        try:
            import torch
            if torch.cuda.is_available():
                record.gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except (ImportError, RuntimeError):
            pass

        # Capture CPU memory (take max of start vs end)
        cpu_now = _get_cpu_memory_mb()
        record.cpu_peak_mb = max(record.cpu_peak_mb, cpu_now)

        self._active_stage = None
        return record

    def mark_skipped(self, name: str, stage_num: int = -1) -> None:
        """Mark a stage as skipped (already complete)."""
        record = StageRecord(name=name, stage_num=stage_num, skipped=True)
        self.records[name] = record

    @property
    def total_elapsed(self) -> float:
        """Total wall time since the profiler was created."""
        return time.perf_counter() - self.pipeline_start

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 0.1:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"

    def summary(self) -> str:
        """Generate a pretty-printed summary table and log it.

        Returns the formatted string for optional further use.
        """
        # Sort records by stage_num, then by insertion order for unnumbered
        sorted_records = sorted(
            self.records.values(),
            key=lambda r: (r.stage_num if r.stage_num >= 0 else 999, r.name),
        )

        if not sorted_records:
            msg = "No stages recorded."
            logger.info(msg)
            return msg

        # Column widths
        name_w = max(len(r.name) for r in sorted_records)
        name_w = max(name_w, 20)  # minimum width

        lines = []
        sep = "+" + "-" * (name_w + 2) + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 11 + "+"
        hdr = (
            f"| {'Stage':<{name_w}} | {'Duration':>10} | {'GPU Peak':>10} | {'CPU Peak':>10} | {'Items/s':>9} |"
        )

        lines.append("")
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)

        total_duration = 0.0
        max_gpu = 0.0

        for r in sorted_records:
            if r.skipped:
                dur_str = "skipped"
                gpu_str = "-"
                cpu_str = "-"
                tp_str = "-"
            elif r.error:
                dur_str = "FAILED"
                gpu_str = "-"
                cpu_str = "-"
                tp_str = "-"
            else:
                dur_str = self._format_duration(r.duration)
                total_duration += r.duration
                gpu_str = f"{r.gpu_peak_gb:.1f}GB" if r.gpu_peak_gb > 0.01 else "-"
                cpu_str = f"{r.cpu_peak_mb:.0f}MB" if r.cpu_peak_mb > 0 else "-"
                tp_str = f"{r.throughput:.1f}/s" if r.throughput > 0 else "-"
                max_gpu = max(max_gpu, r.gpu_peak_gb)

            # Build stage label with number prefix
            if r.stage_num >= 0:
                label = f"{r.stage_num:>2}. {r.name}"
            else:
                label = f"    {r.name}"

            lines.append(
                f"| {label:<{name_w}} | {dur_str:>10} | {gpu_str:>10} | {cpu_str:>10} | {tp_str:>9} |"
            )

        lines.append(sep)

        # Total line
        total_str = self._format_duration(total_duration)
        wall_str = self._format_duration(self.total_elapsed)
        lines.append(
            f"  Total stage time: {total_str}  |  Wall time: {wall_str}"
        )
        if max_gpu > 0.01:
            lines.append(f"  Peak GPU memory across all stages: {max_gpu:.1f}GB")
        lines.append("")

        output = "\n".join(lines)
        logger.info(output)
        return output

    def to_dict(self) -> dict[str, Any]:
        """Serialize profiling data to a JSON-compatible dict."""
        records_list = []
        for r in sorted(
            self.records.values(),
            key=lambda r: (r.stage_num if r.stage_num >= 0 else 999, r.name),
        ):
            records_list.append({
                "name": r.name,
                "stage_num": r.stage_num,
                "duration_s": round(r.duration, 2),
                "gpu_peak_gb": round(r.gpu_peak_gb, 2),
                "cpu_peak_mb": round(r.cpu_peak_mb, 1),
                "num_items": r.num_items,
                "throughput_per_s": round(r.throughput, 2),
                "skipped": r.skipped,
                "error": r.error,
            })

        return {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_wall_time_s": round(self.total_elapsed, 2),
            "total_stage_time_s": round(
                sum(r.duration for r in self.records.values() if not r.skipped), 2
            ),
            "stages": records_list,
        }

    def save_json(self, path: Path | str) -> None:
        """Persist profiling data as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Pipeline profiling data saved to %s", path)
