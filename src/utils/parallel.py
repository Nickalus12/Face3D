"""Parallel processing utilities for the Face3D pipeline.

Provides a unified ``parallel_map`` function that wraps
``concurrent.futures`` with tqdm progress bars, error handling, and
sensible worker-count defaults.

Usage guidelines:
    - **CPU-bound** work (numpy, OpenCV, rawpy): use ``use_threads=False``
      (ProcessPoolExecutor — default).
    - **I/O-bound** work (file reads, network): use ``use_threads=True``
      (ThreadPoolExecutor).
    - **MediaPipe / GPU / TFLite**: NOT safe for multiprocessing or
      multithreading.  Keep on the main thread.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Callable, Iterable, TypeVar

from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def get_optimal_workers(task_type: str = "cpu") -> int:
    """Return an optimal worker count based on *task_type* and system resources.

    Args:
        task_type: ``"cpu"`` for CPU-bound work (default) or ``"io"`` for
            I/O-bound work.

    Returns:
        Positive integer worker count.
    """
    cpu_count = os.cpu_count() or 4

    if task_type == "io":
        # I/O-bound: more threads than cores is fine
        return min(cpu_count * 2, 32)

    # CPU-bound: leave one core free for the OS / main thread
    return max(1, cpu_count - 1)


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
    desc: str = "Processing",
    use_threads: bool = False,
) -> list[R]:
    """Map *fn* over *items* in parallel with a tqdm progress bar.

    Results are returned **in the same order** as the input *items*.
    If any call raises an exception, it is logged and ``None`` is stored
    in its place (the batch continues).

    Args:
        fn: Callable that accepts a single item and returns a result.
        items: Iterable of inputs to *fn*.
        max_workers: Number of parallel workers.  Defaults to
            :func:`get_optimal_workers` for the chosen executor type.
        desc: Label shown in the tqdm progress bar.
        use_threads: If ``True``, use :class:`ThreadPoolExecutor` (good
            for I/O-bound work).  Otherwise use
            :class:`ProcessPoolExecutor` (CPU-bound, default).

    Returns:
        List of results in input order.  Failed items are ``None``.
    """
    items_list = list(items)
    if not items_list:
        return []

    # For very small batches, just run sequentially
    if len(items_list) <= 2:
        results: list[R] = []
        for item in tqdm(items_list, desc=desc):
            try:
                results.append(fn(item))
            except Exception as exc:
                logger.warning("%s: item failed: %s", desc, exc)
                results.append(None)  # type: ignore[arg-type]
        return results

    if max_workers is None:
        task_type = "io" if use_threads else "cpu"
        max_workers = get_optimal_workers(task_type)

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # Map future -> original index so we can reassemble in order
    results_ordered: list[R | None] = [None] * len(items_list)

    def _run_with_executor(cls: type) -> list[R | None]:
        results: list[R | None] = [None] * len(items_list)
        with cls(max_workers=max_workers) as executor:
            future_to_idx: dict[Future, int] = {}
            for idx, item in enumerate(items_list):
                fut = executor.submit(fn, item)
                future_to_idx[fut] = idx

            with tqdm(total=len(items_list), desc=desc) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        logger.warning(
                            "%s: item %d failed: %s", desc, idx, exc
                        )
                    pbar.update(1)
        return results

    # ProcessPoolExecutor can't pickle closures/lambdas/nested functions.
    # Auto-fallback to ThreadPoolExecutor when that happens.
    if executor_cls is ProcessPoolExecutor:
        import pickle
        try:
            pickle.dumps(fn)
        except (pickle.PicklingError, AttributeError, TypeError):
            logger.debug(
                "%s: function %s not picklable, falling back to threads",
                desc, fn.__name__,
            )
            executor_cls = ThreadPoolExecutor

    results_ordered = _run_with_executor(executor_cls)

    return results_ordered  # type: ignore[return-value]
