"""Performance benchmarks for Face3D pipeline stages.

Uses ``pytest-benchmark`` with pedantic mode for reproducible timings.
Each benchmark is grouped by pipeline stage for comparison reports.

Run benchmarks::

    python -m pytest tests/test_benchmarks.py -v --benchmark-enable
    python -m pytest tests/test_benchmarks.py -v --benchmark-json=benchmark.json
    python -m pytest tests/test_benchmarks.py -v --benchmark-compare

Benchmark groups:
    capture     -- frame extraction, color correction, filtering
    colmap_io   -- binary model read/write
    depth       -- depth estimation per frame (CUDA)
    splatting   -- Gaussian initialization, training iteration, export

Design:
    - All benchmarks use synthetic data for consistency
    - GPU benchmarks include torch.cuda.synchronize() in teardown
    - pedantic mode used for functions that need warmup
    - Statistical analysis: min/max/mean/stddev/median/IQR reported
"""

import numpy as np
import pytest

# Guard: only run if pytest-benchmark is installed
try:
    import pytest_benchmark  # noqa: F401
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Capture benchmarks
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="capture")
class TestCapturePerformance:
    """Benchmarks for frame extraction, color correction, and filtering."""

    def test_benchmark_frame_extraction(self, benchmark, synthetic_video, tmp_path):
        """Benchmark frame extraction speed from synthetic video."""
        from capture.frame_extractor import extract_frames

        counter = [0]

        def extract():
            counter[0] += 1
            out_dir = tmp_path / f"bench_frames_{counter[0]}"
            return extract_frames(synthetic_video, out_dir, target_fps=5.0, max_frames=10)

        result = benchmark.pedantic(extract, rounds=3, iterations=1, warmup_rounds=1)
        assert len(result) > 0

    def test_benchmark_slog3_to_linear(self, benchmark):
        """Benchmark S-Log3 EOTF on a 640x480 float32 frame."""
        from capture.color_correction import _slog3_to_linear

        frame = np.random.default_rng(42).uniform(0, 1, size=(480, 640, 3)).astype(np.float32)

        result = benchmark(_slog3_to_linear, frame)
        assert result.dtype == np.float32

    def test_benchmark_slog3_lut_8bit(self, benchmark):
        """Benchmark fast-path 8-bit LUT S-Log3 -> sRGB conversion."""
        from capture.color_correction import _slog3_to_srgb_lut_8bit

        frame = np.random.default_rng(42).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

        result = benchmark(_slog3_to_srgb_lut_8bit, frame)
        assert result.dtype == np.uint8

    def test_benchmark_blur_detection(self, benchmark, sample_frames):
        """Benchmark blur detection (Laplacian variance) on a single frame."""
        import cv2
        from capture.frame_filter import check_blur

        frame = cv2.imread(str(sample_frames[0]))

        is_sharp, var = benchmark(check_blur, frame, threshold=100.0)
        assert isinstance(var, float)

    def test_benchmark_exposure_check(self, benchmark, sample_frames):
        """Benchmark exposure histogram analysis."""
        import cv2
        from capture.frame_filter import check_exposure

        frame = cv2.imread(str(sample_frames[0]))

        ok, stats = benchmark(check_exposure, frame)
        assert isinstance(stats, dict)


# ═══════════════════════════════════════════════════════════════════════════
# COLMAP I/O benchmarks
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="colmap_io")
class TestColmapIOPerformance:
    """Benchmarks for COLMAP binary model read speed."""

    def test_benchmark_read_cameras(self, benchmark, sample_colmap_model):
        """Benchmark cameras.bin read speed."""
        from utils.colmap_io import read_cameras_binary

        cameras = benchmark(read_cameras_binary, sample_colmap_model / "cameras.bin")
        assert len(cameras) == 5

    def test_benchmark_read_images(self, benchmark, sample_colmap_model):
        """Benchmark images.bin read speed."""
        from utils.colmap_io import read_images_binary

        images = benchmark(read_images_binary, sample_colmap_model / "images.bin")
        assert len(images) == 5

    def test_benchmark_read_points3d(self, benchmark, sample_colmap_model):
        """Benchmark points3D.bin read speed (100 points)."""
        from utils.colmap_io import read_points3d_binary

        points = benchmark(read_points3d_binary, sample_colmap_model / "points3D.bin")
        assert len(points) == 100


# ═══════════════════════════════════════════════════════════════════════════
# Splatting benchmarks
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="splatting")
class TestSplattingPerformance:
    """Benchmarks for Gaussian initialization and export."""

    def test_benchmark_initialize_from_colmap(self, benchmark, sample_colmap_model):
        """Benchmark Gaussian initialization from COLMAP sparse points."""
        from splatting.initializer import initialize_from_colmap_sparse

        model = benchmark(initialize_from_colmap_sparse, sample_colmap_model)
        assert model.num_gaussians == 100

    def test_benchmark_ply_export(self, benchmark, tmp_path, sample_gaussians):
        """Benchmark PLY export for 50 Gaussians."""
        from splatting.initializer import _save_gaussians_ply

        counter = [0]

        def export():
            counter[0] += 1
            path = tmp_path / f"bench_{counter[0]}.ply"
            _save_gaussians_ply(sample_gaussians, path)
            return path

        result = benchmark(export)
        assert result.exists()

    def test_benchmark_gaussian_clone(self, benchmark, sample_gaussians):
        """Benchmark GaussianModel deep clone speed."""
        clone = benchmark(sample_gaussians.clone)
        assert clone.num_gaussians == sample_gaussians.num_gaussians


# ═══════════════════════════════════════════════════════════════════════════
# Depth estimation benchmarks (CUDA)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="depth")
class TestDepthPerformance:
    """Benchmarks for depth estimation (requires CUDA + model weights)."""

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_benchmark_depth_estimation(self, benchmark, cuda_available, sample_frames):
        """Benchmark DA2/DA3 inference speed per frame."""
        from depth.depth_estimator import DepthEstimator

        try:
            estimator = DepthEstimator(model_name="auto", device="cuda")
        except (RuntimeError, ImportError) as exc:
            pytest.skip(f"Depth model not available: {exc}")

        import torch

        def estimate_with_sync():
            result = estimator.estimate(sample_frames[0])
            torch.cuda.synchronize()
            return result

        result = benchmark.pedantic(estimate_with_sync, rounds=5, iterations=1, warmup_rounds=2)
        assert result.depth.ndim == 2


# ═══════════════════════════════════════════════════════════════════════════
# Sensor processing benchmarks
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="sensors")
class TestSensorPerformance:
    """Benchmarks for IMU processing and orientation estimation."""

    def test_benchmark_madgwick_filter(self, benchmark, sample_imu_data):
        """Benchmark Madgwick filter processing 100 IMU samples."""
        from sensors.orientation import compute_rotations

        rotations = benchmark(compute_rotations, sample_imu_data, sample_rate=100.0)
        assert rotations.shape == (100, 3, 3)

    def test_benchmark_complementary_filter(self, benchmark, sample_imu_data):
        """Benchmark complementary filter processing 100 IMU samples."""
        from sensors.orientation import compute_rotations

        rotations = benchmark(
            compute_rotations, sample_imu_data,
            sample_rate=100.0, use_complementary=True,
        )
        assert rotations.shape == (100, 3, 3)
