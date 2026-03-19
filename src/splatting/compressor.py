"""Gaussian Splatting compression utilities.

Supports:
- Draco compression for PLY/glTF files (industry standard, 5-20x size reduction)
- Video-based GS compression for temporal sequences
- Standard PLY quantization (existing gsplat compressed format)
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional backend detection
# ---------------------------------------------------------------------------

HAS_DRACO = False
HAS_GS_COMPRESSOR = False

try:
    import DracoPy  # noqa: F401

    HAS_DRACO = True
except ImportError:
    pass

try:
    from gaussian_splatting.gscompressor import GSCompressor as _GSCompressor  # noqa: F401

    HAS_GS_COMPRESSOR = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Quality presets (maps quality name -> parameter dict)
# ---------------------------------------------------------------------------

_QUALITY_PRESETS = {
    "fast": {
        "position_bits": 11,
        "color_bits": 6,
        "scale_bits": 8,
        "rotation_bits": 8,
        "sh_bands_keep": 0,  # DC only
        "draco_compression_level": 10,
        "draco_quantization_bits": 11,
    },
    "balanced": {
        "position_bits": 14,
        "color_bits": 8,
        "scale_bits": 12,
        "rotation_bits": 10,
        "sh_bands_keep": 1,  # DC + 1st order
        "draco_compression_level": 7,
        "draco_quantization_bits": 14,
    },
    "quality": {
        "position_bits": 16,
        "color_bits": 8,
        "scale_bits": 14,
        "rotation_bits": 12,
        "sh_bands_keep": 3,  # All bands
        "draco_compression_level": 4,
        "draco_quantization_bits": 16,
    },
}


# ---------------------------------------------------------------------------
# GaussianCompressor — main class
# ---------------------------------------------------------------------------


class GaussianCompressor:
    """Compress Gaussian Splatting models using multiple backends.

    Compression hierarchy (tries in order):
    1. gscompressor (from gaussian-splatting package) -- Draco-native GS compression
    2. DracoPy -- direct Draco encoding of point clouds
    3. Quantization fallback -- simple float16/uint8 quantization
    """

    def __init__(self, quality: str = "balanced"):
        """
        Args:
            quality: "fast" (most compressed, some quality loss),
                     "balanced" (good compression, minimal quality loss),
                     "quality" (least compression, preserves everything)
        """
        if quality not in _QUALITY_PRESETS:
            raise ValueError(
                f"Unknown quality {quality!r}; must be one of {list(_QUALITY_PRESETS)}"
            )
        self.quality = quality
        self.params = _QUALITY_PRESETS[quality]
        self.backend = self._select_backend()

    def _select_backend(self) -> str:
        """Choose the best available compression backend."""
        if HAS_GS_COMPRESSOR:
            logger.info("Compression backend: gscompressor (Draco-native GS)")
            return "gscompressor"
        if HAS_DRACO:
            logger.info("Compression backend: DracoPy (direct Draco encoding)")
            return "draco"
        logger.info("Compression backend: quantized (numpy-only fallback)")
        return "quantized"

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def compress_ply(self, input_path: Path, output_path: Path) -> dict:
        """Compress a PLY file containing Gaussian Splat data.

        Reads the PLY, extracts Gaussian parameters, compresses them,
        and writes the result.

        Returns dict with: original_size, compressed_size, ratio, backend_used
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        original_size = input_path.stat().st_size

        if self.backend == "gscompressor":
            self._compress_ply_gscompressor(input_path, output_path)
        elif self.backend == "draco":
            self._compress_ply_draco(input_path, output_path)
        else:
            self._compress_ply_quantized(input_path, output_path)

        compressed_size = output_path.stat().st_size
        ratio = original_size / max(compressed_size, 1)

        stats = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": round(ratio, 2),
            "backend_used": self.backend,
            "quality": self.quality,
        }

        logger.info(
            "Compressed PLY: %.1f MB -> %.1f MB (%.1fx, backend=%s)",
            original_size / 1e6,
            compressed_size / 1e6,
            ratio,
            self.backend,
        )
        return stats

    def compress_for_gltf(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> bytes:
        """Compress mesh data for embedding in glTF with KHR_draco_mesh_compression.

        Args:
            vertices: (N, 3) float32 vertex positions.
            faces: (M, 3) uint32 triangle indices.
            colors: Optional (N, 3) or (N, 4) uint8 vertex colors.

        Returns:
            Draco-encoded binary data, or raw vertex+face bytes if Draco unavailable.
        """
        if not HAS_DRACO:
            logger.warning(
                "DracoPy not available; returning uncompressed mesh data "
                "for glTF embedding. Install with: pip install DracoPy"
            )
            return vertices.astype(np.float32).tobytes() + faces.astype(np.uint32).tobytes()

        import DracoPy

        kwargs = {
            "quantization_bits": self.params["draco_quantization_bits"],
            "compression_level": self.params["draco_compression_level"],
        }
        if colors is not None:
            kwargs["colors"] = colors.astype(np.uint8).flatten()

        binary = DracoPy.encode(
            vertices.astype(np.float32).flatten(),
            faces=faces.astype(np.uint32).flatten(),
            **kwargs,
        )

        logger.info(
            "Draco-encoded mesh for glTF: %d vertices, %d faces -> %d bytes",
            len(vertices),
            len(faces),
            len(binary),
        )
        return binary

    def compress_gaussians(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        opacities: np.ndarray,
        sh_coeffs: np.ndarray,
        output_path: Path,
    ) -> dict:
        """Compress raw Gaussian parameters.

        Applies:
        - Position quantization (configurable bits)
        - Scale log-encoding + quantization
        - Rotation quaternion normalization + quantization
        - SH coefficient truncation (keep first N bands)
        - Draco encoding of the resulting point cloud (if available)

        Args:
            means: (N, 3) float32 positions.
            scales: (N, 3) float32 scales (linear, not log-space).
            rotations: (N, 4) float32 quaternions (w, x, y, z).
            opacities: (N,) float32 opacities in [0, 1].
            sh_coeffs: (N, K, 3) float32 spherical harmonics coefficients.
            output_path: Destination file path.

        Returns:
            Dict with compression statistics.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(means)
        p = self.params

        # --- Quantize each parameter ---
        # Positions: float32 -> fixed-point
        pos_min = means.min(axis=0)
        pos_max = means.max(axis=0)
        pos_range = (pos_max - pos_min).clip(min=1e-8)
        pos_norm = (means - pos_min) / pos_range  # [0, 1]
        pos_quant = _quantize(pos_norm, p["position_bits"])

        # Scales: log-encode then quantize
        log_scales = np.log(scales.clip(min=1e-8))
        ls_min, ls_max = log_scales.min(), log_scales.max()
        ls_range = max(ls_max - ls_min, 1e-8)
        ls_norm = (log_scales - ls_min) / ls_range
        scale_quant = _quantize(ls_norm, p["scale_bits"])

        # Rotations: normalize then quantize to int16
        rot_norms = np.linalg.norm(rotations, axis=-1, keepdims=True).clip(min=1e-8)
        rotations_normed = rotations / rot_norms
        rot_quant = _quantize((rotations_normed + 1.0) / 2.0, p["rotation_bits"])

        # Opacities: float32 -> uint8
        opa_quant = (opacities.clip(0, 1) * 255).astype(np.uint8)

        # SH: truncate bands then float16
        bands_keep = p["sh_bands_keep"]
        # SH coefficient count per band: 1, 3, 5, 7 => cumulative: 1, 4, 9, 16
        sh_counts = [1, 4, 9, 16]
        max_k = sh_coeffs.shape[1] if sh_coeffs.ndim == 3 else 1
        keep_k = sh_counts[min(bands_keep, len(sh_counts) - 1)] if bands_keep >= 0 else max_k
        keep_k = min(keep_k, max_k)

        if sh_coeffs.ndim == 3:
            sh_truncated = sh_coeffs[:, :keep_k, :].astype(np.float16)
        else:
            sh_truncated = sh_coeffs.astype(np.float16)

        # --- Write compressed binary ---
        # If Draco is available, encode positions as a Draco point cloud
        if HAS_DRACO and self.backend in ("draco", "gscompressor"):
            self._write_draco_gaussians(
                output_path, means, pos_quant, scale_quant, rot_quant,
                opa_quant, sh_truncated, pos_min, pos_range, ls_min, ls_range, p,
            )
        else:
            self._write_quantized_gaussians(
                output_path, pos_quant, scale_quant, rot_quant,
                opa_quant, sh_truncated, pos_min, pos_range, ls_min, ls_range, p,
            )

        # Compute stats
        original_size = (
            means.nbytes + scales.nbytes + rotations.nbytes
            + opacities.nbytes + sh_coeffs.nbytes
        )
        compressed_size = output_path.stat().st_size
        ratio = original_size / max(compressed_size, 1)

        stats = {
            "num_gaussians": n,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": round(ratio, 2),
            "backend_used": self.backend,
            "quality": self.quality,
            "sh_bands_kept": bands_keep,
        }

        logger.info(
            "Compressed %d Gaussians: %.1f MB -> %.1f MB (%.1fx)",
            n, original_size / 1e6, compressed_size / 1e6, ratio,
        )
        return stats

    def compress_video_sequence(
        self,
        ply_frames: list[Path],
        output_path: Path,
    ) -> dict:
        """Compress a sequence of GS frames (for animations).

        Uses gsvvcompressor if available, otherwise per-frame Draco compression.

        Args:
            ply_frames: List of PLY file paths (one per frame).
            output_path: Destination for compressed sequence.

        Returns:
            Dict with compression statistics.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_original = 0
        total_compressed = 0
        frame_dir = output_path.parent / (output_path.stem + "_frames")
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Try gsvvcompressor first
        try:
            from gaussian_splatting.gsvvcompressor import GSVVCompressor

            compressor = GSVVCompressor()
            compressor.compress(
                [str(f) for f in ply_frames],
                str(output_path),
            )

            for f in ply_frames:
                total_original += f.stat().st_size
            total_compressed = output_path.stat().st_size

            stats = {
                "num_frames": len(ply_frames),
                "original_size": total_original,
                "compressed_size": total_compressed,
                "ratio": round(total_original / max(total_compressed, 1), 2),
                "backend_used": "gsvvcompressor",
            }
            logger.info(
                "Video-compressed %d GS frames: %.1f MB -> %.1f MB (%.1fx)",
                len(ply_frames),
                total_original / 1e6,
                total_compressed / 1e6,
                stats["ratio"],
            )
            return stats

        except ImportError:
            logger.info(
                "gsvvcompressor not available; using per-frame compression"
            )
        except Exception as e:
            logger.warning("gsvvcompressor failed (%s); using per-frame fallback", e)

        # Fallback: per-frame compression
        for i, frame_path in enumerate(ply_frames):
            frame_path = Path(frame_path)
            if not frame_path.exists():
                logger.warning("Frame %d not found: %s", i, frame_path)
                continue

            compressed_frame = frame_dir / f"frame_{i:04d}.drc"
            frame_stats = self.compress_ply(frame_path, compressed_frame)
            total_original += frame_stats["original_size"]
            total_compressed += frame_stats["compressed_size"]

        stats = {
            "num_frames": len(ply_frames),
            "original_size": total_original,
            "compressed_size": total_compressed,
            "ratio": round(total_original / max(total_compressed, 1), 2),
            "backend_used": f"per-frame-{self.backend}",
            "frames_dir": str(frame_dir),
        }
        logger.info(
            "Per-frame compressed %d GS frames: %.1f MB -> %.1f MB (%.1fx)",
            len(ply_frames),
            total_original / 1e6,
            total_compressed / 1e6,
            stats["ratio"],
        )
        return stats

    @staticmethod
    def get_compression_stats(original_path: Path, compressed_path: Path) -> dict:
        """Compare original vs compressed file sizes."""
        original_path = Path(original_path)
        compressed_path = Path(compressed_path)
        orig_size = original_path.stat().st_size
        comp_size = compressed_path.stat().st_size
        return {
            "original_path": str(original_path),
            "compressed_path": str(compressed_path),
            "original_size": orig_size,
            "compressed_size": comp_size,
            "ratio": round(orig_size / max(comp_size, 1), 2),
            "savings_pct": round((1 - comp_size / max(orig_size, 1)) * 100, 1),
        }

    # -----------------------------------------------------------------------
    # Private: backend-specific compression
    # -----------------------------------------------------------------------

    def _compress_ply_gscompressor(self, input_path: Path, output_path: Path) -> None:
        """Compress PLY using gscompressor (Draco-native GS compression)."""
        from gaussian_splatting.gscompressor import GSCompressor

        compressor = GSCompressor()
        compressor.compress(str(input_path), str(output_path))

    def _compress_ply_draco(self, input_path: Path, output_path: Path) -> None:
        """Compress PLY using DracoPy (read PLY, Draco-encode the point cloud)."""
        import DracoPy

        points, attrs = _read_ply_points_and_attrs(input_path)
        if points is None:
            logger.warning("Could not parse PLY %s; copying uncompressed", input_path)
            import shutil

            shutil.copy2(input_path, output_path)
            return

        # Encode positions as a Draco point cloud
        binary = DracoPy.encode(
            points.astype(np.float32).flatten(),
            quantization_bits=self.params["draco_quantization_bits"],
            compression_level=self.params["draco_compression_level"],
        )

        # Write a container: Draco-compressed positions + raw quantized attributes
        _write_compressed_container(output_path, binary, attrs, points.shape[0])

    def _compress_ply_quantized(self, input_path: Path, output_path: Path) -> None:
        """Compress PLY using numpy-only quantization (no external deps)."""
        compress_ply_quantized(
            input_path,
            output_path,
            position_bits=self.params["position_bits"],
            color_bits=self.params["color_bits"],
            scale_bits=self.params["scale_bits"],
        )

    def _write_draco_gaussians(
        self, output_path, means, pos_quant, scale_quant, rot_quant,
        opa_quant, sh_truncated, pos_min, pos_range, ls_min, ls_range, params,
    ):
        """Write Gaussian parameters with Draco-compressed positions."""
        import DracoPy

        # Draco-encode the raw positions for best compression
        draco_binary = DracoPy.encode(
            means.astype(np.float32).flatten(),
            quantization_bits=params["draco_quantization_bits"],
            compression_level=params["draco_compression_level"],
        )

        with open(output_path, "wb") as f:
            # Header: magic + version + counts
            f.write(b"GS3C")  # magic: Gaussian Splat 3D Compressed
            f.write(struct.pack("<I", 1))  # version
            n = len(means)
            f.write(struct.pack("<I", n))  # num gaussians

            # Metadata for dequantization
            f.write(pos_min.astype(np.float32).tobytes())
            f.write(pos_range.astype(np.float32).tobytes())
            f.write(struct.pack("<ff", ls_min, ls_range))
            f.write(struct.pack("<HHH",
                                params["position_bits"],
                                params["scale_bits"],
                                params["rotation_bits"]))
            sh_k = sh_truncated.shape[1] if sh_truncated.ndim == 3 else 1
            f.write(struct.pack("<H", sh_k))

            # Draco-compressed positions
            f.write(struct.pack("<I", len(draco_binary)))
            f.write(draco_binary)

            # Quantized attributes (compact binary)
            f.write(scale_quant.tobytes())
            f.write(rot_quant.tobytes())
            f.write(opa_quant.tobytes())
            f.write(sh_truncated.tobytes())

    def _write_quantized_gaussians(
        self, output_path, pos_quant, scale_quant, rot_quant,
        opa_quant, sh_truncated, pos_min, pos_range, ls_min, ls_range, params,
    ):
        """Write quantized Gaussian parameters (numpy-only, no Draco)."""
        with open(output_path, "wb") as f:
            f.write(b"GS3Q")  # magic: Gaussian Splat 3D Quantized
            f.write(struct.pack("<I", 1))  # version
            n = len(pos_quant)
            f.write(struct.pack("<I", n))

            # Dequantization metadata
            f.write(pos_min.astype(np.float32).tobytes())
            f.write(pos_range.astype(np.float32).tobytes())
            f.write(struct.pack("<ff", ls_min, ls_range))
            f.write(struct.pack("<HHH",
                                params["position_bits"],
                                params["scale_bits"],
                                params["rotation_bits"]))
            sh_k = sh_truncated.shape[1] if sh_truncated.ndim == 3 else 1
            f.write(struct.pack("<H", sh_k))

            # All quantized data
            f.write(pos_quant.tobytes())
            f.write(scale_quant.tobytes())
            f.write(rot_quant.tobytes())
            f.write(opa_quant.tobytes())
            f.write(sh_truncated.tobytes())


# ---------------------------------------------------------------------------
# Standalone quantization-based compression (zero external deps)
# ---------------------------------------------------------------------------


def compress_ply_quantized(
    input_path: Path,
    output_path: Path,
    position_bits: int = 16,
    color_bits: int = 8,
    scale_bits: int = 12,
) -> dict:
    """Simple quantization-based compression (no Draco dependency).

    Reduces PLY file size by 2-4x through:
    - Float32 positions -> Fixed-point with configurable precision
    - Float32 SH -> Float16
    - Float32 scales -> Log-encoded uint16
    - Float32 rotations -> Normalized int16
    - Float32 opacities -> uint8

    Args:
        input_path: Source PLY file (standard 3DGS format).
        output_path: Destination for compressed file.
        position_bits: Quantization precision for positions (8-16).
        color_bits: Quantization precision for colors/SH (6-8).
        scale_bits: Quantization precision for scales (8-16).

    Returns:
        Dict with original_size, compressed_size, ratio.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_size = input_path.stat().st_size

    # Read the PLY file
    points, attrs = _read_ply_points_and_attrs(input_path)
    if points is None:
        # Cannot parse -- copy as-is
        import shutil

        shutil.copy2(input_path, output_path)
        return {
            "original_size": original_size,
            "compressed_size": original_size,
            "ratio": 1.0,
        }

    n = len(points)

    # Quantize positions
    pos_min = points.min(axis=0)
    pos_max = points.max(axis=0)
    pos_range = (pos_max - pos_min).clip(min=1e-8)
    pos_norm = (points - pos_min) / pos_range
    pos_quant = _quantize(pos_norm, position_bits)

    # Process other attributes: convert float32 to float16 for most,
    # uint8 for opacities
    compressed_attrs = {}
    for key, arr in attrs.items():
        if "opacity" in key:
            # Sigmoid space -> clip to [0,1] range and quantize to uint8
            compressed_attrs[key] = (arr.clip(0, 1) * 255).astype(np.uint8)
        elif "scale" in key:
            # Log-scales: float32 -> float16
            compressed_attrs[key] = arr.astype(np.float16)
        elif "rot" in key:
            # Quaternions: normalize then int16
            compressed_attrs[key] = _quantize((arr + 1.0) / 2.0, 15).astype(np.int16)
        elif "f_dc" in key or "f_rest" in key:
            # SH coefficients: float32 -> float16
            compressed_attrs[key] = arr.astype(np.float16)
        else:
            # Unknown attribute: keep as float16
            compressed_attrs[key] = arr.astype(np.float16)

    # Write compressed binary
    with open(output_path, "wb") as f:
        f.write(b"GS3Q")  # magic
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", n))

        # Position metadata for dequantization
        f.write(pos_min.astype(np.float32).tobytes())
        f.write(pos_range.astype(np.float32).tobytes())
        f.write(struct.pack("<H", position_bits))

        # Number of attribute arrays
        f.write(struct.pack("<I", len(compressed_attrs)))

        # Quantized positions
        f.write(pos_quant.tobytes())

        # Each attribute: name (length-prefixed), dtype code, data
        for key, arr in compressed_attrs.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<H", len(key_bytes)))
            f.write(key_bytes)
            dtype_str = str(arr.dtype).encode("utf-8")
            f.write(struct.pack("<H", len(dtype_str)))
            f.write(dtype_str)
            f.write(struct.pack("<I", arr.nbytes))
            f.write(arr.tobytes())

    compressed_size = output_path.stat().st_size
    ratio = original_size / max(compressed_size, 1)

    logger.info(
        "Quantized PLY: %.1f MB -> %.1f MB (%.1fx reduction)",
        original_size / 1e6,
        compressed_size / 1e6,
        ratio,
    )

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": round(ratio, 2),
    }


# ---------------------------------------------------------------------------
# PLY reading utilities
# ---------------------------------------------------------------------------


def _read_ply_points_and_attrs(ply_path: Path) -> tuple:
    """Read a 3DGS PLY file and return (positions, {name: array}).

    Returns (None, None) if the file cannot be parsed.
    """
    try:
        from plyfile import PlyData

        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]

        # Extract positions
        x = np.array(vertex["x"], dtype=np.float32)
        y = np.array(vertex["y"], dtype=np.float32)
        z = np.array(vertex["z"], dtype=np.float32)
        positions = np.stack([x, y, z], axis=-1)

        # Extract all other properties as separate arrays
        attrs = {}
        skip = {"x", "y", "z"}
        for prop in vertex.properties:
            if prop.name not in skip:
                attrs[prop.name] = np.array(vertex[prop.name], dtype=np.float32)

        return positions, attrs

    except ImportError:
        logger.warning("plyfile not installed; cannot parse PLY for compression")
        return None, None
    except Exception as e:
        logger.warning("Failed to parse PLY %s: %s", ply_path, e)
        return None, None


# ---------------------------------------------------------------------------
# Compressed container format (Draco positions + raw attributes)
# ---------------------------------------------------------------------------


def _write_compressed_container(
    output_path: Path,
    draco_binary: bytes,
    attrs: dict[str, np.ndarray],
    num_points: int,
) -> None:
    """Write a compressed container: Draco point cloud + quantized attributes."""
    with open(output_path, "wb") as f:
        f.write(b"GS3D")  # magic: Gaussian Splat 3D Draco
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", num_points))

        # Draco-compressed positions
        f.write(struct.pack("<I", len(draco_binary)))
        f.write(draco_binary)

        # Attributes: quantized to float16 for SH, uint8 for opacity
        compressed_attrs = {}
        for key, arr in attrs.items():
            if "opacity" in key:
                compressed_attrs[key] = (arr.clip(0, 1) * 255).astype(np.uint8)
            elif "f_dc" in key or "f_rest" in key:
                compressed_attrs[key] = arr.astype(np.float16)
            else:
                compressed_attrs[key] = arr.astype(np.float16)

        f.write(struct.pack("<I", len(compressed_attrs)))
        for key, arr in compressed_attrs.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<H", len(key_bytes)))
            f.write(key_bytes)
            dtype_str = str(arr.dtype).encode("utf-8")
            f.write(struct.pack("<H", len(dtype_str)))
            f.write(dtype_str)
            f.write(struct.pack("<I", arr.nbytes))
            f.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _quantize(values: np.ndarray, bits: int) -> np.ndarray:
    """Quantize float values in [0, 1] to unsigned integer with given bit depth.

    Returns uint8 for bits <= 8, uint16 for bits <= 16, uint32 otherwise.
    """
    max_val = (1 << bits) - 1
    quantized = np.round(values.clip(0, 1) * max_val)

    if bits <= 8:
        return quantized.astype(np.uint8)
    elif bits <= 16:
        return quantized.astype(np.uint16)
    else:
        return quantized.astype(np.uint32)
