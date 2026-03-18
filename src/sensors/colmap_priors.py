"""Generate COLMAP-compatible pose priors from IMU-derived rotations."""

import logging
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def rotation_matrix_to_colmap_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a COLMAP-convention quaternion.

    COLMAP uses the Hamilton quaternion convention with scalar-first ordering:
    ``(w, x, y, z)`` where ``w = cos(theta/2)`` and the quaternion is normalised
    with ``w >= 0`` (canonical form) to avoid the double-cover ambiguity.

    Uses ``scipy.spatial.transform.Rotation`` with ``scalar_first=True`` and
    ``canonical=True`` for numerically robust conversion.

    Args:
        R: A (3, 3) rotation matrix (proper orthogonal, det = +1).

    Returns:
        Quaternion array of shape (4,) as ``[w, x, y, z]``.
    """
    rot = Rotation.from_matrix(np.asarray(R, dtype=np.float64))
    return rot.as_quat(scalar_first=True, canonical=True)


def generate_colmap_priors(
    rotations: np.ndarray,
    frame_names: Sequence[str],
    output_path: Union[str, Path],
) -> Path:
    """Write a COLMAP ``image_priors.txt`` file with rotation-only pose priors.

    Each line provides a rotation prior for one image.  Translation components
    are left free (written as 0 with infinite uncertainty) because double-integrated
    accelerometer drift makes position priors unreliable.

    The output format follows COLMAP's ``image_priors.txt`` convention::

        # IMAGE_NAME QW QX QY QZ TX TY TZ QW_STD QX_STD QY_STD QZ_STD TX_STD TY_STD TZ_STD
        frame_0001.jpg 0.999 0.010 -0.005 0.003  0 0 0  0.01 0.01 0.01 0.01  1e6 1e6 1e6

    The quaternion standard deviations default to 0.01 (tight prior) and
    translation standard deviations are set to 1e6 (effectively unconstrained).

    Args:
        rotations: (N, 3, 3) rotation matrices, one per frame.
        frame_names: Sequence of N image filenames (e.g., ``["frame_0001.jpg", ...]``).
        output_path: File path for the output priors file.

    Returns:
        Resolved Path to the written file.

    Raises:
        ValueError: If *rotations* and *frame_names* have mismatched lengths.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    output_path = Path(output_path)

    n_frames = len(frame_names)
    if rotations.shape[0] != n_frames:
        raise ValueError(
            f"Mismatch: {rotations.shape[0]} rotations vs {n_frames} frame names"
        )

    # Uncertainty parameters
    q_std = 0.01  # tight rotation prior
    t_std = 1e6   # unconstrained translation

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# COLMAP image pose priors generated from IMU data\n"
            "# IMAGE_NAME QW QX QY QZ TX TY TZ "
            "QW_STD QX_STD QY_STD QZ_STD TX_STD TY_STD TZ_STD\n"
        )
        for i in range(n_frames):
            q = rotation_matrix_to_colmap_quat(rotations[i])
            line = (
                f"{frame_names[i]} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                f"0.0 0.0 0.0 "
                f"{q_std} {q_std} {q_std} {q_std} "
                f"{t_std:.1f} {t_std:.1f} {t_std:.1f}\n"
            )
            fh.write(line)

    logger.info("Wrote COLMAP priors for %d frames to %s", n_frames, output_path)
    return output_path.resolve()
