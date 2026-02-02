"""Common robotics transformations.

Functions:
- quat_to_rpy(quat_xyzw)

Example:
>>> from mc_robo_utils.transformations import quat_to_rpy
>>> quat_to_rpy([0.0, 0.0, 0.0, 1.0])
"""

from __future__ import annotations

import numpy as np


def quat_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) (x, y, z, w) to roll/pitch/yaw (rad).

    Args:
        quat_xyzw: Array of shape (N, 4) or (4,) in x,y,z,w order.

    Returns:
        Array of shape (N, 3) with roll, pitch, yaw in radians.

    Raises:
        ValueError: If the input array does not have shape ``(4,)`` or ``(N, 4)``.

    Examples:
        >>> import numpy as np
        >>> from mc_robo_utils.transformations import quat_to_rpy
        >>> quat_to_rpy(np.array([0.0, 0.0, 0.0, 1.0]))
    """
    q = np.asarray(quat_xyzw, dtype=float)
    if q.ndim == 1:
        if q.shape[0] != 4:
            raise ValueError(f"quat_to_rpy expects 4 elements, got shape {q.shape}")
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat_to_rpy expects shape (N,4), got {q.shape}")

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack((roll, pitch, yaw))
