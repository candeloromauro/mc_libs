"""LCM channel adapters.

Functions:
- from_imu_kearfott_compas(imu)

Example:
>>> from mc_io_utils.adapters import from_imu_kearfott_compas
>>> data = from_imu_kearfott_compas(imu_dict)
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Sequence

import numpy as np

from mc_robo_utils import McKinematicsData, McKinematicFrame, build_component, default_units
from mc_robo_utils.transformations import quat_to_rpy


def from_imu_kearfott_compas(imu: Dict) -> McKinematicsData:
    """Build KinematicsData from an IMU_KEARFOTT_COMPAS dict.

    Populates:
    - body.pos.psi/theta/phi from quaternion orientation (rad)
    - body.vel.x/y/z from angRate (rad/s)
    - body.acc.x/y/z from xyz_ddot (m/s^2)
    - t_s from lcm_timestamp (or header timestamp)
    - t_datetime from header timestamp when available
    Args:
        imu (Dict): decoded LCM message dict with fields like ``orientation``, ``angRate``,
            ``xyz_ddot``, and timestamps.

    Returns:
        McKinematicsData: populated kinematics data with body frame fields.

    Raises:
        ValueError: If required fields are missing.

    Examples:
        >>> from mc_io_utils.adapters import from_imu_kearfott_compas
        >>> data = from_imu_kearfott_compas(imu_dict)
    """
    if "orientation" not in imu or "angRate" not in imu or "xyz_ddot" not in imu:
        raise ValueError("IMU dict missing required fields")

    orient = np.asarray(imu["orientation"], dtype=float)
    ang_rate = np.asarray(imu["angRate"], dtype=float)
    accel = np.asarray(imu["xyz_ddot"], dtype=float)

    n = min(len(orient), len(ang_rate), len(accel))
    orient = orient[:n]
    ang_rate = ang_rate[:n]
    accel = accel[:n]

    if "lcm_timestamp" in imu:
        t_s = np.asarray(imu["lcm_timestamp"], dtype=float)[:n]
        t_s = t_s - t_s[0]
        t_datetime: Optional[Sequence[datetime]] = None
    else:
        ts = np.asarray(imu["header"]["timestamp"], dtype=float)[:n]
        t_s = (ts - ts[0]) * 1e-6
        t_datetime = [datetime.fromtimestamp(float(us) * 1e-6) for us in ts]

    rpy = quat_to_rpy(orient)

    body = McKinematicFrame(
        pos=build_component(psi=rpy[:, 2], theta=rpy[:, 1], phi=rpy[:, 0]),
        vel=build_component(x=ang_rate[:, 0], y=ang_rate[:, 1], z=ang_rate[:, 2]),
        acc=build_component(x=accel[:, 0], y=accel[:, 1], z=accel[:, 2]),
    )

    return McKinematicsData(t_s=t_s, t_datetime=t_datetime, body=body, units=default_units())
