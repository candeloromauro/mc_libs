"""Kinematics data structures for robotics workflows.

Classes:
- KinematicComponent
- KinematicFrame
- KinematicsData
- McKinematicComponent (alias)
- McKinematicFrame (alias)
- McKinematicsData (alias)

Functions:
- build_component(...)
- default_units()

Example:
>>> from mc_robo_utils import McKinematicsData, McKinematicFrame, build_component
>>> data = McKinematicsData(t_s=[0.0, 1.0], body=McKinematicFrame(pos=build_component(x=[0, 1])))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence, Dict

import numpy as np


@dataclass
class KinematicComponent:
    """Container for x,y,z and optional attitude angles psi, theta, phi."""
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    psi: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None

    def as_dict(self) -> dict:
        """Return the component fields as a dictionary.

        Args:
            None

        Returns:
            dict: mapping with keys ``x``, ``y``, ``z``, ``psi``, ``theta``, ``phi``.

        Examples:
            >>> from mc_robo_utils.kinematics import KinematicComponent
            >>> comp = KinematicComponent(x=[0.0, 1.0], y=[2.0, 3.0])
            >>> comp.as_dict()["x"]
            [0.0, 1.0]
        """
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "psi": self.psi,
            "theta": self.theta,
            "phi": self.phi,
        }


@dataclass
class KinematicFrame:
    """Container for pos, vel, acc groups in a frame (e.g., NED or body)."""
    pos: KinematicComponent = field(default_factory=KinematicComponent)
    vel: KinematicComponent = field(default_factory=KinematicComponent)
    acc: KinematicComponent = field(default_factory=KinematicComponent)


@dataclass
class KinematicsData:
    """Standard kinematics structure shared across projects."""
    t_s: np.ndarray
    t_datetime: Optional[Sequence[datetime]] = None
    ned: KinematicFrame = field(default_factory=KinematicFrame)
    body: KinematicFrame = field(default_factory=KinematicFrame)
    units: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t_s = np.asarray(self.t_s, dtype=float)
        if self.t_datetime is not None:
            if len(self.t_datetime) != len(self.t_s):
                raise ValueError("t_datetime must match length of t_s")

    def copy(self) -> "KinematicsData":
        """Return a shallow copy of the kinematics data.

        Args:
            None

        Returns:
            KinematicsData: new instance with copied time arrays and same frame objects.

        Examples:
            >>> import numpy as np
            >>> from mc_robo_utils.kinematics import KinematicsData
            >>> data = KinematicsData(t_s=np.array([0.0, 1.0]))
            >>> data_copy = data.copy()
        """
        return KinematicsData(
            t_s=self.t_s.copy(),
            t_datetime=list(self.t_datetime) if self.t_datetime is not None else None,
            ned=self.ned,
            body=self.body,
        )


def build_component(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    psi: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
    phi: Optional[np.ndarray] = None,
) -> KinematicComponent:
    """Convenience constructor for ``KinematicComponent``.

    Args:
        x (Optional[np.ndarray]): x samples.
        y (Optional[np.ndarray]): y samples.
        z (Optional[np.ndarray]): z samples.
        psi (Optional[np.ndarray]): yaw samples.
        theta (Optional[np.ndarray]): pitch samples.
        phi (Optional[np.ndarray]): roll samples.

    Returns:
        KinematicComponent: populated component container.

    Examples:
        >>> from mc_robo_utils.kinematics import build_component
        >>> comp = build_component(x=[0.0, 1.0], y=[0.0, 2.0])
    """
    return KinematicComponent(x=x, y=y, z=z, psi=psi, theta=theta, phi=phi)


def default_units() -> Dict[str, str]:
    """Default units for common kinematics fields.

    Args:
        None

    Returns:
        Dict[str, str]: mapping of kinematics field names to unit strings.

    Examples:
        >>> from mc_robo_utils.kinematics import default_units
        >>> units = default_units()
        >>> units["pos.x"]
        'm'
    """
    return {
        "t_s": "s",
        "pos.x": "m",
        "pos.y": "m",
        "pos.z": "m",
        "pos.psi": "rad",
        "pos.theta": "rad",
        "pos.phi": "rad",
        "vel.x": "m/s",
        "vel.y": "m/s",
        "vel.z": "m/s",
        "vel.psi": "rad/s",
        "vel.theta": "rad/s",
        "vel.phi": "rad/s",
        "acc.x": "m/s^2",
        "acc.y": "m/s^2",
        "acc.z": "m/s^2",
        "acc.psi": "rad/s^2",
        "acc.theta": "rad/s^2",
        "acc.phi": "rad/s^2",
    }


# Prefixed aliases for clarity across mc_* libraries.
McKinematicComponent = KinematicComponent
McKinematicFrame = KinematicFrame
McKinematicsData = KinematicsData
