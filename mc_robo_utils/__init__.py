"""Utility helpers for robotics workflows.

Exports:
- quat_to_rpy
- KinematicsData, KinematicFrame, KinematicComponent, build_component, default_units
- McKinematicsData, McKinematicFrame, McKinematicComponent (aliases)
- plot_component, plot_multiple, save_html

Example:
>>> from mc_robo_utils import quat_to_rpy
>>> quat_to_rpy([0.0, 0.0, 0.0, 1.0])
"""

from .transformations import quat_to_rpy
from .kinematics import (
    KinematicsData,
    KinematicFrame,
    KinematicComponent,
    build_component,
    default_units,
)
from .plotting import plot_component, plot_multiple, save_html

__all__ = [
    "quat_to_rpy",
    "KinematicsData",
    "KinematicFrame",
    "KinematicComponent",
    "McKinematicsData",
    "McKinematicFrame",
    "McKinematicComponent",
    "build_component",
    "default_units",
    "plot_component",
    "plot_multiple",
    "save_html",
]

# Prefixed aliases for clarity at call sites
McKinematicsData = KinematicsData
McKinematicFrame = KinematicFrame
McKinematicComponent = KinematicComponent
