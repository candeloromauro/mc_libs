"""IO utilities.

Exports:
- scan_channels
- list_channel_fields
- from_imu_kearfott_compas
- list_files
- load_csv_timeseries_to_mc_kinematics
- PointFileMode
- McPointFileMode
- clear_console
- setup_logging_from_yaml
- check_output_file
- sum_total_bytes
- estimate_total_bytes

LCM/adapters exports are available when optional dependencies are installed.

Example:
>>> from mc_io_utils.lcm import scan_channels
>>> scan_channels("lcmlogs", "*.00")
"""

from __future__ import annotations

from typing import Any

from .io import *  # noqa: F401,F403
from .runtime import (
    clear_console,
    setup_logging_from_yaml,
    check_output_file,
    sum_total_bytes,
    estimate_total_bytes,
)


def _missing_lcm_dependency(*args: Any, **kwargs: Any):
    raise RuntimeError(
        "LCM helpers in mc_io_utils require optional dependencies (navlib, lcm). Install them and retry."
    )


def _missing_adapter_dependency(*args: Any, **kwargs: Any):
    raise RuntimeError(
        "Adapter helpers in mc_io_utils require optional dependencies from mc_robo_utils. Install them and retry."
    )


try:
    from .lcm import (
        scan_channels,
        scan_logs_detailed,
        list_channel_fields,
        list_channel_fields_in_dir,
        lcmlog_to_pickle,
    )
except Exception:  # pragma: no cover - optional dependencies may be missing
    scan_channels = _missing_lcm_dependency
    scan_logs_detailed = _missing_lcm_dependency
    list_channel_fields = _missing_lcm_dependency
    list_channel_fields_in_dir = _missing_lcm_dependency
    lcmlog_to_pickle = _missing_lcm_dependency


try:
    from .adapters import from_imu_kearfott_compas
except Exception:  # pragma: no cover - optional dependencies may be missing
    from_imu_kearfott_compas = _missing_adapter_dependency


__all__ = [
    "scan_channels",
    "scan_logs_detailed",
    "list_channel_fields",
    "list_channel_fields_in_dir",
    "lcmlog_to_pickle",
    "from_imu_kearfott_compas",
    "list_files",
    "load_csv_timeseries_to_mc_kinematics",
    "PointFileMode",
    "McPointFileMode",
    "clear_console",
    "setup_logging_from_yaml",
    "check_output_file",
    "sum_total_bytes",
    "estimate_total_bytes",
]
