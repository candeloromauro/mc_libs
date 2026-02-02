"""General data utilities.

Exports:
- allan2
- estimate_period
- first_local_min, nearest_index
- extract_allan_metrics
- time_window_mask_from_hm, time_window_mask_from_indexes, time_window_mask_from_seconds

Example:
>>> import mc_data_utils
>>> mc_data_utils.allan2(omega, fs=100.0, pts=100)
"""

from .allan import allan2
from .signal import estimate_period
from .metrics import first_local_min, nearest_index
from .allan_metrics import extract_allan_metrics
from .time import time_window_mask_from_hm, time_window_mask_from_indexes, time_window_mask_from_seconds

__all__ = [
    "allan2",
    "estimate_period",
    "first_local_min",
    "nearest_index",
    "extract_allan_metrics",
    "time_window_mask_from_hm",
    "time_window_mask_from_indexes",
    "time_window_mask_from_seconds",
]
