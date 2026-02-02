"""Time-window utilities.

Functions:
- time_window_mask_from_hm(t_dt, start_hm=None, end_hm=None)
- time_window_mask_from_indexes(n, start_idx=None, end_idx=None)
- time_window_mask_from_seconds(t_s, start_s=None, end_s=None)

Example:
>>> from mc_data_utils.time import time_window_mask_from_hm
>>> mask = time_window_mask_from_hm(t_dt, "08:00", "10:00")
"""

from __future__ import annotations

from datetime import datetime, timedelta, time

import numpy as np


def _parse_hhmm(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def time_window_mask_from_hm(
    t_dt: list[datetime], start_hm: str | None, end_hm: str | None
) -> np.ndarray:
    """Return a boolean mask for a HH:MM time window on datetime series.

    If end <= start, the window is assumed to wrap into the next day.

    Args:
        t_dt (list[datetime]): datetime samples to filter.
        start_hm (str | None): start time in ``HH:MM`` format (local time).
        end_hm (str | None): end time in ``HH:MM`` format (local time).

    Returns:
        np.ndarray: boolean mask aligned to ``t_dt``.

    Examples:
        >>> from datetime import datetime
        >>> from mc_data_utils.time import time_window_mask_from_hm
        >>> t = [datetime(2024, 1, 1, 8, 0), datetime(2024, 1, 1, 9, 0)]
        >>> time_window_mask_from_hm(t, "08:30", "09:30").tolist()
        [False, True]
    """
    if not t_dt:
        return np.array([], dtype=bool)
    if start_hm is None and end_hm is None:
        return np.ones(len(t_dt), dtype=bool)

    base_date = t_dt[0].date()
    start_time = _parse_hhmm(start_hm) if start_hm is not None else time.min
    end_time = _parse_hhmm(end_hm) if end_hm is not None else time.max

    start_dt = datetime.combine(base_date, start_time)
    end_dt = datetime.combine(base_date, end_time)
    if end_dt <= start_dt:
        end_dt = end_dt + timedelta(days=1)

    return np.array([(d >= start_dt) and (d <= end_dt) for d in t_dt], dtype=bool)


def time_window_mask_from_indexes(
    n: int, start_idx: int | None = None, end_idx: int | None = None
) -> np.ndarray:
    """Return a boolean mask for a slice by index range.

    start_idx/end_idx are inclusive bounds. If None, defaults to start/end.

    Args:
        n (int): length of the sequence to mask.
        start_idx (int | None): inclusive start index (default 0).
        end_idx (int | None): inclusive end index (default ``n - 1``).

    Returns:
        np.ndarray: boolean mask of length ``n``.

    Raises:
        ValueError: If ``n`` is negative or indices are negative.

    Examples:
        >>> from mc_data_utils.time import time_window_mask_from_indexes
        >>> time_window_mask_from_indexes(5, start_idx=1, end_idx=3).tolist()
        [False, True, True, True, False]
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = n - 1
    if n == 0:
        return np.array([], dtype=bool)
    if start_idx < 0 or end_idx < 0:
        raise ValueError("start_idx/end_idx must be >= 0")
    if start_idx > end_idx:
        return np.zeros(n, dtype=bool)

    start_idx = max(0, start_idx)
    end_idx = min(n - 1, end_idx)
    mask = np.zeros(n, dtype=bool)
    mask[start_idx : end_idx + 1] = True
    return mask


def time_window_mask_from_seconds(
    t_s: np.ndarray, start_s: float | None = None, end_s: float | None = None
) -> np.ndarray:
    """Return a boolean mask for a time window based on seconds array.

    Args:
        t_s (np.ndarray): time vector in seconds.
        start_s (float | None): start time (defaults to first element).
        end_s (float | None): end time (defaults to last element).

    Returns:
        np.ndarray: boolean mask aligned to ``t_s``.

    Examples:
        >>> import numpy as np
        >>> from mc_data_utils.time import time_window_mask_from_seconds
        >>> t = np.array([0.0, 1.0, 2.0, 3.0])
        >>> time_window_mask_from_seconds(t, 1.0, 2.0).tolist()
        [False, True, True, False]
    """
    t_s = np.asarray(t_s, dtype=float)
    if t_s.size == 0:
        return np.array([], dtype=bool)
    if start_s is None:
        start_s = float(t_s[0])
    if end_s is None:
        end_s = float(t_s[-1])
    return (t_s >= start_s) & (t_s <= end_s)
