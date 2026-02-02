"""Signal analysis utilities.

Functions:
- estimate_period(t, y)

Example:
>>> from mc_data_utils.signal import estimate_period
>>> period_s = estimate_period(t, y)
"""

from __future__ import annotations

import numpy as np


def estimate_period(t: np.ndarray, y: np.ndarray) -> float:
    """Estimate dominant period using FFT; returns period in seconds.

    Args:
        t (np.ndarray): time stamps (seconds).
        y (np.ndarray): signal samples aligned with ``t``.

    Returns:
        float: estimated dominant period in seconds (``nan`` if insufficient data).

    Examples:
        >>> import numpy as np
        >>> from mc_data_utils.signal import estimate_period
        >>> t = np.linspace(0, 10, 500)
        >>> y = np.sin(2 * np.pi * t)
        >>> round(estimate_period(t, y), 2)
        1.0
    """
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) < 4:
        return float("nan")
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    span = t[-1] - t[0]
    if span <= 0:
        return float("nan")

    # Detrend and window to reduce leakage
    y = y - np.mean(y)
    p = np.polyfit(t, y, 1)
    y = y - (p[0] * t + p[1])
    if np.allclose(y, 0):
        return float("nan")
    window = np.hanning(len(y))
    y_w = y * window

    freqs = np.fft.rfftfreq(len(y_w), dt)
    power = np.abs(np.fft.rfft(y_w)) ** 2
    fmin = 1.0 / span
    mask_f = freqs > fmin
    if not np.any(mask_f):
        return float("nan")
    f_peak = freqs[mask_f][np.argmax(power[mask_f])]
    if f_peak <= 0:
        return float("nan")
    return float(1.0 / f_peak)
