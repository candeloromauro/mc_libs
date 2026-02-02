"""Allan variance/deviation utilities.

Functions:
- allan2(omega, fs, pts)

Example:
>>> from mc_data_utils.allan import allan2
>>> T, sigma = allan2(omega, fs=100.0, pts=100)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def allan2(omega: np.ndarray, fs: float, pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Allan deviation (sigma) using log-spaced cluster sizes.

    Args:
        omega (np.ndarray): Rate sequence shaped ``(N,)`` or ``(N, M)``, where ``M`` is the
            number of axes.
        fs (float): Sample rate in Hz.
        pts (int): Number of log-spaced cluster sizes (tau points) to evaluate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``T`` (tau values in seconds) and ``sigma`` with
        shape ``(len(T), M)`` containing Allan deviation per axis.

    Raises:
        ValueError: If fewer than 3 samples are provided.

    Examples:
        >>> import numpy as np
        >>> from mc_data_utils.allan import allan2
        >>> omega = np.random.randn(1000)
        >>> T, sigma = allan2(omega, fs=100.0, pts=50)
    """
    if omega.ndim == 1:
        omega = omega[:, None]
    N, M = omega.shape
    if N < 3:
        raise ValueError("allan2 requires at least 3 samples")

    n = 2 ** np.arange(0, int(np.floor(np.log2(N / 2))) + 1)
    maxN = int(n[-1])
    end_log_inc = np.log10(maxN)
    m = np.unique(np.ceil(np.logspace(0, end_log_inc, pts))).astype(int)
    t0 = 1.0 / fs
    T = m * t0
    theta = np.cumsum(omega, axis=0) / fs
    sigma2 = np.zeros((len(T), M))

    for i, mi in enumerate(m):
        kmax = N - 2 * mi
        if kmax <= 0:
            continue
        diff = theta[2 * mi :] - 2.0 * theta[mi:-mi] + theta[: -2 * mi]
        sigma2[i, :] = np.sum(diff * diff, axis=0)

    denom = 2.0 * (T ** 2) * (N - 2 * m)
    valid = denom > 0
    sigma2[valid, :] = sigma2[valid, :] / denom[valid, None]
    sigma2[~valid, :] = np.nan
    sigma = np.sqrt(sigma2)
    return T, sigma
