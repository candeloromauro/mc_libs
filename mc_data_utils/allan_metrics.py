"""Allan deviation metric extraction helpers.

Functions:
- extract_allan_metrics(T, sigma, ...)

Example:
>>> from mc_data_utils.allan_metrics import extract_allan_metrics
>>> metrics = extract_allan_metrics(T, sigma, kind="gyro_deg_per_hr")
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .metrics import first_local_min, nearest_index


def extract_allan_metrics(
    T: np.ndarray,
    sigma: np.ndarray,
    *,
    kind: str = "gyro_deg_per_hr",
    min_points: int = 8,
    slope_tol: float = 0.12,
    bias_slope_tol: float = 0.08,
    bias_divisor: float = 0.664,
) -> Dict[str, Dict[str, float]]:
    """Extract Allan-derived metrics per axis.

    Args:
        T (np.ndarray): averaging times (tau) in seconds, shape ``(K,)``.
        sigma (np.ndarray): Allan deviation array, shape ``(K, M)`` (or ``(K,)``).
            Units must match the input to ``allan2``. For example:
            - ``kind="gyro_deg_per_hr"`` expects ``sigma`` in deg/hr.
            - ``kind="accel_mg"`` expects ``sigma`` in mG.
        kind (str): conversion convention. Supported values:
            - ``"gyro_deg_per_hr"`` → ARW [deg/√hr], RRW [deg/hr/√hr], BS [deg/hr]
            - ``"accel_mg"`` → VRW [mG/√hr], BRW [mG/√hr], BS [mG]
        min_points (int): minimum number of points for slope segment fits.
        slope_tol (float): tolerance for slope matching when finding white/RW segments.
        bias_slope_tol (float): reserved tolerance for bias estimation (kept for API compatibility).
        bias_divisor (float): divisor for bias instability extraction (default 0.664).

    Returns:
        Dict[str, Dict[str, float]]: metrics per axis key (``"X"``, ``"Y"``, ``"Z"``). Each
        axis dict includes ``white_coeff``, ``rw_coeff``, ``bs``, and diagnostic tau ranges.

    Raises:
        ValueError: If ``T`` and ``sigma`` lengths are inconsistent.

    Examples:
        >>> import numpy as np
        >>> from mc_data_utils.allan_metrics import extract_allan_metrics
        >>> T = np.logspace(0, 3, 50)
        >>> sigma = np.random.rand(50, 3)
        >>> metrics = extract_allan_metrics(T, sigma, kind="gyro_deg_per_hr")
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    if sigma.shape[0] != T.shape[0]:
        raise ValueError(f"extract_allan_metrics: T has {T.shape[0]} rows but sigma has {sigma.shape[0]}")

    axes = ["X", "Y", "Z"]
    n_axes = min(sigma.shape[1], 3)
    metrics: Dict[str, Dict[str, float]] = {}

    def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
        segs: list[tuple[int, int]] = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif (not v) and start is not None:
                segs.append((start, i - 1))
                start = None
        if start is not None:
            segs.append((start, len(mask) - 1))
        return segs

    def _best_slope_segment(tau_s: np.ndarray, sig: np.ndarray, target: float, prefer: str) -> tuple[int, int] | None:
        logT = np.log10(tau_s)
        logS = np.log10(sig)
        local_slope = np.gradient(logS, logT)
        ok = np.isfinite(local_slope) & (np.abs(local_slope - target) <= slope_tol)

        segs = [s for s in _segments(ok) if (s[1] - s[0] + 1) >= min_points]
        if not segs:
            return None

        ranked: list[tuple[tuple[float, int, float], tuple[int, int]]] = []
        for i0, i1 in segs:
            seg_sl = local_slope[i0 : i1 + 1]
            slope_err = float(np.nanmedian(np.abs(seg_sl - target)))
            seg_len = (i1 - i0 + 1)
            med_tau = float(np.median(tau_s[i0 : i1 + 1]))
            if prefer == "low":
                key = (slope_err, -seg_len, med_tau)
            else:
                key = (slope_err, -seg_len, -med_tau)
            ranked.append((key, (i0, i1)))

        ranked.sort(key=lambda x: x[0])
        return ranked[0][1]

    for ax_i in range(n_axes):
        axis_name = axes[ax_i]
        sig = sigma[:, ax_i]
        mask = np.isfinite(T) & np.isfinite(sig) & (T > 0) & (sig > 0)
        tau = T[mask]
        sig = sig[mask]

        if len(tau) < min_points:
            metrics[axis_name] = {"white_coeff": np.nan, "rw_coeff": np.nan, "bs": np.nan}
            continue

        seg_white = _best_slope_segment(tau, sig, target=-0.5, prefer="low")
        seg_rw = _best_slope_segment(tau, sig, target=+0.5, prefer="high")

        white_coeff = np.nan
        rw_coeff = np.nan

        if seg_white:
            i0, i1 = seg_white
            m, b = np.polyfit(np.log10(tau[i0 : i1 + 1]), np.log10(sig[i0 : i1 + 1]), 1)
            white_coeff = 10 ** b

        if seg_rw:
            i0, i1 = seg_rw
            m, b = np.polyfit(np.log10(tau[i0 : i1 + 1]), np.log10(sig[i0 : i1 + 1]), 1)
            rw_coeff = 10 ** b

        bs_idx = first_local_min(sig)
        bs = sig[bs_idx] / bias_divisor if np.isfinite(sig[bs_idx]) else np.nan

        metrics[axis_name] = {
            "white_coeff": float(white_coeff),
            "rw_coeff": float(rw_coeff),
            "bs": float(bs),
            "tau_bs": float(tau[bs_idx]) if len(tau) else float("nan"),
            "tau_white0": float(tau[seg_white[0]]) if seg_white else float("nan"),
            "tau_white1": float(tau[seg_white[1]]) if seg_white else float("nan"),
            "tau_rw0": float(tau[seg_rw[0]]) if seg_rw else float("nan"),
            "tau_rw1": float(tau[seg_rw[1]]) if seg_rw else float("nan"),
        }

    return metrics
