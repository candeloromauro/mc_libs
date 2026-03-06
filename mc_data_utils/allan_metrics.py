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

from .metrics import nearest_index

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
            The input Allan curve is fit in tau seconds; coefficients are converted
            to per-``sqrt(hr)`` style metrics commonly reported in datasheets.
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
            elif prefer == "high":
                key = (slope_err, -seg_len, -med_tau)
            else:
                key = (slope_err, -seg_len, 0.0)
            ranked.append((key, (i0, i1)))

        ranked.sort(key=lambda x: x[0])
        return ranked[0][1]

    def _best_window_fit(
        tau_s: np.ndarray, sig: np.ndarray, target: float, prefer: str
    ) -> tuple[int, int, float, float] | None:
        """Fallback fit when strict slope segment detection fails.

        Chooses the window whose fitted log-log slope is closest to ``target``.
        """
        if len(tau_s) < min_points:
            return None

        logT = np.log10(tau_s)
        logS = np.log10(sig)
        max_w = min(len(tau_s), max(min_points + 6, 20))
        best_key: tuple[float, int, float] | None = None
        best_fit: tuple[int, int, float, float] | None = None

        for w in range(min_points, max_w + 1):
            for i0 in range(0, len(tau_s) - w + 1):
                i1 = i0 + w - 1
                x = logT[i0 : i1 + 1]
                y = logS[i0 : i1 + 1]
                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                    continue
                m, b = np.polyfit(x, y, 1)
                slope_err = abs(float(m) - target)
                med_tau = float(np.median(tau_s[i0 : i1 + 1]))
                if prefer == "low":
                    loc_key = med_tau
                elif prefer == "high":
                    loc_key = -med_tau
                else:
                    loc_key = 0.0
                key = (slope_err, -w, loc_key)
                if best_key is None or key < best_key:
                    best_key = key
                    best_fit = (i0, i1, float(m), float(b))

        return best_fit

    def _bias_stability_index(tau_s: np.ndarray, sig: np.ndarray) -> int | None:
        """Return interior local-minimum index for bias stability, if present.

        Bias instability requires a real turning point in the Allan curve.
        If the curve is monotonic over the available tau range, return ``None``.
        """
        n = len(sig)
        if n < 5:
            return None

        logT = np.log10(tau_s)
        logS = np.log10(sig)
        slope = np.gradient(logS, logT)

        # Guard against edge picks; edge minima are often artifacts of finite record length.
        edge_guard = max(3, int(0.05 * n))
        i0 = edge_guard
        i1 = n - edge_guard
        if i1 - i0 < 3:
            return None

        slope_window = max(4, int(0.06 * n))
        slope_eps = 0.03

        cands: list[int] = []
        for i in range(i0, i1):
            if not (sig[i] <= sig[i - 1] and sig[i] <= sig[i + 1]):
                continue
            # Require slope sign change around the minimum.
            if slope[i - 1] < 0.0 and slope[i + 1] > 0.0:
                left_start = max(i0, i - slope_window)
                right_end = min(i1, i + slope_window)
                if i <= left_start or i >= right_end:
                    continue

                left_slopes = slope[left_start:i]
                right_slopes = slope[i + 1 : right_end + 1]
                if len(left_slopes) < 3 or len(right_slopes) < 3:
                    continue
                # Require sustained down-slope then sustained up-slope (not a tail wiggle).
                if float(np.median(left_slopes)) >= -slope_eps:
                    continue
                if float(np.median(right_slopes)) <= slope_eps:
                    continue
                if float(np.mean(left_slopes < 0.0)) < 0.6:
                    continue
                if float(np.mean(right_slopes > 0.0)) < 0.6:
                    continue
                cands.append(i)

        if not cands:
            return None

        # Use the lowest local minimum among valid candidates.
        best = min(cands, key=lambda j: sig[j])
        return int(best)

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

        white_coeff_raw = np.nan
        rw_coeff_raw = np.nan
        white_slope = np.nan
        rw_slope = np.nan

        if seg_white:
            i0, i1 = seg_white
            m, b = np.polyfit(np.log10(tau[i0 : i1 + 1]), np.log10(sig[i0 : i1 + 1]), 1)
            white_slope = float(m)
            white_coeff_raw = 10 ** b
        else:
            fit_white = _best_window_fit(tau, sig, target=-0.5, prefer="low")
            if fit_white is not None:
                i0, i1, m, b = fit_white
                seg_white = (i0, i1)
                white_slope = float(m)
                white_coeff_raw = 10 ** b

        bs_idx = _bias_stability_index(tau, sig)
        if bs_idx is None:
            bs = np.nan
            tau_bs = np.nan
        else:
            bs = sig[bs_idx] / bias_divisor if np.isfinite(sig[bs_idx]) else np.nan
            tau_bs = tau[bs_idx]

        # For RRW/BRW, prioritize the post-BS rising branch when a BS minimum exists.
        rw_offset = 0
        tau_rw = tau
        sig_rw = sig
        if bs_idx is not None:
            # Start strictly after the BS point to avoid fitting the flat minimum itself.
            bs_after = int(bs_idx) + 1
            if (len(tau) - bs_after) >= min_points:
                rw_offset = bs_after
            elif (len(tau) - int(bs_idx)) >= min_points:
                rw_offset = int(bs_idx)
            tau_rw = tau[rw_offset:]
            sig_rw = sig[rw_offset:]

        seg_rw_local = _best_slope_segment(tau_rw, sig_rw, target=+0.5, prefer="high")
        seg_rw = None
        if seg_rw_local is not None:
            seg_rw = (seg_rw_local[0] + rw_offset, seg_rw_local[1] + rw_offset)

        if seg_rw:
            i0, i1 = seg_rw
            m, b = np.polyfit(np.log10(tau[i0 : i1 + 1]), np.log10(sig[i0 : i1 + 1]), 1)
            rw_slope = float(m)
            rw_coeff_raw = 10 ** b
        else:
            fit_rw = _best_window_fit(tau_rw, sig_rw, target=+0.5, prefer="high")
            if fit_rw is not None:
                i0, i1, m, b = fit_rw
                seg_rw = (i0 + rw_offset, i1 + rw_offset)
                rw_slope = float(m)
                rw_coeff_raw = 10 ** b

        # Fitted coefficients are based on tau in seconds:
        #   sigma = c_w * tau^(-1/2), sigma = c_rw * tau^(+1/2)
        # Generic tau-unit conversion to hour-based tau uses:
        #   tau_hr = tau_s / 3600 -> c_w,hr = c_w,s / 60 ; c_rw,hr = c_rw,s * 60
        #
        # NOTE: For accelerometer VRW in m/s/sqrt(hr), the white coefficient from
        # sigma_a(tau) (where sigma_a is in acceleration units) must be converted
        # with an additional *60 factor relative to the generic c_w,hr term.
        # We compute that explicitly below from the raw tau-second fit.
        white_coeff = white_coeff_raw
        rw_coeff = rw_coeff_raw
        extra: Dict[str, float] = {}
        if kind in {"gyro_deg_per_hr", "accel_mg"}:
            white_coeff = white_coeff_raw / 60.0 if np.isfinite(white_coeff_raw) else np.nan
            rw_coeff = rw_coeff_raw * 60.0 if np.isfinite(rw_coeff_raw) else np.nan
        if kind == "gyro_deg_per_hr":
            extra["arw_deg_per_sqrt_hr"] = float(white_coeff)
            extra["rrw_deg_per_hr_per_sqrt_hr"] = float(rw_coeff)
        elif kind == "accel_mg":
            extra["vrw_mg_per_sqrt_hr"] = float(white_coeff)
            extra["vrw_mps_per_sqrt_hr"] = (
                float(white_coeff_raw) * 9.80665e-3 * 60.0 if np.isfinite(white_coeff_raw) else float("nan")
            )
            extra["brw_mg_per_sqrt_hr"] = float(rw_coeff)

        metrics[axis_name] = {
            "white_coeff": float(white_coeff),
            "rw_coeff": float(rw_coeff),
            "white_coeff_raw_tau_s": float(white_coeff_raw),
            "rw_coeff_raw_tau_s": float(rw_coeff_raw),
            "bs": float(bs),
            "bs_found": float(np.isfinite(bs)),
            "white_slope": float(white_slope),
            "rw_slope": float(rw_slope),
            "tau_bs": float(tau_bs) if len(tau) else float("nan"),
            "tau_white0": float(tau[seg_white[0]]) if seg_white else float("nan"),
            "tau_white1": float(tau[seg_white[1]]) if seg_white else float("nan"),
            "tau_rw0": float(tau[seg_rw[0]]) if seg_rw else float("nan"),
            "tau_rw1": float(tau[seg_rw[1]]) if seg_rw else float("nan"),
        }
        metrics[axis_name].update(extra)

    return metrics
