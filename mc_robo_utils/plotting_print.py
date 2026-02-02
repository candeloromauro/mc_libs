"""Matplotlib plotting helpers for kinematics and diagnostics (print/static).

Functions:
- plot_multiple_mpl(data, specs, t_dt=None, title=None)
- plot_series_mpl(x, y, title=None, x_label=None, y_label=None)
- plot_allan_mpl(T_g, sigma_g, T_a, sigma_a)

Example:
>>> from mc_robo_utils.plotting_print import plot_multiple_mpl
>>> fig = plot_multiple_mpl(data, specs=[("body","vel",["x","y","z"])])
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .kinematics import McKinematicsData


_FIELDS = ("x", "y", "z", "psi", "theta", "phi")


def _get_component(data: McKinematicsData, frame: str, group: str):
    if frame not in {"ned", "body"}:
        raise ValueError("frame must be 'ned' or 'body'")
    if group not in {"pos", "vel", "acc"}:
        raise ValueError("group must be one of: pos, vel, acc")
    return getattr(getattr(data, frame), group)


def _get_time(data: McKinematicsData, t_dt: Optional[Sequence] = None):
    if t_dt is not None:
        return t_dt
    return data.t_datetime if data.t_datetime is not None else data.t_s


def _ylabel_for(group: str, field: str, units: dict) -> str:
    key = f"{group}.{field}"
    unit = units.get(key)
    if unit:
        return f"{field} [{unit}]"
    return field


def plot_multiple_mpl(
    data: McKinematicsData,
    specs: Iterable[tuple[str, str, Iterable[str]]],
    *,
    t_dt: Optional[Sequence] = None,
    title: Optional[str] = None,
):
    """Plot multiple components using Matplotlib (one subplot per field).

    Args:
        data (McKinematicsData): kinematics data to plot.
        specs (Iterable[tuple[str, str, Iterable[str]]]): list of (frame, group, fields).
        t_dt (Optional[Sequence]): optional datetime array for the x-axis.
        title (Optional[str]): optional figure title.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing the plots.

    Raises:
        ValueError: If ``specs`` is empty.

    Examples:
        >>> from mc_robo_utils.plotting_print import plot_multiple_mpl
        >>> fig = plot_multiple_mpl(data, specs=[("body", "vel", ["x", "y"])])
    """
    specs = list(specs)
    if not specs:
        raise ValueError("specs must not be empty")

    rows = 0
    for frame, group, fields in specs:
        rows += len(list(fields))

    fig, axes = plt.subplots(rows, 1, figsize=(14, max(4, 2 * rows)), sharex=True)
    if rows == 1:
        axes = [axes]

    t = _get_time(data, t_dt=t_dt)
    row = 0
    for frame, group, fields in specs:
        comp = _get_component(data, frame, group)
        for f in fields:
            y = getattr(comp, f)
            if y is None:
                row += 1
                continue
            axes[row].plot(t, y)
            axes[row].set_title(f"{frame}.{group}.{f}")
            axes[row].set_ylabel(_ylabel_for(group, f, data.units))
            row += 1

    if hasattr(axes[-1].xaxis, "set_major_formatter") and t_dt is not None:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("time")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_series_mpl(x, y, *, title=None, x_label=None, y_label=None):
    """Plot a single series using Matplotlib.

    Args:
        x: x-axis values.
        y: y-axis values.
        title: optional title.
        x_label: optional x-axis label.
        y_label: optional y-axis label.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing the series.

    Examples:
        >>> from mc_robo_utils.plotting_print import plot_series_mpl
        >>> fig = plot_series_mpl([0, 1], [0, 1], title="demo")
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(x, y)
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_allan_mpl(T_g: np.ndarray, sigma_g: np.ndarray, T_a: np.ndarray, sigma_a: np.ndarray):
    """Plot gyro/accel Allan deviation using Matplotlib.

    Args:
        T_g (np.ndarray): gyro tau array.
        sigma_g (np.ndarray): gyro Allan deviation array (K x 3).
        T_a (np.ndarray): accel tau array.
        sigma_a (np.ndarray): accel Allan deviation array (K x 3).

    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing the Allan plots.

    Examples:
        >>> import numpy as np
        >>> from mc_robo_utils.plotting_print import plot_allan_mpl
        >>> T = np.logspace(0, 3, 10)
        >>> sigma = np.random.rand(10, 3)
        >>> fig = plot_allan_mpl(T, sigma, T, sigma)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    colors = ["r", "g", "b"]
    labels = ["X", "Y", "Z"]

    for idx in range(3):
        axes[0].loglog(T_g, sigma_g[:, idx], colors[idx], label=labels[idx])
        axes[1].loglog(T_a, sigma_a[:, idx], colors[idx], label=labels[idx])

    axes[0].set_title("Gyro Allan Deviation")
    axes[0].set_xlabel("Tau [s]")
    axes[0].set_ylabel("[deg/hr]")
    axes[0].legend()
    axes[0].grid(True, which="both")

    axes[1].set_title("Accelerometer Allan Deviation")
    axes[1].set_xlabel("Tau [s]")
    axes[1].set_ylabel("[mG]")
    axes[1].legend()
    axes[1].grid(True, which="both")

    fig.tight_layout()
    return fig
