"""Plotting helpers for kinematics data using Plotly.

Functions:
- plot_component(...)
- plot_multiple(...)
- plot_series(x, y, title=None, x_label="time", y_label=None, height=400)
- plot_allan(T_g, sigma_g, T_a, sigma_a, labels=None, colors=None)
- save_html(fig, path, open_after=False)

Example:
>>> from mc_robo_utils.plotting import plot_component, save_html
>>> fig = plot_component(data, frame="body", group="acc", fields=["x", "y", "z"])
>>> save_html(fig, "body_acc.html")
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .kinematics import McKinematicsData, McKinematicFrame, McKinematicComponent


_FIELDS = ("x", "y", "z", "psi", "theta", "phi")


def _get_component(frame: McKinematicFrame, group: str) -> McKinematicComponent:
    if group not in {"pos", "vel", "acc"}:
        raise ValueError("group must be one of: pos, vel, acc")
    return getattr(frame, group)


def _get_time(data: McKinematicsData) -> Sequence:
    return data.t_datetime if data.t_datetime is not None else data.t_s


def plot_component(
    data: McKinematicsData,
    frame: str = "ned",
    group: str = "pos",
    fields: Optional[Iterable[str]] = None,
    separate_axes: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot a single component (pos/vel/acc) from a frame.

    Args:
        data (McKinematicsData): kinematics data structure.
        frame (str): frame name (``"ned"`` or ``"body"``).
        group (str): component group (``"pos"``, ``"vel"``, or ``"acc"``).
        fields (Optional[Iterable[str]]): subset of ``[x, y, z, psi, theta, phi]`` to plot.
        separate_axes (bool): when True, one subplot per field.
        title (Optional[str]): optional figure title.

    Returns:
        go.Figure: Plotly figure containing the plot(s).

    Raises:
        ValueError: If the frame/group is invalid or no fields are available.

    Examples:
        >>> from mc_robo_utils.plotting import plot_component
        >>> fig = plot_component(data, frame="body", group="acc", fields=["x", "y", "z"])
    """
    if frame not in {"ned", "body"}:
        raise ValueError("frame must be 'ned' or 'body'")

    comp = _get_component(getattr(data, frame), group)
    fields = list(fields) if fields is not None else [f for f in _FIELDS if getattr(comp, f) is not None]
    if not fields:
        raise ValueError("No fields available to plot")

    t = _get_time(data)

    if separate_axes:
        fig = make_subplots(rows=len(fields), cols=1, shared_xaxes=True, subplot_titles=fields)
        for idx, f in enumerate(fields, start=1):
            y = getattr(comp, f)
            if y is None:
                continue
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=f"{frame}.{group}.{f}"), row=idx, col=1)
        fig.update_layout(height=250 * len(fields))
        fig.update_xaxes(title_text="time", row=len(fields), col=1)
    else:
        fig = go.Figure()
        for f in fields:
            y = getattr(comp, f)
            if y is None:
                continue
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=f"{frame}.{group}.{f}"))
        fig.update_layout(height=600)
        fig.update_xaxes(title_text="time")

    fig.update_layout(title=title or f"{frame.upper()} {group.upper()}")
    return fig


def plot_multiple(
    data: McKinematicsData,
    specs: Iterable[tuple[str, str, Iterable[str]]],
    separate_axes: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot multiple components in a single figure.

    Args:
        data (McKinematicsData): kinematics data structure.
        specs (Iterable[tuple[str, str, Iterable[str]]]): list of (frame, group, fields).
        separate_axes (bool): when True, one subplot per field.
        title (Optional[str]): optional figure title.

    Returns:
        go.Figure: Plotly figure containing the plot(s).

    Raises:
        ValueError: If ``specs`` is empty.

    Examples:
        >>> from mc_robo_utils.plotting import plot_multiple
        >>> fig = plot_multiple(data, specs=[("body", "vel", ["x", "y", "z"])])
    """
    specs = list(specs)
    if not specs:
        raise ValueError("specs must not be empty")

    t = _get_time(data)

    total_rows = 0
    for frame, group, fields in specs:
        comp = _get_component(getattr(data, frame), group)
        if fields is None:
            fields = [f for f in _FIELDS if getattr(comp, f) is not None]
        total_rows += len(list(fields)) if separate_axes else 1

    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True)

    row = 1
    for frame, group, fields in specs:
        comp = _get_component(getattr(data, frame), group)
        fields = list(fields) if fields is not None else [f for f in _FIELDS if getattr(comp, f) is not None]
        if not fields:
            continue
        if separate_axes:
            for f in fields:
                y = getattr(comp, f)
                if y is None:
                    continue
                fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=f"{frame}.{group}.{f}"), row=row, col=1)
                row += 1
        else:
            for f in fields:
                y = getattr(comp, f)
                if y is None:
                    continue
                fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=f"{frame}.{group}.{f}"), row=row, col=1)
            row += 1

    fig.update_layout(title=title or "Kinematics", height=250 * total_rows)
    fig.update_xaxes(title_text="time", row=total_rows, col=1)
    return fig


def plot_series(
    x: Sequence | np.ndarray,
    y: Sequence | np.ndarray,
    *,
    title: str | None = None,
    x_label: str = "time",
    y_label: str | None = None,
    height: int = 400,
) -> go.Figure:
    """Plot a single series (x vs y).

    Args:
        x (Sequence | np.ndarray): x values.
        y (Sequence | np.ndarray): y values.
        title (str | None): optional title.
        x_label (str): x-axis label.
        y_label (str | None): y-axis label.
        height (int): figure height in pixels.

    Returns:
        go.Figure: Plotly figure containing the series.

    Examples:
        >>> from mc_robo_utils.plotting import plot_series
        >>> fig = plot_series([0, 1, 2], [0, 1, 0], title="demo")
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=height)
    return fig


def plot_allan(
    T_g: np.ndarray,
    sigma_g: np.ndarray,
    T_a: np.ndarray,
    sigma_a: np.ndarray,
    *,
    labels: Optional[Iterable[str]] = None,
    colors: Optional[Iterable[str]] = None,
    title: str = "Allan Deviation",
) -> go.Figure:
    """Plot gyro/accel Allan deviation with default markers/labels.

    Args:
        T_g (np.ndarray): gyro tau array.
        sigma_g (np.ndarray): gyro Allan deviation array (K x 3).
        T_a (np.ndarray): accel tau array.
        sigma_a (np.ndarray): accel Allan deviation array (K x 3).
        labels (Optional[Iterable[str]]): axis labels (default ``["X", "Y", "Z"]``).
        colors (Optional[Iterable[str]]): trace colors.
        title (str): figure title.

    Returns:
        go.Figure: Plotly figure with two stacked log-log plots.

    Examples:
        >>> import numpy as np
        >>> from mc_robo_utils.plotting import plot_allan
        >>> T = np.logspace(0, 3, 10)
        >>> sigma = np.random.rand(10, 3)
        >>> fig = plot_allan(T, sigma, T, sigma)
    """
    labels = list(labels) if labels is not None else ["X", "Y", "Z"]
    colors = list(colors) if colors is not None else ["red", "green", "blue"]

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Gyro Allan Deviation", "Accelerometer Allan Deviation"])

    for idx, lab in enumerate(labels):
        fig.add_trace(
            go.Scatter(x=T_g, y=sigma_g[:, idx], mode="lines", name=f"Gyro {lab}", line=dict(color=colors[idx])),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=T_a, y=sigma_a[:, idx], mode="lines", name=f"Accel {lab}", line=dict(color=colors[idx])),
            row=2,
            col=1,
        )

    fig.update_xaxes(type="log", title_text="Tau [s]", row=1, col=1)
    fig.update_yaxes(type="log", title_text="[deg/hr]", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Tau [s]", row=2, col=1)
    fig.update_yaxes(type="log", title_text="[mG]", row=2, col=1)
    fig.update_layout(title=title, height=900)
    return fig


def save_html(fig: go.Figure, path: str, open_after: bool = False) -> None:
    """Save a Plotly figure to HTML, optionally opening it.

    Args:
        fig (go.Figure): Plotly figure to save.
        path (str): destination HTML path.
        open_after (bool): when True, open the file after writing.

    Returns:
        None

    Examples:
        >>> from mc_robo_utils.plotting import plot_series, save_html
        >>> fig = plot_series([0, 1], [0, 1])
        >>> save_html(fig, "plot.html", open_after=False)
    """
    fig.write_html(path)
    if open_after:
        import subprocess

        subprocess.run(["open", path], check=False)
