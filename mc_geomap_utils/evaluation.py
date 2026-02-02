from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
import numpy as np


def array_stats(arr: np.ndarray) -> dict[str, float]:
    """Summarise the finite values of a raster with key statistics.

    Args:
        arr (np.ndarray): array from which to compute summary statistics.

    Returns:
        dict[str, float]: dictionary containing counts, percentiles, and extrema.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.evaluation import array_stats
        >>> stats = array_stats(np.array([1.0, 2.0, np.nan]))
        >>> stats["valid_px"]
        2.0
    """

    finite = np.isfinite(arr)
    total = int(arr.size)
    valid = int(finite.sum())
    out: dict[str, float] = {
        "valid_px": float(valid),
        "total_px": float(total),
        "nan_pct": 100.0 * (total - valid) / max(1, total),
    }
    if valid:
        vals = arr[finite]
        p2, p50, p98 = np.nanpercentile(vals, [2, 50, 98])
        out.update({
            "min": float(np.nanmin(vals)),
            "p2": float(p2),
            "median": float(p50),
            "p98": float(p98),
            "max": float(np.nanmax(vals)),
        })
    else:
        out.update({"min": np.nan, "p2": np.nan, "median": np.nan, "p98": np.nan, "max": np.nan})
    return out


def print_array_stats(arr: np.ndarray, name: str) -> None:
    """Pretty-print summary statistics produced by ``array_stats``.

    Args:
        arr (np.ndarray): array to analyse.
        name (str): label used in the console output.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.evaluation import print_array_stats
        >>> print_array_stats(np.array([1.0, 2.0, np.nan]), "demo")
    """

    s = array_stats(arr)

    def _fmt(x: Any) -> str:
        """Format numeric values for rendering into the stats message.

        Args:
            x (Any): value to convert to text.

        Returns:
            str: formatted number or 'NaN'.
        """
        return f"{x:.3f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "NaN"

    print(
        f"[stats] {name}: min={_fmt(s['min'])} "
        f"p2={_fmt(s['p2'])} med={_fmt(s['median'])}/np98={_fmt(s['p98'])} "
        f"max={_fmt(s['max'])} NaN%={s['nan_pct']:.1f}"
    )


def export_label_table(labels: np.ndarray, cell: float, path: Path | str) -> None:
    """Write CSV with per-label pixel counts and area in square metres.

    Args:
        labels (np.ndarray): integer label raster to summarise.
        cell (float): cell size used to convert pixel counts to area.
        path (Path | str): destination CSV file path.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.evaluation import export_label_table
        >>> export_label_table(np.array([[0, 1], [1, 2]]), 0.5, "labels.csv")
    """

    uniq, counts = np.unique(labels, return_counts=True)
    rows = []
    for lab, cnt in zip(uniq, counts):
        if int(lab) == 0:
            continue
        rows.append((int(lab), int(cnt), float(cnt) * (cell * cell)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "pixels", "area_m2"])
        w.writerows(rows)
    print(f"[write] label summary -> {path}")


def export_feature_stats_table(features: dict[str, np.ndarray], path: Path | str) -> None:
    """Serialise feature statistics to CSV using ``array_stats``.

    Args:
        features (dict[str, np.ndarray]): mapping of feature name to raster.
        path (Path | str): destination CSV file path.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.evaluation import export_feature_stats_table
        >>> export_feature_stats_table({"slope": np.array([[1.0, np.nan]])}, "features.csv")
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "min", "p2", "median", "p98", "max", "nan_pct", "valid_px", "total_px"])
        for name, arr in features.items():
            s = array_stats(arr)
            w.writerow(
                [
                    name,
                    s.get("min"),
                    s.get("p2"),
                    s.get("median"),
                    s.get("p98"),
                    s.get("max"),
                    s.get("nan_pct"),
                    s.get("valid_px"),
                    s.get("total_px"),
                ]
            )
    print(f"[write] feature stats -> {path}")
