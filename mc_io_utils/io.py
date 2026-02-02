"""IO helpers for point clouds and raster outputs.

Functions:
- list_files(folder, pattern="*", recursive=False)
- load_points(file_path, mode=...)
- save_points(file_path, points)
- load_grd(...)
- write_geotiff(...)

Example:
>>> from mc_io_utils.io import load_points
>>> pts = load_points(Path("points.xyz"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List
from enum import Enum

import numpy as np
from numpy.typing import NDArray

try:  # optional dependency for GeoTIFF writing
    import rasterio
except Exception:  # pragma: no cover - rasterio might be missing
    rasterio = None  # type: ignore[assignment]


class PointFileMode(str, Enum):
    """Layout of coordinates inside a plain-text point cloud file."""

    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


# Prefixed alias for mc_* libraries.
McPointFileMode = PointFileMode


def load_points(file_path: Path, *, mode: PointFileMode | str = PointFileMode.VERTICAL) -> NDArray[np.float64]:
    """Load a whitespace-delimited point cloud into a NumPy array.

    Args:
        file_path (Path): absolute path to the source XYZ text file.
        mode (PointFileMode | str): "vertical" for one point per line, "horizontal" for per-axis rows.

    Returns:
        NDArray[np.float64]: array with columns ``x, y, z`` describing each point.

    Raises:
        ValueError: If the file contents are malformed or incompatible with the chosen layout.
        FileNotFoundError: If ``file_path`` does not exist.

    Examples:
        >>> from pathlib import Path
        >>> from mc_io_utils.io import load_points
        >>> pts = load_points(Path("points.xyz"))
    """
    try:
        layout = PointFileMode(mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported layout mode: {mode}") from exc

    rows: List[List[float]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or "nan" in line.lower():
                continue
            parts = line.split()
            if layout is PointFileMode.VERTICAL and len(parts) < 3:
                raise ValueError(f"Expected at least 3 values on line {idx}: {raw_line!r}")
            try:
                values = [float(value) for value in parts]
            except ValueError as exc:
                raise ValueError(f"Failed to parse floats on line {idx}: {raw_line!r}") from exc
            rows.append(values)

    if not rows:
        raise ValueError(f"No valid points found in {file_path}")

    # Discards any potential extra column after the third if layout is vertical
    if layout is PointFileMode.VERTICAL:
        cleaned = [[value for value in row[:3]] for row in rows]
        return np.asarray(cleaned, dtype=float)

    # Horizontal layout: each row corresponds to a coordinate axis
    num_axes = len(rows)
    if num_axes < 3:
        raise ValueError(f"Expected at least 3 rows for horizontal layout in {file_path}")

    target_len = len(rows[0])
    if target_len == 0:
        raise ValueError(f"Row 1 in {file_path} does not contain any numeric values")
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != target_len:
            raise ValueError(f"Row {idx} length {len(row)} does not match row 1 length {target_len}")

    arr = np.asarray(rows[:3], dtype=float)
    return arr.T


def list_files(folder: str | Path, pattern: str = "*", recursive: bool = False) -> List[Path]:
    """List files in a folder matching a glob pattern.

    Args:
        folder (str | Path): directory to search.
        pattern (str): glob pattern, e.g. \"*.txt\".
        recursive (bool): when True, search recursively (uses rglob).

    Returns:
        List[Path]: matching file paths.

    Raises:
        FileNotFoundError: If ``folder`` does not exist or is not a directory.

    Examples:
        >>> from mc_io_utils.io import list_files
        >>> files = list_files("data", "*.txt", recursive=True)
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    it = p.rglob(pattern) if recursive else p.glob(pattern)
    return [f for f in it if f.is_file()]


def save_points(file_path: Path, points: np.ndarray) -> Path:
    """Persist a point cloud next to the original file with a ``*_cleaned`` suffix.

    Args:
        file_path (Path): original input file whose directory will host the cleaned copy.
        points (np.ndarray): coordinates to be written (shape N x 3).

    Returns:
        Path: filesystem path of the generated ``*_cleaned.txt`` file.

    Examples:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from mc_io_utils.io import save_points
        >>> out_path = save_points(Path("points.xyz"), np.zeros((3, 3)))
    """
    path = Path(file_path)
    cleaned_path = path.with_name(f"{path.stem}_cleaned.txt")
    np.savetxt(cleaned_path, points, fmt="%.6f", delimiter="	")
    return cleaned_path


def load_grd(
    file_path: Path | str,
    *,
    assume_wgs84: bool = True,
) -> tuple[np.ndarray, Any, Any | None]:
    """Load a GMT/NetCDF ``.grd`` file and return its raster plus metadata.

    Args:
        file_path (Path | str): path to the grid file on disk.
        assume_wgs84 (bool): inject ``EPSG:4326`` when the dataset lacks a CRS.

    Returns:
        tuple[np.ndarray, Any, Any | None]: data array, affine transform, and CRS (if present).

    Raises:
        FileNotFoundError: If the grid file does not exist.
        RuntimeError: If ``rasterio`` is unavailable.

    Examples:
        >>> from mc_io_utils.io import load_grd
        >>> data, transform, crs = load_grd("surface.grd")
    """

    if rasterio is None:  # pragma: no cover - rasterio optional at runtime
        raise RuntimeError("rasterio is required to load .grd files. Install rasterio and retry.")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Grid file not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1, masked=True).filled(np.nan)
        transform = src.transform
        crs = src.crs

    if crs is None and assume_wgs84:
        try:
            from rasterio.crs import CRS
        except Exception:  # pragma: no cover - shouldn't happen with rasterio installed
            CRS = None  # type: ignore[assignment]
        if CRS is not None:
            crs = CRS.from_epsg(4326)

    return np.asarray(data, dtype=float), transform, crs


def write_geotiff(
    path: Path | str,
    arr: np.ndarray,
    transform: Any,
    *,
    crs: str | None = None,
    nodata: float | None = np.nan,
) -> None:
    """Persist a NumPy array as a single-band GeoTIFF (with .npy fallback).

    Args:
        path (Path | str): destination file path for the raster.
        arr (np.ndarray): image data to be written.
        transform (Any): affine transform describing pixel-to-world mapping.
        crs (str | None): coordinate reference system identifier.
        nodata (float | None): value representing no-data cells.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_io_utils.io import write_geotiff
        >>> write_geotiff("out.tif", np.zeros((10, 10)), transform=None)
    """
    target = Path(path)
    if rasterio is None or transform is None:
        np.save(target.with_suffix(".npy"), arr)
        print(f"[warn] rasterio not available; saved {target.with_suffix('.npy').name} instead of GeoTIFF")
        return

    nd = nodata
    if nd is None and np.issubdtype(arr.dtype, np.integer):
        nd = -9999

    with rasterio.open(
        target,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=str(arr.dtype),
        crs=crs,
        transform=transform,
        nodata=nd,
    ) as dst:
        dst.write(arr, 1)
