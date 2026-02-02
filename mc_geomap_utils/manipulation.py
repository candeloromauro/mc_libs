from __future__ import annotations

from enum import Enum
from typing import Iterable, Any

import numpy as np
from numpy.typing import NDArray

try:  # optional dependency for coordinate reprojection
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - pyproj might be missing
    CRS = None  # type: ignore[assignment]
    Transformer = None  # type: ignore[assignment]


class Mode(str, Enum):
    """Strategies for choosing the translation origin."""
    FIRST = "first_point"
    MIN = "min_point"
    CUSTOM = "custom_point"


def shift_point_cloud(points: NDArray[np.floating], *, mode: Mode | str = Mode.FIRST, origin: Iterable[float] | None = None) -> NDArray[np.float64]:
    """Translate a point cloud so a chosen origin maps to (0, 0, ...).

    Args:
        points (NDArray[np.floating]): input coordinates with shape ``(N, >=2)``.
        mode (Mode | str): origin-selection strategy (first/min/custom point).
        origin (Iterable[float] | None): explicit origin coordinates when ``mode`` is custom.

    Returns:
        NDArray[np.float64]: shifted copy of the input point cloud.

    Raises:
        ValueError: If the input shape is invalid or the mode/origin are incompatible.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.manipulation import shift_point_cloud
        >>> pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> shift_point_cloud(pts, mode="min")
    """

    # Checking arguments
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Expected an array of shape (N, >=2) for point cloud data.")
    try:
        shift_mode = Mode(mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported shift mode: {mode}") from exc

    # Shifting dataset based on selected mode
    if shift_mode is Mode.FIRST:
        origin_vec = arr[0]
    elif shift_mode is Mode.MIN:
        origin_vec = arr.min(axis=0)
    else:  # Mode.CUSTOM
        if origin is None:
            raise ValueError("Mode.CUSTOM requires an explicit origin.")
        origin_vec = np.asarray(origin, dtype=arr.dtype)
        if origin_vec.shape != arr.shape[1:]:
            raise ValueError(
                f"Origin shape {origin_vec.shape} incompatible with points {arr.shape}."
            )

    return arr - origin_vec


def grid_to_point_cloud(
    raster: np.ndarray,
    transform: Any,
    *,
    src_crs: Any | None = None,
    dst_crs: Any | None = None,
) -> NDArray[np.float64]:
    """Convert a 2-D raster into ``(x, y, z)`` point samples with optional reprojection.

    Args:
        raster (np.ndarray): grid of Z values (shape ``rows x cols``).
        transform (Any): affine transform describing pixel center coordinates.
        src_crs (Any | None): CRS describing the raster coordinates (defaults to ``EPSG:4326``).
        dst_crs (Any | None): CRS for the output XY values (skip reprojection when ``None``).

    Returns:
        NDArray[np.float64]: flattened point cloud with columns ``[x, y, z]``.

    Raises:
        ValueError: If the input raster is invalid or contains no finite samples.
        RuntimeError: If reprojection is requested but ``pyproj`` is unavailable.
        TypeError: If the transform object is missing required affine attributes.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.manipulation import grid_to_point_cloud
        >>> raster = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> class T: a=1; b=0; c=0; d=0; e=-1; f=0
        >>> pts = grid_to_point_cloud(raster, T())
    """

    arr = np.asarray(raster, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2-D raster array.")

    mask = np.isfinite(arr)
    if not mask.any():
        raise ValueError("Raster does not contain any finite samples.")

    rows, cols = np.nonzero(mask)
    values = arr[rows, cols]

    try:
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    except AttributeError as exc:  # pragma: no cover - depends on transform implementation
        raise TypeError("Transform object must expose attributes a, b, c, d, e, f.") from exc

    # [GPT-5 Pro patch] use pixel-center convention (+0.5) for raster->world mapping
    cols_f = cols.astype(float) + 0.5
    rows_f = rows.astype(float) + 0.5
    xs = a * cols_f + b * rows_f + c
    ys = d * cols_f + e * rows_f + f

    if dst_crs is not None:
        if Transformer is None or CRS is None:
            raise RuntimeError("pyproj is required for coordinate reprojection. Install pyproj and retry.")
        source_crs = CRS.from_user_input(src_crs or "EPSG:4326")
        target_crs = CRS.from_user_input(dst_crs)
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        xs, ys = transformer.transform(xs, ys)

    return np.column_stack([xs, ys, values])
