# geomap_utils/spiral.py
from __future__ import annotations
from typing import Any, List, Tuple, Dict
import math
import numpy as np

def archimedean_spiral(center_xy: Tuple[float,float],
                        r_out: float,
                        d_m: float,
                        r_in: float,
                        *,
                        step_m: float | None = None) -> List[Tuple[float,float]]:
    """Generate a constant-spacing Archimedean spiral polyline.

    The spiral follows ``r(theta) = r_out - b*theta`` with ``2πb = d_m`` and proceeds
    inward from ``r_out`` to ``r_in``.

    Args:
        center_xy (Tuple[float, float]): spiral center in world coordinates.
        r_out (float): outer radius in metres.
        d_m (float): spacing between successive turns in metres.
        r_in (float): inner radius cutoff in metres.
        step_m (float | None): approximate step length along the curve (defaults to ``0.5 * d_m``).

    Returns:
        List[Tuple[float, float]]: list of ``(x, y)`` points defining the spiral polyline.

    Examples:
        >>> from mc_geomap_utils.spiral import archimedean_spiral
        >>> pts = archimedean_spiral((0.0, 0.0), r_out=10.0, d_m=2.0, r_in=1.0)
    """
    if step_m is None:
        step_m = 0.5 * d_m
    b = d_m / (2.0 * math.pi)
    xc, yc = center_xy
    pts: List[Tuple[float,float]] = []
    theta = 0.0
    r = r_out
    # Integrate inward with adaptive dθ so chord length ≈ step_m
    while r > max(r_in, 1e-6):
        x = xc + r * math.cos(theta)
        y = yc + r * math.sin(theta)
        pts.append((x, y))
        # local differential arc length ds ≈ √(r^2 + b^2) dθ  => dθ = step / √(r^2 + b^2)
        dtheta = step_m / math.sqrt(r*r + b*b)
        theta += dtheta
        r = r_out - b*theta
    return pts

def clip_polyline_to_mask(points_xy: List[Tuple[float,float]],
                            mask: np.ndarray,
                            transform: Any) -> List[List[Tuple[float,float]]]:
    """Clip a polyline to a raster mask and return contiguous valid segments.

    Args:
        points_xy (List[Tuple[float, float]]): input polyline points in world coordinates.
        mask (np.ndarray): boolean raster mask (True = keep).
        transform (Any): affine transform mapping pixel centers to world coordinates.

    Returns:
        List[List[Tuple[float, float]]]: list of polyline segments contained within the mask.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.spiral import clip_polyline_to_mask
        >>> mask = np.ones((10, 10), dtype=bool)
        >>> class T: a=1; b=0; c=0; d=0; e=-1; f=0
        >>> segs = clip_polyline_to_mask([(0.5, -0.5), (5.5, -5.5)], mask, T())
    """
    from typing import List as _List
    def xy_to_px(x: float, y: float) -> tuple[int,int] | None:
        a,b,c0,d,e,f = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
        det = a*e - b*d
        col = (( e*(x - c0) - b*(y - f)) / det) - 0.5
        row = ((-d*(x - c0) + a*(y - f)) / det) - 0.5
        ci = int(round(col)); ri = int(round(row))
        if 0 <= ri < mask.shape[0] and 0 <= ci < mask.shape[1]:
            return (ri, ci)
        return None

    segs: _List[_List[Tuple[float,float]]] = []
    cur: _List[Tuple[float,float]] = []
    for (x, y) in points_xy:
        rc = xy_to_px(x, y)
        ok = (rc is not None) and bool(mask[rc])
        if ok:
            cur.append((x, y))
        else:
            if len(cur) >= 2:
                segs.append(cur)
            cur = []
    if len(cur) >= 2:
        segs.append(cur)
    return segs
