# geomap_utils/planning.py
from __future__ import annotations
from typing import Any, Iterable, List, Tuple, Dict
import math
import numpy as np
from scipy import ndimage as ndi
try:
    from shapely.geometry import shape as _shp_shape, LineString as _LineString, Polygon as _Polygon
    from shapely.ops import unary_union as _unary_union
    from rasterio.features import shapes as _rio_shapes
except Exception:  # pragma: no cover
    _shp_shape = None
    _LineString = None
    _Polygon = None
    _unary_union = None
    _rio_shapes = None

def _px_to_xy(transform: Any, cols: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized pixel-center to world coordinates for a rasterio Affine-like transform."""
    # x = a*col + b*row + c ; y = d*col + e*row + f  (with col,row at centers)
    c = cols.astype(float) + 0.5
    r = rows.astype(float) + 0.5
    a,b,c0,d,e,f = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
    x = a*c + b*r + c0
    y = d*c + e*r + f
    return x, y

def _rotate_xy(x: np.ndarray, y: np.ndarray, psi_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate world coordinates by -psi to align strips with the x'-axis."""
    c, s = math.cos(psi_rad), math.sin(psi_rad)
    # [u] = [ c  s][x]
    # [v]   [-s  c][y]
    u =  c * x + s * y
    v = -s * x + c * y
    return u, v

def region_mask(macro_labels: np.ndarray, rid: int) -> np.ndarray:
    """Return a boolean mask for a single macroregion id.

    Args:
        macro_labels (np.ndarray): macroregion label raster.
        rid (int): region id to extract.

    Returns:
        np.ndarray: boolean mask where ``macro_labels == rid``.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.planning import region_mask
        >>> region_mask(np.array([[1, 2], [2, 2]]), 2)
    """
    return (macro_labels == int(rid))

def _split_by_gaps(xs: np.ndarray, ys: np.ndarray, gap_m: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a sorted point sequence into segments when the jump exceeds gap_m."""
    if xs.size == 0:
        return []
    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i in range(1, xs.size):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        if math.hypot(dx, dy) > gap_m:
            segs.append((xs[start:i], ys[start:i]))
            start = i
    segs.append((xs[start:], ys[start:]))
    return segs


def _mask_to_polygon(mask: np.ndarray, transform: Any) -> Any | None:
    """Convert a boolean mask to a Shapely polygon (union of shapes)."""
    if (_rio_shapes is None) or (_shp_shape is None) or (_unary_union is None) or (_Polygon is None):
        return None
    geoms = []
    for geom, val in _rio_shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if val:
            geoms.append(_shp_shape(geom))
    if not geoms:
        return None
    return _unary_union(geoms)


def generate_serpentine_tracks_polygon(mask: np.ndarray,
                                        transform: Any,
                                        psi_deg: float,
                                        spacing_m: float,
                                        *,
                                        min_seg_len_m: float | None = None) -> List[List[Tuple[float, float]]]:
    """Generate parallel strips by intersecting an AOI polygon with evenly spaced lines.

    Args:
        mask (np.ndarray): boolean AOI mask.
        transform (Any): affine transform mapping pixel centers to world coords.
        psi_deg (float): heading angle in degrees.
        spacing_m (float): spacing between tracks in metres.
        min_seg_len_m (float | None): minimum segment length to keep.

    Returns:
        List[List[Tuple[float, float]]]: list of polylines, each a list of ``(x, y)`` points.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.planning import generate_serpentine_tracks_polygon
        >>> mask = np.ones((5, 5), dtype=bool)
        >>> tracks = generate_serpentine_tracks_polygon(mask, transform=None, psi_deg=0.0, spacing_m=1.0)
    """
    if _LineString is None:
        return []
    if min_seg_len_m is None:
        min_seg_len_m = 2.0 * spacing_m
    poly = _mask_to_polygon(mask, transform)
    if poly is None or poly.is_empty:
        return []

    psi = math.radians(psi_deg)
    u = np.array([math.cos(psi), math.sin(psi)], dtype=float)     # along-track
    n = np.array([-math.sin(psi), math.cos(psi)], dtype=float)    # spacing normal

    def _minmax_proj(p: Any) -> Tuple[float, float]:
        coords = np.array(p.exterior.coords, dtype=float)
        proj = coords @ n
        return float(np.min(proj)), float(np.max(proj))

    mins = []; maxs = []
    if _Polygon is not None and isinstance(poly, _Polygon):
        mn, mx = _minmax_proj(poly)
        mins.append(mn); maxs.append(mx)
    else:
        try:
            for geom in poly.geoms:  # type: ignore[union-attr]
                mn, mx = _minmax_proj(geom)
                mins.append(mn); maxs.append(mx)
        except Exception:
            return []
    vmin = min(mins); vmax = max(maxs)
    width = float(np.hypot(*(poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1]))) * 2.0

    segs_out: List[List[Tuple[float, float]]] = []
    k_start = int(math.floor(vmin / spacing_m)) - 1
    k_end   = int(math.ceil(vmax / spacing_m)) + 1
    for k in range(k_start, k_end + 1):
        offset = k * spacing_m
        q = offset * n
        p0 = q - u * width
        p1 = q + u * width
        line = _LineString([tuple(p0.tolist()), tuple(p1.tolist())])
        inter = poly.intersection(line)
        if inter.is_empty:
            continue
        if inter.geom_type == "LineString":
            lines = [inter]
        else:
            try:
                lines = list(inter.geoms)  # type: ignore[attr-defined]
            except Exception:
                lines = []
        for ln in lines:
            if ln.length < min_seg_len_m:
                continue
            coords = list(ln.coords)
            segs_out.append([(float(x), float(y)) for (x, y) in coords])
    return segs_out

def generate_serpentine_tracks(mask: np.ndarray,
                                transform: Any,
                                psi_deg: float,
                                spacing_m: float,
                                cell_m: float,
                                min_seg_len_m: float | None = None,
                                band_tol_frac: float = 0.35) -> List[List[Tuple[float,float]]]:
    """Raster-native generation of serpentine strips inside a binary mask.

    Args:
        mask (np.ndarray): boolean AOI mask.
        transform (Any): affine transform mapping pixel centers to world coords.
        psi_deg (float): heading angle in degrees.
        spacing_m (float): spacing between tracks in metres.
        cell_m (float): raster cell size in metres.
        min_seg_len_m (float | None): minimum segment length to keep.
        band_tol_frac (float): band tolerance as a fraction of spacing.

    Returns:
        List[List[Tuple[float, float]]]: list of polylines in serpentine order.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.planning import generate_serpentine_tracks
        >>> mask = np.ones((5, 5), dtype=bool)
        >>> tracks = generate_serpentine_tracks(mask, transform=None, psi_deg=0.0, spacing_m=1.0, cell_m=1.0)
    """
    if min_seg_len_m is None:
        min_seg_len_m = 2.0 * spacing_m

    # Fill holes / smooth jagged edges so strips stay continuous (fewer turns)
    mask = np.asarray(ndi.binary_closing(mask, iterations=2), dtype=bool)
    mask = np.asarray(ndi.binary_fill_holes(mask), dtype=bool)

    rows, cols = np.where(mask)
    if rows.size == 0:
        return []

    # pixel centers -> world
    x, y = _px_to_xy(transform, cols, rows)

    # rotate so strips align with u-axis
    psi = math.radians(psi_deg)
    u, v = _rotate_xy(x, y, psi)

    # Define strip indices by binning v with width spacing_m
    vmin = float(np.min(v)); vmax = float(np.max(v))
    # Offset so first strip sits just inside vmin
    v0 = vmin + 0.5 * spacing_m
    k = np.round((v - v0) / spacing_m).astype(int)
    # Band tolerance around the center line (in meters)
    band_tol = band_tol_frac * spacing_m
    # Allow bigger jumps before splitting to avoid micro-segments inside one strip
    gap_m = max(2.0 * cell_m, 0.8 * spacing_m)

    # Build mapping k -> points close to that strip
    tracks: List[List[Tuple[float,float]]] = []
    for ki in range(int(np.min(k)), int(np.max(k)) + 1):
        idx = (k == ki) & (np.abs(v - (v0 + ki * spacing_m)) <= band_tol)
        if not np.any(idx):
            continue
        # sort by u to trace the line in order
        order = np.argsort(u[idx])
        xs = x[idx][order]
        ys = y[idx][order]
        # break into segments where geometric jump is large
        segs = _split_by_gaps(xs, ys, gap_m=gap_m)
        if not segs:
            continue
        # keep only the longest segment on this strip (reduces micro-strips)
        segs_sorted = sorted(segs, key=lambda s: float(np.sum(np.hypot(np.diff(s[0]), np.diff(s[1])))), reverse=True)
        sx, sy = segs_sorted[0]
        if sx.size < 2:
            continue
        length = float(np.sum(np.hypot(np.diff(sx), np.diff(sy))))
        if length >= min_seg_len_m:
            tracks.append(list(zip(sx.tolist(), sy.tolist())))

    # Serpentine ordering: alternate direction per consecutive strip
    tracks_ordered: List[List[Tuple[float,float]]] = []
    if not tracks:
        return tracks_ordered
    # sort tracks by the mean v of their points (approximate strip order)
    v_means = []
    for t in tracks:
        tx = np.array([p[0] for p in t]); ty = np.array([p[1] for p in t])
        tu, tv = _rotate_xy(tx, ty, psi)
        v_means.append(float(np.mean(tv)))
    order = np.argsort(v_means)
    for j, idx in enumerate(order):
        seg = tracks[idx]
        if j % 2 == 1:
            seg = list(reversed(seg))
        tracks_ordered.append(seg)
    return tracks_ordered

def attach_altitude_and_tilt(polylines: List[List[Tuple[float,float]]],
                            dem: np.ndarray,
                            transform: Any,
                            h_m: float,
                            beta_deg: float) -> List[List[Dict]]:
    """Sample DEM under each waypoint and attach z and tilt.

    Args:
        polylines (List[List[Tuple[float, float]]]): list of track polylines.
        dem (np.ndarray): DEM raster for altitude sampling.
        transform (Any): affine transform mapping pixel centers to world coords.
        h_m (float): altitude offset to add above the DEM.
        beta_deg (float): tilt angle (degrees) assigned to each waypoint.

    Returns:
        List[List[Dict]]: list of enriched waypoints with ``x``, ``y``, ``z``, and ``tilt_deg``.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.planning import attach_altitude_and_tilt
        >>> polylines = [[(0.0, 0.0), (1.0, 1.0)]]
        >>> dem = np.zeros((5, 5))
        >>> out = attach_altitude_and_tilt(polylines, dem, transform=None, h_m=2.0, beta_deg=5.0)
    """
    a,b,c0,d,e,f = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
    out: List[List[Dict]] = []
    H, W = dem.shape
    for seg in polylines:
        enriched: List[Dict] = []
        for (x, y) in seg:
            # invert affine: col,row from x,y (nearest)
            # Solve: x = a*c + b*r + c0 ; y = d*c + e*r + f
            # For north-up grids: b=d=0, so this is fast:
            if abs(b) < 1e-12 and abs(d) < 1e-12:
                col = (x - c0) / a - 0.5
                row = (y - f)  / e - 0.5
            else:
                # generic 2x2 inverse
                det = a*e - b*d
                col = (( e*(x - c0) - b*(y - f)) / det) - 0.5
                row = ((-d*(x - c0) + a*(y - f)) / det) - 0.5
            ci = int(round(col)); ri = int(round(row))
            if 0 <= ri < H and 0 <= ci < W and np.isfinite(dem[ri, ci]):
                z = float(dem[ri, ci]) + float(h_m)
            else:
                z = float(np.nan)  # outside support; we leave it NaN
            enriched.append({
                "x": float(x), "y": float(y), "z": z,
                "tilt_deg": float(beta_deg)
            })
        out.append(enriched)
    return out

def estimate_turn_count(tracks: List[List[Tuple[float,float]]]) -> int:
    """Approximate number of turns (ends of segments) for quick reporting.

    Args:
        tracks (List[List[Tuple[float, float]]]): list of track polylines.

    Returns:
        int: estimated number of turns (segments minus one).

    Examples:
        >>> from mc_geomap_utils.planning import estimate_turn_count
        >>> estimate_turn_count([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])
        1
    """
    return max(0, sum(1 for _ in tracks) - 1)

def polyline_length_m(points_xy: List[Tuple[float,float]]) -> float:
    """Compute total length of a polyline.

    Args:
        points_xy (List[Tuple[float, float]]): list of ``(x, y)`` points.

    Returns:
        float: polyline length in metres.

    Examples:
        >>> from mc_geomap_utils.planning import polyline_length_m
        >>> polyline_length_m([(0, 0), (3, 4)])
        5.0
    """
    if len(points_xy) < 2:
        return 0.0
    p = np.asarray(points_xy, dtype=float)
    return float(np.sum(np.hypot(np.diff(p[:,0]), np.diff(p[:,1]))))

def tracks_total_length_m(tracks: List[List[Tuple[float,float]]]) -> float:
    """Compute total length across multiple polylines.

    Args:
        tracks (List[List[Tuple[float, float]]]): list of polylines.

    Returns:
        float: total length in metres.

    Examples:
        >>> from mc_geomap_utils.planning import tracks_total_length_m
        >>> tracks_total_length_m([[(0, 0), (1, 0)], [(1, 0), (1, 1)]])
        2.0
    """
    return float(sum(polyline_length_m(t) for t in tracks))


def merge_tracks_continuous(tracks: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Concatenate ordered segments into a single continuous path (bridging jumps).

    Each input list is assumed to be ordered along-strip. We connect the end of one
    segment to the start of the next to keep one continuous polyline per region.

    Args:
        tracks (List[List[Tuple[float, float]]]): ordered track segments.

    Returns:
        List[Tuple[float, float]]: merged continuous polyline.

    Examples:
        >>> from mc_geomap_utils.planning import merge_tracks_continuous
        >>> merged = merge_tracks_continuous([[(0, 0), (1, 0)], [(2, 0), (3, 0)]])
    """
    merged: List[Tuple[float, float]] = []
    for seg in tracks:
        if len(seg) < 2:
            continue
        if merged:
            merged.append(seg[0])  # bridge with a straight hop
        merged.extend(seg)
    return merged


def straighten_tracks(tracks: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
    """Replace each strip polyline with a straight line between its endpoints.

    Args:
        tracks (List[List[Tuple[float, float]]]): list of polylines.

    Returns:
        List[List[Tuple[float, float]]]: list of straightened polylines.

    Examples:
        >>> from mc_geomap_utils.planning import straighten_tracks
        >>> out = straighten_tracks([[(0, 0), (1, 1), (2, 0)]])
    """
    out: List[List[Tuple[float, float]]] = []
    for seg in tracks:
        if len(seg) < 2:
            continue
        out.append([seg[0], seg[-1]])
    return out


def smooth_polyline(points: List[Tuple[float, float]], window: int = 5) -> List[Tuple[float, float]]:
    """Apply a simple moving average to a polyline to soften stair-steps.

    Args:
        points (List[Tuple[float, float]]): polyline points.
        window (int): moving average window size (odd, >=3).

    Returns:
        List[Tuple[float, float]]: smoothed polyline points.

    Examples:
        >>> from mc_geomap_utils.planning import smooth_polyline
        >>> out = smooth_polyline([(0, 0), (1, 1), (2, 0)], window=3)
    """
    n = len(points)
    if n < 3:
        return points
    w = max(3, int(window) | 1)  # odd window >=3
    half = w // 2
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    k = np.ones(w, dtype=float) / float(w)
    # pad with edge values to preserve endpoints
    xs_pad = np.pad(xs, (half, half), mode="edge")
    ys_pad = np.pad(ys, (half, half), mode="edge")
    xs_s = np.convolve(xs_pad, k, mode="valid")
    ys_s = np.convolve(ys_pad, k, mode="valid")
    return list(zip(xs_s.tolist(), ys_s.tolist()))


def simplify_polyline_rdp(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """Douglas–Peucker simplification to reduce vertices while keeping shape.

    Args:
        points (List[Tuple[float, float]]): polyline points.
        epsilon (float): maximum deviation allowed (metres).

    Returns:
        List[Tuple[float, float]]: simplified polyline points.

    Examples:
        >>> from mc_geomap_utils.planning import simplify_polyline_rdp
        >>> out = simplify_polyline_rdp([(0, 0), (1, 0.1), (2, 0)], epsilon=0.2)
    """
    if len(points) < 3 or epsilon <= 0:
        return points
    pts = np.array(points, dtype=float)
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack: List[Tuple[int, int]] = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        p0, p1 = pts[i0], pts[i1]
        v = p1 - p0
        norm2 = float(np.dot(v, v))
        if norm2 == 0.0:
            continue
        # perpendicular distance of each point to segment p0->p1
        t = np.clip(((pts[i0 + 1:i1] - p0) @ v) / norm2, 0.0, 1.0)
        proj = p0 + t[:, None] * v
        dist = np.sqrt(np.sum((pts[i0 + 1:i1] - proj) ** 2, axis=1))
        j_rel = int(np.argmax(dist))
        dmax = float(dist[j_rel])
        if dmax > epsilon:
            j = i0 + 1 + j_rel
            keep[j] = True
            stack.append((i0, j))
            stack.append((j, i1))
    simplified = pts[keep]
    return list(map(tuple, simplified.tolist()))
