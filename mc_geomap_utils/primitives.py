# geomap_utils/primitives.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional, Any, cast
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import disk
from scipy import ndimage as ndi
import logging

@dataclass
class Knoll:
    rid: int                  # macroregion id owning the center
    cy: float                 # center row (pixel)
    cx: float                 # center col (pixel)
    r_m: float                # chosen ring radius (metres)
    r_px: float               # chosen ring radius (pixels)
    score: float              # ring strength score
    cov: float                # coverage fraction
    cv: float                 # circularity (coef. of variation of radius)
    d_m: float                # local spacing to respect
    psi_deg: float            # local heading (keep in table for completeness)
    height_m: float | None = None
    seed_r: int | None = None
    seed_c: int | None = None
    boundary_rc: list[tuple[int,int]] | None = None
    width_m: float | None = None
    length_m: float | None = None
    ellipse_major_m: float | None = None
    ellipse_minor_m: float | None = None
    ellipse_theta_deg: float | None = None
    footprint_rc: list[tuple[int,int]] | None = None

def _grad_mag_deg(slope_deg: np.ndarray) -> np.ndarray:
    """Gradient magnitude of slope (deg) used to highlight rims."""
    sd = np.nan_to_num(slope_deg, nan=0.0)
    gx = ndi.sobel(sd, axis=1, mode="nearest") / 8.0
    gy = ndi.sobel(sd, axis=0, mode="nearest") / 8.0
    return np.hypot(gx, gy)

def _annulus_mask(H: int, W: int, cy: float, cx: float, r_px: float, w_px: float) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (rr >= (r_px - w_px)) & (rr <= (r_px + w_px))

def _boundary_mask_bool(mask: np.ndarray) -> np.ndarray:
    """1-pixel boundary of a binary mask (8-connected)."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    er = ndi.binary_erosion(mask, structure=np.ones((3,3), bool))
    return mask & np.logical_not(er)

def _ellipse_from_mask(mask: np.ndarray, cell_m: float) -> tuple[float | None, float | None, float | None]:
    """Approximate a mask by an ellipse via second moments; return (major_m, minor_m, theta_deg)."""
    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        return None, None, None
    xy = np.column_stack((xs, ys)).astype(float)
    xy -= xy.mean(axis=0, keepdims=True)
    cov = np.cov(xy, rowvar=False)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None, None, None
    vals = np.maximum(vals, 0.0)
    # scale eigenvalues to axis lengths (2σ ellipse)
    axes_px = 2.0 * np.sqrt(vals)
    idx_max = int(np.argmax(axes_px))
    idx_min = 1 - idx_max
    major_m = float(axes_px[idx_max] * cell_m)
    minor_m = float(axes_px[idx_min] * cell_m)
    vx, vy = vecs[:, idx_max]
    theta = float(np.degrees(np.arctan2(vy, vx)))
    return major_m, minor_m, theta

def _footprint_from_descent(
    dem: np.ndarray,
    seed_r: int,
    seed_c: int,
    *,
    cell_m: float,
    rmin_px: float,
    rmax_px_soft: float,
    descent_dmax_m: float,
    dz_m: float,
) -> tuple[np.ndarray | None, float | None, float | None]:
    """Slide a horizontal plane from the seed height down to the ground; return best mask, r_eq_px, r_eq_m."""
    H, W = dem.shape
    if not (0 <= seed_r < H and 0 <= seed_c < W):
        return None, None, None
    z0 = float(dem[seed_r, seed_c])
    if not np.isfinite(z0):
        return None, None, None
    finite = np.isfinite(dem)
    tau_levels = np.arange(z0, z0 - descent_dmax_m - 1e-6, -dz_m)
    best_mask = None
    best_r_eq = None
    for tau in tau_levels:
        mask = finite & (dem >= tau)
        if not mask[seed_r, seed_c]:
            continue
        comp_labels_arr, nlab_val = cast(tuple[np.ndarray, int], ndi.label(mask))
        comp_labels: np.ndarray = np.asarray(comp_labels_arr)
        lab_id = int(comp_labels[seed_r, seed_c])
        if lab_id <= 0:
            continue
        comp = (comp_labels == lab_id)
        r_eq = float(np.sqrt(comp.sum() / np.pi))
        if r_eq < rmin_px or r_eq > rmax_px_soft:
            continue
        best_mask = comp
        best_r_eq = r_eq
        break
    if best_mask is None or best_r_eq is None:
        return None, None, None
    return best_mask, best_r_eq, float(best_r_eq * cell_m)

def _component_at_target_plane(
    dem: np.ndarray,
    seed_r: int,
    seed_c: int,
    target_z: float,
    step: float = 0.25,
) -> tuple[np.ndarray | None, float | None]:
    """Lower a plane from seed height to target_z; return component containing seed at the deepest reachable level."""
    H, W = dem.shape
    if not (0 <= seed_r < H and 0 <= seed_c < W):
        return None, None
    z0 = float(dem[seed_r, seed_c])
    if not np.isfinite(z0):
        return None, None
    finite = np.isfinite(dem)
    if step <= 0:
        step = 0.25
    levels = np.arange(z0, target_z - 1e-6, -step)
    last_mask: np.ndarray | None = None
    last_r_eq: float | None = None
    for lvl in levels:
        mask = finite & (dem >= lvl)
        if not mask[seed_r, seed_c]:
            break
        comp_labels_arr, nlab_val = cast(tuple[np.ndarray, int], ndi.label(mask))
        comp_labels: np.ndarray = np.asarray(comp_labels_arr)
        lab_id = int(comp_labels[seed_r, seed_c])
        if lab_id <= 0:
            break
        comp = (comp_labels == lab_id)
        last_mask = comp
        last_r_eq = float(np.sqrt(comp.sum() / np.pi))
    if last_mask is None or last_r_eq is None:
        return None, None
    return last_mask, last_r_eq

def _ring_stats(G: np.ndarray, cy: float, cx: float, r_px: float, w_px: float, thr: float) -> tuple[float, float]:
    """Return (mean_on_ring, coverage_fraction) from annulus above threshold."""
    H, W = G.shape
    ann = _annulus_mask(H, W, cy, cx, r_px, w_px)
    vals = G[ann]
    if vals.size == 0:
        return 0.0, 0.0
    cov = float(np.mean(vals > thr))
    return float(np.nanmean(vals)), cov

def _estimate_radius_profile(G: np.ndarray, cy: float, cx: float, r_grid: np.ndarray, w_px: float, thr: float) -> tuple[float, float]:
    """Search r_grid for best ring by annular mean; return (r_best_px, mean_val_at_best)."""
    best_v, best_r = -np.inf, float(r_grid[0])
    for r in r_grid:
        v, cov = _ring_stats(G, cy, cx, float(r), w_px, thr)
        if v > best_v:
            best_v, best_r = v, float(r)
    return best_r, best_v

def _circularity(G: np.ndarray, cy: float, cx: float, r_px: float, w_px: float, thr: float, n_angles: int = 180) -> tuple[float, float]:
    """Estimate CV of radius by sampling directions and picking radial maxima of G."""
    H, W = G.shape
    yy, xx = np.indices((H, W))
    radii = []
    cov_cnt = 0
    for t in np.linspace(0, 2*np.pi, n_angles, endpoint=False):
        # sample along a ray
        dx, dy = np.cos(t), np.sin(t)
        # sample candidate radii around r_px
        rs = np.linspace(max(1.0, r_px - 4*w_px), r_px + 4*w_px, 25)
        vals = []
        for r in rs:
            x = cx + r * dx
            y = cy + r * dy
            ix = int(round(x)); iy = int(round(y))
            if 0 <= ix < W and 0 <= iy < H:
                vals.append(G[iy, ix])
            else:
                vals.append(np.nan)
        vals = np.array(vals, float)
        if np.all(~np.isfinite(vals)):
            continue
        k = np.nanargmax(vals)
        r_hat = float(rs[k])
        radii.append(r_hat)
        if np.nanmax(vals) > thr:  # contributes to coverage
            cov_cnt += 1
    if len(radii) < 5:
        return 1.0, 0.0  # poor estimate: treat as non-circular/zero coverage
    radii = np.array(radii, float)
    cv = float(np.nanstd(radii) / (np.nanmean(radii) + 1e-9))
    cov = float(cov_cnt / max(1, n_angles))
    return cv, cov

def detect_knolls(
    feats: dict[str, np.ndarray],
    macro_labels: np.ndarray,
    rows: list[dict],
    *,
    cell_m: float,
    W_m: float,
    slope_ring_deg: float = 12.0,
    circ_cv_max: float = 0.20,
    rmin_mult: float = 1.5,
    rmax_mult: float = 4.0,
    spiral_rmin_m: float = 1.0,
    size_relax: float = 1.0,
    use_grad_seeds: bool = True,  # retained for CLI compatibility
    grad_seed_pct: float = 85.0,
    dem: np.ndarray | None = None,
) -> List[Knoll]:
    """Detect knoll-like blobs on |∇slope| using thresholding and shape filters.

    Args:
        feats (dict[str, np.ndarray]): feature rasters (expects ``"slope"`` and optionally ``"dem"``).
        macro_labels (np.ndarray): macroregion label raster aligned to ``feats``.
        rows (list[dict]): region metadata rows (used for default spacing/heading).
        cell_m (float): raster cell size in metres.
        W_m (float): nominal swath width in metres.
        slope_ring_deg (float): slope ring threshold in degrees.
        circ_cv_max (float): maximum allowed circularity coefficient of variation.
        rmin_mult (float): minimum radius multiplier relative to ``W_m``.
        rmax_mult (float): maximum radius multiplier relative to ``W_m``.
        spiral_rmin_m (float): minimum spiral radius for forced footprints.
        size_relax (float): softening factor for size limits.
        use_grad_seeds (bool): when True, seed from gradient magnitude peaks.
        grad_seed_pct (float): percentile for gradient thresholding when seeding.
        dem (np.ndarray | None): optional DEM for height metadata.

    Returns:
        List[Knoll]: detected knoll objects sorted by score.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.primitives import detect_knolls
        >>> feats = {"slope": np.zeros((10, 10))}
        >>> knolls = detect_knolls(feats, np.zeros((10, 10), dtype=int), [], cell_m=1.0, W_m=3.0, spiral_rmin_m=1.0)
    """
    logger = logging.getLogger("main")

    slope_deg = np.degrees(feats["slope"])
    H, W = slope_deg.shape
    G = _grad_mag_deg(slope_deg)

    finite_G = G[np.isfinite(G)]
    if finite_G.size == 0:
        return []
    try:
        if use_grad_seeds:
            thr_grad = float(np.nanpercentile(finite_G, grad_seed_pct))
        else:
            from skimage.filters import threshold_otsu
            thr_grad = float(threshold_otsu(finite_G))
    except Exception:
        thr_grad = float(np.nanpercentile(finite_G, grad_seed_pct))

    # Size bounds scaled to swath (fallback to reasonable defaults)
    target_r_min_m = max(1.0, rmin_mult * W_m)
    target_r_max_m = max(target_r_min_m + 1.0, rmax_mult * W_m)
    rmin_px = max(1.0, target_r_min_m / cell_m)
    rmax_px = max(rmin_px + 1.0, target_r_max_m / cell_m)
    size_relax = max(1.0, float(size_relax))
    rmax_px_soft = rmax_px * size_relax
    area_min = np.pi * (rmin_px ** 2)
    area_max = np.pi * (rmax_px ** 2)
    area_max_soft = np.pi * (rmax_px_soft ** 2)

    bw: np.ndarray = (G >= thr_grad) & np.isfinite(G)
    labels_arr, nlab_val = cast(tuple[np.ndarray, int], ndi.label(bw))
    labels: np.ndarray = np.asarray(labels_arr)
    nlab_int: int = int(nlab_val)
    logger.info("[primitives] grad blobs=%d thr_grad=%.3f", nlab_int, thr_grad)
    logger.info("[primitives] radius band px=%.2f..%.2f (soft up to %.2f, relax=%.2f)", rmin_px, rmax_px, rmax_px_soft, size_relax)

    out: List[Knoll] = []
    seeds_all_raw: list[tuple[int, int]] = []
    for lab_id in range(1, nlab_int + 1):
        region: np.ndarray = (labels == lab_id)
        area = float(region.sum())
        if area < area_min or area > area_max_soft:
            continue
        eroded: np.ndarray = ndi.binary_erosion(region)
        perim = float((region & np.logical_not(eroded)).sum())
        if perim <= 0:
            continue
        roundness = 4.0 * np.pi * area / (perim * perim)
        if roundness < 0.05:
            continue
        ys, xs = np.where(region)
        if ys.size == 0 or xs.size == 0:
            continue
        cy = float(ys.mean()); cx = float(xs.mean())
        # track every initial candidate center (before eliminations)
        seeds_all_raw.append((int(round(cy)), int(round(cx))))
        macro_vals = macro_labels[region]
        vals, counts = np.unique(macro_vals, return_counts=True)
        mask = vals > 0
        if not mask.any():
            continue
        vals = vals[mask]; counts = counts[mask]
        rid = int(vals[int(np.argmax(counts))])
        r_eq_px = float(np.sqrt(area / np.pi))
        r_eq_m = r_eq_px * cell_m
        mean_g = float(np.nanmean(G[region]))
        # ---- footprint + height metadata (stay inside loop so labels_arr is in scope) ----
        sr = int(round(cy)); sc = int(round(cx))
        mask_macro = (macro_labels == rid)
        base = float("nan")
        peak = float("nan")
        if dem is not None and 0 <= sr < H and 0 <= sc < W:
            try:
                peak = float(dem[sr, sc])
                base = float(np.nanpercentile(dem[mask_macro], 20.0)) if np.any(mask_macro) else float("nan")
            except Exception:
                peak = float("nan"); base = float("nan")
        height_m = float(peak - base) if (np.isfinite(peak) and np.isfinite(base)) else None

        bnd_mask = _boundary_mask_bool(region)
        if (not np.any(bnd_mask)) and np.any(mask_macro):
            bnd_mask = _boundary_mask_bool(mask_macro)
        boundary_rc: list[tuple[int, int]] | None = None
        width_m = None
        length_m = None
        if np.any(bnd_mask):
            by, bx = np.nonzero(bnd_mask)
            boundary_rc = [(int(y), int(x)) for y, x in zip(by, bx)]
            width_m = (float(bx.max()) - float(bx.min()) + 1.0) * cell_m
            length_m = (float(by.max()) - float(by.min()) + 1.0) * cell_m
        major_m, minor_m, theta_deg = _ellipse_from_mask(region, cell_m)

        out.append(Knoll(
            rid=rid, cy=cy, cx=cx,
            r_m=r_eq_m, r_px=r_eq_px,
            score=mean_g, cov=roundness, cv=0.0,
            d_m=float(rows[0].get("d_m", 0.5 * W_m)) if rows else 0.5 * W_m,
            psi_deg=float(rows[0].get("psi_deg", 0.0)) if rows else 0.0,
            height_m=height_m,
            seed_r=sr, seed_c=sc,
            boundary_rc=boundary_rc,
            width_m=width_m, length_m=length_m,
            ellipse_major_m=major_m,
            ellipse_minor_m=minor_m,
            ellipse_theta_deg=theta_deg,
            footprint_rc=[(int(y), int(x)) for y, x in zip(*np.nonzero(region))],
        ))

    # Store seeds for diagnostics plot
    try:
        global DEBUG_MOUND_SEEDS
        seeds_rc = [(int(round(k.seed_r)), int(round(k.seed_c))) for k in out if k.seed_r is not None and k.seed_c is not None]
        DEBUG_MOUND_SEEDS = {
            "seeds_all": list(seeds_all_raw),
            "seeds_macro": list(seeds_rc),
            "seeds_kept": list(seeds_rc),
        }
    except Exception:
        pass

    out.sort(key=lambda k: k.score, reverse=True)
    logger.info("[primitives] knolls kept=%d (blob mode fixed size 8-25 m, Otsu)", len(out))
    return out


def detect_knolls_plane_descent(
    dem: np.ndarray,
    macro_labels: np.ndarray,
    rows: list[dict],
    *,
    cell_m: float,
    W_m: float,
    rmin_mult: float = 1.5,
    rmax_mult: float = 4.0,
    prominence_min_m: float = 3.0,
    descent_dmax_m: float = 30.0,
    dz_m: float = 0.5,
    size_relax: float = 1.0,
    cls_by_id: Dict[int, int] | None = None,
    steep_only: bool = False,
) -> List[Knoll]:
    """Detect knolls by descending a plane and tracking connected super-level sets.

    Args:
        dem (np.ndarray): DEM raster used for plane descent.
        macro_labels (np.ndarray): macroregion label raster aligned to ``dem``.
        rows (list[dict]): region metadata rows (used for default spacing/heading).
        cell_m (float): raster cell size in metres.
        W_m (float): nominal swath width in metres.
        rmin_mult (float): minimum radius multiplier relative to ``W_m``.
        rmax_mult (float): maximum radius multiplier relative to ``W_m``.
        prominence_min_m (float): minimum prominence for a seed peak.
        descent_dmax_m (float): maximum descent depth.
        dz_m (float): vertical step for plane descent.
        size_relax (float): softening factor for size limits.
        cls_by_id (Dict[int, int] | None): optional macroregion class lookup.
        steep_only (bool): when True, keep only steep class regions.

    Returns:
        List[Knoll]: detected knoll objects sorted by score.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.primitives import detect_knolls_plane_descent
        >>> dem = np.random.rand(10, 10)
        >>> knolls = detect_knolls_plane_descent(dem, np.zeros((10, 10), dtype=int), [], cell_m=1.0, W_m=3.0)
    """
    logger = logging.getLogger("main")
    if dem is None:
        logger.warning("[primitives] plane-descent requires DEM; skipping.")
        return []
    Z = np.array(dem, copy=True, dtype=float)
    H, W = Z.shape
    finite = np.isfinite(Z)
    if not finite.any():
        return []

    # Seed: morphological maxima with ~4m separation
    r_sep_px = max(1, int(np.ceil(4.0 / max(cell_m, 1e-6))))
    max_filt = ndi.maximum_filter(Z, size=2 * r_sep_px + 1, mode="nearest")
    seeds_mask = finite & (Z == max_filt)
    seeds_labeled_arr, n_seed_val = cast(tuple[np.ndarray, int], ndi.label(seeds_mask))
    seeds_labeled: np.ndarray = np.asarray(seeds_labeled_arr)
    n_seed: int = int(n_seed_val)

    # Prominence window (~30m)
    prom_win_px = max(1, int(np.ceil(30.0 / max(cell_m, 1e-6))))

    rmin_px = max(1.0, (rmin_mult * W_m) / max(cell_m, 1e-6))
    rmax_px = max(rmin_px + 1.0, (rmax_mult * W_m) / max(cell_m, 1e-6))
    size_relax = max(1.0, float(size_relax))
    rmax_px_soft = rmax_px * size_relax
    logger.info("[primitives] plane size band px=%.2f..%.2f (soft up to %.2f, relax=%.2f)", rmin_px, rmax_px, rmax_px_soft, size_relax)

    out: List[Knoll] = []
    seeds_all_raw: list[tuple[int, int]] = []
    for lab in range(1, int(n_seed) + 1):
        rr, cc = np.where(seeds_labeled == lab)
        if rr.size == 0:
            continue
        # take highest point in plateau
        idx_peak = int(np.argmax(Z[rr, cc]))
        sr, sc = int(rr[idx_peak]), int(cc[idx_peak])
        # store every initial candidate (before prominence/macro filters)
        seeds_all_raw.append((sr, sc))
        z0 = float(Z[sr, sc])
        # prominence check
        r0 = slice(max(0, sr - prom_win_px), min(H, sr + prom_win_px + 1))
        c0 = slice(max(0, sc - prom_win_px), min(W, sc + prom_win_px + 1))
        local_min = float(np.nanmin(Z[r0, c0]))
        if not np.isfinite(local_min) or (z0 - local_min) < prominence_min_m:
            logger.debug(
                "[plane] seed (%d,%d) rejected: prominence %.2f < %.2f",
                sr, sc, (z0 - local_min) if np.isfinite(local_min) else float("nan"), prominence_min_m
            )
            continue

        tau_high = z0 - 0.5 * prominence_min_m
        tau_low = z0 - descent_dmax_m
        tau_levels = np.arange(tau_high, tau_low - 1e-6, -dz_m)
        best_mask: np.ndarray | None = None
        best_tau = None
        for tau in tau_levels:
            mask = finite & (Z >= tau)
            if not mask[sr, sc]:
                continue
            comp_labels_arr, nlab_val = cast(tuple[np.ndarray, int], ndi.label(mask))
            comp_labels: np.ndarray = np.asarray(comp_labels_arr)
            nlab: int = int(nlab_val)
            lab_id = int(comp_labels[sr, sc])
            if lab_id <= 0:
                continue
            comp = (comp_labels == lab_id)
            area = float(comp.sum())
            r_eq = float(np.sqrt(area / np.pi))
            if r_eq < rmin_px or r_eq > rmax_px_soft:
                logger.debug(
                    "[plane] seed (%d,%d) level %.2f rejected: r_eq_px=%.2f outside [%.2f, %.2f] (soft_max=%.2f)",
                    sr, sc, tau, r_eq, rmin_px, rmax_px, rmax_px_soft
                )
                continue
            best_mask = comp
            best_tau = float(tau)
            break  # pick the first (highest) level that satisfies size band

        if best_mask is None:
            logger.debug("[plane] seed (%d,%d) rejected: no level satisfied size band", sr, sc)
            continue

        ys, xs = np.where(best_mask)
        cy = float(ys.mean()); cx = float(xs.mean())
        macro_vals = macro_labels[best_mask]
        vals, counts = np.unique(macro_vals, return_counts=True)
        mask_pos = vals > 0
        if not mask_pos.any():
            logger.debug("[plane] seed (%d,%d) rejected: footprint has no macro ids", sr, sc)
            continue
        rid = int(vals[int(np.argmax(counts))])
        if steep_only and cls_by_id is not None:
            cls_val = int(cls_by_id.get(rid, 0))
            if cls_val != 3:
                logger.debug("[plane] seed (%d,%d) rejected: macro id %d class %d not steep", sr, sc, rid, cls_val)
                continue

        r_eq_px = float(np.sqrt(best_mask.sum() / np.pi))
        r_eq_m = r_eq_px * cell_m
        mean_z = float(np.nanmean(Z[best_mask]))

        bnd_mask = _boundary_mask_bool(best_mask)
        boundary_rc = []
        if np.any(bnd_mask):
            by, bx = np.nonzero(bnd_mask)
            boundary_rc = [(int(y), int(x)) for y, x in zip(by, bx)]
        major_m, minor_m, theta_deg = _ellipse_from_mask(best_mask, cell_m)

        out.append(Knoll(
            rid=rid, cy=cy, cx=cx,
            r_m=r_eq_m, r_px=r_eq_px,
            score=mean_z if best_tau is None else best_tau,
            cov=1.0, cv=0.0,
            d_m=float(rows[0].get("d_m", 0.5 * W_m)) if rows else 0.5 * W_m,
            psi_deg=float(rows[0].get("psi_deg", 0.0)) if rows else 0.0,
            height_m=None,
            seed_r=sr, seed_c=sc,
            boundary_rc=boundary_rc,
            width_m=None, length_m=None,
            ellipse_major_m=major_m,
            ellipse_minor_m=minor_m,
            ellipse_theta_deg=theta_deg,
            footprint_rc=[(int(y), int(x)) for y, x in zip(*np.nonzero(best_mask))],
        ))

    out.sort(key=lambda k: k.score, reverse=True)
    logger.info("[primitives] plane-descent knolls kept=%d", len(out))
    try:
        global DEBUG_MOUND_SEEDS
        seeds_rc = [(int(round(k.seed_r)), int(round(k.seed_c))) for k in out if k.seed_r is not None and k.seed_c is not None]
        DEBUG_MOUND_SEEDS = {
            "seeds_all": list(seeds_all_raw),
            "seeds_macro": list(seeds_rc),
            "seeds_kept": list(seeds_rc),
        }
    except Exception:
        pass
    return out

def augment_regions_with_primitives(
    rows: list[dict],
    macro_labels: np.ndarray,
    feats: dict[str, np.ndarray],
    transform: Any,
    *,
    cell_m: float,
    W_m: float,
    slope_ring_deg: float,
    circ_cv_max: float,
    rmin_mult: float,
    rmax_mult: float,
    size_relax: float = 1.0,
    spiral_rmin_m: float,
    use_grad_seeds: bool = True,
    grad_seed_pct: float = 85.0,
    dem: np.ndarray | None = None,
    method: str = "ring",
    prominence_min_m: float = 3.0,
    descent_dmax_m: float = 30.0,
    dz_m: float = 0.5,
    steep_only: bool = False,
    force_seeds_xy: list[tuple[float, float]] | None = None,
    force_plane_z: float | None = None,
    force_plane_z_list: list[float] | None = None,
    force_min_diam_m: float = 0.0,
) -> list[dict]:
    """Attach primitive metadata (spiral around knoll) to region rows.

    Args:
        rows (list[dict]): region records to augment in place.
        macro_labels (np.ndarray): macroregion label raster.
        feats (dict[str, np.ndarray]): feature rasters (expects ``"slope"`` and optionally ``"dem"``).
        transform (Any): affine transform mapping pixel centers to world coords.
        cell_m (float): raster cell size in metres.
        W_m (float): nominal swath width in metres.
        slope_ring_deg (float): slope ring threshold in degrees.
        circ_cv_max (float): maximum allowed circularity coefficient of variation.
        rmin_mult (float): minimum radius multiplier relative to ``W_m``.
        rmax_mult (float): maximum radius multiplier relative to ``W_m``.
        size_relax (float): softening factor for size limits.
        spiral_rmin_m (float): minimum spiral radius for forced footprints.
        use_grad_seeds (bool): whether to seed from gradient magnitude peaks.
        grad_seed_pct (float): percentile for gradient thresholding when seeding.
        dem (np.ndarray | None): optional DEM for height metadata.
        method (str): detection method (``"ring"`` or ``"plane"``).
        prominence_min_m (float): minimum prominence for plane descent.
        descent_dmax_m (float): maximum descent depth.
        dz_m (float): vertical step for plane descent.
        steep_only (bool): when True, keep only steep class regions.
        force_seeds_xy (list[tuple[float, float]] | None): force-add seeds (world coords).
        force_plane_z (float | None): optional target plane for forced footprints.
        force_plane_z_list (list[float] | None): optional per-seed plane heights.
        force_min_diam_m (float): minimum diameter for forced footprints.

    Returns:
        list[dict]: updated region rows (same list, augmented in place).

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.primitives import augment_regions_with_primitives
        >>> rows = [{"id": 1}]
        >>> feats = {"slope": np.zeros((5, 5))}
        >>> out = augment_regions_with_primitives(rows, np.ones((5, 5), dtype=int), feats, transform=None, cell_m=1.0, W_m=3.0,
        ...                                       slope_ring_deg=12.0, circ_cv_max=0.2, rmin_mult=1.5, rmax_mult=4.0, spiral_rmin_m=1.0)
    """
    logger = logging.getLogger("main")
    global DEBUG_MOUND_SEEDS
    seed_debug_detect: dict | None = None
    mth = (method or "ring").lower()
    if mth == "plane":
        dem_input = dem if dem is not None else feats.get("dem", None)
        if dem_input is None:
            logging.getLogger("main").warning("[primitives] plane-descent requested but DEM missing; skipping knolls.")
            knolls = []
        else:
            cls_by_id: Dict[int, int] = {int(r["id"]): int(r.get("cls", 0)) for r in rows}
            knolls = detect_knolls_plane_descent(
                dem=dem_input,
                macro_labels=macro_labels,
                rows=rows,
                cell_m=cell_m,
                W_m=W_m,
                size_relax=size_relax,
                rmin_mult=rmin_mult,
                rmax_mult=rmax_mult,
                prominence_min_m=prominence_min_m,
                descent_dmax_m=descent_dmax_m,
                dz_m=dz_m,
                cls_by_id=cls_by_id,
                steep_only=steep_only,
            )
        try:
            seed_debug_detect = DEBUG_MOUND_SEEDS.copy()  # type: ignore[name-defined]
        except Exception:
            seed_debug_detect = None
    else:
        knolls = detect_knolls(
            feats, macro_labels, rows,
            cell_m=cell_m, W_m=W_m,
            slope_ring_deg=slope_ring_deg, circ_cv_max=circ_cv_max,
            rmin_mult=rmin_mult, rmax_mult=rmax_mult, spiral_rmin_m=spiral_rmin_m,
            size_relax=size_relax,
            use_grad_seeds=use_grad_seeds, grad_seed_pct=grad_seed_pct,
            dem=dem,
        )
        try:
            seed_debug_detect = DEBUG_MOUND_SEEDS.copy()  # type: ignore[name-defined]
        except Exception:
            seed_debug_detect = None
    # Force-add seeds in world coords (x,y meters), bypassing size/steep gates
    if force_seeds_xy and transform is not None:
        forced: list[Knoll] = []
        H, W = macro_labels.shape
        mask_macro_any = macro_labels > 0
        seeds_iter = list(force_seeds_xy) if force_seeds_xy is not None else []
        for idx, (xw, yw) in enumerate(seeds_iter):
            try:
                col_c, row_c = (~transform) * (float(xw), float(yw))
                cx = float(col_c) - 0.5
                cy = float(row_c) - 0.5
            except Exception:
                cx = float(xw / max(cell_m, 1e-6)) - 0.5
                cy = float(yw / max(cell_m, 1e-6)) - 0.5
            r_int = int(round(cy)); c_int = int(round(cx))
            if not (0 <= r_int < H and 0 <= c_int < W):
                logger.debug("[primitives] force seed (%.2f,%.2f) -> rc=(%d,%d) out of bounds", xw, yw, r_int, c_int)
                continue
            rid = int(macro_labels[r_int, c_int])
            if rid <= 0 and np.any(mask_macro_any):
                # snap to nearest macro pixel if the point landed on a void
                _dist, idx_tuple = cast(
                    tuple[np.ndarray, tuple[np.ndarray, ...]],
                    ndi.distance_transform_edt(~mask_macro_any, return_indices=True),
                )
                rr_idx = np.asarray(idx_tuple[0]); cc_idx = np.asarray(idx_tuple[1])
                nr = int(rr_idx[r_int, c_int]); nc = int(cc_idx[r_int, c_int])
                rid = int(macro_labels[nr, nc])
                r_int, c_int = nr, nc
                cy, cx = float(r_int), float(c_int)
            if rid <= 0:
                logger.debug("[primitives] force seed (%.2f,%.2f) -> rc=(%d,%d) has no macro id even after snap", xw, yw, r_int, c_int)
                continue
            # Try to derive an actual footprint by descending a plane from the seed height downwards
            width_m = length_m = None
            ellipse_major_m = ellipse_minor_m = ellipse_theta_deg = None
            boundary_rc: list[tuple[int, int]] | None = None
            r_px = None
            r_m = None
            mask_fp: np.ndarray | None = None
            if dem is not None:
                # For forced seeds, be very permissive on size to capture the large knoll footprint
                rmin_px = 1.0
                rmax_px = max(1.0, (rmax_mult * W_m) / max(cell_m, 1e-6))
                rmax_px_soft = max(rmax_px * max(1.0, size_relax), float(max(H, W)) * 2.0)
                mask_fp = None; r_eq_px = None; r_eq_m = None
                # First, try fixed horizontal plane if specified (e.g., z=11)
                target_z = None
                if force_plane_z_list and idx < len(force_plane_z_list):
                    target_z = force_plane_z_list[idx]
                elif force_plane_z is not None:
                    target_z = force_plane_z
                if target_z is not None:
                    mask_fp, r_eq_px = _component_at_target_plane(
                        dem, r_int, c_int, target_z=float(target_z), step=0.25
                    )
                    if r_eq_px is not None:
                        r_eq_m = float(r_eq_px * cell_m)
                # If that fails, fall back to descent
                if mask_fp is None:
                    mask_fp, r_eq_px, r_eq_m = _footprint_from_descent(
                        dem, r_int, c_int,
                        cell_m=cell_m,
                        rmin_px=rmin_px,
                        rmax_px_soft=rmax_px_soft,
                        descent_dmax_m=descent_dmax_m,
                        dz_m=dz_m,
                    )
                # Enforce minimum diameter for forced footprints by dilating
                if mask_fp is not None and force_min_diam_m and force_min_diam_m > 0:
                    desired_r_px = (0.5 * force_min_diam_m) / max(cell_m, 1e-6)
                    r_eq_px_curr = r_eq_px if r_eq_px is not None else float(np.sqrt(mask_fp.sum() / np.pi))
                    iter_ct = 0
                    while r_eq_px_curr < desired_r_px and iter_ct < 20:
                        mask_fp = ndi.binary_dilation(mask_fp, structure=disk(1))
                        r_eq_px_curr = float(np.sqrt(mask_fp.sum() / np.pi))
                        iter_ct += 1
                    r_eq_px = r_eq_px_curr
                    r_eq_m = r_eq_px * cell_m
                if mask_fp is not None and r_eq_px is not None and r_eq_m is not None:
                    r_px = r_eq_px
                    r_m = r_eq_m
                    bnd_mask = _boundary_mask_bool(mask_fp)
                    if np.any(bnd_mask):
                        by, bx = np.nonzero(bnd_mask)
                        boundary_rc = [(int(y), int(x)) for y, x in zip(by, bx)]
                        width_m = (float(bx.max()) - float(bx.min()) + 1.0) * cell_m
                        length_m = (float(by.max()) - float(by.min()) + 1.0) * cell_m
                    maj, minu, theta = _ellipse_from_mask(mask_fp, cell_m)
                    ellipse_major_m, ellipse_minor_m, ellipse_theta_deg = maj, minu, theta
            if r_px is None or r_m is None:
                # Fallback synthetic circular footprint so plotting and footprint metadata are present
                r_m = max(spiral_rmin_m, W_m * rmax_mult)
                r_px = r_m / max(cell_m, 1e-6)
                yy, xx = np.ogrid[:H, :W]
                dist = np.hypot(yy - cy, xx - cx)
                boundary_mask = np.logical_and(dist >= (r_px - 0.75), dist <= (r_px + 0.75))
                by, bx = np.nonzero(boundary_mask)
                boundary_rc = [(int(y), int(x)) for y, x in zip(by, bx)]
                width_m = length_m = 2.0 * r_px * cell_m
                ellipse_major_m = ellipse_minor_m = 2.0 * r_px * cell_m
                ellipse_theta_deg = 0.0
                mask_fp = boundary_mask  # reuse as footprint outline (approx)
            forced.append(Knoll(
                rid=rid, cy=cy, cx=cx,
                r_m=r_m, r_px=r_px,
                score=1e6, cov=1.0, cv=0.0,
                d_m=float(rows[0].get("d_m", 0.5 * W_m)) if rows else 0.5 * W_m,
                psi_deg=float(rows[0].get("psi_deg", 0.0)) if rows else 0.0,
                height_m=None,
                seed_r=r_int, seed_c=c_int,
                boundary_rc=boundary_rc,
                width_m=width_m, length_m=length_m,
                ellipse_major_m=ellipse_major_m,
                ellipse_minor_m=ellipse_minor_m,
                ellipse_theta_deg=ellipse_theta_deg,
                footprint_rc=[(int(y), int(x)) for y, x in zip(*np.nonzero(mask_fp))] if mask_fp is not None else None,
            ))
            logger.info("[primitives] force seed kept: (x=%.2f,y=%.2f) -> rc=(%d,%d) macro id=%d", xw, yw, r_int, c_int, rid)
        # When force list is provided, discard other seeds and only keep forced ones
        knolls = forced
    # augment rows in place
    by_id = {int(r["id"]): r for r in rows}
    if knolls:
        # Keep only one per macro unless we're explicitly forcing seeds (then keep them all)
        if not (force_seeds_xy and len(force_seeds_xy) > 0):
            knolls.sort(key=lambda k: k.score, reverse=True)
            uniq: list[Knoll] = []
            seen_rids: set[int] = set()
            for k in knolls:
                rid_k = int(k.rid)
                if rid_k in seen_rids:
                    continue
                seen_rids.add(rid_k)
                uniq.append(k)
            knolls = uniq
        try:
            seeds_rc = [(int(round(k.seed_r)), int(round(k.seed_c))) for k in knolls if k.seed_r is not None and k.seed_c is not None]
            seeds_all = list(seeds_rc)
            seeds_macro = list(seeds_rc)
            if seed_debug_detect:
                seeds_all = list(seed_debug_detect.get("seeds_all", seeds_all))
                seeds_macro = list(seed_debug_detect.get("seeds_macro", seeds_macro))
            DEBUG_MOUND_SEEDS = {
                "seeds_all": seeds_all,
                "seeds_macro": seeds_macro,
                "seeds_kept": list(seeds_rc),
            }
        except Exception:
            pass
        seed_log = []
        for k in knolls:
            if k.seed_r is None or k.seed_c is None:
                continue
            if transform is not None:
                try:
                    a, b, c0, d, e, f0 = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
                    xw = float(c0 + a * (k.cx + 0.5) + b * (k.cy + 0.5))
                    yw = float(f0 + d * (k.cx + 0.5) + e * (k.cy + 0.5))
                except Exception:
                    xw = float("nan"); yw = float("nan")
                seed_log.append((int(k.seed_r), int(k.seed_c), xw, yw))
            else:
                seed_log.append((int(k.seed_r), int(k.seed_c)))
        logger.info("[primitives] seeds_kept=%s", seed_log)
    for k in knolls:
        r = by_id.get(k.rid)
        if r is None:
            continue
        r["primitive"] = "spiral"

        # Generic names that plotting expects
        r["cx_px"] = k.cx
        r["cy_px"] = k.cy
        r["r_m"]   = k.r_m

        # Keep the 'prim_*' synonyms (harmless, and useful if other code uses them)
        r["prim_cx_px"] = k.cx
        r["prim_cy_px"] = k.cy
        r["prim_r_m"]   = k.r_m
        r["prim_score"] = k.score
        r["prim_cov"]   = k.cov
        r["prim_cv"]    = k.cv
        r["height_m"]   = k.height_m
        new_bnd = k.boundary_rc or []
        r["boundary_rc"] = new_bnd or r.get("boundary_rc") or []
        r.setdefault("boundary_rc_list", []).append(new_bnd)
        if k.footprint_rc:
            r.setdefault("footprint_masks", []).append(k.footprint_rc)
        r["seed_r"]     = k.seed_r
        r["seed_c"]     = k.seed_c
        r.setdefault("seeds_list", []).append((k.seed_r, k.seed_c))
        width_candidates = [val for val in (r.get("width_m"), k.width_m) if val is not None]
        r["width_m"]    = max(width_candidates) if width_candidates else None
        length_candidates = [val for val in (r.get("length_m"), k.length_m) if val is not None]
        r["length_m"]   = max(length_candidates) if length_candidates else None
        r["primitive_shape"] = "cylinder"
        r["ellipse_major_m"] = k.ellipse_major_m
        r["ellipse_minor_m"] = k.ellipse_minor_m
        r["ellipse_theta_deg"] = k.ellipse_theta_deg

        # --- NEW: cache world coordinates of the primitive center for plotting/planning ---
        # Note: row/col → world using rasterio Affine:  x = c + (col+0.5)*a,  y = f + (row+0.5)*e
        #  Robust world coords using rasterio Affine multiplication (supports rotation/skew).
        try:
            # (col,row) -> (x,y) with pixel-center convention (+0.5)
            xw, yw = transform * (k.cx + 0.5, k.cy + 0.5)
            xw = float(xw); yw = float(yw)
        except Exception:
            # Fallback that handles generic a,b,c,d,e,f (including b,d != 0); if transform is None, use grid coords.
            try:
                a, b, c0, d, e, f0 = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
                xw = float(c0 + a * (k.cx + 0.5) + b * (k.cy + 0.5))
                yw = float(f0 + d * (k.cx + 0.5) + e * (k.cy + 0.5))
            except Exception:
                xw = float((k.cx + 0.5) * cell_m)
                yw = float((k.cy + 0.5) * cell_m)

        r["prim_x_m"] = xw
        r["prim_y_m"] = yw
        
        logging.getLogger("main").info("[primitives] spiral regions: %d", sum(1 for r in rows if (r.get("primitive") or "").lower() == "spiral"))


    return rows
