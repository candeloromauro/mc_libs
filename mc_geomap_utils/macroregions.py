from __future__ import annotations
import csv
import numpy as np
from scipy import ndimage as ndi
from collections import defaultdict
from skimage.morphology import binary_closing, binary_opening, disk


# ---------- masks & thresholds ----------
def eroded_support_mask(dem: np.ndarray, border: int = 3) -> np.ndarray:
    """Erode valid-data support by N cells to hide edge seams.

    Args:
        dem (np.ndarray): DEM raster with NaNs for no-data.
        border (int): number of cells to erode inward.

    Returns:
        np.ndarray: boolean mask of the eroded valid-data support.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import eroded_support_mask
        >>> mask = eroded_support_mask(np.array([[1.0, np.nan], [2.0, 3.0]]), border=1)
    """
    support = np.isfinite(dem)
    if border <= 0:
        return support
    foot = np.ones((2*border+1, 2*border+1), dtype=bool)
    return ndi.binary_erosion(support, structure=foot)

def classify_policy_cells(features: dict[str, np.ndarray],
                        slope_flat_deg: float = 5.0,
                        slope_steep_deg: float = 15.0,
                        vrm_steep: float = 0.06,
                        bpi_sigma: float | None = None,
                        bpi_sigma_mult: float = 1.0,
                        ridge_min_slope_deg: float = 3.0,   # NEW
                        vrm_min_slope_deg: float = 7.0      # NEW
                        ) -> tuple[np.ndarray, float]:
    """Classify each cell into a policy class based on slope/VRM/BPI.

    Args:
        features (dict[str, np.ndarray]): feature rasters including ``slope``, ``vrm``, and ``bpi``.
        slope_flat_deg (float): upper slope limit for flat class.
        slope_steep_deg (float): lower slope limit for steep class.
        vrm_steep (float): VRM threshold for steep classification.
        bpi_sigma (float | None): override for BPI sigma (otherwise estimated).
        bpi_sigma_mult (float): multiplier for BPI sigma thresholding.
        ridge_min_slope_deg (float): minimum slope for ridge/valley class.
        vrm_min_slope_deg (float): minimum slope for VRM-driven steep class.

    Returns:
        tuple[np.ndarray, float]: ``policy_map`` and computed ``sigma_bpi``.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import classify_policy_cells
        >>> feats = {"slope": np.zeros((5, 5)), "vrm": np.zeros((5, 5)), "bpi": np.zeros((5, 5))}
        >>> policy, sigma_bpi = classify_policy_cells(feats)
    """
    slope_deg = np.degrees(features["slope"])
    vrm = features["vrm"]
    bpi = features["bpi"]
    valid = np.isfinite(slope_deg) & np.isfinite(vrm) & np.isfinite(bpi)

    if bpi_sigma is None:
        sigma_bpi = float(np.nanstd(bpi[valid])) if np.any(valid) else 0.0
    else:
        sigma_bpi = float(bpi_sigma)
    thr = bpi_sigma_mult * sigma_bpi

    out = np.zeros_like(slope_deg, dtype=np.uint8)

    flat   = (slope_deg <= slope_flat_deg) & (vrm <= 0.02) & (np.abs(bpi) <= 0.5*thr)
    gentle = (slope_deg > slope_flat_deg) & (slope_deg <= slope_steep_deg)
    steep  = (slope_deg > slope_steep_deg) | ((vrm > vrm_steep) & (slope_deg >= vrm_min_slope_deg))
    # If dispersion is ~0 (flat/constant surface), avoid labeling everything as R/V
    if thr > 0:
        ridgev = (np.abs(bpi) >= thr) & (slope_deg >= ridge_min_slope_deg)
    else:
        ridgev = np.zeros_like(slope_deg, dtype=bool)

    out[flat]   = 1
    out[gentle] = 2
    out[steep]  = 3
    # only assign Ridge/Valley where NOT steep (so steep keeps priority)
    out[ridgev & ~steep] = 4
    out[~valid] = 0
    return out, sigma_bpi

# ---------- SLIC -> RAG merge ----------
def grad_mag_deg(slope_deg: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude of the slope surface in degrees.

    Args:
        slope_deg (np.ndarray): slope raster expressed in degrees.

    Returns:
        np.ndarray: gradient magnitude at each cell.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import grad_mag_deg
        >>> grad = grad_mag_deg(np.zeros((3, 3)))
    """
    gx = ndi.sobel(slope_deg, axis=1, mode="nearest") / 8.0
    gy = ndi.sobel(slope_deg, axis=0, mode="nearest") / 8.0
    return np.hypot(gx, gy)

def edge_list_with_boundaries(labels: np.ndarray) -> dict[tuple[int,int], list[tuple[int,int]]]:
    """Collect boundary pixel coordinates for each touching superpixel pair.

    Args:
        labels (np.ndarray): label raster describing superpixels.

    Returns:
        dict[tuple[int,int], list[tuple[int,int]]]: map from region pairs to seam coordinates.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import edge_list_with_boundaries
        >>> edges = edge_list_with_boundaries(np.array([[1, 1], [2, 2]]))
    """
    H, W = labels.shape
    edges: dict[tuple[int,int], list[tuple[int,int]]] = {}
    # vertical seams
    L = labels[:, :-1]; R = labels[:, 1:]
    m = (L != R) & (L > 0) & (R > 0)
    ys, xs = np.where(m)
    for y, x in zip(ys, xs):
        a, b = int(L[y, x]), int(R[y, x])
        if a > b:
            a, b = b, a
        edges.setdefault((a, b), []).append((int(y), int(x)))
    # horizontal seams
    T = labels[:-1, :]; B = labels[1:, :]
    m = (T != B) & (T > 0) & (B > 0)
    ys, xs = np.where(m)
    for y, x in zip(ys, xs):
        a, b = int(T[y, x]), int(B[y, x])
        if a > b:
            a, b = b, a
        edges.setdefault((a, b), []).append((int(y), int(x)))
    return edges

def superpixel_stats(labels: np.ndarray, policy_map: np.ndarray,
                    feats: dict[str, np.ndarray]) -> dict[int, dict]:
    """Summarise each superpixel with area, class, and mean features.

    Args:
        labels (np.ndarray): superpixel label raster.
        policy_map (np.ndarray): class map aligned to the labels.
        feats (dict[str, np.ndarray]): feature rasters used for statistics.

    Returns:
        dict[int, dict]: mapping from region id to collected attributes.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import superpixel_stats
        >>> labels = np.array([[1, 1], [2, 2]])
        >>> policy = np.array([[1, 1], [2, 2]])
        >>> feats = {"slope": np.zeros((2, 2)), "vrm": np.zeros((2, 2)), "bpi": np.zeros((2, 2))}
        >>> stats = superpixel_stats(labels, policy, feats)
    """
    ids = np.unique(labels[labels > 0])
    SP: dict[int, dict] = {}
    for rid in ids:
        m = (labels == rid)
        vals, cnts = np.unique(policy_map[m], return_counts=True)
        mask = vals > 0
        if np.any(mask):
            maj = int(vals[mask][np.argmax(cnts[mask])])
        else:
            maj = 0
        SP[rid] = {
            "cls": maj,
            "area_px": int(m.sum()),
            "mean_slope_deg": float(np.nanmean(np.degrees(feats["slope"][m]))),
            "mean_vrm": float(np.nanmean(feats["vrm"][m])),
            "mean_bpi": float(np.nanmean(feats["bpi"][m])),
        }
    return SP

class UF:
    """Lightweight union-find structure for merging region adjacency graph components.

    Args:
        ids (Iterable[int]): region identifiers to initialise.

    Returns:
        None
    """

    def __init__(self, ids):
        """Initialise parent and size dictionaries for all nodes.

        Args:
            ids (Iterable[int]): identifiers representing the initial sets.

        Returns:
            None
        """
        self.p = {i: i for i in ids}
        self.sz = {i: 1 for i in ids}

    def find(self, x):
        """Locate the canonical parent for node x with path compression.

        Args:
            x (int): node identifier to locate.

        Returns:
            int: current root representative for x.

        Examples:
            >>> uf = UF([1, 2])
            >>> uf.find(1)
            1
        """
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        """Merge the sets containing nodes a and b.

        Args:
            a (int): first node to merge.
            b (int): second node to merge.

        Returns:
            int: identifier of the resulting root.

        Examples:
            >>> uf = UF([1, 2])
            >>> uf.union(1, 2)
            1
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return ra

def rag_merge_slic(labels: np.ndarray,
                    policy_map: np.ndarray,
                    feats: dict[str, np.ndarray],
                    area_cap_px: int = 40000,
                    tau_pct: float = 98.0,
                    min_area_px: int = 0) -> tuple[np.ndarray, dict[int, int]]:
    """Merge adjacent superpixels when classes agree and the boundary is weak.

    Boundary strength γ_ij is the mean |∇(slope)| along the interface. Thresholds are
    class-wise percentiles; if a class has no edges, the global percentile is used.

    Args:
        labels (np.ndarray): SLIC superpixel label raster.
        policy_map (np.ndarray): per-cell policy classes aligned to ``labels``.
        feats (dict[str, np.ndarray]): feature rasters (expects ``"slope"``).
        area_cap_px (int): maximum area for merged regions (pixels).
        tau_pct (float): percentile used for boundary strength thresholds.
        min_area_px (int): minimum area for macroregions (smaller ones are absorbed).

    Returns:
        tuple[np.ndarray, dict[int, int]]: macro label raster and mapping from macro id to class.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import rag_merge_slic
        >>> labels = np.array([[1, 1], [2, 2]])
        >>> policy = np.array([[1, 1], [2, 2]])
        >>> feats = {"slope": np.zeros((2, 2))}
        >>> macro, reg_cls = rag_merge_slic(labels, policy, feats)
    """

    print(f"[rag] min_area_px={min_area_px}")
    
    # 1) Per-superpixel stats (majority class, area...)
    SP = superpixel_stats(labels, policy_map, feats)
    ids = [int(i) for i in SP.keys()]

    # 2) Boundary strengths for all adjacent label pairs (i<j)
    gamma = _edge_gamma_from_neighbors(labels, feats["slope"])  # slope in radians

    # 3) Build edge list with classes
    edge_info: list[tuple[float, int, int, int, int]] = []
    for (a, b), g in gamma.items():
        ca, cb = int(SP[a]["cls"]), int(SP[b]["cls"])
        edge_info.append((float(g), int(a), int(b), ca, cb))

    # 4) Class-wise thresholds (with robust global fallback)
    agree = [g for (g, a, b, ca, cb) in edge_info if ca > 0 and ca == cb]
    # ensure plain float for type checkers and downstream comparisons
    tau_global = float(np.percentile(agree, tau_pct)) if agree else float("inf")
    tau: dict[int, float] = {}
    for c in (1, 2, 3, 4):
        vv = [g for (g, a, b, ca, cb) in edge_info if ca == cb == c]
        tau[c] = float(np.percentile(vv, tau_pct)) if vv else tau_global

    # 5) Greedy merges with union–find and area cap
    uf = UF(ids)
    for g, a, b, ca, cb in sorted(edge_info, key=lambda t: t[0]):
        if ca == 0 or ca != cb:
            continue  # only merge edges with same nonzero class
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if (SP[ra]["area_px"] + SP[rb]["area_px"]) > area_cap_px:
            continue
        if g <= tau[ca]:
            root = uf.union(ra, rb)
            # Update aggregate stats on the root
            SP[root]["area_px"] = SP[ra]["area_px"] + SP[rb]["area_px"]
            SP[root]["cls"] = ca  # keep class c for the merged region

    # 6) Post-pass: absorb sub-threshold macros (before rasterization)
    if min_area_px > 0:
        roots = sorted({uf.find(i) for i in ids})
        small = [r for r in roots if SP[r]["area_px"] < min_area_px]
        print(f"[rag] small macros before absorption: {len(small)} (min_area={min_area_px} px)")
    merged_small = _absorb_small_macros(ids, uf, SP, gamma, min_area_px, prefer_same_class=True)
    if merged_small:
        print(f"[rag] absorbed {merged_small} sub-threshold regions (min_area={min_area_px} px)")

    # 7) Rasterize macro labels (after absorption)
    root_of = {rid: uf.find(rid) for rid in ids}
    unique_roots = sorted(set(root_of.values()))
    remap = {r: i + 1 for i, r in enumerate(unique_roots)}
    macro = np.zeros_like(labels, dtype=np.int32)
    for rid in ids:
        macro[labels == rid] = remap[root_of[rid]]

    # 8) Per-macro class lookup
    reg_cls: dict[int, int] = {}
    for r in unique_roots:
        reg_cls[remap[r]] = int(SP[r]["cls"])

    # diagnostics
    num_labels = len(ids)
    num_edges = len(edge_info)
    num_roots = len(unique_roots)
    print(f"[rag] SLIC labels={num_labels} edges={num_edges} -> macro regions={num_roots}")

    return macro, reg_cls

# ---------- per-region policy & export ----------
def _edge_gamma_from_neighbors(labels: np.ndarray, slope_rad: np.ndarray) -> dict[tuple[int,int], float]:
    """Return mean boundary strength γ_ij for each adjacent label pair (i<j).
        γ is the mean gradient magnitude of slope along the interface.
    """
    # gradient magnitude of slope
    gx = ndi.sobel(slope_rad, axis=1, mode="nearest") / 8.0
    gy = ndi.sobel(slope_rad, axis=0, mode="nearest") / 8.0
    G = np.hypot(gx, gy)
    G[~np.isfinite(slope_rad)] = np.nan

    H, W = labels.shape
    sum_g = defaultdict(float)
    cnt_g = defaultdict(int)

    # vertical neighbors (rows differ)
    Lv, Lu = labels[1:, :], labels[:-1, :]
    mv = (Lv > 0) & (Lu > 0) & (Lv != Lu)
    if np.any(mv):
        r, c = np.nonzero(mv)
        r1 = r + 1; c1 = c
        # sample gradient on both sides, average
        # sample gradient on both sides; finite-aware average (no empty-slice warning)
        g1 = G[r1, c1]
        g2 = G[r1-1, c1]
        cnt = np.isfinite(g1).astype(np.int8) + np.isfinite(g2).astype(np.int8)
        num = np.where(np.isfinite(g1), g1, 0.0) + np.where(np.isfinite(g2), g2, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gs = np.where(cnt > 0, num / cnt, np.nan)
        i = np.minimum(Lv[mv], Lu[mv]).astype(int)
        j = np.maximum(Lv[mv], Lu[mv]).astype(int)
        for a, b, g in zip(i, j, gs):
            if np.isfinite(g):
                sum_g[(a, b)] += float(g); cnt_g[(a, b)] += 1

    # horizontal neighbors (cols differ)
    Lr, Ll = labels[:, 1:], labels[:, :-1]
    mh = (Lr > 0) & (Ll > 0) & (Lr != Ll)
    if np.any(mh):
        r, c = np.nonzero(mh)
        r1 = r; c1 = c + 1
        g1 = G[r1, c1]
        g2 = G[r1, c1-1]
        cnt = np.isfinite(g1).astype(np.int8) + np.isfinite(g2).astype(np.int8)
        num = np.where(np.isfinite(g1), g1, 0.0) + np.where(np.isfinite(g2), g2, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gs = np.where(cnt > 0, num / cnt, np.nan)
        i = np.minimum(Lr[mh], Ll[mh]).astype(int)
        j = np.maximum(Lr[mh], Ll[mh]).astype(int)
        for a, b, g in zip(i, j, gs):
            if np.isfinite(g):
                sum_g[(a, b)] += float(g); cnt_g[(a, b)] += 1

    return {k: (sum_g[k] / cnt_g[k]) for k in sum_g if cnt_g[k] > 0}

def axial_mean_heading(mask: np.ndarray, aspect: np.ndarray, slope: np.ndarray, along: bool) -> float:
    """Compute the dominant heading within a region using slope/aspect weights.

    Args:
        mask (np.ndarray): boolean mask describing the macroregion.
        aspect (np.ndarray): aspect raster measured in radians.
        slope (np.ndarray): slope raster measured in radians.
        along (bool): whether to align headings with the slope direction.

    Returns:
        float: heading angle expressed in radians.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import axial_mean_heading
        >>> mask = np.ones((2, 2), dtype=bool)
        >>> aspect = np.zeros((2, 2))
        >>> slope = np.zeros((2, 2))
        >>> axial_mean_heading(mask, aspect, slope, along=False)
        0.0
    """
    theta = aspect + (np.pi/2 if along else 0.0)
    w = (np.degrees(slope)**2) * mask
    th2 = (2.0*theta) % (2*np.pi)
    s = np.nansum(np.sin(th2)*w); c = np.nansum(np.cos(th2)*w)
    if s == 0 and c == 0: return 0.0
    return float(0.5*np.arctan2(s, c) % (2*np.pi))

def policy_for_class(cls: int, W: float, h0: float, hmin: float, betamax_deg: float, mean_slope_deg: float):
    """Derive default survey policy parameters for a macroregion class.

    Args:
        cls (int): policy class identifier.
        W (float): nominal swath width.
        h0 (float): baseline altitude parameter.
        hmin (float): minimum altitude permitted.
        betamax_deg (float): upper bound for pitch/heading offsets.
        mean_slope_deg (float): mean slope in degrees for the region.

    Returns:
        dict: dictionary containing spacing, altitude, and orientation flags.

    Examples:
        >>> from mc_geomap_utils.macroregions import policy_for_class
        >>> policy_for_class(1, W=3.0, h0=5.0, hmin=2.0, betamax_deg=20.0, mean_slope_deg=4.0)
    """
    if cls == 1:  # Flat
        return dict(d=0.8*W, h=h0, beta=5.0, along=False)
    if cls == 2:  # Gentle
        return dict(d=0.75*W, h=h0, beta=10.0, along=False)
    if cls == 3:  # Steep
        return dict(d=0.65*W, h=max(h0-0.3, hmin), beta=min(mean_slope_deg, betamax_deg), along=True)
    if cls == 4:  # Ridge/Valley
        return dict(d=0.60*W, h=max(h0-0.3, hmin), beta=min(mean_slope_deg, betamax_deg), along=True)
    return dict(d=0.75*W, h=h0, beta=5.0, along=False)

def policy_superpixel_consensus(policy: np.ndarray, labels: np.ndarray, min_frac: float = 0.60) -> np.ndarray:
    """Within each SLIC superpixel, if one class occupies ≥min_frac, assign the whole superpixel to it.

    Args:
        policy (np.ndarray): per-cell policy class raster.
        labels (np.ndarray): superpixel labels aligned to ``policy``.
        min_frac (float): minimum fraction for consensus assignment.

    Returns:
        np.ndarray: updated policy raster with superpixel consensus applied.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import policy_superpixel_consensus
        >>> policy = np.array([[1, 1], [0, 2]])
        >>> labels = np.array([[1, 1], [1, 2]])
        >>> out = policy_superpixel_consensus(policy, labels, min_frac=0.5)
    """
    out = policy.copy()
    ids = np.unique(labels[labels > 0])
    for rid in ids:
        m = (labels == rid)
        vals, cnts = np.unique(policy[m], return_counts=True)
        mask = vals > 0
        if not np.any(mask): 
            continue
        vals, cnts = vals[mask], cnts[mask]
        j = int(np.argmax(cnts))
        maj, frac = int(vals[j]), float(cnts[j]) / float(m.sum())
        if frac >= min_frac:
            out[m] = maj
    return out

def policy_mode_filter(policy: np.ndarray, size: int = 3) -> np.ndarray:
    """Mode filter on nonzero classes to remove salt-and-pepper noise.

    Args:
        policy (np.ndarray): policy class raster.
        size (int): neighborhood size (square window).

    Returns:
        np.ndarray: filtered policy raster.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import policy_mode_filter
        >>> out = policy_mode_filter(np.array([[1, 0], [2, 2]]), size=3)
    """
    def _mode(block):
        b = block.astype(np.int32)
        b = b[b > 0]
        if b.size == 0: 
            return 0
        v, c = np.unique(b, return_counts=True)
        return int(v[np.argmax(c)])
    return ndi.generic_filter(policy, _mode, size=size, mode="nearest")

def summarize_regions(macro: np.ndarray, 
                    reg_cls: dict[int,int],
                    feats: dict[str,np.ndarray],
                    W: float,
                    h0: float,
                    hmin: float,
                    betamax_deg: float,
                    cell_m: float) -> list[dict]:
    """Compile region-wise metrics and recommended survey policies.

    Args:
        macro (np.ndarray): macroregion label raster.
        reg_cls (dict[int, int]): mapping from macro id to policy class.
        feats (dict[str, np.ndarray]): feature rasters used for stats.
        W (float): nominal swath width.
        h0 (float): baseline altitude parameter.
        hmin (float): minimum altitude permitted.
        betamax_deg (float): maximum allowed heading offset.
        cell_m (float): raster cell size in metres.

    Returns:
        list[dict]: per-region policy summary rows.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import summarize_regions
        >>> macro = np.array([[1, 1], [2, 2]])
        >>> reg_cls = {1: 1, 2: 2}
        >>> feats = {"slope": np.zeros((2, 2)), "aspect": np.zeros((2, 2)), "vrm": np.zeros((2, 2)), "bpi": np.zeros((2, 2))}
        >>> rows = summarize_regions(macro, reg_cls, feats, W=3.0, h0=5.0, hmin=2.0, betamax_deg=20.0, cell_m=1.0)
    """
    rows = []
    for rid in np.unique(macro[macro>0]):
        m = (macro == rid)
        cls = int(reg_cls.get(int(rid), 0))
        mslope = float(np.nanmean(np.degrees(feats["slope"][m])))
        pq = policy_for_class(cls, W, h0, hmin, betamax_deg, mslope)
        psi = axial_mean_heading(m, feats["aspect"], feats["slope"], along=bool(pq["along"]))
        rows.append(dict(
            id=int(rid), cls=cls,
            area_m2=float(m.sum()) * (cell_m*cell_m),
            psi_deg=float(np.degrees(psi)),
            d_m=float(pq["d"]), h_m=float(pq["h"]), beta_deg=float(pq["beta"]),
            mean_slope_deg=mslope,
            mean_vrm=float(np.nanmean(feats["vrm"][m])),
            mean_bpi=float(np.nanmean(feats["bpi"][m])),
            viewshed_flag=bool(cls==4),
        ))
    return rows

def export_regions_csv(rows: list[dict], path) -> None:
    """Write macroregion summaries to CSV.

    Args:
        rows (list[dict]): list of per-region policy records.
        path (str | Path): destination CSV path.

    Returns:
        None

    Examples:
        >>> from mc_geomap_utils.macroregions import export_regions_csv
        >>> export_regions_csv([{"id": 1, "cls": 1}], "regions.csv")
    """
    keys = [
        "id","cls","area_m2","psi_deg","d_m","h_m","beta_deg",
        "mean_slope_deg","mean_vrm","mean_bpi","viewshed_flag",
        # optional primitive/spiral fields (populated when enabled)
        "plan_type","x0","y0","r_out","r_in","ring_slope_deg","ring_cv",
    ]
    path = str(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"[write] region summary -> {path}")

def _macro_gamma_from_sp_gamma(gamma: dict[tuple[int,int], float], uf) -> dict[tuple[int,int], float]:
    """Aggregate superpixel γ_ij to macro-level γ̄_IJ under current UF mapping."""
    sum_g = defaultdict(float)
    cnt_g = defaultdict(int)
    for (a, b), g in gamma.items():
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        i, j = (ra, rb) if ra < rb else (rb, ra)
        sum_g[(i, j)] += float(g)
        cnt_g[(i, j)] += 1
    return {k: (sum_g[k] / cnt_g[k]) for k in sum_g}

def _absorb_small_macros(ids, uf, SP, gamma_sp, min_area_px: int, prefer_same_class: bool = True) -> int:
    """Merge any macro with area < min_area_px into the best neighbor.
        score = γ̄_IJ + penalty(if class mismatch). Returns number of merges."""
    if min_area_px <= 0:
        return 0
    merged = 0
    # Iterate a few times in case merges create new small macros
    for _ in range(5):
        roots = sorted({uf.find(i) for i in ids})
        small = [r for r in roots if SP[r]["area_px"] < min_area_px]
        if not small:
            break
        macro_gamma = _macro_gamma_from_sp_gamma(gamma_sp, uf)
        typical = float(np.nanmedian(list(macro_gamma.values()))) if macro_gamma else 1.0
        penalty_sameclass = 0.0
        penalty_diffclass = 0.5 * typical if prefer_same_class else 0.0
        for r in small:
            # collect neighbors at macro level
            cands = []
            for (i, j), g in macro_gamma.items():
                if i == r:
                    cands.append((g, j))
                elif j == r:
                    cands.append((g, i))
            if not cands:
                continue
            # choose neighbor with lowest score
            best_score, best_n = None, None
            for g, n in cands:
                pen = penalty_sameclass if SP[r]["cls"] == SP[n]["cls"] else penalty_diffclass
                score = float(g) + pen
                if best_score is None or score < best_score:
                    best_score, best_n = score, n
            if best_n is None:
                continue
            ra, rb = uf.find(r), uf.find(best_n)
            if ra == rb:
                continue
            root = uf.union(ra, rb)
            other = rb if root == ra else ra
            # update macro stats
            SP[root]["area_px"] = int(SP[ra]["area_px"] + SP[rb]["area_px"])
            # keep class of the larger side (or of the neighbor we merged into)
            SP[root]["cls"] = SP[rb]["cls"] if SP[rb]["area_px"] >= SP[ra]["area_px"] else SP[ra]["cls"]
            merged += 1
    return merged


def absorb_small_islands(labels: np.ndarray, min_area_px: int) -> np.ndarray:
    """Replace very small regions with their most common neighboring label.

    Args:
        labels (np.ndarray): macroregion label raster.
        min_area_px (int): regions with area < min_area_px are absorbed.

    Returns:
        np.ndarray: relabeled raster.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import absorb_small_islands
        >>> labels = np.array([[1, 1], [1, 2]])
        >>> absorb_small_islands(labels, min_area_px=2)
    """
    if min_area_px <= 0:
        return labels
    L = labels.copy()
    ids, counts = np.unique(L[L > 0], return_counts=True)
    small = [int(rid) for rid, cnt in zip(ids, counts) if cnt < min_area_px]
    if not small:
        return L
    for rid in small:
        mask = (L == rid)
        if not np.any(mask):
            continue
        dil = ndi.binary_dilation(mask, structure=np.ones((3, 3), bool))
        neighbors = L[dil & (~mask)]
        neighbors = neighbors[neighbors > 0]
        if neighbors.size == 0:
            continue
        vals, cnts = np.unique(neighbors, return_counts=True)
        target = int(vals[np.argmax(cnts)])
        L[mask] = target
    return L


def smooth_macro_labels(labels: np.ndarray, radius_px: int = 2) -> np.ndarray:
    """Morphologically smooth jagged macro boundaries (closing+opening per region).

    Regions are processed in descending area so larger regions keep priority when overlaps occur.

    Args:
        labels (np.ndarray): macroregion label raster.
        radius_px (int): structuring element radius in pixels.

    Returns:
        np.ndarray: smoothed macroregion labels.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.macroregions import smooth_macro_labels
        >>> smooth_macro_labels(np.array([[1, 1], [1, 2]]), radius_px=1)
    """
    if radius_px <= 0:
        return labels
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    if ids.size == 0:
        return labels
    order = [rid for rid, _ in sorted(zip(ids, counts), key=lambda t: t[1], reverse=True)]
    se = disk(radius_px)
    out = np.zeros_like(labels, dtype=np.int32)
    for rid in order:
        mask = (labels == rid)
        if not np.any(mask):
            continue
        sm = binary_closing(mask, se)
        sm = binary_opening(sm, se)
        if not np.any(sm):
            sm = mask  # fallback
        out[(sm) & (out == 0)] = int(rid)
    return out
