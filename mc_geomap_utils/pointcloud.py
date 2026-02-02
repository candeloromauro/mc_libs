from __future__ import annotations

from pathlib import Path
from typing import Any, cast, Tuple

import numpy as np
from scipy.spatial import Delaunay
from scipy import ndimage as ndi

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from rasterio.transform import from_origin as rasterio_from_origin
except ImportError:
    rasterio_from_origin = None

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.segmentation import slic

# --- helpers for robust feature computation on sparse DEMs ---
def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    """Fill NaNs by nearest-neighbor (used only for derivative computations)."""
    mask = ~np.isfinite(a)
    if not np.any(mask):
        return a.copy()
    idx = cast(tuple[np.ndarray, ...], ndi.distance_transform_edt(mask, return_distances=False, return_indices=True))
    return a[tuple(idx)]

def delaunay_mesh(points: np.ndarray) -> Delaunay:
    """Build a 2.5D Delaunay triangulation from XY coordinates.

    Args:
        points (np.ndarray): input point cloud with columns ``x, y, z``.

    Returns:
        Delaunay: triangulation object describing XY simplices.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.pointcloud import delaunay_mesh
        >>> tri = delaunay_mesh(np.random.rand(10, 3))
    """
    return Delaunay(points[:, :2])

def mesh_to_obj(points: np.ndarray, tri: Delaunay, filename: Path | str = "mesh.obj") -> Path:
    """Write a triangulated mesh to disk in OBJ format.

    Args:
        points (np.ndarray): vertex positions used by the triangulation.
        tri (Delaunay): triangulation describing mesh faces.
        filename (Path | str): destination file path for the OBJ export.

    Returns:
        Path: path to the created OBJ file.

    Raises:
        RuntimeError: If ``trimesh`` is unavailable.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.pointcloud import delaunay_mesh, mesh_to_obj
        >>> pts = np.random.rand(10, 3)
        >>> tri = delaunay_mesh(pts)
        >>> _ = mesh_to_obj(pts, tri, "mesh.obj")
    """
    if trimesh is None:  # pragma: no cover
        raise RuntimeError("trimesh is required to export meshes; install trimesh.")
    output_path = Path(filename)
    mesh = trimesh.Trimesh(vertices=points, faces=tri.simplices, process=False)
    mesh.export(output_path)
    print(f"Exported mesh to {output_path}")
    return output_path

def delaunay_to_open3d_mesh(points: np.ndarray, tri: Delaunay) -> Any:
    """Convert a SciPy triangulation to an Open3D TriangleMesh instance.

    Args:
        points (np.ndarray): vertex coordinates (N x 3).
        tri (Delaunay): triangulation whose simplices form mesh faces.

    Returns:
        Any: Open3D ``TriangleMesh`` ready for visualization.

    Raises:
        RuntimeError: If ``open3d`` is unavailable.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.pointcloud import delaunay_mesh, delaunay_to_open3d_mesh
        >>> pts = np.random.rand(10, 3)
        >>> tri = delaunay_mesh(pts)
        >>> mesh = delaunay_to_open3d_mesh(pts, tri)
    """
    if o3d is None:  # pragma: no cover
        raise RuntimeError("Open3D is required for mesh visualization; install open3d.")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    mesh.compute_vertex_normals()
    return mesh

def grid_to_dem(points: np.ndarray, cell: float = 0.5, min_count: int = 3, agg: str = "median"):
    """Rasterize an irregular point cloud into a DEM and count grid.

    Args:
        points (np.ndarray): input XYZ coordinates.
        cell (float): grid resolution in the XY plane.
        min_count (int): minimum samples required to keep a raster cell.
        agg (str): aggregation strategy ("median", "mean", or percentile ``pXX``).

    Returns:
        tuple[np.ndarray, np.ndarray, Any]: DEM array, per-cell counts, and raster transform.

    Raises:
        ValueError: If an unsupported aggregation strategy is requested.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.pointcloud import grid_to_dem
        >>> pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 2.0]])
        >>> dem, counts, transform = grid_to_dem(pts, cell=0.5)
    """
    agg_key = agg.lower()
    percentile: float | None = None
    if agg_key.startswith("p") and len(agg_key) > 1:
        try:
            percentile = float(agg_key[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid percentile aggregator '{agg}'") from exc
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"Percentile aggregator must be between 0 and 100; got {percentile}")
    elif agg_key not in {"median", "mean"}:
        raise ValueError(f"Unsupported aggregator '{agg}'. Use 'median', 'mean', or 'pXX'.")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    nx = int(np.ceil((xmax - xmin) / cell))
    ny = int(np.ceil((ymax - ymin) / cell))
    nx = max(nx, 1)
    ny = max(ny, 1)
    dem = np.full((ny, nx), np.nan, dtype=np.float32)
    count = np.zeros((ny, nx), dtype=np.int32)

    ix = np.clip(((x - xmin) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / cell).astype(int), 0, ny - 1)
    flat = (iy * nx + ix).astype(np.int64)
    order = np.argsort(flat)
    flat_sorted = flat[order]
    z_sorted = z[order]
    unique, idx = np.unique(flat_sorted, return_index=True)
    idx_next = np.r_[idx[1:], z_sorted.size]
    for u, i0, i1 in zip(unique, idx, idx_next):
        r = u // nx
        c = u % nx

        vals = z_sorted[i0:i1]
        if agg_key == "median":
            dem[r, c] = float(np.nanmedian(vals))
        elif agg_key == "mean":
            dem[r, c] = float(np.nanmean(vals))
        else:
            if percentile is None:
                raise RuntimeError("Percentile aggregator misconfigured.")
            dem[r, c] = float(np.nanpercentile(vals, percentile))
        count[r, c] = vals.size

    dem[count < min_count] = np.nan
    transform = rasterio_from_origin(xmin, ymax, cell, cell) if rasterio_from_origin else None
    return dem, count, transform

def dem_to_features(dem: np.ndarray, cell: float = 0.5, smooth_sigma_cells: float = 1.0,
                    vrm_win_cells: int = 11):
    """Derive terrain features (slope, curvature, VRM, BPI) from a DEM.

    Args:
        dem (np.ndarray): raster DEM with NaNs marking gaps.
        cell (float): grid spacing in metres.
        smooth_sigma_cells (float): Gaussian sigma (in cells) for derivative smoothing.

    Returns:
        dict[str, np.ndarray]: mapping of feature names to float32 rasters.

    Raises:
        ValueError: If the DEM contains no finite samples.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.pointcloud import dem_to_features
        >>> dem = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        >>> feats = dem_to_features(dem, cell=1.0)
    """
    Z = dem.copy()
    # original support mask
    support = np.isfinite(Z)
    if not np.any(support):
        raise ValueError("DEM contains no finite cells; check gridding cell size / min_count in grid_to_dem.")
    # Fill NaNs only for derivative computations; we'll restore mask later
    Zf = _fill_nan_nearest(Z)
    # smooth to stabilize derivatives
    if smooth_sigma_cells and smooth_sigma_cells > 0:
        Zf = ndi.gaussian_filter(Zf, sigma=smooth_sigma_cells)
    # gradients
    dzdx = ndi.sobel(Zf, axis=1, mode='nearest') / (8*cell)
    dzdy = ndi.sobel(Zf, axis=0, mode='nearest') / (8*cell)
    slope = np.arctan(np.hypot(dzdx, dzdy))  # radians in [0, π/2]
    aspect = np.arctan2(dzdy, dzdx)          # radians in (-π, π]
    # Hessian & principal curvatures (small-slope approx)
    if hessian_matrix is not None and hessian_matrix_eigvals is not None:
        hm_kwargs = {"sigma": 1, "order": "rc"}
        try:
            Hxx, Hxy, Hyy = cast(
                tuple[np.ndarray, np.ndarray, np.ndarray],
                hessian_matrix(Zf, use_gaussian_derivatives=False, **hm_kwargs),  # type: ignore[call-arg]
            )
        except TypeError:
            # older scikit-image without that kwarg
            Hxx, Hxy, Hyy = cast(
                tuple[np.ndarray, np.ndarray, np.ndarray],
                hessian_matrix(Zf, **hm_kwargs),
            )

        # build a real ndarray for the newer API
        H_elems = np.stack((Hxx, Hxy, Hyy), axis=0).astype(float, copy=False)

        try:
            k1_arr, k2_arr = hessian_matrix_eigvals(H_elems)  # single-arg API
        except TypeError:
            k1_arr, k2_arr = hessian_matrix_eigvals(Hxx, Hxy, Hyy)  # type: ignore[call-arg]

    else:
        # fallback: second differences if scikit-image is unavailable
        Hxx = ndi.laplace(ndi.sobel(Zf, axis=1))
        Hyy = ndi.laplace(ndi.sobel(Zf, axis=0))
        Hxy = ndi.gaussian_filter(ndi.sobel(ndi.sobel(Zf, axis=1), axis=0), 1)
        k1_arr, k2_arr = Hxx, Hyy
    # Convert curvatures to arrays for downstream masking/export
    k1 = np.asarray(k1_arr, dtype=float)
    k2 = np.asarray(k2_arr, dtype=float)

    # VRM (normal dispersion)
    nx, ny, nz = -dzdx, -dzdy, np.ones_like(Zf)
    norm = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-9
    nx, ny, nz = nx/norm, ny/norm, nz/norm
    win = np.ones((vrm_win_cells, vrm_win_cells), dtype=float)
    sx = ndi.convolve(nx, win, mode='nearest')/win.size
    sy = ndi.convolve(ny, win, mode='nearest')/win.size
    sz = ndi.convolve(nz, win, mode='nearest')/win.size
    vrm = 1.0 - np.sqrt(sx*sx + sy*sy + sz*sz)
    # BPI as Difference-of-Gaussians (operate on the filled/smoothed surface)
    Zcoarse = ndi.gaussian_filter(Zf, sigma=25.0/cell)
    Zfine   = ndi.gaussian_filter(Zf, sigma=5.0/cell)
    bpi = Zcoarse - Zfine
    # restore original support: set features to NaN where DEM had no data
    features_to_mask: Tuple[np.ndarray, ...] = (slope, aspect, k1, k2, vrm, bpi)
    for arr_feature in features_to_mask:
        arr_feature[~support] = np.nan
    return {
        "slope":  slope.astype(np.float32),
        "aspect": aspect.astype(np.float32),
        "k1":     k1.astype(np.float32),
        "k2":     k2.astype(np.float32),
        "vrm":    vrm.astype(np.float32),
        "bpi":    bpi.astype(np.float32),
    }

# Basic SLIC segmentation entry point
def feature_to_slic_segmentation(features: dict, cell: float, W: float = 3.0, compactness: float = 30.0):
    """Segment feature rasters with SLIC to produce superpixel labels.

    Args:
        features (dict): dictionary containing feature rasters (slope, k1, k2, vrm, bpi).
        cell (float): grid spacing used when estimating segment sizes.
        W (float): nominal swath width controlling the desired segment area.
        compactness (float): SLIC compactness parameter balancing color and spatial distance.

    Returns:
        np.ndarray: integer label image (same shape as the DEM features).

    Raises:
        RuntimeError: If scikit-image is unavailable.
        ValueError: If no valid pixels exist in the input features.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.pointcloud import feature_to_slic_segmentation
        >>> feats = {"slope": np.ones((10, 10)), "k1": np.zeros((10, 10)), "k2": np.zeros((10, 10)), "vrm": np.zeros((10, 10)), "bpi": np.zeros((10, 10))}
        >>> labels = feature_to_slic_segmentation(feats, cell=1.0)
    """
    if slic is None:
        raise RuntimeError("scikit-image not available: cannot run SLIC segmentation")
    # stack channels; normalize per-channel
    chans = [features[k] for k in ("slope","k1","k2","vrm","bpi") if k in features]
    F = np.stack(chans, axis=-1)
    m = np.nanmean(F, axis=(0,1)); s = np.nanstd(F, axis=(0,1)) + 1e-9
    Fz = (F - m)/s
    # valid only where ALL channels are finite
    valid_mask = np.isfinite(Fz).all(axis=-1)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError("SLIC: no valid pixels (all-NaN features). Check DEM support, cell size, or reduce gridding min_count.")
    # estimate number of segments so each is ~ (3 m)^2 in area, but cap by n_valid
    target_cells = max(16, int((W/cell)**2))
    est = max(1, int(n_valid / target_cells))
    n_segments = int(max(10, min(est, n_valid // 2)))
    Fz = np.ascontiguousarray(Fz.astype(np.float32, copy=False))
    labels = slic(
        Fz,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        mask=valid_mask,
        channel_axis=-1,          # explicit, future-proof
    )

    return labels.astype(np.int32)
