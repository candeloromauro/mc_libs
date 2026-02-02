from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import math

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 11,
})
import numpy as np
from scipy import ndimage as ndi
try:
    import open3d as o3d
except ImportError:
    o3d = None
from matplotlib import cm as _mpl_cm
from matplotlib.colors import Colormap, ListedColormap, BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Ellipse
try:
    import plotly.graph_objs as go
except Exception:
    go = None
from matplotlib.markers import MarkerStyle

from scipy.spatial import Delaunay

from . import primitives as _prim


# Class to organise plots inputs
@dataclass
class PlotDataSource:
    """Container helping turn assorted inputs into labelled XY series.

    Args:
        series (Mapping[str, np.ndarray]): dictionary mapping labels to value arrays.

    Returns:
        PlotDataSource: initialised data source ready for plotting.
    """

    series: dict[str, np.ndarray]

    @classmethod
    def from_input(cls, raw: Any) -> "PlotDataSource":
        """Normalise many possible user inputs into a consistent mapping.

        Args:
            raw (Any): mapping of labels to arrays or alternating key/value sequence.

        Returns:
            PlotDataSource: wrapper holding the parsed series.

        Examples:
            >>> from mc_geomap_utils.plotting import PlotDataSource
            >>> ds = PlotDataSource.from_input({"x": [0, 1], "y": [1, 2]})
        """

        if isinstance(raw, Mapping):
            items = raw.items()
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            if len(raw) % 2 != 0:
                raise ValueError("Expected an even number of elements (key/value pairs).")
            items = zip(raw[0::2], raw[1::2])
        else:
            raise TypeError("Data must be a mapping or a flat list/tuple of key/value pairs.")

        series: dict[str, np.ndarray] = {}
        for key, values in items:
            if not isinstance(key, str):
                raise TypeError(f"Series name must be str, got {type(key).__name__!s}.")
            if not isinstance(values, (Sequence, np.ndarray)) or isinstance(values, (str, bytes, bytearray)):
                raise TypeError(f"Series '{key}' must be a sequence of numbers.")
            series[key] = np.asarray(values, dtype=float)
        if not series:
            raise ValueError("No data provided.")
        return cls(series)

    def iter_xy_pairs(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Yield paired X/Y arrays for plotting convenience.

        Args:
            None

        Returns:
            list[tuple[str, np.ndarray, np.ndarray]]: tuples containing label, x-array, and y-array.

        Examples:
            >>> from mc_geomap_utils.plotting import PlotDataSource
            >>> ds = PlotDataSource.from_input({"x": [0, 1], "y": [1, 2]})
            >>> ds.iter_xy_pairs()
        """

        pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
        for key in sorted(k for k in self.series if k.lower().startswith("x")):
            suffix = key[1:]
            y_key = f"y{suffix}"
            if y_key not in self.series:
                raise KeyError(f"Missing matching series '{y_key}' for '{key}'.")
            x = self.series[key]
            y = self.series[y_key]
            if x.shape != y.shape:
                raise ValueError(
                    f"Series '{key}' and '{y_key}' have mismatched lengths ({x.size} vs {y.size})."
                )
            label = suffix or key
            pairs.append((label, x, y))
        if not pairs:
            raise ValueError("No x?/y? series pairs found.")
        return pairs

def show_plot(data: Any, shared_axes: bool = True, separate_axes: bool = True) -> Figure:
    """Render one or multiple XY series using Matplotlib subplots.

    Args:
        data (Any): sequence, mapping, or alternating list describing XY pairs.
        shared_axes (bool): whether generated subplots share their axes.
        separate_axes (bool): split series across subplots (True) or overlay on one axis (False).

    Returns:
        Figure: Matplotlib figure containing the requested plot layout.

    Examples:
        >>> from mc_geomap_utils.plotting import show_plot
        >>> fig = show_plot({"x": [0, 1], "y": [1, 2]})
    """
    datasource = PlotDataSource.from_input(data)
    pairs = datasource.iter_xy_pairs()

    if separate_axes:
        n_plots = len(pairs)
        n_cols = math.ceil(math.sqrt(n_plots))
        n_rows = math.ceil(n_plots / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 3.0 * n_rows),
            sharex=shared_axes,
            sharey=shared_axes,
            squeeze=False,
        )

        for ax, (label, x, y) in zip(axes.flat, pairs):
            ax.plot(x, y, marker="o")
            ax.set_title(f"Series {label}")
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[len(pairs):]:
            ax.set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for label, x, y in pairs:
            ax.plot(x, y, marker="o", label=f"Series {label}")
        ax.set_title("Combined Series")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    return fig

# Simple mosaic helper from existing PNGs
def save_mosaic(paths: list[Path], out_path: Path, titles: list[str] | None = None, figsize=(12, 10)) -> None:
    """Save a 2x2 mosaic from up to 4 image paths.

    Args:
        paths (list[Path]): image file paths to include (up to 4).
        out_path (Path): output image path for the mosaic.
        titles (list[str] | None): optional per-image titles.
        figsize: Matplotlib figure size.

    Returns:
        None

    Examples:
        >>> from pathlib import Path
        >>> from mc_geomap_utils.plotting import save_mosaic
        >>> save_mosaic([Path("a.png"), Path("b.png")], Path("mosaic.png"))
    """
    imgs_raw: list[np.ndarray] = []
    valid_titles: list[str] = []
    for j, p in enumerate(paths):
        try:
            import matplotlib.image as mpimg
            imgs_raw.append(mpimg.imread(os.fspath(p)))
            valid_titles.append(titles[j] if titles and j < len(titles) else "")
        except Exception:
            continue
    if not imgs_raw:
        return

    # Pad images to the same HxW so their displayed size matches without stretching.
    max_h = max(img.shape[0] for img in imgs_raw)
    max_w = max(img.shape[1] for img in imgs_raw)
    imgs: list[np.ndarray] = []
    for img in imgs_raw:
        h, w = img.shape[:2]
        pad = np.ones((max_h, max_w, img.shape[2]), dtype=img.dtype)
        pad[:h, :w, ...] = img
        imgs.append(pad)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.flat
    for ax, img, ttl in zip(axes_flat, imgs, valid_titles):
        ax.imshow(img)
        if ttl:
            ax.set_title(ttl)
        ax.axis("off")
    for ax in axes_flat[len(imgs):]:
        ax.axis("off")
    fig.savefig(os.fspath(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_mosaic_1x2(paths: list[Path], out_path: Path, titles: list[str] | None = None, figsize=(12, 5)) -> None:
    """Save a 1x2 mosaic from up to 2 image paths, padding sizes to match.

    Args:
        paths (list[Path]): image file paths to include (up to 2).
        out_path (Path): output image path for the mosaic.
        titles (list[str] | None): optional per-image titles.
        figsize: Matplotlib figure size.

    Returns:
        None

    Examples:
        >>> from pathlib import Path
        >>> from mc_geomap_utils.plotting import save_mosaic_1x2
        >>> save_mosaic_1x2([Path("a.png"), Path("b.png")], Path("mosaic_1x2.png"))
    """
    imgs_raw: list[np.ndarray] = []
    valid_titles: list[str] = []
    for j, p in enumerate(paths):
        try:
            import matplotlib.image as mpimg
            imgs_raw.append(mpimg.imread(os.fspath(p)))
            valid_titles.append(titles[j] if titles and j < len(titles) else "")
        except Exception:
            continue
    if not imgs_raw:
        return

    max_h = max(img.shape[0] for img in imgs_raw)
    max_w = max(img.shape[1] for img in imgs_raw)
    imgs: list[np.ndarray] = []
    for img in imgs_raw:
        h, w = img.shape[:2]
        pad = np.ones((max_h, max_w, img.shape[2]), dtype=img.dtype)
        pad[:h, :w, ...] = img
        imgs.append(pad)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax, img, ttl in zip(axes, imgs, valid_titles):
        ax.imshow(img)
        if ttl:
            ax.set_title(ttl)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.savefig(os.fspath(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

def _resolve_draw_geometries() -> Callable[[list[Any]], None]:
    """Fetch ``open3d.visualization.draw_geometries`` with runtime guards.

    Args:
        None

    Returns:
        Callable[[list[Any]], None]: function capable of displaying Open3D geometries.
    """
    if o3d is None:  # pragma: no cover - runtime guard
        raise RuntimeError("Open3D is not available; install open3d to enable 3D visualization.")
    visualization = getattr(o3d, "visualization", None)
    if visualization is None:
        raise AttributeError("open3d.visualization is not available; update Open3D.")
    draw_fn = getattr(visualization, "draw_geometries", None)
    if draw_fn is None:
        raise AttributeError("open3d.visualization.draw_geometries is not available; update Open3D.")
    return cast(Callable[[list[Any]], None], draw_fn)

# --- Visualization and reporting helpers -------------------------------------

def _extent_from_transform(transform: Any, shape: tuple[int, int]) -> tuple[float, float, float, float] | None:
    """Derive imshow extents from an affine transform and array shape.

    Args:
        transform (Any): raster transform describing pixel-to-world mapping.
        shape (tuple[int, int]): raster dimensions (rows, columns).

    Returns:
        tuple[float, float, float, float] | None: bounding box suitable for Matplotlib extent.
    """
    if transform is None:
        return None
    try:
        xmin = float(transform.c)
        xmax = xmin + float(transform.a) * shape[1]
        ymax = float(transform.f)
        ymin = ymax + float(transform.e) * shape[0]
        return (xmin, xmax, ymin, ymax)
    except Exception:
        return None

def _masked_imshow(
    ax,
    arr: np.ndarray,
    extent: tuple[float, float, float, float] | None = None,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    nan_color=(0.7, 0.7, 0.7, 1.0),
):
    """Plot an array while masking invalid cells and styling the colormap.

    Args:
        ax (Any): Matplotlib axis onto which the image will be drawn.
        arr (np.ndarray): data array potentially containing NaNs.
        extent (tuple[float, float, float, float] | None): spatial extent passed to imshow.
        cmap (str): name of the Matplotlib colormap to use.
        vmin (float | None): optional lower bound for color scaling.
        vmax (float | None): optional upper bound for color scaling.
        nan_color (tuple[float, float, float, float]): RGBA color for NaN masking.

    Returns:
        Any: Matplotlib image returned by ``imshow``.
    """
    assert _mpl_cm is not None
    cmap_obj = cast(Colormap, _mpl_cm.get_cmap(cmap))
    try:
        cmap_obj = cmap_obj.copy()
    except Exception:
        import copy as _copy
        cmap_obj = _copy.copy(cmap_obj)
    cmap_obj.set_bad(nan_color)
    data = np.ma.masked_invalid(arr)
    imshow_kwargs: dict[str, Any] = {
        "origin": "upper",
        "cmap": cmap_obj,
        "interpolation": "nearest",
    }
    if extent is not None:
        imshow_kwargs["extent"] = tuple(float(v) for v in extent)
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax
    im = ax.imshow(data, **imshow_kwargs)
    return im

# Labels

def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    """Compute a boolean mask highlighting label boundaries.

    Args:
        labels (np.ndarray): integer label raster.

    Returns:
        np.ndarray: boolean array with True where label transitions occur.
    """
    L = labels
    b = np.zeros(L.shape, dtype=bool)
    b[1:, :]  |= (L[1:, :]  != L[:-1, :])
    b[:-1, :] |= (L[1:, :]  != L[:-1, :])
    b[:, 1:]  |= (L[:, 1:]  != L[:, :-1])
    b[:, :-1]  |= (L[:, 1:] != L[:, :-1])
    return b

def _label_colorize(labels: np.ndarray, bg_label: int = 0, seed: int = 0) -> np.ndarray:
    """Assign reproducible colors to label rasters, with transparent background.

    Args:
        labels (np.ndarray): integer label image.
        bg_label (int): label value to treat as transparent background.
        seed (int): RNG seed used to shuffle the color table.

    Returns:
        np.ndarray: RGBA image matching the label array shape.
    """
    if labels.size == 0:
        return np.zeros((*labels.shape, 4), dtype=float)
    unique = np.unique(labels)
    max_lab = int(unique.max())
    assert _mpl_cm is not None
    cmap = cast(Colormap, _mpl_cm.get_cmap("tab20"))
    rng = np.random.default_rng(seed)
    n_colors = 20
    perm = rng.permutation(n_colors)
    lut = np.zeros((max_lab + 1, 4), dtype=float)
    k = 0
    for lab in unique:
        if lab == bg_label:
            continue
        lut[lab] = cmap(perm[k % n_colors])
        k += 1
    rgba = lut[labels]
    rgba[labels == bg_label, 3] = 0.0
    return rgba

# Boundary helper
def _draw_double_boundary(ax, mask: np.ndarray, extent: tuple[float, float, float, float] | None = None,
                          outer_color: str = "black", inner_color: str = "white",
                          outer_lw: float = 1.2, inner_lw: float = 0.9, alpha: float = 0.9) -> None:
    """Draw a tight double line (outer+inner) for readability."""
    if mask.size == 0:
        return
    levels = [0.5]
    contour_kw: dict[str, Any] = {"levels": levels, "colors": outer_color, "linewidths": outer_lw,
                                  "alpha": alpha, "origin": "upper"}
    if extent is not None:
        contour_kw["extent"] = extent
    try:
        ax.contour(mask.astype(float), **contour_kw)
        contour_kw.update({"colors": inner_color, "linewidths": inner_lw})
        ax.contour(mask.astype(float), **contour_kw)
    except Exception:
        pass

# Plotting functions

def plot_point_cloud(points: np.ndarray, *, draw_fn: Callable[[list[Any]], None] | None = None) -> None:
    """Send a point cloud to Open3D for interactive inspection.

    Args:
        points (np.ndarray): XYZ coordinates to render.
        draw_fn (Callable[[list[Any]], None] | None): optional custom Open3D draw function.

    Returns:
        None: the side effect is an Open3D window displaying the cloud.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_point_cloud
        >>> plot_point_cloud(np.random.rand(100, 3))
    """
    if o3d is None:  # pragma: no cover - runtime guard
        raise RuntimeError("Open3D is not available; install open3d to enable 3D visualization.")
    if draw_fn is None:
        draw_fn = _resolve_draw_geometries()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    draw_fn([pcd])

def plot_mesh(points: np.ndarray, tri: Delaunay, *, show: bool = True) -> Figure:
    """Plot a triangulated surface using Matplotlib's 3D toolkit.

    Args:
        points (np.ndarray): vertex coordinates with columns x, y, z.
        tri (Delaunay): triangulation object describing the mesh faces.
        show (bool): display the plot interactively after creation.

    Returns:
        Figure: Matplotlib figure containing the rendered mesh.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_mesh
        >>> from scipy.spatial import Delaunay
        >>> pts = np.random.rand(10, 3)
        >>> tri = Delaunay(pts[:, :2])
        >>> fig = plot_mesh(pts, tri, show=False)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[call-overload]
    ax.plot_trisurf(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        triangles=tri.simplices,
        cmap="viridis",
        edgecolor="gray",
        linewidth=0.2,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    if show:
        plt.show()
    return fig

def plot_dem(
    dem: np.ndarray,
    transform: Any | None = None,
    title: str = "DEM (Z)",
    cmap: str = "terrain",
    save: Path | None = None,
    show: bool = False,
    colorbar: bool = True,
):
    """Display or export a DEM using masked shading.

    Args:
        dem (np.ndarray): digital elevation model raster.
        transform (Any | None): affine transform for deriving plot extent.
        title (str): title text applied to the figure.
        cmap (str): Matplotlib colormap name used to render elevation.
        save (Path | None): optional location for saving the PNG output.
        show (bool): display the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_dem
        >>> plot_dem(np.random.rand(10, 10), show=False)
    """
    assert plt is not None
    extent = _extent_from_transform(transform, dem.shape)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = _masked_imshow(ax, dem, extent=extent, cmap=cmap)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
        cbar.set_label("Elevation [m]")
    ax.set_title(title)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="box")
    if save is not None:
        fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_feature_map(arr: np.ndarray, name: str, transform: Any | None = None, save: Path | None = None, show: bool = False):
    """Render a derived feature raster (slope, VRM, BPI, etc.).

    Args:
        arr (np.ndarray): feature raster to visualise.
        name (str): human-readable feature name (controls scaling and label).
        transform (Any | None): affine transform for computing the extent.
        save (Path | None): optional output path for a saved image.
        show (bool): whether to display the plot interactively.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_feature_map
        >>> plot_feature_map(np.random.rand(10, 10), "slope", show=False)
    """
    assert plt is not None
    extent = _extent_from_transform(transform, arr.shape)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    lname = name.lower()
    data = np.array(arr, copy=True)
    cmap = "viridis"
    label = name
    if lname == "slope":
        data = np.degrees(data)
        label = "Slope (deg)"
        vmax = np.nanpercentile(data, 98) if np.isfinite(data).any() else None
        im = _masked_imshow(ax, data, extent=extent, cmap="viridis", vmin=0, vmax=vmax)
    elif lname == "bpi":
        vmax = np.nanpercentile(np.abs(data), 98) if np.isfinite(data).any() else None
        im = _masked_imshow(ax, data, extent=extent, cmap="coolwarm", vmin=-vmax if vmax else None, vmax=vmax)
    else:  # vrm, k1, k2, etc.
        im = _masked_imshow(ax, data, extent=extent, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
    cbar.set_label(label)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    if save is not None:
        fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_labels(labels: np.ndarray, transform: Any | None = None, base: np.ndarray | None = None,
                base_name: str = "Base", save: Path | None = None, show: bool = False,
                alpha_labels: float = 0.55, draw_boundaries: bool = True):
    """Overlay colorized labels on an optional base raster and export/display.

    Args:
        labels (np.ndarray): integer label image to visualize.
        transform (Any | None): spatial transform for extent computation.
        base (np.ndarray | None): optional background raster for context.
        base_name (str): descriptive name of the background raster.
        save (Path | None): location for saving the PNG overlay.
        show (bool): show the figure interactively when True.
        alpha_labels (float): opacity applied to the label overlay.
        draw_boundaries (bool): draw boundary outlines when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_labels
        >>> plot_labels(np.random.randint(0, 3, (10, 10)), show=False)
    """
    assert plt is not None
    extent = _extent_from_transform(transform, labels.shape)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    if base is not None:
        # pick colormap sensibly for base
        bname = base_name.lower()
        cmap = "terrain" if "dem" in bname else ("coolwarm" if "bpi" in bname else "viridis")
        _masked_imshow(ax, base, extent=extent, cmap=cmap)
    rgba = _label_colorize(labels, bg_label=0, seed=0)
    im_kwargs: dict[str, Any] = {"interpolation": "nearest", "alpha": alpha_labels}
    if extent is not None:
        im_kwargs["extent"] = tuple(float(v) for v in extent)
    ax.imshow(rgba, **im_kwargs)
    if draw_boundaries:
        bd = _label_boundaries(labels)
        # draw boundaries in black (as overlay)
        bd_img = np.where(bd, 0.0, np.nan)
        _masked_imshow(ax, bd_img, extent=extent, cmap="gray")
    ax.set_title("Labels (SLIC)")
    ax.set_aspect("equal", adjustable="box")
    if save is not None:
        fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_policy_map(policy_map: np.ndarray, transform, *, title="Policy classes",
                    save=None, show=False):
    """Render a policy-class raster with a categorical legend.

    Args:
        policy_map (np.ndarray): integer policy class raster.
        transform: affine transform for computing plot extent.
        title (str): figure title.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_policy_map
        >>> plot_policy_map(np.random.randint(0, 5, (10, 10)), transform=None, show=False)
    """
    def _fill_labels_nn(labels: np.ndarray) -> np.ndarray:
        L = labels.copy()
        mask = (L > 0)
        if not np.any(mask):
            return L
        # nearest-neighbor fill for zeros
        idx = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        if idx is None or len(idx) < 2:
            return L
        r_idx = np.asarray(idx[0], dtype=int)
        c_idx = np.asarray(idx[1], dtype=int)
        L[~mask] = L[r_idx[~mask], c_idx[~mask]]
        return L

    policy_map = _fill_labels_nn(policy_map)
    extent = _extent_from_transform(transform, policy_map.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}
    colors = [
        (0,0,0,0),          # 0 nodata transparent
        (0.85,0.85,0.85,1), # 1 Flat
        (0.65,0.85,0.65,1), # 2 Gentle
        (1.0,0.7,0.2,1),    # 3 Steep
        (0.55,0.45,0.85,1), # 4 Ridge/Valley
    ]
    labels = ["NoData","Flat","Gentle","Steep/Scarp","Ridge/Valley"]
    cmap = ListedColormap(colors)
    n_levels = getattr(cmap, "N", len(colors))
    norm = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5], n_levels)
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    ax.imshow(policy_map, origin="upper", cmap=cmap, norm=norm,
              interpolation="nearest", **extent_kw)
    # boundaries
    L = policy_map.astype(np.int32, copy=False)
    b = np.zeros(L.shape, dtype=bool)
    b[1:, :]   |= (L[1:, :] != L[:-1, :])
    b[:-1, :]  |= (L[1:, :] != L[:-1, :])
    b[:, 1:]   |= (L[:, 1:] != L[:, :-1])
    b[:, :-1]  |= (L[:, 1:] != L[:, :-1])
    _draw_double_boundary(ax, np.where(b, 1.0, 0.0), extent if extent_kw else None, alpha=0.9)
    ax.set_title(title); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    legend_elems = [Patch(facecolor=colors[i], edgecolor='none', label=labels[i]) for i in range(1,5)]
    ax.legend(handles=legend_elems, loc="lower right", frameon=True)
    if save: fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

def plot_macro_overlay(base: np.ndarray, macro_labels: np.ndarray, transform, *, title="Macroregions over base", save=None, show=False):
    """Overlay macroregion boundaries on a base raster.

    Args:
        base (np.ndarray): base raster (e.g., DEM) for context.
        macro_labels (np.ndarray): macroregion labels aligned to ``base``.
        transform: affine transform for computing plot extent.
        title (str): figure title.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_macro_overlay
        >>> plot_macro_overlay(np.random.rand(10, 10), np.ones((10, 10), dtype=int), transform=None, show=False)
    """
    extent = _extent_from_transform(transform, base.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}

    L = macro_labels
    b = np.zeros(L.shape, bool)
    b[1:,:]  |= (L[1:,:]  != L[:-1,:])
    b[:-1,:] |= (L[1:,:]  != L[:-1,:])
    b[:,1:]  |= (L[:,1:]  != L[:, :-1])
    b[:, :-1]|= (L[:,1:]  != L[:, :-1])
    boundary = np.where(b, 1.0, np.nan)

    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    im = ax.imshow(base, origin="upper", cmap=_mpl_cm.get_cmap("terrain"),
                   interpolation="nearest", **extent_kw)
    ax.imshow(boundary, origin="upper", cmap="gray", interpolation="nearest",
              alpha=0.9, **extent_kw)
    ax.set_title(title); ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5); cbar.set_label("Base")
    if save: fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

def plot_macro_with_seeds(
    dem: np.ndarray,
    macro_labels: np.ndarray,
    transform,
    seeds_rc: list[tuple[int, int]] | None = None,
    *,
    macro_transform=None,
    seeds_debug: dict | None = None,
    title: str = "DEM + Seeds + Macro overlay",
    save=None,
    show=False,
) -> None:
    """Render DEM + macro boundaries + seed markers.

    Args:
        dem (np.ndarray): DEM raster.
        macro_labels (np.ndarray): macroregion labels.
        transform: affine transform for DEM extent.
        seeds_rc (list[tuple[int, int]] | None): optional seed pixel coords.
        macro_transform: optional transform for macro labels (defaults to ``transform``).
        seeds_debug (dict | None): optional debug seed sets.
        title (str): figure title.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_macro_with_seeds
        >>> plot_macro_with_seeds(np.random.rand(5, 5), np.ones((5, 5), dtype=int), transform=None, show=False)
    """
    extent = _extent_from_transform(transform, dem.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}
    extent_macro = _extent_from_transform(macro_transform, macro_labels.shape) if macro_transform is not None else extent
    extent_kw_macro: dict[str, Any] = {} if extent_macro is None else {"extent": extent_macro}

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(dem, origin="upper", cmap="terrain", interpolation="nearest", **extent_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.5); cb.set_label("Elevation [m]")

    L = macro_labels.astype(np.int32, copy=False)
    b = _label_boundaries(L)
    _draw_double_boundary(ax, np.where(b, 1.0, 0.0),
                          extent_macro if extent_kw_macro else None,
                          outer_color="black", inner_color="white",
                          outer_lw=1.2, inner_lw=0.6, alpha=0.9)

    # Seeds
    def _plot_seed_set(seeds: list[tuple[int, int]], marker: str, color: str, label: str):
        if not seeds:
            return
        xs_plot: list[float] = []
        ys_plot: list[float] = []
        for rr, cc in seeds:
            if transform is not None:
                xw, yw = _pix_to_world(float(rr), float(cc), transform)
                xs_plot.append(xw); ys_plot.append(yw)
            else:
                xs_plot.append(float(cc)); ys_plot.append(float(rr))
        if xs_plot:
            marker_style = MarkerStyle(marker)
            if marker in {"+", "x"}:
                ax.scatter(xs_plot, ys_plot, s=40, marker=marker_style,
                           c=color, linewidths=1.5, facecolors="none",
                           label=label, zorder=4)
            else:
                ax.scatter(xs_plot, ys_plot, s=40, marker=marker_style,
                           facecolors="none", edgecolors=color,
                           linewidths=1.5, label=label, zorder=4)

    if seeds_debug is not None:
        _plot_seed_set(list(seeds_debug.get("seeds_all", [])), "+", "cyan", "all local maxima")
        _plot_seed_set(list(seeds_debug.get("seeds_macro", [])), "x", "orange", "after macro filter")
        _plot_seed_set(list(seeds_debug.get("seeds_kept", [])), "o", "magenta", "after prominence")
    elif seeds_rc is not None:
        _plot_seed_set(seeds_rc, "o", "magenta", "seeds")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(title)
    if save: fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

def plot_region_headings(
    macro_labels: np.ndarray,
    psi_deg_per_id: dict[int, float],
    transform,
    *,
    base: np.ndarray,                 # e.g., np.degrees(feats["slope"])
    title: str = "Region headings",
    save=None,
    show: bool = False,
    arrow_scale: float = 8.0,         # quiver scale (bigger → shorter arrows)
) -> None:
    """Draw one heading arrow per macroregion, overlaid on a base raster.

    Args:
        macro_labels (np.ndarray): macroregion label raster (0 = NoData).
        psi_deg_per_id (dict[int, float]): mapping from region id to heading in degrees.
        transform: affine transform for computing extents (None for pixel coords).
        base (np.ndarray): background raster (e.g., slope in degrees).
        title (str): figure title.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.
        arrow_scale (float): quiver scale (larger → shorter arrows).

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_region_headings
        >>> plot_region_headings(np.ones((5, 5), dtype=int), {1: 45.0}, transform=None, base=np.zeros((5, 5)), show=False)
    """
    extent = _extent_from_transform(transform, macro_labels.shape)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    # Base raster (e.g., slope deg)
    base_kwargs: dict[str, Any] = {
        "origin": "upper",
        "cmap": "terrain",
        "interpolation": "nearest",
    }
    if extent is not None:
        base_kwargs["extent"] = extent
    ax.imshow(base, **base_kwargs)

    # Draw macroregion boundaries
    L = macro_labels
    b = np.zeros(L.shape, dtype=bool)
    b[1:, :]   |= (L[1:, :] != L[:-1, :])
    b[:-1, :]  |= (L[1:, :] != L[:-1, :])
    b[:, 1:]   |= (L[:, 1:] != L[:, :-1])
    b[:, :-1]  |= (L[:, 1:] != L[:, :-1])
    boundary_kwargs = {
        "origin": "upper",
        "cmap": "gray",
        "interpolation": "nearest",
        "alpha": 0.9,
    }
    if extent is not None:
        boundary_kwargs["extent"] = extent
    ax.imshow(np.where(b, 1.0, np.nan), **boundary_kwargs)

    # One arrow per region (at centroid)
    ids = np.unique(L[L > 0])
    Xc, Yc, U, V = [], [], [], []
    H, W = L.shape
    for rid in ids:
        m = (L == rid)
        # centroid in pixel coords
        ys, xs = np.where(m)
        if xs.size == 0:
            continue
        x0 = xs.mean()
        y0 = ys.mean()
        # map to world coords if we have extent
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
            x = xmin + (x0 + 0.5) * (xmax - xmin) / W
            y = ymin + (y0 + 0.5) * (ymax - ymin) / H
        else:
            x, y = x0, y0

        psi = np.deg2rad(psi_deg_per_id.get(int(rid), 0.0))
        Xc.append(x); Yc.append(y)
        U.append(np.cos(psi)); V.append(np.sin(psi))

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.quiver(Xc, Yc, U, V, angles="xy", scale_units="xy", scale=arrow_scale, pivot="mid", width=0.002)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    if save:
        fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_dem_with_policy_overlay(
    dem: np.ndarray,
    policy_map: np.ndarray,
    transform,
    *,
    title: str = "DEM + Policy overlay",
    base_cmap: str = "terrain",
    hillshade: bool = False,        # set True if you like shaded relief
    policy_alpha: float = 0.45,     # transparency of the overlay
    use_rgba_overlay: bool = True,  # robust path that builds RGBA image explicitly
    save=None,
    show=False
):
    """Overlay a policy-class map on top of a DEM.

    Args:
        dem (np.ndarray): DEM raster.
        policy_map (np.ndarray): policy class raster.
        transform: affine transform for computing extents.
        title (str): figure title.
        base_cmap (str): colormap for the DEM base layer.
        hillshade (bool): whether to use hillshade for the base layer.
        policy_alpha (float): transparency for policy overlay.
        use_rgba_overlay (bool): whether to build an explicit RGBA overlay.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_dem_with_policy_overlay
        >>> plot_dem_with_policy_overlay(np.random.rand(10, 10), np.zeros((10, 10), dtype=int), transform=None, show=False)
    """
    def _fill_labels_nn(labels: np.ndarray) -> np.ndarray:
        L = labels.copy()
        mask = (L > 0)
        if not np.any(mask):
            return L
        idx = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        if idx is None or len(idx) < 2:
            return L
        r_idx = np.asarray(idx[0], dtype=int)
        c_idx = np.asarray(idx[1], dtype=int)
        L[~mask] = L[r_idx[~mask], c_idx[~mask]]
        return L

    policy_map = _fill_labels_nn(policy_map)
    # Use the module-level helper (typed) and only pass extent if present
    extent = _extent_from_transform(transform, dem.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}
    support = np.isfinite(dem)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # ---- base 2.5D map ----
    if hillshade:
        from matplotlib.colors import LightSource
        dem_filled = np.where(support, dem, np.nanmin(dem))
        ls = LightSource(azdeg=315, altdeg=45)
        base_rgb = ls.shade(dem_filled, cmap=_mpl_cm.get_cmap(base_cmap),
                            vert_exag=1.0, blend_mode="overlay")
        ax.imshow(base_rgb, origin="upper", interpolation="nearest", zorder=0, **extent_kw)
    else:
        im = ax.imshow(dem, origin="upper", cmap=base_cmap,
                       interpolation="nearest", zorder=0, **extent_kw)
        # removed colorbar per request

    # ---- policy overlay (two robust options) ----
    class_colors_list: list[tuple[float, float, float, float]] = [
        (0.0, 0.0, 0.0, 0.0),     # 0 -> fully transparent
        (0.85, 0.85, 0.85, 1.0),  # 1 Flat
        (0.65, 0.85, 0.65, 1.0),  # 2 Gentle
        (1.00, 0.70, 0.20, 1.0),  # 3 Steep/Scarp
        (0.55, 0.45, 0.85, 1.0),  # 4 Ridge/Valley
    ]
    class_colors = np.array(class_colors_list, dtype=float)

    if use_rgba_overlay:
        L = np.clip(policy_map.astype(int), 0, 4)
        rgba = class_colors[L]
        rgba[..., 3] *= policy_alpha            # global transparency
        rgba[~support] = [0, 0, 0, 0]          # mask outside DEM support
        ax.imshow(rgba, origin="upper", interpolation="nearest", zorder=2, **extent_kw)
    else:
        cmap = ListedColormap(class_colors_list)
        n_levels = getattr(cmap, "N", len(class_colors_list))
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], n_levels)
        overlay = np.ma.masked_where((policy_map == 0) | (~support), policy_map)
        ax.imshow(overlay, origin="upper", cmap=cmap, norm=norm,
                  interpolation="nearest", alpha=policy_alpha, zorder=2, **extent_kw)

    # Thin boundaries for readability (optional)
    L = policy_map.astype(np.int32, copy=False)
    b = np.zeros(L.shape, dtype=bool)
    b[1:, :]   |= (L[1:, :] != L[:-1, :])
    b[:-1, :]  |= (L[1:, :] != L[:-1, :])
    b[:, 1:]   |= (L[:, 1:] != L[:, :-1])
    b[:, :-1]  |= (L[:, 1:] != L[:, :-1])
    _draw_double_boundary(ax, np.where(b & support, 1.0, 0.0), extent if extent_kw else None,
                          outer_color="black", inner_color="white", outer_lw=1.2, inner_lw=0.5, alpha=0.9)

    # Legend (skip NoData)
    labels = ["NoData", "Flat", "Gentle", "Steep/Scarp", "Ridge/Valley"]
    legend_elems = [Patch(facecolor=class_colors[i], edgecolor='k', label=labels[i], linewidth=0.5, alpha=policy_alpha)
                    for i in range(1, 5)]
    ax.legend(handles=legend_elems, loc="lower right", frameon=True)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    if save:
        fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def make_quicklook_figures(
    dem: np.ndarray,
    count: np.ndarray,
    features: dict[str, np.ndarray] | None = None,
    labels: np.ndarray | None = None,
    transform: Any | None = None,
    outdir: Path | str = "quicklooks",
    cell: float | None = None,
    show: bool = False
) -> dict[str, Path]:
    """Generate quicklook PNGs for DEM, point density, features, and labels.

    Args:
        dem (np.ndarray): elevation raster used as the base layer.
        count (np.ndarray): per-cell point count raster for density plots.
        features (dict[str, np.ndarray] | None): optional feature rasters such as slope, VRM, or BPI.
        labels (np.ndarray | None): label raster to overlay on top of a base layer.
        transform (Any | None): affine transform for computing spatial extents.
        outdir (Path | str): output directory where PNGs are written.
        cell (float | None): grid cell size used for labelling/metadata (currently informational).
        show (bool): display the generated Matplotlib figures interactively.

    Returns:
        dict[str, Path]: mapping between quicklook identifier and saved PNG path.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import make_quicklook_figures
        >>> dem = np.random.rand(5, 5)
        >>> count = np.ones((5, 5))
        >>> outputs = make_quicklook_figures(dem, count, outdir="quicklooks", show=False)
    """

    assert plt is not None
    out: dict[str, Path] = {}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # DEM quicklook
    dem_path = outdir / "1_dem.png"
    plot_dem(dem, transform=transform, title="DEM", save=dem_path, show=False, colorbar=False)
    out["dem"] = dem_path

    # Count quicklook
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    extent = _extent_from_transform(transform, count.shape)
    im = _masked_imshow(ax, count.astype(float), extent=extent, cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Points per cell")
    ax.set_title("Point count")
    ax.set_aspect("equal", adjustable="box")
    count_path = outdir / "2_count.png"
    fig.savefig(str(count_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    out["count"] = count_path

    # Individual feature quicklooks
    if features is not None:
        if "slope" in features:
            slope_path = outdir / "3_slope_deg.png"
            plot_feature_map(features["slope"], "slope", transform=transform, save=slope_path, show=False)
            out["slope"] = slope_path
        if "vrm" in features:
            vrm_path = outdir / "4_vrm.png"
            plot_feature_map(features["vrm"], "vrm", transform=transform, save=vrm_path, show=False)
            out["vrm"] = vrm_path
        if "bpi" in features:
            bpi_path = outdir / "5_bpi.png"
            plot_feature_map(features["bpi"], "bpi", transform=transform, save=bpi_path, show=False)
            out["bpi"] = bpi_path

    # Labels overlay quicklook
    if labels is not None:
        base = None
        base_name = "Base"
        if features is not None and "slope" in features:
            base = features["slope"]
            base_name = "slope"
        else:
            base = dem
            base_name = "dem"
        labels_path = outdir / "6_labels_overlay.png"
        plot_labels(labels, transform=transform, base=base, base_name=base_name, save=labels_path, show=False)
        out["labels_overlay"] = labels_path

    # Mosaic quicklook (2x3 grid)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    extent_dem = _extent_from_transform(transform, dem.shape)
    _masked_imshow(axs[0, 0], dem, extent=extent_dem, cmap="terrain")
    axs[0, 0].set_title("DEM")
    _masked_imshow(axs[0, 1], count.astype(float), extent=extent_dem, cmap="magma")
    axs[0, 1].set_title("Count")

    if features is not None and "slope" in features:
        _masked_imshow(axs[0, 2], np.degrees(features["slope"]), extent=extent_dem, cmap="viridis")
        axs[0, 2].set_title("Slope (deg)")
    else:
        axs[0, 2].axis("off")

    if features is not None and "vrm" in features:
        _masked_imshow(axs[1, 0], features["vrm"], extent=extent_dem, cmap="viridis")
        axs[1, 0].set_title("VRM")
    else:
        axs[1, 0].axis("off")

    if features is not None and "bpi" in features:
        vmax = np.nanpercentile(np.abs(features["bpi"]), 98) if np.isfinite(features["bpi"]).any() else None
        _masked_imshow(axs[1, 1], features["bpi"], extent=extent_dem, cmap="coolwarm", vmin=-vmax if vmax else None, vmax=vmax)
        axs[1, 1].set_title("BPI")
    else:
        axs[1, 1].axis("off")

    if labels is not None:
        rgba = _label_colorize(labels, bg_label=0, seed=0)
        im_kwargs: dict[str, Any] = {"interpolation": "nearest", "alpha": 0.9}
        if extent_dem is not None:
            im_kwargs["extent"] = tuple(float(v) for v in extent_dem)
        axs[1, 2].imshow(rgba, **im_kwargs)
        axs[1, 2].set_title("Labels")
    else:
        axs[1, 2].axis("off")

    for ax in axs.flat:
        if ax.has_data():
            ax.set_aspect("equal", adjustable="box")

    mosaic_path = outdir / "7_quicklook_mosaic.png"
    fig.savefig(str(mosaic_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    out["mosaic"] = mosaic_path

    return out

def plot_tracks_overlay(dem: np.ndarray,
                        transform,
                        tracks: list[list[tuple[float,float]]],
                        *,
                        title: str = "Planned tracks",
                        color: str = "k",
                        linewidth: float = 1.0,
                        save=None,
                        show=False):
    """Overlay survey tracks on a DEM.

    Args:
        dem (np.ndarray): DEM raster.
        transform: affine transform for computing extents.
        tracks (list[list[tuple[float, float]]]): list of track polylines.
        title (str): figure title.
        color (str): track color.
        linewidth (float): track line width.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_tracks_overlay
        >>> plot_tracks_overlay(np.random.rand(5, 5), transform=None, tracks=[[(0, 0), (1, 1)]], show=False)
    """
    extent = _extent_from_transform(transform, dem.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(dem, origin="upper", cmap="terrain", interpolation="nearest", **extent_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5); cb.set_label("Elevation (arb. units)")
    # Lock axes to raster extent; keep world Y orientation consistent
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    # draw tracks
    for seg in tracks:
        if len(seg) < 2:
            continue
        xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
        ax.plot(xs, ys, color=color, linewidth=linewidth)
    ax.set_title(title); ax.set_aspect("equal", adjustable="box")
    if save: fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def plot_tracks_overlay_by_region(
    dem: np.ndarray,
    transform,
    tracks_by_region: dict[int, list[list[tuple[float, float]]]],
    *,
    title: str = "Planned tracks",
    linewidth: float = 1.0,
    save=None,
    show=False,
) -> None:
    """Plot serpentine tracks with a distinct color per region (one continuous line each).

    Args:
        dem (np.ndarray): DEM raster.
        transform: affine transform for computing extents.
        tracks_by_region (dict[int, list[list[tuple[float, float]]]]): mapping from region id to tracks.
        title (str): figure title.
        linewidth (float): track line width.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_tracks_overlay_by_region
        >>> tracks = {1: [[(0, 0), (1, 1)]]}
        >>> plot_tracks_overlay_by_region(np.random.rand(5, 5), transform=None, tracks_by_region=tracks, show=False)
    """
    extent = _extent_from_transform(transform, dem.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(dem, origin="upper", cmap="terrain", interpolation="nearest", **extent_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Elevation (arb. units)")
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # simple color cycle (robustly extract RGBA tuples)
    colors = None
    if _mpl_cm is not None:
        cmap_obj = _mpl_cm.get_cmap("tab20")
        if hasattr(cmap_obj, "colors"):
            colors = list(getattr(cmap_obj, "colors"))
        else:
            colors = [cmap_obj(i / max(1, getattr(cmap_obj, "N", 20))) for i in range(getattr(cmap_obj, "N", 20))]
    ids = sorted(tracks_by_region.keys())
    for j, rid in enumerate(ids):
        seg_list = tracks_by_region[rid]
        color = colors[j % len(colors)] if colors else "k"
        for seg in seg_list:
            if len(seg) < 2:
                continue
            xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
            ax.plot(xs, ys, color=color, linewidth=linewidth)
    ax.set_title(title); ax.set_aspect("equal", adjustable="box")
    if save: fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def plot_macro_tracks_overlay(
    dem: np.ndarray,
    macro_labels: np.ndarray,
    transform,
    tracks_by_region: dict[int, list[list[tuple[float, float]]]],
    *,
    title: str = "Macroregions + Tracks",
    linewidth: float = 1.0,
    save=None,
    show=False,
) -> None:
    """Plot DEM with macroregion boundaries and region-colored tracks.

    Args:
        dem (np.ndarray): DEM raster.
        macro_labels (np.ndarray): macroregion labels aligned to ``dem``.
        transform: affine transform for computing extents.
        tracks_by_region (dict[int, list[list[tuple[float, float]]]]): mapping from region id to tracks.
        title (str): figure title.
        linewidth (float): track line width.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_macro_tracks_overlay
        >>> tracks = {1: [[(0, 0), (1, 1)]]}
        >>> plot_macro_tracks_overlay(np.random.rand(5, 5), np.ones((5, 5), dtype=int), transform=None, tracks_by_region=tracks, show=False)
    """
    extent = _extent_from_transform(transform, dem.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(dem, origin="upper", cmap="terrain", interpolation="nearest", **extent_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.5); cb.set_label("Elevation [m]")
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Macro boundaries
    L = macro_labels.astype(np.int32, copy=False)
    b = np.zeros(L.shape, dtype=bool)
    b[1:, :]   |= (L[1:, :] != L[:-1, :])
    b[:-1, :]  |= (L[1:, :] != L[:-1, :])
    b[:, 1:]   |= (L[:, 1:] != L[:, :-1])
    b[:, :-1]  |= (L[:, 1:] != L[:, :-1])
    ax.imshow(np.where(b, 1.0, np.nan), origin="upper", cmap="gray", interpolation="nearest",
              alpha=0.9, zorder=1, **extent_kw)

    # Tracks
    colors = None
    if _mpl_cm is not None:
        cmap_obj = _mpl_cm.get_cmap("tab20")
        if hasattr(cmap_obj, "colors"):
            colors = list(getattr(cmap_obj, "colors"))
        else:
            colors = [cmap_obj(i / max(1, getattr(cmap_obj, "N", 20))) for i in range(getattr(cmap_obj, "N", 20))]
    ids = sorted(tracks_by_region.keys())
    for j, rid in enumerate(ids):
        seg_list = tracks_by_region[rid]
        color = colors[j % len(colors)] if colors else "k"
        for seg in seg_list:
            if len(seg) < 2:
                continue
            xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
            ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=2)

    ax.set_title(title); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    if save: fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def plot_dem_with_polygon(
    dem: np.ndarray,
    transform,
    polygon_xy: np.ndarray,
    *,
    tracks: dict[int, list[list[tuple[float, float]]]] | None = None,
    macro_labels: np.ndarray | None = None,
    macro_transform=None,
    title: str = "DEM + AOI",
    save=None,
    show=False,
) -> None:
    """Plot DEM with an outline of the polygon AOI and optional tracks.

    Args:
        dem (np.ndarray): DEM raster.
        transform: affine transform for computing extents.
        polygon_xy (np.ndarray): polygon vertices in world coordinates.
        tracks (dict[int, list[list[tuple[float, float]]]] | None): optional tracks by region.
        macro_labels (np.ndarray | None): optional macroregion labels.
        macro_transform: optional transform for macro labels.
        title (str): figure title.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_dem_with_polygon
        >>> plot_dem_with_polygon(np.random.rand(5, 5), transform=None, polygon_xy=np.array([[0,0],[1,0],[1,1],[0,1]]), show=False)
    """
    extent = _extent_from_transform(transform, dem.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(dem, origin="upper", cmap="terrain", interpolation="nearest", **extent_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5); cb.set_label("Elevation (arb. units)")

    # polygon outline
    poly = np.asarray(polygon_xy, dtype=float)
    if poly.shape[0] >= 2:
        ax.plot(poly[:, 0], poly[:, 1], color="magenta", linewidth=2.0, alpha=0.9, zorder=2)
        ax.plot([poly[-1, 0], poly[0, 0]], [poly[-1, 1], poly[0, 1]], color="magenta", linewidth=2.0, alpha=0.9, zorder=2)

    # optional tracks
    if tracks:
        colors = None
        if _mpl_cm is not None:
            cmap_obj = _mpl_cm.get_cmap("tab20")
            if hasattr(cmap_obj, "colors"):
                colors = list(getattr(cmap_obj, "colors"))
            else:
                colors = [cmap_obj(i / max(1, getattr(cmap_obj, "N", 20))) for i in range(getattr(cmap_obj, "N", 20))]
        for j, (rid, seg_list) in enumerate(sorted(tracks.items())):
            color = colors[j % len(colors)] if colors else "k"
            for seg in seg_list:
                if len(seg) < 2:
                    continue
                xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
                ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.9, zorder=3)

    # optional macro boundaries (cropped grid)
    if macro_labels is not None:
        ext_macro = _extent_from_transform(macro_transform, macro_labels.shape)
        ext_kw_macro: dict[str, Any] = {} if ext_macro is None else {"extent": ext_macro}
        L = macro_labels.astype(np.int32, copy=False)
        b = np.zeros(L.shape, dtype=bool)
        b[1:, :]   |= (L[1:, :] != L[:-1, :])
        b[:-1, :]  |= (L[1:, :] != L[:-1, :])
        b[:, 1:]   |= (L[:, 1:] != L[:, :-1])
        b[:, :-1]  |= (L[:, 1:] != L[:, :-1])
        ax.imshow(np.where(b, 1.0, np.nan), origin="upper", cmap="gray", interpolation="nearest",
                  alpha=0.8, zorder=2.5, **ext_kw_macro)

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_title(title); ax.set_aspect("equal", adjustable="box")
    if save: fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

# pixel-center world mapping
def _pix_to_world(r: float, c: float, transform) -> tuple[float, float]:
    """Map pixel coords (row r, col c) to world (x,y) at *pixel centers* using a rasterio Affine."""
    if transform is None:
        return float(c), float(r)
    a, b, c0, d, e, f0 = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
    # Pixel-center convention (+0.5)
    x = c0 + a * (c + 0.5) + b * (r + 0.5)
    y = f0 + d * (c + 0.5) + e * (r + 0.5)
    return float(x), float(y)

def _row_get(r: dict, key: str, *alts: str, default=np.nan):
    for k in (key,)+alts:
        v = r.get(k, None)
        if v is not None:
            return v
    return default

# [GPT-5 Pro patch] robust primitive overlay (prefers prim_* fields; pixel-center mapping)
def plot_primitives_overlay(
    dem: np.ndarray,
    transform,
    rows: list[dict],
    macro_labels: np.ndarray,
    *,
    title: str = "DEM + Primitives overlay",
    base_cmap: str = "terrain",
    save=None,
    show=False,
    macro_transform=None,
):
    """Draw detected primitives on top of the DEM.

    Rows may contain either the new fields (``prim_cx_px``, ``prim_cy_px``, ``prim_r_m``)
    or legacy ones (``cx_px``, ``cy_px``, ``r_m``). New names are preferred.

    Args:
        dem (np.ndarray): DEM raster.
        transform: affine transform for computing extents.
        rows (list[dict]): region rows containing primitive metadata.
        macro_labels (np.ndarray): macroregion labels aligned to ``dem``.
        title (str): figure title.
        base_cmap (str): colormap for the DEM base layer.
        save: optional output path for saving the figure.
        show (bool): show the figure interactively when True.
        macro_transform: optional transform for macro labels.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_primitives_overlay
        >>> plot_primitives_overlay(np.random.rand(5, 5), transform=None, rows=[], macro_labels=np.zeros((5, 5), dtype=int), show=False)
    """
    extent = _extent_from_transform(transform, dem.shape)
    extent_kw: dict[str, Any] = {} if extent is None else {"extent": extent}
    support = np.isfinite(dem)
    extent_macro = _extent_from_transform(macro_transform, macro_labels.shape) if macro_transform is not None else extent
    extent_kw_macro: dict[str, Any] = {} if extent_macro is None else {"extent": extent_macro}

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Base DEM
    im = ax.imshow(dem, origin="upper", cmap=base_cmap,
                   interpolation="nearest", zorder=0, **extent_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.9)
    cb.set_label("Elevation [m]")

    # Macro boundaries
    L = macro_labels.astype(np.int32, copy=False)
    b = np.zeros(L.shape, dtype=bool)
    b[1:, :]   |= (L[1:, :] != L[:-1, :])
    b[:-1, :]  |= (L[1:, :] != L[:-1, :])
    b[:, 1:]   |= (L[:, 1:] != L[:, :-1])
    b[:, :-1]  |= (L[:, 1:] != L[:, :-1])
    _draw_double_boundary(ax, np.where(b & support, 1.0, 0.0), extent if extent_kw else None,
                          outer_color="black", inner_color="white", outer_lw=1.2, inner_lw=0.5, alpha=0.9)
    # If macro grid has its own transform, replot boundaries with that extent to avoid shifts
    if extent_macro is not None and extent_macro != extent:
        _draw_double_boundary(ax, np.where(b, 1.0, 0.0), extent_macro, outer_color="black", inner_color="white",
                              outer_lw=1.0, inner_lw=0.5, alpha=0.6)

    # Primitives
    for r in rows:
        prim = str(r.get("primitive", "") or "").lower()
        if prim != "spiral" and not r.get("primitive_list"):
            continue

        # Prefer explicit list (supports multiple per region); fallback to single entry
        entries = r.get("primitive_list")
        if not entries:
            bnd_list = r.get("boundary_rc_list") or [r.get("boundary_rc", [])]
            foot_list = r.get("footprint_masks") or []
            # Align footprints with boundaries; fallback to empty list if lengths differ
            if len(foot_list) < len(bnd_list):
                foot_list = list(foot_list) + [None] * (len(bnd_list) - len(foot_list))
            entries = []
            for bnd, foot in zip(bnd_list, foot_list):
                entries.append({
                    "cx_px": float(r.get("prim_cx_px", r.get("cx_px", np.nan))),
                    "cy_px": float(r.get("prim_cy_px", r.get("cy_px", np.nan))),
                    "r_m":   float(r.get("prim_r_m",  r.get("r_m",  np.nan))),
                    "seed_r": int(r.get("seed_r", -1)),
                    "seed_c": int(r.get("seed_c", -1)),
                    "boundary_rc": list(bnd or []),
                    "footprint_rc": list(foot or []),
                    "width_m": float(r.get("width_m", np.nan) if r.get("width_m", None) is not None else np.nan),
                    "length_m": float(r.get("length_m", np.nan) if r.get("length_m", None) is not None else np.nan),
                    "shape": str(r.get("primitive_shape", "")),
                })

        for j, entry in enumerate(entries, start=1):
            seed_r = int(entry.get("seed_r", -1))
            seed_c = int(entry.get("seed_c", -1))
            boundary_rc = entry.get("boundary_rc", [])
            width_m = float(entry.get("width_m", np.nan))
            length_m = float(entry.get("length_m", np.nan))
            shape = str(entry.get("shape", "")).lower()
            # draw footprint boundary if present
            if boundary_rc:
                pts = np.array(boundary_rc, dtype=float)
                by, bx = pts[:, 0], pts[:, 1]  # rows, cols
                xw = []; yw = []
                for rr, cc in zip(by, bx):
                    x, y = _pix_to_world(rr, cc, transform)
                    xw.append(x); yw.append(y)
                ax.plot(xw, yw, color="crimson", lw=1.5, zorder=3)
            # draw footprint fill if we have the mask pixels
            footprint_rc = entry.get("footprint_rc", [])
            if footprint_rc:
                pts = np.array(footprint_rc, dtype=float)
                my, mx = pts[:, 0], pts[:, 1]
                xw = []; yw = []
                for rr, cc in zip(my, mx):
                    x, y = _pix_to_world(rr, cc, transform)
                    xw.append(x); yw.append(y)
                ax.scatter(xw, yw, s=6, color="crimson", alpha=0.25, zorder=2)
                # ellipse approximation from footprint
                pts_centered = np.column_stack([mx, my]).astype(float)
                pts_centered -= pts_centered.mean(axis=0, keepdims=True)
                cov = np.cov(pts_centered, rowvar=False)
                try:
                    vals, vecs = np.linalg.eigh(cov)
                    vals = np.maximum(vals, 0.0)
                    axes = 2.0 * np.sqrt(vals)
                    idx_max = int(np.argmax(axes))
                    idx_min = 1 - idx_max
                    vx, vy = vecs[:, idx_max]
                    theta = np.arctan2(vy, vx)
                    cx_ell = np.mean(xw); cy_ell = np.mean(yw)
                    pix_scale = abs(transform.a) if hasattr(transform, "a") else 1.0
                    half0 = axes[idx_max] * 0.5 * pix_scale
                    half1 = axes[idx_min] * 0.5 * pix_scale
                    t = np.linspace(0, 2*np.pi, 200)
                    cos_t = np.cos(t); sin_t = np.sin(t)
                    cos_a = np.cos(theta); sin_a = np.sin(theta)
                    ex = cx_ell + half0 * cos_t * cos_a - half1 * sin_t * sin_a
                    ey = cy_ell + half0 * cos_t * sin_a + half1 * sin_t * cos_a
                    ax.plot(ex, ey, color="crimson", lw=1.2, ls="--", alpha=0.8, zorder=3)
                except Exception:
                    pass
            # draw seed point (peak)
            if seed_r >= 0 and seed_c >= 0:
                xs, ys = _pix_to_world(float(seed_r), float(seed_c), transform)
                ax.plot([xs], [ys], marker="+", mew=1.5, ms=8, color="black", zorder=4)
                # optional ellipse footprint for cylindrical/conical shapes
                if shape and shape in {"cylinder", "elliptic_cyl", "cone"}:
                    # derive ellipse center/axes from boundary (preferred)
                    cx_ell, cy_ell = xs, ys
                    w_ell, l_ell = width_m, length_m
                    angle_deg = 0.0
                    if boundary_rc:
                        pts_world = []
                        for (rr, cc) in boundary_rc:
                            xw, yw = _pix_to_world(float(rr), float(cc), transform)
                            pts_world.append((xw, yw))
                        if pts_world:
                            pw = np.array(pts_world, dtype=float)
                            mean_xy = pw.mean(axis=0)
                            # PCA for orientation
                            pw_centered = pw - mean_xy
                            cov = np.cov(pw_centered.T)
                            vals, vecs = np.linalg.eigh(cov)
                            order = np.argsort(vals)[::-1]
                            vecs = vecs[:, order]
                            vals = vals[order]
                            u0 = vecs[:, 0] / (np.linalg.norm(vecs[:, 0]) + 1e-9)
                            u1 = vecs[:, 1] / (np.linalg.norm(vecs[:, 1]) + 1e-9)
                            proj0 = pw_centered @ u0
                            proj1 = pw_centered @ u1
                            half0 = float(np.max(np.abs(proj0)))
                            half1 = float(np.max(np.abs(proj1)))
                            # small margin so boundary lies inside
                            margin = 1.1
                            w_ell = 2.0 * half0 * margin
                            l_ell = 2.0 * half1 * margin
                            cx_ell, cy_ell = mean_xy
                            angle_deg = float(np.degrees(np.arctan2(u0[1], u0[0])))
                    if np.isfinite(w_ell) and np.isfinite(l_ell) and w_ell > 0 and l_ell > 0:
                        ell = Ellipse((cx_ell, cy_ell), width=w_ell, height=l_ell, angle=angle_deg,
                                      facecolor="none", edgecolor="crimson", linewidth=1.5, zorder=3)
                        ax.add_patch(ell)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if save:
        fig.savefig(str(save), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_knoll_diagnostics(
    dem: np.ndarray,
    transform: Any,
    feats: dict[str, np.ndarray],
    macro_labels: np.ndarray | None,
    *,
    macro_transform=None,
    save: Path | str | None = None,
    show: bool = False,
) -> Figure:
    """Diagnostic plot for DEM-based knoll detection.

    Shows DEM background, macroregion boundaries, and seed points at different
    filtering stages (when available).

    Args:
        dem (np.ndarray): DEM raster.
        transform (Any): affine transform for computing extents.
        feats (dict[str, np.ndarray]): feature rasters used by knoll detection.
        macro_labels (np.ndarray | None): macroregion labels aligned to ``dem``.
        macro_transform: optional transform for macro labels.
        save (Path | str | None): optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        Figure: Matplotlib figure containing the diagnostic plot.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_knoll_diagnostics
        >>> feats = {"slope": np.zeros((5, 5))}
        >>> fig = plot_knoll_diagnostics(np.random.rand(5, 5), transform=None, feats=feats, macro_labels=None, show=False)
    """
    debug = getattr(_prim, "DEBUG_MOUND_SEEDS", None)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    extent = _extent_from_transform(transform, dem.shape)
    extent_macro = _extent_from_transform(macro_transform, macro_labels.shape) if (macro_transform is not None and macro_labels is not None) else extent

    # DEM background
    im = _masked_imshow(ax, dem, extent=extent, cmap="terrain")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Elevation (arb. units)")

    # Macro boundaries
    if macro_labels is not None and macro_labels.size:
        try:
            bnd = _label_boundaries(macro_labels.astype(int))
            ext_bnd = extent_macro
            if ext_bnd is None:
                ax.contour(
                    bnd.astype(float),
                    levels=[0.5],
                    colors="k",
                    linewidths=0.7,
                )
            else:
                xmin, xmax, ymin, ymax = ext_bnd
                xs = np.linspace(xmin, xmax, macro_labels.shape[1])
                ys = np.linspace(ymin, ymax, macro_labels.shape[0])
                ax.contour(xs, ys, bnd.astype(float),
                           levels=[0.5], colors="k", linewidths=0.7)
        except Exception:
            pass  # diagnostics should never crash the run

    # Overlay seeds if available
    if debug is not None:
        def rc_to_xy(rc_list: list[tuple[int, int]]):
            if not rc_list:
                return np.asarray([]), np.asarray([])
            rr = np.asarray([r for (r, c) in rc_list], dtype=float)
            cc = np.asarray([c for (r, c) in rc_list], dtype=float)
            tf = macro_transform if macro_transform is not None else transform
            if tf is None:
                return cc, rr
            xs = []; ys = []
            for r, c in zip(rr, cc):
                xw, yw = _pix_to_world(r, c, tf)
                xs.append(xw); ys.append(yw)
            return np.asarray(xs), np.asarray(ys)

        seeds_all   = list(debug.get("seeds_all", []))
        seeds_macro = list(debug.get("seeds_macro", []))
        seeds_kept  = list(debug.get("seeds_kept", []))

        xa, ya = rc_to_xy(seeds_all)
        xm, ym = rc_to_xy(seeds_macro)
        xk, yk = rc_to_xy(seeds_kept)

        if xa.size:
            ax.scatter(xa, ya, s=18, marker=MarkerStyle("+"), color="cyan",   label="all local maxima")
        if xm.size:
            ax.scatter(xm, ym, s=24, marker=MarkerStyle("x"), color="orange", label="after macro filter")
        if xk.size:
            ax.scatter(
                xk, yk,
                s=40,
                marker=MarkerStyle("o"),
                facecolors="none",
                edgecolors="magenta",
                linewidths=1.2,
                label="after prominence",
            )

        if xa.size or xm.size or xk.size:
            ax.legend(loc="lower right", fontsize=8)

    ax.set_title("Knoll seeds diagnostics")
    ax.set_xlabel("East [m]" if extent is not None else "column index")
    ax.set_ylabel("North [m]" if extent is not None else "row index")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    fig.tight_layout()

    if save is not None:
        fig.savefig(os.fspath(save), dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_primitives_3d(
    dem: np.ndarray,
    transform: Any,
    rows: list[dict],
    *,
    title: str = "DEM + Primitive fit (3D)",
    save: Path | str | None = None,
    show: bool = False,
) -> Figure | None:
    """Render DEM as a surface with fitted primitive footprints (ellipse wires).

    Args:
        dem (np.ndarray): DEM raster.
        transform (Any): affine transform for computing world coordinates.
        rows (list[dict]): region rows containing primitive ellipse metadata.
        title (str): figure title.
        save (Path | str | None): optional output path for saving the figure.
        show (bool): show the figure interactively when True.

    Returns:
        Figure | None: Matplotlib figure, or ``None`` if ``transform`` is missing.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_primitives_3d
        >>> fig = plot_primitives_3d(np.random.rand(5, 5), transform=None, rows=[], show=False)
    """
    assert plt is not None
    if transform is None:
        return None

    H, W = dem.shape
    cols = np.arange(W, dtype=float) + 0.5
    rows_idx = np.arange(H, dtype=float) + 0.5
    a, b, c0, d, e, f0 = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
    X = c0 + a * cols[np.newaxis, :] + b * rows_idx[:, np.newaxis]
    Y = f0 + d * cols[np.newaxis, :] + e * rows_idx[:, np.newaxis]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[call-overload]

    Z = np.array(dem, copy=True, dtype=float)
    surf = ax.plot_surface(X, Y, np.where(np.isfinite(Z), Z, np.nan), cmap="terrain", linewidth=0, antialiased=False, alpha=0.7)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.05, label="Elevation")

    def _ellipse_from_boundary(boundary_rc: list[tuple[int, int]]):
        pts_world: list[tuple[float, float]] = []
        for (rr, cc) in boundary_rc:
            xw, yw = _pix_to_world(float(rr), float(cc), transform)
            pts_world.append((xw, yw))
        if not pts_world:
            return None
        pw = np.array(pts_world, dtype=float)
        mean_xy = pw.mean(axis=0)
        pw_centered = pw - mean_xy
        cov = np.cov(pw_centered.T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order]
        vals = vals[order]
        u0 = vecs[:, 0] / (np.linalg.norm(vecs[:, 0]) + 1e-9)
        proj0 = pw_centered @ u0
        proj1 = pw_centered @ vecs[:, 1]
        half0 = float(np.max(np.abs(proj0)))
        half1 = float(np.max(np.abs(proj1)))
        angle_rad = math.atan2(u0[1], u0[0])
        return mean_xy, half0, half1, angle_rad

    for r in rows:
        prim = str(r.get("primitive_shape", "") or r.get("primitive", "") or "").lower()
        if prim not in {"cylinder", "elliptic_cyl", "cone"}:
            continue
        bnd_list = r.get("boundary_rc_list") or [r.get("boundary_rc", [])]
        seed_r = int(r.get("seed_r", -1))
        seed_c = int(r.get("seed_c", -1))
        height_raw = r.get("height_m", np.nan)
        height_m = float(height_raw) if (height_raw is not None) else np.nan
        if seed_r < 0 or seed_c < 0 or not np.isfinite(height_m):
            continue
        z_top = float(dem[seed_r, seed_c]) if (0 <= seed_r < H and 0 <= seed_c < W and np.isfinite(dem[seed_r, seed_c])) else np.nan
        if not np.isfinite(z_top):
            continue
        for boundary_rc in bnd_list:
            boundary_rc = list(boundary_rc or [])
            if not boundary_rc:
                continue
            ell = _ellipse_from_boundary(boundary_rc)
            if ell is None:
                continue
            (cx, cy), half0, half1, angle = ell
            if not (np.isfinite(cx) and np.isfinite(cy) and half0 > 0 and half1 > 0):
                continue
            z_base = z_top - height_m
            t = np.linspace(0, 2 * np.pi, 200)
            cos_t = np.cos(t); sin_t = np.sin(t)
            cos_a = math.cos(angle); sin_a = math.sin(angle)
            ex = cx + half0 * cos_t * cos_a - half1 * sin_t * sin_a
            ey = cy + half0 * cos_t * sin_a + half1 * sin_t * cos_a
            ax.plot(ex, ey, zs=z_base, color="crimson", linewidth=1.5)
            ax.plot(ex, ey, zs=z_top, color="crimson", linewidth=1.5)
            for k in range(0, t.size, max(1, t.size // 12)):
                ax.plot([ex[k], ex[k]], [ey[k], ey[k]], [z_base, z_top], color="crimson", linewidth=0.8, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.view_init(elev=45, azim=-60)
    if save:
        fig.savefig(os.fspath(save), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_primitives_3d_interactive(
    dem: np.ndarray,
    transform: Any,
    rows: list[dict],
    *,
    title: str = "DEM + Primitive fit (3D, interactive)",
    save: Path | str | None = None,
) -> Any:
    """Export an interactive HTML (Plotly) with DEM surface and fitted primitives.

    Args:
        dem (np.ndarray): DEM raster.
        transform (Any): affine transform for computing world coordinates.
        rows (list[dict]): region rows containing primitive metadata.
        title (str): figure title.
        save (Path | str | None): optional output path for saving the HTML.

    Returns:
        Any: Plotly figure (or ``None`` if dependencies are unavailable).

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_primitives_3d_interactive
        >>> fig = plot_primitives_3d_interactive(np.random.rand(5, 5), transform=None, rows=[], save=None)
    """
    if go is None or transform is None:
        return
    H, W = dem.shape
    cols = np.arange(W, dtype=float) + 0.5
    rows_idx = np.arange(H, dtype=float) + 0.5
    a, b, c0, d, e, f0 = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
    X = c0 + a * cols[np.newaxis, :] + b * rows_idx[:, np.newaxis]
    Y = f0 + d * cols[np.newaxis, :] + e * rows_idx[:, np.newaxis]
    Z = np.array(dem, copy=True, dtype=float)

    data = []
    data.append(go.Surface(x=X, y=Y, z=np.where(np.isfinite(Z), Z, np.nan),
                           colorscale="Earth", opacity=0.7, showscale=True, name="DEM"))

    def _ellipse_from_boundary(boundary_rc: list[tuple[int, int]]):
        pts_world: list[tuple[float, float]] = []
        for (rr, cc) in boundary_rc:
            xw, yw = _pix_to_world(float(rr), float(cc), transform)
            pts_world.append((xw, yw))
        if not pts_world:
            return None
        pw = np.array(pts_world, dtype=float)
        mean_xy = pw.mean(axis=0)
        pw_centered = pw - mean_xy
        cov = np.cov(pw_centered.T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order]
        vals = vals[order]
        u0 = vecs[:, 0] / (np.linalg.norm(vecs[:, 0]) + 1e-9)
        proj0 = pw_centered @ u0
        proj1 = pw_centered @ vecs[:, 1]
        half0 = float(np.max(np.abs(proj0)))
        half1 = float(np.max(np.abs(proj1)))
        angle_rad = math.atan2(u0[1], u0[0])
        return mean_xy, half0, half1, angle_rad

    for r in rows:
        prim = str(r.get("primitive_shape", "") or r.get("primitive", "") or "").lower()
        if prim not in {"cylinder", "elliptic_cyl", "cone"}:
            continue
        boundary_rc = list(r.get("boundary_rc", []))
        seed_r = int(r.get("seed_r", -1))
        seed_c = int(r.get("seed_c", -1))
        height_raw = r.get("height_m", np.nan)
        height_m = float(height_raw if height_raw is not None else np.nan)
        if not boundary_rc or not np.isfinite(height_m) or seed_r < 0 or seed_c < 0:
            continue
        ell = _ellipse_from_boundary(boundary_rc)
        if ell is None:
            continue
        (cx, cy), half0, half1, angle = ell
        if not (np.isfinite(cx) and np.isfinite(cy) and half0 > 0 and half1 > 0):
            continue
        z_top = float(dem[seed_r, seed_c]) if (0 <= seed_r < H and 0 <= seed_c < W and np.isfinite(dem[seed_r, seed_c])) else np.nan
        if not np.isfinite(z_top):
            continue
        z_base = z_top - height_m
        t = np.linspace(0, 2 * np.pi, 200)
        cos_t = np.cos(t); sin_t = np.sin(t)
        cos_a = math.cos(angle); sin_a = math.sin(angle)
        ex = cx + half0 * cos_t * cos_a - half1 * sin_t * sin_a
        ey = cy + half0 * cos_t * sin_a + half1 * sin_t * cos_a

        data.append(go.Scatter3d(x=ex, y=ey, z=np.full_like(ex, z_base),
                                 mode="lines", line=dict(color="red", width=3),
                                 name=f"{prim}_base"))
        data.append(go.Scatter3d(x=ex, y=ey, z=np.full_like(ex, z_top),
                                 mode="lines", line=dict(color="red", width=3),
                                 name=f"{prim}_top"))
        for k in range(0, t.size, max(1, t.size // 12)):
            data.append(go.Scatter3d(x=[ex[k], ex[k]], y=[ey[k], ey[k]], z=[z_base, z_top],
                                     mode="lines", line=dict(color="red", width=2),
                                     name=f"{prim}_side"))

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        showlegend=False,
    )
    fig = go.Figure(data=data, layout=layout)
    if save:
        fig.write_html(os.fspath(save), include_plotlyjs="cdn")
    return fig


def plot_tracks_3d_interactive(
    dem: np.ndarray,
    transform: Any,
    tracks_by_region: dict[int, list[list[tuple[float, float]]]],
    *,
    title: str = "DEM + Tracks (3D, interactive)",
    save: Path | str | None = None,
    fancy: bool = False,
) -> Any:
    """Export an interactive 3D HTML with DEM surface and colored tracks.

    Args:
        dem (np.ndarray): DEM raster.
        transform (Any): affine transform for computing world coordinates.
        tracks_by_region (dict[int, list[list[tuple[float, float]]]]): mapping from region id to tracks.
        title (str): figure title.
        save (Path | str | None): optional output path for saving the HTML.
        fancy (bool): enable enhanced styling.

    Returns:
        Any: Plotly figure (or ``None`` if dependencies are unavailable).

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_tracks_3d_interactive
        >>> fig = plot_tracks_3d_interactive(np.random.rand(5, 5), transform=None, tracks_by_region={}, save=None)
    """
    if go is None or transform is None:
        return
    H, W = dem.shape
    cols = np.arange(W, dtype=float) + 0.5
    rows_idx = np.arange(H, dtype=float) + 0.5
    a, b, c0, d, e, f0 = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
    X = c0 + a * cols[np.newaxis, :] + b * rows_idx[:, np.newaxis]
    Y = f0 + d * cols[np.newaxis, :] + e * rows_idx[:, np.newaxis]
    Z = np.array(dem, copy=True, dtype=float)

    def xy_to_rc(x: float, y: float) -> tuple[int, int] | None:
        det = a * e - b * d
        if det == 0.0:
            return None
        col = ((e * (x - c0) - b * (y - f0)) / det) - 0.5
        row = ((-d * (x - c0) + a * (y - f0)) / det) - 0.5
        ci = int(round(col)); ri = int(round(row))
        if 0 <= ri < H and 0 <= ci < W:
            return ri, ci
        return None

    data = []
    surface_kwargs: dict[str, Any] = {
        "x": X,
        "y": Y,
        "z": np.where(np.isfinite(Z), Z, np.nan),
        "colorscale": "Earth",
        "opacity": 0.7,
        "showscale": True,
        "name": "DEM",
    }
    if fancy:
        surface_kwargs.update({
            "lighting": dict(ambient=0.35, diffuse=0.65, specular=0.25, roughness=0.55, fresnel=0.1),
            "lightposition": dict(x=80, y=200, z=300),
        })
    data.append(go.Surface(**surface_kwargs))

    colors = None
    if _mpl_cm is not None:
        cmap_obj = _mpl_cm.get_cmap("tab20")
        if hasattr(cmap_obj, "colors"):
            colors = list(getattr(cmap_obj, "colors"))
        else:
            colors = [cmap_obj(i / max(1, getattr(cmap_obj, "N", 20))) for i in range(getattr(cmap_obj, "N", 20))]
    for j, (rid, seg_list) in enumerate(sorted(tracks_by_region.items())):
        color = colors[j % len(colors)] if colors else "red"
        for seg in seg_list:
            if len(seg) < 2:
                continue
            xs = np.array([p[0] for p in seg], dtype=float)
            ys = np.array([p[1] for p in seg], dtype=float)
            zs = np.full(xs.shape, np.nan, dtype=float)
            for i, (xv, yv) in enumerate(zip(xs, ys)):
                rc = xy_to_rc(float(xv), float(yv))
                if rc is None:
                    continue
                r, c = rc
                z = dem[r, c]
                if np.isfinite(z):
                    zs[i] = float(z)
            data.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=4),
                name=f"Region {rid}"
            ))

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        showlegend=False,
    )
    fig = go.Figure(data=data, layout=layout)
    if save:
        fig.write_html(os.fspath(save), include_plotlyjs="cdn")
    return fig


def plot_macro_labels_3d_interactive(
    dem: np.ndarray,
    macro_labels: np.ndarray,
    transform: Any,
    *,
    title: str = "DEM + Macroregions (3D, interactive)",
    save: Path | str | None = None,
    policy_map: np.ndarray | None = None,
    fancy: bool = False,
) -> Any:
    """Export an interactive 3D HTML with DEM surface colored by macroregion IDs or policy classes.

    Args:
        dem (np.ndarray): DEM raster.
        macro_labels (np.ndarray): macroregion labels aligned to ``dem``.
        transform (Any): affine transform for computing world coordinates.
        title (str): figure title.
        save (Path | str | None): optional output path for saving the HTML.
        policy_map (np.ndarray | None): optional policy classes for coloring.
        fancy (bool): enable enhanced styling.

    Returns:
        Any: Plotly figure (or ``None`` if dependencies are unavailable).

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_macro_labels_3d_interactive
        >>> fig = plot_macro_labels_3d_interactive(np.random.rand(5, 5), np.ones((5, 5), dtype=int), transform=None, save=None)
    """
    if go is None or transform is None:
        return
    H, W = dem.shape
    cols = np.arange(W, dtype=float) + 0.5
    rows_idx = np.arange(H, dtype=float) + 0.5
    a, b, c0, d, e, f0 = float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)
    X = c0 + a * cols[np.newaxis, :] + b * rows_idx[:, np.newaxis]
    Y = f0 + d * cols[np.newaxis, :] + e * rows_idx[:, np.newaxis]

    Z = np.array(dem, copy=True, dtype=float)
    support = np.isfinite(Z)
    Z_masked = np.where(support, Z, np.nan)

    if policy_map is not None:
        L = np.array(policy_map, copy=True, dtype=float)
        L[~support] = np.nan
        base_colors = [
            (0.85, 0.85, 0.85, 1.0),  # Flat
            (0.65, 0.85, 0.65, 1.0),  # Gentle
            (1.00, 0.70, 0.20, 1.0),  # Steep/Scarp
            (0.55, 0.45, 0.85, 1.0),  # Ridge/Valley
        ]
        colorscale_plotly = []
        for cls_val, rgba in zip([1, 2, 3, 4], base_colors):
            col = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.3f})"
            t = (cls_val - 1) / 3.0
            colorscale_plotly.append((max(0.0, t - 1e-6), col))
            colorscale_plotly.append((t, col))
        vmin, vmax = 1.0, 4.0
    else:
        L = np.array(macro_labels, copy=True, dtype=float)
        L[~support] = np.nan
        unique_ids = sorted(int(v) for v in np.unique(L[np.isfinite(L)]) if v > 0)
        if not unique_ids:
            return
        colorscale = []
        cmap = _mpl_cm.get_cmap("tab20")
        for rid in unique_ids:
            rgba = cmap((rid - 1) % getattr(cmap, "N", 20))
            col = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.3f})"
            colorscale.append((rid, col))
        vmin, vmax = float(min(unique_ids)), float(max(unique_ids))
        norm = (lambda v: 0.0) if vmax == vmin else (lambda v: (v - vmin) / (vmax - vmin))
        colorscale_plotly = []
        for val, col in colorscale:
            t = norm(val)
            colorscale_plotly.append((t - 1e-6 if t > 0 else 0.0, col))
            colorscale_plotly.append((t, col))

    surface_kwargs: dict[str, Any] = {
        "x": X,
        "y": Y,
        "z": Z_masked,
        "surfacecolor": L,
        "colorscale": colorscale_plotly,
        "cmin": vmin,
        "cmax": vmax,
        "showscale": False,
        "opacity": 0.9,
        "name": "DEM",
    }
    if fancy:
        surface_kwargs.update({
            "lighting": dict(ambient=0.35, diffuse=0.65, specular=0.25, roughness=0.55, fresnel=0.1),
            "lightposition": dict(x=80, y=200, z=300),
        })
    surface = go.Surface(**surface_kwargs)
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        showlegend=False,
    )
    if save:
        fig.write_html(os.fspath(save), include_plotlyjs="cdn")
    return fig

def plot_knoll_ellipses_3d(
    dem: np.ndarray,
    transform: Any,
    rows: list[dict],
    *,
    title: str = "DEM + Knoll ellipses (3D)",
    save=None,
    alpha_surface: float = 0.8,
):
    """Interactive 3D DEM with cylindrical ellipses for knoll footprints.

    Args:
        dem (np.ndarray): DEM raster.
        transform (Any): affine transform for computing world coordinates.
        rows (list[dict]): region rows containing ellipse metadata.
        title (str): figure title.
        save: optional output path for saving the HTML.
        alpha_surface (float): surface transparency.

    Returns:
        Any: Plotly figure.

    Raises:
        ImportError: If Plotly is unavailable.

    Examples:
        >>> import numpy as np
        >>> from mc_geomap_utils.plotting import plot_knoll_ellipses_3d
        >>> fig = plot_knoll_ellipses_3d(np.random.rand(5, 5), transform=None, rows=[], save=None)
    """
    if go is None:
        raise ImportError("plotly is required for plot_knoll_ellipses_3d")
    H, W = dem.shape
    rr = np.arange(H); cc = np.arange(W)
    RR, CC = np.meshgrid(rr, cc, indexing="ij")
    X = transform.c + transform.a * (CC + 0.5) + transform.b * (RR + 0.5)
    Y = transform.f + transform.d * (CC + 0.5) + transform.e * (RR + 0.5)

    dem_masked = np.where(np.isfinite(dem), dem, np.nan)
    surface = go.Surface(x=X, y=Y, z=dem_masked, colorscale="Viridis",
                         showscale=True, opacity=alpha_surface, name="DEM")
    traces: list[Any] = [surface]

    t = np.linspace(0, 2 * np.pi, 120)
    cos_t = np.cos(t); sin_t = np.sin(t)
    for r in rows:
        prim = str(r.get("primitive", "") or "").lower()
        if prim != "spiral":
            continue
        entries = r.get("primitive_list")
        if not entries:
            bnd_list = r.get("boundary_rc_list") or [r.get("boundary_rc", [])]
            foot_list = r.get("footprint_masks") or []
            seeds_list = r.get("seeds_list") or []
            if len(foot_list) < len(bnd_list):
                foot_list = list(foot_list) + [None] * (len(bnd_list) - len(foot_list))
            entries = []
            for idx, (bnd, foot) in enumerate(zip(bnd_list, foot_list)):
                sr_sc = seeds_list[idx] if idx < len(seeds_list) else (r.get("seed_r", -1), r.get("seed_c", -1))
                entries.append({
                    "cx_px": float(r.get("prim_cx_px", r.get("cx_px", np.nan))),
                    "cy_px": float(r.get("prim_cy_px", r.get("cy_px", np.nan))),
                    "seed_r": int(sr_sc[0]) if sr_sc is not None else int(r.get("seed_r", -1)),
                    "seed_c": int(sr_sc[1]) if sr_sc is not None else int(r.get("seed_c", -1)),
                    "boundary_rc": list(bnd or []),
                    "footprint_rc": list(foot or []),
                    "ellipse_major_m": r.get("ellipse_major_m"),
                    "ellipse_minor_m": r.get("ellipse_minor_m"),
                    "ellipse_theta_deg": r.get("ellipse_theta_deg", 0.0),
                    "height_m": r.get("height_m", None),
                })

        colors_cycle = ["crimson", "magenta", "orange", "dodgerblue"]
        geoms: list[dict[str, Any]] = []
        for entry_idx, entry in enumerate(entries):
            cx_px = float(entry.get("cx_px", np.nan))
            cy_px = float(entry.get("cy_px", np.nan))
            seed_r = int(entry.get("seed_r", -1))
            seed_c = int(entry.get("seed_c", -1))
            a_full = entry.get("ellipse_major_m")
            b_full = entry.get("ellipse_minor_m")
            theta_deg = entry.get("ellipse_theta_deg", 0.0) or 0.0
            pts_rc = entry.get("footprint_rc") or entry.get("boundary_rc") or []

            a = b = None
            cx = cy = np.nan
            # Always prefer footprint geometry if available; compute directly in world coords
            if len(pts_rc) >= 3:
                pts = np.asarray(pts_rc, float)
                xy = np.array([transform * (float(c) + 0.5, float(r) + 0.5) for r, c in pts], float)
                mean_xy = xy.mean(axis=0, keepdims=True)
                xy_centered = xy - mean_xy
                cov = np.cov(xy_centered, rowvar=False)
                vals, vecs = np.linalg.eigh(cov)
                vals = np.maximum(vals, 0.0)
                idx_max = int(np.argmax(vals)); idx_min = 1 - idx_max
                v0 = vecs[:, idx_max]
                proj0 = xy_centered @ v0
                proj1 = xy_centered @ vecs[:, idx_min]
                half0 = np.max(np.abs(proj0))
                half1 = np.max(np.abs(proj1))
                cx, cy = float(mean_xy[0, 0]), float(mean_xy[0, 1])
                a = half0
                b = half1
                theta_deg = float(np.degrees(np.arctan2(v0[1], v0[0])))
            else:
                if a_full is None or b_full is None or not np.isfinite(cx_px) or not np.isfinite(cy_px):
                    continue
                a = float(a_full) * 0.5
                b = float(b_full) * 0.5
                try:
                    cx, cy = transform * (cx_px + 0.5, cy_px + 0.5)
                except Exception:
                    cx = (cx_px + 0.5) * float(transform.a if hasattr(transform, "a") else 1.0)
                    cy = (cy_px + 0.5) * float(transform.e if hasattr(transform, "e") else 1.0)

            if not np.isfinite(cx) or not np.isfinite(cy) or a is None or b is None:
                continue
            z_top = float(dem[seed_r, seed_c]) if (0 <= seed_r < H and 0 <= seed_c < W and np.isfinite(dem[seed_r, seed_c])) else float(np.nanmean(dem))
            height = float(entry.get("height_m", r.get("height_m", 2.0))) if (entry.get("height_m", None) is not None or r.get("height_m", None) is not None) else 2.0
            z_base = z_top - height
            geoms.append({
                "cx": cx, "cy": cy, "a": a, "b": b,
                "theta_deg": theta_deg,
                "z_top": z_top, "z_base": z_base,
            })

        geoms.sort(key=lambda g: g["a"] * g["b"] if g["a"] is not None and g["b"] is not None else 0.0, reverse=True)
        for geom_idx, g in enumerate(geoms):
            color = colors_cycle[geom_idx % len(colors_cycle)]
            cx, cy = g["cx"], g["cy"]
            a, b = g["a"], g["b"]
            theta_deg = g["theta_deg"]
            z_top = g["z_top"]; z_base = g["z_base"]
            th = np.radians(theta_deg)
            cos_a = np.cos(th); sin_a = np.sin(th)
            ex = cx + a * cos_t * cos_a - b * sin_t * sin_a
            ey = cy + a * cos_t * sin_a + b * sin_t * cos_a
            traces.append(go.Scatter3d(x=ex, y=ey, z=np.full_like(ex, z_base),
                                       mode="lines", line=dict(color=color, width=3), name="footprint base",
                                       showlegend=False))
            traces.append(go.Scatter3d(x=ex, y=ey, z=np.full_like(ex, z_top),
                                       mode="lines", line=dict(color=color, width=3, dash="dash"), name="footprint top",
                                       showlegend=False))
            for k in range(0, len(ex), max(1, len(ex)//12)):
                traces.append(go.Scatter3d(x=[ex[k], ex[k]], y=[ey[k], ey[k]], z=[z_base, z_top],
                                           mode="lines", line=dict(color=color, width=2, dash="dot"),
                                           showlegend=False))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="East [m]",
            yaxis_title="North [m]",
            zaxis_title="Elevation [m]",
            aspectmode="data",
        ),
        showlegend=False,
    )
    if save:
        fig.write_html(os.fspath(save), include_plotlyjs="cdn")
    return fig
