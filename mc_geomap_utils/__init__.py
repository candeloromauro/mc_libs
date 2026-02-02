"""Reusable utilities for point-cloud processing and visualization."""

from .io import (
    PointFileMode,
    load_points,
    save_points,
    write_geotiff,
    load_grd,
)

from .plotting import (
    plot_mesh,
    plot_point_cloud,
    show_plot,
    plot_dem,
    plot_feature_map,
    plot_labels,
    make_quicklook_figures,
    PlotDataSource,
    plot_policy_map,
    plot_macro_overlay,
    plot_region_headings,
    plot_tracks_overlay,
    plot_macro_tracks_overlay,
    plot_dem_with_polygon,
    plot_primitives_overlay,
    plot_knoll_diagnostics,
    plot_primitives_3d_interactive,
    plot_tracks_3d_interactive,
    plot_macro_labels_3d_interactive,
    plot_knoll_ellipses_3d,
    save_mosaic
)

from .evaluation import (
    array_stats,
    print_array_stats,
    export_feature_stats_table,
    export_label_table
)

from .pointcloud import (
    delaunay_mesh,
    mesh_to_obj,
    grid_to_dem,
    delaunay_to_open3d_mesh,
    dem_to_features,
    feature_to_slic_segmentation
)

from .manipulation import (
    Mode,
    shift_point_cloud,
    grid_to_point_cloud
)

from .macroregions import (
    UF,
    eroded_support_mask,
    classify_policy_cells,
    grad_mag_deg,
    edge_list_with_boundaries,
    superpixel_stats,
    rag_merge_slic,
    axial_mean_heading,
    policy_for_class,
    summarize_regions,
    export_regions_csv,
    absorb_small_islands,
    smooth_macro_labels,
    policy_mode_filter,
    policy_superpixel_consensus
)

from .primitives import (
    augment_regions_with_primitives
)

from .planning import (
    region_mask,
    generate_serpentine_tracks,
    generate_serpentine_tracks_polygon,
    attach_altitude_and_tilt,
    estimate_turn_count,
    polyline_length_m,
    tracks_total_length_m,
    merge_tracks_continuous,
    straighten_tracks,
    smooth_polyline,
    simplify_polyline_rdp
)

# Prefixed aliases for clarity across mc_* libraries.
McPointFileMode = PointFileMode
McPlotDataSource = PlotDataSource
McMode = Mode
McUF = UF

__all__ = [
    "PointFileMode",
    "McPointFileMode",
    "load_points",
    "save_points",
    "load_grd",
    "show_plot",
    "plot_mesh",
    "plot_point_cloud",
    "plot_dem",
    "plot_feature_map",
    "plot_labels",
    "make_quicklook_figures",
    "PlotDataSource",
    "McPlotDataSource",
    "array_stats",
    "print_array_stats",
    "export_feature_stats_table",
    "export_label_table",
    "delaunay_mesh",
    "mesh_to_obj",
    "grid_to_dem",
    "delaunay_to_open3d_mesh",
    "Mode",
    "McMode",
    "shift_point_cloud",
    "grid_to_point_cloud",
    "write_geotiff",
    "dem_to_features",
    "feature_to_slic_segmentation",
    "UF",
    "McUF",
    "eroded_support_mask",
    "classify_policy_cells",
    "grad_mag_deg",
    "edge_list_with_boundaries",
    "superpixel_stats",
    "rag_merge_slic",
    "axial_mean_heading",
    "policy_for_class",
    "summarize_regions",
    "export_regions_csv",
    "absorb_small_islands",
    "smooth_macro_labels",
    "plot_policy_map",
    "plot_macro_overlay",
    "plot_region_headings",
    "policy_superpixel_consensus",
    "policy_mode_filter",
    "plot_primitives_overlay",
    "plot_tracks_overlay",
    "plot_macro_tracks_overlay",
    "plot_dem_with_polygon",
    "plot_primitives_3d_interactive",
    "plot_tracks_3d_interactive",
    "plot_macro_labels_3d_interactive",
    "plot_knoll_ellipses_3d",
    "augment_regions_with_primitives",
    "region_mask",
    "generate_serpentine_tracks",
    "attach_altitude_and_tilt",
    "estimate_turn_count",
    "polyline_length_m",
    "tracks_total_length_m",
    "plot_knoll_diagnostics",
    "straighten_tracks",
    "smooth_polyline",
    "simplify_polyline_rdp"
]
