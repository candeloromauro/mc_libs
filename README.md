# mc_libs

Personal mc_* utility libraries.

## Dependencies (for all functions)

Install all of the following to ensure every function works (including optional plotting/IO features):

- numpy
- scipy
- matplotlib
- scikit-image
- rasterio
- pyproj
- shapely
- plotly
- open3d
- trimesh
- lcm
- navlib

Note: If `navlib` or `lcm` are not available on PyPI in your environment, install them from your internal sources; they are required by the LCM utilities.

## Install (all deps + editable)

```sh
python3 -m pip install numpy scipy matplotlib scikit-image rasterio pyproj shapely plotly open3d trimesh lcm navlib && \
python3 -m pip install -e /Users/mcandeloro/python_projects/mc_libs
```

## CLI

```sh
mc-libs tree
mc-libs libs
mc-libs funcs mc_data_utils
mc-libs help mc_data_utils.allan2
mc-libs example mc_data_utils.allan2
```
