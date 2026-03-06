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
mc-libs tree [--json] || mc-libs list-all [--json]                                  # displays all libraries, modules, and functions in the main folder
mc-libs libs [--json] || mc-libs list-libraries [--json]                            # displays all libraries in the main folder
mc-libs funcs [library] [--json] || mc-libs list-functions [library] [--json]       # displays all functions in a certain library's module
mc-libs examples [library] [--json]                                                 # displays all examples for the functions in a certain library
mc-libs cli [--json]                                                                # displays available cli
mc-libs help [path]                                                                 # displays help for the specified function
mc-libs example [path]                                                              # displays example fo the specified function
mc-libs channels [folder] [pattern]                                                 # displays channels for the specified lcmlog (pattern is usually *.nn)
mc-libs fields [folder] [pattern] [channel]                                         # displays all fields of a specified channel in a lcm log
```
