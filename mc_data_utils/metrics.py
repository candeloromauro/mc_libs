"""Small metric helpers.

Functions:
- first_local_min(x)
- nearest_index(x, target)

Example:
>>> from mc_data_utils.metrics import nearest_index
>>> idx = nearest_index([0, 2, 5], 3)
"""

from __future__ import annotations

import numpy as np


def first_local_min(x: np.ndarray) -> int:
    """Return index of first local minimum; fallback to last index.

    Args:
        x (np.ndarray): 1-D array to scan.

    Returns:
        int: index of the first local minimum, or the last index if none found.

    Examples:
        >>> import numpy as np
        >>> from mc_data_utils.metrics import first_local_min
        >>> first_local_min(np.array([3.0, 2.0, 4.0, 1.0]))
        1
    """
    if len(x) < 3:
        return max(len(x) - 1, 0)
    mins = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
    return int(mins[0]) if len(mins) else len(x) - 1


def nearest_index(x: np.ndarray, target: float) -> int:
    """Return index of element in x closest to target.

    Args:
        x (np.ndarray): 1-D array of candidate values.
        target (float): target value to match.

    Returns:
        int: index of the nearest value in ``x``.

    Examples:
        >>> import numpy as np
        >>> from mc_data_utils.metrics import nearest_index
        >>> nearest_index(np.array([0.0, 2.0, 5.0]), 3.1)
        2
    """
    return int(np.argmin(np.abs(x - target)))
