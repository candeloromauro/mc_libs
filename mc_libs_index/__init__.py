"""Index of mc_* libraries and their functions.

Exports:
- list_libraries() -> str (tree)
- list_functions(library) -> str (tree)
- list_examples(library)
- list_all() -> str (tree)
- list_all_auto(base_path=None)
- discover_libraries(base_path=None)
- get_help(func_path)
- get_example(func_path)
- tree(libraries=None)
- print_tree(libraries=None)
- list_cli_commands()

Example:
>>> from mc_libs_index import list_libraries, list_functions
>>> list_libraries()
>>> list_functions('mc_data_utils')
"""

from . import index as _index
from typing import Callable
from .index import (
    list_libraries,
    list_functions,
    list_examples,
    list_all,
    list_all_auto,
    discover_libraries,
    get_help,
    get_example,
    tree,
    print_tree,
)


def list_cli_commands() -> str:
    """Return a tree-like list of CLI commands, usage, and targets.

    Args:
        None

    Returns:
        str: tree-style list of CLI commands.

    Examples:
        >>> from mc_libs_index import list_cli_commands
        >>> print(list_cli_commands())
    """
    func: Callable[[], str] = getattr(_index, "list_cli_commands")  # type: ignore[assignment]
    return func()

__all__ = [
    "list_libraries",
    "list_functions",
    "list_examples",
    "list_all",
    "list_all_auto",
    "discover_libraries",
    "get_help",
    "get_example",
    "tree",
    "print_tree",
    "list_cli_commands",
]
