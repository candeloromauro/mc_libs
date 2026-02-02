"""Index of mc_* libraries, functions, and examples."""

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import importlib
import inspect
import pkgutil
import sys


def _extract_example(doc: Optional[str]) -> str:
    if not doc:
        return "No example available."
    lines = doc.splitlines()
    try:
        idx = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("example"))
    except StopIteration:
        return "No example available."
    example_lines: List[str] = []
    for line in lines[idx + 1 :]:
        if line.strip() == "":
            if example_lines:
                break
            continue
        example_lines.append(line.strip())
    return "\n".join(example_lines) if example_lines else "No example available."


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _missing_packages_message(exc: Exception) -> str:
    missing: List[str] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ModuleNotFoundError):
            name = getattr(cur, "name", None)
            if name:
                missing.append(name)
        cur = cur.__cause__ or cur.__context__

    if missing:
        pkgs = ", ".join(sorted(set(missing)))
        return f"modules cannot be displayed; missing packages: {pkgs} (install and retry)"
    return f"modules cannot be displayed; import failed: {type(exc).__name__}: {exc}"


def discover_libraries(base_path: str | Path | None = None) -> List[str]:
    """Discover mc_* libraries in a given path (defaults to project root).

    Args:
        base_path (str | Path | None): base directory to scan (defaults to project root).

    Returns:
        List[str]: sorted list of discovered package names.

    Examples:
        >>> from mc_libs_index.index import discover_libraries
        >>> libs = discover_libraries()
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parents[1]
    else:
        base_path = Path(base_path)
    names = []
    for mod in pkgutil.iter_modules([str(base_path)]):
        if mod.ispkg and mod.name.startswith("mc_"):
            names.append(mod.name)
    return sorted(names)


def build_registry(base_path: str | Path | None = None) -> Dict[str, List[dict]]:
    """Build a registry by introspecting mc_* libraries.

    Args:
        base_path (str | Path | None): base directory to scan (defaults to project root).

    Returns:
        Dict[str, List[dict]]: mapping from library name to entries with ``name`` and ``example``.

    Examples:
        >>> from mc_libs_index.index import build_registry
        >>> registry = build_registry()
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parents[1]
    else:
        base_path = Path(base_path)
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))

    registry: Dict[str, List[dict]] = {}
    for lib in discover_libraries(base_path):
        entries: List[dict] = []
        try:
            pkg = importlib.import_module(lib)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            registry[lib] = [
                {
                    "name": f"{lib}.__import_error__",
                    "example": "",
                    "error": _missing_packages_message(exc),
                }
            ]
            continue
        if pkg.__file__ is None:
            continue
        pkg_path = Path(pkg.__file__).parent
        for mod in pkgutil.walk_packages([str(pkg_path)], prefix=f"{lib}."):
            module = importlib.import_module(mod.name)
            for name, obj in inspect.getmembers(module):
                if not _is_public(name):
                    continue
                if inspect.isfunction(obj) or inspect.isclass(obj):
                    if getattr(obj, "__module__", "") != module.__name__:
                        continue
                    entries.append(
                        {
                            "name": f"{module.__name__}.{name}",
                            "example": _extract_example(inspect.getdoc(obj)),
                        }
                    )
        registry[lib] = entries
    return registry


def list_libraries() -> str:
    """Return a tree-like string of discovered mc_* libraries.

    Args:
        None

    Returns:
        str: tree-style list of library names.

    Examples:
        >>> from mc_libs_index.index import list_libraries
        >>> print(list_libraries())
    """
    libs = discover_libraries()
    if not libs:
        return ""
    lines: List[str] = []
    for i, lib in enumerate(libs):
        branch = "├─" if i < len(libs) - 1 else "└─"
        lines.append(f"{branch} {lib}")
    return "\n".join(lines)


def list_functions(library: str) -> str:
    """Return a tree-like string of functions for a library.

    Args:
        library (str): library name, e.g. ``"mc_data_utils"``.

    Returns:
        str: tree-style list of functions for the library.

    Raises:
        KeyError: If the library is unknown.

    Examples:
        >>> from mc_libs_index.index import list_functions
        >>> print(list_functions("mc_data_utils"))
    """
    registry = build_registry()
    if library not in registry:
        raise KeyError(f"Unknown library: {library}")
    return tree({library: registry[library]})


def list_examples(library: str) -> Dict[str, str]:
    """Return example calls for a library.

    Args:
        library (str): library name, e.g. ``"mc_data_utils"``.

    Returns:
        Dict[str, str]: mapping from fully qualified names to example strings.

    Raises:
        KeyError: If the library is unknown.

    Examples:
        >>> from mc_libs_index.index import list_examples
        >>> examples = list_examples("mc_data_utils")
    """
    registry = build_registry()
    if library not in registry:
        raise KeyError(f"Unknown library: {library}")
    return {entry["name"]: entry["example"] for entry in registry[library]}


def list_all() -> str:
    """Return the auto-discovered registry as a tree-like string.

    Args:
        None

    Returns:
        str: tree-style list of all libraries and functions.

    Examples:
        >>> from mc_libs_index.index import list_all
        >>> print(list_all())
    """
    return tree(build_registry())


def list_all_auto(base_path: str | Path | None = None) -> Dict[str, List[dict]]:
    """Return an auto-discovered registry by introspection.

    Args:
        base_path (str | Path | None): base directory to scan (defaults to project root).

    Returns:
        Dict[str, List[dict]]: registry mapping for all libraries.

    Examples:
        >>> from mc_libs_index.index import list_all_auto
        >>> registry = list_all_auto()
    """
    return build_registry(base_path=base_path)


def get_help(func_path: str) -> str:
    """Return the docstring for a fully qualified function/class path.

    Args:
        func_path (str): fully qualified name, e.g. ``"mc_data_utils.allan2"``.

    Returns:
        str: docstring text (or a fallback message).

    Raises:
        ValueError: If ``func_path`` is not fully qualified.

    Examples:
        >>> from mc_libs_index.index import get_help
        >>> print(get_help("mc_data_utils.allan2"))
    """
    module_path, _, name = func_path.rpartition(".")
    if not module_path:
        raise ValueError("func_path must be fully qualified, e.g. mc_data_utils.allan2")
    module = importlib.import_module(module_path)
    obj = getattr(module, name)
    doc = inspect.getdoc(obj)
    return doc or "No docstring available."


def get_example(func_path: str) -> str:
    """Return the example block for a fully qualified function/class path.

    Args:
        func_path (str): fully qualified name, e.g. ``"mc_data_utils.allan2"``.

    Returns:
        str: example text if available, else a fallback message.

    Examples:
        >>> from mc_libs_index.index import get_example
        >>> print(get_example("mc_data_utils.allan2"))
    """
    return _extract_example(get_help(func_path))


def tree(libraries: Dict[str, List[dict]] | None = None) -> str:
    """Return a tree-like string grouped by library and module.

    Args:
        libraries (Dict[str, List[dict]] | None): optional registry; when None, build one.

    Returns:
        str: tree-style summary of libraries and functions.

    Examples:
        >>> from mc_libs_index.index import tree
        >>> print(tree())
    """
    if libraries is None:
        libraries = build_registry()

    def split_mod(name: str) -> tuple[str, str]:
        mod, _, func = name.rpartition(".")
        return mod, func

    lines: List[str] = []
    libs = sorted(libraries.keys())
    for li, lib in enumerate(libs):
        is_last_lib = li == len(libs) - 1
        lines.append(lib)
        entries = libraries.get(lib, [])
        if entries and entries[0].get("error"):
            err = entries[0].get("error", "modules cannot be displayed; unknown import error")
            prefix = "│  " if not is_last_lib else "   "
            lines.append(f"{prefix}└─ import-error: {err}")
            continue
        # group by module
        modules: Dict[str, List[str]] = {}
        for entry in entries:
            mod, func = split_mod(entry["name"])
            if mod.startswith(f"{lib}."):
                mod = mod[len(lib) + 1 :]
            modules.setdefault(mod, []).append(func)
        module_names = sorted(modules.keys())
        for mi, mod in enumerate(module_names):
            is_last_mod = mi == len(module_names) - 1
            lib_prefix = "│  " if not is_last_lib else "   "
            mod_branch = "├─" if not is_last_mod else "└─"
            lines.append(f"{lib_prefix}{mod_branch} {mod}")
            funcs = sorted(modules[mod])
            for fi, func in enumerate(funcs):
                is_last_func = fi == len(funcs) - 1
                func_prefix = f"{lib_prefix}{'│  ' if not is_last_mod else '   '}"
                func_branch = "├─" if not is_last_func else "└─"
                lines.append(f"{func_prefix}{func_branch} {func}")
    return "\n".join(lines)


def print_tree(libraries: Dict[str, List[dict]] | None = None) -> None:
    """Print a tree-like list of libraries and function names.

    Args:
        libraries (Dict[str, List[dict]] | None): optional registry; when None, build one.

    Returns:
        None

    Examples:
        >>> from mc_libs_index.index import print_tree
        >>> print_tree()
    """
    print(tree(libraries))


def list_cli_commands() -> str:
    """Return a tree-like list of mc-libs CLI commands, usage, and targets.

    Args:
        None

    Returns:
        str: tree-style list of CLI commands.

    Examples:
        >>> from mc_libs_index.index import list_cli_commands
        >>> print(list_cli_commands())
    """
    commands = [
        ("tree [--json]", "mc_libs_index.list_all"),
        ("list-all [--json]", "mc_libs_index.list_all"),
        ("libs [--json]", "mc_libs_index.list_libraries"),
        ("list-libraries [--json]", "mc_libs_index.list_libraries"),
        ("funcs <library> [--json]", "mc_libs_index.list_functions"),
        ("list-functions <library> [--json]", "mc_libs_index.list_functions"),
        ("examples <library> [--json]", "mc_libs_index.list_examples"),
        ("help <fully.qualified.path>", "mc_libs_index.get_help"),
        ("example <fully.qualified.path>", "mc_libs_index.get_example"),
        ("channels <folder> [pattern]", "mc_io_utils.lcm.scan_channels"),
        ("fields <folder> [pattern] <channel>", "mc_io_utils.lcm.list_channel_fields_in_dir"),
        ("cli [--json]", "mc_libs_index.list_cli_commands"),
    ]
    lines: List[str] = ["mc-libs (usage -> target)"]
    for i, (cmd, target) in enumerate(commands):
        branch = "├─" if i < len(commands) - 1 else "└─"
        lines.append(f"{branch} {cmd} -> {target}")
    return "\n".join(lines)


__all__ = [
    "discover_libraries",
    "build_registry",
    "list_libraries",
    "list_functions",
    "list_examples",
    "list_all",
    "list_all_auto",
    "get_help",
    "get_example",
    "tree",
    "print_tree",
    "list_cli_commands",
]
