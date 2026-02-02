"""CLI for mc_* libraries."""

from __future__ import annotations

import argparse
import json

from mc_libs_index import (
    list_all,
    list_all_auto,
    list_libraries,
    list_functions,
    list_examples,
    discover_libraries,
    get_help,
    get_example,
    list_cli_commands,
)
from mc_io_utils.lcm import scan_channels, list_channel_fields_in_dir


def main() -> int:
    """Entry point for the ``mc-libs`` CLI.

    Args:
        None

    Returns:
        int: exit code (0 for success).

    Examples:
        >>> from mc_cli import main
        >>> isinstance(main(), int)
        True
    """
    parser = argparse.ArgumentParser(prog="mc-libs", description="CLI for mc_* libraries")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tree_cmd = sub.add_parser("tree", help="Show full library tree")
    tree_cmd.add_argument("--json", action="store_true", help="Output JSON instead of text")
    tree_alias = sub.add_parser("list-all", help="Alias for tree (full library tree)")
    tree_alias.add_argument("--json", action="store_true", help="Output JSON instead of text")

    libs_cmd = sub.add_parser("libs", help="List libraries")
    libs_cmd.add_argument("--json", action="store_true", help="Output JSON instead of text")
    libs_alias = sub.add_parser("list-libraries", help="Alias for libs")
    libs_alias.add_argument("--json", action="store_true", help="Output JSON instead of text")

    funcs = sub.add_parser("funcs", help="List functions for a library")
    funcs.add_argument("library", help="Library name, e.g. mc_data_utils")
    funcs.add_argument("--json", action="store_true", help="Output JSON instead of text")
    funcs_alias = sub.add_parser("list-functions", help="Alias for funcs")
    funcs_alias.add_argument("library", help="Library name, e.g. mc_data_utils")
    funcs_alias.add_argument("--json", action="store_true", help="Output JSON instead of text")

    examples = sub.add_parser("examples", help="List examples for a library")
    examples.add_argument("library", help="Library name, e.g. mc_data_utils")
    examples.add_argument("--json", action="store_true", help="Output JSON instead of text")

    cli_cmd = sub.add_parser("cli", help="List mc-libs CLI commands and targets")
    cli_cmd.add_argument("--json", action="store_true", help="Output JSON instead of text")

    help_cmd = sub.add_parser("help", help="Show docstring for a function/class")
    help_cmd.add_argument("path", help="Fully qualified name, e.g. mc_data_utils.allan2")

    ex_cmd = sub.add_parser("example", help="Show example for a function/class")
    ex_cmd.add_argument("path", help="Fully qualified name, e.g. mc_data_utils.allan2")

    channels = sub.add_parser("channels", help="List channels in an LCM log")
    channels.add_argument("folder", help="Folder containing LCM logs")
    channels.add_argument("pattern", nargs="?", default="*.00", help="Glob pattern (default: *.00)")

    fields = sub.add_parser("fields", help="List fields for a channel in an LCM log")
    fields.add_argument("folder", help="Folder containing LCM logs")
    fields.add_argument("pattern", nargs="?", default="*.00", help="Glob pattern (default: *.00)")
    fields.add_argument("channel", help="Channel name, e.g. IMU_KEARFOTT_COMPAS")

    args = parser.parse_args()

    if args.cmd in {"tree", "list-all"}:
        if args.json:
            print(json.dumps(list_all_auto(), indent=2, sort_keys=True))
        else:
            print(list_all())
        return 0
    if args.cmd in {"libs", "list-libraries"}:
        if args.json:
            print(json.dumps(discover_libraries(), indent=2, sort_keys=True))
        else:
            print(list_libraries())
        return 0
    if args.cmd in {"funcs", "list-functions"}:
        if args.json:
            examples_map = list_examples(args.library)
            print(json.dumps(sorted(examples_map.keys()), indent=2))
        else:
            print(list_functions(args.library))
        return 0
    if args.cmd == "examples":
        examples_map = list_examples(args.library)
        if args.json:
            print(json.dumps(examples_map, indent=2, sort_keys=True))
        else:
            for name in sorted(examples_map.keys()):
                print(f"{name}:")
                print(examples_map[name])
                print("")
        return 0
    if args.cmd == "help":
        print(get_help(args.path))
        return 0
    if args.cmd == "example":
        print(get_example(args.path))
        return 0
    if args.cmd == "cli":
        if args.json:
            mapping = {}
            for line in list_cli_commands().splitlines()[1:]:
                if "->" not in line:
                    continue
                _, rest = line.split(" ", 1)
                cmd, target = [p.strip() for p in rest.split("->", 1)]
                mapping[cmd] = target
            print(json.dumps(mapping, indent=2, sort_keys=True))
        else:
            print(list_cli_commands())
        return 0
    if args.cmd == "channels":
        result = scan_channels(args.folder, args.pattern)
        for name, count in result.items():
            print(f"{name} ({count} msgs)")
        return 0
    if args.cmd == "fields":
        result = list_channel_fields_in_dir(args.folder, args.pattern, args.channel)
        print(f"channel: {result.get('channel')}")
        lcmtype = result.get("lcmtype")
        if lcmtype:
            print(f"lcmtype: {lcmtype}")
        fields_list = result.get("fields", [])
        if fields_list:
            print("fields:")
            for field in fields_list:
                print(f"  - {field}")
        header_fields = result.get("header_fields")
        if header_fields:
            print("header fields:")
            for field in header_fields:
                print(f"  - {field}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
