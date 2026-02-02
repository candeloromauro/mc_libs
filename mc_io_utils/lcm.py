"""LCM log IO utilities.

Functions:
- scan_channels(log_dir, file_pattern, max_events=None)
- list_channel_fields(log_path, channels, lcm_packages=None)
- list_channel_fields_in_dir(log_dir, file_pattern, channel, lcm_packages=None)
- main()  # CLI entrypoint

Example:
>>> from mc_io_utils.lcm import scan_channels
>>> scan_channels("lcmlogs", "*.00")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, cast, Any

import argparse
from navlib.lcmlog.log_to_smat import make_lcmtype_dictionary, msg_getfields

try:
    from lcm import EventLog
except Exception:  # pragma: no cover
    EventLog = None


def scan_channels(log_dir: str, file_pattern: str, max_events: int | None = None) -> Dict[str, int]:
    """Scan raw LCM log files and return channel counts.

    Args:
        log_dir (str): directory containing LCM log files.
        file_pattern (str): glob pattern to match log files (e.g. ``"*.00"``).
        max_events (int | None): optional cap on events to scan across files.

    Returns:
        Dict[str, int]: mapping of channel name to message count.

    Raises:
        FileNotFoundError: If the directory or matching files are not found.
        RuntimeError: If the LCM Python bindings are unavailable.

    Examples:
        >>> from mc_io_utils.lcm import scan_channels
        >>> counts = scan_channels("lcmlogs", "*.00", max_events=1000)
    """
    if EventLog is None:
        raise RuntimeError("lcm is not installed or available in this environment.")

    directory = Path(log_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    files = sorted(directory.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No log files found in {log_dir} matching pattern '{file_pattern}'")

    counts: Dict[str, int] = {}
    seen_events = 0
    for fpath in files:
        log = EventLog(str(fpath), "r")
        for event in cast(Iterable[Any], log):
            counts[event.channel] = counts.get(event.channel, 0) + 1
            seen_events += 1
            if max_events is not None and seen_events >= max_events:
                return counts
    return counts


def list_channel_fields(
    log_path: str,
    channels: Iterable[str],
    lcm_packages: Optional[list[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Decode one message per channel and list available fields.

    Returns a dict: {channel: {"lcmtype": str|None, "fields": [..], "header_fields": [..]}}

    Args:
        log_path (str): path to a single LCM log file.
        channels (Iterable[str]): channel names to inspect.
        lcm_packages (Optional[list[str]]): optional list of Python LCM packages to import.

    Returns:
        Dict[str, Dict[str, object]]: per-channel metadata including lcmtype and field lists.

    Raises:
        RuntimeError: If the LCM Python bindings are unavailable.

    Examples:
        >>> from mc_io_utils.lcm import list_channel_fields
        >>> info = list_channel_fields("log.00", ["IMU_KEARFOTT_COMPAS"])
    """
    if EventLog is None:
        raise RuntimeError("lcm is not installed or available in this environment.")

    lcm_packages = lcm_packages or []
    type_db = make_lcmtype_dictionary(lcm_packages=lcm_packages)
    channels = set(channels)

    results: Dict[str, Dict[str, object]] = {}
    log = EventLog(str(log_path), "r")
    for event in cast(Iterable[Any], log):
        ch = event.channel
        if ch in results or ch not in channels:
            continue
        lcmtype = type_db.get(event.data[:8], None)
        if lcmtype is None:
            results[ch] = {"lcmtype": None, "fields": []}
        else:
            msg = lcmtype.decode(event.data)
            fields = list(msg_getfields(msg))
            entry: Dict[str, object] = {"lcmtype": lcmtype.__name__, "fields": fields}
            if "header" in fields:
                header = getattr(msg, "header", None)
                if header is not None and hasattr(header, "__slots__"):
                    entry["header_fields"] = list(header.__slots__)
            results[ch] = entry
        if len(results) == len(channels):
            break

    return results


def list_channel_fields_in_dir(
    log_dir: str,
    file_pattern: str,
    channel: str,
    lcm_packages: Optional[list[str]] = None,
) -> Dict[str, object]:
    """Convenience wrapper to read one log file in a directory and list channel fields.

    Args:
        log_dir (str): directory containing LCM logs.
        file_pattern (str): glob pattern used to select a log file.
        channel (str): channel name to inspect.
        lcm_packages (Optional[list[str]]): optional list of Python LCM packages to import.

    Returns:
        Dict[str, object]: dictionary with ``channel``, ``lcmtype``, and ``fields`` entries.

    Raises:
        FileNotFoundError: If the directory or matching files are not found.

    Examples:
        >>> from mc_io_utils.lcm import list_channel_fields_in_dir
        >>> entry = list_channel_fields_in_dir("lcmlogs", "*.00", "IMU_KEARFOTT_COMPAS")
    """
    directory = Path(log_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    files = sorted(directory.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No log files found in {log_dir} matching pattern '{file_pattern}'")

    info = list_channel_fields(str(files[0]), [channel], lcm_packages=lcm_packages)
    entry = info.get(channel, {"lcmtype": None, "fields": []})
    entry["channel"] = channel
    return entry


def _cmd_scan(args: argparse.Namespace) -> int:
    counts = scan_channels(args.log_dir, args.pattern, args.max_events)
    print("Channels found:")
    for ch in sorted(counts.keys()):
        print(f" - {ch} ({counts[ch]} msgs)")
    return 0


def _cmd_fields(args: argparse.Namespace) -> int:
    channels = args.channels.split(",") if args.channels else []
    if not channels:
        raise SystemExit("No channels provided. Use --channels \"CH1,CH2\"")
    lcm_packages = args.lcm_packages.split(",") if args.lcm_packages else []
    info = list_channel_fields(args.log_path, channels, lcm_packages=lcm_packages)
    for ch in channels:
        entry = info.get(ch, {})
        print(ch)
        print("  lcmtype:", entry.get("lcmtype"))
        print("  fields:")
        for f in cast(list, entry.get("fields", [])):
            print("   -", f)
        if "header_fields" in entry:
            print("  header fields:")
            for hf in cast(list, entry["header_fields"]):
                print("   -", hf)
        print("")
    return 0


def main() -> int:
    """CLI entrypoint for LCM log utilities.

    Args:
        None

    Returns:
        int: exit code (0 for success).

    Examples:
        >>> from mc_io_utils.lcm import main
        >>> isinstance(main(), int)
        True
    """
    parser = argparse.ArgumentParser(description="LCM log utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="List channels in raw LCM logs")
    scan.add_argument("--log-dir", default="lcmlogs", help="Directory containing LCM logs")
    scan.add_argument("--pattern", default="*.00", help="Glob pattern for log files")
    scan.add_argument("--max-events", type=int, default=None, help="Limit number of events scanned")
    scan.set_defaults(func=_cmd_scan)

    fields = sub.add_parser("fields", help="List fields for specific channels")
    fields.add_argument("--log-path", required=True, help="Path to a single LCM log file")
    fields.add_argument("--channels", required=True, help="Comma-separated channel list")
    fields.add_argument("--lcm-packages", default="", help="Comma-separated LCM python packages")
    fields.set_defaults(func=_cmd_fields)

    args = parser.parse_args()
    return cast(int, args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
