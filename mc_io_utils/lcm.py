"""LCM log IO utilities.

Functions:
- scan_channels(log_dir, file_pattern, max_events=None)
- scan_logs_detailed(log_paths, ...)
- list_channel_fields(log_path, channels, lcm_packages=None)
- list_channel_fields_in_dir(log_dir, file_pattern, channel, lcm_packages=None)
- lcmlog_to_pickle(log_dir, file_pattern="*.00", channels=None, out_path="lcmlogs_merged.pkl", lcm_packages=None, format="pickle")
- main()  # CLI entrypoint

Example:
>>> from mc_io_utils.lcm import scan_channels
>>> scan_channels("lcmlogs", "*.00")
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, cast, Any, Callable

import argparse
import importlib
import pickle
import re

try:
    from lcm import EventLog as _EventLog
except Exception:  # pragma: no cover
    _EventLog = None

EventLogType = Callable[[str, str], Iterable[Any]]
EventLog: EventLogType | None = cast(EventLogType, _EventLog) if _EventLog is not None else None


@dataclass
class ChannelTimingStats:
    """Timestamp quality metrics for one LCM channel."""

    events: int = 0
    timestamped_events: int = 0
    first_timestamp_us: int = 0
    last_timestamp_us: int = 0
    out_of_order: int = 0
    duplicate_timestamps: int = 0
    gap_over_threshold: int = 0
    max_gap_us: int = 0

    def add_event(self, timestamp_us: int, gap_warning_us: int) -> None:
        """Update channel timing metrics with one event timestamp.

        Args:
            timestamp_us (int): event timestamp in microseconds.
            gap_warning_us (int): threshold above which a gap is flagged.

        Returns:
            None

        Examples:
            >>> stats = ChannelTimingStats()
            >>> stats.add_event(1_000_000, gap_warning_us=500_000)
        """
        self.events += 1
        if timestamp_us <= 0:
            return

        self.timestamped_events += 1
        if self.timestamped_events == 1:
            self.first_timestamp_us = timestamp_us
            self.last_timestamp_us = timestamp_us
            return

        gap_us = timestamp_us - self.last_timestamp_us
        if timestamp_us < self.last_timestamp_us:
            self.out_of_order += 1
        elif timestamp_us == self.last_timestamp_us:
            self.duplicate_timestamps += 1
        else:
            if gap_us > self.max_gap_us:
                self.max_gap_us = gap_us
            if gap_us > gap_warning_us:
                self.gap_over_threshold += 1
        self.last_timestamp_us = timestamp_us


@dataclass
class SyncAgeStats:
    """Simple sonar-vs-navigation timing age statistics in microseconds."""

    sonar_events: int = 0
    gps_age_samples: int = 0
    rph_age_samples: int = 0
    gps_age_sum_us: int = 0
    rph_age_sum_us: int = 0
    gps_age_max_us: int = 0
    rph_age_max_us: int = 0
    missing_gps: int = 0
    missing_rph: int = 0
    negative_gps_age: int = 0
    negative_rph_age: int = 0

    def observe(self, sonar_timestamp_us: int, last_gps_timestamp_us: int, last_rph_timestamp_us: int) -> None:
        """Update sync-age counters for one sonar event.

        Args:
            sonar_timestamp_us (int): sonar event timestamp in microseconds.
            last_gps_timestamp_us (int): most recent GPS timestamp in microseconds.
            last_rph_timestamp_us (int): most recent RPH timestamp in microseconds.

        Returns:
            None

        Examples:
            >>> sync = SyncAgeStats()
            >>> sync.observe(10_000_000, last_gps_timestamp_us=9_500_000, last_rph_timestamp_us=9_700_000)
        """
        self.sonar_events += 1

        if last_gps_timestamp_us <= 0:
            self.missing_gps += 1
        else:
            gps_age_us = sonar_timestamp_us - last_gps_timestamp_us
            if gps_age_us < 0:
                self.negative_gps_age += 1
            else:
                self.gps_age_samples += 1
                self.gps_age_sum_us += gps_age_us
                if gps_age_us > self.gps_age_max_us:
                    self.gps_age_max_us = gps_age_us

        if last_rph_timestamp_us <= 0:
            self.missing_rph += 1
        else:
            rph_age_us = sonar_timestamp_us - last_rph_timestamp_us
            if rph_age_us < 0:
                self.negative_rph_age += 1
            else:
                self.rph_age_samples += 1
                self.rph_age_sum_us += rph_age_us
                if rph_age_us > self.rph_age_max_us:
                    self.rph_age_max_us = rph_age_us


@dataclass
class FileScanStats:
    """Per-file counters and timestamp range."""

    path: str
    size_bytes: int = 0
    events: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    first_timestamp_us: int = 0
    last_timestamp_us: int = 0


@dataclass
class LCMScanStats:
    """Aggregate scan output for one or more LCM logs."""

    total_events: int = 0
    aggregate_counts: Dict[str, int] = field(default_factory=dict)
    per_file: list[FileScanStats] = field(default_factory=list)
    timing_by_channel: Dict[str, ChannelTimingStats] = field(default_factory=dict)
    first_timestamp_us: int = 0
    last_timestamp_us: int = 0
    sync: SyncAgeStats = field(default_factory=SyncAgeStats)


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


def scan_logs_detailed(
    log_paths: Iterable[str | Path],
    *,
    max_events_per_file: int | None = None,
    gap_warning_seconds: float = 1.0,
    sonar_channels: Optional[Iterable[str]] = None,
    gps_channels: Optional[Iterable[str]] = None,
    rph_channels: Optional[Iterable[str]] = None,
) -> LCMScanStats:
    """Scan LCM logs and return counts, timestamp quality, and optional sync metrics.

    Args:
        log_paths: iterable of LCM log file paths to scan in order.
        max_events_per_file: optional cap of scanned events per file.
        gap_warning_seconds: threshold used for ``gap_over_threshold`` counters.
        sonar_channels: channels used as sync reference events.
        gps_channels: channels treated as GPS time references.
        rph_channels: channels treated as RPH/attitude time references.

    Returns:
        LCMScanStats: aggregate and per-file metrics.

    Raises:
        RuntimeError: If LCM bindings are unavailable.
        ValueError: If ``log_paths`` is empty.
        FileNotFoundError: If any provided file does not exist.

    Examples:
        >>> from mc_io_utils.lcm import scan_logs_detailed
        >>> stats = scan_logs_detailed(["log.00"], gap_warning_seconds=0.5)
        >>> stats.total_events >= 0
        True
    """
    if EventLog is None:
        raise RuntimeError("lcm is not installed or available in this environment.")

    paths = [Path(p) for p in log_paths]
    if not paths:
        raise ValueError("No log paths provided.")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing LCM log file(s): {[str(p) for p in missing]}")

    gap_warning_us = max(1, int(float(gap_warning_seconds) * 1_000_000.0))
    sonar_set = {str(ch) for ch in (sonar_channels or [])}
    gps_set = {str(ch) for ch in (gps_channels or [])}
    rph_set = {str(ch) for ch in (rph_channels or [])}

    stats = LCMScanStats()
    aggregate_counter: Counter[str] = Counter()
    last_gps_timestamp_us = 0
    last_rph_timestamp_us = 0

    for log_path in paths:
        file_counter: Counter[str] = Counter()
        file_stats = FileScanStats(path=str(log_path), size_bytes=log_path.stat().st_size)

        log = EventLog(str(log_path), "r")
        try:
            for event in cast(Iterable[Any], log):
                channel = str(getattr(event, "channel", ""))
                if not channel:
                    continue

                timestamp_us = int(getattr(event, "timestamp", 0) or 0)
                file_counter[channel] += 1
                aggregate_counter[channel] += 1
                stats.total_events += 1
                file_stats.events += 1

                timing = stats.timing_by_channel.setdefault(channel, ChannelTimingStats())
                timing.add_event(timestamp_us, gap_warning_us)

                if timestamp_us > 0:
                    if file_stats.first_timestamp_us == 0:
                        file_stats.first_timestamp_us = timestamp_us
                    file_stats.last_timestamp_us = timestamp_us

                    if stats.first_timestamp_us == 0 or timestamp_us < stats.first_timestamp_us:
                        stats.first_timestamp_us = timestamp_us
                    if timestamp_us > stats.last_timestamp_us:
                        stats.last_timestamp_us = timestamp_us

                    if channel in gps_set:
                        last_gps_timestamp_us = timestamp_us
                    if channel in rph_set:
                        last_rph_timestamp_us = timestamp_us
                    if channel in sonar_set:
                        stats.sync.observe(timestamp_us, last_gps_timestamp_us, last_rph_timestamp_us)

                if max_events_per_file is not None and file_stats.events >= int(max_events_per_file):
                    break
        finally:
            close_fn = getattr(log, "close", None)
            if callable(close_fn):
                close_fn()

        file_stats.counts = dict(file_counter)
        stats.per_file.append(file_stats)

    stats.aggregate_counts = dict(aggregate_counter)
    return stats


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
    try:
        smat = importlib.import_module("navlib.lcmlog.log_to_smat")
        make_lcmtype_dictionary_fn = getattr(smat, "make_lcmtype_dictionary")
        msg_getfields_fn = getattr(smat, "msg_getfields")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("navlib is required to decode LCM messages; install navlib and retry.") from exc

    lcm_packages = lcm_packages or []
    type_db = make_lcmtype_dictionary_fn(lcm_packages=lcm_packages)
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
            fields = list(msg_getfields_fn(msg))
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


def lcmlog_to_pickle(
    log_dir: str,
    file_pattern: str = "*.00",
    channels: Optional[Iterable[str]] = None,
    out_path: str = "lcmlogs_merged.pkl",
    lcm_packages: Optional[list[str]] = None,
    format: str = "pickle",
    *,
    verbose: bool = True,
) -> Path:
    """Convert raw LCM logs to a merged output file.

    Args:
        log_dir: directory containing LCM log files.
        file_pattern: glob pattern for log files (default: *.00).
        channels: optional iterable of channel names to include. If None, includes all.
        out_path: output path.
        lcm_packages: optional list of LCM packages to load.
        format: output format (currently supports "pickle" or "pkl").
        verbose: pass-through verbosity flag to the parser.

    Returns:
        Path: path to the written output file.

    Raises:
        RuntimeError: If ``navlib`` cannot be imported.
        ValueError: If ``format`` is unsupported.

    Examples:
        >>> from mc_io_utils.lcm import lcmlog_to_pickle
        >>> out = lcmlog_to_pickle("lcmlogs", file_pattern="*.00", out_path="merged.pkl")
        >>> out.name
        'merged.pkl'
    """
    try:
        navlib_lcmlog = importlib.import_module("navlib.lcmlog")
        read_logs = getattr(navlib_lcmlog, "read_logs")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("navlib is required to parse LCM logs; install navlib and retry.") from exc

    if channels:
        channels_regex = "|".join(re.escape(ch) for ch in channels)
    else:
        channels_regex = ".*"

    data = read_logs(
        log_dir,
        file_pattern=file_pattern,
        verbose=verbose,
        lcm_packages=lcm_packages or [],
        channels_to_process=channels_regex,
    )

    fmt = format.lower().strip()
    if fmt not in {"pickle", "pkl"}:
        raise ValueError(f"Unsupported format '{format}'. Supported: pickle, pkl")

    data_dict: object = data
    if not hasattr(data, "keys") and hasattr(data, "__dict__"):
        data_dict = data.__dict__

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    with out_path_p.open("wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path_p


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
