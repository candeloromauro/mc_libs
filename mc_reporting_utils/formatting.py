"""Formatting utilities for human-readable reporting."""

from __future__ import annotations

from datetime import datetime, timezone


def format_bytes_human(num_bytes: int | float, precision: int = 2) -> str:
    """Format a byte count into a human-readable binary unit string.

    Args:
        num_bytes (int | float): Size in bytes.
        precision (int): Number of decimal places for non-byte units.

    Returns:
        str: Human-readable size, e.g. ``"254.24 MB"``.

    Examples:
        >>> format_bytes_human(512)
        '512 B'
        >>> format_bytes_human(266591626)
        '254.24 MB'
    """
    value = float(num_bytes)
    if value < 0:
        sign = "-"
        value = abs(value)
    else:
        sign = ""

    units = ("B", "KB", "MB", "GB", "TB", "PB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{sign}{int(value)} {unit}"
            return f"{sign}{value:.{max(0, int(precision))}f} {unit}"
        value /= 1024.0

    # Unreachable fallback.
    return f"{sign}{int(num_bytes)} B"


def format_timestamp_utc_us(timestamp_us: int | float | None) -> str:
    """Format a Unix timestamp in microseconds as UTC text.

    Args:
        timestamp_us (int | float | None): Unix timestamp in microseconds.

    Returns:
        str: formatted UTC string (``YYYY-mm-dd HH:MM:SS.ffffff UTC``) or
        ``"n/a"`` for invalid or non-positive values.

    Examples:
        >>> format_timestamp_utc_us(1_700_000_000_000_000)
        '2023-11-14 22:13:20.000000 UTC'
    """
    if timestamp_us is None:
        return "n/a"
    try:
        value = int(timestamp_us)
    except Exception:
        return "n/a"
    if value <= 0:
        return "n/a"

    dt = datetime.fromtimestamp(value / 1_000_000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC")


def format_duration_us(duration_us: int | float | None, precision: int = 3) -> str:
    """Format a duration in microseconds as seconds text.

    Args:
        duration_us (int | float | None): duration in microseconds.
        precision (int): number of decimal places in seconds.

    Returns:
        str: duration string in seconds (for example ``"0.125s"``). Returns
        ``"0.000s"`` for invalid or non-positive values.

    Examples:
        >>> format_duration_us(125_000)
        '0.125s'
    """
    if duration_us is None:
        return "0.000s"
    try:
        value = float(duration_us)
    except Exception:
        return "0.000s"
    if value <= 0:
        return "0.000s"
    return f"{value / 1_000_000.0:.{max(0, int(precision))}f}s"


def format_mean_ms(total_us: int | float, samples: int) -> str:
    """Compute and format a mean duration in milliseconds.

    Args:
        total_us (int | float): accumulated duration in microseconds.
        samples (int): number of samples in the accumulation.

    Returns:
        str: mean duration formatted in milliseconds (for example ``"3.4 ms"``),
        or ``"n/a"`` when ``samples <= 0`` or ``total_us`` is invalid.

    Examples:
        >>> format_mean_ms(12_500, 5)
        '2.5 ms'
    """
    if samples <= 0:
        return "n/a"
    try:
        total = float(total_us)
    except Exception:
        return "n/a"
    return f"{(total / float(samples)) / 1000.0:.1f} ms"
