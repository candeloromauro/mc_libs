"""Runtime utilities for console, logging, and LCM log progress estimation.

Functions:
- clear_console()
- setup_logging_from_yaml(cfg_path, logs_dir="logs", prefix="compas_run")
- check_output_file(output_path, remove_existing=True)
- sum_total_bytes(file_path)
- estimate_total_bytes(file_path, sample_events=20000)
"""

from __future__ import annotations

import logging
import logging.config
import os
import platform
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def clear_console() -> None:
    """Clear the current terminal screen.

    The command used depends on the host OS:
    ``cls`` on Windows and ``clear`` on Unix-like systems.

    Args:
        None

    Returns:
        None

    Examples:
        >>> from mc_io_utils.runtime import clear_console
        >>> clear_console()
    """
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def setup_logging_from_yaml(
    cfg_path: str | PathLike[str],
    logs_dir: str | PathLike[str] = "logs",
    prefix: str = "compas_run",
) -> None:
    """Configure Python logging from a YAML config file.

    Any handler named ``file`` or ``json_file`` that contains a ``filename``
    entry is rewritten to include a timestamped basename under ``logs_dir``.

    Args:
        cfg_path (str | PathLike[str]): path to a YAML logging configuration file.
        logs_dir (str | PathLike[str]): directory where timestamped log files are created.
        prefix (str): prefix used for generated log filenames.

    Returns:
        None

    Raises:
        RuntimeError: If ``PyYAML`` is not available.

    Examples:
        >>> from mc_io_utils.runtime import setup_logging_from_yaml
        >>> setup_logging_from_yaml("logging.yaml", logs_dir="logs", prefix="run")
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional dependency at import time
        raise RuntimeError("PyYAML is required for setup_logging_from_yaml.") from exc

    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    Path(logs_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{timestamp}"

    for handler in ("file", "json_file"):
        if handler in cfg.get("handlers", {}):
            if "filename" in cfg["handlers"][handler]:
                suffix = Path(cfg["handlers"][handler]["filename"]).suffix
                cfg["handlers"][handler]["filename"] = f"{logs_dir}/{base_name}{suffix}"

    logging.config.dictConfig(cfg)


def check_output_file(output_path: Path, *, remove_existing: bool = True) -> bool:
    """Validate an output path by creating/truncating the file.

    This helper ensures the parent directory exists, optionally removes an
    existing file, and verifies write permission with a short open/close cycle.

    Args:
        output_path (Path): destination path to validate.
        remove_existing (bool): when ``True``, remove an existing file before validating.

    Returns:
        bool: ``True`` if the output file can be created/written, ``False`` otherwise.

    Examples:
        >>> from pathlib import Path
        >>> from mc_io_utils.runtime import check_output_file
        >>> ok = check_output_file(Path("out/data.pkl"))
        >>> isinstance(ok, bool)
        True
    """
    try:
        output_dir = output_path.parent
        if output_dir and not output_dir.exists():
            logger.info("Creating missing output directory: %s", output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if remove_existing and output_path.exists():
            logger.info("Removing existing output file: %s", output_path)
            output_path.unlink()

        output_path.open("wb").close()
        logger.info("Output path is writable: %s", output_path)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Output path %s is not writable: %s", output_path, exc)
        return False


def sum_total_bytes(file_path: str | Path) -> Optional[int]:
    """Return exact payload bytes by scanning every event in an LCM log.

    Args:
        file_path (str | Path): path to an LCM log file.

    Returns:
        Optional[int]: exact sum of ``len(event.data)`` across all events, or
        ``None`` if scanning fails (for example missing LCM bindings).

    Examples:
        >>> from mc_io_utils.runtime import sum_total_bytes
        >>> total = sum_total_bytes("session.00")
    """
    try:
        import lcm

        total = 0
        with lcm.EventLog(str(file_path), "r") as prescan:
            for ev in prescan:
                total += len(ev.data or b"")
        return total
    except Exception:
        return None


def estimate_total_bytes(file_path: str | Path, sample_events: int = 20000) -> Optional[int]:
    """Estimate total payload bytes from an initial sample of events.

    The estimator reads up to ``sample_events`` events, computes average payload
    size, approximates event count from on-disk file size, and returns an
    estimated total payload size. Falls back to coarse file-size heuristics if
    parsing is unavailable.

    Args:
        file_path (str | Path): path to an LCM log file.
        sample_events (int): maximum number of events used for sampling.

    Returns:
        Optional[int]: estimated payload bytes, or ``None`` when estimation fails.

    Examples:
        >>> from mc_io_utils.runtime import estimate_total_bytes
        >>> estimate_total_bytes("session.00", sample_events=5000)
    """
    try:
        import lcm

        n = 0
        payload_sum = 0
        with lcm.EventLog(str(file_path), "r") as prescan:
            for ev in prescan:
                payload_sum += len(ev.data or b"")
                n += 1
                if n >= sample_events:
                    break

        if n == 0:
            return _coarse_fallback(file_path)

        avg_payload = max(1, payload_sum // n)
        total_file_bytes = os.path.getsize(Path(file_path))
        avg_event_bytes = avg_payload + 128
        est_events = max(n, int(total_file_bytes // avg_event_bytes))
        return est_events * avg_payload
    except Exception:
        return _coarse_fallback(file_path)


def _coarse_fallback(file_path: str | Path) -> Optional[int]:
    try:
        return os.path.getsize(Path(file_path))
    except Exception:
        return None
