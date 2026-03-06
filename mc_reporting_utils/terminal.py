"""Terminal interaction helpers for human-facing reporting."""

from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import TextIO


class Spinner:
    """Minimal terminal spinner for long-running operations.

    The spinner animates only when the output stream is a TTY. In non-interactive
    runs it falls back to simple start/done text output.
    """

    def __init__(
        self,
        message: str,
        done_message: str | None = None,
        frames: str = "-\\|/",
        interval_s: float = 0.1,
        stream: TextIO | None = None,
    ) -> None:
        self.message = message
        self.done_message = done_message
        self.frames = tuple(frames) if frames else ("-",)
        self.interval_s = max(0.02, float(interval_s))
        self.stream = stream or sys.stdout

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._running = False

    def start(self) -> "Spinner":
        """Start spinner animation and return the spinner instance.

        Args:
            None

        Returns:
            Spinner: the same spinner object (useful for chaining/context use).

        Examples:
            >>> from mc_reporting_utils.terminal import Spinner
            >>> sp = Spinner("Loading").start()
            >>> sp.stop("Done")
        """
        if self._running:
            return self

        self._running = True
        self._stop_event.clear()
        self._enabled = bool(getattr(self.stream, "isatty", lambda: False)())

        if self._enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            self.stream.write(f"{self.message} ...\n")
            self.stream.flush()

        return self

    def stop(self, done_message: str | None = None) -> None:
        """Stop spinner animation and optionally print a final message.

        Args:
            done_message (str | None): message to print after stopping. If
                ``None``, uses the instance default ``done_message``.

        Returns:
            None

        Examples:
            >>> from mc_reporting_utils.terminal import Spinner
            >>> sp = Spinner("Working", done_message="Finished").start()
            >>> sp.stop()
        """
        if not self._running:
            return

        if self._enabled and self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self.stream.write("\r" + (" " * (len(self.message) + 4)) + "\r")
            self.stream.flush()

        self._running = False
        final_message = done_message if done_message is not None else self.done_message
        if final_message:
            self.stream.write(f"{final_message}\n")
            self.stream.flush()

    def _spin(self) -> None:
        for frame in itertools.cycle(self.frames):
            if self._stop_event.is_set():
                break
            self.stream.write(f"\r{self.message} {frame}")
            self.stream.flush()
            time.sleep(self.interval_s)

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()
