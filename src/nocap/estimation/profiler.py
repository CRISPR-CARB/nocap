"""Optional timing instrumentation for estimation operations."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from time import perf_counter
from typing import Self, TextIO

logger = logging.getLogger(__name__)

__all__ = ["EstimationProfiler"]


@dataclass
class _Timing:
    calls: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    last_metadata: dict[str, object] | None = None


class EstimationProfiler:
    """Collect opt-in timing statistics for distribution estimation.

    The profiler is deliberately lightweight when unused: all estimation APIs
    accept ``profiler=None`` by default. Use :meth:`summary` after an estimate
    to inspect aggregate timings, or :meth:`log_summary` to send them to the
    module logger at INFO level.
    """

    def __init__(self, log_file: str | PathLike[str] | TextIO | None = None) -> None:
        """Initialize the profiler, optionally streaming records to a file.

        ``log_file`` may be a path or an already-open text stream. Records are
        flushed after every completed operation so the file can be followed
        while estimation is still running. File streams opened by this class
        are closed by :meth:`close`.
        """
        self._timings: dict[str, _Timing] = {}
        self._log_stream: TextIO | None = None
        self._owns_log_stream = False
        if isinstance(log_file, (str, PathLike)):
            self._log_stream = open(log_file, "a", encoding="utf-8")
            self._owns_log_stream = True
        elif log_file is not None:
            self._log_stream = log_file

    @contextmanager
    def measure(self, name: str, **metadata: object):
        """Measure one named operation and retain its latest metadata."""
        started = perf_counter()
        try:
            yield
        finally:
            self._record(name, perf_counter() - started, **metadata)

    def summary(self) -> list[dict[str, object]]:
        """Return timing rows ordered from most to least total time."""
        return [
            {
                "operation": name,
                "calls": timing.calls,
                "total_seconds": timing.total_seconds,
                "mean_seconds": timing.total_seconds / timing.calls,
                "max_seconds": timing.max_seconds,
                "last_metadata": timing.last_metadata,
            }
            for name, timing in sorted(
                self._timings.items(), key=lambda item: item[1].total_seconds, reverse=True
            )
        ]

    def _record(self, name: str, elapsed: float, **metadata: object) -> None:
        """Record an already-timed operation."""
        timing = self._timings.setdefault(name, _Timing())
        timing.calls += 1
        timing.total_seconds += elapsed
        timing.max_seconds = max(timing.max_seconds, elapsed)
        timing.last_metadata = metadata or None
        if self._log_stream is not None:
            self._log_stream.write(
                f"operation={name} calls={timing.calls} elapsed={elapsed:.6f}s "
                f"total={timing.total_seconds:.6f}s metadata={metadata or {}}\n"
            )
            self._log_stream.flush()

    def close(self) -> None:
        """Flush and close a log stream opened by this profiler."""
        if self._log_stream is not None:
            self._log_stream.flush()
            if self._owns_log_stream:
                self._log_stream.close()
            self._log_stream = None

    def __enter__(self) -> Self:
        """Return the profiler for use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close any profiler-owned log stream."""
        self.close()

    def log_summary(self) -> None:
        """Log the collected timing rows at INFO level."""
        for row in self.summary():
            logger.info(
                "estimation timing operation=%s calls=%d total=%.6fs mean=%.6fs max=%.6fs metadata=%s",
                row["operation"],
                row["calls"],
                row["total_seconds"],
                row["mean_seconds"],
                row["max_seconds"],
                row["last_metadata"],
            )
