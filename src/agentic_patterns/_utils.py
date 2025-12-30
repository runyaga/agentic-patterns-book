"""Shared utilities for agentic patterns."""

import re
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any


@dataclass
class HistoryBuffer:
    """
    Rolling window buffer for (observation, action) pairs.

    Used for maintaining conversation/execution context with bounded memory.

    Example:
        buf = HistoryBuffer(max_size=5)
        buf.add("User asked about X", "Searched for X")
        buf.add("Found results", "Summarized results")
        context = buf.as_context()
    """

    max_size: int = 10
    _buffer: deque[tuple[str, str]] = field(default_factory=deque)

    def add(self, observation: str, action: str) -> None:
        """Add entry, evicting oldest if at capacity."""
        if len(self._buffer) >= self.max_size:
            self._buffer.popleft()
        self._buffer.append((observation, action))

    def as_context(self, sep: str = "\n") -> str:
        """Format history as context string."""
        return sep.join(
            f"Observation: {obs}\nAction: {act}" for obs, act in self._buffer
        )

    def as_list(self) -> list[tuple[str, str]]:
        """Get history as list of tuples."""
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __bool__(self) -> bool:
        return len(self._buffer) > 0

    def clear(self) -> None:
        """Clear all history."""
        self._buffer.clear()


def parse_xml_tag(text: str, tag: str) -> str | None:
    """
    Extract content from XML-style tag.

    Args:
        text: Raw text containing tags.
        tag: Tag name (without angle brackets).

    Returns:
        Content between tags (stripped), or None if not found.

    Example:
        >>> parse_xml_tag("<plan>Do X then Y</plan>", "plan")
        'Do X then Y'
        >>> parse_xml_tag("No tags here", "plan")
        None
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def strip_xml_tags(text: str, tag: str) -> str:
    """
    Remove tag and its content from text.

    Args:
        text: Raw text containing tags.
        tag: Tag name to remove.

    Returns:
        Text with tag and content removed.

    Example:
        >>> strip_xml_tags("<plan>Do X</plan> Action: move", "plan")
        'Action: move'
    """
    pattern = rf"<{tag}>.*?</{tag}>"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


@dataclass
class MetricsCollector:
    """
    Collect metrics during agent execution.

    Tracks counters and numeric values for calculating frequencies,
    averages, and other statistics.

    Example:
        metrics = MetricsCollector()
        metrics.increment("steps")
        metrics.increment("plans")
        metrics.record("plan_length", 150)
        freq = metrics.get_frequency("plans", "steps")
    """

    _counters: dict[str, int] = field(default_factory=dict)
    _values: dict[str, list[float]] = field(default_factory=dict)
    _start_time: datetime = field(default_factory=datetime.now)

    def increment(self, name: str, by: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + by

    def record(self, name: str, value: float) -> None:
        """Record a numeric value for averaging."""
        if name not in self._values:
            self._values[name] = []
        self._values[name].append(value)

    def get_count(self, name: str) -> int:
        """Get counter value (0 if not set)."""
        return self._counters.get(name, 0)

    def get_sum(self, name: str) -> float:
        """Get sum of recorded values."""
        return sum(self._values.get(name, []))

    def get_average(self, name: str) -> float:
        """Get average of recorded values (0 if none)."""
        values = self._values.get(name, [])
        return sum(values) / len(values) if values else 0.0

    def get_frequency(self, event: str, total: str) -> float:
        """Calculate frequency as event_count / total_count."""
        t = self.get_count(total)
        return self.get_count(event) / t if t > 0 else 0.0

    def elapsed_seconds(self) -> float:
        """Seconds since collector was created."""
        return (datetime.now() - self._start_time).total_seconds()

    def summary(self) -> dict[str, Any]:
        """Get all metrics as dictionary."""
        return {
            "counters": dict(self._counters),
            "averages": {k: self.get_average(k) for k in self._values},
            "elapsed_seconds": self.elapsed_seconds(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._values.clear()
        self._start_time = datetime.now()
