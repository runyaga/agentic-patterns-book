"""Tests for shared utilities."""

import pytest

from agentic_patterns._utils import HistoryBuffer
from agentic_patterns._utils import MetricsCollector
from agentic_patterns._utils import parse_xml_tag
from agentic_patterns._utils import strip_xml_tags


class TestHistoryBuffer:
    """Tests for HistoryBuffer."""

    def test_add_and_len(self):
        buf = HistoryBuffer(max_size=5)
        assert len(buf) == 0
        assert not buf

        buf.add("obs1", "act1")
        assert len(buf) == 1
        assert buf

    def test_rolling_window(self):
        buf = HistoryBuffer(max_size=2)
        buf.add("obs1", "act1")
        buf.add("obs2", "act2")
        buf.add("obs3", "act3")  # evicts (obs1, act1)

        assert len(buf) == 2
        items = buf.as_list()
        assert items[0] == ("obs2", "act2")
        assert items[1] == ("obs3", "act3")

    def test_as_context(self):
        buf = HistoryBuffer(max_size=5)
        buf.add("saw X", "did Y")
        buf.add("saw Z", "did W")

        context = buf.as_context()
        assert "Observation: saw X" in context
        assert "Action: did Y" in context
        assert "Observation: saw Z" in context
        assert "Action: did W" in context

    def test_as_context_custom_separator(self):
        buf = HistoryBuffer(max_size=5)
        buf.add("obs1", "act1")
        buf.add("obs2", "act2")

        context = buf.as_context(sep="\n---\n")
        assert "---" in context

    def test_clear(self):
        buf = HistoryBuffer(max_size=5)
        buf.add("obs1", "act1")
        buf.add("obs2", "act2")
        assert len(buf) == 2

        buf.clear()
        assert len(buf) == 0
        assert not buf


class TestParseXmlTag:
    """Tests for parse_xml_tag."""

    def test_simple_tag(self):
        text = "<plan>Do something</plan>"
        assert parse_xml_tag(text, "plan") == "Do something"

    def test_tag_with_surrounding_text(self):
        text = "Before <plan>The plan content</plan> After"
        assert parse_xml_tag(text, "plan") == "The plan content"

    def test_multiline_content(self):
        text = """<plan>
Step 1: Do X
Step 2: Do Y
Step 3: Do Z
</plan>"""
        result = parse_xml_tag(text, "plan")
        assert "Step 1: Do X" in result
        assert "Step 3: Do Z" in result

    def test_missing_tag(self):
        text = "No tags here"
        assert parse_xml_tag(text, "plan") is None

    def test_different_tag_names(self):
        text = "<thought>thinking...</thought><action>move</action>"
        assert parse_xml_tag(text, "thought") == "thinking..."
        assert parse_xml_tag(text, "action") == "move"

    def test_whitespace_stripped(self):
        text = "<plan>  content with spaces  </plan>"
        assert parse_xml_tag(text, "plan") == "content with spaces"


class TestStripXmlTags:
    """Tests for strip_xml_tags."""

    def test_strip_tag(self):
        text = "<plan>Remove this</plan> Keep this"
        assert strip_xml_tags(text, "plan") == "Keep this"

    def test_strip_preserves_other_content(self):
        text = "Before <plan>Remove</plan> After"
        result = strip_xml_tags(text, "plan")
        assert "Before" in result
        assert "After" in result
        assert "Remove" not in result

    def test_strip_multiline(self):
        text = """<plan>
Line 1
Line 2
</plan>
Action: move"""
        result = strip_xml_tags(text, "plan")
        assert "Action: move" in result
        assert "Line 1" not in result

    def test_no_tag_unchanged(self):
        text = "No tags here"
        assert strip_xml_tags(text, "plan") == "No tags here"


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_increment(self):
        m = MetricsCollector()
        assert m.get_count("steps") == 0

        m.increment("steps")
        assert m.get_count("steps") == 1

        m.increment("steps", by=5)
        assert m.get_count("steps") == 6

    def test_record_and_average(self):
        m = MetricsCollector()
        m.record("length", 10.0)
        m.record("length", 20.0)
        m.record("length", 30.0)

        assert m.get_average("length") == 20.0
        assert m.get_sum("length") == 60.0

    def test_average_empty(self):
        m = MetricsCollector()
        assert m.get_average("missing") == 0.0

    def test_frequency(self):
        m = MetricsCollector()
        m.increment("plans", by=3)
        m.increment("steps", by=10)

        assert m.get_frequency("plans", "steps") == 0.3

    def test_frequency_zero_total(self):
        m = MetricsCollector()
        assert m.get_frequency("plans", "steps") == 0.0

    def test_elapsed_seconds(self):
        m = MetricsCollector()
        elapsed = m.elapsed_seconds()
        assert elapsed >= 0
        assert elapsed < 1  # Should be nearly instant

    def test_summary(self):
        m = MetricsCollector()
        m.increment("steps", by=5)
        m.record("score", 0.8)
        m.record("score", 0.9)

        summary = m.summary()
        assert summary["counters"]["steps"] == 5
        assert summary["averages"]["score"] == pytest.approx(0.85)
        assert "elapsed_seconds" in summary

    def test_reset(self):
        m = MetricsCollector()
        m.increment("steps", by=10)
        m.record("score", 0.5)

        m.reset()
        assert m.get_count("steps") == 0
        assert m.get_average("score") == 0.0
