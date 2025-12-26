"""Tests for the Memory Management Pattern implementation (Idiomatic)."""

from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.memory import BufferMemory
from agentic_patterns.memory import ConversationSummary
from agentic_patterns.memory import MemoryDeps
from agentic_patterns.memory import MemoryMessage
from agentic_patterns.memory import MemoryStats
from agentic_patterns.memory import MessageRole
from agentic_patterns.memory import SummaryMemory
from agentic_patterns.memory import WindowMemory
from agentic_patterns.memory import chat_with_memory


class TestEnums:
    """Test enum definitions."""

    def test_message_role_values(self):
        assert MessageRole.USER.value == "user"
        assert MessageRole.AI.value == "ai"
        assert MessageRole.SYSTEM.value == "system"


class TestMemoryMessage:
    """Test MemoryMessage model."""

    def test_message_basic(self):
        msg = MemoryMessage(
            role=MessageRole.USER,
            content="Hello world",
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello world"
        assert msg.metadata == {}
        assert isinstance(msg.timestamp, datetime)

    def test_message_with_metadata(self):
        msg = MemoryMessage(
            role=MessageRole.AI,
            content="Response",
            metadata={"tokens": 50, "model": "test"},
        )
        assert msg.metadata["tokens"] == 50
        assert msg.metadata["model"] == "test"

    def test_message_system_role(self):
        msg = MemoryMessage(
            role=MessageRole.SYSTEM,
            content="System instruction",
        )
        assert msg.role == MessageRole.SYSTEM


class TestConversationSummary:
    """Test ConversationSummary model."""

    def test_summary_basic(self):
        summary = ConversationSummary(
            summary="Discussion about Python",
            message_count=10,
        )
        assert summary.summary == "Discussion about Python"
        assert summary.message_count == 10
        assert summary.key_points == []

    def test_summary_with_key_points(self):
        summary = ConversationSummary(
            summary="Technical discussion",
            message_count=5,
            key_points=["Async patterns", "Error handling"],
        )
        assert len(summary.key_points) == 2


class TestMemoryStats:
    """Test MemoryStats model."""

    def test_stats_basic(self):
        stats = MemoryStats(
            total_messages=10,
            user_messages=5,
            ai_messages=4,
            system_messages=1,
            approximate_tokens=250,
        )
        assert stats.total_messages == 10
        assert stats.user_messages == 5
        assert stats.ai_messages == 4
        assert stats.system_messages == 1
        assert stats.approximate_tokens == 250


class TestBufferMemory:
    """Test BufferMemory class."""

    def test_init_empty(self):
        memory = BufferMemory()
        assert len(memory.messages) == 0

    def test_add_user_message(self):
        memory = BufferMemory()
        memory.add_user_message("Hello")
        assert len(memory.messages) == 1
        assert memory.messages[0].role == MessageRole.USER
        assert memory.messages[0].content == "Hello"

    def test_add_ai_message(self):
        memory = BufferMemory()
        memory.add_ai_message("Hi there")
        assert len(memory.messages) == 1
        assert memory.messages[0].role == MessageRole.AI

    def test_add_message_generic(self):
        memory = BufferMemory()
        memory.add_message(MessageRole.SYSTEM, "System alert")
        assert len(memory.messages) == 1
        assert memory.messages[0].role == MessageRole.SYSTEM

    def test_get_context_formats_correctly(self):
        memory = BufferMemory()
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi there")
        context = memory.get_context()
        assert "USER: Hello" in context
        assert "AI: Hi there" in context

    def test_get_context_empty(self):
        memory = BufferMemory()
        context = memory.get_context()
        assert context == ""

    def test_get_stats(self):
        memory = BufferMemory()
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi")
        stats = memory.get_stats()
        assert stats.total_messages == 2
        assert stats.user_messages == 1
        assert stats.ai_messages == 1
        assert stats.approximate_tokens > 0

    def test_max_messages_limit(self):
        memory = BufferMemory(max_messages=3)
        memory.add_user_message("One")
        memory.add_user_message("Two")
        memory.add_user_message("Three")
        memory.add_user_message("Four")
        assert len(memory.messages) == 3
        assert memory.messages[0].content == "Two"


class TestWindowMemory:
    """Test WindowMemory class."""

    def test_init_default_window(self):
        memory = WindowMemory()
        assert memory.window_size == 10

    def test_init_custom_window(self):
        memory = WindowMemory(window_size=5)
        assert memory.window_size == 5

    def test_add_message_within_window(self):
        memory = WindowMemory(window_size=3)
        memory.add_user_message("One")
        memory.add_user_message("Two")
        assert len(memory.messages) == 2

    def test_window_trimming(self):
        memory = WindowMemory(window_size=3)
        memory.add_user_message("One")
        memory.add_user_message("Two")
        memory.add_user_message("Three")
        memory.add_user_message("Four")
        assert len(memory.messages) == 3
        assert memory.messages[0].content == "Two"
        assert memory.messages[2].content == "Four"

    def test_add_ai_message(self):
        memory = WindowMemory()
        memory.add_ai_message("Response")
        assert memory.messages[0].role == MessageRole.AI

    def test_get_context(self):
        memory = WindowMemory(window_size=5)
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi")
        context = memory.get_context()
        assert "USER: Hello" in context
        assert "AI: Hi" in context

    def test_get_context_empty(self):
        memory = WindowMemory()
        context = memory.get_context()
        assert context == ""

    def test_get_stats(self):
        memory = WindowMemory()
        memory.add_user_message("Test")
        memory.add_ai_message("Response")
        stats = memory.get_stats()
        assert stats.total_messages == 2
        assert stats.user_messages == 1
        assert stats.ai_messages == 1


class TestSummaryMemory:
    """Test SummaryMemory class."""

    def test_init_defaults(self):
        memory = SummaryMemory()
        assert memory.recent_window == 6
        assert memory.summarize_threshold == 12

    def test_init_custom_values(self):
        memory = SummaryMemory(recent_window=4, summarize_threshold=8)
        assert memory.recent_window == 4
        assert memory.summarize_threshold == 8

    def test_add_messages(self):
        memory = SummaryMemory()
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi")
        assert len(memory.messages) == 2

    def test_get_context_messages_only(self):
        memory = SummaryMemory()
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi")
        context = memory.get_context()
        assert "Recent messages:" in context
        assert "USER: Hello" in context

    def test_get_context_with_summaries(self):
        memory = SummaryMemory()
        memory.summaries.append(
            ConversationSummary(
                summary="Earlier talk about Python",
                message_count=5,
                key_points=["Async patterns"],
            )
        )
        memory.add_user_message("New question")
        context = memory.get_context()
        assert "Previous summary:" in context
        assert "Earlier talk about Python" in context
        assert "New question" in context

    def test_get_stats(self):
        memory = SummaryMemory()
        memory.add_user_message("Test")
        memory.summaries.append(
            ConversationSummary(
                summary="Old summary",
                message_count=10,
            )
        )
        stats = memory.get_stats()
        assert stats.total_messages == 1
        assert stats.approximate_tokens > 0

    def test_summaries_direct_access(self):
        """Test that summaries list can be accessed directly."""
        memory = SummaryMemory()
        memory.summaries.append(
            ConversationSummary(summary="Test", message_count=1)
        )
        assert len(memory.summaries) == 1
        assert memory.summaries[0].summary == "Test"


class TestMemoryDeps:
    """Test MemoryDeps dataclass."""

    def test_deps_with_buffer_memory(self):
        memory = BufferMemory()
        deps = MemoryDeps(memory=memory)
        assert deps.memory is memory

    def test_deps_with_window_memory(self):
        memory = WindowMemory(window_size=5)
        deps = MemoryDeps(memory=memory)
        assert deps.memory is memory

    def test_deps_with_summary_memory(self):
        memory = SummaryMemory()
        deps = MemoryDeps(memory=memory)
        assert deps.memory is memory


class TestChatWithMemory:
    """Test chat_with_memory function with MemoryDeps."""

    @pytest.mark.asyncio
    async def test_chat_adds_messages(self):
        memory = BufferMemory()
        deps = MemoryDeps(memory=memory)

        mock_result = MagicMock()
        mock_result.output = "Hello! How can I help?"

        with patch(
            "agentic_patterns.memory.conversational_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            response = await chat_with_memory(deps, "Hi there")

            assert response == "Hello! How can I help?"
            assert len(memory.messages) == 2  # User + AI
            assert memory.messages[0].role == MessageRole.USER
            assert memory.messages[1].role == MessageRole.AI

    @pytest.mark.asyncio
    async def test_chat_passes_deps(self):
        """Test that deps are passed to the agent."""
        memory = BufferMemory()
        deps = MemoryDeps(memory=memory)

        mock_result = MagicMock()
        mock_result.output = "Response"

        with patch(
            "agentic_patterns.memory.conversational_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            await chat_with_memory(deps, "Test")

            # Verify deps were passed
            call_kwargs = mock_agent.run.call_args[1]
            assert call_kwargs.get("deps") is deps

    @pytest.mark.asyncio
    async def test_chat_with_window_memory(self):
        memory = WindowMemory(window_size=4)
        deps = MemoryDeps(memory=memory)

        mock_result = MagicMock()
        mock_result.output = "Response"

        with patch(
            "agentic_patterns.memory.conversational_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            await chat_with_memory(deps, "Test")

            assert len(memory.messages) == 2

    @pytest.mark.asyncio
    async def test_chat_with_summary_memory(self):
        memory = SummaryMemory()
        deps = MemoryDeps(memory=memory)

        mock_result = MagicMock()
        mock_result.output = "Response"

        with patch(
            "agentic_patterns.memory.conversational_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            await chat_with_memory(deps, "Test")

            assert len(memory.messages) == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_memory_message_empty_content(self):
        msg = MemoryMessage(
            role=MessageRole.USER,
            content="",
        )
        assert msg.content == ""

    def test_buffer_memory_single_message(self):
        memory = BufferMemory()
        memory.add_user_message("Only message")
        stats = memory.get_stats()
        assert stats.total_messages == 1

    def test_window_memory_size_one(self):
        memory = WindowMemory(window_size=1)
        memory.add_user_message("First")
        memory.add_user_message("Second")
        assert len(memory.messages) == 1
        assert memory.messages[0].content == "Second"

    def test_summary_memory_no_key_points_in_context(self):
        memory = SummaryMemory()
        memory.summaries.append(
            ConversationSummary(
                summary="Simple summary",
                message_count=5,
                key_points=[],
            )
        )
        context = memory.get_context()
        assert "Simple summary" in context

    def test_memory_stats_all_zeros(self):
        memory = BufferMemory()
        stats = memory.get_stats()
        assert stats.total_messages == 0
        assert stats.user_messages == 0
        assert stats.ai_messages == 0
        assert stats.approximate_tokens == 0

    def test_conversation_summary_many_key_points(self):
        summary = ConversationSummary(
            summary="Long discussion",
            message_count=50,
            key_points=[f"Point {i}" for i in range(20)],
        )
        assert len(summary.key_points) == 20

    def test_buffer_max_messages_exact_limit(self):
        memory = BufferMemory(max_messages=3)
        memory.add_user_message("One")
        memory.add_user_message("Two")
        memory.add_user_message("Three")
        assert len(memory.messages) == 3
        # Add one more to trigger trim
        memory.add_user_message("Four")
        assert len(memory.messages) == 3

    def test_summary_memory_empty_context(self):
        """SummaryMemory returns header even when empty."""
        memory = SummaryMemory()
        context = memory.get_context()
        # With no messages, still shows structure
        assert "Recent messages:" in context
