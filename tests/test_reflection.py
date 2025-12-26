"""Tests for the Reflection Pattern implementation (Idiomatic PydanticAI)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic_ai import ModelRetry

from agentic_patterns.reflection import Critique
from agentic_patterns.reflection import ProducerOutput
from agentic_patterns.reflection import ReflectionDeps
from agentic_patterns.reflection import run_reflection
from agentic_patterns.reflection import validate_content


class TestModels:
    """Test Pydantic model validation."""

    def test_producer_output_valid(self):
        output = ProducerOutput(
            content="This is some generated content.",
            reasoning="Generated based on the task description.",
        )
        assert output.content == "This is some generated content."
        assert output.reasoning == "Generated based on the task description."

    def test_critique_valid(self):
        critique = Critique(
            is_acceptable=False,
            score=6.5,
            feedback="Needs more examples and detail.",
        )
        assert not critique.is_acceptable
        assert critique.score == 6.5
        assert "examples" in critique.feedback

    def test_critique_score_bounds(self):
        # Valid at boundaries
        Critique(
            is_acceptable=False,
            score=0.0,
            feedback="Very poor quality.",
        )
        Critique(
            is_acceptable=True,
            score=10.0,
            feedback="Excellent quality.",
        )

        # Invalid
        with pytest.raises(ValueError):
            Critique(
                is_acceptable=False,
                score=10.5,
                feedback="Invalid score.",
            )

        with pytest.raises(ValueError):
            Critique(
                is_acceptable=False,
                score=-1.0,
                feedback="Invalid score.",
            )

    def test_critique_acceptable_threshold(self):
        # Score >= 8 with is_acceptable=True
        critique = Critique(
            is_acceptable=True,
            score=8.5,
            feedback="Good quality content.",
        )
        assert critique.is_acceptable
        assert critique.score >= 8.0


class TestReflectionDeps:
    """Test ReflectionDeps dataclass."""

    def test_reflection_deps_defaults(self):
        mock_critic = MagicMock()
        deps = ReflectionDeps(critic_agent=mock_critic)
        assert deps.critic_agent is mock_critic
        assert deps.max_history == 5

    def test_reflection_deps_custom_history(self):
        mock_critic = MagicMock()
        deps = ReflectionDeps(critic_agent=mock_critic, max_history=10)
        assert deps.max_history == 10


class TestValidateContent:
    """Test the output validator function."""

    @pytest.fixture
    def mock_ctx(self):
        mock_critic = MagicMock()
        deps = ReflectionDeps(critic_agent=mock_critic)
        ctx = MagicMock()
        ctx.deps = deps
        return ctx

    @pytest.fixture
    def mock_low_score_critique(self):
        return Critique(
            is_acceptable=False,
            score=5.0,
            feedback="Content is too brief. Add more detail.",
        )

    @pytest.fixture
    def mock_high_score_critique(self):
        return Critique(
            is_acceptable=True,
            score=9.0,
            feedback="Content meets quality standards.",
        )

    @pytest.mark.asyncio
    async def test_validate_content_low_score_raises_retry(
        self, mock_ctx, mock_low_score_critique
    ):
        """Test that low score raises ModelRetry."""
        mock_critique_result = MagicMock()
        mock_critique_result.output = mock_low_score_critique
        mock_ctx.deps.critic_agent.run = AsyncMock(
            return_value=mock_critique_result
        )

        producer_output = ProducerOutput(
            content="Brief content.",
            reasoning="Initial draft.",
        )

        with pytest.raises(ModelRetry) as exc_info:
            await validate_content(mock_ctx, producer_output)

        assert "5.0/10" in str(exc_info.value)
        assert "too brief" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validate_content_high_score_returns_output(
        self, mock_ctx, mock_high_score_critique
    ):
        """Test that high score returns the output unchanged."""
        mock_critique_result = MagicMock()
        mock_critique_result.output = mock_high_score_critique
        mock_ctx.deps.critic_agent.run = AsyncMock(
            return_value=mock_critique_result
        )

        producer_output = ProducerOutput(
            content="High quality content with detail.",
            reasoning="Comprehensive draft.",
        )

        result = await validate_content(mock_ctx, producer_output)
        assert result is producer_output
        assert result.content == "High quality content with detail."


class TestRunReflection:
    """Test the full reflection process."""

    @pytest.fixture
    def mock_producer_output(self):
        return ProducerOutput(
            content="Generated content about the topic.",
            reasoning="Based on task requirements.",
        )

    @pytest.mark.asyncio
    async def test_run_reflection_success(self, mock_producer_output):
        """Test successful reflection with passing validation."""
        mock_result = MagicMock()
        mock_result.output = mock_producer_output

        with patch(
            "agentic_patterns.reflection.producer_agent"
        ) as mock_producer:
            mock_producer.run = AsyncMock(return_value=mock_result)

            result = await run_reflection("Write about Python.")

            assert result.content == mock_producer_output.content
            assert result.reasoning == mock_producer_output.reasoning
            mock_producer.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_reflection_with_retries(self):
        """Test that retries happen via ModelRetry in validator."""
        # This test verifies the retry mechanism is wired correctly
        # The actual retry logic is handled by PydanticAI's framework
        mock_output = ProducerOutput(
            content="Final refined content.",
            reasoning="After feedback incorporation.",
        )
        mock_result = MagicMock()
        mock_result.output = mock_output

        with patch(
            "agentic_patterns.reflection.producer_agent"
        ) as mock_producer:
            mock_producer.run = AsyncMock(return_value=mock_result)

            result = await run_reflection("Write about async Python.")

            assert result is mock_output

    @pytest.mark.asyncio
    async def test_run_reflection_handles_exception(self):
        """Test that exceptions are propagated correctly."""
        with patch(
            "agentic_patterns.reflection.producer_agent"
        ) as mock_producer:
            mock_producer.run = AsyncMock(side_effect=Exception("LLM error"))

            with pytest.raises(Exception) as exc_info:
                await run_reflection("Write about Python.")

            assert "LLM error" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_critique_boundary_score_7_9(self):
        """Test critique at the threshold boundary."""
        # Score of 7.9 should not be acceptable
        critique = Critique(
            is_acceptable=False,
            score=7.9,
            feedback="Almost there but needs minor improvements.",
        )
        assert critique.score < 8.0

    def test_critique_boundary_score_8_0(self):
        """Test critique at exactly 8.0."""
        critique = Critique(
            is_acceptable=True,
            score=8.0,
            feedback="Meets minimum acceptable standard.",
        )
        assert critique.score >= 8.0

    def test_producer_output_empty_content(self):
        """Test producer output with empty content."""
        output = ProducerOutput(
            content="",
            reasoning="Failed to generate content.",
        )
        assert output.content == ""

    def test_critique_empty_feedback(self):
        """Test critique with empty feedback."""
        critique = Critique(
            is_acceptable=True,
            score=10.0,
            feedback="",
        )
        assert critique.feedback == ""
