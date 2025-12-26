"""Tests for the Learning and Adaptation Pattern implementation."""

from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.learning import AdaptedPrompt
from agentic_patterns.learning import Experience
from agentic_patterns.learning import ExperienceStore
from agentic_patterns.learning import ExperienceType
from agentic_patterns.learning import FeedbackLoop
from agentic_patterns.learning import FeedbackResult
from agentic_patterns.learning import LearningStats
from agentic_patterns.learning import adapt_prompt
from agentic_patterns.learning import format_examples_as_context
from agentic_patterns.learning import process_feedback
from agentic_patterns.learning import run_adaptive_task
from agentic_patterns.learning import run_with_learning


class TestEnums:
    """Test enum definitions."""

    def test_experience_type_values(self):
        assert ExperienceType.SUCCESS.value == "success"
        assert ExperienceType.FAILURE.value == "failure"
        assert ExperienceType.NEUTRAL.value == "neutral"


class TestExperience:
    """Test Experience model."""

    def test_experience_basic(self):
        exp = Experience(
            task_type="summarize",
            input_text="Test input",
            output_text="Test output",
            outcome=ExperienceType.SUCCESS,
        )
        assert exp.task_type == "summarize"
        assert exp.outcome == ExperienceType.SUCCESS
        assert exp.feedback == ""
        assert isinstance(exp.timestamp, datetime)

    def test_experience_with_feedback(self):
        exp = Experience(
            task_type="translate",
            input_text="Hello",
            output_text="Hola",
            outcome=ExperienceType.SUCCESS,
            feedback="Great translation",
        )
        assert exp.feedback == "Great translation"

    def test_experience_with_metadata(self):
        exp = Experience(
            task_type="classify",
            input_text="Input",
            output_text="Output",
            outcome=ExperienceType.FAILURE,
            metadata={"model": "test", "tokens": 100},
        )
        assert exp.metadata["model"] == "test"


class TestLearningStats:
    """Test LearningStats model."""

    def test_stats_basic(self):
        stats = LearningStats(
            total_experiences=10,
            successes=7,
            failures=3,
            success_rate=0.7,
        )
        assert stats.total_experiences == 10
        assert stats.successes == 7
        assert stats.success_rate == 0.7
        assert stats.task_types == []

    def test_stats_with_task_types(self):
        stats = LearningStats(
            total_experiences=5,
            successes=4,
            failures=1,
            success_rate=0.8,
            task_types=["summarize", "translate"],
        )
        assert len(stats.task_types) == 2


class TestFeedbackResult:
    """Test FeedbackResult model."""

    def test_feedback_result_basic(self):
        result = FeedbackResult(
            original_output="Test output",
            feedback="Good job",
            was_helpful=True,
        )
        assert result.was_helpful
        assert result.improvement_suggestion == ""

    def test_feedback_result_with_suggestion(self):
        result = FeedbackResult(
            original_output="Output",
            feedback="Too verbose",
            was_helpful=True,
            improvement_suggestion="Be more concise",
        )
        assert result.improvement_suggestion == "Be more concise"


class TestAdaptedPrompt:
    """Test AdaptedPrompt model."""

    def test_adapted_prompt_basic(self):
        adapted = AdaptedPrompt(
            original_prompt="Be helpful",
            adapted_prompt="Be helpful and concise",
        )
        assert adapted.learnings_applied == []

    def test_adapted_prompt_with_learnings(self):
        adapted = AdaptedPrompt(
            original_prompt="Original",
            adapted_prompt="Improved",
            learnings_applied=["Be concise", "Use examples"],
        )
        assert len(adapted.learnings_applied) == 2


class TestExperienceStore:
    """Test ExperienceStore class."""

    def test_init_empty(self):
        store = ExperienceStore()
        assert len(store.experiences) == 0
        assert store.max_experiences == 1000

    def test_add_experience(self):
        store = ExperienceStore()
        exp = store.add_experience(
            task_type="summarize",
            input_text="Input",
            output_text="Output",
            outcome=ExperienceType.SUCCESS,
        )
        assert len(store.experiences) == 1
        assert exp.task_type == "summarize"

    def test_add_success(self):
        store = ExperienceStore()
        exp = store.add_success(
            task_type="translate",
            input_text="Hello",
            output_text="Hola",
        )
        assert exp.outcome == ExperienceType.SUCCESS

    def test_add_failure(self):
        store = ExperienceStore()
        exp = store.add_failure(
            task_type="classify",
            input_text="Data",
            output_text="Wrong",
            feedback="Incorrect category",
        )
        assert exp.outcome == ExperienceType.FAILURE
        assert exp.feedback == "Incorrect category"

    def test_max_experiences_limit(self):
        store = ExperienceStore(max_experiences=3)
        for i in range(5):
            store.add_success(
                task_type="test",
                input_text=f"Input {i}",
                output_text=f"Output {i}",
            )
        assert len(store.experiences) == 3
        assert store.experiences[0].input_text == "Input 2"

    def test_get_relevant_examples(self):
        store = ExperienceStore()
        store.add_success("summarize", "Input 1", "Output 1")
        store.add_success("summarize", "Input 2", "Output 2")
        store.add_success("translate", "Input 3", "Output 3")

        examples = store.get_relevant_examples("summarize", k=5)
        assert len(examples) == 2
        assert all(e.task_type == "summarize" for e in examples)

    def test_get_relevant_examples_limit_k(self):
        store = ExperienceStore()
        for i in range(10):
            store.add_success("test", f"Input {i}", f"Output {i}")

        examples = store.get_relevant_examples("test", k=3)
        assert len(examples) == 3
        # Should return most recent
        assert examples[-1].input_text == "Input 9"

    def test_get_relevant_examples_only_successes(self):
        store = ExperienceStore()
        store.add_success("task", "Input 1", "Output 1")
        store.add_failure("task", "Input 2", "Output 2")
        store.add_success("task", "Input 3", "Output 3")

        examples = store.get_relevant_examples("task", only_successes=True)
        assert len(examples) == 2
        assert all(e.outcome == ExperienceType.SUCCESS for e in examples)

    def test_get_relevant_examples_include_failures(self):
        store = ExperienceStore()
        store.add_success("task", "Input 1", "Output 1")
        store.add_failure("task", "Input 2", "Output 2")

        examples = store.get_relevant_examples("task", only_successes=False)
        assert len(examples) == 2

    def test_get_stats(self):
        store = ExperienceStore()
        store.add_success("summarize", "In1", "Out1")
        store.add_success("summarize", "In2", "Out2")
        store.add_failure("translate", "In3", "Out3")

        stats = store.get_stats()
        assert stats.total_experiences == 3
        assert stats.successes == 2
        assert stats.failures == 1
        assert stats.success_rate == pytest.approx(2 / 3)
        assert set(stats.task_types) == {"summarize", "translate"}

    def test_get_stats_empty(self):
        store = ExperienceStore()
        stats = store.get_stats()
        assert stats.total_experiences == 0
        assert stats.success_rate == 0.0

    def test_get_failure_patterns(self):
        store = ExperienceStore()
        store.add_failure("task", "In1", "Out1", feedback="Too long")
        store.add_failure("task", "In2", "Out2", feedback="Wrong format")
        store.add_success("task", "In3", "Out3")

        patterns = store.get_failure_patterns("task")
        assert len(patterns) == 2
        assert "Too long" in patterns
        assert "Wrong format" in patterns

    def test_get_failure_patterns_no_feedback(self):
        store = ExperienceStore()
        store.add_failure("task", "In", "Out", feedback="")

        patterns = store.get_failure_patterns("task")
        assert len(patterns) == 0

    def test_clear(self):
        store = ExperienceStore()
        store.add_success("task", "In", "Out")
        store.clear()
        assert len(store.experiences) == 0


class TestFeedbackLoop:
    """Test FeedbackLoop class."""

    def test_init_defaults(self):
        loop = FeedbackLoop()
        assert loop.improvement_threshold == 0.7
        assert isinstance(loop.store, ExperienceStore)

    def test_record_outcome_success(self):
        loop = FeedbackLoop()
        exp = loop.record_outcome(
            task_type="test",
            input_text="Input",
            output_text="Output",
            success=True,
        )
        assert exp.outcome == ExperienceType.SUCCESS

    def test_record_outcome_failure(self):
        loop = FeedbackLoop()
        exp = loop.record_outcome(
            task_type="test",
            input_text="Input",
            output_text="Output",
            success=False,
            feedback="Error occurred",
        )
        assert exp.outcome == ExperienceType.FAILURE
        assert exp.feedback == "Error occurred"

    def test_should_adapt_insufficient_data(self):
        loop = FeedbackLoop()
        # Less than 5 experiences
        for i in range(3):
            loop.record_outcome("task", f"In{i}", f"Out{i}", success=True)

        assert not loop.should_adapt("task")

    def test_should_adapt_high_success_rate(self):
        loop = FeedbackLoop(improvement_threshold=0.7)
        # 80% success rate
        for i in range(8):
            loop.record_outcome("task", f"In{i}", f"Out{i}", success=True)
        for i in range(2):
            loop.record_outcome("task", f"Fail{i}", f"Out{i}", success=False)

        assert not loop.should_adapt("task")

    def test_should_adapt_low_success_rate(self):
        loop = FeedbackLoop(improvement_threshold=0.7)
        # 50% success rate
        for i in range(5):
            loop.record_outcome("task", f"In{i}", f"Out{i}", success=True)
        for i in range(5):
            loop.record_outcome("task", f"Fail{i}", f"Out{i}", success=False)

        assert loop.should_adapt("task")

    def test_get_improvement_suggestions(self):
        loop = FeedbackLoop()
        loop.record_outcome(
            "task", "In1", "Out1", success=False, feedback="Issue 1"
        )
        loop.record_outcome(
            "task", "In2", "Out2", success=False, feedback="Issue 2"
        )

        suggestions = loop.get_improvement_suggestions("task")
        assert len(suggestions) == 2

    def test_get_stats(self):
        loop = FeedbackLoop()
        loop.record_outcome("test", "In", "Out", success=True)
        stats = loop.get_stats()
        assert stats.total_experiences == 1


class TestFormatExamplesAsContext:
    """Test format_examples_as_context function."""

    def test_empty_examples(self):
        result = format_examples_as_context([])
        assert result == ""

    def test_single_example(self):
        examples = [
            Experience(
                task_type="test",
                input_text="Test input",
                output_text="Test output",
                outcome=ExperienceType.SUCCESS,
            )
        ]
        result = format_examples_as_context(examples)
        assert "Example 1:" in result
        assert "Test input" in result
        assert "Test output" in result

    def test_multiple_examples(self):
        examples = [
            Experience(
                task_type="test",
                input_text=f"Input {i}",
                output_text=f"Output {i}",
                outcome=ExperienceType.SUCCESS,
            )
            for i in range(3)
        ]
        result = format_examples_as_context(examples)
        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "Example 3:" in result


class TestRunWithLearning:
    """Test run_with_learning function."""

    @pytest.mark.asyncio
    async def test_run_without_examples(self):
        store = ExperienceStore()

        mock_result = MagicMock()
        mock_result.output = "Task completed"

        with patch("agentic_patterns.learning.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            output, _ = await run_with_learning(
                task_type="test",
                input_text="Do something",
                store=store,
            )

            assert output == "Task completed"

    @pytest.mark.asyncio
    async def test_run_with_examples(self):
        store = ExperienceStore()
        store.add_success("test", "Example input", "Example output")

        mock_result = MagicMock()
        mock_result.output = "Learned output"

        with patch("agentic_patterns.learning.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            output, _ = await run_with_learning(
                task_type="test",
                input_text="New input",
                store=store,
            )

            # Examples are now injected via @system_prompt decorator
            # The user message should be just the input text
            call_args = mock_agent.run.call_args[0][0]
            assert call_args == "New input"

            # Deps should be passed with store and task_type
            call_kwargs = mock_agent.run.call_args[1]
            deps = call_kwargs.get("deps")
            assert deps is not None
            assert deps.store is store
            assert deps.task_type == "test"
            assert deps.use_examples is True

    @pytest.mark.asyncio
    async def test_run_without_example_context(self):
        store = ExperienceStore()
        store.add_success("test", "Example", "Output")

        mock_result = MagicMock()
        mock_result.output = "Output"

        with patch("agentic_patterns.learning.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            await run_with_learning(
                task_type="test",
                input_text="Input",
                store=store,
                use_examples=False,
            )

            # Deps should have use_examples=False
            call_kwargs = mock_agent.run.call_args[1]
            deps = call_kwargs.get("deps")
            assert deps is not None
            assert deps.use_examples is False


class TestProcessFeedback:
    """Test process_feedback function."""

    @pytest.mark.asyncio
    async def test_process_feedback_basic(self):
        mock_result = MagicMock()
        mock_result.output = FeedbackResult(
            original_output="Output",
            feedback="Good",
            was_helpful=True,
        )

        with patch(
            "agentic_patterns.learning.feedback_evaluator"
        ) as mock_eval:
            mock_eval.run = AsyncMock(return_value=mock_result)

            result = await process_feedback("Output", "Good")

            assert result.was_helpful
            mock_eval.run.assert_called_once()


class TestAdaptPrompt:
    """Test adapt_prompt function."""

    @pytest.mark.asyncio
    async def test_adapt_prompt_no_feedback(self):
        result = await adapt_prompt("Original prompt", [])

        assert result.original_prompt == "Original prompt"
        assert result.adapted_prompt == "Original prompt"
        assert result.learnings_applied == []

    @pytest.mark.asyncio
    async def test_adapt_prompt_with_feedback(self):
        mock_result = MagicMock()
        mock_result.output = AdaptedPrompt(
            original_prompt="Original",
            adapted_prompt="Improved prompt",
            learnings_applied=["Be concise"],
        )

        with patch(
            "agentic_patterns.learning.prompt_adapter_agent"
        ) as mock_adapter:
            mock_adapter.run = AsyncMock(return_value=mock_result)

            result = await adapt_prompt(
                "Original",
                ["Issue 1", "Issue 2"],
            )

            assert result.adapted_prompt == "Improved prompt"


class TestRunAdaptiveTask:
    """Test run_adaptive_task function."""

    @pytest.mark.asyncio
    async def test_adaptive_task_basic(self):
        loop = FeedbackLoop()

        mock_result = MagicMock()
        mock_result.output = "Output"

        with patch("agentic_patterns.learning.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            output = await run_adaptive_task(
                task_type="test",
                input_text="Input",
                feedback_loop=loop,
            )

            assert output == "Output"

    @pytest.mark.asyncio
    async def test_adaptive_task_record_outcome(self):
        loop = FeedbackLoop()

        mock_result = MagicMock()
        mock_result.output = "Output"

        with patch("agentic_patterns.learning.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            await run_adaptive_task(
                task_type="test",
                input_text="Input",
                feedback_loop=loop,
                record_outcome=True,
                success=True,
                feedback="Good result",
            )

            assert len(loop.store.experiences) == 1
            assert loop.store.experiences[0].outcome == ExperienceType.SUCCESS


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_experience_empty_strings(self):
        exp = Experience(
            task_type="",
            input_text="",
            output_text="",
            outcome=ExperienceType.NEUTRAL,
        )
        assert exp.task_type == ""

    def test_experience_store_single_experience(self):
        store = ExperienceStore()
        store.add_success("task", "In", "Out")
        stats = store.get_stats()
        assert stats.success_rate == 1.0

    def test_feedback_loop_different_task_types(self):
        loop = FeedbackLoop()
        loop.record_outcome("task1", "In", "Out", success=True)
        loop.record_outcome("task2", "In", "Out", success=False)

        stats = loop.get_stats()
        assert set(stats.task_types) == {"task1", "task2"}

    def test_get_relevant_examples_no_matches(self):
        store = ExperienceStore()
        store.add_success("task_a", "In", "Out")

        examples = store.get_relevant_examples("task_b")
        assert len(examples) == 0

    def test_learning_stats_all_failures(self):
        store = ExperienceStore()
        store.add_failure("task", "In1", "Out1")
        store.add_failure("task", "In2", "Out2")

        stats = store.get_stats()
        assert stats.successes == 0
        assert stats.failures == 2
        assert stats.success_rate == 0.0

    def test_format_long_examples_truncated(self):
        long_text = "x" * 500
        examples = [
            Experience(
                task_type="test",
                input_text=long_text,
                output_text=long_text,
                outcome=ExperienceType.SUCCESS,
            )
        ]
        result = format_examples_as_context(examples)
        # Should truncate to 200 chars
        assert len(result) < len(long_text) * 2

    def test_experience_with_metadata(self):
        store = ExperienceStore()
        exp = store.add_experience(
            task_type="test",
            input_text="In",
            output_text="Out",
            outcome=ExperienceType.SUCCESS,
            metadata={"key": "value"},
        )
        assert exp.metadata["key"] == "value"
