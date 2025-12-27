"""Tests for the Evaluation module using pydantic-evals."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agentic_patterns.evaluation import PATTERN_DATASETS
from agentic_patterns.evaluation import Case
from agentic_patterns.evaluation import CollaborationSuccessEvaluator
from agentic_patterns.evaluation import ConfidenceThresholdEvaluator
from agentic_patterns.evaluation import Dataset
from agentic_patterns.evaluation import IntentMatchEvaluator
from agentic_patterns.evaluation import TaskCompletionEvaluator
from agentic_patterns.evaluation import create_multi_agent_dataset
from agentic_patterns.evaluation import create_routing_dataset
from agentic_patterns.evaluation import evaluate_pattern
from agentic_patterns.evaluation import evaluate_pattern_sync


# Mock models for testing
class MockRouteDecision(BaseModel):
    """Mock routing decision."""

    intent: str = "order_status"
    confidence: float = 0.95
    reasoning: str = "Test reasoning"

    class Intent:
        value = "order_status"


class MockRouteDecisionWithIntent(BaseModel):
    """Mock routing decision with intent enum-like."""

    confidence: float = 0.95
    reasoning: str = "Test reasoning"

    class intent:  # noqa: N801
        value = "order_status"


class MockCollaborationResult(BaseModel):
    """Mock collaboration result."""

    success: bool = True
    task_results: list = []


class MockTaskResult(BaseModel):
    """Mock task result."""

    success: bool = True


class TestIntentMatchEvaluator:
    """Tests for IntentMatchEvaluator."""

    def test_matching_intent(self) -> None:
        """Test evaluator returns 1.0 for matching intent."""
        evaluator = IntentMatchEvaluator(expected_intent="order_status")

        # Create mock context
        ctx = MagicMock()
        ctx.output = (MockRouteDecisionWithIntent(),)

        score = evaluator.evaluate(ctx)
        assert score == 1.0

    def test_non_matching_intent(self) -> None:
        """Test evaluator returns 0.0 for non-matching intent."""
        evaluator = IntentMatchEvaluator(expected_intent="product_info")

        ctx = MagicMock()
        ctx.output = (MockRouteDecisionWithIntent(),)

        score = evaluator.evaluate(ctx)
        assert score == 0.0

    def test_empty_output(self) -> None:
        """Test evaluator handles empty output."""
        evaluator = IntentMatchEvaluator(expected_intent="order_status")

        ctx = MagicMock()
        ctx.output = None

        score = evaluator.evaluate(ctx)
        assert score == 0.0

    def test_output_without_intent(self) -> None:
        """Test evaluator handles output without intent attribute."""
        evaluator = IntentMatchEvaluator(expected_intent="order_status")

        ctx = MagicMock()
        ctx.output = ("no intent here",)

        score = evaluator.evaluate(ctx)
        assert score == 0.0


class TestConfidenceThresholdEvaluator:
    """Tests for ConfidenceThresholdEvaluator."""

    def test_above_threshold(self) -> None:
        """Test returns confidence when above threshold."""
        evaluator = ConfidenceThresholdEvaluator(min_confidence=0.7)

        ctx = MagicMock()
        decision = MagicMock()
        decision.confidence = 0.95
        ctx.output = (decision,)

        score = evaluator.evaluate(ctx)
        assert score == 0.95

    def test_below_threshold(self) -> None:
        """Test returns 0.0 when below threshold."""
        evaluator = ConfidenceThresholdEvaluator(min_confidence=0.7)

        ctx = MagicMock()
        decision = MagicMock()
        decision.confidence = 0.5
        ctx.output = (decision,)

        score = evaluator.evaluate(ctx)
        assert score == 0.0

    def test_at_threshold(self) -> None:
        """Test returns confidence when at threshold."""
        evaluator = ConfidenceThresholdEvaluator(min_confidence=0.7)

        ctx = MagicMock()
        decision = MagicMock()
        decision.confidence = 0.7
        ctx.output = (decision,)

        score = evaluator.evaluate(ctx)
        assert score == 0.7

    def test_empty_output(self) -> None:
        """Test handles empty output."""
        evaluator = ConfidenceThresholdEvaluator(min_confidence=0.7)

        ctx = MagicMock()
        ctx.output = None

        score = evaluator.evaluate(ctx)
        assert score == 0.0


class TestCollaborationSuccessEvaluator:
    """Tests for CollaborationSuccessEvaluator."""

    def test_successful_collaboration(self) -> None:
        """Test returns 1.0 for successful collaboration."""
        evaluator = CollaborationSuccessEvaluator()

        ctx = MagicMock()
        ctx.output = MockCollaborationResult(success=True)

        score = evaluator.evaluate(ctx)
        assert score == 1.0

    def test_failed_collaboration(self) -> None:
        """Test returns 0.0 for failed collaboration."""
        evaluator = CollaborationSuccessEvaluator()

        ctx = MagicMock()
        ctx.output = MockCollaborationResult(success=False)

        score = evaluator.evaluate(ctx)
        assert score == 0.0

    def test_none_output(self) -> None:
        """Test handles None output."""
        evaluator = CollaborationSuccessEvaluator()

        ctx = MagicMock()
        ctx.output = None

        score = evaluator.evaluate(ctx)
        assert score == 0.0


class TestTaskCompletionEvaluator:
    """Tests for TaskCompletionEvaluator."""

    def test_all_tasks_successful(self) -> None:
        """Test returns 1.0 when all tasks succeed."""
        evaluator = TaskCompletionEvaluator()

        ctx = MagicMock()
        result = MagicMock()
        result.task_results = [
            MockTaskResult(success=True),
            MockTaskResult(success=True),
            MockTaskResult(success=True),
        ]
        ctx.output = result

        score = evaluator.evaluate(ctx)
        assert score == 1.0

    def test_partial_success(self) -> None:
        """Test returns ratio for partial success."""
        evaluator = TaskCompletionEvaluator()

        ctx = MagicMock()
        result = MagicMock()
        result.task_results = [
            MockTaskResult(success=True),
            MockTaskResult(success=False),
            MockTaskResult(success=True),
        ]
        ctx.output = result

        score = evaluator.evaluate(ctx)
        assert abs(score - 2 / 3) < 0.01

    def test_all_tasks_failed(self) -> None:
        """Test returns 0.0 when all tasks fail."""
        evaluator = TaskCompletionEvaluator()

        ctx = MagicMock()
        result = MagicMock()
        result.task_results = [
            MockTaskResult(success=False),
            MockTaskResult(success=False),
        ]
        ctx.output = result

        score = evaluator.evaluate(ctx)
        assert score == 0.0

    def test_empty_tasks(self) -> None:
        """Test handles empty task list."""
        evaluator = TaskCompletionEvaluator()

        ctx = MagicMock()
        result = MagicMock()
        result.task_results = []
        ctx.output = result

        score = evaluator.evaluate(ctx)
        assert score == 0.0

    def test_none_output(self) -> None:
        """Test handles None output."""
        evaluator = TaskCompletionEvaluator()

        ctx = MagicMock()
        ctx.output = None

        score = evaluator.evaluate(ctx)
        assert score == 0.0


class TestCreateRoutingDataset:
    """Tests for create_routing_dataset factory."""

    def test_creates_dataset(self) -> None:
        """Test creates a valid dataset."""
        dataset = create_routing_dataset()

        assert isinstance(dataset, Dataset)
        assert len(dataset.cases) == 4

    def test_has_expected_cases(self) -> None:
        """Test dataset has expected case names."""
        dataset = create_routing_dataset()

        case_names = {case.name for case in dataset.cases}
        expected = {
            "order_status_query",
            "product_info_query",
            "technical_support_query",
            "ambiguous_query",
        }
        assert case_names == expected

    def test_has_evaluators(self) -> None:
        """Test dataset has evaluators."""
        dataset = create_routing_dataset()

        assert len(dataset.evaluators) >= 2

    def test_without_llm_judge(self) -> None:
        """Test dataset without LLM judge."""
        dataset = create_routing_dataset(include_llm_judge=False)

        # Should have 2 evaluators (IsInstance, ConfidenceThreshold)
        assert len(dataset.evaluators) == 2

    def test_with_llm_judge(self) -> None:
        """Test dataset with LLM judge adds evaluator."""
        dataset = create_routing_dataset(
            include_llm_judge=True,
            llm_judge_model="test:model",
        )

        # Should have 3 evaluators
        assert len(dataset.evaluators) == 3


class TestCreateMultiAgentDataset:
    """Tests for create_multi_agent_dataset factory."""

    def test_creates_dataset(self) -> None:
        """Test creates a valid dataset."""
        dataset = create_multi_agent_dataset()

        assert isinstance(dataset, Dataset)
        assert len(dataset.cases) == 2

    def test_has_expected_cases(self) -> None:
        """Test dataset has expected case names."""
        dataset = create_multi_agent_dataset()

        case_names = {case.name for case in dataset.cases}
        expected = {"research_task", "analysis_task"}
        assert case_names == expected

    def test_has_evaluators(self) -> None:
        """Test dataset has evaluators."""
        dataset = create_multi_agent_dataset()

        assert len(dataset.evaluators) >= 2


class TestPatternDatasets:
    """Tests for PATTERN_DATASETS registry."""

    def test_has_routing(self) -> None:
        """Test registry has routing pattern."""
        assert "routing" in PATTERN_DATASETS

    def test_has_multi_agent(self) -> None:
        """Test registry has multi_agent pattern."""
        assert "multi_agent" in PATTERN_DATASETS

    def test_factories_callable(self) -> None:
        """Test all factories are callable."""
        for name, factory in PATTERN_DATASETS.items():
            assert callable(factory), f"{name} factory not callable"


class TestEvaluatePattern:
    """Tests for evaluate_pattern function."""

    @pytest.mark.asyncio
    async def test_unknown_pattern_raises(self) -> None:
        """Test raises ValueError for unknown pattern."""
        with pytest.raises(ValueError, match="Unknown pattern"):
            await evaluate_pattern("unknown", lambda x: x)

    @pytest.mark.asyncio
    async def test_routing_pattern(self) -> None:
        """Test evaluating routing pattern."""

        async def mock_route(query: str) -> tuple:
            decision = MagicMock()
            decision.confidence = 0.9
            response = MagicMock()
            return decision, response

        report = await evaluate_pattern("routing", mock_route)

        assert report is not None
        assert len(report.cases) == 4

    @pytest.mark.asyncio
    async def test_with_custom_cases(self) -> None:
        """Test with custom cases."""
        custom_cases = [
            Case(name="custom_test", inputs="test query"),
        ]

        async def mock_route(query: str) -> tuple:
            return MagicMock(), MagicMock()

        report = await evaluate_pattern(
            "routing",
            mock_route,
            custom_cases=custom_cases,
        )

        assert len(report.cases) == 1
        assert report.cases[0].name == "custom_test"

    @pytest.mark.asyncio
    async def test_with_metadata(self) -> None:
        """Test metadata is included in report."""

        async def mock_route(query: str) -> tuple:
            return MagicMock(), MagicMock()

        report = await evaluate_pattern(
            "routing",
            mock_route,
            metadata={"version": "test"},
        )

        assert report is not None


class TestEvaluatePatternSync:
    """Tests for evaluate_pattern_sync function."""

    def test_routing_pattern_sync(self) -> None:
        """Test synchronous evaluation of routing pattern."""

        async def mock_route(query: str) -> tuple:
            decision = MagicMock()
            decision.confidence = 0.9
            response = MagicMock()
            return decision, response

        report = evaluate_pattern_sync("routing", mock_route)

        assert report is not None
        assert len(report.cases) == 4

    def test_unknown_pattern_raises_sync(self) -> None:
        """Test raises ValueError for unknown pattern in sync."""
        with pytest.raises(ValueError, match="Unknown pattern"):
            evaluate_pattern_sync("unknown", lambda x: x)


class TestCase:
    """Tests for Case model from pydantic-evals."""

    def test_case_creation(self) -> None:
        """Test creating a case."""
        case = Case(
            name="test_case",
            inputs="test input",
            expected_output="expected",
            metadata={"key": "value"},
        )

        assert case.name == "test_case"
        assert case.inputs == "test input"
        assert case.expected_output == "expected"
        assert case.metadata == {"key": "value"}

    def test_case_minimal(self) -> None:
        """Test case with minimal fields."""
        case = Case(inputs="test")

        assert case.inputs == "test"
        assert case.expected_output is None


class TestDataset:
    """Tests for Dataset model from pydantic-evals."""

    def test_dataset_creation(self) -> None:
        """Test creating a dataset."""
        cases = [Case(inputs="test1"), Case(inputs="test2")]
        dataset = Dataset(cases=cases)

        assert len(dataset.cases) == 2

    def test_dataset_with_evaluators(self) -> None:
        """Test dataset with evaluators."""
        cases = [Case(inputs="test")]
        evaluators = [ConfidenceThresholdEvaluator()]
        dataset = Dataset(cases=cases, evaluators=evaluators)

        assert len(dataset.evaluators) == 1
