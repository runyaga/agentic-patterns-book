"""Tests for the Resource-Aware Optimization pattern module."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.resource_aware import ComplexityAssessment
from agentic_patterns.resource_aware import ComplexityClassifier
from agentic_patterns.resource_aware import ModelConfig
from agentic_patterns.resource_aware import ModelSelector
from agentic_patterns.resource_aware import ModelTier
from agentic_patterns.resource_aware import ResourceAwareExecutor
from agentic_patterns.resource_aware import ResourceBudget
from agentic_patterns.resource_aware import ResourceStats
from agentic_patterns.resource_aware import ResourceType
from agentic_patterns.resource_aware import ResourceUsage
from agentic_patterns.resource_aware import TaskComplexity
from agentic_patterns.resource_aware import estimate_cost
from agentic_patterns.resource_aware import estimate_tokens
from agentic_patterns.resource_aware import prune_context


class TestTaskComplexity:
    """Tests for TaskComplexity enum."""

    def test_complexity_values(self) -> None:
        """Test all complexity values exist."""
        assert TaskComplexity.SIMPLE == "simple"
        assert TaskComplexity.MEDIUM == "medium"
        assert TaskComplexity.COMPLEX == "complex"


class TestModelTier:
    """Tests for ModelTier enum."""

    def test_tier_values(self) -> None:
        """Test all tier values exist."""
        assert ModelTier.LIGHTWEIGHT == "lightweight"
        assert ModelTier.STANDARD == "standard"
        assert ModelTier.ADVANCED == "advanced"


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_resource_type_values(self) -> None:
        """Test all resource type values exist."""
        assert ResourceType.TOKENS == "tokens"
        assert ResourceType.COST == "cost"
        assert ResourceType.TIME == "time"
        assert ResourceType.CALLS == "calls"


class TestResourceUsage:
    """Tests for ResourceUsage model."""

    def test_resource_usage_creation(self) -> None:
        """Test creating resource usage record."""
        usage = ResourceUsage(
            operation_id="op_001",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost=0.001,
            execution_time_ms=500,
            model_tier=ModelTier.STANDARD,
            task_complexity=TaskComplexity.MEDIUM,
        )
        assert usage.operation_id == "op_001"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.estimated_cost == 0.001
        assert usage.execution_time_ms == 500
        assert usage.model_tier == ModelTier.STANDARD

    def test_resource_usage_defaults(self) -> None:
        """Test resource usage default values."""
        usage = ResourceUsage(
            operation_id="op_002",
            model_tier=ModelTier.LIGHTWEIGHT,
            task_complexity=TaskComplexity.SIMPLE,
        )
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.estimated_cost == 0.0
        assert usage.execution_time_ms == 0


class TestComplexityAssessment:
    """Tests for ComplexityAssessment model."""

    def test_assessment_creation(self) -> None:
        """Test creating complexity assessment."""
        assessment = ComplexityAssessment(
            complexity=TaskComplexity.COMPLEX,
            confidence=0.85,
            reasoning="Task requires detailed analysis",
            estimated_tokens=500,
        )
        assert assessment.complexity == TaskComplexity.COMPLEX
        assert assessment.confidence == 0.85
        assert "analysis" in assessment.reasoning
        assert assessment.estimated_tokens == 500

    def test_assessment_defaults(self) -> None:
        """Test assessment default values."""
        assessment = ComplexityAssessment(
            complexity=TaskComplexity.SIMPLE,
            confidence=0.9,
        )
        assert assessment.reasoning == ""
        assert assessment.estimated_tokens == 0


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_config_creation(self) -> None:
        """Test creating model config."""
        config = ModelConfig(
            tier=ModelTier.STANDARD,
            model_name="test-model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            max_context=8192,
            avg_latency_ms=1000,
        )
        assert config.tier == ModelTier.STANDARD
        assert config.model_name == "test-model"
        assert config.cost_per_1k_input == 0.001
        assert config.cost_per_1k_output == 0.002
        assert config.max_context == 8192

    def test_config_defaults(self) -> None:
        """Test config default values."""
        config = ModelConfig(
            tier=ModelTier.LIGHTWEIGHT,
            model_name="small-model",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0002,
        )
        assert config.max_context == 4096
        assert config.avg_latency_ms == 1000


class TestResourceStats:
    """Tests for ResourceStats model."""

    def test_stats_creation(self) -> None:
        """Test creating resource stats."""
        stats = ResourceStats(
            total_operations=100,
            total_tokens=50000,
            total_cost=5.0,
            avg_tokens_per_op=500.0,
            budget_remaining_tokens=50000,
            budget_remaining_cost=5.0,
            operations_by_tier={"standard": 80, "lightweight": 20},
        )
        assert stats.total_operations == 100
        assert stats.total_tokens == 50000
        assert stats.total_cost == 5.0
        assert stats.avg_tokens_per_op == 500.0


class TestResourceBudget:
    """Tests for ResourceBudget dataclass."""

    def test_default_budget(self) -> None:
        """Test default budget values."""
        budget = ResourceBudget()
        assert budget.max_tokens == 100000
        assert budget.max_cost == 10.0
        assert budget.max_operations == 1000
        assert budget.tokens_used == 0
        assert budget.cost_incurred == 0.0

    def test_custom_budget(self) -> None:
        """Test custom budget values."""
        budget = ResourceBudget(
            max_tokens=5000,
            max_cost=1.0,
            max_operations=50,
        )
        assert budget.max_tokens == 5000
        assert budget.max_cost == 1.0
        assert budget.max_operations == 50

    def test_can_afford_within_budget(self) -> None:
        """Test can_afford when within budget."""
        budget = ResourceBudget(max_tokens=1000, max_cost=1.0)
        assert budget.can_afford(500, 0.5) is True

    def test_can_afford_exceeds_tokens(self) -> None:
        """Test can_afford when exceeding token budget."""
        budget = ResourceBudget(max_tokens=1000, max_cost=1.0)
        assert budget.can_afford(1500, 0.1) is False

    def test_can_afford_exceeds_cost(self) -> None:
        """Test can_afford when exceeding cost budget."""
        budget = ResourceBudget(max_tokens=10000, max_cost=1.0)
        assert budget.can_afford(100, 1.5) is False

    def test_can_afford_exceeds_operations(self) -> None:
        """Test can_afford when exceeding operation limit."""
        budget = ResourceBudget(max_operations=1)
        budget.operations_count = 1
        assert budget.can_afford(100, 0.1) is False

    def test_record_usage(self) -> None:
        """Test recording usage."""
        budget = ResourceBudget()
        usage = ResourceUsage(
            operation_id="op_001",
            total_tokens=100,
            estimated_cost=0.01,
            model_tier=ModelTier.STANDARD,
            task_complexity=TaskComplexity.SIMPLE,
        )
        budget.record_usage(usage)

        assert budget.tokens_used == 100
        assert budget.cost_incurred == 0.01
        assert budget.operations_count == 1
        assert len(budget.usage_history) == 1

    def test_get_remaining(self) -> None:
        """Test getting remaining budget."""
        budget = ResourceBudget(
            max_tokens=1000,
            max_cost=1.0,
            max_operations=10,
        )
        budget.tokens_used = 400
        budget.cost_incurred = 0.3
        budget.operations_count = 3

        remaining = budget.get_remaining()
        assert remaining["tokens"] == 600
        assert remaining["cost"] == pytest.approx(0.7)
        assert remaining["operations"] == 7

    def test_get_usage_percent(self) -> None:
        """Test getting usage percentages."""
        budget = ResourceBudget(
            max_tokens=1000,
            max_cost=1.0,
            max_operations=10,
        )
        budget.tokens_used = 500
        budget.cost_incurred = 0.2
        budget.operations_count = 4

        usage = budget.get_usage_percent()
        assert usage["tokens"] == 50.0
        assert usage["cost"] == 20.0
        assert usage["operations"] == 40.0

    def test_get_usage_percent_zero_max(self) -> None:
        """Test usage percent with zero max values."""
        budget = ResourceBudget(
            max_tokens=0,
            max_cost=0,
            max_operations=0,
        )
        usage = budget.get_usage_percent()
        assert usage["tokens"] == 0
        assert usage["cost"] == 0
        assert usage["operations"] == 0

    def test_reset(self) -> None:
        """Test resetting budget."""
        budget = ResourceBudget()
        budget.tokens_used = 500
        budget.cost_incurred = 0.5
        budget.operations_count = 5
        budget.usage_history.append(
            ResourceUsage(
                operation_id="test",
                model_tier=ModelTier.STANDARD,
                task_complexity=TaskComplexity.SIMPLE,
            )
        )

        budget.reset()

        assert budget.tokens_used == 0
        assert budget.cost_incurred == 0.0
        assert budget.operations_count == 0
        assert budget.usage_history == []


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_string(self) -> None:
        """Test token estimate for empty string."""
        assert estimate_tokens("") == 1  # Minimum of 1

    def test_short_string(self) -> None:
        """Test token estimate for short string."""
        # 4 chars = 1 token
        assert estimate_tokens("test") == 1

    def test_longer_string(self) -> None:
        """Test token estimate for longer string."""
        # 100 chars = ~25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_realistic_text(self) -> None:
        """Test token estimate for realistic text."""
        text = "This is a sample sentence for testing token estimation."
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_zero_tokens(self) -> None:
        """Test cost with zero tokens."""
        config = ModelConfig(
            tier=ModelTier.STANDARD,
            model_name="test",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )
        assert estimate_cost(0, 0, config) == 0.0

    def test_typical_usage(self) -> None:
        """Test cost for typical token usage."""
        config = ModelConfig(
            tier=ModelTier.STANDARD,
            model_name="test",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )
        # 1000 input + 500 output
        cost = estimate_cost(1000, 500, config)
        expected = 0.001 + 0.001  # 0.001 for input + 0.001 for output
        assert cost == pytest.approx(expected)


class TestComplexityClassifier:
    """Tests for ComplexityClassifier dataclass."""

    def test_default_classifier(self) -> None:
        """Test default classifier configuration."""
        classifier = ComplexityClassifier()
        assert classifier.simple_threshold == 50
        assert classifier.medium_threshold == 200
        assert "analyze" in classifier.keywords_complex
        assert "what is" in classifier.keywords_simple

    def test_classify_simple_task(self) -> None:
        """Test classifying a simple task."""
        classifier = ComplexityClassifier()
        assessment = classifier.classify("What is Python?")

        assert assessment.complexity == TaskComplexity.SIMPLE
        assert assessment.confidence >= 0.7

    def test_classify_complex_task(self) -> None:
        """Test classifying a complex task."""
        classifier = ComplexityClassifier()
        assessment = classifier.classify(
            "Analyze and compare the pros and cons of different "
            "database systems for high-throughput applications"
        )

        assert assessment.complexity == TaskComplexity.COMPLEX
        assert assessment.confidence >= 0.6

    def test_classify_medium_task(self) -> None:
        """Test classifying a medium complexity task."""
        classifier = ComplexityClassifier()
        assessment = classifier.classify(
            "How do I implement a function to sort a list?"
        )

        # Should be medium (no strong simple/complex indicators)
        assert assessment.complexity in [
            TaskComplexity.MEDIUM,
            TaskComplexity.SIMPLE,
        ]

    def test_classify_by_length(self) -> None:
        """Test that very long tasks are classified as complex."""
        classifier = ComplexityClassifier()
        # Create a long task (>200 tokens estimated)
        long_task = "word " * 250  # ~1250 chars = ~312 tokens
        assessment = classifier.classify(long_task)

        assert assessment.complexity == TaskComplexity.COMPLEX


class TestModelSelector:
    """Tests for ModelSelector dataclass."""

    def test_default_configs(self) -> None:
        """Test default model configurations are created."""
        selector = ModelSelector()
        assert ModelTier.LIGHTWEIGHT in selector.model_configs
        assert ModelTier.STANDARD in selector.model_configs
        assert ModelTier.ADVANCED in selector.model_configs

    def test_select_for_simple_task(self) -> None:
        """Test selecting model for simple task."""
        selector = ModelSelector()
        config = selector.select_for_complexity(TaskComplexity.SIMPLE)

        assert config.tier == ModelTier.LIGHTWEIGHT

    def test_select_for_medium_task(self) -> None:
        """Test selecting model for medium task."""
        selector = ModelSelector()
        config = selector.select_for_complexity(TaskComplexity.MEDIUM)

        assert config.tier == ModelTier.STANDARD

    def test_select_for_complex_task(self) -> None:
        """Test selecting model for complex task."""
        selector = ModelSelector()
        config = selector.select_for_complexity(TaskComplexity.COMPLEX)

        assert config.tier == ModelTier.ADVANCED

    def test_select_with_low_budget_downgrades(self) -> None:
        """Test that low budget forces downgrade to lightweight."""
        selector = ModelSelector()
        budget = ResourceBudget(max_tokens=500, max_cost=0.001)
        budget.tokens_used = 400  # Only 100 tokens left

        config = selector.select_for_complexity(
            TaskComplexity.COMPLEX,
            budget=budget,
        )

        assert config.tier == ModelTier.LIGHTWEIGHT

    def test_get_config(self) -> None:
        """Test getting specific tier config."""
        selector = ModelSelector()
        config = selector.get_config(ModelTier.ADVANCED)

        assert config.tier == ModelTier.ADVANCED


class TestPruneContext:
    """Tests for prune_context function."""

    def test_short_context_unchanged(self) -> None:
        """Test short context is not pruned."""
        context = "This is a short context."
        result = prune_context(context, max_tokens=100)
        assert result == context

    def test_long_context_truncated(self) -> None:
        """Test long context is truncated."""
        context = "word " * 2000  # 10000 chars = ~2500 tokens
        result = prune_context(context, max_tokens=100)

        assert len(result) < len(context)
        assert "[...truncated]" in result

    def test_truncation_finds_period(self) -> None:
        """Test truncation prefers breaking at periods when possible."""
        context = "First sentence. Second sentence. Third sentence."
        # Use a larger token budget so period finding works
        result = prune_context(context, max_tokens=8)

        # Result should be truncated
        assert len(result) < len(context)
        # Should have truncation marker
        assert "[...truncated]" in result


class TestResourceAwareExecutor:
    """Tests for ResourceAwareExecutor dataclass."""

    def test_executor_initialization(self) -> None:
        """Test executor initialization."""
        budget = ResourceBudget()
        executor = ResourceAwareExecutor(budget=budget)

        assert executor.budget == budget
        assert executor.operation_counter == 0
        assert executor.use_llm_classification is False

    def test_generate_operation_id(self) -> None:
        """Test operation ID generation."""
        budget = ResourceBudget()
        executor = ResourceAwareExecutor(budget=budget)

        id1 = executor._generate_operation_id()
        id2 = executor._generate_operation_id()

        assert id1 == "op_0001"
        assert id2 == "op_0002"

    @pytest.mark.asyncio
    async def test_classify_task_heuristic(self) -> None:
        """Test task classification using heuristics."""
        budget = ResourceBudget()
        executor = ResourceAwareExecutor(budget=budget)

        assessment = await executor.classify_task("What is 2 + 2?")

        assert assessment.complexity == TaskComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_run_basic(self) -> None:
        """Test basic task execution."""
        budget = ResourceBudget(max_tokens=10000, max_cost=1.0)
        executor = ResourceAwareExecutor(budget=budget)

        mock_result = MagicMock()
        mock_result.output = "The answer is 4."

        with patch("agentic_patterns.resource_aware.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result, usage = await executor.run("What is 2 + 2?")

        assert result == "The answer is 4."
        assert usage.operation_id == "op_0001"
        assert usage.total_tokens > 0
        assert budget.operations_count == 1

    @pytest.mark.asyncio
    async def test_run_insufficient_budget(self) -> None:
        """Test execution with insufficient budget."""
        budget = ResourceBudget(max_tokens=1, max_cost=0.0001)
        executor = ResourceAwareExecutor(budget=budget)

        result, usage = await executor.run("What is Python?")

        assert "Insufficient budget" in result
        assert usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_run_with_forced_tier(self) -> None:
        """Test execution with forced model tier."""
        budget = ResourceBudget()
        executor = ResourceAwareExecutor(budget=budget)

        mock_result = MagicMock()
        mock_result.output = "Response"

        with patch("agentic_patterns.resource_aware.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result, usage = await executor.run(
                "Simple question",
                force_tier=ModelTier.ADVANCED,
            )

        assert usage.model_tier == ModelTier.ADVANCED

    @pytest.mark.asyncio
    async def test_run_batch(self) -> None:
        """Test batch execution."""
        budget = ResourceBudget(max_tokens=10000)
        executor = ResourceAwareExecutor(budget=budget)

        mock_result = MagicMock()
        mock_result.output = "Answer"

        with patch("agentic_patterns.resource_aware.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            results = await executor.run_batch(
                [
                    "Question 1",
                    "Question 2",
                    "Question 3",
                ]
            )

        assert len(results) == 3
        assert budget.operations_count == 3

    @pytest.mark.asyncio
    async def test_run_batch_stops_on_budget(self) -> None:
        """Test batch execution stops when budget exhausted."""
        budget = ResourceBudget(max_tokens=50)  # Very low
        executor = ResourceAwareExecutor(budget=budget)

        mock_result = MagicMock()
        mock_result.output = "Answer " * 20  # Uses ~20 tokens output

        with patch("agentic_patterns.resource_aware.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            results = await executor.run_batch(
                ["Q1", "Q2", "Q3", "Q4", "Q5"],
                stop_on_budget=True,
            )

        # Should stop before completing all 5
        assert len(results) < 5

    def test_get_stats(self) -> None:
        """Test getting executor statistics."""
        budget = ResourceBudget()
        executor = ResourceAwareExecutor(budget=budget)

        # Add some usage
        usage1 = ResourceUsage(
            operation_id="op_001",
            total_tokens=100,
            estimated_cost=0.01,
            model_tier=ModelTier.STANDARD,
            task_complexity=TaskComplexity.MEDIUM,
        )
        usage2 = ResourceUsage(
            operation_id="op_002",
            total_tokens=50,
            estimated_cost=0.005,
            model_tier=ModelTier.LIGHTWEIGHT,
            task_complexity=TaskComplexity.SIMPLE,
        )
        budget.record_usage(usage1)
        budget.record_usage(usage2)

        stats = executor.get_stats()

        assert stats.total_operations == 2
        assert stats.total_tokens == 150
        assert stats.total_cost == pytest.approx(0.015)
        assert stats.avg_tokens_per_op == 75.0
        assert stats.operations_by_tier["standard"] == 1
        assert stats.operations_by_tier["lightweight"] == 1


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complexity_based_routing(self) -> None:
        """Test that tasks are routed based on complexity."""
        budget = ResourceBudget(max_tokens=10000)
        executor = ResourceAwareExecutor(budget=budget)

        mock_result = MagicMock()
        mock_result.output = "Response"

        with patch("agentic_patterns.resource_aware.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            # Simple task
            _, usage1 = await executor.run("Define Python")
            assert usage1.model_tier == ModelTier.LIGHTWEIGHT

            # Complex task
            _, usage2 = await executor.run(
                "Analyze and compare microservices vs monolithic "
                "architecture patterns in detail"
            )
            assert usage2.model_tier == ModelTier.ADVANCED

    @pytest.mark.asyncio
    async def test_budget_tracking_accuracy(self) -> None:
        """Test budget tracking across multiple operations."""
        budget = ResourceBudget(max_tokens=10000, max_cost=1.0)
        executor = ResourceAwareExecutor(budget=budget)

        mock_result = MagicMock()
        mock_result.output = "Short answer"

        with patch("agentic_patterns.resource_aware.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            # Execute multiple tasks
            for i in range(5):
                await executor.run(f"Question {i}")

        # Verify budget tracking
        assert budget.operations_count == 5
        assert budget.tokens_used > 0
        assert budget.cost_incurred > 0

        # Stats should match
        stats = executor.get_stats()
        assert stats.total_operations == 5
        assert stats.total_tokens == budget.tokens_used
