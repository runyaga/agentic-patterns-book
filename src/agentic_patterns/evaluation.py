"""
Evaluation module using pydantic-evals.

This module provides evaluation capabilities for agentic patterns using
the pydantic-evals framework with Logfire integration.

Key components:
- Dataset factories for each pattern (routing, multi_agent)
- Built-in evaluators: LLMJudge, IsInstance, Contains, EqualsExpected
- Logfire integration for observability

Example usage:
    from agentic_patterns.evaluation import evaluate_pattern
    from agentic_patterns.routing import route_query

    report = await evaluate_pattern("routing", route_query)
    report.print(include_input=True, include_output=True)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from typing import TypeVar

import logfire
from pydantic_evals import Case
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Contains
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.evaluators import IsInstance
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.reporting import EvaluationReport

from agentic_patterns._models import get_model

# Type variables for generic evaluators
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# --8<-- [start:config]
def configure_logfire_for_evals(
    service_name: str = "agentic-patterns-evals",
    environment: str = "development",
) -> None:
    """
    Configure Logfire for evaluation runs.

    Disables scrubbing to capture full evaluation data.

    Args:
        service_name: Service name for Logfire.
        environment: Environment name (development, staging, production).
    """
    logfire.configure(
        scrubbing=False,  # Disable scrubbing for full eval visibility
        send_to_logfire="if-token-present",
        service_name=service_name,
        environment=environment,
    )
    logfire.instrument_pydantic_ai()


# --8<-- [end:config]


# --8<-- [start:evaluators]
class IntentMatchEvaluator(Evaluator[str, tuple]):
    """
    Check if routing decision matches expected intent.

    Works with route_query() output: tuple[RouteDecision, RouteResponse].
    """

    def __init__(self, expected_intent: str) -> None:
        self.expected_intent = expected_intent

    def evaluate(self, ctx: EvaluatorContext[str, tuple]) -> float:
        """Check if the routing decision intent matches expected."""
        if not ctx.output or len(ctx.output) < 1:
            return 0.0

        decision = ctx.output[0]
        if hasattr(decision, "intent"):
            actual = decision.intent.value
            return 1.0 if actual == self.expected_intent else 0.0
        return 0.0


class ConfidenceThresholdEvaluator(Evaluator[str, tuple]):
    """Check if routing confidence meets threshold."""

    def __init__(self, min_confidence: float = 0.7) -> None:
        self.min_confidence = min_confidence

    def evaluate(self, ctx: EvaluatorContext[str, tuple]) -> float:
        """Return confidence score if above threshold, else 0."""
        if not ctx.output or len(ctx.output) < 1:
            return 0.0

        decision = ctx.output[0]
        if hasattr(decision, "confidence"):
            conf = decision.confidence
            return conf if conf >= self.min_confidence else 0.0
        return 0.0


class CollaborationSuccessEvaluator(Evaluator[str, Any]):
    """Check if multi-agent collaboration succeeded."""

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Return 1.0 if collaboration succeeded, else 0.0."""
        if ctx.output is None:
            return 0.0

        if hasattr(ctx.output, "success"):
            return 1.0 if ctx.output.success else 0.0
        return 0.0


class TaskCompletionEvaluator(Evaluator[str, Any]):
    """Evaluate task completion ratio in multi-agent collaboration."""

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Return ratio of successful tasks."""
        if ctx.output is None:
            return 0.0

        if hasattr(ctx.output, "task_results"):
            results = ctx.output.task_results
            if not results:
                return 0.0
            successful = sum(1 for r in results if r.success)
            return successful / len(results)
        return 0.0


# --8<-- [end:evaluators]


# --8<-- [start:datasets]
def create_routing_dataset(
    include_llm_judge: bool = False,
    llm_judge_model: str | None = None,
) -> Dataset:
    """
    Create evaluation dataset for routing pattern.

    Args:
        include_llm_judge: Whether to include LLM-as-judge evaluation.
        llm_judge_model: Model to use for LLM judge (default: project model).

    Returns:
        Dataset with routing test cases and evaluators.
    """
    cases = [
        Case(
            name="order_status_query",
            inputs="Where is my order ORD-12345?",
            expected_output={"intent": "order_status"},
            metadata={"category": "orders", "difficulty": "easy"},
        ),
        Case(
            name="product_info_query",
            inputs="What features does the Pro model have?",
            expected_output={"intent": "product_info"},
            metadata={"category": "products", "difficulty": "easy"},
        ),
        Case(
            name="technical_support_query",
            inputs="My device won't turn on after the update.",
            expected_output={"intent": "technical_support"},
            metadata={"category": "support", "difficulty": "medium"},
        ),
        Case(
            name="ambiguous_query",
            inputs="Hello, I need help with something.",
            expected_output={"intent": "clarification"},
            metadata={"category": "unclear", "difficulty": "hard"},
        ),
    ]

    evaluators: list[Evaluator] = [
        IsInstance(type_name="tuple"),
        ConfidenceThresholdEvaluator(min_confidence=0.5),
    ]

    if include_llm_judge:
        model = llm_judge_model or str(get_model())
        evaluators.append(
            LLMJudge(
                rubric="Response correctly identifies user intent and "
                "provides helpful, relevant information.",
                model=model,
                include_input=True,
            )
        )

    return Dataset(cases=cases, evaluators=evaluators)


def create_multi_agent_dataset(
    include_llm_judge: bool = False,
    llm_judge_model: str | None = None,
) -> Dataset:
    """
    Create evaluation dataset for multi-agent collaboration pattern.

    Args:
        include_llm_judge: Whether to include LLM-as-judge evaluation.
        llm_judge_model: Model to use for LLM judge (default: project model).

    Returns:
        Dataset with multi-agent test cases and evaluators.
    """
    cases = [
        Case(
            name="research_task",
            inputs="Research Python async patterns and summarize the key "
            "benefits for developers new to async programming.",
            metadata={"complexity": "medium", "expected_agents": 3},
        ),
        Case(
            name="analysis_task",
            inputs="Analyze the pros and cons of microservices architecture "
            "compared to monolithic applications.",
            metadata={"complexity": "high", "expected_agents": 4},
        ),
    ]

    evaluators: list[Evaluator] = [
        CollaborationSuccessEvaluator(),
        TaskCompletionEvaluator(),
    ]

    if include_llm_judge:
        model = llm_judge_model or str(get_model())
        evaluators.append(
            LLMJudge(
                rubric="The collaboration produced a comprehensive, "
                "well-structured response that addresses the objective.",
                model=model,
                include_input=True,
            )
        )

    return Dataset(cases=cases, evaluators=evaluators)


PATTERN_DATASETS = {
    "routing": create_routing_dataset,
    "multi_agent": create_multi_agent_dataset,
}
# --8<-- [end:datasets]


# --8<-- [start:api]
async def evaluate_pattern(
    pattern_name: str,
    task_function: Callable,
    *,
    include_llm_judge: bool = False,
    llm_judge_model: str | None = None,
    custom_cases: list[Case] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """
    Evaluate a pattern implementation.

    Args:
        pattern_name: Name of pattern ("routing", "multi_agent").
        task_function: Async function to evaluate.
        include_llm_judge: Whether to include LLM-as-judge evaluation.
        llm_judge_model: Model for LLM judge (uses project default if None).
        custom_cases: Optional custom test cases to use instead.
        metadata: Optional metadata to include in report.

    Returns:
        EvaluationReport with scores and details.

    Raises:
        ValueError: If pattern_name is not recognized.
    """
    if pattern_name not in PATTERN_DATASETS:
        valid = ", ".join(PATTERN_DATASETS.keys())
        raise ValueError(f"Unknown pattern: {pattern_name}. Valid: {valid}")

    dataset_factory = PATTERN_DATASETS[pattern_name]
    dataset = dataset_factory(
        include_llm_judge=include_llm_judge,
        llm_judge_model=llm_judge_model,
    )

    if custom_cases:
        dataset = Dataset(cases=custom_cases, evaluators=dataset.evaluators)

    report_metadata = {
        "pattern": pattern_name,
        "timestamp": datetime.now().isoformat(),
        **(metadata or {}),
    }

    return await dataset.evaluate(task_function, metadata=report_metadata)


def evaluate_pattern_sync(
    pattern_name: str,
    task_function: Callable,
    *,
    include_llm_judge: bool = False,
    llm_judge_model: str | None = None,
    custom_cases: list[Case] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """
    Synchronous version of evaluate_pattern.

    See evaluate_pattern for full documentation.
    """
    import asyncio

    return asyncio.run(
        evaluate_pattern(
            pattern_name,
            task_function,
            include_llm_judge=include_llm_judge,
            llm_judge_model=llm_judge_model,
            custom_cases=custom_cases,
            metadata=metadata,
        )
    )


# --8<-- [end:api]


# Re-export pydantic-evals types for convenience
__all__ = [
    # Core pydantic-evals types
    "Case",
    "Dataset",
    "EvaluationReport",
    "Evaluator",
    "EvaluatorContext",
    # Built-in evaluators
    "LLMJudge",
    "IsInstance",
    "Contains",
    # Custom evaluators
    "IntentMatchEvaluator",
    "ConfidenceThresholdEvaluator",
    "CollaborationSuccessEvaluator",
    "TaskCompletionEvaluator",
    # Dataset factories
    "create_routing_dataset",
    "create_multi_agent_dataset",
    "PATTERN_DATASETS",
    # Main API
    "evaluate_pattern",
    "evaluate_pattern_sync",
    "configure_logfire_for_evals",
]


if __name__ == "__main__":
    import asyncio

    async def demo() -> None:
        """Demonstrate evaluation capabilities."""
        print("=" * 60)
        print("Evaluation Demo with pydantic-evals")
        print("=" * 60)

        # Configure logfire
        configure_logfire_for_evals()

        # Create a mock routing function for demo
        async def mock_route_query(query: str) -> tuple:
            from pydantic import BaseModel

            class MockDecision(BaseModel):
                intent: str = "order_status"
                confidence: float = 0.95
                reasoning: str = "User asked about order"

            class MockResponse(BaseModel):
                message: str = "Your order is on the way"

            return MockDecision(), MockResponse()

        # Run evaluation
        print("\n--- Routing Pattern Evaluation ---")
        report = await evaluate_pattern(
            "routing",
            mock_route_query,
            metadata={"model": "mock", "version": "demo"},
        )

        report.print(include_input=True, include_output=True)

        print("\n--- Programmatic Access ---")
        for case in report.cases:
            print(f"Case: {case.name}")
            print(f"  Scores: {case.scores}")
            print(f"  Assertions: {case.assertions}")

    asyncio.run(demo())
