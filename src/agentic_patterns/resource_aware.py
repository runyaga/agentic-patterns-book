"""
Resource-Aware Optimization Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 16:
Manage computational, temporal, and financial resources dynamically.

Key concepts:
- Dynamic Model Switching: Select models based on task complexity
- Token Budget Management: Track and manage token usage
- Task Complexity Assessment: Classify tasks for routing
- Contextual Pruning: Minimize prompt token count
- Resource Tracking: Monitor costs and usage

This module implements:
- ResourceBudget: Track and manage resource constraints
- ComplexityClassifier: Assess task complexity
- ModelSelector: Choose appropriate model for task
- ResourceAwareExecutor: Execute with resource optimization

Example usage:
    budget = ResourceBudget(max_tokens=10000, max_cost=1.0)
    executor = ResourceAwareExecutor(budget)
    result = await executor.run("Simple question")
"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from agentic_patterns._models import get_model


class TaskComplexity(str, Enum):
    """Complexity level of a task."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ModelTier(str, Enum):
    """Model tier based on capability and cost."""

    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    ADVANCED = "advanced"


class ResourceType(str, Enum):
    """Type of resource being tracked."""

    TOKENS = "tokens"
    COST = "cost"
    TIME = "time"
    CALLS = "calls"


class ResourceUsage(BaseModel):
    """Record of resource usage for a single operation."""

    operation_id: str = Field(description="Unique operation identifier")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")
    estimated_cost: float = Field(default=0.0, description="Estimated cost")
    execution_time_ms: int = Field(default=0, description="Execution time")
    model_tier: ModelTier = Field(description="Model tier used")
    task_complexity: TaskComplexity = Field(description="Task complexity")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When operation occurred",
    )


class ComplexityAssessment(BaseModel):
    """Assessment of task complexity."""

    complexity: TaskComplexity = Field(description="Determined complexity")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in assessment",
    )
    reasoning: str = Field(default="", description="Assessment reasoning")
    estimated_tokens: int = Field(default=0, description="Estimated tokens")


class ModelConfig(BaseModel):
    """Configuration for a model tier."""

    tier: ModelTier = Field(description="Model tier")
    model_name: str = Field(description="Model identifier")
    cost_per_1k_input: float = Field(description="Cost per 1K input tokens")
    cost_per_1k_output: float = Field(description="Cost per 1K output tokens")
    max_context: int = Field(default=4096, description="Max context length")
    avg_latency_ms: int = Field(default=1000, description="Average latency")


class ResourceStats(BaseModel):
    """Statistics about resource usage."""

    total_operations: int = Field(description="Total operations performed")
    total_tokens: int = Field(description="Total tokens consumed")
    total_cost: float = Field(description="Total cost incurred")
    avg_tokens_per_op: float = Field(description="Average tokens per op")
    budget_remaining_tokens: int = Field(description="Remaining token budget")
    budget_remaining_cost: float = Field(description="Remaining cost budget")
    operations_by_tier: dict[str, int] = Field(
        default_factory=dict,
        description="Operations count by model tier",
    )


@dataclass
class ResourceBudget:
    """
    Resource budget tracker.

    Manages constraints on tokens, cost, and operations.
    """

    max_tokens: int = 100000
    max_cost: float = 10.0
    max_operations: int = 1000
    tokens_used: int = 0
    cost_incurred: float = 0.0
    operations_count: int = 0
    usage_history: list[ResourceUsage] = field(default_factory=list)

    def can_afford(
        self,
        estimated_tokens: int,
        estimated_cost: float = 0.0,
    ) -> bool:
        """
        Check if budget allows an operation.

        Args:
            estimated_tokens: Estimated tokens for operation.
            estimated_cost: Estimated cost for operation.

        Returns:
            True if operation is within budget.
        """
        token_ok = (self.tokens_used + estimated_tokens) <= self.max_tokens
        cost_ok = (self.cost_incurred + estimated_cost) <= self.max_cost
        ops_ok = self.operations_count < self.max_operations
        return token_ok and cost_ok and ops_ok

    def record_usage(
        self,
        usage: ResourceUsage,
    ) -> None:
        """
        Record resource usage.

        Args:
            usage: Usage record to add.
        """
        self.tokens_used += usage.total_tokens
        self.cost_incurred += usage.estimated_cost
        self.operations_count += 1
        self.usage_history.append(usage)

    def get_remaining(self) -> dict[str, float]:
        """Get remaining budget amounts."""
        return {
            "tokens": self.max_tokens - self.tokens_used,
            "cost": self.max_cost - self.cost_incurred,
            "operations": self.max_operations - self.operations_count,
        }

    def get_usage_percent(self) -> dict[str, float]:
        """Get budget usage as percentages."""
        return {
            "tokens": (self.tokens_used / self.max_tokens * 100)
            if self.max_tokens > 0
            else 0,
            "cost": (self.cost_incurred / self.max_cost * 100)
            if self.max_cost > 0
            else 0,
            "operations": (self.operations_count / self.max_operations * 100)
            if self.max_operations > 0
            else 0,
        }

    def reset(self) -> None:
        """Reset budget counters."""
        self.tokens_used = 0
        self.cost_incurred = 0.0
        self.operations_count = 0
        self.usage_history = []


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Approximation: ~4 characters per token for English.

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    config: ModelConfig,
) -> float:
    """
    Estimate cost for an operation.

    Args:
        input_tokens: Input token count.
        output_tokens: Output token count.
        config: Model configuration.

    Returns:
        Estimated cost in dollars.
    """
    input_cost = (input_tokens / 1000) * config.cost_per_1k_input
    output_cost = (output_tokens / 1000) * config.cost_per_1k_output
    return input_cost + output_cost


@dataclass
class ComplexityClassifier:
    """
    Classifier for task complexity.

    Uses heuristics and optional LLM assessment.
    """

    simple_threshold: int = 50
    medium_threshold: int = 200
    keywords_complex: list[str] = field(
        default_factory=lambda: [
            "analyze",
            "compare",
            "evaluate",
            "synthesize",
            "design",
            "explain in detail",
            "comprehensive",
            "step by step",
        ]
    )
    keywords_simple: list[str] = field(
        default_factory=lambda: [
            "what is",
            "define",
            "list",
            "name",
            "when",
            "where",
            "yes or no",
            "true or false",
        ]
    )

    def classify(self, task: str) -> ComplexityAssessment:
        """
        Classify task complexity.

        Args:
            task: Task description to classify.

        Returns:
            ComplexityAssessment with determined complexity.
        """
        task_lower = task.lower()
        token_estimate = estimate_tokens(task)

        # Check for simple keywords
        has_simple_keywords = any(
            kw in task_lower for kw in self.keywords_simple
        )

        # Check for complex keywords
        has_complex_keywords = any(
            kw in task_lower for kw in self.keywords_complex
        )

        # Determine complexity
        if has_complex_keywords or token_estimate > self.medium_threshold:
            complexity = TaskComplexity.COMPLEX
            confidence = 0.8 if has_complex_keywords else 0.6
            reasoning = "Task contains complex indicators"
        elif has_simple_keywords or token_estimate < self.simple_threshold:
            complexity = TaskComplexity.SIMPLE
            confidence = 0.9 if has_simple_keywords else 0.7
            reasoning = "Task appears straightforward"
        else:
            complexity = TaskComplexity.MEDIUM
            confidence = 0.7
            reasoning = "Task has moderate complexity"

        return ComplexityAssessment(
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            estimated_tokens=token_estimate * 3,  # Estimate response tokens
        )


@dataclass
class ModelSelector:
    """
    Select appropriate model based on task and resources.

    Maps complexity to model tiers with fallback logic.
    """

    model_configs: dict[ModelTier, ModelConfig] = field(default_factory=dict)
    default_tier: ModelTier = ModelTier.STANDARD

    def __post_init__(self) -> None:
        """Initialize default model configs if not provided."""
        if not self.model_configs:
            self.model_configs = {
                ModelTier.LIGHTWEIGHT: ModelConfig(
                    tier=ModelTier.LIGHTWEIGHT,
                    model_name="gpt-oss:7b",
                    cost_per_1k_input=0.0001,
                    cost_per_1k_output=0.0002,
                    max_context=4096,
                    avg_latency_ms=500,
                ),
                ModelTier.STANDARD: ModelConfig(
                    tier=ModelTier.STANDARD,
                    model_name="gpt-oss:20b",
                    cost_per_1k_input=0.0005,
                    cost_per_1k_output=0.0015,
                    max_context=8192,
                    avg_latency_ms=1000,
                ),
                ModelTier.ADVANCED: ModelConfig(
                    tier=ModelTier.ADVANCED,
                    model_name="gpt-oss:70b",
                    cost_per_1k_input=0.001,
                    cost_per_1k_output=0.003,
                    max_context=16384,
                    avg_latency_ms=2000,
                ),
            }

    def select_for_complexity(
        self,
        complexity: TaskComplexity,
        budget: ResourceBudget | None = None,
    ) -> ModelConfig:
        """
        Select model based on complexity and budget.

        Args:
            complexity: Task complexity level.
            budget: Optional resource budget for constraints.

        Returns:
            Selected ModelConfig.
        """
        # Map complexity to preferred tier
        tier_map = {
            TaskComplexity.SIMPLE: ModelTier.LIGHTWEIGHT,
            TaskComplexity.MEDIUM: ModelTier.STANDARD,
            TaskComplexity.COMPLEX: ModelTier.ADVANCED,
        }
        preferred_tier = tier_map.get(complexity, self.default_tier)

        # Check budget constraints if provided
        if budget:
            remaining = budget.get_remaining()
            # If budget is low, downgrade tier
            if remaining["tokens"] < 1000 or remaining["cost"] < 0.01:
                preferred_tier = ModelTier.LIGHTWEIGHT

        return self.model_configs.get(
            preferred_tier,
            self.model_configs[self.default_tier],
        )

    def get_config(self, tier: ModelTier) -> ModelConfig:
        """Get config for a specific tier."""
        return self.model_configs.get(
            tier,
            self.model_configs[self.default_tier],
        )


# Initialize default model for agents
model = get_model()

# Complexity assessment agent
complexity_agent = Agent(
    model,
    system_prompt=(
        "You are a task complexity analyzer. Assess the complexity of "
        "tasks as SIMPLE, MEDIUM, or COMPLEX. Consider factors like: "
        "required reasoning depth, information needed, output length, "
        "and domain expertise required."
    ),
    output_type=ComplexityAssessment,
)

# Task execution agent (uses selected model in practice)
task_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful assistant. Complete the given task efficiently. "
        "Be concise but thorough. Adapt your response length to the "
        "complexity of the question."
    ),
    output_type=str,
)


@dataclass
class ResourceAwareExecutor:
    """
    Execute tasks with resource-aware optimization.

    Manages model selection, budget tracking, and execution.
    """

    budget: ResourceBudget
    classifier: ComplexityClassifier = field(
        default_factory=ComplexityClassifier
    )
    selector: ModelSelector = field(default_factory=ModelSelector)
    operation_counter: int = 0
    use_llm_classification: bool = False

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        self.operation_counter += 1
        return f"op_{self.operation_counter:04d}"

    async def classify_task(
        self,
        task: str,
    ) -> ComplexityAssessment:
        """
        Classify task complexity.

        Args:
            task: Task to classify.

        Returns:
            ComplexityAssessment.
        """
        if self.use_llm_classification:
            result = await complexity_agent.run(
                f"Assess the complexity of this task: {task}"
            )
            return result.output
        else:
            return self.classifier.classify(task)

    async def run(
        self,
        task: str,
        force_tier: ModelTier | None = None,
    ) -> tuple[str, ResourceUsage]:
        """
        Execute a task with resource optimization.

        Args:
            task: Task to execute.
            force_tier: Optional tier to force (bypasses selection).

        Returns:
            Tuple of (result string, resource usage record).
        """
        op_id = self._generate_operation_id()
        start_time = datetime.now()

        # Classify complexity
        assessment = await self.classify_task(task)

        # Select model
        if force_tier:
            config = self.selector.get_config(force_tier)
        else:
            config = self.selector.select_for_complexity(
                assessment.complexity,
                self.budget,
            )

        # Estimate tokens
        input_tokens = estimate_tokens(task)
        estimated_output = assessment.estimated_tokens

        # Check budget
        est_cost = estimate_cost(input_tokens, estimated_output, config)
        if not self.budget.can_afford(
            input_tokens + estimated_output, est_cost
        ):
            return (
                "Error: Insufficient budget for this operation",
                ResourceUsage(
                    operation_id=op_id,
                    model_tier=config.tier,
                    task_complexity=assessment.complexity,
                ),
            )

        # Execute task
        result = await task_agent.run(task)
        output = result.output

        # Calculate actual usage
        end_time = datetime.now()
        output_tokens = estimate_tokens(output)
        total_tokens = input_tokens + output_tokens
        actual_cost = estimate_cost(input_tokens, output_tokens, config)
        exec_time = int((end_time - start_time).total_seconds() * 1000)

        # Record usage
        usage = ResourceUsage(
            operation_id=op_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=actual_cost,
            execution_time_ms=exec_time,
            model_tier=config.tier,
            task_complexity=assessment.complexity,
        )
        self.budget.record_usage(usage)

        return output, usage

    async def run_batch(
        self,
        tasks: list[str],
        stop_on_budget: bool = True,
    ) -> list[tuple[str, ResourceUsage]]:
        """
        Execute multiple tasks.

        Args:
            tasks: List of tasks to execute.
            stop_on_budget: Stop if budget exhausted.

        Returns:
            List of (result, usage) tuples.
        """
        results = []
        for task in tasks:
            remaining = self.budget.get_remaining()
            if stop_on_budget and remaining["tokens"] <= 0:
                break
            result = await self.run(task)
            results.append(result)
        return results

    def get_stats(self) -> ResourceStats:
        """Get resource usage statistics."""
        ops_by_tier: dict[str, int] = {}
        for usage in self.budget.usage_history:
            tier = usage.model_tier.value
            ops_by_tier[tier] = ops_by_tier.get(tier, 0) + 1

        avg_tokens = 0.0
        if self.budget.operations_count > 0:
            avg_tokens = self.budget.tokens_used / self.budget.operations_count

        remaining = self.budget.get_remaining()

        return ResourceStats(
            total_operations=self.budget.operations_count,
            total_tokens=self.budget.tokens_used,
            total_cost=self.budget.cost_incurred,
            avg_tokens_per_op=avg_tokens,
            budget_remaining_tokens=int(remaining["tokens"]),
            budget_remaining_cost=remaining["cost"],
            operations_by_tier=ops_by_tier,
        )


def prune_context(
    context: str,
    max_tokens: int = 2000,
) -> str:
    """
    Prune context to fit within token budget.

    Args:
        context: Context text to prune.
        max_tokens: Maximum tokens allowed.

    Returns:
        Pruned context string.
    """
    current_tokens = estimate_tokens(context)
    if current_tokens <= max_tokens:
        return context

    # Calculate target character length
    target_chars = max_tokens * 4  # Approx 4 chars per token

    # Simple truncation with indicator
    if len(context) > target_chars:
        truncated = context[: target_chars - 20]
        # Find a good break point
        last_period = truncated.rfind(".")
        if last_period > target_chars // 2:
            truncated = truncated[: last_period + 1]
        return truncated + "\n[...truncated]"

    return context


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Resource-Aware Optimization Pattern")
        print("=" * 60)

        # Create budget and executor
        budget = ResourceBudget(
            max_tokens=5000,
            max_cost=1.0,
            max_operations=10,
        )
        executor = ResourceAwareExecutor(budget=budget)

        # Test tasks of varying complexity
        tasks = [
            "What is 2 + 2?",
            "Explain the concept of recursion in programming.",
            "Analyze the pros and cons of microservices architecture",
        ]

        print("\n--- Executing Tasks ---")
        for task in tasks:
            print(f"\nTask: {task[:50]}...")

            # Classify first
            assessment = await executor.classify_task(task)
            print(f"  Complexity: {assessment.complexity.value}")

            # Execute
            result, usage = await executor.run(task)
            print(f"  Model tier: {usage.model_tier.value}")
            print(f"  Tokens: {usage.total_tokens}")
            print(f"  Cost: ${usage.estimated_cost:.4f}")
            print(f"  Result: {result[:80]}...")

        # Show stats
        stats = executor.get_stats()
        print("\n--- Resource Statistics ---")
        print(f"Total operations: {stats.total_operations}")
        print(f"Total tokens: {stats.total_tokens}")
        print(f"Total cost: ${stats.total_cost:.4f}")
        print(f"Remaining tokens: {stats.budget_remaining_tokens}")
        print(f"Remaining cost: ${stats.budget_remaining_cost:.4f}")
        print(f"Operations by tier: {stats.operations_by_tier}")

    asyncio.run(main())
