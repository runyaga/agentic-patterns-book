# Spec 03: Evaluation Refactoring to pydantic-evals

**Status**: DRAFT
**Decision**: REFACTOR - replace custom evaluation with pydantic-evals
**Priority**: P0 (do first - establishes evaluation foundation)
**Complexity**: Medium

---

## 1. Implementation Details

### 1.1 Overview

Refactor the evaluation module to use `pydantic-evals` as the core evaluation framework. This replaces custom in-memory tracking with a standardized, extensible evaluation system that integrates with Logfire.

### 1.2 File Changes

```
src/agentic_patterns/
├── evaluation.py           # REPLACE with pydantic-evals based implementation
└── evaluators/             # NEW - custom evaluators directory
    ├── __init__.py
    ├── llm_judge.py        # LLMJudge as pydantic-evals Evaluator
    ├── trajectory.py       # TrajectoryEvaluator as pydantic-evals Evaluator
    └── drift.py            # DriftEvaluator for performance drift
```

### 1.3 pydantic-evals Core Concepts

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

# Cases define test inputs and expected outputs
case = Case(
    name="order_query",
    inputs="Where is my order ORD-123?",
    expected_output={"intent": "order_status"},
    metadata={"difficulty": "easy"},
)

# Datasets group cases with evaluators
dataset = Dataset(
    cases=[case1, case2, ...],
    evaluators=[MyEvaluator(), AnotherEvaluator()],
)

# Run evaluation
report = await dataset.evaluate(my_agent_function)
report.print()
```

### 1.4 Custom Evaluators

#### LLMJudgeEvaluator

```python
# evaluators/llm_judge.py

from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_ai import Agent

@dataclass
class LLMJudgeEvaluator(Evaluator[str, str]):
    """
    Use LLM-as-Judge to evaluate response quality.

    Scores quality, helpfulness, and accuracy on 0-1 scale.
    """

    quality_threshold: float = 0.7
    model_name: str = "gpt-oss:20b"

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        """Synchronous evaluation - returns quality score."""
        # For sync, we need to run async in event loop
        import asyncio
        return asyncio.run(self._evaluate_async(ctx))

    async def _evaluate_async(self, ctx: EvaluatorContext[str, str]) -> float:
        """Async LLM evaluation."""
        judge_agent = Agent(
            get_model(self.model_name),
            output_type=JudgmentResult,
            system_prompt=self._get_system_prompt(),
        )

        prompt = f"Query: {ctx.inputs}\n\nResponse: {ctx.output}"
        if ctx.expected_output:
            prompt += f"\n\nExpected: {ctx.expected_output}"

        result = await judge_agent.run(prompt)
        return result.output.quality_score

    def _get_system_prompt(self) -> str:
        return """You are an expert evaluator. Score the response:
        - quality_score (0-1): Overall response quality
        - helpfulness_score (0-1): How helpful is the response
        - accuracy_score (0-1): Factual correctness
        Provide reasoning and improvement suggestions."""


class JudgmentResult(BaseModel):
    """Result from LLM judge evaluation."""

    quality_score: float = Field(ge=0.0, le=1.0)
    helpfulness_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    improvement_suggestions: list[str] = Field(default_factory=list)
```

#### TrajectoryEvaluator

```python
# evaluators/trajectory.py

from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class TrajectoryMatchEvaluator(Evaluator):
    """
    Evaluate if agent took expected action sequence.

    Supports exact, in-order, and any-order matching.
    """

    expected_actions: list[str]
    match_type: str = "in_order"  # exact, in_order, any_order

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Check if output contains expected trajectory."""
        actual = self._extract_actions(ctx.output)

        if self.match_type == "exact":
            return 1.0 if actual == self.expected_actions else 0.0

        elif self.match_type == "in_order":
            return self._in_order_score(actual, self.expected_actions)

        else:  # any_order
            return self._any_order_score(actual, self.expected_actions)

    def _extract_actions(self, output) -> list[str]:
        """Extract action list from output."""
        if hasattr(output, "actions"):
            return output.actions
        if isinstance(output, list):
            return [str(a) for a in output]
        return []

    def _in_order_score(self, actual: list[str], expected: list[str]) -> float:
        matched = 0
        exp_idx = 0
        for action in actual:
            if exp_idx < len(expected) and action == expected[exp_idx]:
                matched += 1
                exp_idx += 1
        return matched / len(expected) if expected else 1.0

    def _any_order_score(self, actual: list[str], expected: list[str]) -> float:
        matched = len(set(actual) & set(expected))
        return matched / len(expected) if expected else 1.0
```

#### DriftEvaluator

```python
# evaluators/drift.py

from dataclasses import dataclass, field
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class DriftEvaluator(Evaluator):
    """
    Detect performance drift from baseline.

    Compares current metrics against stored baseline.
    """

    baseline_score: float
    drift_threshold: float = 0.15  # 15% deviation triggers alert

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Compare output score against baseline."""
        current_score = self._extract_score(ctx.output)
        drift = abs(current_score - self.baseline_score) / self.baseline_score

        if drift > self.drift_threshold:
            ctx.set_attribute("drift_detected", True)
            ctx.set_attribute("drift_percentage", drift * 100)

        return current_score

    def _extract_score(self, output) -> float:
        if hasattr(output, "score"):
            return output.score
        if isinstance(output, (int, float)):
            return float(output)
        return 0.0
```

### 1.5 Dataset Definitions

```python
# evaluation.py - main module

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, IsInstance, Contains
from agentic_patterns.evaluators import LLMJudgeEvaluator, TrajectoryMatchEvaluator

# Routing evaluation dataset
def create_routing_dataset() -> Dataset:
    """Create evaluation dataset for routing pattern."""
    return Dataset(
        name="routing_evaluation",
        cases=[
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
        ],
        evaluators=[
            IsInstance(type_name="tuple"),
            LLMJudgeEvaluator(quality_threshold=0.7),
        ],
    )

# Multi-agent evaluation dataset
def create_multi_agent_dataset() -> Dataset:
    """Create evaluation dataset for multi-agent pattern."""
    return Dataset(
        name="multi_agent_evaluation",
        cases=[
            Case(
                name="research_task",
                inputs="Research Python async patterns and summarize",
                metadata={"complexity": "medium", "expected_agents": 3},
            ),
            Case(
                name="analysis_task",
                inputs="Analyze the pros and cons of microservices",
                metadata={"complexity": "high", "expected_agents": 4},
            ),
        ],
        evaluators=[
            IsInstance(type_name="CollaborationResult"),
            LLMJudgeEvaluator(quality_threshold=0.6),
        ],
    )
```

### 1.6 Public API

```python
# evaluation.py

async def evaluate_pattern(
    pattern_name: str,
    task_function,
    custom_cases: list[Case] | None = None,
) -> EvaluationReport:
    """
    Evaluate a pattern implementation.

    Args:
        pattern_name: Name of pattern ("routing", "multi_agent", etc.)
        task_function: Async function to evaluate
        custom_cases: Optional custom test cases

    Returns:
        EvaluationReport with scores and details
    """
    if pattern_name == "routing":
        dataset = create_routing_dataset()
    elif pattern_name == "multi_agent":
        dataset = create_multi_agent_dataset()
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")

    if custom_cases:
        dataset = Dataset(cases=custom_cases, evaluators=dataset.evaluators)

    report = await dataset.evaluate(
        task_function,
        metadata={"pattern": pattern_name, "timestamp": datetime.now().isoformat()},
    )

    return report


def evaluate_pattern_sync(
    pattern_name: str,
    task_function,
    custom_cases: list[Case] | None = None,
) -> EvaluationReport:
    """Synchronous version of evaluate_pattern."""
    import asyncio
    return asyncio.run(evaluate_pattern(pattern_name, task_function, custom_cases))
```

### 1.7 What to DELETE

| Component | Lines | Replacement |
|-----------|-------|-------------|
| `AgentMetrics` | 178-257 | pydantic-evals metrics |
| `PerformanceMonitor` | 260-366 | pydantic-evals + Logfire |
| `MetricValue` | 74-81 | pydantic-evals Case metadata |
| `PerformanceSummary` | 148-159 | pydantic-evals report |
| `TrajectoryEvaluator` (class) | 369-495 | `TrajectoryMatchEvaluator` |
| `LLMJudge` (class) | 498-551 | `LLMJudgeEvaluator` |
| `DriftDetector` | 554-679 | `DriftEvaluator` |
| `ABTestRunner` | 682-752 | pydantic-evals with variants |
| `generate_evaluation_report()` | 820-867 | `evaluate_pattern()` |

### 1.8 Implementation Tasks

1. Add `pydantic-evals` to dependencies in pyproject.toml
2. Create `evaluators/` directory with `__init__.py`
3. Implement `LLMJudgeEvaluator` in `evaluators/llm_judge.py`
4. Implement `TrajectoryMatchEvaluator` in `evaluators/trajectory.py`
5. Implement `DriftEvaluator` in `evaluators/drift.py`
6. Rewrite `evaluation.py` with Dataset factories
7. Add `evaluate_pattern()` and `evaluate_pattern_sync()` APIs
8. Delete old classes
9. Create `scripts/run_evals.sh`
10. Update tests to use new API
11. Update `__init__.py` exports

---

## 2. Value vs Complexity Analysis

### 2.1 Benefits

| Benefit | Impact |
|---------|--------|
| Standardized evaluation framework | High - consistent patterns |
| Built-in reporting (print, render) | High - better UX |
| Logfire integration | High - production observability |
| Extensible evaluators | Medium - easy customization |
| Dataset serialization (YAML/JSON) | Medium - shareable test suites |
| Metadata and metrics tracking | Medium - rich analysis |

### 2.2 Complexity Assessment

| Factor | Rating | Notes |
|--------|--------|-------|
| Lines changed | ~500 LOC | Significant rewrite |
| New dependency | Low | pydantic-evals is part of pydantic-ai |
| Learning curve | Medium | New API to learn |
| Test migration | Medium | Tests need updating |

**Overall Complexity**: Medium

### 2.3 Risk Factors

| Risk | Mitigation |
|------|------------|
| API differences | Map old concepts to new |
| Custom evaluator complexity | Start with simple evaluators |
| Performance overhead | pydantic-evals is optimized |

### 2.4 Recommendation

**Priority P0** - Do first. Establishes evaluation foundation for testing graph refactoring.

---

## 3. Pre/Post Code Analysis

### 3.1 Before: Custom Classes

```
# BEFORE: Custom, verbose implementation

@dataclass
class AgentMetrics:
    agent_id: str
    metrics: list[MetricValue] = field(default_factory=list)

    def record(self, metric_type, value):
        self.metrics.append(MetricValue(metric_type, value))

    def get_average(self, metric_type):
        values = [m.value for m in self.metrics if m.metric_type == metric_type]
        return sum(values) / len(values)

@dataclass
class PerformanceMonitor:
    agents: dict[str, AgentMetrics] = field(default_factory=dict)

    async def record_execution(self, agent_id, func, *args):
        start = datetime.now()
        result = await func(*args)
        latency = (datetime.now() - start).total_seconds() * 1000
        self.get_or_create_metrics(agent_id).record(LATENCY, latency)
        return result, latency

@dataclass
class LLMJudge:
    async def evaluate(self, query, response, context=None):
        agent = self._get_judge_agent()
        return await agent.run(f"Query: {query}\nResponse: {response}")

# Usage:
monitor = PerformanceMonitor()
result, latency = await monitor.record_execution("agent-1", func, query)
monitor.record_accuracy("agent-1", 0.95)
summary = monitor.get_summary("agent-1")
```

### 3.2 After: pydantic-evals

```
# AFTER: Clean, standardized pydantic-evals

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class LLMJudgeEvaluator(Evaluator[str, str]):
    quality_threshold: float = 0.7

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        # Evaluator logic
        return quality_score

# Create dataset with cases and evaluators
dataset = Dataset(
    cases=[
        Case(name="test1", inputs="query", expected_output="response"),
        Case(name="test2", inputs="query2", expected_output="response2"),
    ],
    evaluators=[
        LLMJudgeEvaluator(quality_threshold=0.7),
        IsInstance(type_name="str"),
    ],
)

# Run evaluation - simple!
report = await dataset.evaluate(my_function)
report.print()

# Access programmatically
for case in report.cases:
    print(f"{case.name}: {case.scores}")
```

**Improvements**:
- Declarative test cases
- Reusable evaluators
- Built-in reporting
- Logfire integration
- Serializable datasets

---

## 4. Logfire Integration

pydantic-evals automatically integrates with Logfire when configured:

### 4.1 Automatic Instrumentation

```python
# Logfire traces each evaluation
report = await dataset.evaluate(
    my_function,
    metadata={"model": "gpt-oss:20b", "version": "v1.0"},
)

# Traces include:
# - Each case execution
# - Evaluator results
# - Timing information
# - Metadata
```

### 4.2 Logfire Queries

```sql
-- Evaluation runs
SELECT
    attributes->>'pattern' as pattern,
    span_name,
    duration_ms,
    start_timestamp
FROM records
WHERE span_name LIKE 'pydantic_evals:%'
ORDER BY start_timestamp DESC;

-- Evaluator scores
SELECT
    attributes->>'case_name' as case_name,
    attributes->>'evaluator' as evaluator,
    (attributes->>'score')::float as score
FROM records
WHERE attributes->>'evaluator' IS NOT NULL;
```

---

## 5. Bash Script for Reports

```bash
#!/bin/bash
# scripts/run_evals.sh - Run evaluations and generate markdown report

set -e

PATTERN=${1:-routing}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="reports"
REPORT_FILE="${REPORT_DIR}/eval_${PATTERN}_${TIMESTAMP}.md"

mkdir -p "$REPORT_DIR"

echo "Running pydantic-evals for: $PATTERN"

# Generate report
.venv/bin/python << PYTHON_SCRIPT
import asyncio
from datetime import datetime

from agentic_patterns.evaluation import evaluate_pattern
from agentic_patterns.routing import route_query
from agentic_patterns.multi_agent import run_collaborative_task

async def main():
    pattern = "${PATTERN}"

    if pattern == "routing":
        task_fn = route_query
    elif pattern == "multi_agent":
        task_fn = run_collaborative_task
    else:
        print(f"Unknown pattern: {pattern}")
        return

    report = await evaluate_pattern(pattern, task_fn)

    # Print to console
    print("# Evaluation Report: ${PATTERN}")
    print(f"Generated: {datetime.now().isoformat()}")
    print()
    report.print(include_input=True, include_output=True)

    # Also show Logfire link
    print()
    print("## Logfire Dashboard")
    print("View detailed traces: https://logfire.pydantic.dev/")

asyncio.run(main())
PYTHON_SCRIPT | tee "$REPORT_FILE"

echo ""
echo "Report saved to: $REPORT_FILE"
```

### 5.1 Usage

```bash
# Make executable
chmod +x scripts/run_evals.sh

# Run for different patterns
./scripts/run_evals.sh routing
./scripts/run_evals.sh multi_agent

# Output saved to reports/eval_routing_20251227_150000.md
```

---

## 6. Testing Strategy

### 6.1 Evaluator Unit Tests

```python
# tests/test_evaluators.py

import pytest
from agentic_patterns.evaluators import (
    LLMJudgeEvaluator,
    TrajectoryMatchEvaluator,
    DriftEvaluator,
)
from pydantic_evals.evaluators import EvaluatorContext

def test_trajectory_exact_match():
    evaluator = TrajectoryMatchEvaluator(
        expected_actions=["parse", "search", "format"],
        match_type="exact",
    )
    ctx = EvaluatorContext(
        inputs="query",
        output=["parse", "search", "format"],
        expected_output=None,
    )
    assert evaluator.evaluate(ctx) == 1.0

def test_trajectory_in_order_match():
    evaluator = TrajectoryMatchEvaluator(
        expected_actions=["parse", "search"],
        match_type="in_order",
    )
    ctx = EvaluatorContext(
        inputs="query",
        output=["parse", "validate", "search", "format"],
        expected_output=None,
    )
    assert evaluator.evaluate(ctx) == 1.0  # Both found in order

def test_drift_evaluator_detects_drift():
    evaluator = DriftEvaluator(baseline_score=0.9, drift_threshold=0.1)
    ctx = EvaluatorContext(inputs="query", output=0.7, expected_output=None)
    score = evaluator.evaluate(ctx)
    assert ctx.get_attribute("drift_detected") is True
```

### 6.2 Dataset Tests

```python
# tests/test_evaluation.py

import pytest
from agentic_patterns.evaluation import (
    create_routing_dataset,
    create_multi_agent_dataset,
    evaluate_pattern,
)

def test_routing_dataset_has_cases():
    dataset = create_routing_dataset()
    assert len(dataset.cases) >= 4
    assert all(c.name for c in dataset.cases)

def test_routing_dataset_has_evaluators():
    dataset = create_routing_dataset()
    assert len(dataset.evaluators) >= 1

async def test_evaluate_pattern_routing(mock_route_query):
    report = await evaluate_pattern("routing", mock_route_query)
    assert report is not None
    assert len(report.cases) >= 4

async def test_evaluate_pattern_unknown_raises():
    with pytest.raises(ValueError, match="Unknown pattern"):
        await evaluate_pattern("unknown", lambda x: x)
```

### 6.3 Coverage Target

- 80%+ coverage
- All custom evaluators tested
- Dataset factories tested
- Integration with patterns verified
