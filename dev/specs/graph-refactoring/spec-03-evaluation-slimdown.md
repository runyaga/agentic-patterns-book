# Spec 03: Evaluation Refactoring to pydantic-evals

**Status**: FINAL
**Decision**: REFACTOR - replaced custom evaluation with pydantic-evals
**Priority**: P0 (completed first - establishes evaluation foundation)
**Complexity**: Medium

---

## 1. Implementation Summary

### 1.1 What Was Done

Refactored the evaluation module to use `pydantic-evals` as the core evaluation framework. This replaced custom in-memory tracking (~1000 lines) with a streamlined implementation (~300 lines) that integrates with Logfire.

### 1.2 File Changes

```
src/agentic_patterns/
├── evaluation.py           # REPLACED - now uses pydantic-evals

scripts/
├── run_evals.sh            # NEW - bash script for report generation

tests/
├── test_evaluation.py      # UPDATED - tests for new API
```

### 1.3 Key Decisions

1. **Use built-in pydantic-evals evaluators** instead of custom evaluators directory
   - `LLMJudge` - built-in evaluator for LLM-as-judge
   - `IsInstance` - type checking
   - `Contains` - substring matching

2. **Create pattern-specific evaluators** as simple classes in evaluation.py:
   - `IntentMatchEvaluator` - checks routing intent matches expected
   - `ConfidenceThresholdEvaluator` - validates confidence above threshold
   - `CollaborationSuccessEvaluator` - checks multi-agent success
   - `TaskCompletionEvaluator` - calculates task completion ratio

3. **Logfire integration** with scrubbing disabled for full eval visibility

---

## 2. Implementation Details

### 2.1 Custom Evaluators

```python
class IntentMatchEvaluator(Evaluator[str, tuple]):
    """Check if routing decision matches expected intent."""

    def __init__(self, expected_intent: str) -> None:
        self.expected_intent = expected_intent

    def evaluate(self, ctx: EvaluatorContext[str, tuple]) -> float:
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
        if ctx.output is None:
            return 0.0
        if hasattr(ctx.output, "success"):
            return 1.0 if ctx.output.success else 0.0
        return 0.0


class TaskCompletionEvaluator(Evaluator[str, Any]):
    """Evaluate task completion ratio in multi-agent collaboration."""

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        if ctx.output is None:
            return 0.0
        if hasattr(ctx.output, "task_results"):
            results = ctx.output.task_results
            if not results:
                return 0.0
            successful = sum(1 for r in results if r.success)
            return successful / len(results)
        return 0.0
```

### 2.2 Dataset Factories

```python
def create_routing_dataset(
    include_llm_judge: bool = False,
    llm_judge_model: str | None = None,
) -> Dataset:
    """Create evaluation dataset for routing pattern."""
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
    """Create evaluation dataset for multi-agent collaboration pattern."""
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
```

### 2.3 Public API

```python
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


def evaluate_pattern_sync(...) -> EvaluationReport:
    """Synchronous version of evaluate_pattern."""
```

### 2.4 Logfire Configuration

```python
def configure_logfire_for_evals(
    service_name: str = "agentic-patterns-evals",
    environment: str = "development",
) -> None:
    """
    Configure Logfire for evaluation runs.

    Disables scrubbing to capture full evaluation data.
    """
    logfire.configure(
        scrubbing=False,  # Disable scrubbing for full eval visibility
        send_to_logfire="if-token-present",
        service_name=service_name,
        environment=environment,
    )
    logfire.instrument_pydantic_ai()
```

---

## 3. Bash Script for Reports

### 3.1 scripts/run_evals.sh

```bash
#!/bin/bash
# Run pydantic-evals evaluations and generate reports
#
# Usage:
#   ./scripts/run_evals.sh routing           # Evaluate routing pattern
#   ./scripts/run_evals.sh multi_agent       # Evaluate multi-agent pattern
#   ./scripts/run_evals.sh routing --llm     # Include LLM judge
#   ./scripts/run_evals.sh all               # Evaluate all patterns
#
# Environment:
#   LOGFIRE_TOKEN - Set to enable Logfire tracing
#   OLLAMA_URL    - Ollama server URL (default: http://localhost:11434)

set -e

PATTERN=${1:-routing}
USE_LLM_JUDGE=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="reports"

# Parse flags
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --llm)
            USE_LLM_JUDGE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$REPORT_DIR"

# Run evaluation with embedded Python script
# Outputs report to console and saves to file
```

### 3.2 Usage

```bash
# Make executable
chmod +x scripts/run_evals.sh

# Run for different patterns
./scripts/run_evals.sh routing
./scripts/run_evals.sh multi_agent
./scripts/run_evals.sh routing --llm  # Include LLM judge
./scripts/run_evals.sh all            # Run all patterns

# Output saved to reports/eval_routing_20251227_150000.txt
```

---

## 4. Value vs Complexity Analysis

### 4.1 Benefits Achieved

| Benefit | Impact |
|---------|--------|
| **~700 lines removed** | High - significantly cleaner codebase |
| Standardized evaluation framework | High - consistent patterns |
| Built-in reporting (print, render) | High - better UX |
| Logfire integration with scrubbing disabled | High - full observability |
| Extensible evaluators | Medium - easy customization |
| Dataset serialization (YAML/JSON) | Medium - shareable test suites |

### 4.2 What Was Deleted

| Component | Status |
|-----------|--------|
| `AgentMetrics` | Deleted - replaced by pydantic-evals |
| `PerformanceMonitor` | Deleted - replaced by Logfire |
| `MetricValue` | Deleted - replaced by Case metadata |
| `PerformanceSummary` | Deleted - replaced by EvaluationReport |
| `TrajectoryEvaluator` | Deleted - use pydantic-evals built-in |
| `LLMJudge` (custom) | Deleted - use pydantic-evals LLMJudge |
| `DriftDetector` | Deleted - use Logfire queries |
| `ABTestRunner` | Deleted - use pydantic-evals variants |

### 4.3 Final Assessment

| Factor | Rating |
|--------|--------|
| Lines changed | ~700 lines removed |
| New dependency | None - pydantic-evals is part of pydantic-ai |
| Learning curve | Low - simple API |
| Test migration | Complete - 37 tests passing |

---

## 5. Testing

### 5.1 Test Coverage

All evaluators and APIs are tested:

```python
# tests/test_evaluation.py

class TestIntentMatchEvaluator:
    def test_matching_intent(self) -> None: ...
    def test_non_matching_intent(self) -> None: ...
    def test_empty_output(self) -> None: ...
    def test_output_without_intent(self) -> None: ...

class TestConfidenceThresholdEvaluator:
    def test_above_threshold(self) -> None: ...
    def test_below_threshold(self) -> None: ...
    def test_at_threshold(self) -> None: ...
    def test_empty_output(self) -> None: ...

class TestCollaborationSuccessEvaluator:
    def test_successful_collaboration(self) -> None: ...
    def test_failed_collaboration(self) -> None: ...
    def test_none_output(self) -> None: ...

class TestTaskCompletionEvaluator:
    def test_all_tasks_successful(self) -> None: ...
    def test_partial_success(self) -> None: ...
    def test_all_tasks_failed(self) -> None: ...
    def test_empty_tasks(self) -> None: ...
    def test_none_output(self) -> None: ...

class TestCreateRoutingDataset:
    def test_creates_dataset(self) -> None: ...
    def test_has_expected_cases(self) -> None: ...
    def test_has_evaluators(self) -> None: ...
    def test_without_llm_judge(self) -> None: ...
    def test_with_llm_judge(self) -> None: ...

class TestCreateMultiAgentDataset:
    def test_creates_dataset(self) -> None: ...
    def test_has_expected_cases(self) -> None: ...
    def test_has_evaluators(self) -> None: ...

class TestEvaluatePattern:
    async def test_unknown_pattern_raises(self) -> None: ...
    async def test_routing_pattern(self) -> None: ...
    async def test_with_custom_cases(self) -> None: ...
    async def test_with_metadata(self) -> None: ...

class TestEvaluatePatternSync:
    def test_routing_pattern_sync(self) -> None: ...
    def test_unknown_pattern_raises_sync(self) -> None: ...
```

### 5.2 Quality Gates

```bash
# All passed:
uv run ruff check src/ tests/     # 0 errors
uv run ruff format --check src/   # 34 files formatted
uv run pytest                     # 580 tests, 86% coverage
```

---

## 6. Logfire Queries

With scrubbing disabled, full evaluation data is available in Logfire:

```sql
-- Evaluation runs by pattern
SELECT
    attributes->>'pattern' as pattern,
    span_name,
    duration_ms,
    start_timestamp
FROM records
WHERE span_name LIKE 'pydantic_evals:%'
ORDER BY start_timestamp DESC;

-- Evaluator scores per case
SELECT
    attributes->>'case_name' as case_name,
    attributes->>'evaluator' as evaluator,
    (attributes->>'score')::float as score
FROM records
WHERE attributes->>'evaluator' IS NOT NULL;

-- Token usage during evaluation
SELECT
    attributes->>'model_name' as model,
    SUM((attributes->>'input_tokens')::int) as input_tokens,
    SUM((attributes->>'output_tokens')::int) as output_tokens
FROM records
WHERE span_name LIKE 'agent:%'
GROUP BY 1;
```

---

## 7. Exports

```python
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
```
