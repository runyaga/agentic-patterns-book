# Spec 05: Verification Standards

**Status**: FINAL
**Purpose**: Document implementation standards, testing strategy, and verification checklist
**Priority**: P3 (finalized after implementation)

---

## 1. Implementation Standards

### 1.1 Node Definition Standards

All graph nodes must follow these conventions:

```python
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, GraphRunContext

@dataclass
class MyNode(BaseNode[StateType, DepsType, ReturnType]):
    """
    Docstring describing the node's purpose.

    Transitions:
        - NextNode: when condition X
        - OtherNode: when condition Y
        - End: when complete
    """

    # Optional node-specific data fields
    field_name: FieldType = default_value

    async def run(
        self,
        ctx: GraphRunContext[StateType, DepsType],
    ) -> NextNode | OtherNode | End[ReturnType]:
        """Execute node logic and return next node."""
        # Node implementation
        ...
```

**Checklist**:
- [x] Use `@dataclass` decorator
- [x] Inherit from `BaseNode[State, Deps, Return]`
- [x] Document transitions in docstring
- [x] Type-hint return with union of possible next nodes
- [x] Use `async def run()` signature

### 1.2 State Design Patterns

```python
from dataclasses import dataclass, field

@dataclass
class MyGraphState:
    """
    State for MyGraph workflow.

    Fields should be:
    - Immutable inputs (set once)
    - Mutable accumulators (updated during execution)
    - Derived properties (computed from other fields)
    """

    # Immutable input
    objective: str

    # Mutable accumulators (use default_factory for collections)
    results: list[Result] = field(default_factory=list)
    current_step: int = 0

    # Optional/nullable fields
    analysis: str | None = None

    # Helper methods for common operations
    @property
    def is_complete(self) -> bool:
        """Derived property."""
        return self.current_step >= len(self.results)

    def add_result(self, result: Result) -> None:
        """Mutation helper."""
        self.results.append(result)
```

**Checklist**:
- [x] Use `@dataclass` decorator
- [x] Document field purposes
- [x] Use `field(default_factory=list)` for mutable defaults
- [x] Add helper methods for complex state operations
- [x] Use `@property` for derived values

### 1.3 Graph Definition Standards

```python
from pydantic_graph import Graph

# Define nodes first, then graph
my_graph = Graph(
    nodes=[
        StartNode,
        ProcessNode,
        EndNode,
    ],
)

# Optional: Add mermaid generation
def get_my_graph_diagram() -> str:
    """Generate Mermaid diagram for documentation."""
    return my_graph.mermaid_code()
```

**Checklist**:
- [x] List nodes in logical order (start -> end)
- [x] Add diagram generation function
- [x] Export graph in `__all__`

### 1.4 Transition Type Safety

Return types define valid transitions. The type checker enforces this:

```python
# GOOD: Explicit union of allowed transitions
async def run(self, ctx) -> NodeA | NodeB | End[Result]:
    if condition_a:
        return NodeA()
    elif condition_b:
        return NodeB()
    else:
        return End(result)

# BAD: Returning node not in type hint
async def run(self, ctx) -> NodeA | End[Result]:
    return NodeC()  # Type error! NodeC not allowed
```

---

## 2. Testing Strategy

### 2.1 Test Organization

```
tests/
├── test_routing.py           # Routing tests
├── test_multi_agent.py       # Multi-agent tests
├── test_evaluation.py        # Evaluation tests (pydantic-evals)
├── test_human_in_loop.py     # HITL tests (existing functional)
└── conftest.py               # Shared fixtures
```

### 2.2 Test Categories

#### Unit Tests (per evaluator/component)

```python
def test_confidence_threshold_evaluator_above():
    """Test individual evaluator behavior."""
    evaluator = ConfidenceThresholdEvaluator(min_confidence=0.7)
    ctx = MagicMock()
    ctx.output = (MagicMock(confidence=0.95),)

    score = evaluator.evaluate(ctx)
    assert score == 0.95
```

#### Integration Tests (full pattern)

```python
async def test_evaluate_pattern_routing():
    """Test complete evaluation flow."""
    async def mock_route(query: str) -> tuple:
        return MagicMock(confidence=0.9), MagicMock()

    report = await evaluate_pattern("routing", mock_route)
    assert len(report.cases) == 4
```

#### State Transition Tests

```python
@pytest.mark.parametrize("decision,expected", [
    (Decision.APPROVED, True),
    (Decision.REJECTED, False),
])
def test_collaboration_success_evaluator(decision, expected):
    """Test evaluator with different states."""
    ...
```

### 2.3 Coverage Requirements

```bash
# Minimum coverage: 80%
uv run pytest --cov=agentic_patterns --cov-fail-under=80

# Coverage report
uv run pytest --cov=agentic_patterns --cov-report=html
```

**Current coverage**: 86%

---

## 3. Quality Gates

### 3.1 Pre-Commit Checks

Run before every commit:

```bash
# All three must pass
uv run ruff check src/ tests/        # 0 errors, 0 warnings
uv run ruff format --check src/ tests/  # All files formatted
uv run pytest                         # 80%+ coverage
```

### 3.2 Milestone Checklist

After completing each spec:

- [x] Code changes complete
- [x] `uv run ruff check src/ tests/` passes (0 errors, 0 warnings)
- [x] `uv run ruff format --check src/ tests/` passes
- [x] `uv run pytest --cov-fail-under=80` passes
- [x] Demo runs successfully (`python -m agentic_patterns.<module>`)
- [x] Documentation updated (spec files)

### 3.3 Integration Test

Full integration with Logfire:

```bash
# Set environment
export LOGFIRE_TOKEN=your_token_here
export OLLAMA_URL=http://localhost:11434

# Run specific pattern
./scripts/integration_test.sh evaluation -v

# Run all patterns
./scripts/integration_test.sh

# Verify in Logfire dashboard
echo "Check traces at: https://logfire.pydantic.dev/"
```

---

## 4. Logfire Verification

### 4.1 Verification Checklist

With `configure_logfire_for_evals()`:

- [x] **Scrubbing disabled**: Full eval data captured
- [x] **Agent spans**: Each agent call creates a span
- [x] **Latency**: Duration recorded accurately
- [x] **Token usage**: Input/output tokens tracked
- [x] **Errors**: Exceptions captured with stack traces

### 4.2 Test Queries

Verify data in Logfire with these queries:

```sql
-- Check agent spans exist
SELECT span_name, COUNT(*)
FROM records
WHERE span_name LIKE 'pydantic_ai:%'
  AND start_timestamp > NOW() - INTERVAL '1 hour'
GROUP BY span_name;

-- Check evaluation spans
SELECT span_name, duration_ms
FROM records
WHERE span_name LIKE 'pydantic_evals:%'
  AND start_timestamp > NOW() - INTERVAL '1 hour'
ORDER BY start_timestamp DESC
LIMIT 10;

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

## 5. Completed Specs Summary

### 5.1 Spec 01: Routing - KEEP

**Decision**: Keep simple functional implementation

- [x] Reviewed graph alternative
- [x] Decided dictionary dispatch is simpler and sufficient
- [x] No code changes needed
- [x] Spec marked FINAL

### 5.2 Spec 02: Multi-Agent Graph - COMPLETED

**Decision**: Replace loop with pydantic_graph

- [x] Defined `CollaborationState` with helpers
- [x] Implemented `PlanNode`
- [x] Implemented `ExecuteTaskNode` (cyclic)
- [x] Implemented `SynthesizeNode`
- [x] Defined `collaboration_graph`
- [x] Updated `run_collaborative_task()` to use graph
- [x] Tests updated and passing
- [x] Quality gates passed
- [x] Committed

### 5.3 Spec 03: Evaluation Slimdown - COMPLETED

**Decision**: Replace custom evaluation with pydantic-evals

- [x] Deleted `AgentMetrics`, `PerformanceMonitor`, etc. (~700 lines)
- [x] Added pydantic-evals integration
- [x] Created custom evaluators: `IntentMatchEvaluator`, `ConfidenceThresholdEvaluator`, `CollaborationSuccessEvaluator`, `TaskCompletionEvaluator`
- [x] Created dataset factories: `create_routing_dataset`, `create_multi_agent_dataset`
- [x] Added `evaluate_pattern()` and `evaluate_pattern_sync()` APIs
- [x] Configured logfire with `scrubbing=False`
- [x] Created `scripts/run_evals.sh`
- [x] Updated `__main__` to run real evals (not mocks)
- [x] Tests updated (37 tests passing)
- [x] Quality gates passed
- [x] Committed

### 5.4 Spec 04: Human-in-Loop Graph - DEFERRED

**Decision**: Punt for now

- [ ] Not implemented
- [ ] Existing functional `human_in_loop.py` remains

---

## 6. Scripts Reference

### 6.1 Integration Test

```bash
# Run all patterns
./scripts/integration_test.sh

# Run single pattern with verbose output
./scripts/integration_test.sh evaluation -v
./scripts/integration_test.sh routing -v
./scripts/integration_test.sh multi_agent -v

# Help
./scripts/integration_test.sh --help
```

### 6.2 Evaluation Reports

```bash
# Run evaluations and generate reports
./scripts/run_evals.sh routing
./scripts/run_evals.sh multi_agent
./scripts/run_evals.sh routing --llm    # Include LLM judge
./scripts/run_evals.sh all              # All patterns

# Reports saved to reports/eval_<pattern>_<timestamp>.txt
```

### 6.3 Direct Module Execution

```bash
# Run pattern demos
.venv/bin/python -m agentic_patterns.routing
.venv/bin/python -m agentic_patterns.multi_agent
.venv/bin/python -m agentic_patterns.evaluation routing
.venv/bin/python -m agentic_patterns.evaluation multi_agent --llm-judge
```

---

## 7. Final State

| Spec | Status | Decision |
|------|--------|----------|
| Spec 01: Routing | FINAL | KEEP simple implementation |
| Spec 02: Multi-Agent | FINAL | REPLACED with pydantic_graph |
| Spec 03: Evaluation | FINAL | REFACTORED to pydantic-evals |
| Spec 04: HITL Graph | DEFERRED | Existing functional impl remains |
| Spec 05: Standards | FINAL | This document |

**Quality Metrics**:
- Tests: 580 passing
- Coverage: 86%
- Lint: 0 errors, 0 warnings
