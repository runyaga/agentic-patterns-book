# Implementation Idioms

Framework idioms for pydantic-ai, pydantic_graph, and pydantic-evals.

Use these patterns when implementing agentic workflows to ensure idiomatic,
maintainable code.

---

## PydanticAI Idioms

Use these native features instead of manual Python control flow.

### Output Validation with ModelRetry

```python
@agent.output_validator
async def validate(ctx: RunContext[Deps], output: Output) -> Output:
    if not meets_criteria(output):
        raise ModelRetry("Feedback for improvement")
    return output
```

**Use when:** Quality criteria exist, failed outputs should retry with feedback.

### Dynamic System Prompts

```python
@agent.system_prompt
def add_context(ctx: RunContext[Deps]) -> str:
    return f"Context: {ctx.deps.some_field}"
```

**Use when:** Injecting persistent context (history, preferences, knowledge).

### Tools with RunContext

```python
@agent.tool
async def my_tool(ctx: RunContext[Deps], query: str) -> str:
    return ctx.deps.service.process(query)
```

**Use when:** Agent needs to dynamically fetch external data.

### Dependencies Pattern

```python
@dataclass
class MyDeps:
    threshold: float = 0.8
    service: SomeService | None = None

result = await agent.run("query", deps=MyDeps(threshold=0.9))
```

**Use when:** Runtime configuration, shared services, state across decorators.

---

## pydantic_graph: Two APIs

`pydantic_graph` has two distinct APIs:

| API | Import | Status | Style |
|-----|--------|--------|-------|
| **Node-based** | `from pydantic_graph import BaseNode, Graph` | Stable | Class-based nodes, explicit transitions |
| **Builder** | `from pydantic_graph.beta import GraphBuilder` | Beta | Declarative, functional, parallel support |

This project uses the **stable node-based API**.

---

## pydantic_graph (Stable Node API)

The stable API uses `BaseNode` classes where each node returns the next node.

### When to Use

**Good fit:**

| Use Case | Why |
|----------|-----|
| Supervisor/worker patterns | Explicit Plan → Execute → Synthesize flow |
| Cyclic workflows | Nodes can return themselves for loops |
| State machine semantics | Transitions are type-checked |
| Mermaid visualization | `graph.mermaid_code()` for docs |
| Human-in-the-loop gates | Pause at specific nodes, resume later |

**Poor fit:**

| Use Case | Better Alternative |
|----------|-------------------|
| Simple linear chains | Direct function calls or prompt chaining |
| Single-agent tasks | Plain `agent.run()` |
| Dictionary dispatch routing | `HANDLERS[intent]()` is simpler |
| Parallel fan-out/fan-in | `pydantic_graph.beta` or `asyncio.gather()` |

### Graph State Design

Define immutable inputs and mutable accumulators:

```python
from dataclasses import dataclass, field

@dataclass
class WorkflowState:
    # Immutable input (set once at start)
    objective: str

    # Mutable accumulators (updated during execution)
    results: list[str] = field(default_factory=list)
    current_step: int = 0

    # Optional/nullable intermediate values
    plan: Plan | None = None
```

**Key principle:** State is a dataclass, not Pydantic BaseModel. Use `field(default_factory=list)` for mutable defaults.

### Node Definition Pattern

Each node returns the next node to execute:

```python
from pydantic_graph import BaseNode, End, GraphRunContext

@dataclass
class ProcessNode(BaseNode[WorkflowState, None, Result]):
    """
    Process one item from the queue.

    Transitions:
        - ProcessNode: more items pending (cyclic)
        - FinalizeNode: queue empty
    """

    async def run(
        self,
        ctx: GraphRunContext[WorkflowState]
    ) -> "ProcessNode" | FinalizeNode | End[Result]:
        # Access and mutate state
        ctx.state.results.append("processed")
        ctx.state.current_step += 1

        # Return next node (type-checked!)
        if ctx.state.current_step < len(ctx.state.pending):
            return ProcessNode()  # Cycle back
        return FinalizeNode()
```

**Key principle:** Return type is a union of allowed transitions. Type checker enforces valid transitions.

### Graph Definition

```python
from pydantic_graph import Graph

my_graph = Graph(nodes=[StartNode, ProcessNode, FinalizeNode])

# Run to completion
result = await my_graph.run(StartNode(), state=WorkflowState(objective="task"))

# Get mermaid diagram
print(my_graph.mermaid_code())
```

### Cyclic Graphs (Loops)

Nodes can return themselves to create loops:

```python
@dataclass
class ExecuteTaskNode(BaseNode[State, None, Result]):
    async def run(self, ctx: GraphRunContext[State]) -> "ExecuteTaskNode" | SynthesizeNode:
        task = ctx.state.pending_tasks.pop(0)
        result = await execute(task)
        ctx.state.completed.append(result)

        if ctx.state.pending_tasks:
            return ExecuteTaskNode()  # More work
        return SynthesizeNode()  # Done
```

### Limitations (Stable API)

1. **No parallel node execution** - Nodes run sequentially. Use `asyncio.gather()` inside a node if needed.

2. **State must be mutable** - Graph mutates state in place. Immutable state patterns don't work well.

3. **No built-in persistence** - State serialization for pause/resume is manual.

4. **Limited error recovery** - No built-in retry or fallback transitions.

5. **Verbose for simple flows** - A 3-step linear flow needs 3 node classes. Sometimes a function is simpler.

---

## pydantic_graph.beta (Builder API)

The beta API provides a more declarative, functional approach with built-in parallelism.

### Key Features (Beta Only)

- **`GraphBuilder`** - Declarative graph construction
- **`@g.step`** - Decorator-based step definition
- **`.map()`** - Fan-out parallel execution over iterables
- **`.transform()`** - Inline data transformation on edges
- **`g.join()`** - Fan-in with reducers (`reduce_sum`, `reduce_list_append`)
- **`g.decision()`** - Conditional branching

### Example: Parallel Map-Reduce

```python
from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

@dataclass
class SimpleState:
    pass

g = GraphBuilder(state_type=SimpleState, output_type=list[int])

@g.step
async def generate_list(ctx: StepContext[SimpleState, None, None]) -> list[int]:
    return [1, 2, 3, 4, 5]

@g.step
async def square(ctx: StepContext[SimpleState, None, int]) -> int:
    return ctx.inputs * ctx.inputs

collect = g.join(reduce_list_append, initial_factory=list[int])

g.add(
    g.edge_from(g.start_node).to(generate_list),
    g.edge_from(generate_list).map().to(square),  # Parallel!
    g.edge_from(square).to(collect),
    g.edge_from(collect).to(g.end_node),
)

graph = g.build()
result = await graph.run(state=SimpleState())
# [1, 4, 9, 16, 25]
```

### When to Use Beta vs Stable

| Scenario | Recommended API |
|----------|-----------------|
| Need parallel fan-out/fan-in | `pydantic_graph.beta` |
| Need inline edge transforms | `pydantic_graph.beta` |
| Need reduce/join operations | `pydantic_graph.beta` |
| Supervisor/worker with cycles | Either works, stable is simpler |
| Human-in-the-loop gates | `pydantic_graph` (stable) |

### Beta Caveats

- API may change before stable release
- Less documentation and examples
- More complex mental model (edges, joins, decisions)

---

## Decision: Graph vs Functions

```
Q: Is it a multi-step workflow with branching/cycles?
├─ Yes → Consider pydantic_graph
│   Q: Need parallel fan-out?
│   ├─ Yes → pydantic_graph.beta
│   └─ No → pydantic_graph (stable)
└─ No → Use plain functions or agent.run()
```

---

## pydantic-evals Idioms

Use `pydantic-evals` for systematic evaluation of agent outputs.

### Dataset and Cases

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge

cases = [
    Case(
        name="order_query",
        inputs="Where is my order?",
        expected_output={"intent": "order_status"},
        metadata={"category": "orders"},
    ),
]

dataset = Dataset(
    cases=cases,
    evaluators=[IsInstance(type_name="tuple")],
)
```

### Custom Evaluators

```python
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

class ConfidenceEvaluator(Evaluator[str, tuple]):
    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def evaluate(self, ctx: EvaluatorContext[str, tuple]) -> float:
        if ctx.output and hasattr(ctx.output[0], "confidence"):
            conf = ctx.output[0].confidence
            return conf if conf >= self.min_confidence else 0.0
        return 0.0
```

### Running Evaluations

```python
async def my_task(query: str) -> tuple:
    return await route_query(query)

report = await dataset.evaluate(my_task)
report.print(include_input=True, include_output=True)
```

**Use when:** Systematic testing, regression detection, A/B comparisons, quality metrics.

---

## Decision Framework

### Agent Features (pydantic-ai)

1. **Is there a retry loop?** → Use `@output_validator` + `ModelRetry`
2. **Is context injected into system prompt?** → Use `@system_prompt`
3. **Is there shared runtime state?** → Use `deps_type`
4. **Does the agent need to fetch data dynamically?** → Use `@tool`

### Workflow Structure (pydantic_graph)

5. **Is it a multi-step stateful workflow?** → Use `pydantic_graph` (stable)
6. **Are there cyclic transitions (loops)?** → Use graph nodes with self-return
7. **Is it a supervisor/worker pattern?** → Use graph with Plan → Execute → Synthesize nodes
8. **Need parallel fan-out/fan-in?** → Use `pydantic_graph.beta` with `.map()` and `g.join()`

### Evaluation (pydantic-evals)

9. **Need systematic quality testing?** → Use `Dataset` with `Case` objects
10. **Need custom scoring logic?** → Create custom `Evaluator` subclass
11. **Need LLM-based judgment?** → Use `LLMJudge` evaluator

If "no" to all, the pattern may not need these features.
