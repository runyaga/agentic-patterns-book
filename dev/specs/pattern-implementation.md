# Pattern Implementation Specification

**Status**: FINAL v2
**Date**: 2025-12-26

Comprehensive guide for implementing agentic design patterns with idiomatic
PydanticAI architecture.

---

## 1. PydanticAI API Reference

### 1.1 Output Validation with ModelRetry

Validate agent outputs and trigger retries with feedback to the LLM:

```python
from pydantic_ai import Agent, RunContext, ModelRetry

agent = Agent(
    model,
    output_type=OutputType,
    deps_type=DepsType,
)

@agent.output_validator
async def validate(ctx: RunContext[DepsType], output: OutputType) -> OutputType:
    """Validate output. Raise ModelRetry to retry with feedback."""
    if not meets_criteria(output):
        raise ModelRetry("Feedback message sent back to LLM for improvement")
    return output
```

**Key points:**
- Decorator is `@agent.output_validator` (not `result_validator`)
- Signature: `async def(ctx: RunContext[T], output: O) -> O`
- Raise `ModelRetry(message)` to trigger retry with feedback
- Return output unchanged if valid

### 1.2 Dynamic System Prompts

Inject context from deps into system prompts:

```python
from pydantic_ai import Agent, RunContext

agent = Agent(
    model,
    system_prompt="Base instructions here.",
    deps_type=DepsType,
)

@agent.system_prompt
def add_context(ctx: RunContext[DepsType]) -> str:
    """Dynamic prompt added to base system prompt."""
    return f"Additional context: {ctx.deps.some_field}"
```

**Key points:**
- Multiple `@system_prompt` decorators are concatenated
- Function receives `RunContext` with access to `ctx.deps`
- Can be sync or async

### 1.3 Tools with RunContext

Define tools that access dependencies:

```python
from pydantic_ai import Agent, RunContext

agent = Agent(model, deps_type=DepsType)

@agent.tool
async def my_tool(ctx: RunContext[DepsType], query: str) -> str:
    """Tool docstring becomes the tool description for the LLM."""
    return ctx.deps.service.process(query)
```

**Key points:**
- `@agent.tool` for async tools with context
- `@agent.tool_plain` for sync tools without context
- First param must be `ctx: RunContext[DepsType]`

### 1.4 Deps Pattern

Use `@dataclass` for dependencies (not Pydantic models):

```python
from dataclasses import dataclass

@dataclass
class MyDeps:
    threshold: float = 0.8
    service: SomeService | None = None

# Pass at runtime
result = await agent.run("query", deps=MyDeps(threshold=0.9))
```

---

## 2. When to Use Each Feature

Apply idiomatic features where they add value; don't force them where they
don't fit.

### 2.1 Decision Framework

When evaluating a pattern, ask:

1. **Is there a retry loop?** → Consider `@output_validator` + `ModelRetry`
2. **Is context injected into system prompt?** → Consider `@system_prompt`
3. **Is there shared runtime state?** → Consider `deps_type`
4. **Does the agent need to fetch data dynamically?** → Consider `@tool`

If the answer is "no" to all, the pattern may already be idiomatic.

### 2.2 `@output_validator` + `ModelRetry`

**Use when:**
- You have explicit quality criteria (score thresholds, schema validation)
- Failed outputs should trigger automatic retry with feedback
- The model can meaningfully improve based on the error message

**Don't use when:**
- Output is pass-through with no quality gate
- Validation is binary pass/fail with no useful feedback
- Retrying won't help (e.g., missing information the model can't invent)

### 2.3 `@system_prompt` Decorator

**Use when:**
- Injecting **persistent context** into the system prompt (conversation history,
  user preferences, retrieved knowledge)
- Context comes from `deps` and should be available for every call

**Don't use when:**
- Building the **user message** with task-specific data
- Data is one-time input, not persistent context

### 2.4 `deps_type` + `RunContext`

**Use when:**
- Runtime configuration (thresholds, limits, feature flags)
- Shared services (database connections, API clients, stores)
- State that multiple decorators/tools need access to

**Don't use when:**
- No shared state between agent components
- All configuration is static/hardcoded

### 2.5 `@tool` Decorator

**Use when:**
- Agent needs to **dynamically fetch** external data during generation
- The agent should decide when/whether to call the tool

**Don't use when:**
- All context is available before the agent runs
- The "tool" would always be called exactly once

---

## 3. Implementation Workflow

### Phase 1: Discovery
1. Query agent-book MCP for chapter content
2. Extract pattern description and use cases
3. Identify key concepts and data flows

### Phase 2: Design
1. Define Pydantic models for inputs/outputs
2. Design agent structure (single vs multi-agent)
3. Plan error handling strategy

### Phase 3: Implementation
1. Create `src/agentic_patterns/{pattern}.py`
2. Use `from agentic_patterns._models import get_model`
3. Follow code style: 79 char lines, type hints, docstrings

### Phase 4: Testing
1. Create `tests/test_{pattern}.py`
2. Unit tests for all Pydantic models
3. Integration tests with mocked agents
4. Achieve >= 80% test coverage

### Phase 5: Validation
1. Run `uv run ruff check src/ tests/` - zero warnings
2. Run `uv run pytest` - all passing
3. Submit to Blacksmith agent for code review
4. Address feedback (max 2 cycles)

### Phase 6: Documentation
1. Verify module has docstring with pattern description
2. Update pattern doc in `docs/patterns/`
3. Add lessons learned to `dev/LESSONS.md`

---

## 4. Pattern Status & Assessments

### 4.1 Status Table

| Ch | Pattern | Status | Idiomatic |
|----|---------|--------|-----------|
| 1 | Prompt Chaining | DONE | ✅ Already idiomatic |
| 2 | Routing | DONE | ✅ Already idiomatic |
| 3 | Parallelization | DONE | ✅ Already idiomatic |
| 4 | Reflection | DONE | ✅ Uses `@output_validator` |
| 5 | Tool Use | DONE | ✅ Reference implementation |
| 6 | Planning | DONE | ✅ Uses deps + `@system_prompt` |
| 7 | Multi-Agent | DONE | ✅ Uses `deps_type` |
| 8 | Memory | DONE | ✅ Uses `@system_prompt` + deps |
| 9 | Learning | DONE | ✅ Uses `@system_prompt` + deps |
| 13 | Human-in-Loop | DONE | ✅ Uses `output_type` |
| 14 | Knowledge Retrieval | DONE | ✅ Uses `@tool` |
| 16 | Resource-Aware | DONE | ✅ Uses deps |
| 18 | Guardrails | DONE | ✅ Already idiomatic |
| 19 | Evaluation | DONE | ✅ Already idiomatic |
| 20 | Prioritization | DONE | ✅ Already idiomatic |
| 10 | MCP Integration | DONE | ✅ Uses `toolsets` + `MCPServerStdio` |
| 11 | Teleological Engine (Goal Setting) | pending | TBD |
| 12 | Phoenix Protocol (Exception Handling) | DONE | ✅ Uses `recoverable_run` wrapper |
| 15 | Agora (Inter-Agent Comm) | pending | TBD |
| 17 | Cognitive Weaver (Reasoning) | pending | TBD |
| 21 | Cartographer (Exploration) | pending | TBD |

### 4.2 Pattern Assessments

**prompt_chaining.py** - f-strings build user messages with step data, not
system context. This is correct and idiomatic.

**routing.py** - f-strings build task inputs for router and handlers.
Already idiomatic.

**parallelization.py** - Workers are independent, no shared state needed.
Already idiomatic.

**reflection.py** - Uses `@output_validator` + `ModelRetry` for critic loop.
`ReflectionDeps` holds critic agent reference.

**tool_use.py** - Reference implementation. Uses `deps_type`, `RunContext`,
`@agent.tool` and `@agent.tool_plain` decorators.

**memory.py** - Uses `@system_prompt` to inject conversation history from
`MemoryDeps`. Clean idiomatic pattern.

**knowledge_retrieval.py** - Uses `@tool` for dynamic retrieval. Agent decides
when to search via `search_knowledge` tool.

**human_in_loop.py** - Agent returns `AgentOutput` with self-assessed
confidence via `output_type`, not wrapper-assigned.

**learning.py** - Uses `@system_prompt` to inject past experiences from
`LearningDeps.store`.

**guardrails.py** - Guardrails are Python-level gates, not agent retry loops.
Intentionally doesn't use `@output_validator`.

---

## 5. Project Structure

```
src/agentic_patterns/
├── __init__.py
├── _models.py              # Shared model configuration
├── prompt_chaining.py      # Chapter 1
├── routing.py              # Chapter 2
├── parallelization.py      # Chapter 3
└── ...

tests/
├── conftest.py
├── test_prompt_chaining.py
├── test_routing.py
└── ...
```

### Model Configuration

Use the shared `get_model()` function:

```python
from agentic_patterns._models import get_model

model = get_model()  # defaults: gpt-oss:20b, localhost:11434
model = get_model(model_name="other-model")  # override
```

---

## 6. Error Handling & Quality Gates

### Retry Logic

| Parameter | Value |
|-----------|-------|
| Max retries per agent call | 3 |
| Backoff strategy | Fixed 1s delay |
| Timeout per agent call | 90s |

### When to Give Up

Abandon implementation attempt when:
1. Agent returns malformed output after **3 retries**
2. Implementation fails after **3 complete attempts**
3. Blacksmith rejects after **2 revision cycles**

### Quality Gates

Each pattern must pass ALL gates:

- [ ] `ruff check` with zero warnings
- [ ] `ruff format` applied
- [ ] `pytest` all tests passing
- [ ] Test coverage >= 80%
- [ ] Blacksmith approval
- [ ] Documentation updated

---

## 7. Blacksmith Integration

### When to Invoke
- After all tests pass
- After linter passes
- Before marking pattern as "complete"

### Review Criteria
1. Code correctness
2. Error handling completeness
3. Test coverage >= 80%
4. Documentation quality
5. Adherence to project standards

### Handling Feedback

```
Blacksmith Response
       │
       ├─── APPROVED ──────────> Mark Complete
       │
       ├─── MINOR ISSUES ──────> Auto-fix, Re-submit (cycle 1)
       │                              │
       │                              └──> If fails again ──> Human Review
       │
       └─── MAJOR ISSUES ──────> Revise, Re-submit (cycle 1)
                                      │
                                      └──> If fails again ──> Mark Blocked
```

Max Blacksmith cycles: **2**

---

## 8. Test Guidelines

### Mock Patterns

When agents are in lookup dicts, patch the dict directly:

```python
# RIGHT - patch the dict entry
with patch.dict("module.INTENT_HANDLERS", {Intent.ORDER_STATUS: mock}):
    ...

# WRONG - handler lookup still gets original
with patch("module.order_status_agent") as mock:
    ...
```

### Mock Result Structure

pydantic-ai `agent.run()` returns result with `.output`:

```python
mock_result = MagicMock()
mock_result.output = YourPydanticModel(...)
mock_agent.run = AsyncMock(return_value=mock_result)
```

### Test Decorated Tools Directly

`@tool` functions need direct testing with mock `RunContext`:

```python
@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.deps = ToolDependencies(weather_data={"berlin": {...}})
    return ctx

async def test_get_weather_direct(mock_context):
    result = await get_weather(mock_context, "Berlin")
    assert result.temperature == 5.0
```

---

## 9. Verification Checklist

For each pattern, verify:

- [ ] `uv run ruff check src/agentic_patterns/{pattern}.py` passes
- [ ] `uv run ruff format src/agentic_patterns/{pattern}.py` passes
- [ ] `uv run pytest tests/test_{pattern}.py` passes
- [ ] `.venv/bin/python -m agentic_patterns.{pattern}` demo runs
- [ ] No manual retry loops (use `@output_validator` + `ModelRetry`)
- [ ] Context injection uses `@system_prompt` (not f-string prompt building)
- [ ] Runtime state passed via `deps`
- [ ] `deps_type` generic matches `RunContext` type parameter

---

## 10. Definition of Done

A pattern is "Idiomatic PydanticAI" when:

1. **No manual string concatenation** for context injection - use
   `@system_prompt`
2. **No manual retry loops** for validation - use `@output_validator` +
   `ModelRetry`
3. **Models define output structure** - use `output_type`, not Python wrappers
4. **All state via deps** - runtime configuration passed through `deps_type`
