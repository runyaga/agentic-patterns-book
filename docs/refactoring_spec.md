# Revision Specification: Agentic Design Refactoring

**Version:** 3.0 (AI-Implementable)
**Date:** December 26, 2025
**Objective:** Refactor agentic patterns to use PydanticAI's native runtime
mechanisms (`ModelRetry`, `output_validator`, `@system_prompt`, `@tool`,
`RunContext`) instead of manual Python control flow.

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

These are **principles, not mandates**. Apply idiomatic features where they add
value; don't force them where they don't fit.

### 2.1 `@output_validator` + `ModelRetry`

**Use when:**
- You have explicit quality criteria (score thresholds, schema validation)
- Failed outputs should trigger automatic retry with feedback
- The model can meaningfully improve based on the error message

**Don't use when:**
- Output is pass-through with no quality gate
- Validation is binary pass/fail with no useful feedback
- Retrying won't help (e.g., missing information the model can't invent)

**Example patterns:** reflection (critic score), planning (step validation)

### 2.2 `@system_prompt` Decorator

**Use when:**
- Injecting **persistent context** into the system prompt (conversation history,
  user preferences, retrieved knowledge)
- Context comes from `deps` and should be available for every call
- You want to separate "what the agent is" from "what context it has"

**Don't use when:**
- Building the **user message** with task-specific data
- Data is one-time input, not persistent context
- The f-string is constructing what the user is asking, not agent context

**Example patterns:** memory (conversation history), learning (past experiences)

**Counter-example:** prompt_chaining - f-strings build user messages with step
data, not system context. This is correct.

### 2.3 `deps_type` + `RunContext`

**Use when:**
- Runtime configuration (thresholds, limits, feature flags)
- Shared services (database connections, API clients, stores)
- State that multiple decorators/tools need access to

**Don't use when:**
- No shared state between agent components
- All configuration is static/hardcoded
- Single-use agents with no tools or validators

**Example patterns:** reflection (acceptable_score), RAG (vector store),
resource_aware (budget tracker)

### 2.4 `@tool` Decorator

**Use when:**
- Agent needs to **dynamically fetch** external data during generation
- The agent should decide when/whether to call the tool
- Data isn't known upfront and depends on the conversation

**Don't use when:**
- All context is available before the agent runs
- You're just passing data through the prompt
- The "tool" would always be called exactly once

**Example patterns:** knowledge_retrieval (search tool), tool_use (calculators,
APIs)

**Counter-example:** RAG with pre-retrieved context - if you always retrieve
before calling the agent, that's not a tool, that's prompt construction.

### 2.5 Decision Framework

When evaluating a pattern, ask:

1. **Is there a retry loop?** → Consider `@output_validator` + `ModelRetry`
2. **Is context injected into system prompt?** → Consider `@system_prompt`
3. **Is there shared runtime state?** → Consider `deps_type`
4. **Does the agent need to fetch data dynamically?** → Consider `@tool`

If the answer is "no" to all, the pattern may already be idiomatic or may not
need these features.

---

## 3. Pattern Assessments

Each pattern is assessed against the decision framework in Section 2.5.

### 3.1 reflection.py

**Decision Framework:**
- ✅ Has retry loop (produce → critique → retry) → needs `@output_validator`
- ❌ No persistent context injection → `@system_prompt` not needed
- ✅ Has runtime config (acceptable_score) → needs `deps_type`
- ❌ No dynamic data fetching → `@tool` not needed

**Assessment:** Needs refactoring. Manual while loop should become
`@output_validator` + `ModelRetry`.

**Current issues:**
- Manual `while` loop for produce/critique/retry cycle
- Threshold hardcoded instead of via deps

**Target approach:**
- `@output_validator` calls critic, raises `ModelRetry` on low score
- `ReflectionDeps(acceptable_score=8.0)` for threshold
- Remove `reflect_once()`, `refiner_agent`, manual loop

**Keep:** `self_reflect()` (different pattern), all Pydantic models,
`critic_agent`

---

### 3.2 human_in_loop.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❌ No persistent context → `@system_prompt` not needed
- ❌ No shared runtime state → `deps_type` not strictly needed
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Needs refactoring, but not for idiomatic reasons. Issue is
`output_type=str` with fake confidence assigned by wrapper.

**Current issues:**
- `task_agent` returns `str`, then wrapper creates `AgentOutput` with
  `auto_confidence` parameter (fake confidence)
- Model should self-assess confidence, not have it assigned

**Target approach:**
- Change `output_type=AgentOutput` so model returns confidence directly
- Remove `auto_confidence` parameter
- System prompt instructs model on confidence rating

**Keep:** All models, escalation logic, `decision_agent`, `augment_decision()`

---

### 3.3 memory.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ✅ Injects conversation history into system prompt → needs `@system_prompt`
- ✅ Memory instance is shared state → needs `deps_type`
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Already refactored. Uses `@system_prompt` and `MemoryDeps`.

**Current state (already idiomatic):**
- Has `MemoryDeps` dataclass with memory instance
- Has `@conversational_agent.system_prompt` decorator
- Uses `RunContext[MemoryDeps]` to access memory

**No changes needed.**

**Keep:** All memory classes, all models, `summarizer_agent`,
`run_conversation()`

---

### 3.4 knowledge_retrieval.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❌ Context is retrieved, not persistent → `@system_prompt` not ideal
- ✅ VectorStore is shared service → needs `deps_type`
- ✅ Agent should search dynamically → needs `@tool`

**Assessment:** Needs refactoring. Current implementation pre-retrieves and
builds prompt with f-string. Should use `@tool` for dynamic retrieval.

**Current issues:**
- `RAGPipeline.query()` retrieves chunks first, then builds prompt
- `expand_query` parameter adds complexity
- Agent doesn't control when/how to search

**Target approach:**
- `@rag_agent.tool` for `search_knowledge` function
- `RAGDeps` with `store`, `top_k`, `min_score`
- Agent decides when to search based on question
- Remove `expand_query`, `query_agent`, `_format_context()`

**Keep:** All utility functions, all models, `VectorStore`,
`build_knowledge_base()`

---

### 3.5 planning.py

**Decision Framework:**
- ✅ Has step execution validation → could use `@output_validator`
- ✅ Injects completed steps as context → could use `@system_prompt`
- ✅ Plan state is shared across steps → needs `deps_type`
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Needs refactoring. Manual for-loop with f-string prompts.

**Current issues:**
- Manual for-loop executes plan steps
- f-string prompts inject completed step results
- Plan state passed as function parameters

**Target approach:**
- `PlanningDeps` with `plan`, `completed_steps`, `max_replans`
- `@system_prompt` to inject completed step context
- Consider `@output_validator` for step validation (optional)
- Pass state via deps instead of parameters

**Keep:** All models, all agents, core function signatures

---

### 3.6 prompt_chaining.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❌ No persistent context (step data is user message) → `@system_prompt` not
  needed
- ❌ No shared runtime config → `deps_type` not needed
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Already idiomatic. No changes needed.

**Why f-strings are correct here:**
- f-strings build the **user message** with data from previous steps
- This is task input, not system context injection
- Each step's output becomes the next step's input - that's prompt chaining
- `@system_prompt` would be wrong: step data isn't persistent agent context

**No changes needed.**

**Keep:** All models, all agents, chain execution logic

---

### 3.7 routing.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❌ f-strings build user messages (task input) → `@system_prompt` not needed
- ❌ No shared state between router and handlers → `deps_type` not needed
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Already idiomatic. No changes needed.

**Why f-strings are correct here:**
- `f"Classify the intent of this customer query:\n\n{user_query}"` - user message
- `f"Handle this customer query:\n\n{user_query}"` - user message
- Same pattern as prompt_chaining: constructing task input, not system context

**Keep:** All models, `router_agent`, handler agents, `INTENT_HANDLERS` mapping

---

### 3.8 parallelization.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❌ f-strings build user messages (task inputs) → `@system_prompt` not needed
- ❌ Parallel workers are independent → `deps_type` not needed
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Already idiomatic. No changes needed.

**Why f-strings are correct here:**
- `f"Topic: {topic}\nSection focus: {section_name}..."` - task input for worker
- `f"Synthesize these section results:\n\n{sections_text}"` - task input
- `f"Voter {voter_id}: Answer this question:\n{question}"` - task input
- All construct what the user/orchestrator is asking each worker to do

**Keep:** All parallel patterns (sectioning, voting, map-reduce),
`asyncio.gather()`, all models

---

### 3.9 tool_use.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❌ No persistent context → `@system_prompt` not needed
- ✅ Shared services (calculator, etc.) → has `deps_type`
- ✅ Dynamic tool calls → has `@tool`

**Assessment:** Already idiomatic. Reference implementation.

**Current state (already idiomatic):**
- Has `deps_type=ToolDependencies`
- Has `RunContext[ToolDependencies]` in tools
- Uses `@agent.tool` and `@agent.tool_plain` decorators

**No changes needed.** Use as template for other patterns.

---

### 3.10 multi_agent.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❓ f-strings may be task assignment → evaluate case-by-case
- ✅ Collaboration state → already has `deps_type=CollaborationContext`
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Mostly idiomatic. Already uses `RunContext` and `deps_type`.
Review f-string usage - likely user messages for task assignment.

**Evaluate:** Are f-strings building task assignments (user messages) or
injecting collaboration context (system prompt)?

**Keep:** `CollaborationContext` deps, supervisor/worker structure, all models

---

### 3.11 guardrails.py

**Decision Framework:**
- ❌ Guardrails are pass/fail gates, not retry mechanisms → `@output_validator`
  not appropriate (toxic content won't improve with retry)
- ❌ Single f-string builds user message → `@system_prompt` not needed
- ❌ Guardrail state not shared with agent → `deps_type` not needed
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Already idiomatic. No changes needed.

**Why guardrails don't use `@output_validator`:**
- `@output_validator` is for quality improvement with retry feedback
- Guardrails are safety gates that block/filter, not improve
- If content is toxic, retrying won't produce safe content
- Guardrails intentionally run AFTER the main agent as separate validation

**Why this is a Python pattern, not an agent pattern:**
- Core logic is regex/pattern matching (`InputGuardrail`, `OutputGuardrail`)
- Agents are simple pass-through (`safety_agent`, `task_agent`)
- `GuardedExecutor` orchestrates Python checks around agent calls

**Keep:** All guardrail classes, all models, `safety_agent`, `task_agent`

---

### 3.12 learning.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ✅ Injects past experiences into context → needs `@system_prompt`
- ✅ ExperienceStore is shared service → needs `deps_type`
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Needs refactoring. f-strings inject experience context into
system prompt - should use `@system_prompt` decorator.

**Target approach:**
- `LearningDeps` with `experience_store`
- `@system_prompt` to inject relevant experiences
- Pass store via deps instead of building prompt in function

**Keep:** `ExperienceStore`, all learning models

---

### 3.13 resource_aware.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ✅ Injects budget/limits into context → needs `@system_prompt`
- ✅ Budget and tracker are shared state → needs `deps_type`
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Needs refactoring. Budget context should use `@system_prompt`
and deps pattern.

**Target approach:**
- `ResourceDeps` with `budget`, `usage_tracker`
- `@system_prompt` to inject resource constraints
- Pass trackers via deps instead of function parameters

**Keep:** `ResourceBudget`, `UsageTracker`, complexity estimation logic

---

### 3.14 evaluation.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❓ Light f-strings → evaluate if user messages or context
- ❓ Could add deps for evaluation config → optional
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Mostly idiomatic. `LLMJudge` already well-structured. Review
light f-string usage but likely already correct.

**Keep:** `LLMJudge` agent, all evaluation models and metrics

---

### 3.15 prioritization.py

**Decision Framework:**
- ❌ No retry loop → `@output_validator` not needed
- ❓ Inline system prompt in `_get_agent` → could extract to decorator
- ❓ Could add deps for priority config → optional
- ❌ No dynamic fetching → `@tool` not needed

**Assessment:** Review needed. Inline system prompt construction could use
`@system_prompt` decorator for cleaner separation. Light changes.

**Keep:** Prioritization logic, all models

---

## 4. Test Updates Required

### Full Refactor Patterns

| Test File | Changes |
|-----------|---------|
| `test_reflection.py` | Remove `reflect_once` tests. Update `run_reflection` to not pass `max_iterations`/`acceptable_score` as positional args - use `ReflectionDeps` instead. Mock `output_validator` behavior. |
| `test_human_in_loop.py` | Remove `auto_confidence` parameter from test calls. Mock `task_agent` to return `AgentOutput` directly. |
| `test_memory.py` | Minimal changes - API signature preserved. May need to mock `@system_prompt` injection. |
| `test_knowledge_retrieval.py` | Remove `expand_query` parameter tests. Add mocks for `search_knowledge` tool. Update `RAGPipeline` tests for simplified API. |
| `test_planning.py` | Update function signatures for new deps-based API. Mock `@output_validator` for step execution. |

### Minor Change Patterns

| Test File | Changes |
|-----------|---------|
| `test_prompt_chaining.py` | Update mocks if deps added. Minimal API changes. |
| `test_routing.py` | Minimal changes - mostly internal refactoring. |
| `test_parallelization.py` | No API changes - tests should pass as-is. |
| `test_tool_use.py` | **No changes needed** - already idiomatic. |
| `test_multi_agent.py` | Minimal changes - deps already in use. |
| `test_guardrails.py` | Minimal changes if deps added. |
| `test_learning.py` | Update mocks for deps-based experience injection. |
| `test_resource_aware.py` | Update mocks for deps-based budget tracking. |
| `test_evaluation.py` | Minimal changes. |
| `test_prioritization.py` | Minimal changes. |

---

## 5. Verification Checklist

For each refactored pattern, verify:

- [ ] `uv run ruff check src/agentic_patterns/{pattern}.py` passes
- [ ] `uv run ruff format src/agentic_patterns/{pattern}.py` passes
- [ ] `uv run pytest tests/test_{pattern}.py` passes
- [ ] `.venv/bin/python -m agentic_patterns.{pattern}` demo runs
- [ ] No manual retry loops (except `self_reflect` in reflection.py)
- [ ] No f-string prompt construction in function bodies
- [ ] All runtime state passed via `deps`
- [ ] `deps_type` generic matches `RunContext` type parameter

---

## 6. Definition of Done

A pattern is "Idiomatic PydanticAI" when:

1. **No manual string concatenation** for context injection - use
   `@system_prompt`
2. **No manual retry loops** for validation - use `@output_validator` +
   `ModelRetry`
3. **Models define output structure** - use `output_type`, not Python wrappers
4. **All state via deps** - runtime configuration passed through `deps_type`

---

## 7. Implementation Order

### Phase 1: Full Refactors (establish patterns)

1. **reflection.py** - Establishes `@output_validator` + `ModelRetry` pattern
2. **human_in_loop.py** - Demonstrates `output_type` for structured responses
3. **memory.py** - Clean `@system_prompt` with deps demonstration
4. **knowledge_retrieval.py** - Tool-based retrieval pattern
5. **planning.py** - Complex `@output_validator` with state management

### Phase 2: Minor Changes (apply patterns)

6. **prompt_chaining.py** - Add deps + `@system_prompt`
7. **multi_agent.py** - Replace f-string prompts (deps already good)
8. **learning.py** - Add deps for experience store
9. **resource_aware.py** - Add deps for budget tracking
10. **guardrails.py** - Extract prompts to decorators
11. **routing.py** - Minor prompt cleanup
12. **parallelization.py** - Minor prompt cleanup
13. **evaluation.py** - Minor prompt cleanup
14. **prioritization.py** - Extract inline prompts

### Phase 3: Verification

15. **tool_use.py** - No changes, verify still works as reference

---

## 8. Pattern Assessment Summary

| Pattern | Assessment | Effort |
|---------|------------|--------|
| reflection | Needs refactor (`@output_validator`) | High |
| human_in_loop | Needs refactor (`output_type`) | Medium |
| memory | **Already idiomatic** | None |
| knowledge_retrieval | Needs refactor (`@tool`) | Medium |
| planning | Needs refactor (`deps` + `@system_prompt`) | High |
| prompt_chaining | **Already idiomatic** | None |
| routing | **Already idiomatic** | None |
| parallelization | **Already idiomatic** | None |
| tool_use | **Already idiomatic** | None |
| multi_agent | Mostly idiomatic (has `deps`) | Low |
| guardrails | **Already idiomatic** | None |
| learning | Needs refactor (`@system_prompt` + `deps`) | Medium |
| resource_aware | Needs refactor (`@system_prompt` + `deps`) | Medium |
| evaluation | Mostly idiomatic | Low |
| prioritization | Review needed | Low |

---

## 9. Pre-Implementation Protocol

Before implementing any pattern refactoring, follow this protocol to ensure
alignment between current code and target architecture.

### 8.1 Generate Gap Analysis

For each pattern, the implementing agent MUST:

1. **Read the current source file**:
   ```bash
   # Read entire source
   cat src/agentic_patterns/{pattern}.py
   ```

2. **Read the corresponding test file**:
   ```bash
   cat tests/test_{pattern}.py
   ```

3. **Read the documentation** (if exists):
   ```bash
   cat docs/patterns/*{pattern}*.md
   ```

4. **Compare against spec section 2.X** for this pattern

5. **Generate a gap analysis** answering:
   - What does current code do vs. what spec requires?
   - Which functions/classes need modification?
   - Which functions/classes need removal?
   - Which tests will break and need updates?
   - Are there any API compatibility concerns?

### 8.2 Write Implementation Plan

Create a plan file at `docs/plans/{pattern}-refactor.md` with:

```markdown
# {Pattern} Refactoring Plan

## Current State
- Brief description of current implementation
- Key deviations from spec

## Gap Analysis

### Code Changes Required
| File | Location | Current | Target | Action |
|------|----------|---------|--------|--------|
| {file} | line X-Y | description | spec requirement | add/modify/remove |

### Test Changes Required
| Test File | Test Name | Change Required |
|-----------|-----------|-----------------|
| test_{pattern}.py | test_xyz | description |

## Implementation Steps
1. Step one
2. Step two
...

## Verification
- [ ] Lint passes
- [ ] Tests pass
- [ ] Demo runs
```

### 8.3 Execute Implementation

After plan is written:
1. Implement changes per plan
2. Verify each checklist item from Section 4
3. Run tests and fix any failures

### 8.4 Cleanup

After pattern is verified complete:
1. **Delete the plan file**: `rm docs/plans/{pattern}-refactor.md`
2. Mark pattern as done in any tracking system

### 8.5 Rationale

This protocol ensures:
- **No blind implementation**: Agent understands current state before changing
- **Traceable decisions**: Plan documents why changes were made
- **Clean workspace**: Ephemeral plans don't clutter repo long-term
- **Consistent approach**: Each pattern follows same process
