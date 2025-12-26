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

## 2. Pattern Specifications

### 2.1 reflection.py

**Current state:** Manual `while` loop with separate producer/critic/refiner
agents.

**Target:** `@output_validator` on producer calls critic internally.

#### Deps Definition

```python
from dataclasses import dataclass

@dataclass
class ReflectionDeps:
    """Dependencies for reflection pattern."""
    acceptable_score: float = 8.0
```

#### Target Architecture

```python
from pydantic_ai import Agent, RunContext, ModelRetry

model = get_model()

# Producer agent with deps
producer_agent = Agent(
    model,
    system_prompt=(
        "You are a skilled content producer. Generate high-quality content "
        "based on the given task. Be thorough, accurate, and well-structured. "
        "If you receive feedback from a previous attempt, incorporate it to "
        "improve your output significantly."
    ),
    output_type=ProducerOutput,
    deps_type=ReflectionDeps,
)

# Critic agent (no deps needed)
critic_agent = Agent(
    model,
    system_prompt=(
        "You are a critical reviewer. Evaluate the given content objectively. "
        "Score from 0-10 where 8+ means acceptable quality. "
        "Identify specific strengths and weaknesses. "
        "Provide actionable suggestions for improvement."
    ),
    output_type=Critique,
)


@producer_agent.output_validator
async def validate_quality(
    ctx: RunContext[ReflectionDeps],
    output: ProducerOutput,
) -> ProducerOutput:
    """Validate output quality via critic agent."""
    critique_result = await critic_agent.run(
        f"Evaluate this content:\n\n{output.content}"
    )
    critique = critique_result.output

    if critique.score < ctx.deps.acceptable_score:
        suggestions = "\n".join(f"- {s}" for s in critique.suggestions)
        weaknesses = "\n".join(f"- {w}" for w in critique.weaknesses)
        raise ModelRetry(
            f"Score: {critique.score}/10 (need {ctx.deps.acceptable_score}+)\n"
            f"Weaknesses:\n{weaknesses}\n"
            f"Suggestions:\n{suggestions}\n"
            f"Please improve the content addressing these issues."
        )
    return output
```

#### Simplified Public API

```python
async def run_reflection(
    task: str,
    deps: ReflectionDeps | None = None,
) -> ReflectionResult:
    """
    Run reflection with automatic quality validation.

    Args:
        task: The task/prompt for content generation.
        deps: Optional ReflectionDeps (defaults to acceptable_score=8.0).

    Returns:
        ReflectionResult with final content and metadata.
    """
    deps = deps or ReflectionDeps()
    result = await producer_agent.run(task, deps=deps)

    return ReflectionResult(
        final_content=result.output.content,
        iterations=len(result.all_messages()),  # Approximate
        final_score=deps.acceptable_score,  # Met threshold
        improvement_history=[],
        converged=True,
    )
```

#### What to Remove

- `reflect_once()` function - logic moves to validator
- `refiner_agent` - producer handles refinement via retry
- Manual `while` loop in `run_reflection`
- `max_iterations` parameter (PydanticAI handles retry limits)

#### What to Keep

- `self_reflect()` function - different pattern, keep manual loop
- All Pydantic models: `ProducerOutput`, `Critique`, `RefinedOutput`,
  `ReflectionResult`
- `critic_agent` - called from validator

---

### 2.2 human_in_loop.py

**Current state:** `task_agent` returns `str`, wrapper assigns fake confidence.

**Target:** `task_agent` returns `AgentOutput` with model-generated confidence.

#### Change task_agent

```python
# BEFORE:
task_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful assistant. Complete the given task to the best "
        "of your ability. Be clear about any uncertainties or limitations "
        "in your response. If you're not confident, say so."
    ),
    output_type=str,
)

# AFTER:
task_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful assistant. Complete tasks and self-assess your "
        "confidence in your response.\n\n"
        "Rate your confidence 0.0-1.0 based on:\n"
        "- 0.9-1.0: Certain, factual, well-documented information\n"
        "- 0.7-0.9: Confident but some assumptions or uncertainty\n"
        "- 0.5-0.7: Moderate confidence, may need verification\n"
        "- Below 0.5: Low confidence, likely needs human review\n\n"
        "Always explain your reasoning for the confidence level."
    ),
    output_type=AgentOutput,  # content + confidence + reasoning
)
```

#### Simplified execute_with_oversight

```python
async def execute_with_oversight(
    task: str,
    policy: EscalationPolicy,
    workflow: ApprovalWorkflow,
    task_type: str = "",
) -> tuple[str, bool, EscalationRequest | None]:
    """
    Execute a task with human oversight based on policy.

    Args:
        task: The task to execute.
        policy: Escalation policy to apply.
        workflow: Approval workflow to use.
        task_type: Type of task for policy checks.

    Returns:
        Tuple of (result, was_escalated, escalation_request).
    """
    print(f"Executing task: {task[:50]}...")

    result = await task_agent.run(task)
    output = result.output  # AgentOutput with real confidence

    should_escalate, reason = policy.should_escalate(output, task_type)

    if should_escalate and reason:
        print(f"  Escalating: {reason.value}")
        request = workflow.submit_for_review(
            output=output,
            task_description=task,
            reason=reason,
        )
        return output.content, True, request

    print("  Auto-approved")
    workflow.task_counter += 1
    return output.content, False, None
```

#### What to Remove

- `auto_confidence` parameter from `execute_with_oversight`
- Manual `AgentOutput` construction with fake confidence

#### What to Keep

- All models: `AgentOutput`, `EscalationPolicy`, `ApprovalWorkflow`, etc.
- `confidence_evaluator` agent (may be useful for validation)
- `decision_agent` and `augment_decision()` function

---

### 2.3 memory.py

**Current state:** f-string concatenation builds prompt with history.

**Target:** `@system_prompt` decorator injects history from deps.

#### Deps Definition

```python
from dataclasses import dataclass
from typing import Union

MemoryType = Union[BufferMemory, WindowMemory, SummaryMemory]


@dataclass
class MemoryDeps:
    """Dependencies for memory-enabled conversation."""
    memory: MemoryType
```

#### Target Architecture

```python
from pydantic_ai import Agent, RunContext

model = get_model()

conversational_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful assistant with conversation memory. "
        "Use the provided context from previous messages to maintain "
        "continuity and provide relevant responses. "
        "Reference previous topics when appropriate."
    ),
    deps_type=MemoryDeps,
    output_type=str,
)


@conversational_agent.system_prompt
def inject_memory_context(ctx: RunContext[MemoryDeps]) -> str:
    """Inject conversation history into system prompt."""
    context = ctx.deps.memory.get_context()
    if not context:
        return ""
    return (
        f"\nConversation history:\n{context}\n\n"
        f"Respond to the latest user message while maintaining context."
    )
```

#### Simplified Public API

```python
async def chat_with_memory(
    memory: MemoryType,
    user_input: str,
) -> str:
    """
    Process a user message with memory context.

    Args:
        memory: Memory instance to use for context.
        user_input: The user's message.

    Returns:
        AI response string.
    """
    memory.add_user_message(user_input)
    deps = MemoryDeps(memory=memory)

    result = await conversational_agent.run(user_input, deps=deps)
    response = result.output

    memory.add_ai_message(response)
    return response
```

#### What Changes

- No f-string prompt building in `chat_with_memory` body
- Context injection via `@system_prompt` decorator
- Deps pattern for memory passing

#### What to Keep

- All memory classes: `BufferMemory`, `WindowMemory`, `SummaryMemory`
- All models: `MemoryMessage`, `ConversationSummary`, `MemoryStats`
- `summarizer_agent` and `SummaryMemory.summarize()` method
- `run_conversation()` demo function

---

### 2.4 knowledge_retrieval.py

**Current state:** RAGPipeline builds prompt with retrieved context via
f-string.

**Target:** Tool-based retrieval during generation.

#### Deps Definition

```python
from dataclasses import dataclass


@dataclass
class RAGDeps:
    """Dependencies for RAG pattern."""
    store: VectorStore
    top_k: int = 3
    min_score: float = 0.1
```

#### Target Architecture

```python
from pydantic_ai import Agent, RunContext

model = get_model()

rag_agent = Agent(
    model,
    system_prompt=(
        "You answer questions using retrieved context from a knowledge base. "
        "ALWAYS use the search_knowledge tool to find relevant information "
        "before answering. Base your answer ONLY on the retrieved content. "
        "If no relevant information is found, say so clearly."
    ),
    deps_type=RAGDeps,
    output_type=str,
)


@rag_agent.tool
async def search_knowledge(
    ctx: RunContext[RAGDeps],
    query: str,
) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        query: The search query to find relevant content.

    Returns:
        Retrieved context from the knowledge base.
    """
    chunks = ctx.deps.store.search(
        query=query,
        k=ctx.deps.top_k,
        threshold=ctx.deps.min_score,
    )

    if not chunks:
        return "No relevant information found in the knowledge base."

    context_parts = []
    for rc in chunks:
        source = rc.chunk.metadata.get("title", "unknown")
        context_parts.append(f"[Source: {source}]\n{rc.chunk.content}")

    return "\n\n---\n\n".join(context_parts)
```

#### Simplified RAGPipeline

```python
@dataclass
class RAGPipeline:
    """RAG pipeline using tool-based retrieval."""

    store: VectorStore
    top_k: int = 3
    min_score: float = 0.1

    async def query(self, question: str) -> RAGResponse:
        """
        Process a question through the RAG pipeline.

        Args:
            question: User's question.

        Returns:
            RAGResponse with answer and metadata.
        """
        deps = RAGDeps(
            store=self.store,
            top_k=self.top_k,
            min_score=self.min_score,
        )

        result = await rag_agent.run(question, deps=deps)

        return RAGResponse(
            answer=result.output,
            query=question,
            context_chunks=[],  # Tool handles retrieval internally
            confidence=0.8,  # Could extract from result metadata
        )

    async def batch_query(self, questions: list[str]) -> list[RAGResponse]:
        """Process multiple questions."""
        return [await self.query(q) for q in questions]
```

#### What to Remove

- `expand_query` parameter (agent decides when to refine)
- `query_agent` (no longer needed)
- `_format_context()` method (tool handles formatting)
- Manual prompt construction in `query()` method

#### What to Keep

- All utility functions: `simple_embedding()`, `cosine_similarity()`,
  `chunk_text()`
- All models: `Document`, `Chunk`, `RetrievedChunk`, `RAGResponse`, etc.
- `VectorStore` class
- `build_knowledge_base()` function

---

### 2.5 planning.py

**Current state:** Manual for-loop executes plan steps with f-string prompts.

**Target:** `@output_validator` for step execution with RunContext state.

**Assessment:** Needs full refactor.

#### Deps Definition

```python
@dataclass
class PlanningDeps:
    """Dependencies for planning pattern."""
    plan: Plan | None = None
    completed_steps: list[StepResult] = field(default_factory=list)
    max_replans: int = 2
```

#### Changes Required

1. Add `deps_type=PlanningDeps` to `executor_agent`
2. Replace f-string prompts (lines 151-154, 179-183, 217-221, 296-299) with
   `@system_prompt` decorators
3. Convert step execution loop to use `@output_validator` for iterative
   validation
4. Pass plan state via deps instead of function parameters

#### What to Keep

- All Pydantic models: `Plan`, `PlanStep`, `StepResult`, `ExecutionResult`
- `planner_agent`, `executor_agent`, `replanner_agent`
- `create_plan()`, `execute_plan()`, `run_planning()` functions (with new
  signatures)

---

### 2.6 prompt_chaining.py

**Current state:** Heavy f-string prompt building (lines 112-141).

**Target:** `@system_prompt` decorator for chain context injection.

**Assessment:** Minor changes.

#### Deps Definition

```python
@dataclass
class ChainDeps:
    """Dependencies for prompt chaining."""
    previous_outputs: list[str] = field(default_factory=list)
    step_index: int = 0
```

#### Changes Required

1. Add `deps_type=ChainDeps` to chain agents
2. Replace f-string prompt building with `@system_prompt`:
   ```python
   @chain_agent.system_prompt
   def inject_chain_context(ctx: RunContext[ChainDeps]) -> str:
       if not ctx.deps.previous_outputs:
           return ""
       prev = "\n".join(ctx.deps.previous_outputs)
       return f"\nPrevious steps output:\n{prev}"
   ```

#### What to Keep

- All models and existing structure
- Chain execution logic

---

### 2.7 routing.py

**Current state:** Moderate f-string prompts (lines 194, 205).

**Target:** `@system_prompt` for handler context.

**Assessment:** Minor changes.

#### Changes Required

1. Replace f-string prompts in handler functions with `@system_prompt`
2. Optionally add `deps_type` for route tracking state

#### What to Keep

- `RouteDecision`, `RouteHandler` models
- `router_agent`, intent classification logic
- All handler agents

---

### 2.8 parallelization.py

**Current state:** Moderate f-string prompts (lines 152-154, 202-203, 252-254).

**Target:** `@system_prompt` for parallel task context.

**Assessment:** Minor changes.

#### Changes Required

1. Replace f-string prompts with `@system_prompt` decorators
2. Already properly async/concurrent - no structural changes needed

#### What to Keep

- All parallel execution patterns (sectioning, voting, map-reduce)
- `asyncio.gather()` usage
- All models

---

### 2.9 tool_use.py

**Current state:** Already uses `RunContext[ToolDependencies]` and `@tool`
decorators properly.

**Target:** No changes needed.

**Assessment:** Already idiomatic - reference implementation.

#### Reference Patterns

This file demonstrates correct usage of:
- `deps_type=ToolDependencies` (line 86)
- `RunContext[ToolDependencies]` in tools (lines 132-135)
- `@agent.tool` and `@agent.tool_plain` decorators

Use as template for other pattern refactoring.

---

### 2.10 multi_agent.py

**Current state:** Good RunContext usage, moderate f-string prompts (lines
284-287, 328-332, 369-371).

**Target:** Replace f-string prompts with `@system_prompt`.

**Assessment:** Minor changes.

#### Changes Required

1. Replace f-string prompt building with `@system_prompt` decorators
2. Already good use of `RunContext` and `deps_type=CollaborationContext`

#### What to Keep

- `CollaborationContext` deps pattern (already good)
- Supervisor/worker agent structure
- All models

---

### 2.11 guardrails.py

**Current state:** Moderate f-string prompts (lines 511, 522).

**Target:** `@system_prompt` for guardrail context.

**Assessment:** Minor changes.

#### Deps Definition (Optional)

```python
@dataclass
class GuardrailDeps:
    """Dependencies for guardrail tracking."""
    violation_log: list[str] = field(default_factory=list)
```

#### Changes Required

1. Extract prompt text to `@system_prompt` decorators
2. Optionally add `RunContext` for violation log state

#### What to Keep

- All guardrail models and validation logic
- Input/output filter agents

---

### 2.12 learning.py

**Current state:** Moderate f-string prompts (lines 424, 450-452, 480-483).

**Target:** `@system_prompt` with RunContext for experience store.

**Assessment:** Minor changes.

#### Deps Definition

```python
@dataclass
class LearningDeps:
    """Dependencies for learning pattern."""
    experience_store: ExperienceStore
```

#### Changes Required

1. Add `deps_type=LearningDeps` to learning agent
2. Replace f-string prompts with `@system_prompt`:
   ```python
   @learning_agent.system_prompt
   def inject_experience(ctx: RunContext[LearningDeps]) -> str:
       relevant = ctx.deps.experience_store.get_relevant(...)
       return f"\nRelevant past experiences:\n{relevant}"
   ```

#### What to Keep

- `ExperienceStore` class
- All learning models

---

### 2.13 resource_aware.py

**Current state:** Moderate f-string prompts (lines 442-443, 511).

**Target:** `@system_prompt` with RunContext for budget tracking.

**Assessment:** Minor changes.

#### Deps Definition

```python
@dataclass
class ResourceDeps:
    """Dependencies for resource-aware execution."""
    budget: ResourceBudget
    usage_tracker: UsageTracker
```

#### Changes Required

1. Add `deps_type=ResourceDeps` to resource-aware agents
2. Replace complexity assessment prompt with `@system_prompt`
3. Use RunContext for budget state instead of passing around objects

#### What to Keep

- `ResourceBudget`, `UsageTracker` classes
- Complexity estimation logic

---

### 2.14 evaluation.py

**Current state:** Light f-string prompts (lines 548, 641-643). LLMJudge
prompt already in system_prompt.

**Target:** Minor `@system_prompt` cleanup.

**Assessment:** Minor changes.

#### Changes Required

1. Small f-string prompts could move to `@system_prompt`
2. Overall clean usage - minimal changes needed

#### What to Keep

- `LLMJudge` agent (already well-structured)
- All evaluation models and metrics

---

### 2.15 prioritization.py

**Current state:** Light f-string prompts (lines 641-643). System prompt
inline in `_get_agent` (lines 607-620).

**Target:** Extract to `@system_prompt` decorator.

**Assessment:** Minor changes.

#### Changes Required

1. Move inline system prompt to `@system_prompt` decorator
2. Extract prioritization prompt building to decorator

#### What to Keep

- Prioritization logic
- All models

---

## 3. Test Updates Required

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

## 4. Verification Checklist

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

## 5. Definition of Done

A pattern is "Idiomatic PydanticAI" when:

1. **No manual string concatenation** for context injection - use
   `@system_prompt`
2. **No manual retry loops** for validation - use `@output_validator` +
   `ModelRetry`
3. **Models define output structure** - use `output_type`, not Python wrappers
4. **All state via deps** - runtime configuration passed through `deps_type`

---

## 6. Implementation Order

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

## 7. Pattern Assessment Summary

| Pattern | Assessment | Effort |
|---------|------------|--------|
| reflection | Full refactor | High |
| human_in_loop | Full refactor | Medium |
| memory | Full refactor | Medium |
| knowledge_retrieval | Full refactor | Medium |
| planning | Full refactor | High |
| prompt_chaining | Minor changes | Low |
| routing | Minor changes | Low |
| parallelization | Minor changes | Low |
| tool_use | **Already idiomatic** | None |
| multi_agent | Minor changes | Low |
| guardrails | Minor changes | Low |
| learning | Minor changes | Medium |
| resource_aware | Minor changes | Medium |
| evaluation | Minor changes | Low |
| prioritization | Minor changes | Low |
