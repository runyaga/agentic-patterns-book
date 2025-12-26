# Standard Operating Procedure: Agentic Refactoring

**Objective:** Systematically refactor all agentic patterns to strictly adhere to idiomatic PydanticAI architecture, maximizing the use of library features over manual Python control flow.

## The Idiomatic Standard

For every pattern, we apply these four rules:

1.  **State via Dependencies (`Deps`)**:
    *   *Anti-Pattern:* Passing context (history, plans, docs) as string arguments to functions.
    *   *Standard:* Define a `Dataclass` or `Pydantic Model` for state. Pass it via `agent.run(..., deps=deps)`.

2.  **Context via System Prompts (`@system_prompt`)**:
    *   *Anti-Pattern:* Manually formatting prompt strings: `f"Context: {data}\nInput: {user_input}" `.
    *   *Standard:* Use `@agent.system_prompt` to inject formatted context from `ctx.deps` automatically.

3.  **Flow via Native Mechanics (`ModelRetry`, `Validator`)**:
    *   *Anti-Pattern:* Manual `while` loops for validation, correction, or reflection.
    *   *Standard:* Use `@agent.result_validator` to check logic/scores and raise `ModelRetry` to trigger the library's internal loop.

4.  **Structure via Models (`output_type`)**:
    *   *Anti-Pattern:* Wrapper classes that add metadata (like fake confidence) *after* the agent returns.
    *   *Standard:* Define the full schema in `output_type` so the LLM populates all fields (including confidence/reasoning) natively.

---

## Workflow Per File

For each target file in `src/agentic_patterns/`:

1.  **Analyze**: Read the file. Map "Anti-Patterns" to "Standards".
2.  **Design**:
    *   Define the `Deps` class (if missing).
    *   Identify which logic moves to `@system_prompt`.
    *   Identify which loops move to `result_validator`.
3.  **Refactor**: Apply changes using `pydantic_ai` imports (`RunContext`, `ModelRetry`).
4.  **Verify**: Run the file (`python -m agentic_patterns.pattern_name`). Ensure `Exit Code: 0` and correct behavior.
5.  **Finalize**: Ensure no "string stuffing" remains in the `run` calls.

---

## Order of Operations

We will proceed through the patterns in complexity order:

1.  **Foundation (Context Injection)**
    *   `memory.py` (History via Deps)
    *   `knowledge_retrieval.py` (RAG via Deps)
    *   `learning.py` (Examples via Deps)

2.  **Control Flow (Deep Mechanics)**
    *   `reflection.py` (Loop -> ModelRetry)
    *   `human_in_loop.py` (Wrapper -> Output Model)
    *   `planning.py` (State passing -> RunContext)

3.  **Coordination (Multi-Agent)**
    *   `prompt_chaining.py` (Clean up string passing)
    *   `routing.py` (Verify Deps usage)
    *   `parallelization.py` (Verify Deps usage)
    *   `multi_agent.py` (Shared Context via Deps)
    *   `tool_use.py` (Verify Gold Standard)
