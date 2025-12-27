# Specification: The Phoenix Protocol (Exception Handling)

**Chapter:** 12
**Pattern Name:** The Phoenix Protocol
**Status:** Draft
**Related:** `src/agentic_patterns/exception_recovery.py`

## 1. Overview
Standard exception handling (`try/except`) is insufficient for agents that may fail due to semantic errors (hallucinations), context overflows, or tool misuse. The **Phoenix Protocol** introduces a "Clinic Agent" that treats failures as clinical cases to be diagnosed and healed.

## 2. Core Concept
Instead of a simple retry loop, we implement a **Recovery Pipeline**:
1.  **Triage:** Catch the exception and capture the full "Trauma Context" (prompt, history, error, tool state).
2.  **Diagnosis:** A specialized `ClinicAgent` analyzes the context to determine the *root cause*.
3.  **Prescription:** The Clinic modifies the input (e.g., summarizing history to fix context overflow, clarifying the prompt to fix hallucinations) or patches the agent's instructions.
4.  **Rehabilitation:** The task is retried with the prescribed changes.

## 3. Architecture

### 3.1 Data Models
```python
class TraumaContext(BaseModel):
    """Snapshot of the agent's state at the moment of failure."""
    original_prompt: str
    error_type: str
    error_message: str
    stack_trace: str
    tool_calls: list[ToolCall] | None
    history_summary: str | None  # If history was too long

class Diagnosis(BaseModel):
    """The Clinic's analysis of the failure."""
    root_cause: Literal[
        "context_length_exceeded",
        "hallucinated_tool",
        "invalid_tool_arguments",
        "ambiguous_instruction",
        "logic_error",
        "transient_error"
    ]
    explanation: str
    confidence: float

class Prescription(BaseModel):
    """Actionable steps to recover."""
    action: Literal["retry", "modify_prompt", "summarize_history", "abort"]
    modified_prompt: str | None
    modified_system_prompt: str | None
```

### 3.2 The Clinic Agent
A specialized `pydantic-ai` agent with the system prompt:
> "You are an expert AI debugger and psychologist. Your goal is to analyze why a worker agent failed and prescribe a fix. You do not execute the task yourself; you repair the request."

### 3.3 The Wrapper
A utility function `recoverable_run` that wraps any `Agent.run()` call:

```python
async def recoverable_run(
    agent: Agent,
    prompt: str,
    max_recoveries: int = 2,
    deps: Any = None
) -> RunResult:
    # Implementation logic...
```

## 4. Implementation Details

### 4.1 File Structure
*   `src/agentic_patterns/exception_recovery.py`: Core logic.

### 4.2 Dependencies
*   `pydantic-ai` (Core framework)
*   `traceback` (For stack traces)

### 4.3 Recovery Strategies
*   **Context Overflow:** Use a `Summarizer` (simple LLM call) to compress the message history.
*   **Hallucinated Tool:** Inject a specific system prompt reminder listing the *actual* available tools.
*   **Invalid Arguments:** Rewrite the user prompt to be more explicit about the required format.

## 5. Documentation Plan
*   **Mermaid Diagram:** State machine showing Healthy -> Injured -> Clinic -> Healthy.
*   **Example:** A script where an agent is forced to fail (e.g., asked to use a non-existent tool), and the Clinic automatically fixes the prompt to use a valid tool.
