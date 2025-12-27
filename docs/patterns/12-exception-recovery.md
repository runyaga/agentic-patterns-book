# Exception Recovery (Phoenix Protocol)

**Chapter 12** · [Source Code](../../src/agentic_patterns/exception_recovery.py)

The **Phoenix Protocol** pattern provides a robust recovery mechanism for agent exceptions. Unlike standard retry loops that blindly restart, this pattern uses a "Clinic Agent" to diagnose the root cause of the failure and prescribe a specific fix (e.g., summarizing history to fix context overflow, or clarifying instructions to fix hallucinations).

## Key Concepts

```mermaid
graph TD
    Start[Agent.run] -->|Success| End[Result]
    Start -->|Exception| Classify{Classify Error}
    
    Classify -->|Timeout/RateLimit| FastFix[Wait & Retry]
    Classify -->|ContextOverflow| FastFix2[Truncate Prompt]
    Classify -->|Unknown Error| Clinic[Clinic Agent]
    
    FastFix --> Start
    FastFix2 --> Start
    
    Clinic -->|Diagnosis: Retry| Start
    Clinic -->|Diagnosis: Abort| Fail[Raise Exception]
```

-   **Deterministic First**: Uses pattern matching to catch common errors (Timeout, Rate Limit) instantly without wasting tokens on an LLM diagnosis.
-   **Clinic Agent**: A specialized LLM agent that acts as a doctor for other agents. It analyzes the error message and "prescribes" a fix.
-   **Smart Retry**: Implements intelligent backoff, truncation, and circuit breaking based on the error type.
-   **Non-Intrusive Wrapper**: Implemented as a simple `recoverable_run()` wrapper function that can be applied to any existing `pydantic-ai` agent.
-   **Full Proxy**: Returns the complete `AgentRunResult` with all metadata (`.output`, `.usage()`, `.all_messages()`).

## Implementation

The implementation follows a "V1 Smart Retry" approach that prioritizes simplicity and token efficiency.

### Recovery Logic

The core logic uses a `recoverable_run` wrapper that catches exceptions and decides whether to retry deterministically or consult the Clinic.

```python
--8<-- "src/agentic_patterns/exception_recovery.py:recoverable_run"
```

### Deterministic Classification

We avoid expensive LLM calls for known errors by using Python heuristics.

```python
--8<-- "src/agentic_patterns/exception_recovery.py:classify"
```

### The Clinic Agent

For unknown or ambiguous errors, we ask the Clinic Agent if a retry is worth it.

```python
--8<-- "src/agentic_patterns/exception_recovery.py:clinic"
```

## Use Cases

1.  **Unreliable APIs**: Wrapping agents that call flaky external tools.
2.  **Long-Running Tasks**: Automatically recovering from timeouts or rate limits.
3.  **Context Management**: Automatically truncating prompts when context limits are hit.
4.  **Production Safety**: preventing a single crashed agent from bringing down the entire application.

## When to Use

| Use Case | Recommended Approach |
| :--- | :--- |
| **Output Validation** | Use `@output_validator` with `ModelRetry`. This handles *semantic* errors where the model ran successfully but produced bad data. |
| **Exception Handling** | Use **Phoenix Protocol**. This handles *runtime* errors (crashes, timeouts, API failures) where `agent.run()` failed. |
| **Complex Workflows** | Use `pydantic_graph`. If recovery requires complex state transitions, a graph is better than a wrapper. |

## Streaming Limitation

`recoverable_run()` only supports `agent.run()`, **NOT** `agent.run_stream()`.

Streaming recovery is fundamentally broken:

| Error Timing | ~% of Cases | Retry Works? |
|--------------|-------------|--------------|
| Before any data | 30% | Yes |
| Mid-stream | 60% | No - duplicates/inconsistency |
| After stream done | 10% | Maybe |

The 60% mid-stream case is fatal: user already printed partial output,
retry re-yields, app state is now wrong.

For streaming with recovery, use the `is_retryable()` helper at the
application level:

```python
from agentic_patterns.exception_recovery import is_retryable

for attempt in range(max_attempts):
    try:
        async with agent.run_stream("prompt") as stream:
            async for chunk in stream.stream_text():
                print(chunk, end="", flush=True)
        break  # Success
    except Exception as e:
        if not is_retryable(e) or attempt == max_attempts - 1:
            raise
        # Clear output, wait, retry at application level
```

## Example

```bash
# Run the included demo
.venv/bin/python -m agentic_patterns.exception_recovery
```

## API Reference

::: agentic_patterns.exception_recovery
    options:
      show_root_heading: true
      members:
        - ErrorCategory
        - RecoveryConfig
        - recoverable_run
        - is_retryable
        - classify_error
        - get_recovery_action
