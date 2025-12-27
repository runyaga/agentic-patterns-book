# Specification: The Phoenix Protocol (Exception Handling)

**Chapter:** 12
**Pattern Name:** The Phoenix Protocol
**Status:** Draft v3 (Simplified)
**Module:** `src/agentic_patterns/exception_recovery.py`

## 1. Overview

When `agent.run()` raises an exception, the default behavior is to propagate it.
The **Phoenix Protocol** wraps agent execution with automatic recovery using
**deterministic heuristics first**, falling back to an LLM diagnosis only when
the error is ambiguous.

### 1.1 Design Philosophy

1. **Deterministic first:** Pattern-match known errors before invoking LLM
2. **Minimal token usage:** Don't send stack traces to LLM unless necessary
3. **Transparent:** Recovery attempts are logged, not hidden
4. **Simple API:** Single `recoverable_run()` function

### 1.2 Distinction from ModelRetry

| Mechanism | Handles | Triggered By |
|-----------|---------|--------------|
| `ModelRetry` | Semantic failures during generation | `@output_validator` |
| Phoenix | Python exceptions from `agent.run()` | `try/except` wrapper |

**Use ModelRetry when:** Output quality is wrong but execution succeeded.

**Use Phoenix when:** `agent.run()` itself raised an exception.

## 2. Architecture

### 2.1 Recovery Flow

```
agent.run() raises Exception
        │
        ▼
┌───────────────────┐
│ Classify Error    │  ← Deterministic pattern matching
│ (no LLM call)     │
└─────────┬─────────┘
          │
          ▼
    ┌─────────────┐
    │ Known Type? │
    └──────┬──────┘
           │
     Yes   │   No
     ▼     │   ▼
┌─────────┐│┌──────────────┐
│ Apply   │││ Clinic Agent │  ← LLM only for ambiguous
│ Fix     │││ (diagnose)   │
└────┬────┘│└──────┬───────┘
     │     │       │
     ▼     ▼       ▼
   Retry agent.run() with fix
           │
           ▼
    Success or max_retries
```

### 2.2 Error Classification (Deterministic)

```python
from enum import Enum

class ErrorCategory(str, Enum):
    """Known error categories with deterministic fixes."""
    CONTEXT_LENGTH = "context_length"      # Truncate/summarize
    TIMEOUT = "timeout"                    # Retry with shorter prompt
    RATE_LIMIT = "rate_limit"              # Wait and retry
    INVALID_JSON = "invalid_json"          # Retry (model glitch)
    CONNECTION = "connection"              # Retry with backoff
    TOOL_ERROR = "tool_error"              # Retry without tool result
    UNKNOWN = "unknown"                    # Needs LLM diagnosis


def classify_error(exc: Exception) -> ErrorCategory:
    """
    Classify exception into known category using pattern matching.
    No LLM call - pure Python heuristics.
    """
    msg = str(exc).lower()
    exc_type = type(exc).__name__

    # Context length patterns
    if any(p in msg for p in ["context length", "too many tokens", "maximum context"]):
        return ErrorCategory.CONTEXT_LENGTH

    # Timeout patterns
    if "timeout" in msg or exc_type in ("TimeoutError", "asyncio.TimeoutError"):
        return ErrorCategory.TIMEOUT

    # Rate limit patterns
    if any(p in msg for p in ["rate limit", "429", "too many requests"]):
        return ErrorCategory.RATE_LIMIT

    # JSON parsing (model returned malformed output)
    if "json" in msg and any(p in msg for p in ["decode", "parse", "invalid"]):
        return ErrorCategory.INVALID_JSON

    # Connection errors
    if any(p in msg for p in ["connection", "network", "refused", "reset"]):
        return ErrorCategory.CONNECTION

    # Tool execution errors (pydantic-ai specific)
    if "tool" in msg and any(p in msg for p in ["failed", "error", "exception"]):
        return ErrorCategory.TOOL_ERROR

    return ErrorCategory.UNKNOWN
```

### 2.3 Data Models

```python
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field


class RecoveryAction(BaseModel):
    """What to do to recover from the error."""
    action: Literal[
        "retry",              # Retry as-is (transient error)
        "retry_shorter",      # Retry with truncated prompt
        "wait_and_retry",     # Rate limit - wait then retry
        "abort",              # Give up
    ]
    wait_seconds: float = 0.0
    truncate_to: int | None = None  # Target prompt length
    reason: str = ""


class RecoveryResult(BaseModel):
    """Outcome of recovery attempt."""
    success: bool
    attempts: int
    categories_seen: list[str] = Field(default_factory=list)
    final_error: str | None = None


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    rate_limit_wait: float = 60.0
    truncate_ratio: float = 0.7  # Keep 70% of prompt on truncation
    use_clinic_for_unknown: bool = True  # Call LLM for unknown errors
```

### 2.4 Recovery Strategies (Deterministic)

```python
import asyncio

def get_recovery_action(
    category: ErrorCategory,
    attempt: int,
    config: RecoveryConfig,
) -> RecoveryAction:
    """
    Determine recovery action for error category.
    Pure Python - no LLM call.
    """
    if attempt >= config.max_attempts:
        return RecoveryAction(action="abort", reason="Max attempts reached")

    match category:
        case ErrorCategory.CONTEXT_LENGTH:
            return RecoveryAction(
                action="retry_shorter",
                truncate_to=int(1000 * (config.truncate_ratio ** attempt)),
                reason="Prompt too long, truncating",
            )

        case ErrorCategory.TIMEOUT:
            return RecoveryAction(
                action="retry_shorter",
                truncate_to=int(2000 * (config.truncate_ratio ** attempt)),
                reason="Timeout, trying shorter prompt",
            )

        case ErrorCategory.RATE_LIMIT:
            return RecoveryAction(
                action="wait_and_retry",
                wait_seconds=config.rate_limit_wait,
                reason="Rate limited, waiting",
            )

        case ErrorCategory.INVALID_JSON | ErrorCategory.CONNECTION:
            return RecoveryAction(
                action="retry",
                wait_seconds=config.backoff_seconds * attempt,
                reason=f"Transient error, retry #{attempt + 1}",
            )

        case ErrorCategory.TOOL_ERROR:
            return RecoveryAction(
                action="retry",
                wait_seconds=config.backoff_seconds,
                reason="Tool error, retrying",
            )

        case ErrorCategory.UNKNOWN:
            # For unknown errors, default to simple retry
            # Clinic agent can override if configured
            return RecoveryAction(
                action="retry",
                wait_seconds=config.backoff_seconds * attempt,
                reason="Unknown error, attempting retry",
            )
```

### 2.5 Clinic Agent (LLM Fallback - Optional)

Only invoked when `category == UNKNOWN` and `config.use_clinic_for_unknown == True`.

```python
from pydantic_ai import Agent
from agentic_patterns._models import get_model


class ClinicDiagnosis(BaseModel):
    """LLM diagnosis for unknown errors."""
    should_retry: bool = Field(description="Whether retrying might help")
    suggestion: str = Field(description="Brief suggestion for the user")


clinic_agent = Agent(
    get_model(),
    system_prompt=(
        "You diagnose agent errors. Given an error message, determine if "
        "retrying would help. Be concise. Don't suggest code changes."
    ),
    output_type=ClinicDiagnosis,
)
```

### 2.6 Entry Point

```python
from typing import TypeVar
from pydantic_ai import Agent

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


async def recoverable_run(
    agent: Agent[DepsT, OutputT],
    prompt: str,
    *,
    deps: DepsT | None = None,
    config: RecoveryConfig | None = None,
) -> OutputT:
    """
    Run agent with automatic exception recovery.

    Uses deterministic heuristics for known errors, optional LLM
    diagnosis for unknown errors.

    Args:
        agent: The agent to run.
        prompt: User prompt.
        deps: Agent dependencies.
        config: Recovery configuration.

    Returns:
        Agent output on success.

    Raises:
        Exception: Original exception if recovery fails.

    Example:
        result = await recoverable_run(
            my_agent,
            "Process this data",
            deps=MyDeps(),
            config=RecoveryConfig(max_attempts=3),
        )
    """
```

## 3. Idiomatic Feature Table

| Feature | Used? | Implementation |
|---------|-------|----------------|
| `@output_validator` + `ModelRetry` | No | Phoenix handles exceptions, not output quality |
| `@system_prompt` | No | Clinic agent has static prompt |
| `deps_type` + `RunContext` | No | No runtime state injection needed |
| `@tool` | No | Pure diagnosis, no external data |
| `pydantic_graph` | No | Simple wrapper function |

## 4. Test Strategy

### 4.1 Unit Tests

```python
import pytest

def test_classify_context_length():
    exc = ValueError("maximum context length exceeded")
    assert classify_error(exc) == ErrorCategory.CONTEXT_LENGTH

def test_classify_timeout():
    exc = TimeoutError("Request timed out")
    assert classify_error(exc) == ErrorCategory.TIMEOUT

def test_classify_rate_limit():
    exc = Exception("Error 429: rate limit exceeded")
    assert classify_error(exc) == ErrorCategory.RATE_LIMIT

def test_classify_unknown():
    exc = ValueError("Something weird happened")
    assert classify_error(exc) == ErrorCategory.UNKNOWN

def test_recovery_action_context_length():
    action = get_recovery_action(ErrorCategory.CONTEXT_LENGTH, 0, RecoveryConfig())
    assert action.action == "retry_shorter"
    assert action.truncate_to is not None

def test_recovery_action_max_attempts():
    action = get_recovery_action(ErrorCategory.TIMEOUT, 5, RecoveryConfig(max_attempts=3))
    assert action.action == "abort"
```

### 4.2 Integration Tests

```python
from unittest.mock import AsyncMock, MagicMock

async def test_recoverable_run_retries_on_transient():
    """Should retry and succeed after transient failure."""
    mock_agent = MagicMock()
    call_count = 0

    async def mock_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Connection reset")
        return MagicMock(output="Success")

    mock_agent.run = mock_run

    result = await recoverable_run(mock_agent, "test")
    assert result == "Success"
    assert call_count == 2

async def test_recoverable_run_gives_up():
    """Should give up after max attempts."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=TimeoutError("Always times out"))

    with pytest.raises(TimeoutError):
        await recoverable_run(
            mock_agent,
            "test",
            config=RecoveryConfig(max_attempts=2),
        )
```

## 5. What's NOT Implemented

Documented edge cases we're skipping:

1. **Message history summarization:** If context is too long, we truncate the
   prompt, not summarize history. Summarization loses message structure.

2. **Tool call introspection:** We don't inspect which tool failed. The error
   message is enough for classification.

3. **Stack trace analysis:** We don't send stack traces to the LLM. Error
   messages are sufficient and save tokens.

4. **Nested recovery:** If the Clinic agent fails, we don't try to recover it.
   We just use the default retry action.

## 6. Integration & Documentation

**Integration:**
- [x] Added to `scripts/integration_test.sh` ALL_PATTERNS array
- [x] Exported from `src/agentic_patterns/__init__.py`
- [x] `if __name__ == "__main__"` demo block

**Documentation:**
- **Pattern page:** `docs/patterns/12-exception-recovery.md`
- **Key insight:** Deterministic heuristics handle 90% of cases without LLM
- **When to use:** Wrap unreliable agent calls (external APIs, long prompts)
