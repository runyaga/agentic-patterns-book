"""
Exception Recovery Pattern (Phoenix Protocol).

Based on the Agentic Design Patterns book Chapter 12:
Automatic recovery from agent exceptions using deterministic heuristics
first, with optional LLM diagnosis for unknown errors.

Key concepts:
- Deterministic First: Pattern-match known errors before invoking LLM
- Minimal Token Usage: Don't send stack traces to LLM
- Simple API: Single recoverable_run() wrapper function

Distinction from ModelRetry:
- ModelRetry: semantic failures during generation (via @output_validator)
- Phoenix: Python exceptions from agent.run() itself

Streaming Limitation:
This pattern only supports agent.run(), NOT agent.run_stream().
Streaming recovery is fundamentally broken - mid-stream errors (~60% of
cases) cause duplicate/inconsistent output. If you need streaming with
recovery, handle retries at the application level using is_retryable().
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import TypeVar

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import UserContent
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from agentic_patterns._models import get_model


# --8<-- [start:models]
class ErrorCategory(str, Enum):
    """Known error categories with deterministic fixes."""

    CONTEXT_LENGTH = "context_length"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INVALID_JSON = "invalid_json"
    CONNECTION = "connection"
    TOOL_ERROR = "tool_error"
    UNKNOWN = "unknown"


class RecoveryAction(BaseModel):
    """What to do to recover from the error."""

    action: str = Field(
        description="retry, retry_shorter, wait_and_retry, abort"
    )
    wait_seconds: float = 0.0
    truncate_to: int | None = None
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
    truncate_ratio: float = 0.7
    use_clinic_for_unknown: bool = True


# --8<-- [end:models]


# --8<-- [start:classify]
def classify_error(exc: Exception) -> ErrorCategory:
    """
    Classify exception into known category using pattern matching.

    No LLM call - pure Python heuristics.

    Args:
        exc: The exception to classify.

    Returns:
        ErrorCategory for the exception type.
    """
    msg = str(exc).lower()
    exc_type = type(exc).__name__

    # Context length patterns
    context_patterns = ["context length", "too many tokens", "maximum context"]
    if any(p in msg for p in context_patterns):
        return ErrorCategory.CONTEXT_LENGTH

    # Timeout patterns
    if "timeout" in msg or exc_type in ("TimeoutError", "AsyncioTimeoutError"):
        return ErrorCategory.TIMEOUT

    # Rate limit patterns
    rate_patterns = ["rate limit", "429", "too many requests"]
    if any(p in msg for p in rate_patterns):
        return ErrorCategory.RATE_LIMIT

    # JSON parsing (model returned malformed output)
    json_error_patterns = ["decode", "parse", "invalid"]
    if "json" in msg and any(p in msg for p in json_error_patterns):
        return ErrorCategory.INVALID_JSON

    # Connection errors
    conn_patterns = ["connection", "network", "refused", "reset"]
    if any(p in msg for p in conn_patterns):
        return ErrorCategory.CONNECTION

    # Tool execution errors
    tool_error_patterns = ["failed", "error", "exception"]
    if "tool" in msg and any(p in msg for p in tool_error_patterns):
        return ErrorCategory.TOOL_ERROR

    return ErrorCategory.UNKNOWN


# --8<-- [end:classify]


# --8<-- [start:recovery_action]
def get_recovery_action(
    category: ErrorCategory,
    attempt: int,
    config: RecoveryConfig,
) -> RecoveryAction:
    """
    Determine recovery action for error category.

    Pure Python - no LLM call.

    Args:
        category: The classified error category.
        attempt: Current attempt number (0-indexed).
        config: Recovery configuration.

    Returns:
        RecoveryAction describing what to do next.
    """
    if attempt >= config.max_attempts:
        return RecoveryAction(action="abort", reason="Max attempts reached")

    match category:
        case ErrorCategory.CONTEXT_LENGTH:
            truncate = int(1000 * (config.truncate_ratio**attempt))
            return RecoveryAction(
                action="retry_shorter",
                truncate_to=truncate,
                reason="Prompt too long, truncating",
            )

        case ErrorCategory.TIMEOUT:
            truncate = int(2000 * (config.truncate_ratio**attempt))
            return RecoveryAction(
                action="retry_shorter",
                truncate_to=truncate,
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
                wait_seconds=config.backoff_seconds * (attempt + 1),
                reason=f"Transient error, retry #{attempt + 1}",
            )

        case ErrorCategory.TOOL_ERROR:
            return RecoveryAction(
                action="retry",
                wait_seconds=config.backoff_seconds,
                reason="Tool error, retrying",
            )

        case _:
            return RecoveryAction(
                action="retry",
                wait_seconds=config.backoff_seconds * (attempt + 1),
                reason="Unknown error, attempting retry",
            )


# --8<-- [end:recovery_action]


# --8<-- [start:clinic]
class ClinicDiagnosis(BaseModel):
    """LLM diagnosis for unknown errors."""

    should_retry: bool = Field(description="Whether retrying might help")
    suggestion: str = Field(description="Brief suggestion for the user")


CLINIC_SYSTEM_PROMPT = (
    "You diagnose agent errors. Given an error message, determine if "
    "retrying would help. Be concise. Don't suggest code changes."
)


def create_clinic_agent(
    model: Model | None = None,
) -> Agent[None, ClinicDiagnosis]:
    """
    Create a clinic agent for unknown error diagnosis.

    Args:
        model: pydantic-ai Model instance. If None, uses default model.

    Returns:
        Configured clinic agent.
    """
    return Agent(
        model or get_model(),
        system_prompt=CLINIC_SYSTEM_PROMPT,
        output_type=ClinicDiagnosis,
    )


# Backward compatibility alias
def _create_clinic_agent() -> Agent[None, ClinicDiagnosis]:
    """Create the clinic agent for unknown error diagnosis."""
    return create_clinic_agent()


async def _diagnose_unknown(
    exc: Exception,
    clinic_agent: Agent[None, ClinicDiagnosis],
) -> RecoveryAction:
    """
    Use LLM to diagnose unknown errors.

    Only called when error category is UNKNOWN and use_clinic_for_unknown=True.

    Args:
        exc: The exception to diagnose.
        clinic_agent: The clinic agent instance.

    Returns:
        RecoveryAction based on LLM diagnosis.
    """
    try:
        result = await clinic_agent.run(
            f"Error type: {type(exc).__name__}\nMessage: {str(exc)[:500]}"
        )
        diagnosis = result.output

        if diagnosis.should_retry:
            return RecoveryAction(
                action="retry",
                wait_seconds=1.0,
                reason=f"LLM diagnosis: {diagnosis.suggestion}",
            )
        return RecoveryAction(
            action="abort",
            reason=f"LLM suggests not retrying: {diagnosis.suggestion}",
        )
    except Exception:
        # If clinic agent fails, fall back to retry
        return RecoveryAction(
            action="retry",
            wait_seconds=1.0,
            reason="Clinic agent failed, attempting retry anyway",
        )


# --8<-- [end:clinic]


# --8<-- [start:is_retryable]
def is_retryable(exc: Exception) -> bool:
    """
    Check if exception is likely retryable.

    Use this helper when handling streaming errors at the application level,
    since recoverable_run() doesn't support streaming.

    Args:
        exc: The exception to check.

    Returns:
        True if the error category suggests retry might help.

    Example:
        async with agent.run_stream("prompt") as stream:
            try:
                async for chunk in stream.stream_text():
                    print(chunk)
            except Exception as e:
                if is_retryable(e):
                    # Handle retry at application level
                    pass
    """
    category = classify_error(exc)
    return category != ErrorCategory.UNKNOWN


# --8<-- [end:is_retryable]


# --8<-- [start:recoverable_run]
DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


def _truncate_prompt(
    prompt: str | Sequence[UserContent] | None,
    truncate_to: int,
) -> str | Sequence[UserContent] | None:
    """Truncate prompt to specified length."""
    if prompt is None:
        return None
    if isinstance(prompt, str):
        if len(prompt) > truncate_to:
            return prompt[:truncate_to] + "\n[Truncated due to length]"
        return prompt
    # Sequence[UserContent] - just return as-is, can't easily truncate
    return prompt


def _truncate_history(
    history: Sequence[ModelMessage] | None,
    ratio: float,
) -> list[ModelMessage] | None:
    """Truncate message history, keeping most recent messages."""
    if history is None or len(history) == 0:
        return None
    keep = max(1, int(len(history) * ratio))
    return list(history[-keep:])


async def recoverable_run(
    agent: Agent[DepsT, OutputT],
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    deps: DepsT | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    recovery_config: RecoveryConfig | None = None,
    **kwargs: Any,
) -> AgentRunResult[OutputT]:
    """
    Run agent with automatic exception recovery.

    Returns the full AgentRunResult, preserving all metadata including
    .output, .usage(), .all_messages(), etc.

    Uses deterministic heuristics for known errors, optional LLM
    diagnosis for unknown errors.

    Note: This function only supports agent.run(), NOT agent.run_stream().
    Streaming recovery is fundamentally broken for mid-stream errors.
    Use is_retryable() to handle streaming retries at the application level.

    Args:
        agent: The agent to run.
        user_prompt: User prompt (string or structured content).
        deps: Agent dependencies.
        message_history: Prior conversation messages.
        recovery_config: Recovery behavior configuration.
        **kwargs: Additional args forwarded to agent.run().

    Returns:
        Full AgentRunResult with .output, .usage(), .all_messages(), etc.
        Usage is accumulated across all retry attempts.

    Raises:
        Exception: Original exception if recovery fails.

    Example:
        result = await recoverable_run(
            my_agent,
            "Process this data",
            deps=MyDeps(),
            recovery_config=RecoveryConfig(max_attempts=3),
        )
        print(result.output)  # The agent's output
        print(result.usage())  # Token usage (includes failed attempts)
    """
    config = recovery_config or RecoveryConfig()
    clinic_agent: Agent[None, ClinicDiagnosis] | None = None
    categories_seen: list[str] = []
    current_prompt = user_prompt
    current_history = message_history
    last_exception: Exception | None = None

    # Accumulate usage across all attempts
    accumulated_usage = RunUsage()

    for attempt in range(config.max_attempts + 1):
        try:
            run_kwargs: dict[str, Any] = dict(kwargs)
            if deps is not None:
                run_kwargs["deps"] = deps
            if current_history is not None:
                run_kwargs["message_history"] = current_history

            result = await agent.run(current_prompt, **run_kwargs)

            # Add this attempt's usage to accumulated total
            ru = result.usage()
            au = accumulated_usage
            accumulated_usage = RunUsage(
                requests=au.requests + ru.requests,
                input_tokens=au.input_tokens + ru.input_tokens,
                output_tokens=au.output_tokens + ru.output_tokens,
            )

            # Update result's usage to reflect all attempts
            # Note: AgentRunResult._usage is the internal storage
            result._usage = accumulated_usage
            return result

        except Exception as exc:
            last_exception = exc
            category = classify_error(exc)
            categories_seen.append(category.value)

            msg = f"[Recovery] Attempt {attempt + 1}: {category.value}"
            print(f"{msg} - {exc}")

            # Get recovery action
            use_clinic = config.use_clinic_for_unknown
            if category == ErrorCategory.UNKNOWN and use_clinic:
                if clinic_agent is None:
                    clinic_agent = _create_clinic_agent()
                action = await _diagnose_unknown(exc, clinic_agent)
            else:
                action = get_recovery_action(category, attempt, config)

            print(f"[Recovery] Action: {action.action} - {action.reason}")

            if action.action == "abort":
                break

            # Apply recovery action
            if action.wait_seconds > 0:
                await asyncio.sleep(action.wait_seconds)

            if action.action == "retry_shorter" and action.truncate_to:
                current_prompt = _truncate_prompt(
                    user_prompt, action.truncate_to
                )
                current_history = _truncate_history(
                    message_history, config.truncate_ratio
                )

    # All attempts failed
    if last_exception:
        raise last_exception
    raise RuntimeError("Recovery failed with no exception captured")


# --8<-- [end:recoverable_run]


if __name__ == "__main__":

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Exception Recovery (Phoenix Protocol)")
        print("=" * 60)

        # Create a simple agent for demonstration
        demo_agent: Agent[None, str] = Agent(
            get_model(),
            system_prompt="You are a helpful assistant. Be concise.",
            output_type=str,
        )

        # Run with recovery - returns full AgentRunResult
        result = await recoverable_run(
            demo_agent,
            "What is 2 + 2?",
            recovery_config=RecoveryConfig(max_attempts=3),
        )

        print(f"\nOutput: {result.output}")
        print(f"Usage: {result.usage()}")

    asyncio.run(main())
