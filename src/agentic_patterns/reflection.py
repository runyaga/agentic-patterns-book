"""
Reflection Pattern Implementation (Idiomatic PydanticAI).

Based on the Agentic Design Patterns book Chapter 4:
An agent evaluates its own work and uses that evaluation to improve.

Key concepts:
- Producer-Critic Model: Separate agents for generation and evaluation
- Automated Iteration: Use ModelRetry to loop until quality standards are met
- Native Control Flow: No manual Python loops; leverage the framework

This module implements reflection using PydanticAI's `result_validator`
to automatically retry generation if the critic's score is too low.
"""

from dataclasses import dataclass

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai import ModelRetry
from pydantic_ai import RunContext
from pydantic_ai.models import Model

from agentic_patterns._models import get_model


# --8<-- [start:models]
class ProducerOutput(BaseModel):
    """Output from the producer agent."""

    content: str = Field(description="The generated content")
    reasoning: str = Field(description="Reasoning behind this draft")


class Critique(BaseModel):
    """Critique from the critic agent."""

    is_acceptable: bool = Field(
        description="Whether the output meets quality standards (8/10+)"
    )
    score: float = Field(ge=0.0, le=10.0, description="Quality score (0-10)")
    feedback: str = Field(description="Constructive feedback and suggestions")


@dataclass
class ReflectionDeps:
    """Dependencies for the reflection process."""

    critic_agent: Agent[None, Critique]
    max_history: int = 5


# --8<-- [end:models]


# --8<-- [start:agents]
CRITIC_SYSTEM_PROMPT = (
    "You are a critical reviewer. Evaluate the given content objectively. "
    "Score from 0-10 where 8+ means acceptable quality. "
    "Provide specific, actionable suggestions for improvement. "
    "If content is repetitive or fails to address the prompt, score low."
)

PRODUCER_SYSTEM_PROMPT = (
    "You are a skilled content producer. Generate high-quality content "
    "based on the given task. If you receive feedback, use it to "
    "improve your next draft significantly."
)


def _producer_validator(
    ctx: RunContext[ReflectionDeps], result: ProducerOutput
) -> ProducerOutput:
    """Validator logic for producer output."""
    raise NotImplementedError("Use async version")


async def _async_producer_validator(
    ctx: RunContext[ReflectionDeps], result: ProducerOutput
) -> ProducerOutput:
    """
    Validate content quality using the Critic agent.
    If quality is low, raise ModelRetry to trigger a new attempt.
    """
    # Ask the critic to evaluate the result
    print("\n[Validator] Critiquing draft...")
    critique_result = await ctx.deps.critic_agent.run(result.content)
    critique = critique_result.output

    print(f"  Score: {critique.score}/10")

    # Check acceptance criteria
    if critique.score < 8.0 and not critique.is_acceptable:
        print(f"  Feedback: {critique.feedback[:100]}...")
        raise ModelRetry(
            f"Critique score {critique.score}/10. "
            f"Feedback: {critique.feedback}. "
            "Please rewrite the content to address this feedback."
        )

    print("  Content accepted!")
    return result


# Alias for backward compatibility with tests
validate_content = _async_producer_validator


def create_critic_agent(
    model: Model | None = None,
) -> Agent[None, Critique]:
    """
    Create a critic agent with optional model override.

    Args:
        model: pydantic-ai Model instance. If None, uses default model.

    Returns:
        Configured critic agent.
    """
    return Agent(
        model or get_model(),
        system_prompt=CRITIC_SYSTEM_PROMPT,
        output_type=Critique,
    )


def create_producer_agent(
    model: Model | None = None,
    retries: int = 3,
) -> Agent[ReflectionDeps, ProducerOutput]:
    """
    Create a producer agent with optional model override.

    Args:
        model: pydantic-ai Model instance. If None, uses default model.
        retries: Max improvement cycles (default: 3).

    Returns:
        Configured producer agent with validator.
    """
    agent: Agent[ReflectionDeps, ProducerOutput] = Agent(
        model or get_model(),
        system_prompt=PRODUCER_SYSTEM_PROMPT,
        deps_type=ReflectionDeps,
        output_type=ProducerOutput,
        retries=retries,
    )
    agent.output_validator(_async_producer_validator)
    return agent


# Default agents (created lazily for backward compatibility)
_default_critic: Agent[None, Critique] | None = None
_default_producer: Agent[ReflectionDeps, ProducerOutput] | None = None


def _get_default_critic() -> Agent[None, Critique]:
    """Get or create the default critic agent."""
    global _default_critic
    if _default_critic is None:
        _default_critic = create_critic_agent()
    return _default_critic


def _get_default_producer() -> Agent[ReflectionDeps, ProducerOutput]:
    """Get or create the default producer agent."""
    global _default_producer
    if _default_producer is None:
        _default_producer = create_producer_agent()
    return _default_producer


# Module-level aliases for backward compatibility with tests
producer_agent = create_producer_agent()
critic_agent = create_critic_agent()


# --8<-- [end:agents]


# --8<-- [start:reflection]
async def run_reflection(
    task: str,
    *,
    producer: Agent[ReflectionDeps, ProducerOutput] | None = None,
    critic: Agent[None, Critique] | None = None,
) -> ProducerOutput:
    """
    Run the reflection process.

    The manual loop is gone. We simply run the producer,
    and PydanticAI handles the critique/retry loop internally.

    Args:
        task: The content generation task.
        producer: Optional producer agent. If None, uses default.
        critic: Optional critic agent. If None, uses default.

    Returns:
        ProducerOutput with the final content.
    """
    print(f"Starting reflection task: {task}")

    the_producer = producer or producer_agent
    the_critic = critic or critic_agent

    deps = ReflectionDeps(critic_agent=the_critic)

    try:
        # The agent will automatically retry if the validator raises ModelRetry
        result = await the_producer.run(task, deps=deps)
        return result.output
    except Exception as e:
        print(f"Reflection failed after retries: {e}")
        raise


# --8<-- [end:reflection]


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Idiomatic Reflection (Validator + Retry)")
        print("=" * 60)

        # We intentionally give a hard task to trigger reflection
        # (or just ask for high quality)
        output = await run_reflection(
            "Write a short, engaging tweet about the importance of "
            "dependency injection in Python software architecture."
        )

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(output.content)
        print(f"\nReasoning: {output.reasoning}")

    asyncio.run(main())
