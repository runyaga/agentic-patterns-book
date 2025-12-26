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
# Initialize the model
model = get_model()

# Critic agent - evaluates content
# Note: Critic is stateless, so deps_type is None
critic_agent = Agent(
    model,
    system_prompt=(
        "You are a critical reviewer. Evaluate the given content objectively. "
        "Score from 0-10 where 8+ means acceptable quality. "
        "Provide specific, actionable suggestions for improvement. "
        "If content is repetitive or fails to address the prompt, score low."
    ),
    output_type=Critique,
)

# Producer agent - generates content
producer_agent = Agent(
    model,
    system_prompt=(
        "You are a skilled content producer. Generate high-quality content "
        "based on the given task. If you receive feedback, use it to "
        "improve your next draft significantly."
    ),
    deps_type=ReflectionDeps,
    output_type=ProducerOutput,
    retries=3,  # Allow up to 3 improvement cycles
)


@producer_agent.output_validator
async def validate_content(
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
        # Raising ModelRetry automatically feeds the error back to the model
        # The model sees this as a previous tool error/rejection
        raise ModelRetry(
            f"Critique score {critique.score}/10. "
            f"Feedback: {critique.feedback}. "
            "Please rewrite the content to address this feedback."
        )

    print("  Content accepted!")
    return result
# --8<-- [end:agents]


# --8<-- [start:reflection]
async def run_reflection(task: str) -> ProducerOutput:
    """
    Run the reflection process.
    The manual loop is gone. We simply run the producer,
    and PydanticAI handles the critique/retry loop internally.
    """
    print(f"Starting reflection task: {task}")

    deps = ReflectionDeps(critic_agent=critic_agent)

    try:
        # The agent will automatically retry if the validator raises ModelRetry
        result = await producer_agent.run(task, deps=deps)
        return result.output
    except Exception as e:
        print(f"Reflection failed after retries: {e}")
        # Return what we have, or re-raise
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
