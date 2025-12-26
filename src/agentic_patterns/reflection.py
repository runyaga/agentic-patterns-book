"""
Reflection Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 4:
An agent evaluates its own work and uses that evaluation to improve.

Core process:
1. Execution - Producer generates initial output
2. Evaluation/Critique - Critic analyzes the output
3. Refinement - Producer improves based on feedback
4. Iteration - Repeat until satisfactory or max iterations

Key implementation: Producer-Critic model with separate agents.
"""

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from agentic_patterns._models import get_model


class ProducerOutput(BaseModel):
    """Output from the producer agent."""

    content: str = Field(description="The generated content")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Producer's confidence in output"
    )


class Critique(BaseModel):
    """Critique from the critic agent."""

    is_acceptable: bool = Field(
        description="Whether the output meets quality standards"
    )
    score: float = Field(ge=0.0, le=10.0, description="Quality score (0-10)")
    strengths: list[str] = Field(
        default_factory=list, description="What the output does well"
    )
    weaknesses: list[str] = Field(
        default_factory=list, description="Areas needing improvement"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Specific improvement suggestions"
    )


class RefinedOutput(BaseModel):
    """Refined output after incorporating feedback."""

    content: str = Field(description="The refined content")
    changes_made: list[str] = Field(
        default_factory=list, description="List of changes made"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in refined output"
    )


class ReflectionResult(BaseModel):
    """Final result of the reflection process."""

    final_content: str = Field(description="The final refined content")
    iterations: int = Field(description="Number of iterations performed")
    final_score: float = Field(description="Final quality score")
    improvement_history: list[str] = Field(
        default_factory=list,
        description="Summary of improvements per iteration",
    )
    converged: bool = Field(
        description="Whether process converged (vs hit max iterations)"
    )


# Initialize the model
model = get_model()

# Producer agent - generates content
producer_agent = Agent(
    model,
    system_prompt=(
        "You are a skilled content producer. Generate high-quality content "
        "based on the given task. Be thorough, accurate, and well-structured. "
        "If given feedback from a previous iteration, incorporate it to "
        "improve your output."
    ),
    output_type=ProducerOutput,
)

# Critic agent - evaluates content
critic_agent = Agent(
    model,
    system_prompt=(
        "You are a critical reviewer. Evaluate the given content objectively. "
        "Score from 0-10 where 8+ means acceptable quality. "
        "Identify specific strengths and weaknesses. "
        "Provide actionable suggestions for improvement. "
        "Be constructive but thorough in your critique."
    ),
    output_type=Critique,
)

# Refiner agent - improves based on feedback
refiner_agent = Agent(
    model,
    system_prompt=(
        "You are a content refiner. Given the original content and critique, "
        "produce an improved version addressing the weaknesses. "
        "Preserve the strengths while fixing the issues. "
        "List the specific changes you made."
    ),
    output_type=RefinedOutput,
)


async def reflect_once(
    content: str,
    task_description: str,
) -> tuple[Critique, RefinedOutput]:
    """
    Perform a single reflection cycle (critique + refine).

    Args:
        content: The content to critique and refine.
        task_description: Original task for context.

    Returns:
        Tuple of (critique, refined_output).
    """
    # Critique the content
    critique_result = await critic_agent.run(
        f"Task: {task_description}\n\nContent to evaluate:\n{content}"
    )
    critique = critique_result.output

    # If already acceptable, return with minimal changes
    if critique.is_acceptable:
        refined = RefinedOutput(
            content=content,
            changes_made=["No changes needed - already acceptable"],
            confidence=0.95,
        )
        return critique, refined

    # Refine based on critique
    suggestions_text = "\n".join(f"- {s}" for s in critique.suggestions)
    weaknesses_text = "\n".join(f"- {w}" for w in critique.weaknesses)

    refine_result = await refiner_agent.run(
        f"Original task: {task_description}\n\n"
        f"Current content:\n{content}\n\n"
        f"Weaknesses identified:\n{weaknesses_text}\n\n"
        f"Suggestions:\n{suggestions_text}\n\n"
        f"Please produce an improved version."
    )

    return critique, refine_result.output


async def run_reflection(
    task: str,
    max_iterations: int = 3,
    acceptable_score: float = 8.0,
) -> ReflectionResult:
    """
    Run the full reflection process with iterative refinement.

    Args:
        task: The task/prompt for content generation.
        max_iterations: Maximum reflection iterations (default: 3).
        acceptable_score: Score threshold to stop (default: 8.0).

    Returns:
        ReflectionResult with final content and process metadata.
    """
    print(f"Reflection: Starting with max {max_iterations} iterations...")

    # Step 1: Initial production
    print("  Iteration 0: Generating initial content...")
    initial_result = await producer_agent.run(task)
    current_content = initial_result.output.content

    improvement_history = []
    final_score = 0.0
    converged = False

    # Step 2: Iterative reflection
    for i in range(max_iterations):
        print(f"  Iteration {i + 1}: Critiquing and refining...")

        critique, refined = await reflect_once(current_content, task)
        final_score = critique.score

        # Record improvement
        if critique.weaknesses:
            summary = f"Iter {i + 1}: Fixed {len(critique.weaknesses)} issues"
        else:
            summary = f"Iter {i + 1}: No issues found"
        improvement_history.append(summary)

        print(f"    Score: {critique.score:.1f}/10")

        # Check if acceptable
        if critique.is_acceptable or critique.score >= acceptable_score:
            print(f"  Converged at iteration {i + 1} with score {final_score}")
            converged = True
            current_content = refined.content
            break

        current_content = refined.content

    if not converged:
        print(f"  Max iterations reached. Final score: {final_score}")

    return ReflectionResult(
        final_content=current_content,
        iterations=len(improvement_history),
        final_score=final_score,
        improvement_history=improvement_history,
        converged=converged,
    )


async def self_reflect(
    task: str,
    max_iterations: int = 2,
) -> ReflectionResult:
    """
    Simplified self-reflection using a single agent.

    The same agent generates, critiques, and refines its own work.
    Useful when you want simpler architecture.

    Args:
        task: The task/prompt for content generation.
        max_iterations: Maximum iterations (default: 2).

    Returns:
        ReflectionResult with final content.
    """
    # Self-reflecting agent does all roles
    self_reflect_agent = Agent(
        model,
        system_prompt=(
            "You are a self-improving content creator. "
            "Generate content, then critically evaluate it yourself, "
            "and produce an improved version. Be honest about weaknesses."
        ),
        output_type=RefinedOutput,
    )

    print(f"Self-reflection: Starting with max {max_iterations} iterations...")

    # Initial generation
    print("  Generating initial content...")
    initial = await producer_agent.run(task)
    current = initial.output.content
    history = []

    for i in range(max_iterations):
        print(f"  Iteration {i + 1}: Self-reflecting...")

        result = await self_reflect_agent.run(
            f"Task: {task}\n\n"
            f"Your previous output:\n{current}\n\n"
            f"Critically evaluate this and produce an improved version."
        )

        refined = result.output
        changes = len(refined.changes_made)
        history.append(f"Iter {i + 1}: Made {changes} changes")
        current = refined.content

        if changes == 0 or "no changes" in str(refined.changes_made).lower():
            print(f"  Self-reflection complete at iteration {i + 1}")
            break

    return ReflectionResult(
        final_content=current,
        iterations=len(history),
        final_score=8.0,  # Self-reflection doesn't score
        improvement_history=history,
        converged=True,
    )


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        # Demo: Producer-Critic reflection
        print("=" * 60)
        print("DEMO: Producer-Critic Reflection")
        print("=" * 60)

        result = await run_reflection(
            task=(
                "Write a short paragraph explaining why Python is popular "
                "for data science. Include at least 3 specific reasons."
            ),
            max_iterations=3,
            acceptable_score=8.0,
        )

        print(f"\nFinal Content:\n{result.final_content}")
        print(f"\nIterations: {result.iterations}")
        print(f"Final Score: {result.final_score}")
        print(f"Converged: {result.converged}")
        print(f"History: {result.improvement_history}")

    asyncio.run(main())
