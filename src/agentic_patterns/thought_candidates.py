"""
Thought Candidates Pattern (Best-of-N Sampling).

Based on the Agentic Design Patterns book Chapter 17a:
Single-level parallel generation and evaluation of candidate solutions.

This pattern generates N candidate thoughts for a problem, evaluates each
in parallel, and selects the highest-scoring candidate. It serves as the
foundation for the Tree of Thoughts pattern (17b).

Key concepts:
- Best-of-N Sampling: Generate multiple solutions, pick highest-scoring
- Parallel Evaluation: Score all candidates concurrently
- Typed Context: Use Pydantic models and deps_type, not string concatenation
"""

import asyncio
from dataclasses import dataclass

import logfire
from pydantic import BaseModel
from pydantic import Field
from pydantic import computed_field
from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.models import Model

from agentic_patterns._models import get_fast_model
from agentic_patterns._models import get_strong_model


# --8<-- [start:models]
class ProblemStatement(BaseModel):
    """Structured problem input - avoids raw strings."""

    description: str = Field(description="The problem to solve")
    constraints: list[str] = Field(
        default_factory=list, description="Constraints to satisfy"
    )
    context: str | None = Field(
        default=None, description="Previous step context for chaining"
    )


class Thought(BaseModel):
    """A single candidate thought or solution step."""

    content: str = Field(description="The thought or partial solution")
    reasoning: str = Field(description="Explanation of this approach")


class ThoughtEvaluation(BaseModel):
    """Evaluation result for a single thought."""

    score: float = Field(ge=0.0, le=10.0, description="Quality score (0-10)")
    is_valid: bool = Field(description="Whether logically valid")
    feedback: str = Field(description="Evaluation rationale and suggestions")


class ScoredThought(BaseModel):
    """Thought combined with its evaluation - immutable."""

    thought: Thought
    evaluation: ThoughtEvaluation

    @computed_field
    @property
    def score(self) -> float:
        """Derived score for sorting and comparison."""
        return self.evaluation.score


class BestOfNResult(BaseModel):
    """Result from best-of-N sampling."""

    problem: ProblemStatement
    candidates: list[ScoredThought] = Field(
        description="All evaluated candidates, sorted by score descending"
    )
    best: ScoredThought = Field(description="The highest-scoring candidate")
    generation_count: int = Field(description="Number of candidates generated")


# Dependency injection contexts (typed, not strings)
@dataclass
class OutputConfig:
    """Configuration for LLM output constraints."""

    max_words: int = 100
    ascii_only: bool = True


@dataclass
class GenerationContext:
    """Context for thought generation - passed via deps_type."""

    problem: ProblemStatement
    config: OutputConfig | None = None

    @property
    def output_config(self) -> OutputConfig:
        return self.config or OutputConfig()


@dataclass
class EvaluationContext:
    """Context for thought evaluation - passed via deps_type."""

    problem: ProblemStatement
    thought: Thought
    config: OutputConfig | None = None

    @property
    def output_config(self) -> OutputConfig:
        return self.config or OutputConfig(max_words=50)


# --8<-- [end:models]


# --8<-- [start:agents]
GENERATOR_SYSTEM_PROMPT = (
    "You are a creative problem solver. Generate a single approach "
    "or step toward solving the given problem. Focus on making progress."
)

EVALUATOR_SYSTEM_PROMPT = (
    "You are a critical evaluator. Score the proposed approach from "
    "0 to 10 based on: correctness, feasibility, and progress toward "
    "the goal. Be strict but fair."
)


def _generator_prompt(ctx: RunContext[GenerationContext]) -> str:
    """Inject problem details and output constraints from typed context."""
    cfg = ctx.deps.output_config
    parts = [f"Problem: {ctx.deps.problem.description}"]
    if ctx.deps.problem.constraints:
        parts.append(f"Constraints: {', '.join(ctx.deps.problem.constraints)}")
    if ctx.deps.problem.context:
        parts.append(f"Previous step: {ctx.deps.problem.context}")

    # Output constraints
    output_rules = [f"Keep response under {cfg.max_words} words."]
    if cfg.ascii_only:
        output_rules.append("Use plain ASCII characters only.")
    parts.append(f"Output rules: {' '.join(output_rules)}")

    return "\n".join(parts)


def _evaluator_prompt(ctx: RunContext[EvaluationContext]) -> str:
    """Inject problem, thought, and output constraints from typed context."""
    cfg = ctx.deps.output_config
    parts = [
        f"Problem: {ctx.deps.problem.description}",
        f"Proposed approach: {ctx.deps.thought.content}",
        f"Author's reasoning: {ctx.deps.thought.reasoning}",
    ]
    if ctx.deps.problem.constraints:
        constraints = ", ".join(ctx.deps.problem.constraints)
        parts.insert(1, f"Constraints: {constraints}")

    # Output constraints
    output_rules = [f"Keep feedback under {cfg.max_words} words."]
    if cfg.ascii_only:
        output_rules.append("Use plain ASCII only.")
    parts.append(f"Output rules: {' '.join(output_rules)}")

    return "\n".join(parts)


def create_generator_agent(
    model: Model | None = None,
) -> Agent[GenerationContext, Thought]:
    """
    Create a generator agent with optional model override.

    Args:
        model: pydantic-ai Model instance. If None, uses default fast model.

    Returns:
        Configured generator agent.
    """
    agent: Agent[GenerationContext, Thought] = Agent(
        model or get_fast_model(),
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        deps_type=GenerationContext,
        output_type=Thought,
    )
    agent.system_prompt(_generator_prompt)
    return agent


def create_evaluator_agent(
    model: Model | None = None,
) -> Agent[EvaluationContext, ThoughtEvaluation]:
    """
    Create an evaluator agent with optional model override.

    Args:
        model: pydantic-ai Model instance. If None, uses default strong model.

    Returns:
        Configured evaluator agent.
    """
    agent: Agent[EvaluationContext, ThoughtEvaluation] = Agent(
        model or get_strong_model(),
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        deps_type=EvaluationContext,
        output_type=ThoughtEvaluation,
    )
    agent.system_prompt(_evaluator_prompt)
    return agent


# Default agents (created lazily on first use for backward compatibility)
_default_generator: Agent[GenerationContext, Thought] | None = None
_default_evaluator: Agent[EvaluationContext, ThoughtEvaluation] | None = None


def _get_default_generator() -> Agent[GenerationContext, Thought]:
    """Get or create the default generator agent."""
    global _default_generator
    if _default_generator is None:
        _default_generator = create_generator_agent()
    return _default_generator


# Module-level aliases for backward compatibility with tests
generator_agent = create_generator_agent()
evaluator_agent = create_evaluator_agent()


def _get_default_evaluator() -> Agent[EvaluationContext, ThoughtEvaluation]:
    """Get or create the default evaluator agent."""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = create_evaluator_agent()
    return _default_evaluator


# --8<-- [end:agents]


# --8<-- [start:patterns]
async def generate_thought(
    problem: ProblemStatement,
    config: OutputConfig | None = None,
    *,
    generator: Agent[GenerationContext, Thought] | None = None,
) -> Thought:
    """
    Generate a single candidate thought for a problem.

    Reusable by other patterns (e.g., Tree of Thoughts).

    Args:
        problem: The structured problem statement.
        config: Output constraints (word limits, ASCII-only, etc.).
        generator: Optional generator agent. If None, uses default.

    Returns:
        A Thought with content and reasoning.
    """
    with logfire.span("generate_thought"):
        agent = generator or generator_agent
        ctx = GenerationContext(problem=problem, config=config)
        result = await agent.run("Generate an approach", deps=ctx)
        return result.output


async def evaluate_thought(
    problem: ProblemStatement,
    thought: Thought,
    config: OutputConfig | None = None,
    *,
    evaluator: Agent[EvaluationContext, ThoughtEvaluation] | None = None,
) -> ThoughtEvaluation:
    """
    Evaluate a single thought against the problem.

    Reusable by other patterns (e.g., Tree of Thoughts).

    Args:
        problem: The structured problem statement.
        thought: The thought to evaluate.
        config: Output constraints (word limits, ASCII-only, etc.).
        evaluator: Optional evaluator agent. If None, uses default.

    Returns:
        ThoughtEvaluation with score, validity, and feedback.
    """
    with logfire.span("evaluate_thought"):
        agent = evaluator or evaluator_agent
        ctx = EvaluationContext(
            problem=problem, thought=thought, config=config
        )
        result = await agent.run("Evaluate this approach", deps=ctx)
        return result.output


async def generate_and_evaluate(
    problem: ProblemStatement,
    config: OutputConfig | None = None,
    *,
    generator: Agent[GenerationContext, Thought] | None = None,
    evaluator: Agent[EvaluationContext, ThoughtEvaluation] | None = None,
) -> ScoredThought:
    """
    Generate and evaluate a single thought atomically.

    Core reusable operation - generates one thought then evaluates it.
    This is the building block for both best-of-N and tree exploration.

    Args:
        problem: The structured problem statement.
        config: Output constraints (word limits, ASCII-only, etc.).
        generator: Optional generator agent. If None, uses default.
        evaluator: Optional evaluator agent. If None, uses default.

    Returns:
        ScoredThought combining the thought with its evaluation.
    """
    with logfire.span("generate_and_evaluate"):
        thought = await generate_thought(problem, config, generator=generator)
        evaluation = await evaluate_thought(
            problem, thought, config, evaluator=evaluator
        )
        return ScoredThought(thought=thought, evaluation=evaluation)


async def run_best_of_n(
    problem: ProblemStatement,
    n: int = 5,
    config: OutputConfig | None = None,
    *,
    generator: Agent[GenerationContext, Thought] | None = None,
    evaluator: Agent[EvaluationContext, ThoughtEvaluation] | None = None,
) -> BestOfNResult:
    """
    Generate N candidate thoughts in parallel and select the best.

    This is single-level exploration (depth=1). Each thought is
    independently generated and evaluated, then the highest-scoring
    one is selected.

    Args:
        problem: The structured problem statement.
        n: Number of candidates to generate (default: 5).
        config: Output constraints (word limits, ASCII-only, etc.).
        generator: Optional generator agent. If None, uses default.
        evaluator: Optional evaluator agent. If None, uses default.

    Returns:
        BestOfNResult with all candidates sorted by score and the best.
    """
    with logfire.span("run_best_of_n", n=n):
        logfire.info(f"Generating {n} candidates in parallel")

        # Generate and evaluate all candidates in parallel
        tasks = [
            generate_and_evaluate(
                problem, config, generator=generator, evaluator=evaluator
            )
            for _ in range(n)
        ]
        candidates = await asyncio.gather(*tasks)

        # Sort by score descending
        sorted_candidates = sorted(
            candidates, key=lambda c: c.score, reverse=True
        )
        best = sorted_candidates[0]

        logfire.info(
            "Best candidate selected",
            score=best.score,
            content=best.thought.content[:100],
        )

        return BestOfNResult(
            problem=problem,
            candidates=list(sorted_candidates),
            best=best,
            generation_count=n,
        )


# --8<-- [end:patterns]


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    def render_candidates(result: BestOfNResult, console: Console) -> None:
        """Render Best-of-N results with rich visualization."""
        console.print()
        console.rule("[bold blue]Thought Candidates (Best-of-N)")
        console.print()

        # Problem panel
        constraints = ", ".join(result.problem.constraints)
        console.print(
            Panel(
                f"[bold]{result.problem.description}[/bold]\n\n"
                f"[dim]Constraints: {constraints}[/dim]",
                title="Problem",
                border_style="blue",
            )
        )

        # Candidates table
        table = Table(
            title=f"[bold]All Candidates[/bold] ({result.generation_count})",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", justify="center", width=8)
        table.add_column("Valid", justify="center", width=6)
        table.add_column("Approach", width=50)
        table.add_column("Feedback", width=40)

        for i, candidate in enumerate(result.candidates):
            is_best = candidate == result.best
            score_style = (
                "bold green"
                if is_best
                else (
                    "green"
                    if candidate.score >= 7
                    else "yellow"
                    if candidate.score >= 5
                    else "red"
                )
            )
            valid_icon = "✓" if candidate.evaluation.is_valid else "✗"
            valid_style = "green" if candidate.evaluation.is_valid else "red"
            row_style = "on grey23" if is_best else ""

            table.add_row(
                f"{'★' if is_best else str(i + 1)}",
                f"[{score_style}]{candidate.score:.1f}[/]",
                f"[{valid_style}]{valid_icon}[/]",
                candidate.thought.content[:47] + "..."
                if len(candidate.thought.content) > 50
                else candidate.thought.content,
                candidate.evaluation.feedback[:37] + "..."
                if len(candidate.evaluation.feedback) > 40
                else candidate.evaluation.feedback,
                style=row_style,
            )

        console.print(table)

        # Best candidate detail
        console.print()
        console.print(
            Panel(
                f"[bold green]Score: {result.best.score:.1f}/10[/]\n\n"
                f"[bold]Approach:[/]\n{result.best.thought.content}\n\n"
                f"[bold]Reasoning:[/]\n{result.best.thought.reasoning}\n\n"
                f"[bold]Evaluation:[/]\n{result.best.evaluation.feedback}",
                title="[bold green]★ Best Candidate[/]",
                border_style="green",
            )
        )

    async def main() -> None:
        console = Console()
        console.print()
        console.rule("[bold]DEMO: Thought Candidates (Best-of-N)")

        # Creative problem: multiple valid approaches to explore
        problem = ProblemStatement(
            description=(
                "How would you explain recursion to someone who has never "
                "programmed before? Generate a clear analogy or example."
            ),
            constraints=[
                "Use everyday concepts, no code",
                "Must convey the self-referential nature",
                "Keep it memorable and intuitive",
            ],
        )

        # Output constraints - tight for local models, expand for cloud
        output_config = OutputConfig(max_words=80, ascii_only=True)

        console.print(
            f"\n[dim]Generating 3 candidates | "
            f"max_words={output_config.max_words}[/dim]"
        )
        result = await run_best_of_n(problem, n=3, config=output_config)
        render_candidates(result, console)

    asyncio.run(main())
