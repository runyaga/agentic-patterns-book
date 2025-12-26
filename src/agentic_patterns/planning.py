"""
Planning Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 6:
Break down complex goals into manageable steps and execute sequentially.

The Planning pattern enables agents to:
1. Decompose high-level objectives into actionable steps
2. Maintain a roadmap to track progress
3. Handle dependencies between steps
4. Adapt plans dynamically if errors occur

Key implementation: Plan-then-Execute model with optional re-planning.
"""

from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from agentic_patterns._models import get_model


# --8<-- [start:models]
class StepStatus(str, Enum):
    """Status of a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """A single step in a plan."""

    step_number: int = Field(description="Step order (1-indexed)")
    description: str = Field(description="What this step accomplishes")
    expected_output: str = Field(description="Expected result of this step")
    dependencies: list[int] = Field(
        default_factory=list,
        description="Step numbers this depends on",
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Current status",
    )


class Plan(BaseModel):
    """A complete plan with multiple steps."""

    goal: str = Field(description="The high-level objective")
    steps: list[PlanStep] = Field(description="Ordered list of steps")
    reasoning: str = Field(description="Why this plan was chosen")


class StepResult(BaseModel):
    """Result from executing a single step."""

    step_number: int = Field(description="Which step was executed")
    success: bool = Field(description="Whether the step succeeded")
    output: str = Field(description="Output or error message")
    needs_replan: bool = Field(
        default=False,
        description="Whether re-planning is recommended",
    )


class PlanExecutionResult(BaseModel):
    """Final result after executing a plan."""

    goal: str = Field(description="Original goal")
    success: bool = Field(description="Overall success")
    completed_steps: int = Field(description="Number of completed steps")
    total_steps: int = Field(description="Total steps in plan")
    step_results: list[StepResult] = Field(
        default_factory=list,
        description="Results from each step",
    )
    final_output: str = Field(description="Final synthesized output")
    replanned: bool = Field(
        default=False,
        description="Whether re-planning occurred",
    )


# --8<-- [end:models]


# --8<-- [start:agents]
# Initialize the model
model = get_model()

# Planner agent - creates plans from goals
planner_agent = Agent(
    model,
    system_prompt=(
        "You are a strategic planner. Given a goal, create a detailed plan "
        "with clear, actionable steps. Each step needs: step_number (int), "
        "description (what to do), expected_output (result), and "
        "dependencies (list of step numbers it depends on, or empty list). "
        "Order steps logically from first to last."
    ),
    output_type=Plan,
)

# Executor agent - executes individual steps
executor_agent = Agent(
    model,
    system_prompt=(
        "You are a task executor. Given a step description and context, "
        "execute the step and report the result. Be thorough but concise. "
        "If the step cannot be completed, explain why clearly. "
        "Respond with step_number, success (true/false), output, and "
        "needs_replan (true/false). Do not wrap output in markdown."
    ),
    output_type=StepResult,
)

# Replanner agent - adjusts plans when needed
replanner_agent = Agent(
    model,
    system_prompt=(
        "You are a plan adjuster. Given a failed step and current context, "
        "decide whether to retry, skip, or modify the remaining plan. "
        "Provide an updated plan that accounts for the failure."
    ),
    output_type=Plan,
)

# Synthesizer agent - combines results into final output
synthesizer_agent = Agent(
    model,
    system_prompt=(
        "You are a result synthesizer. Given completed step results, "
        "combine them into a coherent final answer that addresses "
        "the original goal. Be comprehensive but concise."
    ),
    output_type=str,
)
# --8<-- [end:agents]


# --8<-- [start:planning]
async def create_plan(goal: str, max_steps: int = 5) -> Plan:
    """
    Create a plan to achieve a goal.

    Args:
        goal: The high-level objective to accomplish.
        max_steps: Maximum number of steps in the plan (default: 5).

    Returns:
        A Plan with ordered steps to achieve the goal.
    """
    print(f"Planning: Creating plan for '{goal}'...")

    result = await planner_agent.run(
        f"Create a plan to achieve this goal: {goal}\n"
        f"Limit the plan to at most {max_steps} steps.\n"
        f"Each step should be actionable and specific."
    )

    plan = result.output
    print(f"  Created plan with {len(plan.steps)} steps")

    return plan


async def execute_step(
    step: PlanStep,
    context: str = "",
) -> StepResult:
    """
    Execute a single plan step.

    Args:
        step: The step to execute.
        context: Context from previous steps (optional).

    Returns:
        StepResult with success status and output.
    """
    print(f"  Executing step {step.step_number}: {step.description[:50]}...")

    result = await executor_agent.run(
        f"Execute this step:\n"
        f"Step {step.step_number}: {step.description}\n"
        f"Expected output: {step.expected_output}\n\n"
        f"Context from previous steps:\n{context or 'No previous context'}"
    )

    step_result = result.output
    step_result.step_number = step.step_number

    status = "SUCCESS" if step_result.success else "FAILED"
    print(f"    Step {step.step_number}: {status}")

    return step_result


async def replan(
    original_plan: Plan,
    failed_step: StepResult,
    completed_results: list[StepResult],
) -> Plan:
    """
    Create a new plan after a step failure.

    Args:
        original_plan: The original plan that had a failure.
        failed_step: The step that failed.
        completed_results: Results from previously completed steps.

    Returns:
        A revised Plan to continue toward the goal.
    """
    print(f"  Re-planning after step {failed_step.step_number} failure...")

    context = "\n".join(
        f"Step {r.step_number}: {r.output}" for r in completed_results
    )

    result = await replanner_agent.run(
        f"Original goal: {original_plan.goal}\n\n"
        f"Failed step {failed_step.step_number}: {failed_step.output}\n\n"
        f"Completed steps context:\n{context}\n\n"
        f"Create a revised plan to continue toward the goal."
    )

    new_plan = result.output
    print(f"  Created revised plan with {len(new_plan.steps)} steps")

    return new_plan


async def execute_plan(
    plan: Plan,
    allow_replan: bool = True,
    max_retries: int = 1,
) -> PlanExecutionResult:
    """
    Execute a complete plan step by step.

    Args:
        plan: The plan to execute.
        allow_replan: Whether to allow re-planning on failures.
        max_retries: Max re-planning attempts (default: 1).

    Returns:
        PlanExecutionResult with all step outcomes.
    """
    print(f"Executing plan for: {plan.goal}")
    print(f"  Total steps: {len(plan.steps)}")

    current_plan = plan
    step_results: list[StepResult] = []
    replanned = False
    retry_count = 0

    # Build context as we execute
    context_parts: list[str] = []

    for step in current_plan.steps:
        # Check dependencies
        deps_met = all(
            any(r.step_number == d and r.success for r in step_results)
            for d in step.dependencies
        )

        if not deps_met and step.dependencies:
            print(f"  Skipping step {step.step_number}: dependencies not met")
            step_results.append(
                StepResult(
                    step_number=step.step_number,
                    success=False,
                    output="Skipped: dependencies not met",
                )
            )
            continue

        # Execute the step
        context = "\n".join(context_parts)
        result = await execute_step(step, context)
        step_results.append(result)

        if result.success:
            context_parts.append(f"Step {step.step_number}: {result.output}")
        elif allow_replan and retry_count < max_retries:
            # Attempt to replan
            retry_count += 1
            replanned = True
            current_plan = await replan(current_plan, result, step_results)
            # Continue with new plan (simplified - just log)
            print("  Continuing with revised plan...")

    # Synthesize final output
    completed = [r for r in step_results if r.success]
    print(f"  Synthesizing results from {len(completed)} completed steps...")

    outputs = "\n".join(f"Step {r.step_number}: {r.output}" for r in completed)

    synth_result = await synthesizer_agent.run(
        f"Goal: {plan.goal}\n\n"
        f"Completed step outputs:\n{outputs}\n\n"
        f"Provide a final answer that addresses the goal."
    )

    return PlanExecutionResult(
        goal=plan.goal,
        success=len(completed) == len(current_plan.steps),
        completed_steps=len(completed),
        total_steps=len(current_plan.steps),
        step_results=step_results,
        final_output=synth_result.output,
        replanned=replanned,
    )


async def plan_and_execute(
    goal: str,
    max_steps: int = 5,
    allow_replan: bool = True,
) -> PlanExecutionResult:
    """
    Convenience function: create a plan and execute it.

    Args:
        goal: The high-level objective to accomplish.
        max_steps: Maximum steps in the plan (default: 5).
        allow_replan: Whether to allow re-planning on failures.

    Returns:
        PlanExecutionResult with the final outcome.
    """
    plan = await create_plan(goal, max_steps)
    return await execute_plan(plan, allow_replan)


# --8<-- [end:planning]


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Planning Pattern")
        print("=" * 60)

        # Demo: Plan and execute a research task
        result = await plan_and_execute(
            goal=(
                "Research the key benefits of Python for data science "
                "and summarize the top 3 reasons"
            ),
            max_steps=4,
            allow_replan=True,
        )

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(f"Goal: {result.goal}")
        print(f"Success: {result.success}")
        print(f"Steps: {result.completed_steps}/{result.total_steps}")
        print(f"Replanned: {result.replanned}")
        print(f"\nFinal Output:\n{result.final_output}")

    asyncio.run(main())
