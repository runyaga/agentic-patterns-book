"""
Multi-Agent Collaboration Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 7:
Multiple specialized agents collaborate to achieve complex goals that
exceed the capabilities of a single agent.

Key concepts:
- Specialized roles: Each agent has domain expertise
- Collaboration structures: Supervisor, Network, Hierarchical
- Inter-agent communication: Message passing between agents
- Task decomposition: Break complex problems into sub-tasks

This module implements the Supervisor pattern where a coordinator
delegates tasks to specialized worker agents.

Example usage:
    result = await run_collaborative_task(
        "Research and summarize Python async patterns"
    )
"""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai import RunContext

from agentic_patterns._models import get_model


# --8<-- [start:models]
class AgentRole(str, Enum):
    """Roles that agents can have in collaboration."""

    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"


class TaskStatus(str, Enum):
    """Status of a delegated task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentMessage(BaseModel):
    """Message passed between agents."""

    sender: AgentRole = Field(description="Role of the sending agent")
    recipient: AgentRole = Field(description="Role of the receiving agent")
    content: str = Field(description="Message content")
    task_id: int | None = Field(
        default=None, description="Associated task ID if applicable"
    )


class DelegatedTask(BaseModel):
    """A task delegated from supervisor to worker."""

    task_id: int = Field(description="Unique task identifier")
    assigned_to: AgentRole = Field(description="Agent role to handle this")
    description: str = Field(description="What needs to be done")
    context: str = Field(
        default="", description="Additional context for the task"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current status"
    )


class TaskResult(BaseModel):
    """Result from a worker agent completing a task."""

    task_id: int = Field(description="Which task was completed")
    agent_role: AgentRole = Field(description="Which agent completed it")
    success: bool = Field(description="Whether the task succeeded")
    output: str = Field(description="Task output or error message")
    artifacts: list[str] = Field(
        default_factory=list,
        description="Any artifacts produced (summaries, code, etc.)",
    )


class SupervisorPlan(BaseModel):
    """Supervisor's plan for delegating work."""

    objective: str = Field(description="High-level objective")
    tasks: list[DelegatedTask] = Field(description="Tasks to delegate")
    reasoning: str = Field(description="Why this delegation strategy")


class CollaborationResult(BaseModel):
    """Final result from multi-agent collaboration."""

    objective: str = Field(description="Original objective")
    success: bool = Field(description="Overall success")
    task_results: list[TaskResult] = Field(
        description="Results from each delegated task"
    )
    final_output: str = Field(description="Synthesized final output")
    messages_exchanged: int = Field(
        default=0, description="Number of inter-agent messages"
    )


@dataclass
class CollaborationContext:
    """Runtime context for collaboration."""

    messages: list[AgentMessage]
    task_results: list[TaskResult]
# --8<-- [end:models]


# --8<-- [start:agents]
# Initialize the model
model = get_model()


# Supervisor agent - coordinates and delegates
supervisor_agent = Agent(
    model,
    system_prompt=(
        "You are a supervisor agent coordinating a team of specialists. "
        "Given an objective, break it into tasks and assign to workers:\n"
        "- researcher: Gathers information, finds sources\n"
        "- analyst: Analyzes data, identifies patterns\n"
        "- writer: Creates content, drafts documents\n"
        "- reviewer: Reviews quality, suggests improvements\n\n"
        "Create a clear delegation plan with specific tasks."
    ),
    output_type=SupervisorPlan,
)

# Researcher agent - gathers information
researcher_agent = Agent(
    model,
    system_prompt=(
        "You are a research specialist. Your job is to gather information, "
        "find relevant sources, and compile research findings. "
        "Be thorough but focused on the specific research request. "
        "Provide factual, well-organized research output."
    ),
    output_type=TaskResult,
    deps_type=CollaborationContext,
)

# Analyst agent - analyzes information
analyst_agent = Agent(
    model,
    system_prompt=(
        "You are an analysis specialist. Your job is to analyze information, "
        "identify patterns, draw insights, and provide structured analysis. "
        "Be analytical and evidence-based in your conclusions. "
        "Highlight key findings and their implications."
    ),
    output_type=TaskResult,
    deps_type=CollaborationContext,
)

# Writer agent - creates content
writer_agent = Agent(
    model,
    system_prompt=(
        "You are a writing specialist. Your job is to create clear, "
        "well-structured content based on provided information. "
        "Adapt your writing style to the task requirements. "
        "Focus on clarity, coherence, and engagement."
    ),
    output_type=TaskResult,
    deps_type=CollaborationContext,
)

# Reviewer agent - reviews and improves
reviewer_agent = Agent(
    model,
    system_prompt=(
        "You are a review specialist. Your job is to evaluate work quality, "
        "identify issues, and suggest improvements. "
        "Be constructive in your feedback. "
        "Focus on accuracy, clarity, and completeness."
    ),
    output_type=TaskResult,
    deps_type=CollaborationContext,
)

# Synthesizer agent - combines results
synthesizer_agent = Agent(
    model,
    system_prompt=(
        "You are a synthesis specialist. Given multiple task results from "
        "different agents, combine them into a coherent final output. "
        "Integrate the best elements from each contribution. "
        "Ensure the final output addresses the original objective."
    ),
    output_type=str,
)

# Map roles to agents
ROLE_AGENTS = {
    AgentRole.RESEARCHER: researcher_agent,
    AgentRole.ANALYST: analyst_agent,
    AgentRole.WRITER: writer_agent,
    AgentRole.REVIEWER: reviewer_agent,
}
# --8<-- [end:agents]


# --8<-- [start:collaboration]
def _find_result_by_role(
    task_results: list[TaskResult],
    role: str,
) -> str:
    """Helper to find results from a specific role."""
    for result in task_results:
        if result.agent_role.value == role:
            return result.output
    return "No previous results found for this role."


@researcher_agent.tool
async def researcher_get_previous(
    ctx: RunContext[CollaborationContext],
    role: str,
) -> str:
    """
    Get results from a previous agent's work.

    Args:
        ctx: Run context with collaboration state.
        role: The role whose results to retrieve.

    Returns:
        Previous results or empty string if none found.
    """
    return _find_result_by_role(ctx.deps.task_results, role)


@analyst_agent.tool
async def analyst_get_previous(
    ctx: RunContext[CollaborationContext],
    role: str,
) -> str:
    """Get results from a previous agent's work."""
    return _find_result_by_role(ctx.deps.task_results, role)


@writer_agent.tool
async def writer_get_previous(
    ctx: RunContext[CollaborationContext],
    role: str,
) -> str:
    """Get results from a previous agent's work."""
    return _find_result_by_role(ctx.deps.task_results, role)


@reviewer_agent.tool
async def reviewer_get_previous(
    ctx: RunContext[CollaborationContext],
    role: str,
) -> str:
    """Get results from a previous agent's work."""
    return _find_result_by_role(ctx.deps.task_results, role)


async def create_delegation_plan(objective: str) -> SupervisorPlan:
    """
    Create a delegation plan for an objective.

    The supervisor agent analyzes the objective and creates tasks
    for specialized worker agents.

    Args:
        objective: The high-level goal to accomplish.

    Returns:
        SupervisorPlan with delegated tasks.
    """
    print(f"Supervisor: Planning delegation for '{objective[:50]}...'")

    result = await supervisor_agent.run(
        f"Create a delegation plan for this objective:\n\n{objective}\n\n"
        f"Assign tasks to appropriate specialists. "
        f"Consider dependencies between tasks."
    )

    plan = result.output
    print(f"  Created plan with {len(plan.tasks)} tasks")
    for task in plan.tasks:
        print(f"    - Task {task.task_id}: {task.assigned_to.value}")

    return plan


async def execute_task(
    task: DelegatedTask,
    context: CollaborationContext,
) -> TaskResult:
    """
    Execute a delegated task using the appropriate worker agent.

    Args:
        task: The task to execute.
        context: Collaboration context with previous results.

    Returns:
        TaskResult from the worker agent.
    """
    print(f"  Worker ({task.assigned_to.value}): Starting task {task.task_id}")

    agent = ROLE_AGENTS.get(task.assigned_to)
    if agent is None:
        return TaskResult(
            task_id=task.task_id,
            agent_role=task.assigned_to,
            success=False,
            output=f"No agent available for role: {task.assigned_to.value}",
        )

    # Build context from previous results
    prev_outputs = "\n\n".join(
        f"[{r.agent_role.value}]: {r.output}" for r in context.task_results
    )

    prompt = (
        f"Task: {task.description}\n\n"
        f"Context: {task.context}\n\n"
        f"Previous work:\n{prev_outputs or 'None'}\n\n"
        f"Complete this task and provide your output."
    )

    result = await agent.run(prompt, deps=context)
    task_result = result.output
    task_result.task_id = task.task_id
    task_result.agent_role = task.assigned_to

    status = "SUCCESS" if task_result.success else "FAILED"
    print(f"    Task {task.task_id}: {status}")

    return task_result


async def synthesize_results(
    objective: str,
    task_results: list[TaskResult],
) -> str:
    """
    Synthesize task results into a final output.

    Args:
        objective: Original objective.
        task_results: Results from all completed tasks.

    Returns:
        Synthesized final output string.
    """
    print("  Synthesizer: Combining results...")

    completed = [r for r in task_results if r.success]
    outputs = "\n\n".join(
        f"[{r.agent_role.value}] Task {r.task_id}:\n{r.output}"
        for r in completed
    )

    result = await synthesizer_agent.run(
        f"Objective: {objective}\n\n"
        f"Completed task outputs:\n{outputs}\n\n"
        f"Synthesize these into a final, coherent response."
    )

    return result.output


async def run_collaborative_task(
    objective: str,
    max_tasks: int = 4,
) -> CollaborationResult:
    """
    Run a collaborative task with multiple agents.

    This implements the Supervisor pattern where a supervisor agent
    delegates tasks to specialized workers, then synthesizes results.

    Args:
        objective: The high-level goal to accomplish.
        max_tasks: Maximum number of tasks to create.

    Returns:
        CollaborationResult with all task outcomes.
    """
    print("=" * 60)
    print("Multi-Agent Collaboration: Starting")
    print("=" * 60)
    print(f"Objective: {objective[:80]}...")

    # Initialize collaboration context
    context = CollaborationContext(messages=[], task_results=[])

    # Phase 1: Supervisor creates delegation plan
    plan = await create_delegation_plan(objective)

    # Limit tasks if needed
    tasks = plan.tasks[:max_tasks]

    # Phase 2: Execute tasks sequentially (respecting dependencies)
    print("\nExecuting tasks...")
    for task in tasks:
        result = await execute_task(task, context)
        context.task_results.append(result)

        # Add message to log
        msg_content = f"Completed task {task.task_id}: {result.output[:100]}"
        context.messages.append(
            AgentMessage(
                sender=task.assigned_to,
                recipient=AgentRole.SUPERVISOR,
                content=msg_content,
                task_id=task.task_id,
            )
        )

    # Phase 3: Synthesize final output
    print("\nSynthesizing results...")
    final_output = await synthesize_results(objective, context.task_results)

    # Calculate success
    completed = sum(1 for r in context.task_results if r.success)
    success = completed == len(tasks) and len(tasks) > 0

    print("\n" + "=" * 60)
    print(f"Collaboration complete: {completed}/{len(tasks)} tasks succeeded")

    return CollaborationResult(
        objective=objective,
        success=success,
        task_results=context.task_results,
        final_output=final_output,
        messages_exchanged=len(context.messages),
    )


async def run_network_collaboration(
    objective: str,
    agents_to_consult: list[AgentRole],
) -> CollaborationResult:
    """
    Run a network-style collaboration where agents work in parallel.

    In network collaboration, multiple agents work on the same objective
    independently, and their outputs are combined.

    Args:
        objective: The goal for all agents.
        agents_to_consult: Which agent roles to involve.

    Returns:
        CollaborationResult with combined outputs.
    """
    import asyncio

    print("=" * 60)
    print("Network Collaboration: Starting")
    print("=" * 60)
    print(f"Consulting: {[r.value for r in agents_to_consult]}")

    context = CollaborationContext(messages=[], task_results=[])

    async def consult_agent(role: AgentRole, task_id: int) -> TaskResult:
        agent = ROLE_AGENTS.get(role)
        if agent is None:
            return TaskResult(
                task_id=task_id,
                agent_role=role,
                success=False,
                output=f"No agent for role: {role.value}",
            )

        result = await agent.run(
            f"Provide your perspective on: {objective}",
            deps=context,
        )
        task_result = result.output
        task_result.task_id = task_id
        task_result.agent_role = role
        return task_result

    # Run all agents in parallel
    tasks = [
        consult_agent(role, i)
        for i, role in enumerate(agents_to_consult, start=1)
    ]
    task_results = await asyncio.gather(*tasks)

    context.task_results = list(task_results)

    # Synthesize
    final_output = await synthesize_results(objective, context.task_results)

    completed = sum(1 for r in task_results if r.success)
    success = completed == len(agents_to_consult)

    return CollaborationResult(
        objective=objective,
        success=success,
        task_results=list(task_results),
        final_output=final_output,
        messages_exchanged=0,
    )
# --8<-- [end:collaboration]


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Multi-Agent Collaboration Pattern")
        print("=" * 60)

        # Demo 1: Supervisor pattern
        print("\n--- Supervisor Pattern ---")
        result = await run_collaborative_task(
            objective=(
                "Research the key benefits of using async/await in Python "
                "and create a brief summary for developers new to async."
            ),
            max_tasks=3,
        )

        print("\nFINAL RESULT (Supervisor):")
        print(f"Success: {result.success}")
        print(f"Tasks completed: {len(result.task_results)}")
        print(f"Messages exchanged: {result.messages_exchanged}")
        print(f"\nOutput:\n{result.final_output[:500]}...")

        # Demo 2: Network pattern
        print("\n\n--- Network Pattern ---")
        result2 = await run_network_collaboration(
            objective="What are best practices for Python error handling?",
            agents_to_consult=[
                AgentRole.RESEARCHER,
                AgentRole.ANALYST,
                AgentRole.WRITER,
            ],
        )

        print("\nFINAL RESULT (Network):")
        print(f"Success: {result2.success}")
        print(f"Agents consulted: {len(result2.task_results)}")
        print(f"\nOutput:\n{result2.final_output[:500]}...")

    asyncio.run(main())
