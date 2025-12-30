"""
Dynamic Planning Pattern Implementation.

Based on "Learning When to Plan" (Paglieri et al., 2025):
Agents learn WHEN to plan, not just HOW to plan.

Key insight: Each task has a "Goldilocks" frequency for planning that
outperforms both always-planning (ReAct) and never-planning strategies.

The agent decides at each step whether to:
- Generate a new plan (dt=1): outputs <plan>...</plan> followed by action
- Continue with existing plan (dt=0): outputs action only

This extends Chapter 6 (static Planning) with runtime planning decisions.
"""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum

import logfire
from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from agentic_patterns._models import get_model
from agentic_patterns._utils import HistoryBuffer
from agentic_patterns._utils import MetricsCollector
from agentic_patterns._utils import parse_xml_tag
from agentic_patterns._utils import strip_xml_tags


# --8<-- [start:models]
class PlanningMode(str, Enum):
    """How the agent decides when to plan."""

    DYNAMIC = "dynamic"  # Agent decides (Prompt 20)
    ALWAYS = "always"  # Plan every step (ReAct baseline)
    NEVER = "never"  # Never plan (action-only baseline)


@dataclass
class DynamicPlan:
    """A natural language plan with metadata."""

    content: str
    created_at_step: int
    steps_since_creation: int = 0

    def age(self) -> int:
        """How many steps since plan was created."""
        return self.steps_since_creation


class StepOutput(BaseModel):
    """Output from a single agent step."""

    plan: str | None = Field(
        default=None,
        description="New plan if agent decided to plan, None otherwise",
    )
    action: str = Field(description="The action to execute")
    decided_to_plan: bool = Field(
        default=False,
        description="Whether the agent generated a new plan this step",
    )
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")


class DynamicPlanningResult(BaseModel):
    """Result of a complete dynamic planning episode."""

    goal: str = Field(description="The original goal")
    total_steps: int = Field(description="Total steps executed")
    plans_generated: int = Field(description="Number of times agent planned")
    planning_frequency: float = Field(
        description="plans_generated / total_steps"
    )
    actions: list[str] = Field(description="All actions taken")
    final_observation: str = Field(description="Final environment state")
    success: bool = Field(
        default=True, description="Whether goal was achieved"
    )
    # Token accounting
    total_input_tokens: int = Field(
        default=0, description="Total input tokens"
    )
    total_output_tokens: int = Field(
        default=0, description="Total output tokens"
    )
    plan_tokens: int = Field(
        default=0, description="Tokens used when planning"
    )
    action_tokens: int = Field(
        default=0, description="Tokens used for action-only steps"
    )


class PlanningMetrics(BaseModel):
    """Metrics from a dynamic planning run."""

    total_plans: int
    total_steps: int
    planning_frequency: float
    avg_plan_age_at_replan: float
    plan_lengths: list[int]


# --8<-- [end:models]


# --8<-- [start:prompts]
# Prompt 20 from paper: Dynamic Planning (agent decides)
DYNAMIC_PLANNING_PROMPT = """\
You are an agent that decides when to create or update plans.

Review your current plan and observations.
- If you do not have a plan yet, create one.
- If your plan is outdated or needs changes, create a new plan.

If you create a new plan, output it in the following format:
<plan>YOUR_NEW_PLAN</plan>

If your current plan is still valid, proceed without outputting it again.

After this evaluation (and any necessary replanning), output exactly ONE \
action from the allowed actions.
Output nothing else except an optional <plan>...</plan> block and that \
single action.
"""

# High-level strategic planning
HIGH_LEVEL_PROMPT = """\
You are a strategic planner focused on long-horizon goals.

Re-evaluate the overall strategy. Outline your current high-level plan
for completing the task, focusing on the major phases ahead.

Output format:
<plan>YOUR_HIGH_LEVEL_PLAN</plan>
[Your chosen action]
"""

# Short-term tactical planning
SHORT_TERM_PROMPT = """\
You are a tactical planner focused on immediate next steps.

Detail the specific sequence of actions you intend to take over the
next few steps. Explain the purpose in relation to the current situation.

Output format:
<plan>YOUR_SHORT_TERM_PLAN</plan>
[Your chosen action]
"""

# Action-only baseline (never plan)
ACTION_ONLY_PROMPT = """\
You are an agent that takes actions to achieve goals.

Look at your previous plan and observations, then choose exactly ONE
action from the allowed actions. Output no other text.
"""
# --8<-- [end:prompts]


# --8<-- [start:context]
@dataclass
class AgentContext:
    """
    Context passed to the agent at each step.

    Includes current observation, history, and existing plan.
    """

    goal: str
    observation: str
    history: HistoryBuffer = field(default_factory=HistoryBuffer)
    current_plan: DynamicPlan | None = None
    step: int = 0
    available_actions: list[str] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format context as prompt input."""
        parts = [f"Goal: {self.goal}", f"Step: {self.step}"]

        if self.current_plan:
            parts.append(f"Current Plan: {self.current_plan.content}")
            parts.append(
                f"Plan Age: {self.current_plan.steps_since_creation} steps"
            )
        else:
            parts.append("Current Plan: None")

        if self.history:
            parts.append(f"History:\n{self.history.as_context()}")

        parts.append(f"Current Observation: {self.observation}")

        if self.available_actions:
            actions_str = ", ".join(self.available_actions)
            parts.append(f"Available Actions: {actions_str}")

        return "\n\n".join(parts)


# --8<-- [end:context]


# --8<-- [start:agents]
def get_default_prompt(mode: PlanningMode) -> str:
    """Get the default system prompt for a planning mode."""
    prompts = {
        PlanningMode.DYNAMIC: DYNAMIC_PLANNING_PROMPT,
        PlanningMode.ALWAYS: HIGH_LEVEL_PROMPT,
        PlanningMode.NEVER: ACTION_ONLY_PROMPT,
    }
    return prompts[mode]


def create_dynamic_agent(
    model: Model | None = None,
    mode: PlanningMode = PlanningMode.DYNAMIC,
    system_prompt: str | None = None,
) -> Agent[None, str]:
    """
    Create a dynamic planning agent.

    Args:
        model: pydantic-ai Model instance. If None, uses default.
        mode: Planning mode (DYNAMIC, ALWAYS, or NEVER).
        system_prompt: Custom system prompt. If None, uses default for mode.

    Returns:
        Configured agent that outputs raw text (plan + action).
    """
    prompt = system_prompt if system_prompt else get_default_prompt(mode)

    return Agent(
        model or get_model(),
        system_prompt=prompt,
        output_type=str,
    )


# Default agent (created lazily)
_default_agent: Agent[None, str] | None = None


def _get_default_agent() -> Agent[None, str]:
    """Get or create the default dynamic planning agent."""
    global _default_agent
    if _default_agent is None:
        _default_agent = create_dynamic_agent()
    return _default_agent


# --8<-- [end:agents]


# --8<-- [start:parsing]
def parse_agent_output(raw: str, mode: PlanningMode) -> StepOutput:
    """
    Parse raw agent output into structured StepOutput.

    Extracts <plan>...</plan> if present, and the action.

    Args:
        raw: Raw text output from agent.
        mode: Planning mode (affects parsing expectations).

    Returns:
        StepOutput with plan (if any) and action.
    """
    plan = parse_xml_tag(raw, "plan")
    decided_to_plan = plan is not None

    # Extract action (everything after plan tag, or whole output if no plan)
    action = strip_xml_tags(raw, "plan") if plan else raw.strip()

    # In ALWAYS mode, we expect a plan every time
    if mode == PlanningMode.ALWAYS and not plan:
        # Treat entire output as both plan and action
        plan = raw.strip()
        decided_to_plan = True

    return StepOutput(
        plan=plan,
        action=action,
        decided_to_plan=decided_to_plan,
    )


# --8<-- [end:parsing]


# --8<-- [start:step]
async def step(
    ctx: AgentContext,
    agent: Agent[None, str] | None = None,
    mode: PlanningMode = PlanningMode.DYNAMIC,
) -> StepOutput:
    """
    Execute a single step: decide whether to plan, then act.

    Args:
        ctx: Current agent context.
        agent: Agent to use. If None, uses default.
        mode: Planning mode.

    Returns:
        StepOutput with optional new plan, action, and token usage.
    """
    the_agent = agent or _get_default_agent()

    # Format context for prompt
    prompt = ctx.format_for_prompt()

    # Run agent with logfire span
    with logfire.span(
        "dynamic_planning.step",
        step=ctx.step,
        has_plan=ctx.current_plan is not None,
    ):
        result = await the_agent.run(prompt)
        raw_output = result.output

        # Extract token usage from pydantic_ai result
        usage = result.usage()
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0

    # Parse output
    output = parse_agent_output(raw_output, mode)

    # Add token info
    output.input_tokens = input_tokens
    output.output_tokens = output_tokens

    logfire.info(
        "step_complete",
        step=ctx.step,
        decided_to_plan=output.decided_to_plan,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return output


# --8<-- [end:step]


# --8<-- [start:episode]
async def run_episode(
    goal: str,
    get_observation: callable,
    execute_action: callable,
    available_actions: list[str] | None = None,
    max_steps: int = 100,
    mode: PlanningMode = PlanningMode.DYNAMIC,
    agent: Agent[None, str] | None = None,
    history_size: int = 10,
    on_step: callable | None = None,
    is_done: callable | None = None,
) -> DynamicPlanningResult:
    """
    Run a complete episode with dynamic planning.

    Args:
        goal: The goal to achieve.
        get_observation: Async callable returning current observation string.
        execute_action: Async callable that executes an action string.
        available_actions: List of allowed action strings (optional).
        max_steps: Maximum steps before stopping.
        mode: Planning mode (DYNAMIC, ALWAYS, NEVER).
        agent: Agent to use. If None, creates default.
        history_size: Max history entries to keep.
        on_step: Optional callback(step_num, output) for progress reporting.
        is_done: Optional callable returning bool, checked after each step.

    Returns:
        DynamicPlanningResult with metrics, token usage, and outcomes.
    """
    the_agent = agent or create_dynamic_agent(mode=mode)
    metrics = MetricsCollector()
    history = HistoryBuffer(max_size=history_size)
    current_plan: DynamicPlan | None = None
    actions: list[str] = []

    # Token accounting
    total_input_tokens = 0
    total_output_tokens = 0
    plan_tokens = 0
    action_tokens = 0

    # Get initial observation
    observation = await get_observation()

    with logfire.span("dynamic_planning.episode", goal=goal, mode=mode.value):
        for step_num in range(max_steps):
            # Build context
            ctx = AgentContext(
                goal=goal,
                observation=observation,
                history=history,
                current_plan=current_plan,
                step=step_num,
                available_actions=available_actions or [],
            )

            # Execute step
            output = await step(ctx, the_agent, mode)
            metrics.increment("steps")

            # Track tokens
            step_tokens = output.input_tokens + output.output_tokens
            total_input_tokens += output.input_tokens
            total_output_tokens += output.output_tokens

            # Update plan if agent decided to plan
            if output.decided_to_plan and output.plan:
                if current_plan:
                    age = current_plan.steps_since_creation
                    metrics.record("plan_age", age)
                current_plan = DynamicPlan(
                    content=output.plan,
                    created_at_step=step_num,
                )
                metrics.increment("plans")
                metrics.record("plan_length", len(output.plan))
                plan_tokens += step_tokens
            else:
                action_tokens += step_tokens

            # Execute action
            await execute_action(output.action)
            actions.append(output.action)

            # Callback for progress reporting
            if on_step:
                on_step(step_num, output)

            # Check for early termination
            if is_done and is_done():
                break

            # Update history
            history.add(observation, output.action)

            # Get new observation
            observation = await get_observation()

            # Age the plan
            if current_plan:
                current_plan.steps_since_creation += 1

        # Calculate results
        total_steps = metrics.get_count("steps")
        plans_generated = metrics.get_count("plans")

        logfire.info(
            "episode_complete",
            total_steps=total_steps,
            plans_generated=plans_generated,
            planning_frequency=(
                plans_generated / total_steps if total_steps else 0
            ),
            total_tokens=total_input_tokens + total_output_tokens,
            plan_tokens=plan_tokens,
            action_tokens=action_tokens,
        )

    return DynamicPlanningResult(
        goal=goal,
        total_steps=total_steps,
        plans_generated=plans_generated,
        planning_frequency=(
            plans_generated / total_steps if total_steps > 0 else 0.0
        ),
        actions=actions,
        final_observation=observation,
        success=True,  # Caller should determine actual success
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        plan_tokens=plan_tokens,
        action_tokens=action_tokens,
    )


# --8<-- [end:episode]


# --8<-- [start:main]
if __name__ == "__main__":
    import asyncio

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    def render_results(
        result: DynamicPlanningResult,
        step_log: list[StepOutput],
        console: Console,
    ) -> None:
        """Render episode results with rich visualization."""
        console.print()
        console.rule("[bold blue]Dynamic Planning Results")
        console.print()

        # Step-by-step table
        table = Table(
            title="[bold]Step Execution Log[/bold]",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Step", style="dim", width=5, justify="center")
        table.add_column("Decision", width=10, justify="center")
        table.add_column("Action", width=40)
        table.add_column("Tokens", width=12, justify="right")

        for i, step_out in enumerate(step_log):
            decided = step_out.decided_to_plan
            decision = "[green]PLAN[/]" if decided else "[dim]act[/]"
            act = step_out.action
            action_text = act[:37] + "..." if len(act) > 40 else act
            tok = f"{step_out.input_tokens}+{step_out.output_tokens}"
            table.add_row(str(i), decision, action_text, tok)

        console.print(table)
        console.print()

        # Token breakdown
        total_tokens = result.total_input_tokens + result.total_output_tokens
        plan_pct = (
            result.plan_tokens / total_tokens * 100 if total_tokens > 0 else 0
        )
        action_pct = (
            result.action_tokens / total_tokens * 100
            if total_tokens > 0
            else 0
        )

        token_table = Table(
            title="[bold]Token Accounting[/bold]",
            show_header=True,
            header_style="bold magenta",
        )
        token_table.add_column("Category", width=20)
        token_table.add_column("Tokens", width=10, justify="right")
        token_table.add_column("Percent", width=10, justify="right")

        token_table.add_row(
            "[green]Planning steps[/]",
            str(result.plan_tokens),
            f"{plan_pct:.1f}%",
        )
        token_table.add_row(
            "[blue]Action-only steps[/]",
            str(result.action_tokens),
            f"{action_pct:.1f}%",
        )
        token_table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_tokens}[/bold]",
            "100%",
            style="bold",
        )

        console.print(token_table)
        console.print()

        # Goldilocks frequency visualization
        freq = result.planning_frequency
        freq_bar = "█" * int(freq * 20) + "░" * (20 - int(freq * 20))

        if 0.2 <= freq <= 0.5:
            freq_color = "green"
            freq_label = "Goldilocks zone"
        elif freq < 0.2:
            freq_color = "yellow"
            freq_label = "Rarely planning"
        else:
            freq_color = "red"
            freq_label = "Over-planning"

        console.print(
            Panel(
                f"[bold]Planning Frequency:[/bold] {freq:.1%}\n\n"
                f"Never  [{freq_color}]{freq_bar}[/]  Always\n"
                f"0%                              100%\n\n"
                f"[dim]{freq_label} - "
                f"{result.plans_generated} plans / "
                f"{result.total_steps} steps[/dim]",
                title="[bold yellow]★ Goldilocks Indicator[/]",
                border_style=freq_color,
            )
        )

        console.print()
        steps = result.total_steps
        tps = total_tokens / steps if steps > 0 else 0
        console.print(
            Panel(
                f"[bold]Goal:[/bold] {result.goal}\n"
                f"[bold]Steps:[/bold] {result.total_steps}\n"
                f"[bold]Plans:[/bold] {result.plans_generated}\n"
                f"[bold]Total Tokens:[/bold] {total_tokens:,}\n"
                f"[bold]Tokens/Step:[/bold] {tps:.0f}",
                title="[bold]Episode Summary[/bold]",
                border_style="blue",
            )
        )

    # =========================================================================
    # PROJECT SETUP ENVIRONMENT - Task Decomposition Demo
    # =========================================================================
    # Demonstrates dynamic planning in a practical scenario:
    # - Agent sets up a Python project with tests, CI, and docs
    # - Plans when entering new phases, acts within phases
    # - No spatial reasoning - just file/task management
    # =========================================================================

    class ProjectPhase(str, Enum):
        """Phases of project setup."""

        INIT = "init"  # Create basic structure
        TESTS = "tests"  # Add test infrastructure
        CI = "ci"  # Set up CI/CD
        DOCS = "docs"  # Add documentation
        DONE = "done"  # All complete

    @dataclass
    class ProjectFile:
        """A file in the project."""

        name: str
        content: str = ""
        exists: bool = False

    class ProjectSetupEnvironment:
        """
        Simulates setting up a Python project.

        Phases:
        1. INIT: Create pyproject.toml, src/, README
        2. TESTS: Add pytest, conftest.py, test_main.py
        3. CI: Create .github/workflows/ci.yml
        4. DOCS: Add mkdocs.yml, docs/index.md

        Agent should plan at phase transitions, act within phases.
        """

        PHASE_ORDER = [
            ProjectPhase.INIT,
            ProjectPhase.TESTS,
            ProjectPhase.CI,
            ProjectPhase.DOCS,
            ProjectPhase.DONE,
        ]

        # Required files per phase
        PHASE_REQUIREMENTS: dict[ProjectPhase, list[str]] = {
            ProjectPhase.INIT: [
                "pyproject.toml", "src/__init__.py", "README.md"
            ],
            ProjectPhase.TESTS: ["tests/conftest.py", "tests/test_main.py"],
            ProjectPhase.CI: [".github/workflows/ci.yml"],
            ProjectPhase.DOCS: ["mkdocs.yml", "docs/index.md"],
            ProjectPhase.DONE: [],
        }

        # File templates (abbreviated content)
        FILE_TEMPLATES: dict[str, str] = {
            "pyproject.toml": '[project]\nname = "myproject"',
            "src/__init__.py": '__version__ = "0.1.0"',
            "README.md": "# My Project",
            "tests/conftest.py": "import pytest",
            "tests/test_main.py": "def test_version(): pass",
            ".github/workflows/ci.yml": "name: CI\non: [push]",
            "mkdocs.yml": "site_name: My Project",
            "docs/index.md": "# Welcome",
        }

        def __init__(self) -> None:
            self.current_phase = ProjectPhase.INIT
            self.files: dict[str, ProjectFile] = {}
            self.last_action_result = "Project setup started"
            self.completed = False
            self.phase_just_changed = True  # Start requires planning

            # Initialize file tracking
            for name in self.FILE_TEMPLATES:
                self.files[name] = ProjectFile(name=name, exists=False)

        def _phase_progress(self, phase: ProjectPhase) -> tuple[int, int]:
            """Return (completed, total) files for a phase."""
            required = self.PHASE_REQUIREMENTS.get(phase, [])
            done = sum(
                1 for f in required
                if self.files.get(f, ProjectFile(f)).exists
            )
            return done, len(required)

        def _is_phase_complete(self, phase: ProjectPhase) -> bool:
            """Check if all files for a phase are created."""
            done, total = self._phase_progress(phase)
            return done == total

        def _advance_phase(self) -> None:
            """Move to next phase if current is complete."""
            while self._is_phase_complete(self.current_phase):
                idx = self.PHASE_ORDER.index(self.current_phase)
                if idx < len(self.PHASE_ORDER) - 1:
                    self.current_phase = self.PHASE_ORDER[idx + 1]
                    self.phase_just_changed = True  # Trigger planning
                    if self.current_phase == ProjectPhase.DONE:
                        self.completed = True
                        break
                else:
                    break

        def render(self) -> str:
            """Render project status as ASCII."""
            lines = ["╔══════════════════════════════════════╗"]
            lines.append("║     PROJECT SETUP STATUS             ║")
            lines.append("╠══════════════════════════════════════╣")

            for phase in self.PHASE_ORDER[:-1]:  # Skip DONE
                done, total = self._phase_progress(phase)
                is_current = phase == self.current_phase
                is_complete = done == total

                if is_complete:
                    icon = "✓"
                elif is_current:
                    icon = "▶"
                else:
                    icon = "○"

                bar = "█" * done + "░" * (total - done)
                name = phase.value.upper().ljust(6)
                lines.append(f"║ {icon} {name} [{bar}] {done}/{total}      ║")

            lines.append("╚══════════════════════════════════════╝")

            if self.completed:
                lines.append("\n  ★ PROJECT SETUP COMPLETE! ★")

            return "\n".join(lines)

        def get_observation(self) -> str:
            """Generate observation for agent."""
            obs = [self.render()]

            # What needs to be done
            required = self.PHASE_REQUIREMENTS.get(self.current_phase, [])
            pending = [f for f in required if not self.files[f].exists]

            if not pending:
                if self.completed:
                    obs.append("\nDONE!")
                return "\n".join(obs)

            # Phase transition trigger - MUST PLAN vs just act
            if self.phase_just_changed:
                phase = self.current_phase.value.upper()
                obs.append(f"\n[NEW PHASE: {phase}]")
                obs.append(f"Output: <plan>Plan for {phase}</plan>")
                obs.append(f"create {pending[0]}")
                self.phase_just_changed = False
            else:
                obs.append(f"\nOutput: create {pending[0]}")

            return "\n".join(obs)

        def execute(self, action: str) -> bool:
            """
            Execute an action. Returns True if project is complete.

            Actions:
            - create <filename>: Create a file
            - check: Check phase status
            - next_phase: Advance to next phase (if current complete)
            """
            action = action.lower().strip()

            # Parse create action
            if "create" in action:
                # Extract filename from action
                for fname in self.FILE_TEMPLATES:
                    if fname.lower() in action.lower():
                        return self._create_file(fname)
                # Try to extract any quoted filename
                import re
                match = re.search(r'["\']([^"\']+)["\']', action)
                if match:
                    return self._create_file(match.group(1))
                # Try last word
                words = action.split()
                if len(words) > 1:
                    return self._create_file(words[-1])
                self.last_action_result = "Create what? Specify filename"
                return False

            elif "check" in action or "status" in action:
                done, total = self._phase_progress(self.current_phase)
                phase = self.current_phase.value
                self.last_action_result = f"Phase {phase}: {done}/{total}"
                return False

            elif "next" in action or "advance" in action:
                if self._is_phase_complete(self.current_phase):
                    self._advance_phase()
                    phase = self.current_phase.value
                    self.last_action_result = f"Advanced to {phase}"
                    return self.completed
                self.last_action_result = "Current phase not complete yet"
                return False

            else:
                self.last_action_result = f"Unknown: {action[:30]}"
                return False

        def _create_file(self, filename: str) -> bool:
            """Create a file if it's in current phase."""
            # Normalize filename
            filename = filename.strip().strip("'\"")

            if filename not in self.FILE_TEMPLATES:
                self.last_action_result = f"Unknown file: {filename}"
                return False

            # Only allow files from current phase
            allowed = self.PHASE_REQUIREMENTS.get(self.current_phase, [])
            if filename not in allowed:
                phase = self.current_phase.value.upper()
                self.last_action_result = f"Not in {phase} phase"
                return False

            f = self.files[filename]
            if f.exists:
                self.last_action_result = f"{filename} exists"
                return False

            # Create the file
            f.exists = True
            f.content = self.FILE_TEMPLATES[filename]
            self.last_action_result = f"Created {filename}"

            # Check if we should advance phase
            self._advance_phase()

            return self.completed

    # Project setup system prompt
    PROJECT_PLANNING_PROMPT = """\
You are setting up a Python project.

IMPORTANT: Copy the "Output:" line from observation EXACTLY.
Do not add anything else. No code, no explanation.

Example outputs:
- create pyproject.toml
- <plan>Init: create 3 files</plan> create pyproject.toml
"""

    async def main() -> None:
        console = Console()
        console.print()
        console.rule("[bold]DEMO: Dynamic Planning Pattern")
        console.print()
        console.print("[dim]Task: Set up Python project (4 phases)[/dim]")
        console.print("[dim]Agent plans at transitions, acts within[/dim]")
        console.print()

        # Initialize project environment
        project = ProjectSetupEnvironment()

        # Show initial state
        console.print("[bold]Initial State:[/bold]")
        console.print(project.render())
        console.print()

        # Track steps for visualization
        step_log: list[StepOutput] = []

        def on_step(step_num: int, output: StepOutput) -> None:
            step_log.append(output)
            decided = output.decided_to_plan
            decision = "[green]PLAN[/]" if decided else "[dim]act[/dim]"
            tok = f"{output.input_tokens}+{output.output_tokens}"

            # Show what agent decided
            act = output.action
            act_short = act[:35] + "..." if len(act) > 38 else act
            console.print(f"  Step {step_num}: {decision} {act_short}")
            console.print(f"    [dim]({tok} tokens)[/dim]")

            # Show plan if created
            if output.plan:
                p = output.plan
                p_short = p[:50] + "..." if len(p) > 53 else p
                console.print(f"    [cyan]Plan: {p_short}[/cyan]")

            # Show project status
            console.print(project.render())
            console.print()

        async def get_obs() -> str:
            return project.get_observation()

        async def execute(action: str) -> None:
            project.execute(action)

        def check_done() -> bool:
            return project.completed

        # Create agent with project-specific prompt
        project_agent = create_dynamic_agent(
            system_prompt=PROJECT_PLANNING_PROMPT
        )

        console.print("[dim]Running episode with dynamic planning...[/dim]\n")

        result = await run_episode(
            goal="Set up Python project: init, tests, CI, docs",
            get_observation=get_obs,
            execute_action=execute,
            available_actions=["create <filename>", "check"],
            max_steps=15,  # Should complete in ~8 steps
            mode=PlanningMode.DYNAMIC,
            agent=project_agent,
            on_step=on_step,
            is_done=check_done,
        )

        # Update success based on completion
        result.success = project.completed

        render_results(result, step_log, console)

    asyncio.run(main())
# --8<-- [end:main]
