"""
Goal Monitoring Pattern (Teleological Engine).

Based on the Agentic Design Patterns book Chapter 11:
Proactive goal monitoring with automatic remediation.

Key concepts:
- Goals: measurable targets with async evaluators
- Monitor loop: Wait → Check → Remediate cycle
- Escalation stub: production-ready integration point

V1 is intentionally lean. See spec for production TODOs (P1-P5).
"""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode
from pydantic_graph import End
from pydantic_graph import Graph
from pydantic_graph import GraphRunContext

from agentic_patterns._models import get_model


# --8<-- [start:models]
@dataclass
class Goal:
    """
    A single monitorable goal.

    Args:
        name: Human-readable goal identifier.
        target: Target value to achieve.
        evaluator: Async function returning current value.
        comparator: How to compare current vs target.
        remediation_hint: Hint for remediation agent.
    """

    name: str
    target: float
    evaluator: Callable[[], Awaitable[float]]
    comparator: Literal[">=", "<=", "==", ">", "<"] = ">="
    remediation_hint: str = ""


@dataclass
class GoalStatus:
    """Result of checking a goal."""

    goal_name: str
    current_value: float
    target_value: float
    is_met: bool
    checked_at: datetime


@dataclass
class MonitorState:
    """Graph state for goal monitoring."""

    goals: list[Goal]
    check_interval: float = 60.0
    shutdown: bool = False
    current_gap: Goal | None = None
    last_status: list[GoalStatus] = field(default_factory=list)


# --8<-- [end:models]


# --8<-- [start:remediation]
class RemediationResult(BaseModel):
    """Result of a remediation attempt."""

    success: bool
    action_taken: str = Field(description="What was done to fix the gap")
    error: str | None = None


model = get_model()

remediation_agent = Agent(
    model,
    system_prompt=(
        "You are a maintenance agent. When a goal is not met, "
        "describe what action should be taken to fix it. "
        "Be specific about the remediation steps."
    ),
    output_type=RemediationResult,
)


async def on_escalate(goal: Goal, error: str) -> None:
    """
    Stub for escalation handling.

    Production TODO (P1): Integrate with alerting system
    (Slack, PagerDuty, email, etc.)

    Args:
        goal: The goal that failed remediation.
        error: Error message from remediation attempt.
    """
    print(f"ESCALATE: Goal '{goal.name}' - {error}")


# --8<-- [end:remediation]


# --8<-- [start:nodes]
@dataclass
class WaitNode(BaseNode[MonitorState, None, None]):
    """Wait for next check interval."""

    async def run(
        self,
        ctx: GraphRunContext[MonitorState],
    ) -> CheckNode | End[None]:
        """Wait for interval or exit on shutdown."""
        if ctx.state.shutdown:
            return End(None)

        await asyncio.sleep(ctx.state.check_interval)
        return CheckNode()


@dataclass
class CheckNode(BaseNode[MonitorState, None, None]):
    """Evaluate all goals."""

    async def run(
        self,
        ctx: GraphRunContext[MonitorState],
    ) -> WaitNode | RemediateNode:
        """Check goals and transition based on results."""
        state = ctx.state
        state.last_status = []

        for goal in state.goals:
            try:
                current = await goal.evaluator()
            except Exception as e:
                # Evaluator failed - treat as gap
                print(f"Evaluator error for '{goal.name}': {e}")
                current = (
                    float("-inf")
                    if goal.comparator in (">=", ">")
                    else float("inf")
                )

            is_met = self._check_met(current, goal.target, goal.comparator)

            state.last_status.append(
                GoalStatus(
                    goal_name=goal.name,
                    current_value=current,
                    target_value=goal.target,
                    is_met=is_met,
                    checked_at=datetime.now(),
                )
            )

            if not is_met:
                state.current_gap = goal
                return RemediateNode()

        return WaitNode()

    def _check_met(
        self,
        current: float,
        target: float,
        comparator: str,
    ) -> bool:
        """Compare current value against target."""
        ops: dict[str, Callable[[float, float], bool]] = {
            ">=": lambda c, t: c >= t,
            "<=": lambda c, t: c <= t,
            "==": lambda c, t: c == t,
            ">": lambda c, t: c > t,
            "<": lambda c, t: c < t,
        }
        return ops[comparator](current, target)


@dataclass
class RemediateNode(BaseNode[MonitorState, None, None]):
    """Attempt to fix a detected gap."""

    async def run(
        self,
        ctx: GraphRunContext[MonitorState],
    ) -> CheckNode:
        """Run remediation agent and return to check."""
        goal = ctx.state.current_gap
        if goal is None:
            return CheckNode()

        try:
            result = await remediation_agent.run(
                f"Goal '{goal.name}' is not met.\n"
                f"Hint: {goal.remediation_hint}\n"
                f"Take action to fix this."
            )

            if not result.output.success:
                await on_escalate(
                    goal, result.output.error or "Remediation failed"
                )
        except Exception as e:
            await on_escalate(goal, str(e))

        ctx.state.current_gap = None
        return CheckNode()


# Graph definition
goal_monitor_graph: Graph[MonitorState, None, None] = Graph(
    nodes=[WaitNode, CheckNode, RemediateNode],
)
# --8<-- [end:nodes]


# --8<-- [start:monitor]
class GoalMonitor:
    """
    Manages goal monitoring lifecycle.

    Example:
        monitor = GoalMonitor(goals=[...], check_interval=60.0)
        await monitor.start()
        # ... later
        await monitor.stop()
    """

    def __init__(
        self,
        goals: list[Goal],
        check_interval: float = 60.0,
    ) -> None:
        """
        Initialize goal monitor.

        Args:
            goals: Goals to monitor.
            check_interval: Seconds between checks.
        """
        self.goals = goals
        self.check_interval = check_interval
        self._task: asyncio.Task[None] | None = None
        self._state: MonitorState | None = None

    async def start(self) -> None:
        """Start the monitoring loop."""
        self._state = MonitorState(
            goals=self.goals,
            check_interval=self.check_interval,
        )
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the monitoring loop gracefully."""
        if self._state:
            self._state.shutdown = True
        if self._task:
            await self._task

    async def _run(self) -> None:
        """Run the graph."""
        if self._state:
            await goal_monitor_graph.run(WaitNode(), state=self._state)

    def get_status(self) -> list[GoalStatus]:
        """Get last check results."""
        return self._state.last_status if self._state else []


async def run_goal_monitor(
    goals: list[Goal],
    check_interval: float = 60.0,
) -> None:
    """
    Run goal monitoring until interrupted.

    Args:
        goals: Goals to monitor.
        check_interval: Seconds between checks.
    """
    monitor = GoalMonitor(goals, check_interval)
    await monitor.start()
    try:
        while monitor._state and not monitor._state.shutdown:
            await asyncio.sleep(1)
    finally:
        await monitor.stop()


# --8<-- [end:monitor]


# --8<-- [start:main]
if __name__ == "__main__":

    async def demo_evaluator() -> float:
        """Demo evaluator that returns random value."""
        import random

        return random.uniform(0, 100)

    async def main() -> None:
        """Demo goal monitoring."""
        goals = [
            Goal(
                name="demo_metric",
                target=50.0,
                comparator=">=",
                evaluator=demo_evaluator,
                remediation_hint="Increase the demo metric value",
            ),
        ]

        print("Starting goal monitor (will run 3 checks)...")
        monitor = GoalMonitor(goals, check_interval=2.0)
        await monitor.start()

        # Run for a few cycles
        for i in range(3):
            await asyncio.sleep(2.5)
            status = monitor.get_status()
            if status:
                for s in status:
                    met = "MET" if s.is_met else "NOT MET"
                    print(
                        f"  Check {i + 1}: {s.goal_name} = "
                        f"{s.current_value:.1f} (target {s.target_value}) "
                        f"[{met}]"
                    )

        await monitor.stop()
        print("Monitor stopped.")

    asyncio.run(main())
# --8<-- [end:main]
