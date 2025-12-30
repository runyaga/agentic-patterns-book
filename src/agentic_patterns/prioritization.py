"""
Prioritization pattern implementation.

This module provides tools for ranking tasks and goals based on established
criteria such as urgency, importance, dependencies, and resource constraints.

Key components:
- PriorityEvaluator: Score tasks based on multiple criteria
- TaskQueue: Priority-based task management with dynamic reordering
- DynamicReprioritizer: Adjust priorities based on context changes
- PrioritizationAgent: LLM-based intelligent task prioritization

Example usage:
    queue = TaskQueue()
    queue.add_task(Task(title="Fix bug", urgency=0.9, importance=0.8))
    queue.add_task(Task(title="Write docs", urgency=0.3, importance=0.5))
    next_task = queue.get_next()
    print(next_task.title)  # "Fix bug"
"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from agentic_patterns import get_model


class PriorityLevel(str, Enum):
    """Priority level classifications."""

    CRITICAL = "P0"  # Must be done immediately
    HIGH = "P1"  # Should be done soon
    MEDIUM = "P2"  # Can wait but is important
    LOW = "P3"  # Nice to have


class TaskStatus(str, Enum):
    """Task status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ReprioritizationReason(str, Enum):
    """Reasons for reprioritization."""

    DEADLINE_APPROACHING = "deadline_approaching"
    DEPENDENCY_RESOLVED = "dependency_resolved"
    RESOURCE_AVAILABLE = "resource_available"
    CONTEXT_CHANGE = "context_change"
    USER_REQUEST = "user_request"
    BLOCKER_REMOVED = "blocker_removed"


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


class Task(BaseModel):
    """A task to be prioritized and executed."""

    id: str = Field(
        default_factory=lambda: f"task-{datetime.now().timestamp()}"
    )
    title: str
    description: str = ""
    urgency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Time sensitivity (0=not urgent, 1=critical)",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Impact on objectives (0=low, 1=critical)",
    )
    effort: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Resource cost (0=trivial, 1=massive)",
    )
    dependencies: list[str] = Field(default_factory=list)
    deadline: datetime | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority_level: PriorityLevel | None = None
    priority_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PriorityScore(BaseModel):
    """Detailed priority score breakdown."""

    task_id: str
    urgency_score: float = Field(ge=0.0, le=1.0)
    importance_score: float = Field(ge=0.0, le=1.0)
    deadline_score: float = Field(ge=0.0, le=1.0)
    dependency_score: float = Field(ge=0.0, le=1.0)
    effort_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    priority_level: PriorityLevel
    reasoning: str = ""


class ReprioritizationEvent(BaseModel):
    """Record of a reprioritization event."""

    task_id: str
    reason: ReprioritizationReason
    old_score: float
    new_score: float
    old_level: PriorityLevel | None
    new_level: PriorityLevel
    timestamp: datetime = Field(default_factory=datetime.now)


class PrioritizationResult(BaseModel):
    """Result from LLM-based prioritization."""

    priority_level: PriorityLevel
    urgency_assessment: float = Field(ge=0.0, le=1.0)
    importance_assessment: float = Field(ge=0.0, le=1.0)
    reasoning: str
    suggested_deadline: str | None = None
    dependencies_identified: list[str] = Field(default_factory=list)


class QueueStats(BaseModel):
    """Statistics about the task queue."""

    total_tasks: int
    pending_tasks: int
    in_progress_tasks: int
    completed_tasks: int
    blocked_tasks: int
    critical_tasks: int
    high_priority_tasks: int
    avg_priority_score: float


# ---------------------------------------------------------------------
# Dataclasses for runtime state
# ---------------------------------------------------------------------


@dataclass
class PriorityCriteria:
    """
    Criteria weights for priority calculation.

    Weights should sum to 1.0 for normalized scores.
    """

    urgency_weight: float = 0.30
    importance_weight: float = 0.30
    deadline_weight: float = 0.20
    dependency_weight: float = 0.10
    effort_weight: float = 0.10  # Lower effort = higher priority


@dataclass
class PriorityEvaluator:
    """
    Evaluate and score tasks based on priority criteria.

    Uses weighted scoring across multiple factors to produce
    a normalized priority score.
    """

    criteria: PriorityCriteria = field(default_factory=PriorityCriteria)

    def evaluate(
        self,
        task: Task,
        resolved_dependencies: set[str] | None = None,
    ) -> PriorityScore:
        """
        Evaluate a task's priority.

        Args:
            task: The task to evaluate.
            resolved_dependencies: Set of completed task IDs.

        Returns:
            PriorityScore with detailed breakdown.
        """
        resolved = resolved_dependencies or set()

        # Calculate component scores
        urgency_score = task.urgency
        importance_score = task.importance

        # Deadline score (higher = more urgent)
        deadline_score = self._calculate_deadline_score(task.deadline)

        # Dependency score (higher = fewer blockers)
        dependency_score = self._calculate_dependency_score(
            task.dependencies,
            resolved,
        )

        # Effort score (invert: lower effort = higher priority)
        effort_score = 1.0 - task.effort

        # Calculate weighted final score
        final_score = (
            urgency_score * self.criteria.urgency_weight
            + importance_score * self.criteria.importance_weight
            + deadline_score * self.criteria.deadline_weight
            + dependency_score * self.criteria.dependency_weight
            + effort_score * self.criteria.effort_weight
        )

        # Determine priority level
        priority_level = self._score_to_level(final_score)

        return PriorityScore(
            task_id=task.id,
            urgency_score=urgency_score,
            importance_score=importance_score,
            deadline_score=deadline_score,
            dependency_score=dependency_score,
            effort_score=effort_score,
            final_score=final_score,
            priority_level=priority_level,
            reasoning=self._generate_reasoning(task, final_score),
        )

    def _calculate_deadline_score(
        self,
        deadline: datetime | None,
    ) -> float:
        """Calculate score based on deadline proximity."""
        if deadline is None:
            return 0.5  # Neutral score for no deadline

        now = datetime.now()
        if deadline <= now:
            return 1.0  # Overdue

        time_left = (deadline - now).total_seconds()
        hours_left = time_left / 3600

        if hours_left <= 1:
            return 0.95
        elif hours_left <= 4:
            return 0.85
        elif hours_left <= 24:
            return 0.7
        elif hours_left <= 72:
            return 0.5
        elif hours_left <= 168:  # 1 week
            return 0.3
        else:
            return 0.2

    def _calculate_dependency_score(
        self,
        dependencies: list[str],
        resolved: set[str],
    ) -> float:
        """Calculate score based on resolved dependencies."""
        if not dependencies:
            return 1.0  # No dependencies = fully available

        resolved_count = sum(1 for d in dependencies if d in resolved)
        return resolved_count / len(dependencies)

    def _score_to_level(self, score: float) -> PriorityLevel:
        """Convert numeric score to priority level."""
        if score >= 0.8:
            return PriorityLevel.CRITICAL
        elif score >= 0.6:
            return PriorityLevel.HIGH
        elif score >= 0.4:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW

    def _generate_reasoning(self, task: Task, score: float) -> str:
        """Generate human-readable reasoning for priority."""
        parts = []
        if task.urgency >= 0.8:
            parts.append("highly urgent")
        if task.importance >= 0.8:
            parts.append("critical importance")
        deadline_threshold = datetime.now() + timedelta(hours=24)
        if task.deadline and task.deadline <= deadline_threshold:
            parts.append("deadline approaching")
        if task.effort <= 0.2:
            parts.append("quick win")

        if not parts:
            parts.append("standard priority")

        return f"Score {score:.2f}: {', '.join(parts)}"


@dataclass
class TaskQueue:
    """
    Priority-based task queue with automatic ordering.

    Tasks are automatically sorted by priority score when accessed.
    """

    tasks: dict[str, Task] = field(default_factory=dict)
    evaluator: PriorityEvaluator = field(default_factory=PriorityEvaluator)
    completed_tasks: set[str] = field(default_factory=set)

    def add_task(self, task: Task) -> PriorityScore:
        """Add a task and calculate its priority."""
        score = self.evaluator.evaluate(task, self.completed_tasks)
        task.priority_score = score.final_score
        task.priority_level = score.priority_level
        self.tasks[task.id] = task
        return score

    def get_next(self) -> Task | None:
        """Get the highest priority pending task."""
        pending = [
            t for t in self.tasks.values() if t.status == TaskStatus.PENDING
        ]
        if not pending:
            return None

        # Sort by priority score descending
        pending.sort(key=lambda t: t.priority_score, reverse=True)
        return pending[0]

    def get_by_level(self, level: PriorityLevel) -> list[Task]:
        """Get all tasks at a specific priority level."""
        return [t for t in self.tasks.values() if t.priority_level == level]

    def get_blocked(self) -> list[Task]:
        """Get tasks blocked by unresolved dependencies."""
        blocked = []
        for task in self.tasks.values():
            if task.status == TaskStatus.BLOCKED:
                blocked.append(task)
            elif task.dependencies:
                unresolved = [
                    d
                    for d in task.dependencies
                    if d not in self.completed_tasks
                ]
                if unresolved:
                    blocked.append(task)
        return blocked

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed and reprioritize dependents."""
        if task_id not in self.tasks:
            return False

        self.tasks[task_id].status = TaskStatus.COMPLETED
        self.completed_tasks.add(task_id)

        # Reprioritize tasks that depended on this one
        for task in self.tasks.values():
            if task_id in task.dependencies:
                score = self.evaluator.evaluate(task, self.completed_tasks)
                task.priority_score = score.final_score
                task.priority_level = score.priority_level
                if task.status == TaskStatus.BLOCKED:
                    # Check if all dependencies now resolved
                    unresolved = [
                        d
                        for d in task.dependencies
                        if d not in self.completed_tasks
                    ]
                    if not unresolved:
                        task.status = TaskStatus.PENDING

        return True

    def update_task(
        self,
        task_id: str,
        **updates: Any,
    ) -> PriorityScore | None:
        """Update a task and recalculate priority."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        score = self.evaluator.evaluate(task, self.completed_tasks)
        task.priority_score = score.final_score
        task.priority_level = score.priority_level
        return score

    def get_ordered_list(self) -> list[Task]:
        """Get all pending tasks in priority order."""
        pending = [
            t for t in self.tasks.values() if t.status == TaskStatus.PENDING
        ]
        pending.sort(key=lambda t: t.priority_score, reverse=True)
        return pending

    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        tasks = list(self.tasks.values())
        scores = [t.priority_score for t in tasks if t.priority_score > 0]

        return QueueStats(
            total_tasks=len(tasks),
            pending_tasks=len(
                [t for t in tasks if t.status == TaskStatus.PENDING]
            ),
            in_progress_tasks=len(
                [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
            ),
            completed_tasks=len(
                [t for t in tasks if t.status == TaskStatus.COMPLETED]
            ),
            blocked_tasks=len(
                [t for t in tasks if t.status == TaskStatus.BLOCKED]
            ),
            critical_tasks=len(
                [
                    t
                    for t in tasks
                    if t.priority_level == PriorityLevel.CRITICAL
                ]
            ),
            high_priority_tasks=len(
                [t for t in tasks if t.priority_level == PriorityLevel.HIGH]
            ),
            avg_priority_score=sum(scores) / len(scores) if scores else 0.0,
        )


@dataclass
class DynamicReprioritizer:
    """
    Dynamically adjust priorities based on context changes.

    Monitors for events that should trigger reprioritization.
    """

    queue: TaskQueue
    history: list[ReprioritizationEvent] = field(default_factory=list)
    deadline_threshold_hours: float = 24.0  # Alert when deadline this close

    def check_deadlines(self) -> list[ReprioritizationEvent]:
        """Check for approaching deadlines and reprioritize."""
        events = []
        now = datetime.now()
        threshold = now + timedelta(hours=self.deadline_threshold_hours)

        for task in self.queue.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if task.deadline is None:
                continue
            if task.deadline > threshold:
                continue

            # Deadline is approaching - boost priority
            old_score = task.priority_score
            old_level = task.priority_level

            # Recalculate with urgency boost
            original_urgency = task.urgency
            task.urgency = min(1.0, task.urgency + 0.2)

            score = self.queue.evaluator.evaluate(
                task,
                self.queue.completed_tasks,
            )
            task.priority_score = score.final_score
            task.priority_level = score.priority_level
            task.urgency = original_urgency  # Restore

            if score.final_score > old_score:
                event = ReprioritizationEvent(
                    task_id=task.id,
                    reason=ReprioritizationReason.DEADLINE_APPROACHING,
                    old_score=old_score,
                    new_score=score.final_score,
                    old_level=old_level,
                    new_level=score.priority_level,
                )
                events.append(event)
                self.history.append(event)

        return events

    def on_dependency_resolved(
        self,
        dependency_id: str,
    ) -> list[ReprioritizationEvent]:
        """Handle a dependency being resolved."""
        events = []

        for task in self.queue.tasks.values():
            if dependency_id not in task.dependencies:
                continue

            old_score = task.priority_score
            old_level = task.priority_level

            score = self.queue.evaluator.evaluate(
                task,
                self.queue.completed_tasks,
            )
            task.priority_score = score.final_score
            task.priority_level = score.priority_level

            if score.final_score != old_score:
                event = ReprioritizationEvent(
                    task_id=task.id,
                    reason=ReprioritizationReason.DEPENDENCY_RESOLVED,
                    old_score=old_score,
                    new_score=score.final_score,
                    old_level=old_level,
                    new_level=score.priority_level,
                )
                events.append(event)
                self.history.append(event)

        return events

    def manual_boost(
        self,
        task_id: str,
        reason: ReprioritizationReason = ReprioritizationReason.USER_REQUEST,
    ) -> ReprioritizationEvent | None:
        """Manually boost a task's priority."""
        if task_id not in self.queue.tasks:
            return None

        task = self.queue.tasks[task_id]
        old_score = task.priority_score
        old_level = task.priority_level

        # Boost by increasing urgency and importance slightly
        task.urgency = min(1.0, task.urgency + 0.15)
        task.importance = min(1.0, task.importance + 0.15)

        score = self.queue.evaluator.evaluate(
            task,
            self.queue.completed_tasks,
        )
        task.priority_score = score.final_score
        task.priority_level = score.priority_level

        event = ReprioritizationEvent(
            task_id=task_id,
            reason=reason,
            old_score=old_score,
            new_score=score.final_score,
            old_level=old_level,
            new_level=score.priority_level,
        )
        self.history.append(event)
        return event

    def get_history(
        self,
        task_id: str | None = None,
    ) -> list[ReprioritizationEvent]:
        """Get reprioritization history, optionally filtered by task."""
        if task_id is None:
            return list(self.history)
        return [e for e in self.history if e.task_id == task_id]


PRIORITIZATION_SYSTEM_PROMPT = """You are a task prioritization expert.
Analyze the given task and determine its priority level.

Consider:
- Urgency: How time-sensitive is this?
- Importance: What's the impact on goals?
- Dependencies: What blocks or enables this?
- Effort: How much work is required?

Priority levels:
- P0 (CRITICAL): Must do immediately, blocking issue
- P1 (HIGH): Should do soon, significant impact
- P2 (MEDIUM): Important but can wait
- P3 (LOW): Nice to have, low impact"""


def create_prioritization_agent(
    model: Model | None = None,
) -> Agent[None, PrioritizationResult]:
    """
    Create a prioritization agent with optional model override.

    Args:
        model: pydantic-ai Model instance. If None, uses default model.

    Returns:
        Configured prioritization agent.
    """
    return Agent(
        model or get_model(),
        output_type=PrioritizationResult,
        system_prompt=PRIORITIZATION_SYSTEM_PROMPT,
    )


# Default agent (created lazily for backward compatibility)
_default_prioritization_agent: Agent[None, PrioritizationResult] | None = None


def _get_default_prioritization_agent() -> Agent[None, PrioritizationResult]:
    """Get or create the default prioritization agent."""
    global _default_prioritization_agent
    if _default_prioritization_agent is None:
        _default_prioritization_agent = create_prioritization_agent()
    return _default_prioritization_agent


@dataclass
class PrioritizationAgent:
    """
    LLM-based intelligent task prioritization.

    Uses an LLM to assess task priority with contextual understanding.
    """

    agent: Agent[None, PrioritizationResult] | None = None
    # Backward compatibility attributes (deprecated - use agent parameter)
    model_name: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1"

    def _get_agent(self) -> Agent[None, PrioritizationResult]:
        """Get the configured agent (backward compat method)."""
        if self.agent:
            return self.agent
        return _get_default_prioritization_agent()

    async def prioritize(
        self,
        title: str,
        description: str,
        context: str | None = None,
    ) -> PrioritizationResult:
        """
        Use LLM to prioritize a task.

        Args:
            title: Task title.
            description: Task description.
            context: Optional context about current priorities.

        Returns:
            PrioritizationResult with assessment.
        """
        agent = self._get_agent()
        prompt = f"Task: {title}\n\nDescription: {description}"
        if context:
            prompt += f"\n\nContext: {context}"

        result = await agent.run(prompt)
        return result.output


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def create_task_from_description(
    title: str,
    description: str = "",
    deadline_hours: float | None = None,
    depends_on: list[str] | None = None,
) -> Task:
    """
    Create a task with reasonable defaults.

    Args:
        title: Task title.
        description: Optional description.
        deadline_hours: Hours until deadline (None for no deadline).
        depends_on: List of task IDs this depends on.

    Returns:
        Task with populated fields.
    """
    deadline = None
    if deadline_hours is not None:
        deadline = datetime.now() + timedelta(hours=deadline_hours)

    return Task(
        title=title,
        description=description,
        deadline=deadline,
        dependencies=depends_on or [],
    )


def eisenhower_classify(
    urgency: float,
    importance: float,
) -> str:
    """
    Classify task using Eisenhower matrix.

    Args:
        urgency: 0-1 urgency score.
        importance: 0-1 importance score.

    Returns:
        Quadrant classification.
    """
    if urgency >= 0.5 and importance >= 0.5:
        return "DO: Urgent & Important"
    elif urgency < 0.5 and importance >= 0.5:
        return "SCHEDULE: Not Urgent & Important"
    elif urgency >= 0.5 and importance < 0.5:
        return "DELEGATE: Urgent & Not Important"
    else:
        return "ELIMINATE: Not Urgent & Not Important"


def batch_prioritize(
    tasks: list[Task],
    evaluator: PriorityEvaluator | None = None,
    resolved: set[str] | None = None,
) -> list[tuple[Task, PriorityScore]]:
    """
    Prioritize multiple tasks and return sorted list.

    Args:
        tasks: List of tasks to prioritize.
        evaluator: Optional custom evaluator.
        resolved: Set of resolved dependency IDs.

    Returns:
        List of (task, score) tuples sorted by priority.
    """
    eval_instance = evaluator or PriorityEvaluator()
    resolved_deps = resolved or set()

    scored = []
    for task in tasks:
        score = eval_instance.evaluate(task, resolved_deps)
        task.priority_score = score.final_score
        task.priority_level = score.priority_level
        scored.append((task, score))

    # Sort by final score descending
    scored.sort(key=lambda x: x[1].final_score, reverse=True)
    return scored


def get_actionable_tasks(
    queue: TaskQueue,
    max_count: int = 5,
) -> list[Task]:
    """
    Get top actionable tasks (not blocked).

    Args:
        queue: Task queue to search.
        max_count: Maximum tasks to return.

    Returns:
        List of actionable tasks in priority order.
    """
    actionable = []
    for task in queue.get_ordered_list():
        if task.status != TaskStatus.PENDING:
            continue

        # Check if blocked by dependencies
        unresolved = [
            d for d in task.dependencies if d not in queue.completed_tasks
        ]
        if unresolved:
            continue

        actionable.append(task)
        if len(actionable) >= max_count:
            break

    return actionable


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------


if __name__ == "__main__":
    import asyncio

    async def demo() -> None:
        """Demonstrate prioritization capabilities."""
        print("=" * 60)
        print("Prioritization Pattern Demo")
        print("=" * 60)

        # 1. Create task queue
        print("\n--- Task Queue Setup ---")
        queue = TaskQueue()

        # Add various tasks
        task1 = Task(
            id="task-1",
            title="Fix production bug",
            description="Critical system down",
            urgency=0.95,
            importance=0.95,
            effort=0.3,
        )
        task2 = Task(
            id="task-2",
            title="Write documentation",
            description="Update API docs",
            urgency=0.2,
            importance=0.6,
            effort=0.4,
        )
        task3 = Task(
            id="task-3",
            title="Code review",
            description="Review pending PR",
            urgency=0.5,
            importance=0.7,
            effort=0.2,
            dependencies=["task-1"],
        )
        task4 = Task(
            id="task-4",
            title="Team meeting prep",
            description="Prepare slides",
            urgency=0.8,
            importance=0.5,
            effort=0.3,
            deadline=datetime.now() + timedelta(hours=2),
        )

        for task in [task1, task2, task3, task4]:
            score = queue.add_task(task)
            print(f"Added: {task.title}")
            print(f"  Priority: {score.priority_level.value}")
            print(f"  Score: {score.final_score:.2f}")

        # 2. Get prioritized list
        print("\n--- Priority Order ---")
        for i, task in enumerate(queue.get_ordered_list(), 1):
            level = task.priority_level.value if task.priority_level else "?"
            score_str = f"{task.priority_score:.2f}"
            print(f"{i}. [{level}] {task.title} (score: {score_str})")

        # 3. Eisenhower matrix classification
        print("\n--- Eisenhower Classification ---")
        for task in queue.tasks.values():
            classification = eisenhower_classify(task.urgency, task.importance)
            print(f"{task.title}: {classification}")

        # 4. Dynamic reprioritization
        print("\n--- Dynamic Reprioritization ---")
        reprioritizer = DynamicReprioritizer(queue)

        # Check deadlines
        deadline_events = reprioritizer.check_deadlines()
        print(f"Deadline alerts: {len(deadline_events)}")
        for event in deadline_events:
            task = queue.tasks[event.task_id]
            print(f"  {task.title}: {event.old_level} -> {event.new_level}")

        # Complete a task and see reprioritization
        print("\n--- Completing Task ---")
        queue.complete_task("task-1")
        print("Completed: Fix production bug")

        dep_events = reprioritizer.on_dependency_resolved("task-1")
        print(f"Dependency updates: {len(dep_events)}")
        for event in dep_events:
            task = queue.tasks[event.task_id]
            print(
                f"  {task.title}: score {event.old_score:.2f} -> "
                f"{event.new_score:.2f}"
            )

        # 5. Get actionable tasks
        print("\n--- Actionable Tasks ---")
        actionable = get_actionable_tasks(queue, max_count=3)
        for task in actionable:
            print(f"- {task.title} (score: {task.priority_score:.2f})")

        # 6. Queue statistics
        print("\n--- Queue Stats ---")
        stats = queue.get_stats()
        print(f"Total: {stats.total_tasks}")
        print(f"Pending: {stats.pending_tasks}")
        print(f"Completed: {stats.completed_tasks}")
        print(f"Critical: {stats.critical_tasks}")
        print(f"Avg Score: {stats.avg_priority_score:.2f}")

        # 7. Batch prioritization
        print("\n--- Batch Prioritization ---")
        new_tasks = [
            Task(title="Quick fix", urgency=0.6, importance=0.4),
            Task(title="Big refactor", urgency=0.3, importance=0.8),
            Task(title="Customer request", urgency=0.9, importance=0.9),
        ]
        sorted_tasks = batch_prioritize(new_tasks)
        for task, score in sorted_tasks:
            print(
                f"{task.title}: {score.priority_level.value} "
                f"({score.final_score:.2f})"
            )

        print("\n" + "=" * 60)
        print("Demo complete!")

    asyncio.run(demo())
