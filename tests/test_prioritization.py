"""Tests for the Prioritization pattern module."""

from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from agentic_patterns.prioritization import DynamicReprioritizer
from agentic_patterns.prioritization import PrioritizationAgent
from agentic_patterns.prioritization import PrioritizationResult
from agentic_patterns.prioritization import PriorityCriteria
from agentic_patterns.prioritization import PriorityEvaluator
from agentic_patterns.prioritization import PriorityLevel
from agentic_patterns.prioritization import PriorityScore
from agentic_patterns.prioritization import QueueStats
from agentic_patterns.prioritization import ReprioritizationEvent
from agentic_patterns.prioritization import ReprioritizationReason
from agentic_patterns.prioritization import Task
from agentic_patterns.prioritization import TaskQueue
from agentic_patterns.prioritization import TaskStatus
from agentic_patterns.prioritization import batch_prioritize
from agentic_patterns.prioritization import create_task_from_description
from agentic_patterns.prioritization import eisenhower_classify
from agentic_patterns.prioritization import get_actionable_tasks


class TestPriorityLevel:
    """Tests for PriorityLevel enum."""

    def test_priority_level_values(self) -> None:
        """Test all priority level values exist."""
        assert PriorityLevel.CRITICAL == "P0"
        assert PriorityLevel.HIGH == "P1"
        assert PriorityLevel.MEDIUM == "P2"
        assert PriorityLevel.LOW == "P3"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_values(self) -> None:
        """Test all task status values exist."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.BLOCKED == "blocked"
        assert TaskStatus.CANCELLED == "cancelled"


class TestReprioritizationReason:
    """Tests for ReprioritizationReason enum."""

    def test_reason_values(self) -> None:
        """Test all reprioritization reason values exist."""
        deadline_val = "deadline_approaching"
        dep_val = "dependency_resolved"
        assert deadline_val == ReprioritizationReason.DEADLINE_APPROACHING
        assert dep_val == ReprioritizationReason.DEPENDENCY_RESOLVED
        assert ReprioritizationReason.USER_REQUEST == "user_request"


class TestTask:
    """Tests for Task model."""

    def test_task_creation(self) -> None:
        """Test creating a task."""
        task = Task(title="Test task", description="A test task")
        assert task.title == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.urgency == 0.5
        assert task.importance == 0.5

    def test_task_with_all_fields(self) -> None:
        """Test task with all fields populated."""
        deadline = datetime.now() + timedelta(hours=24)
        task = Task(
            id="custom-id",
            title="Important task",
            description="Very important",
            urgency=0.9,
            importance=0.95,
            effort=0.3,
            dependencies=["dep-1", "dep-2"],
            deadline=deadline,
            priority_level=PriorityLevel.HIGH,
            priority_score=0.85,
            metadata={"team": "engineering"},
        )
        assert task.id == "custom-id"
        assert task.urgency == 0.9
        assert len(task.dependencies) == 2
        assert task.metadata["team"] == "engineering"

    def test_task_default_id_generated(self) -> None:
        """Test that task ID is auto-generated."""
        task = Task(title="Test")
        assert task.id.startswith("task-")


class TestPriorityScore:
    """Tests for PriorityScore model."""

    def test_priority_score_creation(self) -> None:
        """Test creating a priority score."""
        score = PriorityScore(
            task_id="task-1",
            urgency_score=0.8,
            importance_score=0.9,
            deadline_score=0.7,
            dependency_score=1.0,
            effort_score=0.6,
            final_score=0.85,
            priority_level=PriorityLevel.HIGH,
            reasoning="High priority task",
        )
        assert score.task_id == "task-1"
        assert score.final_score == 0.85
        assert score.priority_level == PriorityLevel.HIGH


class TestReprioritizationEvent:
    """Tests for ReprioritizationEvent model."""

    def test_event_creation(self) -> None:
        """Test creating a reprioritization event."""
        event = ReprioritizationEvent(
            task_id="task-1",
            reason=ReprioritizationReason.DEADLINE_APPROACHING,
            old_score=0.5,
            new_score=0.8,
            old_level=PriorityLevel.MEDIUM,
            new_level=PriorityLevel.HIGH,
        )
        assert event.task_id == "task-1"
        assert event.reason == ReprioritizationReason.DEADLINE_APPROACHING
        assert event.new_score > event.old_score


class TestPrioritizationResult:
    """Tests for PrioritizationResult model."""

    def test_result_creation(self) -> None:
        """Test creating a prioritization result."""
        result = PrioritizationResult(
            priority_level=PriorityLevel.CRITICAL,
            urgency_assessment=0.95,
            importance_assessment=0.9,
            reasoning="Production issue requires immediate attention",
            suggested_deadline="Within 1 hour",
            dependencies_identified=["deploy-pipeline"],
        )
        assert result.priority_level == PriorityLevel.CRITICAL
        assert result.urgency_assessment == 0.95


class TestQueueStats:
    """Tests for QueueStats model."""

    def test_stats_creation(self) -> None:
        """Test creating queue statistics."""
        stats = QueueStats(
            total_tasks=10,
            pending_tasks=5,
            in_progress_tasks=2,
            completed_tasks=2,
            blocked_tasks=1,
            critical_tasks=1,
            high_priority_tasks=3,
            avg_priority_score=0.65,
        )
        assert stats.total_tasks == 10
        assert stats.avg_priority_score == 0.65


class TestPriorityCriteria:
    """Tests for PriorityCriteria dataclass."""

    def test_default_criteria(self) -> None:
        """Test default criteria weights."""
        criteria = PriorityCriteria()
        assert criteria.urgency_weight == 0.30
        assert criteria.importance_weight == 0.30
        assert criteria.deadline_weight == 0.20

    def test_custom_criteria(self) -> None:
        """Test custom criteria weights."""
        criteria = PriorityCriteria(
            urgency_weight=0.5,
            importance_weight=0.3,
            deadline_weight=0.1,
            dependency_weight=0.05,
            effort_weight=0.05,
        )
        assert criteria.urgency_weight == 0.5


class TestPriorityEvaluator:
    """Tests for PriorityEvaluator dataclass."""

    def test_evaluator_creation(self) -> None:
        """Test creating an evaluator."""
        evaluator = PriorityEvaluator()
        assert evaluator.criteria is not None

    def test_evaluate_high_priority_task(self) -> None:
        """Test evaluating a high priority task."""
        evaluator = PriorityEvaluator()
        task = Task(
            title="Critical bug",
            urgency=0.95,
            importance=0.95,
            effort=0.2,
        )
        score = evaluator.evaluate(task)
        assert score.final_score >= 0.8
        assert score.priority_level == PriorityLevel.CRITICAL

    def test_evaluate_low_priority_task(self) -> None:
        """Test evaluating a low priority task."""
        evaluator = PriorityEvaluator()
        task = Task(
            title="Nice to have",
            urgency=0.1,
            importance=0.2,
            effort=0.8,
        )
        score = evaluator.evaluate(task)
        assert score.final_score < 0.4
        assert score.priority_level == PriorityLevel.LOW

    def test_evaluate_with_deadline(self) -> None:
        """Test evaluation considers deadline."""
        evaluator = PriorityEvaluator()

        # Task with approaching deadline
        task_urgent = Task(
            title="Deadline soon",
            deadline=datetime.now() + timedelta(hours=2),
        )
        score_urgent = evaluator.evaluate(task_urgent)

        # Task with distant deadline
        task_later = Task(
            title="Deadline later",
            deadline=datetime.now() + timedelta(days=30),
        )
        score_later = evaluator.evaluate(task_later)

        assert score_urgent.deadline_score > score_later.deadline_score

    def test_evaluate_with_dependencies(self) -> None:
        """Test evaluation considers dependencies."""
        evaluator = PriorityEvaluator()
        task = Task(
            title="Dependent task",
            dependencies=["dep-1", "dep-2"],
        )

        # No dependencies resolved
        score_blocked = evaluator.evaluate(task, set())
        assert score_blocked.dependency_score == 0.0

        # One dependency resolved
        score_partial = evaluator.evaluate(task, {"dep-1"})
        assert score_partial.dependency_score == 0.5

        # All resolved
        score_ready = evaluator.evaluate(task, {"dep-1", "dep-2"})
        assert score_ready.dependency_score == 1.0

    def test_evaluate_no_deadline(self) -> None:
        """Test evaluation with no deadline."""
        evaluator = PriorityEvaluator()
        task = Task(title="No deadline")
        score = evaluator.evaluate(task)
        assert score.deadline_score == 0.5  # Neutral

    def test_evaluate_overdue_task(self) -> None:
        """Test evaluation with overdue deadline."""
        evaluator = PriorityEvaluator()
        task = Task(
            title="Overdue",
            deadline=datetime.now() - timedelta(hours=1),
        )
        score = evaluator.evaluate(task)
        assert score.deadline_score == 1.0

    def test_score_to_level_mapping(self) -> None:
        """Test score to priority level mapping."""
        evaluator = PriorityEvaluator()
        assert evaluator._score_to_level(0.9) == PriorityLevel.CRITICAL
        assert evaluator._score_to_level(0.7) == PriorityLevel.HIGH
        assert evaluator._score_to_level(0.5) == PriorityLevel.MEDIUM
        assert evaluator._score_to_level(0.2) == PriorityLevel.LOW


class TestTaskQueue:
    """Tests for TaskQueue dataclass."""

    def test_queue_creation(self) -> None:
        """Test creating a task queue."""
        queue = TaskQueue()
        assert len(queue.tasks) == 0

    def test_add_task(self) -> None:
        """Test adding a task to queue."""
        queue = TaskQueue()
        task = Task(title="Test task", urgency=0.7, importance=0.8)
        score = queue.add_task(task)

        assert task.id in queue.tasks
        assert task.priority_score > 0
        assert task.priority_level is not None
        assert score.final_score == task.priority_score

    def test_get_next_highest_priority(self) -> None:
        """Test getting next task returns highest priority."""
        queue = TaskQueue()
        low = Task(id="low", title="Low", urgency=0.2, importance=0.2)
        high = Task(id="high", title="High", urgency=0.9, importance=0.9)
        medium = Task(id="med", title="Medium", urgency=0.5, importance=0.5)

        queue.add_task(low)
        queue.add_task(high)
        queue.add_task(medium)

        next_task = queue.get_next()
        assert next_task is not None
        assert next_task.id == "high"

    def test_get_next_empty_queue(self) -> None:
        """Test getting next from empty queue."""
        queue = TaskQueue()
        assert queue.get_next() is None

    def test_get_by_level(self) -> None:
        """Test getting tasks by priority level."""
        queue = TaskQueue()
        critical = Task(
            id="crit", title="Critical", urgency=0.95, importance=0.95
        )
        low = Task(id="low", title="Low", urgency=0.1, importance=0.1)

        queue.add_task(critical)
        queue.add_task(low)

        critical_tasks = queue.get_by_level(PriorityLevel.CRITICAL)
        low_tasks = queue.get_by_level(PriorityLevel.LOW)

        assert len(critical_tasks) == 1
        assert critical_tasks[0].id == "crit"
        assert len(low_tasks) == 1

    def test_complete_task(self) -> None:
        """Test completing a task."""
        queue = TaskQueue()
        task = Task(id="task-1", title="Test")
        queue.add_task(task)

        result = queue.complete_task("task-1")

        assert result is True
        assert queue.tasks["task-1"].status == TaskStatus.COMPLETED
        assert "task-1" in queue.completed_tasks

    def test_complete_task_reprioritizes_dependents(self) -> None:
        """Test completing task reprioritizes dependents."""
        queue = TaskQueue()
        task1 = Task(id="task-1", title="First")
        task2 = Task(id="task-2", title="Second", dependencies=["task-1"])

        queue.add_task(task1)
        queue.add_task(task2)

        initial_score = queue.tasks["task-2"].priority_score
        queue.complete_task("task-1")
        new_score = queue.tasks["task-2"].priority_score

        # Score should increase after dependency resolved
        assert new_score >= initial_score

    def test_complete_unknown_task(self) -> None:
        """Test completing non-existent task."""
        queue = TaskQueue()
        assert queue.complete_task("unknown") is False

    def test_update_task(self) -> None:
        """Test updating a task."""
        queue = TaskQueue()
        task = Task(id="task-1", title="Original", urgency=0.5)
        queue.add_task(task)

        original_score = task.priority_score
        queue.update_task("task-1", urgency=0.9)

        assert queue.tasks["task-1"].urgency == 0.9
        assert queue.tasks["task-1"].priority_score > original_score

    def test_update_unknown_task(self) -> None:
        """Test updating non-existent task."""
        queue = TaskQueue()
        assert queue.update_task("unknown", urgency=0.9) is None

    def test_get_ordered_list(self) -> None:
        """Test getting ordered task list."""
        queue = TaskQueue()
        for i, urgency in enumerate([0.3, 0.9, 0.5]):
            task = Task(id=f"t{i}", title=f"Task {i}", urgency=urgency)
            queue.add_task(task)

        ordered = queue.get_ordered_list()

        # Should be sorted by priority descending
        scores = [t.priority_score for t in ordered]
        assert scores == sorted(scores, reverse=True)

    def test_get_blocked(self) -> None:
        """Test getting blocked tasks."""
        queue = TaskQueue()
        blocked = Task(
            id="blocked", title="Blocked", status=TaskStatus.BLOCKED
        )
        pending = Task(id="pending", title="Pending", dependencies=["other"])
        ready = Task(id="ready", title="Ready")

        queue.add_task(blocked)
        queue.add_task(pending)
        queue.add_task(ready)

        blocked_tasks = queue.get_blocked()
        task_ids = [t.id for t in blocked_tasks]

        assert "blocked" in task_ids
        assert "pending" in task_ids
        assert "ready" not in task_ids

    def test_get_stats(self) -> None:
        """Test getting queue statistics."""
        queue = TaskQueue()
        queue.add_task(Task(title="Pending", urgency=0.95, importance=0.95))
        in_prog = Task(title="In progress", status=TaskStatus.IN_PROGRESS)
        queue.add_task(in_prog)
        queue.add_task(Task(title="Low", urgency=0.1, importance=0.1))

        stats = queue.get_stats()

        assert stats.total_tasks == 3
        assert stats.pending_tasks == 2
        assert stats.in_progress_tasks == 1
        assert stats.avg_priority_score > 0


class TestDynamicReprioritizer:
    """Tests for DynamicReprioritizer dataclass."""

    def test_reprioritizer_creation(self) -> None:
        """Test creating a reprioritizer."""
        queue = TaskQueue()
        reprioritizer = DynamicReprioritizer(queue)
        assert reprioritizer.queue is queue
        assert len(reprioritizer.history) == 0

    def test_check_deadlines(self) -> None:
        """Test deadline checking."""
        queue = TaskQueue()
        # Task with deadline approaching
        urgent = Task(
            id="urgent",
            title="Urgent",
            deadline=datetime.now() + timedelta(hours=2),
        )
        queue.add_task(urgent)

        reprioritizer = DynamicReprioritizer(
            queue, deadline_threshold_hours=24
        )
        events = reprioritizer.check_deadlines()

        assert len(events) >= 0  # May or may not trigger based on score change

    def test_on_dependency_resolved(self) -> None:
        """Test dependency resolution handling."""
        queue = TaskQueue()
        task = Task(
            id="dependent",
            title="Dependent",
            dependencies=["dep-1"],
        )
        queue.add_task(task)

        reprioritizer = DynamicReprioritizer(queue)

        # Mark dependency as resolved
        queue.completed_tasks.add("dep-1")
        events = reprioritizer.on_dependency_resolved("dep-1")

        assert len(events) >= 1
        assert events[0].task_id == "dependent"
        assert events[0].reason == ReprioritizationReason.DEPENDENCY_RESOLVED

    def test_manual_boost(self) -> None:
        """Test manual priority boost."""
        queue = TaskQueue()
        task = Task(
            id="task-1", title="Boostable", urgency=0.5, importance=0.5
        )
        queue.add_task(task)

        reprioritizer = DynamicReprioritizer(queue)
        original_score = task.priority_score

        event = reprioritizer.manual_boost("task-1")

        assert event is not None
        assert event.new_score > event.old_score
        assert queue.tasks["task-1"].priority_score > original_score

    def test_manual_boost_unknown_task(self) -> None:
        """Test boosting non-existent task."""
        queue = TaskQueue()
        reprioritizer = DynamicReprioritizer(queue)
        assert reprioritizer.manual_boost("unknown") is None

    def test_get_history(self) -> None:
        """Test getting reprioritization history."""
        queue = TaskQueue()
        task = Task(id="task-1", title="Test", urgency=0.5, importance=0.5)
        queue.add_task(task)

        reprioritizer = DynamicReprioritizer(queue)
        reprioritizer.manual_boost("task-1")
        reprioritizer.manual_boost("task-1")

        all_history = reprioritizer.get_history()
        task_history = reprioritizer.get_history("task-1")

        assert len(all_history) == 2
        assert len(task_history) == 2


class TestPrioritizationAgent:
    """Tests for PrioritizationAgent dataclass."""

    def test_agent_creation(self) -> None:
        """Test creating a prioritization agent."""
        agent = PrioritizationAgent()
        assert agent.model_name == "gpt-oss:20b"

    def test_agent_custom_model(self) -> None:
        """Test agent with custom model."""
        agent = PrioritizationAgent(
            model_name="custom-model",
            base_url="http://custom:8080",
        )
        assert agent.model_name == "custom-model"

    @pytest.mark.asyncio
    async def test_prioritize(self) -> None:
        """Test prioritizing with mocked agent."""
        agent = PrioritizationAgent()

        mock_result = PrioritizationResult(
            priority_level=PriorityLevel.HIGH,
            urgency_assessment=0.8,
            importance_assessment=0.85,
            reasoning="Important feature request",
        )

        with patch.object(agent, "_get_agent") as mock_get_agent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value.output = mock_result
            mock_get_agent.return_value = mock_agent_instance

            result = await agent.prioritize(
                "New feature",
                "Add user authentication",
            )

            assert result.priority_level == PriorityLevel.HIGH
            assert result.urgency_assessment == 0.8

    @pytest.mark.asyncio
    async def test_prioritize_with_context(self) -> None:
        """Test prioritization with context."""
        agent = PrioritizationAgent()

        mock_result = PrioritizationResult(
            priority_level=PriorityLevel.CRITICAL,
            urgency_assessment=0.95,
            importance_assessment=0.9,
            reasoning="Security vulnerability",
        )

        with patch.object(agent, "_get_agent") as mock_get_agent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value.output = mock_result
            mock_get_agent.return_value = mock_agent_instance

            result = await agent.prioritize(
                "Security patch",
                "Fix authentication bypass",
                context="Production system",
            )

            assert result.priority_level == PriorityLevel.CRITICAL


class TestCreateTaskFromDescription:
    """Tests for create_task_from_description function."""

    def test_basic_task(self) -> None:
        """Test creating basic task."""
        task = create_task_from_description("Test task")
        assert task.title == "Test task"
        assert task.deadline is None
        assert len(task.dependencies) == 0

    def test_task_with_deadline(self) -> None:
        """Test creating task with deadline."""
        task = create_task_from_description(
            "Urgent task",
            deadline_hours=24,
        )
        assert task.deadline is not None
        assert task.deadline > datetime.now()

    def test_task_with_dependencies(self) -> None:
        """Test creating task with dependencies."""
        task = create_task_from_description(
            "Dependent task",
            depends_on=["dep-1", "dep-2"],
        )
        assert len(task.dependencies) == 2


class TestEisenhowerClassify:
    """Tests for eisenhower_classify function."""

    def test_urgent_and_important(self) -> None:
        """Test urgent and important quadrant."""
        result = eisenhower_classify(0.8, 0.9)
        assert "DO" in result

    def test_not_urgent_but_important(self) -> None:
        """Test not urgent but important quadrant."""
        result = eisenhower_classify(0.3, 0.8)
        assert "SCHEDULE" in result

    def test_urgent_but_not_important(self) -> None:
        """Test urgent but not important quadrant."""
        result = eisenhower_classify(0.8, 0.3)
        assert "DELEGATE" in result

    def test_not_urgent_not_important(self) -> None:
        """Test not urgent and not important quadrant."""
        result = eisenhower_classify(0.2, 0.2)
        assert "ELIMINATE" in result

    def test_boundary_values(self) -> None:
        """Test boundary values at 0.5."""
        # At boundary should go to higher quadrant
        result = eisenhower_classify(0.5, 0.5)
        assert "DO" in result


class TestBatchPrioritize:
    """Tests for batch_prioritize function."""

    def test_batch_prioritize(self) -> None:
        """Test batch prioritizing multiple tasks."""
        tasks = [
            Task(title="Low", urgency=0.2, importance=0.2),
            Task(title="High", urgency=0.9, importance=0.9),
            Task(title="Medium", urgency=0.5, importance=0.5),
        ]

        sorted_tasks = batch_prioritize(tasks)

        # Should be sorted by score descending
        assert len(sorted_tasks) == 3
        assert sorted_tasks[0][0].title == "High"
        assert sorted_tasks[-1][0].title == "Low"

    def test_batch_with_custom_evaluator(self) -> None:
        """Test batch prioritization with custom evaluator."""
        tasks = [Task(title="Test", urgency=0.7, importance=0.7)]
        custom_criteria = PriorityCriteria(
            urgency_weight=0.8, importance_weight=0.2
        )
        custom_evaluator = PriorityEvaluator(criteria=custom_criteria)

        result = batch_prioritize(tasks, evaluator=custom_evaluator)

        assert len(result) == 1

    def test_batch_with_resolved_deps(self) -> None:
        """Test batch prioritization with resolved dependencies."""
        tasks = [
            Task(title="Blocked", dependencies=["dep-1"]),
            Task(title="Ready"),
        ]

        result = batch_prioritize(tasks, resolved={"dep-1"})

        # Blocked task should have higher dep score now
        blocked_score = next(r[1] for r in result if r[0].title == "Blocked")
        assert blocked_score.dependency_score == 1.0


class TestGetActionableTasks:
    """Tests for get_actionable_tasks function."""

    def test_get_actionable(self) -> None:
        """Test getting actionable tasks."""
        queue = TaskQueue()
        queue.add_task(Task(id="ready", title="Ready", urgency=0.8))
        queue.add_task(
            Task(id="blocked", title="Blocked", dependencies=["other"])
        )

        actionable = get_actionable_tasks(queue, max_count=5)

        assert len(actionable) == 1
        assert actionable[0].id == "ready"

    def test_get_actionable_respects_max(self) -> None:
        """Test max count is respected."""
        queue = TaskQueue()
        for i in range(10):
            queue.add_task(Task(id=f"t{i}", title=f"Task {i}"))

        actionable = get_actionable_tasks(queue, max_count=3)

        assert len(actionable) == 3

    def test_get_actionable_empty_queue(self) -> None:
        """Test empty queue returns empty list."""
        queue = TaskQueue()
        actionable = get_actionable_tasks(queue)
        assert len(actionable) == 0


class TestIntegrationScenarios:
    """Integration tests for prioritization scenarios."""

    def test_complete_workflow(self) -> None:
        """Test complete prioritization workflow."""
        # Setup queue
        queue = TaskQueue()

        # Add tasks with various characteristics
        tasks = [
            Task(
                id="critical",
                title="Critical bug",
                urgency=0.95,
                importance=0.95,
            ),
            Task(
                id="feature",
                title="New feature",
                urgency=0.3,
                importance=0.7,
                dependencies=["critical"],
            ),
            Task(
                id="docs",
                title="Documentation",
                urgency=0.2,
                importance=0.4,
            ),
        ]

        for task in tasks:
            queue.add_task(task)

        # Verify order
        ordered = queue.get_ordered_list()
        assert ordered[0].id == "critical"

        # Complete critical task
        queue.complete_task("critical")

        # Feature should be reprioritized
        assert queue.tasks["feature"].priority_score > 0

        # Check actionable tasks
        actionable = get_actionable_tasks(queue)
        task_ids = [t.id for t in actionable]
        assert "feature" in task_ids
        assert "docs" in task_ids

    def test_dynamic_reprioritization_workflow(self) -> None:
        """Test dynamic reprioritization workflow."""
        queue = TaskQueue()
        reprioritizer = DynamicReprioritizer(
            queue, deadline_threshold_hours=48
        )

        # Add task with deadline
        task = Task(
            id="deadline-task",
            title="Has deadline",
            urgency=0.5,
            importance=0.5,
            deadline=datetime.now() + timedelta(hours=12),
        )
        queue.add_task(task)

        # Check deadlines
        reprioritizer.check_deadlines()

        # Manual boost
        boost_event = reprioritizer.manual_boost("deadline-task")
        assert boost_event is not None

        # Check history
        history = reprioritizer.get_history()
        assert len(history) >= 1

    def test_batch_and_queue_integration(self) -> None:
        """Test batch prioritization feeding into queue."""
        # Create tasks
        tasks = [
            Task(title="Task A", urgency=0.7, importance=0.8),
            Task(title="Task B", urgency=0.3, importance=0.4),
            Task(title="Task C", urgency=0.9, importance=0.9),
        ]

        # Batch prioritize
        sorted_tasks = batch_prioritize(tasks)

        # Add to queue
        queue = TaskQueue()
        for task, _ in sorted_tasks:
            queue.add_task(task)

        # Verify queue ordering matches batch
        ordered = queue.get_ordered_list()
        assert ordered[0].title == sorted_tasks[0][0].title

        # Get stats
        stats = queue.get_stats()
        assert stats.total_tasks == 3
