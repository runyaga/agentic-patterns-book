"""Tests for goal_monitoring pattern."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic_graph import End

from agentic_patterns.goal_monitoring import CheckNode
from agentic_patterns.goal_monitoring import Goal
from agentic_patterns.goal_monitoring import GoalMonitor
from agentic_patterns.goal_monitoring import GoalStatus
from agentic_patterns.goal_monitoring import MonitorState
from agentic_patterns.goal_monitoring import RemediateNode
from agentic_patterns.goal_monitoring import RemediationResult
from agentic_patterns.goal_monitoring import WaitNode
from agentic_patterns.goal_monitoring import on_escalate


class TestModels:
    """Test data model validation."""

    def test_goal_creation_defaults(self):
        """Goal should have sensible defaults."""

        async def dummy_eval():
            return 50.0

        goal = Goal(
            name="test",
            target=80.0,
            evaluator=dummy_eval,
        )
        assert goal.name == "test"
        assert goal.target == 80.0
        assert goal.comparator == ">="
        assert goal.remediation_hint == ""

    def test_goal_with_all_fields(self):
        """Goal should accept all fields."""

        async def dummy_eval():
            return 50.0

        goal = Goal(
            name="disk_usage",
            target=80.0,
            evaluator=dummy_eval,
            comparator="<=",
            remediation_hint="Clean up disk space",
        )
        assert goal.comparator == "<="
        assert goal.remediation_hint == "Clean up disk space"

    def test_goal_status_creation(self):
        """GoalStatus should capture check results."""
        now = datetime.now()
        status = GoalStatus(
            goal_name="test",
            current_value=75.0,
            target_value=80.0,
            is_met=False,
            checked_at=now,
        )
        assert status.goal_name == "test"
        assert status.current_value == 75.0
        assert status.is_met is False
        assert status.checked_at == now

    def test_monitor_state_defaults(self):
        """MonitorState should have sensible defaults."""
        state = MonitorState(goals=[])
        assert state.check_interval == 60.0
        assert state.shutdown is False
        assert state.current_gap is None
        assert state.last_status == []

    def test_remediation_result_validation(self):
        """RemediationResult should validate fields."""
        result = RemediationResult(
            success=True,
            action_taken="Cleaned up temp files",
        )
        assert result.success is True
        assert result.error is None

        result_failed = RemediationResult(
            success=False,
            action_taken="",
            error="Permission denied",
        )
        assert result_failed.success is False
        assert result_failed.error == "Permission denied"


class TestCheckNode:
    """Test CheckNode behavior."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock GraphRunContext."""
        ctx = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_check_detects_gap_less_than_target(self, mock_context):
        """CheckNode should detect when current < target (for >=)."""

        async def failing_eval():
            return 50.0  # Below target of 80

        goal = Goal(name="test", target=80.0, evaluator=failing_eval)
        state = MonitorState(goals=[goal])
        mock_context.state = state

        node = CheckNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, RemediateNode)
        assert state.current_gap == goal
        assert len(state.last_status) == 1
        assert state.last_status[0].is_met is False

    @pytest.mark.asyncio
    async def test_check_passes_when_met(self, mock_context):
        """CheckNode should pass when goal is met."""

        async def passing_eval():
            return 90.0  # Above target of 80

        goal = Goal(name="test", target=80.0, evaluator=passing_eval)
        state = MonitorState(goals=[goal])
        mock_context.state = state

        node = CheckNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, WaitNode)
        assert state.current_gap is None
        assert state.last_status[0].is_met is True

    @pytest.mark.asyncio
    async def test_check_less_than_comparator(self, mock_context):
        """CheckNode should handle <= comparator."""

        async def eval_under():
            return 70.0  # Under target of 80, so met for <=

        goal = Goal(
            name="disk",
            target=80.0,
            comparator="<=",
            evaluator=eval_under,
        )
        state = MonitorState(goals=[goal])
        mock_context.state = state

        node = CheckNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, WaitNode)
        assert state.last_status[0].is_met is True

    @pytest.mark.asyncio
    async def test_check_equal_comparator(self, mock_context):
        """CheckNode should handle == comparator."""

        async def exact_eval():
            return 80.0  # Exactly target

        goal = Goal(
            name="exact",
            target=80.0,
            comparator="==",
            evaluator=exact_eval,
        )
        state = MonitorState(goals=[goal])
        mock_context.state = state

        node = CheckNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, WaitNode)
        assert state.last_status[0].is_met is True

    @pytest.mark.asyncio
    async def test_check_multiple_goals_first_fails(self, mock_context):
        """CheckNode should stop at first failing goal."""

        async def failing_eval():
            return 50.0

        async def passing_eval():
            return 90.0

        goal1 = Goal(name="first", target=80.0, evaluator=failing_eval)
        goal2 = Goal(name="second", target=80.0, evaluator=passing_eval)
        state = MonitorState(goals=[goal1, goal2])
        mock_context.state = state

        node = CheckNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, RemediateNode)
        assert state.current_gap == goal1
        # Only first goal was checked
        assert len(state.last_status) == 1

    @pytest.mark.asyncio
    async def test_check_evaluator_exception(self, mock_context):
        """CheckNode should handle evaluator exceptions as gaps."""

        async def broken_eval():
            raise ValueError("Connection failed")

        goal = Goal(name="broken", target=80.0, evaluator=broken_eval)
        state = MonitorState(goals=[goal])
        mock_context.state = state

        node = CheckNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, RemediateNode)
        assert state.current_gap == goal

    @pytest.mark.asyncio
    async def test_check_no_goals(self, mock_context):
        """CheckNode should return WaitNode when no goals."""
        state = MonitorState(goals=[])
        mock_context.state = state

        node = CheckNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, WaitNode)


class TestWaitNode:
    """Test WaitNode behavior."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock GraphRunContext."""
        ctx = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_wait_returns_check_node(self, mock_context):
        """WaitNode should return CheckNode after interval."""
        state = MonitorState(goals=[], check_interval=0.01)
        mock_context.state = state

        node = WaitNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, CheckNode)

    @pytest.mark.asyncio
    async def test_wait_exits_on_shutdown(self, mock_context):
        """WaitNode should return End when shutdown is True."""
        state = MonitorState(goals=[], shutdown=True)
        mock_context.state = state

        node = WaitNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, End)

    @pytest.mark.asyncio
    async def test_wait_respects_interval(self, mock_context):
        """WaitNode should sleep for check_interval."""
        state = MonitorState(goals=[], check_interval=0.5)
        mock_context.state = state

        with patch("agentic_patterns.goal_monitoring.asyncio.sleep") as mock:
            mock.return_value = None
            node = WaitNode()
            await node.run(mock_context)

            mock.assert_called_once_with(0.5)


class TestRemediateNode:
    """Test RemediateNode behavior."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock GraphRunContext."""
        ctx = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_remediate_calls_agent(self, mock_context):
        """RemediateNode should call remediation agent."""

        async def dummy_eval():
            return 50.0

        goal = Goal(
            name="test",
            target=80.0,
            evaluator=dummy_eval,
            remediation_hint="Fix the thing",
        )
        state = MonitorState(goals=[goal], current_gap=goal)
        mock_context.state = state

        mock_result = MagicMock()
        mock_result.output = RemediationResult(
            success=True,
            action_taken="Fixed it",
        )

        patch_path = "agentic_patterns.goal_monitoring.remediation_agent"
        with patch(patch_path) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            node = RemediateNode()
            next_node = await node.run(mock_context)

            assert isinstance(next_node, CheckNode)
            assert state.current_gap is None
            mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_remediate_escalates_on_failure(self, mock_context):
        """RemediateNode should call on_escalate when remediation fails."""

        async def dummy_eval():
            return 50.0

        goal = Goal(name="test", target=80.0, evaluator=dummy_eval)
        state = MonitorState(goals=[goal], current_gap=goal)
        mock_context.state = state

        mock_result = MagicMock()
        mock_result.output = RemediationResult(
            success=False,
            action_taken="",
            error="Could not fix",
        )

        patch_agent = "agentic_patterns.goal_monitoring.remediation_agent"
        patch_escalate = "agentic_patterns.goal_monitoring.on_escalate"
        with (
            patch(patch_agent) as mock_agent,
            patch(patch_escalate) as mock_escalate,
        ):
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_escalate.return_value = None

            node = RemediateNode()
            await node.run(mock_context)

            mock_escalate.assert_called_once_with(goal, "Could not fix")

    @pytest.mark.asyncio
    async def test_remediate_no_gap(self, mock_context):
        """RemediateNode should return CheckNode when no gap."""
        state = MonitorState(goals=[], current_gap=None)
        mock_context.state = state

        node = RemediateNode()
        next_node = await node.run(mock_context)

        assert isinstance(next_node, CheckNode)

    @pytest.mark.asyncio
    async def test_remediate_agent_exception(self, mock_context):
        """RemediateNode should escalate on agent exception."""

        async def dummy_eval():
            return 50.0

        goal = Goal(name="test", target=80.0, evaluator=dummy_eval)
        state = MonitorState(goals=[goal], current_gap=goal)
        mock_context.state = state

        patch_agent = "agentic_patterns.goal_monitoring.remediation_agent"
        patch_escalate = "agentic_patterns.goal_monitoring.on_escalate"
        with (
            patch(patch_agent) as mock_agent,
            patch(patch_escalate) as mock_escalate,
        ):
            mock_agent.run = AsyncMock(side_effect=Exception("LLM error"))
            mock_escalate.return_value = None

            node = RemediateNode()
            next_node = await node.run(mock_context)

            assert isinstance(next_node, CheckNode)
            mock_escalate.assert_called_once()


class TestGoalMonitor:
    """Test GoalMonitor lifecycle."""

    @pytest.mark.asyncio
    async def test_monitor_start_creates_task(self):
        """GoalMonitor.start() should create background task."""

        async def always_met():
            return 100.0

        goal = Goal(name="test", target=80.0, evaluator=always_met)
        monitor = GoalMonitor([goal], check_interval=0.01)

        await monitor.start()

        assert monitor._task is not None
        assert not monitor._task.done()
        assert monitor._state is not None

        # Stop to clean up
        monitor._state.shutdown = True
        await asyncio.sleep(0.02)
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_monitor_stop_sets_shutdown(self):
        """GoalMonitor.stop() should set shutdown flag."""

        async def always_met():
            return 100.0

        goal = Goal(name="test", target=80.0, evaluator=always_met)
        monitor = GoalMonitor([goal], check_interval=0.01)

        await monitor.start()
        await monitor.stop()

        assert monitor._state.shutdown is True
        assert monitor._task.done()

    @pytest.mark.asyncio
    async def test_monitor_get_status_empty(self):
        """GoalMonitor.get_status() returns empty before start."""
        monitor = GoalMonitor([], check_interval=60.0)
        assert monitor.get_status() == []

    @pytest.mark.asyncio
    async def test_monitor_get_status_after_check(self):
        """GoalMonitor.get_status() returns results after check."""

        async def eval_passing():
            return 100.0  # Above target, so no remediation needed

        goal = Goal(name="test", target=80.0, evaluator=eval_passing)
        monitor = GoalMonitor([goal], check_interval=0.001)

        await monitor.start()

        # Poll until status is populated or timeout
        for _ in range(50):
            await asyncio.sleep(0.01)
            status = monitor.get_status()
            if status:
                break

        # Request shutdown
        monitor._state.shutdown = True
        await asyncio.sleep(0.01)
        await monitor.stop()

        assert len(status) >= 1
        assert status[0].goal_name == "test"
        assert status[0].is_met is True


class TestOnEscalate:
    """Test escalation stub."""

    @pytest.mark.asyncio
    async def test_on_escalate_prints(self, capsys):
        """on_escalate should print escalation message."""

        async def dummy_eval():
            return 50.0

        goal = Goal(name="critical", target=80.0, evaluator=dummy_eval)

        await on_escalate(goal, "System failure")

        captured = capsys.readouterr()
        assert "ESCALATE" in captured.out
        assert "critical" in captured.out
        assert "System failure" in captured.out
