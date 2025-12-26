"""Tests for the Planning Pattern implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.planning import Plan
from agentic_patterns.planning import PlanExecutionResult
from agentic_patterns.planning import PlanStep
from agentic_patterns.planning import StepResult
from agentic_patterns.planning import StepStatus
from agentic_patterns.planning import create_plan
from agentic_patterns.planning import execute_plan
from agentic_patterns.planning import execute_step
from agentic_patterns.planning import plan_and_execute
from agentic_patterns.planning import replan


class TestModels:
    """Test Pydantic model validation."""

    def test_step_status_enum(self):
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"

    def test_plan_step_valid(self):
        step = PlanStep(
            step_number=1,
            description="Research Python benefits",
            expected_output="List of benefits",
        )
        assert step.step_number == 1
        assert step.status == StepStatus.PENDING
        assert step.dependencies == []

    def test_plan_step_with_dependencies(self):
        step = PlanStep(
            step_number=3,
            description="Combine results",
            expected_output="Final summary",
            dependencies=[1, 2],
        )
        assert step.dependencies == [1, 2]

    def test_plan_valid(self):
        steps = [
            PlanStep(
                step_number=1,
                description="Step 1",
                expected_output="Output 1",
            ),
            PlanStep(
                step_number=2,
                description="Step 2",
                expected_output="Output 2",
                dependencies=[1],
            ),
        ]
        plan = Plan(
            goal="Achieve something",
            steps=steps,
            reasoning="This is the best approach",
        )
        assert plan.goal == "Achieve something"
        assert len(plan.steps) == 2
        assert plan.reasoning != ""

    def test_plan_empty_steps(self):
        plan = Plan(
            goal="Simple goal",
            steps=[],
            reasoning="No steps needed",
        )
        assert len(plan.steps) == 0

    def test_step_result_success(self):
        result = StepResult(
            step_number=1,
            success=True,
            output="Step completed successfully",
        )
        assert result.success
        assert result.needs_replan is False

    def test_step_result_failure(self):
        result = StepResult(
            step_number=2,
            success=False,
            output="Error: could not complete",
            needs_replan=True,
        )
        assert not result.success
        assert result.needs_replan

    def test_plan_execution_result_success(self):
        result = PlanExecutionResult(
            goal="Test goal",
            success=True,
            completed_steps=3,
            total_steps=3,
            step_results=[],
            final_output="All done",
        )
        assert result.success
        assert result.completed_steps == result.total_steps
        assert result.replanned is False

    def test_plan_execution_result_partial(self):
        result = PlanExecutionResult(
            goal="Test goal",
            success=False,
            completed_steps=2,
            total_steps=4,
            step_results=[],
            final_output="Partial completion",
            replanned=True,
        )
        assert not result.success
        assert result.completed_steps < result.total_steps
        assert result.replanned


class TestCreatePlan:
    """Test plan creation."""

    @pytest.fixture
    def mock_plan(self):
        return Plan(
            goal="Research Python",
            steps=[
                PlanStep(
                    step_number=1,
                    description="Search for Python benefits",
                    expected_output="List of benefits",
                ),
                PlanStep(
                    step_number=2,
                    description="Analyze findings",
                    expected_output="Analysis summary",
                    dependencies=[1],
                ),
            ],
            reasoning="Two-step research approach",
        )

    @pytest.mark.asyncio
    async def test_create_plan_basic(self, mock_plan):
        """Test basic plan creation."""
        mock_result = MagicMock()
        mock_result.output = mock_plan

        with patch("agentic_patterns.planning.planner_agent") as mock_planner:
            mock_planner.run = AsyncMock(return_value=mock_result)

            plan = await create_plan("Research Python benefits")

            assert plan.goal == "Research Python"
            assert len(plan.steps) == 2
            mock_planner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_plan_with_max_steps(self, mock_plan):
        """Test plan respects max_steps parameter."""
        mock_result = MagicMock()
        mock_result.output = mock_plan

        with patch("agentic_patterns.planning.planner_agent") as mock_planner:
            mock_planner.run = AsyncMock(return_value=mock_result)

            await create_plan("Test goal", max_steps=3)

            call_args = mock_planner.run.call_args[0][0]
            assert "3 steps" in call_args


class TestExecuteStep:
    """Test individual step execution."""

    @pytest.fixture
    def sample_step(self):
        return PlanStep(
            step_number=1,
            description="Find Python documentation",
            expected_output="Links to documentation",
        )

    @pytest.fixture
    def mock_step_result_success(self):
        return StepResult(
            step_number=1,
            success=True,
            output="Found official docs at python.org",
        )

    @pytest.fixture
    def mock_step_result_failure(self):
        return StepResult(
            step_number=1,
            success=False,
            output="Could not access documentation",
            needs_replan=True,
        )

    @pytest.mark.asyncio
    async def test_execute_step_success(
        self, sample_step, mock_step_result_success
    ):
        """Test successful step execution."""
        mock_result = MagicMock()
        mock_result.output = mock_step_result_success

        with patch(
            "agentic_patterns.planning.executor_agent"
        ) as mock_executor:
            mock_executor.run = AsyncMock(return_value=mock_result)

            result = await execute_step(sample_step)

            assert result.success
            assert result.step_number == 1
            mock_executor.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_step_with_context(
        self, sample_step, mock_step_result_success
    ):
        """Test step execution with previous context."""
        mock_result = MagicMock()
        mock_result.output = mock_step_result_success

        with patch(
            "agentic_patterns.planning.executor_agent"
        ) as mock_executor:
            mock_executor.run = AsyncMock(return_value=mock_result)

            await execute_step(
                sample_step,
                context="Previous step found Python 3.12",
            )

            call_args = mock_executor.run.call_args[0][0]
            assert "Python 3.12" in call_args

    @pytest.mark.asyncio
    async def test_execute_step_failure(
        self, sample_step, mock_step_result_failure
    ):
        """Test failed step execution."""
        mock_result = MagicMock()
        mock_result.output = mock_step_result_failure

        with patch(
            "agentic_patterns.planning.executor_agent"
        ) as mock_executor:
            mock_executor.run = AsyncMock(return_value=mock_result)

            result = await execute_step(sample_step)

            assert not result.success
            assert result.needs_replan


class TestReplan:
    """Test re-planning functionality."""

    @pytest.fixture
    def original_plan(self):
        return Plan(
            goal="Research topic",
            steps=[
                PlanStep(
                    step_number=1,
                    description="Search",
                    expected_output="Results",
                ),
                PlanStep(
                    step_number=2,
                    description="Analyze",
                    expected_output="Analysis",
                    dependencies=[1],
                ),
            ],
            reasoning="Original approach",
        )

    @pytest.fixture
    def failed_step_result(self):
        return StepResult(
            step_number=1,
            success=False,
            output="Search failed due to timeout",
            needs_replan=True,
        )

    @pytest.fixture
    def revised_plan(self):
        return Plan(
            goal="Research topic",
            steps=[
                PlanStep(
                    step_number=1,
                    description="Try alternative search",
                    expected_output="Results",
                ),
            ],
            reasoning="Alternative approach after failure",
        )

    @pytest.mark.asyncio
    async def test_replan_creates_new_plan(
        self, original_plan, failed_step_result, revised_plan
    ):
        """Test that replan creates a new plan."""
        mock_result = MagicMock()
        mock_result.output = revised_plan

        with patch(
            "agentic_patterns.planning.replanner_agent"
        ) as mock_replanner:
            mock_replanner.run = AsyncMock(return_value=mock_result)

            new_plan = await replan(
                original_plan,
                failed_step_result,
                [],
            )

            assert new_plan.reasoning == "Alternative approach after failure"
            assert len(new_plan.steps) == 1
            mock_replanner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_replan_includes_context(
        self, original_plan, failed_step_result, revised_plan
    ):
        """Test that replan includes completed step context."""
        mock_result = MagicMock()
        mock_result.output = revised_plan

        completed = [
            StepResult(
                step_number=0,
                success=True,
                output="Setup completed",
            )
        ]

        with patch(
            "agentic_patterns.planning.replanner_agent"
        ) as mock_replanner:
            mock_replanner.run = AsyncMock(return_value=mock_result)

            await replan(original_plan, failed_step_result, completed)

            call_args = mock_replanner.run.call_args[0][0]
            assert "Setup completed" in call_args


class TestExecutePlan:
    """Test full plan execution."""

    @pytest.fixture
    def simple_plan(self):
        return Plan(
            goal="Simple goal",
            steps=[
                PlanStep(
                    step_number=1,
                    description="Step one",
                    expected_output="Output one",
                ),
                PlanStep(
                    step_number=2,
                    description="Step two",
                    expected_output="Output two",
                ),
            ],
            reasoning="Simple two-step plan",
        )

    @pytest.fixture
    def plan_with_deps(self):
        return Plan(
            goal="Dependent goal",
            steps=[
                PlanStep(
                    step_number=1,
                    description="First step",
                    expected_output="First output",
                ),
                PlanStep(
                    step_number=2,
                    description="Depends on first",
                    expected_output="Second output",
                    dependencies=[1],
                ),
            ],
            reasoning="Plan with dependencies",
        )

    @pytest.mark.asyncio
    async def test_execute_plan_all_success(self, simple_plan):
        """Test executing a plan where all steps succeed."""
        step_results = [
            StepResult(step_number=1, success=True, output="Done 1"),
            StepResult(step_number=2, success=True, output="Done 2"),
        ]

        mock_synth_result = MagicMock()
        mock_synth_result.output = "Final answer"

        call_count = 0

        async def mock_exec_run(prompt):
            nonlocal call_count
            result = MagicMock()
            result.output = step_results[call_count]
            call_count += 1
            return result

        with (
            patch("agentic_patterns.planning.executor_agent") as mock_executor,
            patch("agentic_patterns.planning.synthesizer_agent") as mock_synth,
        ):
            mock_executor.run = AsyncMock(side_effect=mock_exec_run)
            mock_synth.run = AsyncMock(return_value=mock_synth_result)

            result = await execute_plan(simple_plan)

            assert result.success
            assert result.completed_steps == 2
            assert result.total_steps == 2
            assert result.final_output == "Final answer"

    @pytest.mark.asyncio
    async def test_execute_plan_skips_unmet_deps(self, plan_with_deps):
        """Test that steps with unmet dependencies are skipped."""
        # First step fails, so second should be skipped
        step1_result = StepResult(
            step_number=1,
            success=False,
            output="Failed",
        )

        mock_synth_result = MagicMock()
        mock_synth_result.output = "Partial result"

        with (
            patch("agentic_patterns.planning.executor_agent") as mock_executor,
            patch("agentic_patterns.planning.synthesizer_agent") as mock_synth,
            patch("agentic_patterns.planning.replanner_agent"),
        ):
            mock_exec_result = MagicMock()
            mock_exec_result.output = step1_result
            mock_executor.run = AsyncMock(return_value=mock_exec_result)
            mock_synth.run = AsyncMock(return_value=mock_synth_result)

            # Disable replanning for this test
            result = await execute_plan(plan_with_deps, allow_replan=False)

            # Step 2 should be skipped because step 1 failed
            assert not result.success
            assert len(result.step_results) == 2  # Both attempted
            # Second result should indicate skip
            skipped = [r for r in result.step_results if "Skipped" in r.output]
            assert len(skipped) == 1


class TestPlanAndExecute:
    """Test the convenience function."""

    @pytest.mark.asyncio
    async def test_plan_and_execute_full_flow(self):
        """Test complete plan-and-execute flow."""
        mock_plan = Plan(
            goal="Test goal",
            steps=[
                PlanStep(
                    step_number=1,
                    description="Single step",
                    expected_output="Output",
                ),
            ],
            reasoning="Simple plan",
        )

        mock_step_result = StepResult(
            step_number=1,
            success=True,
            output="Completed",
        )

        mock_planner_result = MagicMock()
        mock_planner_result.output = mock_plan

        mock_exec_result = MagicMock()
        mock_exec_result.output = mock_step_result

        mock_synth_result = MagicMock()
        mock_synth_result.output = "Final answer"

        with (
            patch("agentic_patterns.planning.planner_agent") as mock_planner,
            patch("agentic_patterns.planning.executor_agent") as mock_executor,
            patch("agentic_patterns.planning.synthesizer_agent") as mock_synth,
        ):
            mock_planner.run = AsyncMock(return_value=mock_planner_result)
            mock_executor.run = AsyncMock(return_value=mock_exec_result)
            mock_synth.run = AsyncMock(return_value=mock_synth_result)

            result = await plan_and_execute("Test goal", max_steps=3)

            assert result.success
            assert result.goal == "Test goal"
            mock_planner.run.assert_called_once()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_plan_step_zero_dependencies(self):
        step = PlanStep(
            step_number=1,
            description="Independent step",
            expected_output="Output",
            dependencies=[],
        )
        assert len(step.dependencies) == 0

    def test_plan_step_many_dependencies(self):
        step = PlanStep(
            step_number=10,
            description="Final step",
            expected_output="Final output",
            dependencies=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        assert len(step.dependencies) == 9

    def test_step_result_empty_output(self):
        result = StepResult(
            step_number=1,
            success=True,
            output="",
        )
        assert result.output == ""

    def test_plan_execution_result_no_step_results(self):
        result = PlanExecutionResult(
            goal="Empty plan",
            success=True,
            completed_steps=0,
            total_steps=0,
            step_results=[],
            final_output="Nothing to do",
        )
        assert len(result.step_results) == 0

    def test_plan_single_step(self):
        plan = Plan(
            goal="Quick task",
            steps=[
                PlanStep(
                    step_number=1,
                    description="Only step",
                    expected_output="Done",
                ),
            ],
            reasoning="Single action needed",
        )
        assert len(plan.steps) == 1

    def test_step_status_transitions(self):
        step = PlanStep(
            step_number=1,
            description="Test step",
            expected_output="Output",
            status=StepStatus.PENDING,
        )
        assert step.status == StepStatus.PENDING

        step.status = StepStatus.IN_PROGRESS
        assert step.status == StepStatus.IN_PROGRESS

        step.status = StepStatus.COMPLETED
        assert step.status == StepStatus.COMPLETED
