"""Tests for the Multi-Agent Collaboration Pattern implementation."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.multi_agent import AgentMessage
from agentic_patterns.multi_agent import AgentRole
from agentic_patterns.multi_agent import CollaborationContext
from agentic_patterns.multi_agent import CollaborationResult
from agentic_patterns.multi_agent import DelegatedTask
from agentic_patterns.multi_agent import SupervisorPlan
from agentic_patterns.multi_agent import TaskResult
from agentic_patterns.multi_agent import TaskStatus
from agentic_patterns.multi_agent import create_delegation_plan
from agentic_patterns.multi_agent import execute_task
from agentic_patterns.multi_agent import run_collaborative_task
from agentic_patterns.multi_agent import run_network_collaboration
from agentic_patterns.multi_agent import synthesize_results


class TestEnums:
    """Test enum definitions."""

    def test_agent_role_values(self):
        assert AgentRole.SUPERVISOR.value == "supervisor"
        assert AgentRole.RESEARCHER.value == "researcher"
        assert AgentRole.ANALYST.value == "analyst"
        assert AgentRole.WRITER.value == "writer"
        assert AgentRole.REVIEWER.value == "reviewer"

    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


class TestAgentMessage:
    """Test AgentMessage model."""

    def test_message_basic(self):
        msg = AgentMessage(
            sender=AgentRole.RESEARCHER,
            recipient=AgentRole.SUPERVISOR,
            content="Research complete",
        )
        assert msg.sender == AgentRole.RESEARCHER
        assert msg.recipient == AgentRole.SUPERVISOR
        assert msg.content == "Research complete"
        assert msg.task_id is None

    def test_message_with_task_id(self):
        msg = AgentMessage(
            sender=AgentRole.ANALYST,
            recipient=AgentRole.WRITER,
            content="Analysis ready",
            task_id=42,
        )
        assert msg.task_id == 42


class TestDelegatedTask:
    """Test DelegatedTask model."""

    def test_task_basic(self):
        task = DelegatedTask(
            task_id=1,
            assigned_to=AgentRole.RESEARCHER,
            description="Research Python patterns",
        )
        assert task.task_id == 1
        assert task.assigned_to == AgentRole.RESEARCHER
        assert task.status == TaskStatus.PENDING
        assert task.context == ""

    def test_task_with_context(self):
        task = DelegatedTask(
            task_id=2,
            assigned_to=AgentRole.WRITER,
            description="Write summary",
            context="Focus on async patterns",
            status=TaskStatus.IN_PROGRESS,
        )
        assert task.context == "Focus on async patterns"
        assert task.status == TaskStatus.IN_PROGRESS


class TestTaskResult:
    """Test TaskResult model."""

    def test_result_success(self):
        result = TaskResult(
            task_id=1,
            agent_role=AgentRole.RESEARCHER,
            success=True,
            output="Found 5 relevant sources",
        )
        assert result.success
        assert result.artifacts == []

    def test_result_failure(self):
        result = TaskResult(
            task_id=2,
            agent_role=AgentRole.ANALYST,
            success=False,
            output="Insufficient data",
        )
        assert not result.success

    def test_result_with_artifacts(self):
        result = TaskResult(
            task_id=3,
            agent_role=AgentRole.WRITER,
            success=True,
            output="Document created",
            artifacts=["summary.md", "code_examples.py"],
        )
        assert len(result.artifacts) == 2


class TestSupervisorPlan:
    """Test SupervisorPlan model."""

    def test_plan_basic(self):
        tasks = [
            DelegatedTask(
                task_id=1,
                assigned_to=AgentRole.RESEARCHER,
                description="Research topic",
            ),
        ]
        plan = SupervisorPlan(
            objective="Learn about Python",
            tasks=tasks,
            reasoning="Start with research",
        )
        assert plan.objective == "Learn about Python"
        assert len(plan.tasks) == 1

    def test_plan_multiple_tasks(self):
        tasks = [
            DelegatedTask(
                task_id=1,
                assigned_to=AgentRole.RESEARCHER,
                description="Research",
            ),
            DelegatedTask(
                task_id=2,
                assigned_to=AgentRole.ANALYST,
                description="Analyze",
            ),
            DelegatedTask(
                task_id=3,
                assigned_to=AgentRole.WRITER,
                description="Write",
            ),
        ]
        plan = SupervisorPlan(
            objective="Complete project",
            tasks=tasks,
            reasoning="Three-phase approach",
        )
        assert len(plan.tasks) == 3

    def test_plan_empty_tasks(self):
        plan = SupervisorPlan(
            objective="Simple task",
            tasks=[],
            reasoning="No delegation needed",
        )
        assert len(plan.tasks) == 0


class TestCollaborationResult:
    """Test CollaborationResult model."""

    def test_result_success(self):
        result = CollaborationResult(
            objective="Test objective",
            success=True,
            task_results=[],
            final_output="All done",
        )
        assert result.success
        assert result.messages_exchanged == 0

    def test_result_with_tasks(self):
        task_results = [
            TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Research done",
            ),
            TaskResult(
                task_id=2,
                agent_role=AgentRole.WRITER,
                success=True,
                output="Writing done",
            ),
        ]
        result = CollaborationResult(
            objective="Collaborative project",
            success=True,
            task_results=task_results,
            final_output="Combined output",
            messages_exchanged=4,
        )
        assert len(result.task_results) == 2
        assert result.messages_exchanged == 4


class TestCollaborationContext:
    """Test CollaborationContext dataclass."""

    def test_context_empty(self):
        ctx = CollaborationContext(messages=[], task_results=[])
        assert len(ctx.messages) == 0
        assert len(ctx.task_results) == 0

    def test_context_with_data(self):
        messages = [
            AgentMessage(
                sender=AgentRole.RESEARCHER,
                recipient=AgentRole.SUPERVISOR,
                content="Done",
            )
        ]
        results = [
            TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Output",
            )
        ]
        ctx = CollaborationContext(messages=messages, task_results=results)
        assert len(ctx.messages) == 1
        assert len(ctx.task_results) == 1


class TestCreateDelegationPlan:
    """Test supervisor delegation planning."""

    @pytest.fixture
    def mock_plan(self):
        return SupervisorPlan(
            objective="Test objective",
            tasks=[
                DelegatedTask(
                    task_id=1,
                    assigned_to=AgentRole.RESEARCHER,
                    description="Research first",
                ),
                DelegatedTask(
                    task_id=2,
                    assigned_to=AgentRole.WRITER,
                    description="Write summary",
                ),
            ],
            reasoning="Research then write",
        )

    @pytest.mark.asyncio
    async def test_create_plan_basic(self, mock_plan):
        """Test basic plan creation."""
        mock_result = MagicMock()
        mock_result.output = mock_plan

        with patch(
            "agentic_patterns.multi_agent.supervisor_agent"
        ) as mock_supervisor:
            mock_supervisor.run = AsyncMock(return_value=mock_result)

            plan = await create_delegation_plan("Test objective")

            assert plan.objective == "Test objective"
            assert len(plan.tasks) == 2
            mock_supervisor.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_plan_prompt_content(self, mock_plan):
        """Test that prompt includes objective."""
        mock_result = MagicMock()
        mock_result.output = mock_plan

        with patch(
            "agentic_patterns.multi_agent.supervisor_agent"
        ) as mock_supervisor:
            mock_supervisor.run = AsyncMock(return_value=mock_result)

            await create_delegation_plan("Research Python async patterns")

            call_args = mock_supervisor.run.call_args[0][0]
            assert "Research Python async patterns" in call_args


class TestExecuteTask:
    """Test task execution by worker agents."""

    @pytest.fixture
    def sample_task(self):
        return DelegatedTask(
            task_id=1,
            assigned_to=AgentRole.RESEARCHER,
            description="Research Python best practices",
            context="Focus on modern Python 3.10+",
        )

    @pytest.fixture
    def sample_context(self):
        return CollaborationContext(messages=[], task_results=[])

    @pytest.fixture
    def mock_task_result_success(self):
        return TaskResult(
            task_id=1,
            agent_role=AgentRole.RESEARCHER,
            success=True,
            output="Found 10 best practices",
        )

    @pytest.mark.asyncio
    async def test_execute_task_success(
        self, sample_task, sample_context, mock_task_result_success
    ):
        """Test successful task execution."""
        mock_result = MagicMock()
        mock_result.output = mock_task_result_success

        with patch(
            "agentic_patterns.multi_agent.ROLE_AGENTS",
            {AgentRole.RESEARCHER: MagicMock()},
        ) as mock_agents:
            mock_agents[AgentRole.RESEARCHER].run = AsyncMock(
                return_value=mock_result
            )

            result = await execute_task(sample_task, sample_context)

            assert result.success
            assert result.task_id == 1

    @pytest.mark.asyncio
    async def test_execute_task_unknown_role(self, sample_context):
        """Test execution with unknown agent role."""
        task = DelegatedTask(
            task_id=1,
            assigned_to=AgentRole.SUPERVISOR,  # Supervisor not in ROLE_AGENTS
            description="Invalid task",
        )

        result = await execute_task(task, sample_context)

        assert not result.success
        assert "No agent available" in result.output

    @pytest.mark.asyncio
    async def test_execute_task_with_previous_results(
        self, sample_task, mock_task_result_success
    ):
        """Test task execution uses previous results."""
        prev_result = TaskResult(
            task_id=0,
            agent_role=AgentRole.ANALYST,
            success=True,
            output="Previous analysis data",
        )
        context = CollaborationContext(
            messages=[],
            task_results=[prev_result],
        )

        mock_result = MagicMock()
        mock_result.output = mock_task_result_success

        with patch(
            "agentic_patterns.multi_agent.ROLE_AGENTS",
            {AgentRole.RESEARCHER: MagicMock()},
        ) as mock_agents:
            mock_agents[AgentRole.RESEARCHER].run = AsyncMock(
                return_value=mock_result
            )

            await execute_task(sample_task, context)

            call_args = mock_agents[AgentRole.RESEARCHER].run.call_args
            prompt = call_args[0][0]
            assert "Previous analysis data" in prompt


class TestSynthesizeResults:
    """Test result synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_basic(self):
        """Test basic synthesis."""
        task_results = [
            TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Research findings here",
            ),
        ]

        mock_result = MagicMock()
        mock_result.output = "Synthesized output"

        with patch(
            "agentic_patterns.multi_agent.synthesizer_agent"
        ) as mock_synth:
            mock_synth.run = AsyncMock(return_value=mock_result)

            output = await synthesize_results("Test objective", task_results)

            assert output == "Synthesized output"
            mock_synth.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_filters_failed(self):
        """Test that synthesis only includes successful results."""
        task_results = [
            TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Good data",
            ),
            TaskResult(
                task_id=2,
                agent_role=AgentRole.ANALYST,
                success=False,
                output="Failed analysis",
            ),
        ]

        mock_result = MagicMock()
        mock_result.output = "Final output"

        with patch(
            "agentic_patterns.multi_agent.synthesizer_agent"
        ) as mock_synth:
            mock_synth.run = AsyncMock(return_value=mock_result)

            await synthesize_results("Test", task_results)

            call_args = mock_synth.run.call_args[0][0]
            assert "Good data" in call_args
            assert "Failed analysis" not in call_args


class TestRunCollaborativeTask:
    """Test full collaborative task execution."""

    @pytest.fixture
    def mock_plan(self):
        return SupervisorPlan(
            objective="Test collaboration",
            tasks=[
                DelegatedTask(
                    task_id=1,
                    assigned_to=AgentRole.RESEARCHER,
                    description="Research phase",
                ),
                DelegatedTask(
                    task_id=2,
                    assigned_to=AgentRole.WRITER,
                    description="Writing phase",
                ),
            ],
            reasoning="Two-phase approach",
        )

    @pytest.mark.asyncio
    async def test_collaborative_task_success(self, mock_plan):
        """Test successful collaborative task."""
        mock_supervisor_result = MagicMock()
        mock_supervisor_result.output = mock_plan

        mock_task_results = [
            TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Research done",
            ),
            TaskResult(
                task_id=2,
                agent_role=AgentRole.WRITER,
                success=True,
                output="Writing done",
            ),
        ]

        mock_synth_result = MagicMock()
        mock_synth_result.output = "Final combined output"

        call_count = 0

        async def mock_worker_run(prompt, deps=None):
            nonlocal call_count
            result = MagicMock()
            result.output = mock_task_results[call_count]
            call_count += 1
            return result

        with (
            patch(
                "agentic_patterns.multi_agent.supervisor_agent"
            ) as mock_supervisor,
            patch(
                "agentic_patterns.multi_agent.ROLE_AGENTS",
                {
                    AgentRole.RESEARCHER: MagicMock(),
                    AgentRole.WRITER: MagicMock(),
                },
            ) as mock_agents,
            patch(
                "agentic_patterns.multi_agent.synthesizer_agent"
            ) as mock_synth,
        ):
            mock_supervisor.run = AsyncMock(
                return_value=mock_supervisor_result
            )
            mock_agents[AgentRole.RESEARCHER].run = AsyncMock(
                side_effect=mock_worker_run
            )
            mock_agents[AgentRole.WRITER].run = AsyncMock(
                side_effect=mock_worker_run
            )
            mock_synth.run = AsyncMock(return_value=mock_synth_result)

            result = await run_collaborative_task("Test collaboration")

            assert result.success
            assert len(result.task_results) == 2
            assert result.final_output == "Final combined output"

    @pytest.mark.asyncio
    async def test_collaborative_task_max_tasks(self, mock_plan):
        """Test that max_tasks limits number of tasks."""
        mock_plan.tasks.append(
            DelegatedTask(
                task_id=3,
                assigned_to=AgentRole.REVIEWER,
                description="Review phase",
            )
        )

        mock_supervisor_result = MagicMock()
        mock_supervisor_result.output = mock_plan

        mock_task_result = TaskResult(
            task_id=1,
            agent_role=AgentRole.RESEARCHER,
            success=True,
            output="Done",
        )

        async def mock_worker_run(prompt, deps=None):
            result = MagicMock()
            result.output = mock_task_result
            return result

        mock_synth_result = MagicMock()
        mock_synth_result.output = "Output"

        with (
            patch(
                "agentic_patterns.multi_agent.supervisor_agent"
            ) as mock_supervisor,
            patch(
                "agentic_patterns.multi_agent.ROLE_AGENTS",
                {
                    AgentRole.RESEARCHER: MagicMock(),
                    AgentRole.WRITER: MagicMock(),
                },
            ) as mock_agents,
            patch(
                "agentic_patterns.multi_agent.synthesizer_agent"
            ) as mock_synth,
        ):
            mock_supervisor.run = AsyncMock(
                return_value=mock_supervisor_result
            )
            mock_agents[AgentRole.RESEARCHER].run = AsyncMock(
                side_effect=mock_worker_run
            )
            mock_agents[AgentRole.WRITER].run = AsyncMock(
                side_effect=mock_worker_run
            )
            mock_synth.run = AsyncMock(return_value=mock_synth_result)

            result = await run_collaborative_task(
                "Test collaboration", max_tasks=2
            )

            # Should only execute 2 tasks even though plan has 3
            assert len(result.task_results) == 2


class TestRunNetworkCollaboration:
    """Test network-style collaboration."""

    @pytest.mark.asyncio
    async def test_network_collaboration_basic(self):
        """Test basic network collaboration."""
        mock_task_result = TaskResult(
            task_id=1,
            agent_role=AgentRole.RESEARCHER,
            success=True,
            output="Network contribution",
        )

        async def mock_worker_run(prompt, deps=None):
            result = MagicMock()
            result.output = mock_task_result
            return result

        mock_synth_result = MagicMock()
        mock_synth_result.output = "Combined network output"

        with (
            patch(
                "agentic_patterns.multi_agent.ROLE_AGENTS",
                {
                    AgentRole.RESEARCHER: MagicMock(),
                    AgentRole.ANALYST: MagicMock(),
                },
            ) as mock_agents,
            patch(
                "agentic_patterns.multi_agent.synthesizer_agent"
            ) as mock_synth,
        ):
            mock_agents[AgentRole.RESEARCHER].run = AsyncMock(
                side_effect=mock_worker_run
            )
            mock_agents[AgentRole.ANALYST].run = AsyncMock(
                side_effect=mock_worker_run
            )
            mock_synth.run = AsyncMock(return_value=mock_synth_result)

            result = await run_network_collaboration(
                "Test network task",
                agents_to_consult=[AgentRole.RESEARCHER, AgentRole.ANALYST],
            )

            assert result.success
            assert len(result.task_results) == 2
            assert result.messages_exchanged == 0  # Network has no messages

    @pytest.mark.asyncio
    async def test_network_collaboration_parallel_execution(self):
        """Test that network agents run in parallel."""
        execution_order = []

        async def mock_researcher_run(prompt, deps=None):
            execution_order.append("researcher_start")
            await asyncio.sleep(0.1)
            execution_order.append("researcher_end")
            result = MagicMock()
            result.output = TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Research",
            )
            return result

        async def mock_analyst_run(prompt, deps=None):
            execution_order.append("analyst_start")
            await asyncio.sleep(0.1)
            execution_order.append("analyst_end")
            result = MagicMock()
            result.output = TaskResult(
                task_id=2,
                agent_role=AgentRole.ANALYST,
                success=True,
                output="Analysis",
            )
            return result

        mock_synth_result = MagicMock()
        mock_synth_result.output = "Output"

        with (
            patch(
                "agentic_patterns.multi_agent.ROLE_AGENTS",
                {
                    AgentRole.RESEARCHER: MagicMock(),
                    AgentRole.ANALYST: MagicMock(),
                },
            ) as mock_agents,
            patch(
                "agentic_patterns.multi_agent.synthesizer_agent"
            ) as mock_synth,
        ):
            mock_agents[AgentRole.RESEARCHER].run = mock_researcher_run
            mock_agents[AgentRole.ANALYST].run = mock_analyst_run
            mock_synth.run = AsyncMock(return_value=mock_synth_result)

            await run_network_collaboration(
                "Parallel test",
                agents_to_consult=[AgentRole.RESEARCHER, AgentRole.ANALYST],
            )

            # Both should start before either ends (parallel execution)
            assert execution_order[0] in ["researcher_start", "analyst_start"]
            assert execution_order[1] in ["researcher_start", "analyst_start"]

    @pytest.mark.asyncio
    async def test_network_collaboration_unknown_role(self):
        """Test network with unknown agent role."""
        mock_task_result = TaskResult(
            task_id=1,
            agent_role=AgentRole.RESEARCHER,
            success=True,
            output="Done",
        )

        async def mock_worker_run(prompt, deps=None):
            result = MagicMock()
            result.output = mock_task_result
            return result

        mock_synth_result = MagicMock()
        mock_synth_result.output = "Output"

        with (
            patch(
                "agentic_patterns.multi_agent.ROLE_AGENTS",
                {AgentRole.RESEARCHER: MagicMock()},
            ) as mock_agents,
            patch(
                "agentic_patterns.multi_agent.synthesizer_agent"
            ) as mock_synth,
        ):
            mock_agents[AgentRole.RESEARCHER].run = AsyncMock(
                side_effect=mock_worker_run
            )
            mock_synth.run = AsyncMock(return_value=mock_synth_result)

            # SUPERVISOR is not in ROLE_AGENTS
            result = await run_network_collaboration(
                "Test",
                agents_to_consult=[AgentRole.RESEARCHER, AgentRole.SUPERVISOR],
            )

            # Should have partial success
            assert not result.success
            failed = [r for r in result.task_results if not r.success]
            assert len(failed) == 1


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_find_result_by_role_found(self):
        from agentic_patterns.multi_agent import _find_result_by_role

        results = [
            TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Research data here",
            ),
            TaskResult(
                task_id=2,
                agent_role=AgentRole.ANALYST,
                success=True,
                output="Analysis data here",
            ),
        ]
        output = _find_result_by_role(results, "researcher")
        assert output == "Research data here"

    def test_find_result_by_role_not_found(self):
        from agentic_patterns.multi_agent import _find_result_by_role

        results = [
            TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Research data",
            ),
        ]
        output = _find_result_by_role(results, "writer")
        assert output == "No previous results found for this role."

    def test_find_result_by_role_empty_list(self):
        from agentic_patterns.multi_agent import _find_result_by_role

        output = _find_result_by_role([], "researcher")
        assert output == "No previous results found for this role."


class TestToolFunctions:
    """Test agent tool functions."""

    @pytest.fixture
    def mock_context_with_results(self):
        ctx = MagicMock()
        ctx.deps = CollaborationContext(
            messages=[],
            task_results=[
                TaskResult(
                    task_id=1,
                    agent_role=AgentRole.RESEARCHER,
                    success=True,
                    output="Researcher output",
                ),
                TaskResult(
                    task_id=2,
                    agent_role=AgentRole.ANALYST,
                    success=True,
                    output="Analyst output",
                ),
            ],
        )
        return ctx

    @pytest.mark.asyncio
    async def test_researcher_get_previous(self, mock_context_with_results):
        from agentic_patterns.multi_agent import researcher_get_previous

        result = await researcher_get_previous(
            mock_context_with_results, "analyst"
        )
        assert result == "Analyst output"

    @pytest.mark.asyncio
    async def test_analyst_get_previous(self, mock_context_with_results):
        from agentic_patterns.multi_agent import analyst_get_previous

        result = await analyst_get_previous(
            mock_context_with_results, "researcher"
        )
        assert result == "Researcher output"

    @pytest.mark.asyncio
    async def test_writer_get_previous(self, mock_context_with_results):
        from agentic_patterns.multi_agent import writer_get_previous

        result = await writer_get_previous(
            mock_context_with_results, "researcher"
        )
        assert result == "Researcher output"

    @pytest.mark.asyncio
    async def test_reviewer_get_previous(self, mock_context_with_results):
        from agentic_patterns.multi_agent import reviewer_get_previous

        result = await reviewer_get_previous(
            mock_context_with_results, "analyst"
        )
        assert result == "Analyst output"

    @pytest.mark.asyncio
    async def test_tool_not_found(self, mock_context_with_results):
        from agentic_patterns.multi_agent import researcher_get_previous

        result = await researcher_get_previous(
            mock_context_with_results, "writer"
        )
        assert result == "No previous results found for this role."


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_agent_message_empty_content(self):
        msg = AgentMessage(
            sender=AgentRole.RESEARCHER,
            recipient=AgentRole.SUPERVISOR,
            content="",
        )
        assert msg.content == ""

    def test_task_result_empty_output(self):
        result = TaskResult(
            task_id=1,
            agent_role=AgentRole.RESEARCHER,
            success=True,
            output="",
        )
        assert result.output == ""

    def test_delegated_task_status_transitions(self):
        task = DelegatedTask(
            task_id=1,
            assigned_to=AgentRole.RESEARCHER,
            description="Test",
            status=TaskStatus.PENDING,
        )
        assert task.status == TaskStatus.PENDING

        task.status = TaskStatus.IN_PROGRESS
        assert task.status == TaskStatus.IN_PROGRESS

        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED

    def test_collaboration_result_partial_success(self):
        task_results = [
            TaskResult(
                task_id=1,
                agent_role=AgentRole.RESEARCHER,
                success=True,
                output="Done",
            ),
            TaskResult(
                task_id=2,
                agent_role=AgentRole.ANALYST,
                success=False,
                output="Failed",
            ),
        ]
        result = CollaborationResult(
            objective="Mixed results",
            success=False,
            task_results=task_results,
            final_output="Partial output",
        )
        assert not result.success
        successful = [r for r in result.task_results if r.success]
        assert len(successful) == 1

    def test_supervisor_plan_all_same_role(self):
        """Test plan where all tasks go to same role."""
        tasks = [
            DelegatedTask(
                task_id=i,
                assigned_to=AgentRole.RESEARCHER,
                description=f"Research task {i}",
            )
            for i in range(1, 4)
        ]
        plan = SupervisorPlan(
            objective="Deep research",
            tasks=tasks,
            reasoning="All research tasks",
        )
        assert all(t.assigned_to == AgentRole.RESEARCHER for t in plan.tasks)

    def test_task_result_many_artifacts(self):
        result = TaskResult(
            task_id=1,
            agent_role=AgentRole.WRITER,
            success=True,
            output="Created documents",
            artifacts=[f"doc_{i}.md" for i in range(10)],
        )
        assert len(result.artifacts) == 10
