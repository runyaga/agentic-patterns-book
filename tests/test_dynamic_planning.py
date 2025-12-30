"""Tests for the Dynamic Planning Pattern implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns._utils import HistoryBuffer
from agentic_patterns.dynamic_planning import AgentContext
from agentic_patterns.dynamic_planning import DynamicPlan
from agentic_patterns.dynamic_planning import DynamicPlanningResult
from agentic_patterns.dynamic_planning import PlanningMetrics
from agentic_patterns.dynamic_planning import PlanningMode
from agentic_patterns.dynamic_planning import StepOutput
from agentic_patterns.dynamic_planning import create_dynamic_agent
from agentic_patterns.dynamic_planning import parse_agent_output
from agentic_patterns.dynamic_planning import run_episode
from agentic_patterns.dynamic_planning import step


class TestModels:
    """Test Pydantic model validation."""

    def test_planning_mode_enum(self):
        assert PlanningMode.DYNAMIC.value == "dynamic"
        assert PlanningMode.ALWAYS.value == "always"
        assert PlanningMode.NEVER.value == "never"

    def test_dynamic_plan(self):
        plan = DynamicPlan(
            content="Step 1: Do X\nStep 2: Do Y",
            created_at_step=5,
        )
        assert plan.content == "Step 1: Do X\nStep 2: Do Y"
        assert plan.created_at_step == 5
        assert plan.steps_since_creation == 0
        assert plan.age() == 0

    def test_dynamic_plan_age(self):
        plan = DynamicPlan(
            content="Plan",
            created_at_step=0,
            steps_since_creation=10,
        )
        assert plan.age() == 10

    def test_step_output_with_plan(self):
        output = StepOutput(
            plan="New plan here",
            action="move_forward",
            decided_to_plan=True,
        )
        assert output.plan == "New plan here"
        assert output.action == "move_forward"
        assert output.decided_to_plan

    def test_step_output_no_plan(self):
        output = StepOutput(
            action="move_forward",
        )
        assert output.plan is None
        assert output.action == "move_forward"
        assert not output.decided_to_plan

    def test_dynamic_planning_result(self):
        result = DynamicPlanningResult(
            goal="Complete task",
            total_steps=10,
            plans_generated=3,
            planning_frequency=0.3,
            actions=["a", "b", "c"],
            final_observation="Done",
        )
        assert result.goal == "Complete task"
        assert result.planning_frequency == 0.3
        assert len(result.actions) == 3
        assert result.success

    def test_planning_metrics(self):
        metrics = PlanningMetrics(
            total_plans=5,
            total_steps=20,
            planning_frequency=0.25,
            avg_plan_age_at_replan=4.0,
            plan_lengths=[100, 150, 120],
        )
        assert metrics.total_plans == 5
        assert metrics.planning_frequency == 0.25


class TestAgentContext:
    """Test AgentContext formatting."""

    def test_format_basic(self):
        ctx = AgentContext(
            goal="Find the key",
            observation="You are in a room",
            step=0,
        )
        prompt = ctx.format_for_prompt()
        assert "Goal: Find the key" in prompt
        assert "Step: 0" in prompt
        assert "Current Observation: You are in a room" in prompt
        assert "Current Plan: None" in prompt

    def test_format_with_plan(self):
        ctx = AgentContext(
            goal="Find the key",
            observation="You are in a room",
            current_plan=DynamicPlan(
                content="1. Search room\n2. Check drawers",
                created_at_step=0,
                steps_since_creation=3,
            ),
            step=3,
        )
        prompt = ctx.format_for_prompt()
        assert "Current Plan: 1. Search room" in prompt
        assert "Plan Age: 3 steps" in prompt

    def test_format_with_history(self):
        history = HistoryBuffer(max_size=5)
        history.add("Saw a door", "opened door")
        history.add("Entered room", "looked around")

        ctx = AgentContext(
            goal="Explore",
            observation="In new room",
            history=history,
            step=2,
        )
        prompt = ctx.format_for_prompt()
        assert "History:" in prompt
        assert "Saw a door" in prompt
        assert "opened door" in prompt

    def test_format_with_available_actions(self):
        ctx = AgentContext(
            goal="Navigate",
            observation="At intersection",
            step=0,
            available_actions=["go_north", "go_south", "go_east"],
        )
        prompt = ctx.format_for_prompt()
        assert "Available Actions:" in prompt
        assert "go_north" in prompt
        assert "go_south" in prompt


class TestParseAgentOutput:
    """Test output parsing."""

    def test_parse_with_plan(self):
        raw = "<plan>Step 1: Do X\nStep 2: Do Y</plan>\nmove_forward"
        output = parse_agent_output(raw, PlanningMode.DYNAMIC)

        assert output.plan == "Step 1: Do X\nStep 2: Do Y"
        assert output.action == "move_forward"
        assert output.decided_to_plan

    def test_parse_without_plan(self):
        raw = "move_forward"
        output = parse_agent_output(raw, PlanningMode.DYNAMIC)

        assert output.plan is None
        assert output.action == "move_forward"
        assert not output.decided_to_plan

    def test_parse_action_only_mode(self):
        raw = "just_an_action"
        output = parse_agent_output(raw, PlanningMode.NEVER)

        assert output.plan is None
        assert output.action == "just_an_action"
        assert not output.decided_to_plan

    def test_parse_always_mode_extracts_plan(self):
        raw = "<plan>Always plan</plan>\naction"
        output = parse_agent_output(raw, PlanningMode.ALWAYS)

        assert output.plan == "Always plan"
        assert output.decided_to_plan

    def test_parse_always_mode_treats_raw_as_plan(self):
        # In ALWAYS mode, if no <plan> tag, entire output is plan
        raw = "action_without_tags"
        output = parse_agent_output(raw, PlanningMode.ALWAYS)

        assert output.plan == "action_without_tags"
        assert output.decided_to_plan

    def test_parse_multiline_plan(self):
        raw = """<plan>
1. First step
2. Second step
3. Third step
</plan>
execute_step_1"""
        output = parse_agent_output(raw, PlanningMode.DYNAMIC)

        assert "First step" in output.plan
        assert "Third step" in output.plan
        assert output.action == "execute_step_1"

    def test_parse_whitespace_handling(self):
        raw = "  <plan>  plan content  </plan>  action  "
        output = parse_agent_output(raw, PlanningMode.DYNAMIC)

        assert output.plan == "plan content"
        assert output.action == "action"


class TestStep:
    """Test single step execution."""

    @pytest.mark.asyncio
    async def test_step_with_plan_output(self):
        """Test step when agent outputs a plan."""
        mock_result = MagicMock()
        mock_result.output = "<plan>New plan</plan>\ndo_action"

        ctx = AgentContext(
            goal="Test",
            observation="State",
            step=0,
        )

        with patch(
            "agentic_patterns.dynamic_planning._get_default_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_get_agent.return_value = mock_agent

            output = await step(ctx)

            assert output.plan == "New plan"
            assert output.action == "do_action"
            assert output.decided_to_plan
            mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_without_plan_output(self):
        """Test step when agent doesn't output a plan."""
        mock_result = MagicMock()
        mock_result.output = "just_action"

        ctx = AgentContext(
            goal="Test",
            observation="State",
            step=5,
        )

        with patch(
            "agentic_patterns.dynamic_planning._get_default_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_get_agent.return_value = mock_agent

            output = await step(ctx)

            assert output.plan is None
            assert output.action == "just_action"
            assert not output.decided_to_plan


class TestRunEpisode:
    """Test full episode execution."""

    @pytest.mark.asyncio
    async def test_run_episode_basic(self):
        """Test basic episode execution."""
        observations = ["obs0", "obs1", "obs2", "obs3"]
        obs_index = 0

        async def mock_get_obs():
            nonlocal obs_index
            obs = observations[min(obs_index, len(observations) - 1)]
            return obs

        actions_executed = []

        async def mock_execute(action):
            nonlocal obs_index
            actions_executed.append(action)
            obs_index += 1

        # Mock agent that plans on step 0, then just acts
        call_count = 0

        async def mock_agent_run(prompt):
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.output = "<plan>Do stuff</plan>\naction_0"
            else:
                result.output = f"action_{call_count}"
            call_count += 1
            return result

        with patch(
            "agentic_patterns.dynamic_planning.create_dynamic_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_agent_run)
            mock_create.return_value = mock_agent

            result = await run_episode(
                goal="Test goal",
                get_observation=mock_get_obs,
                execute_action=mock_execute,
                max_steps=3,
            )

            assert result.total_steps == 3
            assert result.plans_generated == 1
            assert result.planning_frequency == pytest.approx(1 / 3)
            assert len(result.actions) == 3

    @pytest.mark.asyncio
    async def test_run_episode_tracks_metrics(self):
        """Test that episode tracks planning metrics."""

        async def mock_get_obs():
            return "observation"

        async def mock_execute(action):
            pass

        # Agent plans every other step
        call_count = 0

        async def mock_agent_run(prompt):
            nonlocal call_count
            result = MagicMock()
            if call_count % 2 == 0:
                result.output = f"<plan>Plan {call_count}</plan>\naction"
            else:
                result.output = "action"
            call_count += 1
            return result

        with patch(
            "agentic_patterns.dynamic_planning.create_dynamic_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_agent_run)
            mock_create.return_value = mock_agent

            result = await run_episode(
                goal="Test",
                get_observation=mock_get_obs,
                execute_action=mock_execute,
                max_steps=4,
            )

            # Plans at step 0, 2 (even steps)
            assert result.plans_generated == 2
            assert result.planning_frequency == 0.5


class TestCreateDynamicAgent:
    """Test agent creation."""

    def test_create_with_dynamic_mode(self):
        agent_path = "agentic_patterns.dynamic_planning.Agent"
        model_path = "agentic_patterns.dynamic_planning.get_model"
        with patch(agent_path) as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            with patch(model_path) as mock_model:
                mock_model.return_value = MagicMock()
                agent = create_dynamic_agent(mode=PlanningMode.DYNAMIC)
                assert agent is not None
                mock_agent_cls.assert_called_once()

    def test_create_with_always_mode(self):
        agent_path = "agentic_patterns.dynamic_planning.Agent"
        model_path = "agentic_patterns.dynamic_planning.get_model"
        with patch(agent_path) as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            with patch(model_path) as mock_model:
                mock_model.return_value = MagicMock()
                agent = create_dynamic_agent(mode=PlanningMode.ALWAYS)
                assert agent is not None

    def test_create_with_never_mode(self):
        agent_path = "agentic_patterns.dynamic_planning.Agent"
        model_path = "agentic_patterns.dynamic_planning.get_model"
        with patch(agent_path) as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            with patch(model_path) as mock_model:
                mock_model.return_value = MagicMock()
                agent = create_dynamic_agent(mode=PlanningMode.NEVER)
                assert agent is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_dynamic_plan_zero_age(self):
        plan = DynamicPlan(content="New", created_at_step=0)
        assert plan.age() == 0

    def test_step_output_empty_action(self):
        output = StepOutput(action="")
        assert output.action == ""

    def test_agent_context_empty_history(self):
        ctx = AgentContext(
            goal="Test",
            observation="State",
            step=0,
        )
        prompt = ctx.format_for_prompt()
        assert "History:" not in prompt  # Empty history not shown

    def test_parse_empty_plan_tags(self):
        raw = "<plan></plan>\naction"
        output = parse_agent_output(raw, PlanningMode.DYNAMIC)
        assert output.plan == ""
        assert output.decided_to_plan

    def test_planning_result_zero_steps(self):
        result = DynamicPlanningResult(
            goal="Empty",
            total_steps=0,
            plans_generated=0,
            planning_frequency=0.0,
            actions=[],
            final_observation="",
        )
        assert result.planning_frequency == 0.0

    @pytest.mark.asyncio
    async def test_run_episode_max_steps_zero(self):
        """Test episode with max_steps=0."""

        async def mock_get_obs():
            return "obs"

        async def mock_execute(action):
            pass

        with patch(
            "agentic_patterns.dynamic_planning.create_dynamic_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock()
            mock_create.return_value = mock_agent

            result = await run_episode(
                goal="Test",
                get_observation=mock_get_obs,
                execute_action=mock_execute,
                max_steps=0,
            )

            assert result.total_steps == 0
            assert result.plans_generated == 0
            assert len(result.actions) == 0


class TestIntegration:
    """Integration tests for the full dynamic planning flow."""

    @pytest.mark.asyncio
    async def test_full_episode_with_replanning(self):
        """Test episode where agent replans mid-execution."""
        observations = [
            "Start state",
            "After action 1",
            "Unexpected obstacle",  # Should trigger replan
            "After action 3",
            "Goal reached",
        ]
        obs_idx = 0

        async def get_obs():
            nonlocal obs_idx
            return observations[min(obs_idx, len(observations) - 1)]

        executed = []

        async def execute(action):
            nonlocal obs_idx
            executed.append(action)
            obs_idx += 1

        # Agent plans at step 0, replans at step 2 (obstacle), acts otherwise
        call_count = 0

        async def mock_run(prompt):
            nonlocal call_count
            result = MagicMock()

            if call_count == 0:
                # Initial plan
                result.output = "<plan>1. Move forward\n2. Turn</plan>\nmove"
            elif call_count == 2 and "obstacle" in prompt.lower():
                # Replan due to obstacle
                result.output = "<plan>Revised: go around</plan>\ndetour"
            else:
                result.output = f"action_{call_count}"

            call_count += 1
            return result

        with patch(
            "agentic_patterns.dynamic_planning.create_dynamic_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_create.return_value = mock_agent

            result = await run_episode(
                goal="Reach the destination",
                get_observation=get_obs,
                execute_action=execute,
                max_steps=4,
            )

            assert result.total_steps == 4
            assert result.plans_generated >= 1
            assert len(executed) == 4

    @pytest.mark.asyncio
    async def test_episode_modes_comparison(self):
        """Compare behavior across DYNAMIC, ALWAYS, and NEVER modes."""

        async def get_obs():
            return "state"

        async def execute(action):
            pass

        results = {}

        for mode in [
            PlanningMode.DYNAMIC,
            PlanningMode.ALWAYS,
            PlanningMode.NEVER,
        ]:
            counter = {"count": 0}

            async def mock_run(prompt, m=mode, ctr=counter):
                result = MagicMock()
                if m == PlanningMode.ALWAYS:
                    result.output = f"<plan>Plan {ctr['count']}</plan>\naction"
                elif m == PlanningMode.NEVER:
                    result.output = "action"
                else:  # DYNAMIC - plan every other step
                    if ctr["count"] % 2 == 0:
                        c = ctr["count"]
                        result.output = f"<plan>Plan {c}</plan>\naction"
                    else:
                        result.output = "action"
                ctr["count"] += 1
                return result

            with patch(
                "agentic_patterns.dynamic_planning.create_dynamic_agent"
            ) as mock_create:
                mock_agent = MagicMock()
                mock_agent.run = AsyncMock(side_effect=mock_run)
                mock_create.return_value = mock_agent

                result = await run_episode(
                    goal="Test",
                    get_observation=get_obs,
                    execute_action=execute,
                    max_steps=4,
                    mode=mode,
                )
                results[mode] = result

        # ALWAYS should have highest planning frequency
        assert results[PlanningMode.ALWAYS].planning_frequency == 1.0
        # DYNAMIC should be between 0 and 1
        assert 0 < results[PlanningMode.DYNAMIC].planning_frequency < 1
        # NEVER should have 0 planning frequency
        assert results[PlanningMode.NEVER].planning_frequency == 0.0

    @pytest.mark.asyncio
    async def test_history_accumulation(self):
        """Test that history accumulates correctly during episode."""
        obs_sequence = ["obs_0", "obs_1", "obs_2", "obs_3"]
        obs_idx = 0

        async def get_obs():
            nonlocal obs_idx
            return obs_sequence[min(obs_idx, len(obs_sequence) - 1)]

        async def execute(action):
            nonlocal obs_idx
            obs_idx += 1

        prompts_received = []

        async def mock_run(prompt):
            prompts_received.append(prompt)
            result = MagicMock()
            result.output = "action"
            return result

        with patch(
            "agentic_patterns.dynamic_planning.create_dynamic_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_create.return_value = mock_agent

            await run_episode(
                goal="Test history",
                get_observation=get_obs,
                execute_action=execute,
                max_steps=3,
                history_size=10,
            )

        # Later prompts should contain history from earlier steps
        assert len(prompts_received) == 3
        # Third prompt should contain history
        assert "History:" in prompts_received[2]
        assert "obs_0" in prompts_received[2]

    @pytest.mark.asyncio
    async def test_plan_aging(self):
        """Test that plan age is tracked and included in context."""

        async def get_obs():
            return "state"

        async def execute(action):
            pass

        prompts_received = []

        async def mock_run(prompt):
            prompts_received.append(prompt)
            result = MagicMock()
            # Only plan on first step
            if len(prompts_received) == 1:
                result.output = "<plan>My plan</plan>\naction"
            else:
                result.output = "action"
            return result

        with patch(
            "agentic_patterns.dynamic_planning.create_dynamic_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_create.return_value = mock_agent

            await run_episode(
                goal="Test aging",
                get_observation=get_obs,
                execute_action=execute,
                max_steps=4,
            )

        # Step 3 should show plan age of 3 (created at step 0, now at step 3)
        assert "Plan Age: 3 steps" in prompts_received[3]

    @pytest.mark.asyncio
    async def test_available_actions_in_context(self):
        """Test that available actions are passed to agent."""

        async def get_obs():
            return "state"

        async def execute(action):
            pass

        prompt_received = None

        async def mock_run(prompt):
            nonlocal prompt_received
            prompt_received = prompt
            result = MagicMock()
            result.output = "go_north"
            return result

        with patch(
            "agentic_patterns.dynamic_planning.create_dynamic_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_create.return_value = mock_agent

            await run_episode(
                goal="Navigate",
                get_observation=get_obs,
                execute_action=execute,
                available_actions=["go_north", "go_south", "wait"],
                max_steps=1,
            )

        assert "Available Actions:" in prompt_received
        assert "go_north" in prompt_received
        assert "go_south" in prompt_received
        assert "wait" in prompt_received

    @pytest.mark.asyncio
    async def test_goldilocks_frequency_tracking(self):
        """
        Test the 'Goldilocks' frequency concept from the paper.

        Verify that planning frequency is correctly calculated and
        represents the ratio of planning decisions to total steps.
        """

        async def get_obs():
            return "state"

        async def execute(action):
            pass

        # Agent plans on steps 0, 3, 6 (every 3rd step)
        step_counter = {"count": 0}

        async def mock_run(prompt, ctr=step_counter):
            result = MagicMock()
            if ctr["count"] % 3 == 0:
                result.output = f"<plan>Plan at {ctr['count']}</plan>\naction"
            else:
                result.output = "action"
            ctr["count"] += 1
            return result

        with patch(
            "agentic_patterns.dynamic_planning.create_dynamic_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_create.return_value = mock_agent

            result = await run_episode(
                goal="Test Goldilocks",
                get_observation=get_obs,
                execute_action=execute,
                max_steps=9,
            )

            # Plans at steps 0, 3, 6 = 3 plans out of 9 steps
            assert result.plans_generated == 3
            assert result.total_steps == 9
            assert result.planning_frequency == pytest.approx(1 / 3)
