"""Tests for the Agent Marketplace Pattern (The Agora)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import uuid4

import pytest

from agentic_patterns.agent_marketplace import AgentBid
from agentic_patterns.agent_marketplace import AgentCapability
from agentic_patterns.agent_marketplace import AgoraState
from agentic_patterns.agent_marketplace import BidResponse
from agentic_patterns.agent_marketplace import CollectBidsNode
from agentic_patterns.agent_marketplace import ExecuteTaskNode
from agentic_patterns.agent_marketplace import PostRFPNode
from agentic_patterns.agent_marketplace import SelectWinnerNode
from agentic_patterns.agent_marketplace import TaskResult
from agentic_patterns.agent_marketplace import TaskRFP
from agentic_patterns.agent_marketplace import create_bidder_agent
from agentic_patterns.agent_marketplace import run_marketplace_task


class TestModels:
    """Test Pydantic model validation."""

    def test_agent_capability_valid(self):
        cap = AgentCapability(
            agent_id="test-agent",
            name="Test Agent",
            skills=["skill_a", "skill_b"],
            description="A test agent",
        )
        assert cap.agent_id == "test-agent"
        assert len(cap.skills) == 2

    def test_task_rfp_defaults(self):
        rfp = TaskRFP(requirement="Do something")
        assert rfp.id is not None
        assert rfp.required_skills == []
        assert rfp.context == {}
        assert rfp.min_confidence == 0.5

    def test_task_rfp_with_skills(self):
        rfp = TaskRFP(
            requirement="Summarize text",
            required_skills=["brevity", "clarity"],
            min_confidence=0.7,
        )
        assert len(rfp.required_skills) == 2
        assert rfp.min_confidence == 0.7

    def test_agent_bid_confidence_bounds(self):
        rfp_id = uuid4()
        bid = AgentBid(
            rfp_id=rfp_id,
            agent_id="test",
            confidence=0.85,
            proposal="My approach",
        )
        assert bid.confidence == 0.85

    def test_agent_bid_confidence_zero(self):
        rfp_id = uuid4()
        bid = AgentBid(
            rfp_id=rfp_id,
            agent_id="test",
            confidence=0.0,
            proposal="Low confidence",
        )
        assert bid.confidence == 0.0

    def test_agent_bid_confidence_one(self):
        rfp_id = uuid4()
        bid = AgentBid(
            rfp_id=rfp_id,
            agent_id="test",
            confidence=1.0,
            proposal="Full confidence",
        )
        assert bid.confidence == 1.0

    def test_agent_bid_invalid_confidence_high(self):
        with pytest.raises(ValueError):
            AgentBid(
                rfp_id=uuid4(),
                agent_id="test",
                confidence=1.5,
                proposal="Invalid",
            )

    def test_agent_bid_invalid_confidence_low(self):
        with pytest.raises(ValueError):
            AgentBid(
                rfp_id=uuid4(),
                agent_id="test",
                confidence=-0.1,
                proposal="Invalid",
            )

    def test_task_result_success(self):
        result = TaskResult(
            rfp_id=uuid4(),
            agent_id="winner",
            success=True,
            output="Task completed successfully",
        )
        assert result.success
        assert result.error_message is None

    def test_task_result_failure(self):
        result = TaskResult(
            rfp_id=uuid4(),
            agent_id="loser",
            success=False,
            output="",
            error_message="Something went wrong",
        )
        assert not result.success
        assert result.error_message == "Something went wrong"

    def test_bid_response_valid(self):
        response = BidResponse(
            will_bid=True,
            confidence=0.9,
            proposal="I can do this",
            reasoning="Skills match well",
        )
        assert response.will_bid
        assert response.confidence == 0.9

    def test_bid_response_no_bid(self):
        response = BidResponse(
            will_bid=False,
            confidence=0.2,
            proposal="",
            reasoning="Skills don't match",
        )
        assert not response.will_bid


class TestBidderAgent:
    """Test bidder agent creation."""

    def test_create_bidder_agent(self):
        cap = AgentCapability(
            agent_id="test",
            name="Test Agent",
            skills=["skill_a"],
            description="Test description",
        )
        agent = create_bidder_agent(cap)
        assert agent is not None


class TestAgoraState:
    """Test AgoraState dataclass."""

    def test_agora_state_defaults(self):
        rfp = TaskRFP(requirement="Test")
        state = AgoraState(
            rfp=rfp,
            registered_bidders=[],
            bidder_agents={},
        )
        assert state.bids == []
        assert state.winning_bid is None
        assert state.bid_timeout_seconds == 5.0


# Fixtures for testing
@pytest.fixture
def sample_rfp():
    return TaskRFP(
        requirement="Test task",
        required_skills=["skill_a"],
        min_confidence=0.5,
    )


@pytest.fixture
def sample_capabilities():
    return [
        AgentCapability(
            agent_id="agent_a",
            name="Agent A",
            skills=["skill_a", "skill_b"],
            description="Good at skill_a",
        ),
        AgentCapability(
            agent_id="agent_b",
            name="Agent B",
            skills=["skill_c"],
            description="Good at skill_c",
        ),
    ]


@pytest.fixture
def mock_bid_response_will_bid():
    return BidResponse(
        will_bid=True,
        confidence=0.8,
        proposal="I can do this task",
        reasoning="Skills match",
    )


@pytest.fixture
def mock_bid_response_wont_bid():
    return BidResponse(
        will_bid=False,
        confidence=0.2,
        proposal="",
        reasoning="Skills don't match",
    )


class TestPostRFPNode:
    """Test PostRFPNode behavior."""

    @pytest.mark.asyncio
    async def test_no_bidders_returns_error(self, sample_rfp):
        """Empty bidder list returns error result."""
        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=[],
            bidder_agents={},
        )
        ctx = MagicMock()
        ctx.state = state

        node = PostRFPNode()
        result = await node.run(ctx)

        # Should return End with error
        assert hasattr(result, "data")
        assert not result.data.success
        assert "No bidders" in result.data.error_message

    @pytest.mark.asyncio
    async def test_with_bidders_proceeds(
        self, sample_rfp, sample_capabilities
    ):
        """With bidders, proceeds to CollectBidsNode."""
        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=sample_capabilities,
            bidder_agents={},
        )
        ctx = MagicMock()
        ctx.state = state

        node = PostRFPNode()
        result = await node.run(ctx)

        assert isinstance(result, CollectBidsNode)


class TestCollectBidsNode:
    """Test CollectBidsNode behavior."""

    @pytest.mark.asyncio
    async def test_collects_bids_from_willing_agents(
        self,
        sample_rfp,
        sample_capabilities,
        mock_bid_response_will_bid,
        mock_bid_response_wont_bid,
    ):
        """Only agents that will_bid are included in bids."""
        mock_agent_a = MagicMock()
        mock_result_a = MagicMock()
        mock_result_a.output = mock_bid_response_will_bid
        mock_agent_a.run = AsyncMock(return_value=mock_result_a)

        mock_agent_b = MagicMock()
        mock_result_b = MagicMock()
        mock_result_b.output = mock_bid_response_wont_bid
        mock_agent_b.run = AsyncMock(return_value=mock_result_b)

        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=sample_capabilities,
            bidder_agents={
                "agent_a": mock_agent_a,
                "agent_b": mock_agent_b,
            },
        )
        ctx = MagicMock()
        ctx.state = state

        node = CollectBidsNode()
        result = await node.run(ctx)

        assert isinstance(result, SelectWinnerNode)
        assert len(state.bids) == 1
        assert state.bids[0].agent_id == "agent_a"

    @pytest.mark.asyncio
    async def test_handles_missing_agent(
        self, sample_rfp, sample_capabilities
    ):
        """Missing agent in bidder_agents is skipped."""
        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=sample_capabilities,
            bidder_agents={},  # No agents registered
        )
        ctx = MagicMock()
        ctx.state = state

        node = CollectBidsNode()
        result = await node.run(ctx)

        assert isinstance(result, SelectWinnerNode)
        assert len(state.bids) == 0


class TestSelectWinnerNode:
    """Test SelectWinnerNode behavior."""

    @pytest.mark.asyncio
    async def test_no_valid_bids_returns_error(self, sample_rfp):
        """No bids meeting min_confidence returns error."""
        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=[],
            bidder_agents={},
            bids=[],
        )
        ctx = MagicMock()
        ctx.state = state

        node = SelectWinnerNode()
        result = await node.run(ctx)

        assert hasattr(result, "data")
        assert not result.data.success
        assert "No bids" in result.data.error_message

    @pytest.mark.asyncio
    async def test_filters_low_confidence_bids(
        self, sample_rfp, sample_capabilities
    ):
        """Bids below min_confidence are filtered out."""
        rfp = TaskRFP(
            requirement="Test",
            required_skills=["skill_a"],
            min_confidence=0.7,
        )

        low_bid = AgentBid(
            rfp_id=rfp.id,
            agent_id="agent_a",
            confidence=0.5,  # Below threshold
            proposal="Low confidence",
        )

        state = AgoraState(
            rfp=rfp,
            registered_bidders=sample_capabilities,
            bidder_agents={},
            bids=[low_bid],
        )
        ctx = MagicMock()
        ctx.state = state

        node = SelectWinnerNode()
        result = await node.run(ctx)

        assert hasattr(result, "data")
        assert not result.data.success

    @pytest.mark.asyncio
    async def test_selects_highest_score(
        self, sample_rfp, sample_capabilities
    ):
        """Winner is selected by weighted score."""
        high_bid = AgentBid(
            rfp_id=sample_rfp.id,
            agent_id="agent_a",  # Has skill_a (matches requirement)
            confidence=0.8,
            proposal="Good match",
        )
        low_bid = AgentBid(
            rfp_id=sample_rfp.id,
            agent_id="agent_b",  # Has skill_c (no match)
            confidence=0.9,  # Higher confidence but no skill match
            proposal="No skill match",
        )

        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=sample_capabilities,
            bidder_agents={},
            bids=[high_bid, low_bid],
        )
        ctx = MagicMock()
        ctx.state = state

        node = SelectWinnerNode()
        result = await node.run(ctx)

        assert isinstance(result, ExecuteTaskNode)
        # agent_a should win due to skill match despite lower confidence
        assert state.winning_bid.agent_id == "agent_a"


class TestExecuteTaskNode:
    """Test ExecuteTaskNode behavior."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, sample_rfp, sample_capabilities):
        """Successful execution returns success result."""
        winning_bid = AgentBid(
            rfp_id=sample_rfp.id,
            agent_id="agent_a",
            confidence=0.9,
            proposal="Will execute",
        )

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "Task completed!"
        mock_agent.run = AsyncMock(return_value=mock_result)

        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=sample_capabilities,
            bidder_agents={"agent_a": mock_agent},
            winning_bid=winning_bid,
        )
        ctx = MagicMock()
        ctx.state = state

        node = ExecuteTaskNode()
        result = await node.run(ctx)

        assert result.data.success
        assert result.data.agent_id == "agent_a"
        assert "Task completed!" in result.data.output

    @pytest.mark.asyncio
    async def test_execution_failure(self, sample_rfp, sample_capabilities):
        """Execution error is captured in result."""
        winning_bid = AgentBid(
            rfp_id=sample_rfp.id,
            agent_id="agent_a",
            confidence=0.9,
            proposal="Will fail",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("Execution error"))

        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=sample_capabilities,
            bidder_agents={"agent_a": mock_agent},
            winning_bid=winning_bid,
        )
        ctx = MagicMock()
        ctx.state = state

        node = ExecuteTaskNode()
        result = await node.run(ctx)

        assert not result.data.success
        assert "Execution error" in result.data.error_message

    @pytest.mark.asyncio
    async def test_no_winning_bid(self, sample_rfp):
        """No winning bid returns error."""
        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=[],
            bidder_agents={},
            winning_bid=None,
        )
        ctx = MagicMock()
        ctx.state = state

        node = ExecuteTaskNode()
        result = await node.run(ctx)

        assert not result.data.success
        assert "No winning bid" in result.data.error_message

    @pytest.mark.asyncio
    async def test_agent_not_found(self, sample_rfp):
        """Missing agent returns error."""
        winning_bid = AgentBid(
            rfp_id=sample_rfp.id,
            agent_id="missing_agent",
            confidence=0.9,
            proposal="Agent gone",
        )

        state = AgoraState(
            rfp=sample_rfp,
            registered_bidders=[],
            bidder_agents={},  # Agent not in dict
            winning_bid=winning_bid,
        )
        ctx = MagicMock()
        ctx.state = state

        node = ExecuteTaskNode()
        result = await node.run(ctx)

        assert not result.data.success
        assert "not found" in result.data.error_message


class TestRunMarketplaceTask:
    """Test the full marketplace flow."""

    @pytest.mark.asyncio
    async def test_empty_bidders_returns_error(self):
        """Empty bidder list returns error."""
        rfp = TaskRFP(requirement="Test task")
        result = await run_marketplace_task(rfp, [])

        assert not result.success
        assert "No bidders" in result.error_message

    @pytest.mark.asyncio
    async def test_full_flow_success(self):
        """Full flow with mocked agents succeeds."""
        rfp = TaskRFP(
            requirement="Summarize text",
            required_skills=["brevity"],
        )

        cap = AgentCapability(
            agent_id="summarizer",
            name="Summarizer",
            skills=["brevity", "clarity"],
            description="Good at summaries",
        )

        # Mock agent responses
        bid_response = BidResponse(
            will_bid=True,
            confidence=0.9,
            proposal="Will summarize concisely",
            reasoning="Skills match perfectly",
        )

        mock_agent = MagicMock()
        mock_bid_result = MagicMock()
        mock_bid_result.output = bid_response
        mock_exec_result = MagicMock()
        mock_exec_result.output = "Summary: Key points..."

        mock_agent.run = AsyncMock(
            side_effect=[mock_bid_result, mock_exec_result]
        )

        with patch(
            "agentic_patterns.agent_marketplace.agora_graph"
        ) as mock_graph:
            # Set up mock graph result
            mock_graph_result = MagicMock()
            mock_graph_result.output = TaskResult(
                rfp_id=rfp.id,
                agent_id="summarizer",
                success=True,
                output="Summary: Key points...",
            )
            mock_graph.run = AsyncMock(return_value=mock_graph_result)

            result = await run_marketplace_task(rfp, [(cap, mock_agent)])

            assert result.success
            assert result.agent_id == "summarizer"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_skills_list(self):
        cap = AgentCapability(
            agent_id="test",
            name="Test",
            skills=[],
            description="No skills",
        )
        assert len(cap.skills) == 0

    def test_empty_context(self):
        rfp = TaskRFP(
            requirement="Test",
            context={},
        )
        assert rfp.context == {}

    def test_rfp_with_context(self):
        rfp = TaskRFP(
            requirement="Test",
            context={"key": "value", "num": 42},
        )
        assert rfp.context["key"] == "value"
        assert rfp.context["num"] == 42

    def test_min_confidence_zero(self):
        rfp = TaskRFP(
            requirement="Test",
            min_confidence=0.0,
        )
        assert rfp.min_confidence == 0.0

    def test_min_confidence_one(self):
        rfp = TaskRFP(
            requirement="Test",
            min_confidence=1.0,
        )
        assert rfp.min_confidence == 1.0


class TestSelectionStrategies:
    """Test selection strategy implementations."""

    @pytest.fixture
    def sample_bids(self):
        rfp_id = uuid4()
        return [
            AgentBid(
                rfp_id=rfp_id,
                agent_id="agent_a",
                confidence=0.7,
                proposal="Approach A",
            ),
            AgentBid(
                rfp_id=rfp_id,
                agent_id="agent_b",
                confidence=0.9,
                proposal="Approach B",
            ),
            AgentBid(
                rfp_id=rfp_id,
                agent_id="agent_c",
                confidence=0.6,
                proposal="Approach C",
            ),
        ]

    @pytest.fixture
    def sample_caps_dict(self):
        return {
            "agent_a": AgentCapability(
                agent_id="agent_a",
                name="Agent A",
                skills=["skill_a", "skill_b"],
                description="Good at A",
            ),
            "agent_b": AgentCapability(
                agent_id="agent_b",
                name="Agent B",
                skills=["skill_c"],
                description="Good at B",
            ),
            "agent_c": AgentCapability(
                agent_id="agent_c",
                name="Agent C",
                skills=["skill_a", "skill_b", "skill_c"],
                description="Good at C",
            ),
        }


class TestHighestConfidenceStrategy:
    """Test HighestConfidenceStrategy."""

    @pytest.mark.asyncio
    async def test_selects_highest_confidence(self):
        from agentic_patterns.agent_marketplace import (
            HighestConfidenceStrategy,
        )

        strategy = HighestConfidenceStrategy()
        rfp_id = uuid4()
        bids = [
            AgentBid(
                rfp_id=rfp_id, agent_id="a", confidence=0.7, proposal=""
            ),
            AgentBid(
                rfp_id=rfp_id, agent_id="b", confidence=0.9, proposal=""
            ),
            AgentBid(
                rfp_id=rfp_id, agent_id="c", confidence=0.6, proposal=""
            ),
        ]
        rfp = TaskRFP(requirement="Test")

        winner = await strategy.select(bids, rfp, {})

        assert winner is not None
        assert winner.agent_id == "b"

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_bids(self):
        from agentic_patterns.agent_marketplace import (
            HighestConfidenceStrategy,
        )

        strategy = HighestConfidenceStrategy()
        rfp = TaskRFP(requirement="Test")

        winner = await strategy.select([], rfp, {})

        assert winner is None


class TestBestSkillMatchStrategy:
    """Test BestSkillMatchStrategy."""

    @pytest.mark.asyncio
    async def test_prefers_skill_match_over_confidence(self):
        from agentic_patterns.agent_marketplace import BestSkillMatchStrategy

        strategy = BestSkillMatchStrategy()
        rfp_id = uuid4()
        bids = [
            AgentBid(
                rfp_id=rfp_id, agent_id="a", confidence=0.9, proposal=""
            ),
            AgentBid(
                rfp_id=rfp_id, agent_id="b", confidence=0.6, proposal=""
            ),
        ]
        caps = {
            "a": AgentCapability(
                agent_id="a", name="A", skills=["x"], description=""
            ),
            "b": AgentCapability(
                agent_id="b", name="B", skills=["skill_a"], description=""
            ),
        }
        rfp = TaskRFP(requirement="Test", required_skills=["skill_a"])

        winner = await strategy.select(bids, rfp, caps)

        assert winner is not None
        assert winner.agent_id == "b"  # Better skill match

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_bids(self):
        from agentic_patterns.agent_marketplace import BestSkillMatchStrategy

        strategy = BestSkillMatchStrategy()
        rfp = TaskRFP(requirement="Test")

        winner = await strategy.select([], rfp, {})

        assert winner is None

    @pytest.mark.asyncio
    async def test_no_required_skills_returns_first(self):
        from agentic_patterns.agent_marketplace import BestSkillMatchStrategy

        strategy = BestSkillMatchStrategy()
        rfp_id = uuid4()
        bids = [
            AgentBid(
                rfp_id=rfp_id, agent_id="a", confidence=0.7, proposal=""
            ),
            AgentBid(
                rfp_id=rfp_id, agent_id="b", confidence=0.9, proposal=""
            ),
        ]
        # No required_skills means all skill scores are 0
        rfp = TaskRFP(requirement="Test", required_skills=[])

        winner = await strategy.select(bids, rfp, {})

        # With all zeros, max returns first element
        assert winner is not None


class TestWeightedScoreStrategy:
    """Test WeightedScoreStrategy."""

    @pytest.mark.asyncio
    async def test_default_weights(self):
        from agentic_patterns.agent_marketplace import WeightedScoreStrategy

        strategy = WeightedScoreStrategy()
        assert strategy.confidence_weight == 0.6
        assert strategy.skill_weight == 0.4

    @pytest.mark.asyncio
    async def test_custom_weights(self):
        from agentic_patterns.agent_marketplace import WeightedScoreStrategy

        strategy = WeightedScoreStrategy(
            confidence_weight=0.2, skill_weight=0.8
        )
        rfp_id = uuid4()
        # With high skill weight, skill match matters more
        bids = [
            AgentBid(
                rfp_id=rfp_id, agent_id="a", confidence=0.9, proposal=""
            ),
            AgentBid(
                rfp_id=rfp_id, agent_id="b", confidence=0.5, proposal=""
            ),
        ]
        caps = {
            "a": AgentCapability(
                agent_id="a", name="A", skills=["x"], description=""
            ),
            "b": AgentCapability(
                agent_id="b", name="B", skills=["skill_a"], description=""
            ),
        }
        rfp = TaskRFP(requirement="Test", required_skills=["skill_a"])

        winner = await strategy.select(bids, rfp, caps)

        # With 80% skill weight, b should win despite lower confidence
        assert winner is not None
        assert winner.agent_id == "b"

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_bids(self):
        from agentic_patterns.agent_marketplace import WeightedScoreStrategy

        strategy = WeightedScoreStrategy()
        rfp = TaskRFP(requirement="Test")

        winner = await strategy.select([], rfp, {})

        assert winner is None

    @pytest.mark.asyncio
    async def test_no_skills_uses_confidence_only(self):
        from agentic_patterns.agent_marketplace import WeightedScoreStrategy

        strategy = WeightedScoreStrategy()
        rfp_id = uuid4()
        bids = [
            AgentBid(
                rfp_id=rfp_id, agent_id="a", confidence=0.7, proposal=""
            ),
            AgentBid(
                rfp_id=rfp_id, agent_id="b", confidence=0.9, proposal=""
            ),
        ]
        # No required_skills means only confidence matters
        rfp = TaskRFP(requirement="Test", required_skills=[])

        winner = await strategy.select(bids, rfp, {})

        assert winner is not None
        assert winner.agent_id == "b"  # Higher confidence wins


class TestAgentJudgmentStrategy:
    """Test AgentJudgmentStrategy."""

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_bids(self):
        from agentic_patterns.agent_marketplace import AgentJudgmentStrategy

        strategy = AgentJudgmentStrategy()
        rfp = TaskRFP(requirement="Test")

        winner = await strategy.select([], rfp, {})

        assert winner is None

    @pytest.mark.asyncio
    async def test_uses_selector_agent(self):
        from agentic_patterns.agent_marketplace import AgentJudgmentStrategy
        from agentic_patterns.agent_marketplace import JudgmentResult

        strategy = AgentJudgmentStrategy()
        rfp_id = uuid4()
        bids = [
            AgentBid(
                rfp_id=rfp_id, agent_id="a", confidence=0.7, proposal="Plan A"
            ),
            AgentBid(
                rfp_id=rfp_id, agent_id="b", confidence=0.9, proposal="Plan B"
            ),
        ]
        caps = {
            "a": AgentCapability(
                agent_id="a", name="A", skills=["skill_a"], description=""
            ),
            "b": AgentCapability(
                agent_id="b", name="B", skills=["skill_b"], description=""
            ),
        }
        rfp = TaskRFP(requirement="Test task", required_skills=["skill_a"])

        # Mock the selector agent
        mock_result = MagicMock()
        mock_result.output = JudgmentResult(
            selected_agent_id="a",
            reasoning="Better skill match",
        )
        strategy.selector.run = AsyncMock(return_value=mock_result)

        winner = await strategy.select(bids, rfp, caps)

        assert winner is not None
        assert winner.agent_id == "a"

    @pytest.mark.asyncio
    async def test_fallback_to_first_if_invalid_id(self):
        from agentic_patterns.agent_marketplace import AgentJudgmentStrategy
        from agentic_patterns.agent_marketplace import JudgmentResult

        strategy = AgentJudgmentStrategy()
        rfp_id = uuid4()
        bids = [
            AgentBid(
                rfp_id=rfp_id, agent_id="a", confidence=0.7, proposal="Plan A"
            ),
        ]
        rfp = TaskRFP(requirement="Test")

        # Mock selector returning invalid agent ID
        mock_result = MagicMock()
        mock_result.output = JudgmentResult(
            selected_agent_id="invalid_agent",
            reasoning="Selected nonexistent agent",
        )
        strategy.selector.run = AsyncMock(return_value=mock_result)

        winner = await strategy.select(bids, rfp, {})

        # Should fallback to first bid
        assert winner is not None
        assert winner.agent_id == "a"


class TestProtocolCompliance:
    """Test that strategies comply with SelectionStrategy protocol."""

    def test_highest_confidence_is_strategy(self):
        from agentic_patterns.agent_marketplace import (
            HighestConfidenceStrategy,
        )
        from agentic_patterns.agent_marketplace import SelectionStrategy

        strategy = HighestConfidenceStrategy()
        assert isinstance(strategy, SelectionStrategy)

    def test_best_skill_match_is_strategy(self):
        from agentic_patterns.agent_marketplace import BestSkillMatchStrategy
        from agentic_patterns.agent_marketplace import SelectionStrategy

        strategy = BestSkillMatchStrategy()
        assert isinstance(strategy, SelectionStrategy)

    def test_weighted_score_is_strategy(self):
        from agentic_patterns.agent_marketplace import SelectionStrategy
        from agentic_patterns.agent_marketplace import WeightedScoreStrategy

        strategy = WeightedScoreStrategy()
        assert isinstance(strategy, SelectionStrategy)

    def test_agent_judgment_is_strategy(self):
        from agentic_patterns.agent_marketplace import AgentJudgmentStrategy
        from agentic_patterns.agent_marketplace import SelectionStrategy

        strategy = AgentJudgmentStrategy()
        assert isinstance(strategy, SelectionStrategy)


class TestStrategyInFullFlow:
    """Test strategies in the full marketplace flow."""

    @pytest.mark.asyncio
    async def test_run_marketplace_with_custom_strategy(self):
        from agentic_patterns.agent_marketplace import (
            HighestConfidenceStrategy,
        )

        rfp = TaskRFP(requirement="Test task")

        cap = AgentCapability(
            agent_id="test",
            name="Test",
            skills=["skill_a"],
            description="Test agent",
        )

        bid_response = BidResponse(
            will_bid=True,
            confidence=0.9,
            proposal="Will do",
            reasoning="Skills match",
        )

        mock_agent = MagicMock()
        mock_bid_result = MagicMock()
        mock_bid_result.output = bid_response
        mock_exec_result = MagicMock()
        mock_exec_result.output = "Done!"

        mock_agent.run = AsyncMock(
            side_effect=[mock_bid_result, mock_exec_result]
        )

        with patch(
            "agentic_patterns.agent_marketplace.agora_graph"
        ) as mock_graph:
            mock_graph_result = MagicMock()
            mock_graph_result.output = TaskResult(
                rfp_id=rfp.id,
                agent_id="test",
                success=True,
                output="Done!",
            )
            mock_graph.run = AsyncMock(return_value=mock_graph_result)

            result = await run_marketplace_task(
                rfp,
                [(cap, mock_agent)],
                strategy=HighestConfidenceStrategy(),
            )

            assert result.success
