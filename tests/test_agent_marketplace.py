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
