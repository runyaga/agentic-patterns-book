"""
Agent Marketplace Pattern Implementation (The Agora).

Based on the Agentic Design Patterns book Chapter 15:
A decentralized marketplace where agents bid on tasks based on their
capabilities. Unlike Multi-Agent (Ch 7) where a supervisor assigns tasks,
here agents self-select through competitive bidding.

Key concepts:
- RFP (Request for Proposal): Task announcement seeking bids
- Bidding: Agents evaluate tasks and submit confidence-based proposals
- Selection: Winner chosen by weighted score (confidence + skill match)
- Execution: Winning agent performs the task

Flow diagram:

```mermaid
--8<-- [start:diagram]
stateDiagram-v2
    [*] --> PostRFPNode: Task submitted
    PostRFPNode --> CollectBidsNode: RFP posted
    PostRFPNode --> [*]: No bidders
    CollectBidsNode --> SelectWinnerNode: Bids collected
    SelectWinnerNode --> ExecuteTaskNode: Winner selected
    SelectWinnerNode --> [*]: No valid bids
    ExecuteTaskNode --> [*]: Task complete
--8<-- [end:diagram]
```

Example usage:
    result = await run_marketplace_task(
        rfp=TaskRFP(requirement="Summarize", required_skills=["brevity"]),
        bidders=[(capability, agent), ...],
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import runtime_checkable
from uuid import UUID
from uuid import uuid4

if TYPE_CHECKING:
    pass

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode
from pydantic_graph import End
from pydantic_graph import Graph
from pydantic_graph import GraphRunContext

from agentic_patterns._models import get_model


# --8<-- [start:models]
class AgentCapability(BaseModel):
    """Describes what an agent can do."""

    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Human-readable name")
    skills: list[str] = Field(description="Skill tags")
    description: str = Field(description="Specialization description")


class TaskRFP(BaseModel):
    """Request for Proposal - a task seeking bids."""

    id: UUID = Field(default_factory=uuid4)
    requirement: str = Field(description="What needs to be done")
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills the bidder should have",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )
    min_confidence: float = Field(
        default=0.5,
        description="Minimum acceptable confidence",
    )


class AgentBid(BaseModel):
    """A bid from an agent on an RFP."""

    rfp_id: UUID = Field(description="RFP this bid is for")
    agent_id: str = Field(description="Bidding agent ID")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in ability to complete (0-1)",
    )
    proposal: str = Field(description="Brief approach description")


class TaskResult(BaseModel):
    """Result from the winning bidder."""

    rfp_id: UUID = Field(description="RFP that was executed")
    agent_id: str = Field(description="Agent that executed the task")
    success: bool = Field(description="Whether execution succeeded")
    output: str = Field(description="Task output or empty on failure")
    error_message: str | None = Field(
        default=None,
        description="Error message if failed",
    )


class BidResponse(BaseModel):
    """What a bidder returns when asked to bid."""

    will_bid: bool = Field(description="Whether to submit a bid")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in ability to complete",
    )
    proposal: str = Field(description="Brief approach description")
    reasoning: str = Field(description="Why this confidence level")


@dataclass
class BidderContext:
    """Context passed to bidder agents."""

    rfp: TaskRFP
    own_capabilities: AgentCapability


@dataclass
class AgoraState:
    """Graph state for the marketplace."""

    rfp: TaskRFP
    registered_bidders: list[AgentCapability]
    bidder_agents: dict[str, Agent[BidderContext, BidResponse]]
    strategy: Any = None  # SelectionStrategy, default: WeightedScore
    bid_timeout_seconds: float = 5.0
    bids: list[AgentBid] = field(default_factory=list)
    winning_bid: AgentBid | None = None


# --8<-- [end:models]


# --8<-- [start:strategies]
@runtime_checkable
class SelectionStrategy(Protocol):
    """Protocol for bid selection strategies."""

    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        """
        Select the winning bid.

        Args:
            bids: Valid bids (already filtered by min_confidence).
            rfp: The original task request.
            capabilities: Map of agent_id -> AgentCapability.

        Returns:
            Winning bid, or None if no suitable bid found.
        """
        ...


class HighestConfidenceStrategy:
    """Select the bid with highest confidence score."""

    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        """Select bid with highest confidence."""
        if not bids:
            return None
        return max(bids, key=lambda b: b.confidence)


class BestSkillMatchStrategy:
    """Select the bid with best skill overlap."""

    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        """Select bid with best skill match."""
        if not bids:
            return None

        def skill_score(bid: AgentBid) -> float:
            cap = capabilities.get(bid.agent_id)
            if not cap or not rfp.required_skills:
                return 0.0
            matched = len(set(cap.skills) & set(rfp.required_skills))
            return matched / len(rfp.required_skills)

        return max(bids, key=skill_score)


@dataclass
class WeightedScoreStrategy:
    """
    Select using weighted combination of confidence and skill match.

    This is the default strategy from Milestone 1.
    """

    confidence_weight: float = 0.6
    skill_weight: float = 0.4

    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        """Select bid with best weighted score."""
        if not bids:
            return None

        def score(bid: AgentBid) -> float:
            cap = capabilities.get(bid.agent_id)
            if not cap or not rfp.required_skills:
                return bid.confidence * self.confidence_weight

            matched = len(set(cap.skills) & set(rfp.required_skills))
            skill_score = matched / len(rfp.required_skills)

            return (
                self.confidence_weight * bid.confidence
                + self.skill_weight * skill_score
            )

        return max(bids, key=score)


class JudgmentResult(BaseModel):
    """Result from the selector agent."""

    selected_agent_id: str = Field(description="ID of the winning agent")
    reasoning: str = Field(description="Why this agent was selected")


class AgentJudgmentStrategy:
    """
    Use an LLM agent to qualitatively evaluate bids.

    Best for tasks where proposal quality matters more than
    simple metrics (creative tasks, complex analysis).
    """

    def __init__(self) -> None:
        """Initialize with a selector agent."""
        self.selector = Agent(
            get_model(),
            system_prompt=(
                "You are a procurement specialist. Given a task and "
                "multiple bids, select the best bidder based on:\n"
                "1. How well their skills match the requirements\n"
                "2. Quality and clarity of their proposal\n"
                "3. Confidence level (but don't over-weight it)\n\n"
                "Choose the agent most likely to succeed."
            ),
            output_type=JudgmentResult,
        )

    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        """Select bid using LLM judgment."""
        if not bids:
            return None

        # Format bids for the selector
        bids_text = "\n".join(
            f"- {b.agent_id}: confidence={b.confidence:.2f}, "
            f'proposal="{b.proposal}"'
            for b in bids
        )

        caps_text = "\n".join(
            f"- {c.agent_id}: skills={c.skills}"
            for c in capabilities.values()
            if c.agent_id in {b.agent_id for b in bids}
        )

        result = await self.selector.run(
            f"Task: {rfp.requirement}\n"
            f"Required skills: {rfp.required_skills}\n\n"
            f"Agent capabilities:\n{caps_text}\n\n"
            f"Bids:\n{bids_text}\n\n"
            f"Select the best agent."
        )

        winner_id = result.output.selected_agent_id
        return next((b for b in bids if b.agent_id == winner_id), bids[0])


# --8<-- [end:strategies]


# --8<-- [start:agents]
model = get_model()


def create_bidder_agent(
    capability: AgentCapability,
) -> Agent[BidderContext, BidResponse]:
    """
    Create a bidder agent with specific capabilities.

    Each bidder has a system prompt describing its specialization.
    When presented with an RFP, it evaluates whether to bid based
    on skill match.

    Args:
        capability: The agent's capabilities.

    Returns:
        A configured Agent that can bid on RFPs.
    """
    return Agent(
        model,
        system_prompt=(
            f"You are {capability.name}, specializing in: "
            f"{', '.join(capability.skills)}.\n\n"
            f"{capability.description}\n\n"
            "When presented with a task RFP, evaluate if you can help. "
            "Set will_bid=True only if your skills match the requirements. "
            "Be honest about your confidence level (0-1)."
        ),
        output_type=BidResponse,
        deps_type=BidderContext,
    )


# --8<-- [end:agents]


# --8<-- [start:graph_nodes]
@dataclass
class PostRFPNode(BaseNode[AgoraState, None, TaskResult]):
    """Post the RFP and validate bidders exist."""

    async def run(
        self,
        ctx: GraphRunContext[AgoraState],
    ) -> CollectBidsNode | End[TaskResult]:
        """Validate bidders and proceed to collection."""
        if not ctx.state.registered_bidders:
            return End(
                TaskResult(
                    rfp_id=ctx.state.rfp.id,
                    agent_id="",
                    success=False,
                    output="",
                    error_message="No bidders registered",
                )
            )

        print(f"RFP Posted: {ctx.state.rfp.requirement[:50]}...")
        print(f"Required skills: {ctx.state.rfp.required_skills}")
        print(f"Registered bidders: {len(ctx.state.registered_bidders)}")

        return CollectBidsNode()


@dataclass
class CollectBidsNode(BaseNode[AgoraState, None, TaskResult]):
    """Collect bids from all registered bidders in parallel."""

    async def run(
        self,
        ctx: GraphRunContext[AgoraState],
    ) -> SelectWinnerNode:
        """Gather bids from all agents concurrently."""
        rfp = ctx.state.rfp
        timeout = ctx.state.bid_timeout_seconds

        async def get_bid(cap: AgentCapability) -> AgentBid | None:
            agent = ctx.state.bidder_agents.get(cap.agent_id)
            if not agent:
                return None

            try:
                result = await asyncio.wait_for(
                    agent.run(
                        f"Evaluate this RFP and decide if you want to bid:\n\n"
                        f"Requirement: {rfp.requirement}\n"
                        f"Required skills: {rfp.required_skills}\n"
                        f"Context: {rfp.context}",
                        deps=BidderContext(rfp=rfp, own_capabilities=cap),
                    ),
                    timeout=timeout,
                )
                response = result.output

                if response.will_bid:
                    return AgentBid(
                        rfp_id=rfp.id,
                        agent_id=cap.agent_id,
                        confidence=response.confidence,
                        proposal=response.proposal,
                    )
            except TimeoutError:
                print(f"  Bid timeout: {cap.agent_id}")
            except Exception as e:
                print(f"  Bid error from {cap.agent_id}: {e}")

            return None

        tasks = [get_bid(cap) for cap in ctx.state.registered_bidders]
        results = await asyncio.gather(*tasks)
        ctx.state.bids = [b for b in results if b is not None]

        print(f"Received {len(ctx.state.bids)} bids")
        return SelectWinnerNode()


@dataclass
class SelectWinnerNode(BaseNode[AgoraState, None, TaskResult]):
    """Select the winning bid using the configured strategy."""

    async def run(
        self,
        ctx: GraphRunContext[AgoraState],
    ) -> ExecuteTaskNode | End[TaskResult]:
        """Select winner using the configured selection strategy."""
        rfp = ctx.state.rfp
        bids = ctx.state.bids
        strategy = ctx.state.strategy or WeightedScoreStrategy()

        # Filter by minimum confidence
        valid_bids = [b for b in bids if b.confidence >= rfp.min_confidence]

        if not valid_bids:
            return End(
                TaskResult(
                    rfp_id=rfp.id,
                    agent_id="",
                    success=False,
                    output="",
                    error_message="No bids met minimum confidence threshold",
                )
            )

        # Build capabilities dict for strategy
        capabilities = {
            cap.agent_id: cap for cap in ctx.state.registered_bidders
        }

        # Use the strategy to select the winner
        winner = await strategy.select(valid_bids, rfp, capabilities)

        if not winner:
            return End(
                TaskResult(
                    rfp_id=rfp.id,
                    agent_id="",
                    success=False,
                    output="",
                    error_message="Strategy returned no winner",
                )
            )

        ctx.state.winning_bid = winner
        print(f"Winner: {winner.agent_id}")

        return ExecuteTaskNode()


@dataclass
class ExecuteTaskNode(BaseNode[AgoraState, None, TaskResult]):
    """Execute the task with the winning bidder."""

    async def run(
        self,
        ctx: GraphRunContext[AgoraState],
    ) -> End[TaskResult]:
        """Run the winning agent on the task."""
        rfp = ctx.state.rfp
        bid = ctx.state.winning_bid

        if not bid:
            return End(
                TaskResult(
                    rfp_id=rfp.id,
                    agent_id="",
                    success=False,
                    output="",
                    error_message="No winning bid",
                )
            )

        agent = ctx.state.bidder_agents.get(bid.agent_id)

        if not agent:
            return End(
                TaskResult(
                    rfp_id=rfp.id,
                    agent_id=bid.agent_id,
                    success=False,
                    output="",
                    error_message=f"Agent {bid.agent_id} not found",
                )
            )

        try:
            result = await agent.run(
                f"Execute this task using your proposed approach:\n\n"
                f"Task: {rfp.requirement}\n"
                f"Your proposal: {bid.proposal}\n"
                f"Context: {rfp.context}",
            )
            return End(
                TaskResult(
                    rfp_id=rfp.id,
                    agent_id=bid.agent_id,
                    success=True,
                    output=str(result.output),
                )
            )
        except Exception as e:
            return End(
                TaskResult(
                    rfp_id=rfp.id,
                    agent_id=bid.agent_id,
                    success=False,
                    output="",
                    error_message=str(e),
                )
            )


# Define the marketplace graph
agora_graph: Graph[AgoraState, None, TaskResult] = Graph(
    nodes=[PostRFPNode, CollectBidsNode, SelectWinnerNode, ExecuteTaskNode],
)
# --8<-- [end:graph_nodes]


# --8<-- [start:marketplace]
async def run_marketplace_task(
    rfp: TaskRFP,
    bidders: list[tuple[AgentCapability, Agent[BidderContext, BidResponse]]],
    strategy: SelectionStrategy | None = None,
    bid_timeout_seconds: float = 5.0,
) -> TaskResult:
    """
    Run a task through the marketplace.

    Posts an RFP, collects bids from registered agents, selects a winner
    using the configured strategy, and executes the task with the winner.

    Args:
        rfp: The task request for proposal.
        bidders: List of (capability, agent) tuples.
        strategy: Selection strategy (default: WeightedScoreStrategy).
        bid_timeout_seconds: Timeout for each bid (default: 5.0).

    Returns:
        TaskResult from the winning bidder's execution.

    Example:
        summarizer = AgentCapability(
            agent_id="summarizer",
            name="Fast Summarizer",
            skills=["speed", "brevity"],
            description="Quick, concise summaries",
        )
        result = await run_marketplace_task(
            rfp=TaskRFP(
                requirement="Summarize this article for executives",
                required_skills=["brevity"],
            ),
            bidders=[(summarizer, create_bidder_agent(summarizer))],
        )

        # With custom strategy:
        result = await run_marketplace_task(
            rfp=rfp,
            bidders=bidders,
            strategy=HighestConfidenceStrategy(),
        )
    """
    print("=" * 60)
    print("Agent Marketplace: Starting")
    print("=" * 60)

    if strategy is None:
        strategy = WeightedScoreStrategy()

    state = AgoraState(
        rfp=rfp,
        registered_bidders=[cap for cap, _ in bidders],
        bidder_agents={cap.agent_id: agent for cap, agent in bidders},
        strategy=strategy,
        bid_timeout_seconds=bid_timeout_seconds,
    )

    result = await agora_graph.run(PostRFPNode(), state=state)

    print("=" * 60)
    return result.output


# --8<-- [end:marketplace]


if __name__ == "__main__":

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Agent Marketplace Pattern (The Agora)")
        print("=" * 60)

        # Create specialized bidders
        summarizer = AgentCapability(
            agent_id="summarizer",
            name="Fast Summarizer",
            skills=["speed", "brevity", "extraction"],
            description="Quick summaries in executive style",
        )
        analyzer = AgentCapability(
            agent_id="analyzer",
            name="Deep Analyzer",
            skills=["thoroughness", "citations", "research"],
            description="Deep analysis with sources and citations",
        )
        writer = AgentCapability(
            agent_id="writer",
            name="Creative Writer",
            skills=["engagement", "narrative", "storytelling"],
            description="Engaging narrative content for broad audiences",
        )

        bidders = [
            (summarizer, create_bidder_agent(summarizer)),
            (analyzer, create_bidder_agent(analyzer)),
            (writer, create_bidder_agent(writer)),
        ]

        # Post RFP for executive summary
        rfp = TaskRFP(
            requirement=(
                "Summarize the key advances in quantum computing "
                "for a board of directors presentation"
            ),
            required_skills=["brevity", "extraction"],
        )

        result = await run_marketplace_task(rfp, bidders)

        print("\nRESULT:")
        print(f"Winner: {result.agent_id}")
        print(f"Success: {result.success}")
        if result.success:
            print(f"Output: {result.output[:500]}...")
        else:
            print(f"Error: {result.error_message}")

    asyncio.run(main())
