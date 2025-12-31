# Specification: The Agora - Selection Strategies (Milestone 2)

**Chapter:** 15b
**Pattern Name:** The Agora (Strategies)
**Status:** Draft
**Depends On:** spec-15a-agora-core.md

## 1. Overview

Milestone 1 uses a fixed weighted scoring strategy. This milestone adds:
- A `SelectionStrategy` protocol for extensibility
- Four built-in strategies
- Support for custom user-defined strategies

## 2. SelectionStrategy Protocol

```python
from typing import Protocol, runtime_checkable
from agentic_patterns.agent_marketplace import (
    AgentBid,
    AgentCapability,
    TaskRFP,
)


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
```

## 3. Built-in Strategies

### 3.1 HighestConfidenceStrategy

```python
class HighestConfidenceStrategy:
    """Select the bid with highest confidence score."""

    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        if not bids:
            return None
        return max(bids, key=lambda b: b.confidence)
```

### 3.2 BestSkillMatchStrategy

```python
class BestSkillMatchStrategy:
    """Select the bid with best skill overlap."""

    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        if not bids:
            return None

        def skill_score(bid: AgentBid) -> float:
            cap = capabilities.get(bid.agent_id)
            if not cap or not rfp.required_skills:
                return 0.0
            matched = len(set(cap.skills) & set(rfp.required_skills))
            return matched / len(rfp.required_skills)

        return max(bids, key=skill_score)
```

### 3.3 WeightedScoreStrategy (Default)

```python
from dataclasses import dataclass


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
        if not bids:
            return None

        def score(bid: AgentBid) -> float:
            cap = capabilities.get(bid.agent_id)
            if not cap or not rfp.required_skills:
                return bid.confidence * self.confidence_weight

            matched = len(set(cap.skills) & set(rfp.required_skills))
            skill_score = matched / len(rfp.required_skills)

            return (
                self.confidence_weight * bid.confidence +
                self.skill_weight * skill_score
            )

        return max(bids, key=score)
```

### 3.4 AgentJudgmentStrategy

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from agentic_patterns._models import get_model


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

    def __init__(self):
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
        if not bids:
            return None

        # Format bids for the selector
        bids_text = "\n".join(
            f"- {b.agent_id}: confidence={b.confidence:.2f}, "
            f"proposal=\"{b.proposal}\""
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
```

## 4. Updated Entry Point

```python
async def run_marketplace_task(
    rfp: TaskRFP,
    bidders: list[tuple[AgentCapability, Agent]],
    strategy: SelectionStrategy | None = None,
) -> TaskResult:
    """
    Run a task through the marketplace.

    Args:
        rfp: The task request.
        bidders: List of (capability, agent) tuples.
        strategy: Selection strategy (default: WeightedScoreStrategy).

    Example with custom strategy:
        result = await run_marketplace_task(
            rfp=rfp,
            bidders=bidders,
            strategy=AgentJudgmentStrategy(),
        )
    """
    if strategy is None:
        strategy = WeightedScoreStrategy()

    state = AgoraState(
        rfp=rfp,
        registered_bidders=[cap for cap, _ in bidders],
        bidder_agents={cap.agent_id: agent for cap, agent in bidders},
        strategy=strategy,  # Added to state
    )
    result = await agora_graph.run(PostRFPNode(), state=state)
    return result.output
```

## 5. Custom Strategy Example

Users can implement their own strategies:

```python
class CostAwareStrategy:
    """
    Example: Select based on estimated token cost.

    Demonstrates extensibility of the protocol.
    """

    def __init__(self, max_tokens: int = 1000):
        self.max_tokens = max_tokens

    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        # Filter by token budget (if agents provide estimates)
        affordable = [
            b for b in bids
            if b.metadata.get("estimated_tokens", 0) <= self.max_tokens
        ]
        if not affordable:
            affordable = bids  # Fallback to all if none affordable

        # Among affordable, pick highest confidence
        return max(affordable, key=lambda b: b.confidence) if affordable else None


# Usage
result = await run_marketplace_task(
    rfp=rfp,
    bidders=bidders,
    strategy=CostAwareStrategy(max_tokens=500),
)
```

## 6. Test Strategy

```python
import pytest


class TestHighestConfidenceStrategy:
    async def test_selects_highest_confidence(self):
        strategy = HighestConfidenceStrategy()
        bids = [
            AgentBid(rfp_id=uuid4(), agent_id="a", confidence=0.7, proposal=""),
            AgentBid(rfp_id=uuid4(), agent_id="b", confidence=0.9, proposal=""),
        ]
        winner = await strategy.select(bids, sample_rfp, {})
        assert winner.agent_id == "b"

    async def test_returns_none_for_empty_bids(self):
        strategy = HighestConfidenceStrategy()
        winner = await strategy.select([], sample_rfp, {})
        assert winner is None


class TestBestSkillMatchStrategy:
    async def test_prefers_skill_match_over_confidence(self):
        strategy = BestSkillMatchStrategy()
        bids = [
            AgentBid(rfp_id=uuid4(), agent_id="a", confidence=0.9, proposal=""),
            AgentBid(rfp_id=uuid4(), agent_id="b", confidence=0.6, proposal=""),
        ]
        caps = {
            "a": AgentCapability(agent_id="a", name="A", skills=["x"], description=""),
            "b": AgentCapability(agent_id="b", name="B", skills=["skill_a"], description=""),
        }
        rfp = TaskRFP(requirement="", required_skills=["skill_a"])
        winner = await strategy.select(bids, rfp, caps)
        assert winner.agent_id == "b"  # Better skill match


class TestWeightedScoreStrategy:
    async def test_custom_weights(self):
        strategy = WeightedScoreStrategy(confidence_weight=0.2, skill_weight=0.8)
        # With high skill weight, skill match matters more
        ...


class TestAgentJudgmentStrategy:
    async def test_uses_selector_agent(self):
        # Mock the selector agent
        ...


class TestCustomStrategy:
    async def test_protocol_compliance(self):
        """Custom strategies must implement the protocol."""
        strategy = CostAwareStrategy()
        assert isinstance(strategy, SelectionStrategy)
```

## 7. Integration Updates

Update `AgoraState` to include strategy:

```python
@dataclass
class AgoraState:
    rfp: TaskRFP
    registered_bidders: list[AgentCapability]
    bidder_agents: dict[str, Agent]
    strategy: SelectionStrategy = field(default_factory=WeightedScoreStrategy)
    bids: list[AgentBid] = field(default_factory=list)
    winning_bid: AgentBid | None = None
```

Update `SelectWinnerNode.run()` to use `ctx.state.strategy.select()`.
