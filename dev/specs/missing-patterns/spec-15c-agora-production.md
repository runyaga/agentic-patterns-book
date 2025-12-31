# Specification: The Agora - Production Features (Milestone 3)

**Chapter:** 15c
**Pattern Name:** The Agora (Production)
**Status:** Draft
**Depends On:** spec-15a-agora-core.md, spec-15b-agora-strategies.md

## 1. Overview

This milestone adds production-ready features:
- Event callbacks for observability
- Load balancing via capacity tracking
- Configurable timeouts

These features make the Agora suitable for real-world deployments while
remaining optional (backward compatible with M1/M2).

## 2. Event Callbacks

### 2.1 Callback Types

```python
from typing import Callable, Awaitable


# Callback type aliases
OnBidReceived = Callable[[AgentBid], Awaitable[None]]
OnWinnerSelected = Callable[[AgentBid, list[AgentBid]], Awaitable[None]]
OnTaskComplete = Callable[[TaskResult], Awaitable[None]]


@dataclass
class AgoraCallbacks:
    """Optional callbacks for marketplace events."""
    on_bid_received: OnBidReceived | None = None
    on_winner_selected: OnWinnerSelected | None = None
    on_task_complete: OnTaskComplete | None = None
```

### 2.2 Updated State

```python
@dataclass
class AgoraState:
    rfp: TaskRFP
    registered_bidders: list[AgentCapability]
    bidder_agents: dict[str, Agent]
    strategy: SelectionStrategy = field(default_factory=WeightedScoreStrategy)
    callbacks: AgoraCallbacks = field(default_factory=AgoraCallbacks)
    bid_timeout_seconds: float = 5.0

    # Mutable state
    bids: list[AgentBid] = field(default_factory=list)
    winning_bid: AgentBid | None = None
```

### 2.3 Callback Integration

In `CollectBidsNode`:
```python
async def run(self, ctx: GraphRunContext[AgoraState]) -> ...:
    # ... collect bids ...

    # Fire callbacks
    if ctx.state.callbacks.on_bid_received:
        for bid in ctx.state.bids:
            await ctx.state.callbacks.on_bid_received(bid)

    return SelectWinnerNode()
```

In `SelectWinnerNode`:
```python
async def run(self, ctx: GraphRunContext[AgoraState]) -> ...:
    # ... select winner ...

    if ctx.state.callbacks.on_winner_selected:
        await ctx.state.callbacks.on_winner_selected(
            ctx.state.winning_bid,
            ctx.state.bids,
        )

    return ExecuteTaskNode()
```

### 2.4 Usage Example

```python
async def log_bid(bid: AgentBid) -> None:
    print(f"Bid received: {bid.agent_id} @ {bid.confidence}")


async def log_winner(winner: AgentBid, all_bids: list[AgentBid]) -> None:
    print(f"Winner: {winner.agent_id} from {len(all_bids)} bids")


result = await run_marketplace_task(
    rfp=rfp,
    bidders=bidders,
    callbacks=AgoraCallbacks(
        on_bid_received=log_bid,
        on_winner_selected=log_winner,
    ),
)
```

## 3. Load Balancing

### 3.1 Enhanced AgentCapability

```python
class AgentCapability(BaseModel):
    """Agent capability with load tracking."""
    agent_id: str
    name: str
    skills: list[str]
    description: str
    max_concurrent: int = Field(default=3, description="Max parallel tasks")
    current_load: int = Field(default=0, description="Active task count")

    @property
    def available_capacity(self) -> int:
        """Remaining capacity for new tasks."""
        return max(0, self.max_concurrent - self.current_load)

    @property
    def is_available(self) -> bool:
        """Whether agent can accept new tasks."""
        return self.available_capacity > 0
```

### 3.2 Capacity-Aware Bidding

In `CollectBidsNode`, skip agents at capacity:

```python
async def get_bid(cap: AgentCapability) -> AgentBid | None:
    # Skip agents at capacity
    if not cap.is_available:
        return None

    agent = ctx.state.bidder_agents.get(cap.agent_id)
    # ... rest of bidding logic ...
```

### 3.3 CapacityAwareStrategy

```python
@dataclass
class CapacityAwareStrategy:
    """
    Weighted strategy that also considers agent availability.

    Prevents overloading popular agents.
    """
    confidence_weight: float = 0.5
    skill_weight: float = 0.3
    capacity_weight: float = 0.2

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
            if not cap:
                return 0.0

            # Skill match
            skill_score = 0.0
            if rfp.required_skills:
                matched = len(set(cap.skills) & set(rfp.required_skills))
                skill_score = matched / len(rfp.required_skills)

            # Capacity (prefer less loaded agents)
            capacity_score = cap.available_capacity / cap.max_concurrent

            return (
                self.confidence_weight * bid.confidence +
                self.skill_weight * skill_score +
                self.capacity_weight * capacity_score
            )

        return max(bids, key=score)
```

### 3.4 Load Tracking

Update load before/after execution:

```python
@dataclass
class ExecuteTaskNode(BaseNode[AgoraState, None, TaskResult]):
    async def run(self, ctx: GraphRunContext[AgoraState]) -> End[TaskResult]:
        bid = ctx.state.winning_bid
        cap = next(
            (c for c in ctx.state.registered_bidders
             if c.agent_id == bid.agent_id),
            None
        )

        # Increment load
        if cap:
            cap.current_load += 1

        try:
            # ... execute task ...
            result = ...
        finally:
            # Decrement load
            if cap:
                cap.current_load = max(0, cap.current_load - 1)

        return End(result)
```

## 4. Configurable Timeouts

### 4.1 Timeout Configuration

```python
@dataclass
class AgoraConfig:
    """Configuration for marketplace behavior."""
    bid_timeout_seconds: float = 5.0
    execution_timeout_seconds: float = 30.0
    max_retries: int = 0  # 0 = no retries
```

### 4.2 Updated Entry Point

```python
async def run_marketplace_task(
    rfp: TaskRFP,
    bidders: list[tuple[AgentCapability, Agent]],
    strategy: SelectionStrategy | None = None,
    callbacks: AgoraCallbacks | None = None,
    config: AgoraConfig | None = None,
) -> TaskResult:
    """
    Run a task through the marketplace.

    Args:
        rfp: The task request.
        bidders: List of (capability, agent) tuples.
        strategy: Selection strategy (default: WeightedScoreStrategy).
        callbacks: Optional event callbacks.
        config: Optional configuration.
    """
    if strategy is None:
        strategy = WeightedScoreStrategy()
    if config is None:
        config = AgoraConfig()

    state = AgoraState(
        rfp=rfp,
        registered_bidders=[cap for cap, _ in bidders],
        bidder_agents={cap.agent_id: agent for cap, agent in bidders},
        strategy=strategy,
        callbacks=callbacks or AgoraCallbacks(),
        bid_timeout_seconds=config.bid_timeout_seconds,
    )
    # ...
```

## 5. Future Considerations (Not Implemented)

### 5.1 Coalition Support

Agents could form teams to bid together:
```python
class Coalition(BaseModel):
    coalition_id: str
    members: list[str]  # agent_ids
    combined_skills: list[str]
    coordinator: str  # Lead agent
```

### 5.2 Deadlock Prevention

For recursive RFPs (agents posting tasks to other agents):
```python
class CircuitBreaker:
    """Prevent infinite agent recursion."""
    max_depth: int = 3
    current_depth: int = 0
```

### 5.3 Durable Message Bus

For production, replace in-memory with:
- Redis Streams
- RabbitMQ
- NATS

### 5.4 Distributed Tracing

Integration with OpenTelemetry/Logfire:
```python
with logfire.span("marketplace.collect_bids"):
    # ... bidding logic ...
```

## 6. Test Strategy

```python
class TestCallbacks:
    async def test_on_bid_received_called(self):
        received = []
        async def capture(bid: AgentBid):
            received.append(bid)

        await run_marketplace_task(
            rfp=sample_rfp,
            bidders=sample_bidders,
            callbacks=AgoraCallbacks(on_bid_received=capture),
        )
        assert len(received) > 0

    async def test_on_winner_selected_called(self):
        ...


class TestLoadBalancing:
    async def test_skips_agents_at_capacity(self):
        cap = AgentCapability(
            agent_id="busy",
            name="Busy Agent",
            skills=["test"],
            description="",
            max_concurrent=2,
            current_load=2,
        )
        assert not cap.is_available

    async def test_capacity_score_in_selection(self):
        strategy = CapacityAwareStrategy()
        # Agent with more capacity should score higher
        ...


class TestTimeouts:
    async def test_bid_timeout_respected(self):
        # Mock slow agent that exceeds timeout
        ...
```

## 7. Migration Guide

### From M1 to M3

No breaking changes. New parameters are optional:

```python
# M1 style (still works)
result = await run_marketplace_task(rfp, bidders)

# M3 style (with all features)
result = await run_marketplace_task(
    rfp,
    bidders,
    strategy=CapacityAwareStrategy(),
    callbacks=AgoraCallbacks(on_bid_received=log_bid),
    config=AgoraConfig(bid_timeout_seconds=10.0),
)
```

### From M2 to M3

Add `capacity_weight` to existing strategies or use `CapacityAwareStrategy`:

```python
# M2 style
strategy = WeightedScoreStrategy(confidence_weight=0.6, skill_weight=0.4)

# M3 style with capacity
strategy = CapacityAwareStrategy(
    confidence_weight=0.5,
    skill_weight=0.3,
    capacity_weight=0.2,
)
```
