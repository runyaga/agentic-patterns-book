# Agent Marketplace (The Agora)

**Chapter 15** · *Specification / Planned*

> **Note:** This pattern is currently in the specification phase. The documentation below reflects the planned design.

In complex systems, agents shouldn't be hard-wired to each other. **The Agora** implements a decentralized communication pattern where agents function like participants in a marketplace. They "bid" on tasks based on their specialized skills and current capacity.

## Key Concepts

```mermaid
sequenceDiagram
    participant Requester
    participant Agora as Marketplace
    participant AgentA as Regex Expert
    participant AgentB as SQL Expert

    Requester->>Agora: Post RFP: "Validate email regex"
    Agora->>AgentA: Notify
    Agora->>AgentB: Notify
    
    par Bidding
        AgentA->>Agora: Bid: "I can do this (0.95 conf)"
        AgentB->>Agora: No Bid (skills mismatch)
    end

    Agora->>Requester: Select Winner (AgentA)
    Requester->>AgentA: Execute Task
    AgentA-->>Requester: Result
```

-   **RFP (Request for Proposal)**: A structured task definition including requirements, skills needed, and budget/deadline.
-   **Bidding**: Agents autonomously decide whether to bid on a task based on their capabilities and load.
-   **Selection Strategy**: The marketplace (or requester) selects the winning bid using strategies like "Highest Confidence", "Lowest Cost", or "Agent Judgment" (asking another LLM to pick).
-   **Decoupling**: The requester does not need to know which agents exist or how to call them.

## Use Cases

1.  **Dynamic Worker Pools**: When the set of available agents changes at runtime (e.g., plugins loaded dynamically).
2.  **Competitive Execution**: When multiple agents *could* do a task, but one might be better suited based on context (e.g., "Fast Linter" vs "Deep Auditor").
3.  **Resource Optimization**: Selecting the cheapest agent that meets the confidence threshold for a given task.

## Proposed Implementation

The core architecture uses `pydantic_graph` to model the RFP lifecycle:

```python
class TaskRFP(BaseModel):
    requirement: str
    required_skills: list[str]
    min_confidence: float = 0.5

class AgentBid(BaseModel):
    agent_id: str
    confidence: float
    proposal: str

# Workflow: PostRFP -> CollectBids -> SelectWinner -> ExecuteTask
```

### Example Scenario

A "Project Manager" agent needs a code review.
1.  It posts an RFP to the Agora.
2.  **Agent A (Linter)** bids: "I can check syntax instantly. Cost: $0.01. Confidence: 1.0 for syntax."
3.  **Agent B (Security)** bids: "I can check for vulnerabilities. Cost: $0.50. Confidence: 0.9 for security."
4.  The Manager selects **Agent A** first for a quick check, then **Agent B** if the syntax passes.

## Production Reality Check

### When to Use
- Worker pool is dynamic (agents join/leave at runtime, plugins loaded dynamically)
- Cost/latency tradeoffs vary per request (cheap fast agent vs. expensive thorough one)
- Competitive execution is needed (multiple agents could handle task, pick best)
- Decoupling is valuable (requesters shouldn't know which agents exist)
- *Comparison*: Router or Supervisor patterns require knowing agents at design time;
  Agora enables runtime discovery

### When NOT to Use
- Agent pool is static and known at design time (use Router or Supervisor instead)
- Latency is critical (bidding adds round-trip overhead)
- Coordination complexity outweighs benefits for your use case
- Single agent can handle all task types adequately
- *Anti-pattern*: Small fixed set of 3-4 tasks where a rules-based router is enough—
  marketplace overhead dominates the work

### Production Considerations
- **Message bus**: Replace `asyncio.Queue` with durable message systems (Redis
  Streams, RabbitMQ, NATS) to handle process crashes.
- **Distributed tracing**: OpenTelemetry/Logfire is critical to visualize the
  async fan-out and debug "why did agent X win?"
- **Economic layer**: Real marketplaces need quota tracking, rate limiting, or
  credits to prevent "Winner's Curse" (agents over-promising).
- **Agent trust**: Third-party agents may misbehave. Sandbox execution, limit
  permissions, and consider reputational scoring over time.
- **Bid timeout**: Set timeouts for bid collection. Don't wait forever for slow
  agents.
- **Selection strategy**: "Highest confidence" vs "lowest cost" vs "agent
  judgment" have different tradeoffs. Make the strategy configurable.
- **Gaming risks**: Agents may learn to game selection criteria. Monitor for
  bid inflation and add safeguards.

## Example

> **Note:** This pattern is in specification phase. No runnable example yet.
