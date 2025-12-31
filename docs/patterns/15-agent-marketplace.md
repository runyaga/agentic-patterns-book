# Chapter 15: Agent Marketplace (The Agora)

A decentralized marketplace where agents bid on tasks based on capabilities.

## Flow Diagram

```mermaid
--8<-- "src/agentic_patterns/agent_marketplace.py:diagram"
```

## Implementation

Source: [`src/agentic_patterns/agent_marketplace.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/agent_marketplace.py)

### Data Models

```python
--8<-- "src/agentic_patterns/agent_marketplace.py:models"
```

### Bidder Agents

```python
--8<-- "src/agentic_patterns/agent_marketplace.py:agents"
```

### Graph Nodes

```python
--8<-- "src/agentic_patterns/agent_marketplace.py:graph_nodes"
```

### Selection Strategies

```python
--8<-- "src/agentic_patterns/agent_marketplace.py:strategies"
```

### Marketplace Logic

```python
--8<-- "src/agentic_patterns/agent_marketplace.py:marketplace"
```

## Selection Strategies

The marketplace supports pluggable selection strategies via the
`SelectionStrategy` protocol:

| Strategy | Description |
|----------|-------------|
| `WeightedScoreStrategy` | Default. Combines confidence (60%) + skill match (40%) |
| `HighestConfidenceStrategy` | Select bid with highest confidence score |
| `BestSkillMatchStrategy` | Select bid with best skill overlap |
| `AgentJudgmentStrategy` | LLM agent evaluates bids qualitatively |

### Using Custom Strategies

```python
from agentic_patterns import (
    run_marketplace_task,
    HighestConfidenceStrategy,
    TaskRFP,
)

# Use highest confidence strategy
result = await run_marketplace_task(
    rfp=TaskRFP(requirement="...", required_skills=["..."]),
    bidders=[...],
    strategy=HighestConfidenceStrategy(),
)
```

### Creating Custom Strategies

Implement the `SelectionStrategy` protocol:

```python
class MyStrategy:
    async def select(
        self,
        bids: list[AgentBid],
        rfp: TaskRFP,
        capabilities: dict[str, AgentCapability],
    ) -> AgentBid | None:
        # Your selection logic here
        return max(bids, key=lambda b: b.confidence) if bids else None
```

## Distinction from Multi-Agent

| Aspect | Multi-Agent (Ch 7) | Agora (Ch 15) |
|--------|-------------------|---------------|
| Routing | Supervisor assigns tasks | Agents self-select via bidding |
| Coupling | Supervisor knows all workers | Requester doesn't know bidders |
| Discovery | Static worker registry | Dynamic capability matching |
| Selection | Supervisor decides | Market-based (best score wins) |

## Use Cases

- **Dynamic Worker Pools**: Agents join/leave at runtime (plugins, services).
- **Competitive Execution**: Multiple agents could handle task; pick best fit.
- **Skill-Based Routing**: Route to agent with matching capabilities.
- **Research Tasks**: Summarizer vs Analyzer vs Writer compete for tasks.

## Production Reality Check

### When to Use
- Worker pool is dynamic (agents join/leave at runtime)
- Cost/latency tradeoffs vary per request
- Multiple agents could handle task, need to pick best
- Decoupling is valuable (requesters shouldn't know which agents exist)
- *Comparison*: Router or Supervisor patterns require knowing agents at design
  time; Agora enables runtime discovery

### When NOT to Use
- Agent pool is static and known at design time (use Router instead)
- Latency is critical (bidding adds round-trip overhead)
- Single agent can handle all task types adequately
- Small fixed set of 3-4 tasks (rules-based router is simpler)

### Production Considerations
- **Bid timeout**: Set timeouts for bid collection (default: 5 seconds)
- **Selection strategy**: Pluggable via `SelectionStrategy` protocol;
  default is `WeightedScoreStrategy` (60% confidence, 40% skill match)
- **Minimum confidence**: Filter bids below threshold (default: 0.5)
- **Parallel bidding**: All agents bid concurrently via `asyncio.gather`
- **Error handling**: Timeout and exception handling per bidder

## Example

```bash
.venv/bin/python -m agentic_patterns.agent_marketplace
```

Output:
```
============================================================
Agent Marketplace: Starting
============================================================
RFP Posted: Summarize the key advances in quantum computing...
Required skills: ['brevity', 'extraction']
Registered bidders: 3
Received 1 bids
Winner: summarizer (score: 0.88)
============================================================

RESULT:
Winner: summarizer
Success: True
Output: Key advances in quantum computing...
```
