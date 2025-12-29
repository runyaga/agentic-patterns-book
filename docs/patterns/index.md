# Agentic Design Patterns

Documentation for each implemented pattern from the "Agentic Design Patterns"
book, ported to pydantic-ai.

See [Idioms](../idioms.md) for pydantic-ai, pydantic_graph, and pydantic-evals patterns.

## Pattern Index

### Foundation

| Ch | Pattern | Description | Status |
|----|---------|-------------|--------|
| 1 | [Prompt Chaining](01-prompt-chaining.md) | Chain LLM calls sequentially | ✅ |
| 2 | [Routing](02-routing.md) | Classify and route to handlers | ✅ |
| 3 | [Parallelization](03-parallelization.md) | Execute tasks concurrently | ✅ |

### Reasoning

| Ch | Pattern | Description | Status |
|----|---------|-------------|--------|
| 4 | [Reflection](04-reflection.md) | Self-evaluate and improve | ✅ |
| 5 | [Tool Use](05-tool-use.md) | Interact with external tools | ✅ |
| 6 | [Planning](06-planning.md) | Decompose goals into steps | ✅ |

### Coordination

| Ch | Pattern | Description | Status |
|----|---------|-------------|--------|
| 7 | [Multi-Agent](07-multi-agent.md) | Coordinate specialized agents | ✅ |
| 8 | [Memory](08-memory.md) | Maintain conversation context | ✅ |
| 9 | [Learning](09-learning.md) | Improve through experience | ✅ |

### Integration

| Ch | Pattern | Description | Status |
|----|---------|-------------|--------|
| 10 | [MCP Integration](10-mcp-integration.md) | Extensibility via MCP tools | ✅ |
| 11 | [Goal Monitoring](11-goal-monitoring.md) | Proactive state monitoring | ✅ |
| 12 | [Exception Recovery](12-exception-recovery.md) | Robust error handling | ✅ |

### Advanced

| Ch | Pattern | Description | Status |
|----|---------|-------------|--------|
| 13 | [Human-in-the-Loop](13-human-in-loop.md) | Integrate human oversight | ✅ |
| 14 | [Knowledge Retrieval](14-knowledge-retrieval.md) | RAG pattern | ✅ |
| 15 | [Agent Marketplace](15-agent-marketplace.md) | Decentralized bidding | 📋 Spec |
| 16 | [Resource Optimization](16-resource-optimization.md) | Cost/latency routing | ✅ |
| 17a | [Thought Candidates](17a-thought-candidates.md) | Best-of-N sampling | ✅ |
| 17b | [Tree of Thoughts](17b-tree-of-thoughts.md) | Multi-level exploration | ✅ |
| 18 | [Guardrails](18-guardrails.md) | Safety and filtering | ✅ |
| 19 | [Evaluation & Monitoring](19-evaluation-monitoring.md) | Observability | ✅ |
| 20 | [Prioritization](20-prioritization.md) | Task scheduling | ✅ |
| 21 | [Domain Exploration](21-domain-exploration.md) | Knowledge graph mapping | 📋 Spec |

## Pattern Complexity

Patterns build on each other:

```
Ch 1: Prompt Chaining (foundation)
  |
  +-> Ch 2: Routing
  +-> Ch 3: Parallelization
  +-> Ch 4: Reflection
  |   |
  |   +-> Ch 12: Exception Recovery (depends on Ch 4 logic)
  |   +-> Ch 17a: Thought Candidates (best-of-N)
  |       +-> Ch 17b: Tree of Thoughts (extends 17a)
  |
  +-> Ch 5: Tool Use (depends on Ch 1-2)
      |
      +-> Ch 10: MCP Integration (extends Ch 5)
      |
      +-> Ch 6: Planning (depends on Ch 1-5)
          |
          +-> Ch 7: Multi-Agent (depends on Ch 1-6)
              |
              +-> Ch 11: Goal Monitoring (autonomous loop)
              +-> Ch 13: Human-in-Loop (depends on Ch 1-7)
              +-> Ch 15: Agent Marketplace (decentralized multi-agent)

Ch 8: Memory (standalone, depends on Ch 1)
Ch 9: Learning (depends on Ch 4, 8)
Ch 14: Knowledge Retrieval (depends on Ch 5)
    |
    +-> Ch 21: Domain Exploration (active retrieval)

Ch 16: Resource Optimization (depends on Ch 2 routing)
Ch 18: Guardrails (standalone, applies to all patterns)
Ch 19: Evaluation & Monitoring (standalone, applies to all patterns)
Ch 20: Prioritization (depends on Ch 6 planning)
```

## Running Examples

Each pattern has a runnable example:

```bash
# Run any pattern
.venv/bin/python -m agentic_patterns.<pattern_name>

# Examples
.venv/bin/python -m agentic_patterns.prompt_chaining
.venv/bin/python -m agentic_patterns.routing
.venv/bin/python -m agentic_patterns.parallelization
```
