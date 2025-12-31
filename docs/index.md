# Agentic Patterns Book

**[View on GitHub](https://github.com/runyaga/agentic-patterns-book)**

**Idiomatic pydantic-ai implementations of agentic design patterns.**

This is a **vibe-coded** repository - built using Claude Code with
[haiku-rag](https://github.com/ggozad/haiku.rag) as the RAG mechanism
for reading the source material.

## Why This Exists

I came late to GenAI and missed a lot of the "old ways" of building agents.
[Soliplex](https://github.com/soliplex/soliplex), a project I'm involved with,
uses newer techniques for agent orchestration.

I built this repository to:

1. **Learn the patterns** - Understand how traditional agentic code looks
2. **Evaluate for Soliplex** - Determine which patterns translate to modern
   approaches
3. **Compare paradigms** - See how things like Human-in-the-Loop differ (spoiler:
   much nicer in Soliplex and [ag-ui](https://github.com/ag-ui-protocol/ag-ui))
4. **Try haiku-rag** - First time using it in a vibe coding session

This is a learning exercise, not a production recommendation.

## What is this?

A Python port of 23 agentic AI patterns from the **Gulli** book:

> *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems*
> by Antonio Gulli (Springer, 2025)
>
> [Purchase from Springer](https://link.springer.com/book/10.1007/978-3-032-01402-3)

Each pattern is implemented as a standalone module with:

- Type-safe Pydantic models
- Async-first design using pydantic-ai
- Comprehensive tests (80%+ branch coverage)
- Runnable examples

## Quick Start

```bash
git clone https://github.com/runyaga/agentic-patterns-book.git
cd agentic-patterns-book
uv venv && uv pip install -e ".[dev]" --python .venv/bin/python

# Run any pattern
.venv/bin/python -m agentic_patterns.prompt_chaining
```

See [Getting Started](getting-started.md) for full setup instructions.

## Patterns Overview

23 implemented patterns. See [full pattern index](patterns/index.md).

| Category | Patterns |
|----------|----------|
| **Foundation** | [Prompt Chaining](patterns/01-prompt-chaining.md), [Routing](patterns/02-routing.md), [Parallelization](patterns/03-parallelization.md) |
| **Reasoning** | [Reflection](patterns/04-reflection.md), [Tool Use](patterns/05-tool-use.md), [Planning](patterns/06-planning.md), [Dynamic Planning](patterns/06b-dynamic-planning.md) |
| **Coordination** | [Multi-Agent](patterns/07-multi-agent.md), [Memory](patterns/08-memory.md), [Learning](patterns/09-learning.md) |
| **Integration** | [MCP Integration](patterns/10-mcp-integration.md), [Goal Monitoring](patterns/11-goal-monitoring.md), [Exception Recovery](patterns/12-exception-recovery.md) |
| **Advanced** | [Human-in-Loop](patterns/13-human-in-loop.md), [Knowledge Retrieval](patterns/14-knowledge-retrieval.md), [Agent Marketplace](patterns/15-agent-marketplace.md), [Resource Optimization](patterns/16-resource-optimization.md), [Thought Candidates](patterns/17a-thought-candidates.md), [Tree of Thoughts](patterns/17b-tree-of-thoughts.md), [Guardrails](patterns/18-guardrails.md), [Evaluation](patterns/19-evaluation-monitoring.md), [Prioritization](patterns/20-prioritization.md), [Domain Exploration](patterns/21-domain-exploration.md) |

## Reproduce This

Want to build your own pattern implementations? See
[Reproduce This](reproduce.md) for the complete workflow including the
haiku-rag setup used to create this repository.

## LLM-Friendly Documentation

This site provides machine-readable documentation for AI assistants:

- [llms.txt](https://runyaga.github.io/agentic-patterns-book/llms.txt) - Concise site index
- [llms-full.txt](https://runyaga.github.io/agentic-patterns-book/llms-full.txt) - Full content for RAG ingestion

## License

MIT - See [Attribution](attribution.md) for book credits.
