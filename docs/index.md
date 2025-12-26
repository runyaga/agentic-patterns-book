# Agentic Patterns Book

**Idiomatic pydantic-ai implementations of agentic design patterns.**

This is a **vibe-coded** repository - built using Claude Code with
[haiku-rag](https://github.com/anthropics/haiku-rag) as the RAG mechanism
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

A Python port of 15 agentic AI patterns from the **Gulli** book:

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

| Chapter | Pattern | Description |
|---------|---------|-------------|
| 1 | [Prompt Chaining](patterns/01-prompt-chaining.md) | Sequential LLM calls |
| 2 | [Routing](patterns/02-routing.md) | Intent classification |
| 3 | [Parallelization](patterns/03-parallelization.md) | Concurrent execution |
| 4 | [Reflection](patterns/04-reflection.md) | Self-improvement loops |
| 5 | [Tool Use](patterns/05-tool-use.md) | External tool integration |
| 6 | [Planning](patterns/06-planning.md) | Goal decomposition |
| 7 | [Multi-Agent](patterns/07-multi-agent.md) | Agent collaboration |
| 8 | [Memory](patterns/08-memory.md) | Conversation context |
| 9 | [Learning](patterns/09-learning.md) | Experience-based adaptation |
| 13 | [Human-in-Loop](patterns/13-human-in-loop.md) | Human oversight |
| 14 | [Knowledge Retrieval](patterns/14-knowledge-retrieval.md) | RAG pipelines |

## Reproduce This

Want to build your own pattern implementations? See
[Reproduce This](reproduce.md) for the complete workflow including the
haiku-rag setup used to create this repository.

## LLM-Friendly Documentation

This site provides machine-readable documentation for AI assistants:

- [llms.txt](/llms.txt) - Concise site index
- [llms-full.txt](/llms-full.txt) - Full content for RAG ingestion

## License

MIT - See [Attribution](attribution.md) for book credits.
