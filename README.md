# Agentic Patterns Book

**[Documentation](https://runyaga.github.io/agentic-patterns-book/)** |
**[Patterns](https://runyaga.github.io/agentic-patterns-book/patterns/)** |
**[Getting Started](https://runyaga.github.io/agentic-patterns-book/getting-started/)**

[![CI](https://github.com/runyaga/agentic-patterns-book/actions/workflows/ci.yml/badge.svg)](https://github.com/runyaga/agentic-patterns-book/actions/workflows/ci.yml)
[![Documentation](https://github.com/runyaga/agentic-patterns-book/actions/workflows/docs.yml/badge.svg)](https://runyaga.github.io/agentic-patterns-book/)

**Idiomatic pydantic-ai implementations of agentic design patterns.**

> **Vibe Coded** - This repository was built using Claude Code with
> [haiku-rag](https://github.com/ggozad/haiku.rag) as the RAG mechanism
> for reading the source material. See [Reproduce This](https://runyaga.github.io/agentic-patterns-book/reproduce/)
> for the complete workflow.

## About

A Python port of 15 agentic AI patterns from the **Gulli** book:

> **Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems**
> by Antonio Gulli (Springer, 2025)
>
> [Purchase from Springer](https://link.springer.com/book/10.1007/978-3-032-01402-3)

Each pattern is implemented as a standalone module with type-safe Pydantic
models, async-first design, comprehensive tests (80%+ branch coverage), and
runnable examples.

## Documentation

Full documentation: **https://runyaga.github.io/agentic-patterns-book/**

- [Getting Started](https://runyaga.github.io/agentic-patterns-book/getting-started/)
- [Pattern Documentation](https://runyaga.github.io/agentic-patterns-book/patterns/)
- [Reproduce This Repository](https://runyaga.github.io/agentic-patterns-book/reproduce/)

## Quick Start

```bash
git clone https://github.com/runyaga/agentic-patterns-book.git
cd agentic-patterns-book

# Install
uv venv && uv pip install -e ".[dev]" --python .venv/bin/python

# Run a pattern
.venv/bin/python -m agentic_patterns.prompt_chaining
```

## Requirements

- Python 3.14+
- Ollama running locally (`http://localhost:11434`)

## Implemented Patterns

| Pattern | Chapter | Description |
|---------|---------|-------------|
| [Prompt Chaining](src/agentic_patterns/prompt_chaining.py) | 1 | Sequential LLM calls |
| [Routing](src/agentic_patterns/routing.py) | 2 | Intent classification and dispatch |
| [Parallelization](src/agentic_patterns/parallelization.py) | 3 | Concurrent execution: sectioning, voting, map-reduce |
| [Reflection](src/agentic_patterns/reflection.py) | 4 | Producer-critic self-improvement loop |
| [Tool Use](src/agentic_patterns/tool_use.py) | 5 | External tools (weather, calculator, search) |
| [Planning](src/agentic_patterns/planning.py) | 6 | Goal decomposition into steps |
| [Multi-Agent](src/agentic_patterns/multi_agent.py) | 7 | Supervisor and network collaboration |
| [Memory](src/agentic_patterns/memory.py) | 8 | Buffer, window, and summary memory |
| [Learning](src/agentic_patterns/learning.py) | 9 | Experience-based adaptation |
| [MCP Integration](src/agentic_patterns/mcp_integration.py) | 10 | Model Context Protocol tool usage |
| [Exception Recovery](src/agentic_patterns/exception_recovery.py) | 12 | Phoenix Protocol: Clinic agent for error diagnosis |
| [Human-in-Loop](src/agentic_patterns/human_in_loop.py) | 13 | Escalation and approval workflows |
| [Knowledge Retrieval](src/agentic_patterns/knowledge_retrieval.py) | 14 | RAG pipeline |
| [Agent Marketplace](docs/patterns/15-agent-marketplace.md) | 15 | (Spec) The Agora: Decentralized bidding |
| [Resource-Aware](src/agentic_patterns/resource_aware.py) | 16 | Budget management and model selection |
| [Reasoning Weaver](docs/patterns/17-reasoning-weaver.md) | 17 | (Spec) Tree of Thoughts topology |
| [Guardrails](src/agentic_patterns/guardrails.py) | 18 | Input/output validation and safety |
| [Evaluation](src/agentic_patterns/evaluation.py) | 19 | Performance metrics and monitoring |
| [Prioritization](src/agentic_patterns/prioritization.py) | 20 | Task ranking and priority queues |
| [Domain Exploration](docs/patterns/21-domain-exploration.md) | 21 | (Spec) The Cartographer: Knowledge graph mapping |

## Development

```bash
# Run tests
uv run pytest

# Lint (must pass with zero warnings)
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Build docs locally
uv pip install -e ".[docs]" --python .venv/bin/python
mkdocs serve
```

## Integration Tests

Run all patterns against a live Ollama instance:

```bash
./scripts/integration_test.sh
```

**Runtime:** ~20-30 minutes for all 15 patterns (with retries). Individual
patterns take 30-120 seconds depending on complexity.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `REQUIRED_MODEL` | `gpt-oss:20b` | Model to use |
| `RETRY_COUNT` | `2` | Retries per pattern |
| `TIMEOUT_SECS` | `120` | Timeout per pattern |
| `LOGFIRE_TOKEN` | (unset) | Enable observability |

Example with remote Ollama:

```bash
OLLAMA_URL=http://remote-host:11434 ./scripts/integration_test.sh
```

## Observability

This project uses [Logfire](https://logfire.pydantic.dev/) for tracing agent
runs. When `LOGFIRE_TOKEN` is set, all pydantic-ai calls are instrumented.

### Setup

1. Create a project at [logfire.pydantic.dev](https://logfire.pydantic.dev/)
2. Get your write token from project settings
3. Export it:

```bash
export LOGFIRE_TOKEN=your_token_here
```

### Logfire MCP Server (Claude Code)

Query your traces directly from Claude Code:

```bash
# Get a read token from Logfire project settings
claude mcp add logfire -e LOGFIRE_READ_TOKEN=your_read_token -- uvx logfire-mcp@latest
```

Then ask Claude Code questions like "show me the last 10 agent runs" or "what
errors occurred in the last hour".

## Project Structure

```
src/agentic_patterns/
├── __init__.py           # Exports get_model()
├── _models.py            # Shared model configuration
└── *.py                  # Pattern implementations

tests/
├── conftest.py           # Shared fixtures
└── test_*.py             # Pattern tests

docs/                     # MkDocs site
dev/                      # Developer/maintainer docs
```

## Attribution

This project is a port of patterns from *Agentic Design Patterns: A Hands-On
Guide to Building Intelligent Systems* by Antonio Gulli.

See [docs/attribution.md](docs/attribution.md) for full credits.

## License

MIT
