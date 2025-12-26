# Agentic Patterns Book

[![CI](https://github.com/runyaga/agentic-patterns-book/actions/workflows/ci.yml/badge.svg)](https://github.com/runyaga/agentic-patterns-book/actions/workflows/ci.yml)
[![Documentation](https://github.com/runyaga/agentic-patterns-book/actions/workflows/docs.yml/badge.svg)](https://runyaga.github.io/agentic-patterns-book/)

**Idiomatic pydantic-ai implementations of agentic design patterns.**

> **Vibe Coded** - This repository was built using Claude Code with
> [haiku-rag](https://github.com/anthropics/haiku-rag) as the RAG mechanism
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
| [Human-in-Loop](src/agentic_patterns/human_in_loop.py) | 13 | Escalation and approval workflows |
| [Knowledge Retrieval](src/agentic_patterns/knowledge_retrieval.py) | 14 | RAG pipeline |
| [Resource-Aware](src/agentic_patterns/resource_aware.py) | 16 | Budget management and model selection |
| [Guardrails](src/agentic_patterns/guardrails.py) | 18 | Input/output validation and safety |
| [Evaluation](src/agentic_patterns/evaluation.py) | 19 | Performance metrics and monitoring |
| [Prioritization](src/agentic_patterns/prioritization.py) | 20 | Task ranking and priority queues |

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
