# Claude Code Instructions

Project-specific instructions for agentic-patterns.

## Project Purpose

Port of agentic design patterns from "Agentic Design Patterns" book to pydantic-ai.
Each module implements a pattern with working code and tests.

## Tech Stack

- **Python**: 3.14+
- **Agent Framework**: pydantic-ai
- **LLM Provider**: Ollama (gpt-oss:20b default, configurable)
- **Tests**: pytest with 80% coverage requirement

## Key Commands

```bash
# Install dev dependencies
uv pip install -e ".[dev]" --python .venv/bin/python

# Run tests with coverage
uv run pytest

# Lint (must pass with zero warnings)
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Run a specific pattern example
.venv/bin/python -m agentic_patterns.prompt_chaining
.venv/bin/python -m agentic_patterns.routing
```

## Code Style

- Line length: 79 chars (ruff enforced)
- Single-line imports (isort force-single-line)
- Type hints required for public APIs
- Docstrings for public functions

## Test Requirements

- 80% branch coverage (`--cov-fail-under=80`)
- Tests in `tests/` directory (separate from source)
- Use pytest fixtures from `tests/conftest.py`
- Async tests with `pytest-asyncio`

## Project Layout

```
src/agentic_patterns/
├── __init__.py
├── _models.py           # Shared model configuration
├── prompt_chaining.py   # Chapter 1: Prompt Chaining
├── routing.py           # Chapter 2: Routing
├── parallelization.py   # Chapter 3: Parallelization (pending)
└── ...

tests/
├── conftest.py
├── test_prompt_chaining.py
├── test_routing.py
└── ...
```

## Pattern Implementation Workflow

1. Query agent-book MCP for chapter content
2. Design Pydantic models for inputs/outputs
3. Implement in `src/agentic_patterns/{pattern}.py`
4. Write tests in `tests/test_{pattern}.py`
5. Run `uv run ruff check --fix` and `uv run pytest`

## Reference Documentation

- `/docs/specs/pattern-implementation.md` - Implementation spec
- `/docs/LESSONS.md` - Lessons learned
