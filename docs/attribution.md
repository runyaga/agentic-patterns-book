# Attribution

## Source Material

This project implements patterns from the book by **Gulli**:

> **Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems**
>
> by Antonio Gulli
>
> Springer, 2025
>
> [https://link.springer.com/book/10.1007/978-3-032-01402-3](https://link.springer.com/book/10.1007/978-3-032-01402-3)

All concepts and pattern descriptions are adapted from this work into working
Python code using the pydantic-ai framework.

## Implementation

The patterns in this repository are **original implementations**, not direct
copies of any code from the source material. The Gulli book provides conceptual
frameworks and descriptions; this project translates those concepts into:

- Production-ready Python code
- pydantic-ai agent implementations
- Comprehensive test suites
- Runnable examples

## Framework

- **pydantic-ai**: The agent framework used for all implementations
  - Repository: https://github.com/pydantic/pydantic-ai
  - License: MIT

## LLM Provider

Default configuration uses Ollama for local LLM inference:

- **Ollama**: https://ollama.ai
- **Default Model**: gpt-oss:20b

## Dependencies

See `pyproject.toml` for full dependency list. Key dependencies:

- `pydantic-ai` - Agent framework
- `haiku-rag` - RAG utilities
- `pytest` - Testing framework
- `ruff` - Linting and formatting

## License

This implementation is released under the MIT License.

The original book content remains under its original copyright. This repository
contains only original code implementations inspired by the patterns described
in the Gulli work.
