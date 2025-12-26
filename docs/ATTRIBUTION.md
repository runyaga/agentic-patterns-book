# Attribution

## Source Material

This project implements patterns from the **"Agentic Design Patterns"** book,
adapting the concepts and examples into working Python code using the
pydantic-ai framework.

### Original Work

- **Book**: Agentic Design Patterns
- **Content**: Conceptual patterns for building agentic AI systems
- **Adaptation**: All code in this repository is original implementation
  inspired by the book's pattern descriptions

## Implementation

The patterns in this repository are **original implementations**, not direct
copies of any code from the source material. The book provides conceptual
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

This implementation is released under the MIT License. See the root LICENSE
file for details.

The original "Agentic Design Patterns" book content remains under its original
copyright. This repository contains only original code implementations inspired
by the patterns described in that work.
