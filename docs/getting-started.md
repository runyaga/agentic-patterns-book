# Getting Started

## Prerequisites

- **Python 3.14+**
- **Ollama** running locally at `http://localhost:11434`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/runyaga/agentic-patterns-book.git
cd agentic-patterns-book
```

### 2. Set up Ollama

Install [Ollama](https://ollama.ai/) and pull the required model:

```bash
ollama pull gpt-oss:20b
```

The project uses `gpt-oss:20b` by default but any Ollama model works.

### 3. Create virtual environment and install

```bash
uv venv
uv pip install -e ".[dev]" --python .venv/bin/python
```

## Running Examples

Each pattern has a runnable `__main__` block:

```bash
# Prompt chaining: market research pipeline
.venv/bin/python -m agentic_patterns.prompt_chaining

# Routing: intent classification
.venv/bin/python -m agentic_patterns.routing

# Parallelization: sectioning, voting, map-reduce
.venv/bin/python -m agentic_patterns.parallelization

# Reflection: producer-critic improvement loop
.venv/bin/python -m agentic_patterns.reflection

# Tool use: weather, calculator, search tools
.venv/bin/python -m agentic_patterns.tool_use

# Planning: goal decomposition
.venv/bin/python -m agentic_patterns.planning

# Multi-agent: supervisor and network topologies
.venv/bin/python -m agentic_patterns.multi_agent

# Memory: buffer, window, summary patterns
.venv/bin/python -m agentic_patterns.memory

# Learning: experience-based adaptation
.venv/bin/python -m agentic_patterns.learning

# Human-in-loop: escalation workflows
.venv/bin/python -m agentic_patterns.human_in_loop

# Knowledge retrieval: RAG pipeline
.venv/bin/python -m agentic_patterns.knowledge_retrieval

# Resource-aware: budget management
.venv/bin/python -m agentic_patterns.resource_aware

# Guardrails: validation and safety
.venv/bin/python -m agentic_patterns.guardrails

# Evaluation: performance metrics
.venv/bin/python -m agentic_patterns.evaluation

# Prioritization: task ranking
.venv/bin/python -m agentic_patterns.prioritization
```

## Using in Your Code

```python
from agentic_patterns import get_model
from agentic_patterns.routing import router_agent, IntentClassification

# Get the configured model (Ollama gpt-oss:20b)
model = get_model()

# Use a pattern
result = await router_agent.run("I need help with my order")
classification: IntentClassification = result.output
print(f"Intent: {classification.intent}")
print(f"Confidence: {classification.confidence}")
```

## Development

### Run tests

```bash
uv run pytest
```

Tests require 80% branch coverage to pass.

### Lint and format

```bash
# Check for issues
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Configuration

### Custom model

Set `OLLAMA_MODEL` environment variable:

```bash
export OLLAMA_MODEL=llama3:8b
.venv/bin/python -m agentic_patterns.routing
```

### Custom Ollama URL

Set `OLLAMA_BASE_URL`:

```bash
export OLLAMA_BASE_URL=http://remote-server:11434/v1
```

## Project Structure

```
src/agentic_patterns/
├── __init__.py           # Exports get_model()
├── _models.py            # Model configuration
├── prompt_chaining.py    # Chapter 1
├── routing.py            # Chapter 2
├── parallelization.py    # Chapter 3
├── reflection.py         # Chapter 4
├── tool_use.py           # Chapter 5
├── planning.py           # Chapter 6
├── multi_agent.py        # Chapter 7
├── memory.py             # Chapter 8
├── learning.py           # Chapter 9
├── human_in_loop.py      # Chapter 13
├── knowledge_retrieval.py # Chapter 14
├── resource_aware.py     # Chapter 16
├── guardrails.py         # Chapter 18
├── evaluation.py         # Chapter 19
└── prioritization.py     # Chapter 20

tests/
├── conftest.py           # Shared fixtures
└── test_*.py             # Pattern tests
```

## Next Steps

- Browse [Patterns](patterns/index.md) to understand each implementation
- See [Reproduce This](reproduce.md) to build your own implementation
