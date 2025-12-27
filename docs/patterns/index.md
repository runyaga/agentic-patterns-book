# Agentic Design Patterns

Documentation for each implemented pattern from the "Agentic Design Patterns"
book, ported to pydantic-ai.

See [Idioms](../idioms.md) for pydantic-ai, pydantic_graph, and pydantic-evals patterns.

## Pattern Index

| Chapter | Pattern | Description |
|---------|---------|-------------|
| 1 | [Prompt Chaining](01-prompt-chaining.md) | Chain LLM calls sequentially |
| 2 | [Routing](02-routing.md) | Classify and route to handlers |
| 3 | [Parallelization](03-parallelization.md) | Execute tasks concurrently |
| 4 | [Reflection](04-reflection.md) | Self-evaluate and improve |
| 5 | [Tool Use](05-tool-use.md) | Interact with external tools |
| 6 | [Planning](06-planning.md) | Decompose goals into steps |
| 7 | [Multi-Agent](07-multi-agent.md) | Coordinate specialized agents |
| 8 | [Memory](08-memory.md) | Maintain conversation context |
| 9 | [Learning](09-learning.md) | Improve through experience |
| 13 | [Human-in-the-Loop](13-human-in-loop.md) | Integrate human oversight |
| 14 | [Knowledge Retrieval](14-knowledge-retrieval.md) | RAG pattern |

## Pattern Complexity

Patterns build on each other:

```
Ch 1: Prompt Chaining (foundation)
  |
  +-> Ch 2: Routing
  +-> Ch 3: Parallelization
  +-> Ch 4: Reflection
  |
  +-> Ch 5: Tool Use (depends on Ch 1-2)
      |
      +-> Ch 6: Planning (depends on Ch 1-5)
          |
          +-> Ch 7: Multi-Agent (depends on Ch 1-6)
              |
              +-> Ch 13: Human-in-Loop (depends on Ch 1-7)

Ch 8: Memory (standalone, depends on Ch 1)
Ch 9: Learning (depends on Ch 4, 8)
Ch 14: Knowledge Retrieval (depends on Ch 5)
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
