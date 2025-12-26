# Chapter 3: Parallelization

Execute independent sub-tasks concurrently (`asyncio.gather`) to reduce latency.

## Implementation

Source: [`src/agentic_patterns/parallelization.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/parallelization.py)

### Data Models

```python
--8<-- "src/agentic_patterns/parallelization.py:models"
```

### Agents

```python
--8<-- "src/agentic_patterns/parallelization.py:agents"
```

### Parallelization Patterns

```python
--8<-- "src/agentic_patterns/parallelization.py:patterns"
```

## Use Cases

- **Sectioning**: Research (History, Pros/Cons), Content generation (Intro, Body, Conclusion).
- **Voting**: Fact-checking, Content safety classification, Creative brainstorming (best of N).
- **Map-Reduce**: Log analysis, Document summarization, Batch data extraction.

## When to Use

- **Sectioning**: Task divides into distinct, independent sub-topics.
- **Voting**: High accuracy needed; models may hallucinate or vary.
- **Map-Reduce**: Large datasets where items can be processed individually first.

## Example

```bash
.venv/bin/python -m agentic_patterns.parallelization
```
