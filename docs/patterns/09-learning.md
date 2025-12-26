# Chapter 9: Learning and Adaptation

Improve performance over time by storing experiences (Few-Shot) and adapting prompts.

## Implementation

Source: [`src/agentic_patterns/learning.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/learning.py)

### Data Models & Experience Store

```python
--8<-- "src/agentic_patterns/learning.py:models"
```

### Agents with Example Injection

```python
--8<-- "src/agentic_patterns/learning.py:agents"
```

### Learning & Adaptation Logic

```python
--8<-- "src/agentic_patterns/learning.py:learning"
```

## Use Cases

- **Style Adaptation**: Mimic user's writing style based on past edits.
- **Code Correction**: Learn from linter errors to avoid repeating them.
- **Classification**: Improve label accuracy by providing "gold" examples.

## When to Use

- Tasks are repetitive and belong to distinct categories.
- Quality feedback is available (explicit or implicit).
- Zero-shot performance is insufficient, but Few-shot is promising.

## Example

```bash
.venv/bin/python -m agentic_patterns.learning
```
