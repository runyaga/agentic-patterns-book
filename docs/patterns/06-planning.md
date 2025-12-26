# Chapter 6: Planning

Dynamically decompose goals into steps, execute them, and adapt to failures.

## Implementation

Source: [`src/agentic_patterns/planning.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/planning.py)

### Data Models

```python
--8<-- "src/agentic_patterns/planning.py:models"
```

### Agents

```python
--8<-- "src/agentic_patterns/planning.py:agents"
```

### Planning & Execution Logic

```python
--8<-- "src/agentic_patterns/planning.py:planning"
```

## Use Cases

- **Complex Research**: Decompose "Write report on X" into "Search", "Read", "Outline", "Write".
- **Code Refactoring**: "Analyze file", "Plan changes", "Apply edits", "Run tests".
- **Multi-step Analysis**: "Fetch data", "Clean data", "Run stats", "Visualize".

## When to Use

- Goal is too complex for a single prompt ("one-shot").
- Steps have strict dependencies (B needs output of A).
- Error recovery is needed (if step 2 fails, try step 2b).
- Transparency in process is required.

## Example

```bash
.venv/bin/python -m agentic_patterns.planning
```
