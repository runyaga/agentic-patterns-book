# Chapter 4: Reflection

Self-correction via automated validation loops.

## Implementation

Source: [`src/agentic_patterns/reflection.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/reflection.py)

### Data Models & Dependencies

```python
--8<-- "src/agentic_patterns/reflection.py:models"
```

### Agents with Output Validator

```python
--8<-- "src/agentic_patterns/reflection.py:agents"
```

### Reflection Execution

```python
--8<-- "src/agentic_patterns/reflection.py:reflection"
```

## Use Cases

- **Code Generation**: Validator runs unit tests; retries on failure.
- **Content Writing**: Validator checks style guidelines; retries on violations.
- **Data Extraction**: Validator checks schema constraints; retries on mismatches.

## When to Use

- You have a clear "pass/fail" or scoring criteria.
- You want to keep your main application logic linear (`result = run()`) rather than looping.
- You want the model to see *why* it failed (the `ModelRetry` message becomes conversation history).

## Example

```bash
.venv/bin/python -m agentic_patterns.reflection
```
