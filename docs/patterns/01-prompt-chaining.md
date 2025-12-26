# Chapter 1: Prompt Chaining

Chain multiple LLM calls where each step's output becomes the next input.

## Implementation

Source: [`src/agentic_patterns/prompt_chaining.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/prompt_chaining.py)

### Data Models

```python
--8<-- "src/agentic_patterns/prompt_chaining.py:models"
```

### Agents with System Prompts

```python
--8<-- "src/agentic_patterns/prompt_chaining.py:agents"
```

### Chain Execution

```python
--8<-- "src/agentic_patterns/prompt_chaining.py:chain"
```

## Use Cases

- **Document Processing**: Extract -> Analyze -> Generate
- **Research Pipelines**: Gather -> Synthesize -> Report
- **Content Creation**: Outline -> Draft -> Refine
- **Data Transformation**: Parse -> Process -> Format

## When to Use

- Tasks naturally decompose into sequential steps
- Intermediate results need validation or structured formatting
- Complex reasoning benefits from breaking down the problem

## Example

```bash
.venv/bin/python -m agentic_patterns.prompt_chaining
```
