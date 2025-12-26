# Chapter 5: Tool Use

Enable agents to execute external functions (APIs, DBs, calculations).

## Implementation

Source: [`src/agentic_patterns/tool_use.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/tool_use.py)

### Data Models

```python
--8<-- "src/agentic_patterns/tool_use.py:models"
```

### Agent Definition

```python
--8<-- "src/agentic_patterns/tool_use.py:agent"
```

### Tool Definitions

```python
--8<-- "src/agentic_patterns/tool_use.py:tools"
```

### Execution

```python
--8<-- "src/agentic_patterns/tool_use.py:run"
```

## Use Cases

- **Data Retrieval**: Database queries, Search APIs, File reading.
- **Computation**: Math, Date/Time, Data transformation.
- **Action Execution**: Sending emails, Posting to Slack, Updating records.

## When to Use

- Tasks require capabilities beyond text generation (math, current data).
- Real-time information is needed (weather, stock prices).
- Interaction with external systems is required.

## Example

```bash
.venv/bin/python -m agentic_patterns.tool_use
```
