# Chapter 8: Memory Management

Maintain conversation context across interactions using different storage strategies.

## Implementation

Source: [`src/agentic_patterns/memory.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/memory.py)

### Memory Strategies

1.  **BufferMemory**: Stores complete history. Simplest, but hits token limits fast.
2.  **WindowMemory**: Keeps last $N$ messages. Efficient, but loses distant context.
3.  **SummaryMemory**: Summarizes old messages, keeps recent ones. Balanced.

### Data Models & Memory Classes

```python
--8<-- "src/agentic_patterns/memory.py:models"
```

### Agent with Memory Injection

```python
--8<-- "src/agentic_patterns/memory.py:agent"
```

### Memory-Enabled Chat

```python
--8<-- "src/agentic_patterns/memory.py:memory"
```

## Use Cases

- **Chatbots**: Maintaining user persona and preference context.
- **Long-running Sessions**: Coding assistants, RPG game masters.
- **Summarization**: Digesting long transcripts into key points.

## When to Use

| Type | Best For | Trade-off |
|------|----------|-----------|
| **Buffer** | Short, high-detail tasks | High token cost |
| **Window** | Infinite streams, reactive bots | Amnesia of old facts |
| **Summary** | Long-term coherent conversations | Lossy compression of details |

## Example

```bash
.venv/bin/python -m agentic_patterns.memory
```
