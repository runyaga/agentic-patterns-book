# Chapter 8: Memory Management

Maintain conversation context across interactions using different storage strategies.

## Implementation

Source: `src/agentic_patterns/memory.py`

### Memory Strategies

1.  **BufferMemory**: Stores complete history. Simplest, but hits token limits fast.
2.  **WindowMemory**: Keeps last $N$ messages. Efficient, but loses distant context.
3.  **SummaryMemory**: Summarizes old messages, keeps recent ones. Balanced.

### Idiomatic Pattern (Dependency Injection)

PydanticAI allows decoupling memory management from the run loop using `Deps` and `@system_prompt`.

```python
@dataclass
class MemoryDeps:
    """Dependencies for conversational agent."""
    memory: BufferMemory | WindowMemory | SummaryMemory

# Agent definition with dependency type
conversational_agent = Agent(
    model,
    deps_type=MemoryDeps,
    system_prompt="You are a helpful assistant."
)

@conversational_agent.system_prompt
def add_memory_context(ctx: RunContext[MemoryDeps]) -> str:
    """The Agent pulls its own history from deps automatically."""
    context = ctx.deps.memory.get_context()
    return f"Conversation history:\n{context}"
```

### Execution

The caller only provides the user input and the dependency object.

```python
async def chat_with_memory(deps: MemoryDeps, user_input: str) -> str:
    # 1. Update memory state
    deps.memory.add_user_message(user_input)
    
    # 2. Run agent (context is injected via @system_prompt hook)
    result = await conversational_agent.run(user_input, deps=deps)
    
    # 3. Update memory with AI response
    deps.memory.add_ai_message(result.output)
    return result.output
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

```