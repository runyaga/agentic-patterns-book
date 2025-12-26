# Chapter 5: Tool Use

Enable agents to execute external functions (APIs, DBs, calculations).

## Implementation

Source: `src/agentic_patterns/tool_use.py`

### Tool Definitions

PydanticAI uses decorators to register tools.

```python
@dataclass
class ToolDependencies:
    weather_data: dict[str, dict] | None = None

# Agent with dependencies
tool_agent = Agent(
    model,
    deps_type=ToolDependencies,
    system_prompt="Use tools to answer questions...",
)

# Async Tool (with context)
@tool_agent.tool
async def get_weather(ctx: RunContext[ToolDependencies], location: str) -> WeatherResult:
    """Get weather for a location."""
    data = ctx.deps.weather_data.get(location)
    return WeatherResult(temp=data["temp"], ...)

# Sync Tool (plain function)
@tool_agent.tool_plain
def calculate(expression: str) -> float:
    """Safe calculation tool."""
    return eval(expression, {"__builtins__": {}}, safe_math_funcs)
```

### Execution

```python
async def run_tool_agent(query: str):
    deps = ToolDependencies(weather_data={...})
    # Agent decides when to call tools based on query
    result = await tool_agent.run(query, deps=deps)
    return result.output
```

## Use Cases

- **Data Retrieval**: Database queries, Search APIs, File reading.
- **Computation**: Math, Date/Time, Data transformation.
- **Action Execution**: Sending emails, Posting to Slack, Updating records.

## When to Use

- Tasks require capabilities beyond text generation (math, current data).
- Real-time information is needed (weather, stock prices).
- Interaction with external systems is required.

## Testing

```python
async def test_weather_tool():
    ctx = MagicMock(deps=ToolDependencies(weather_data={"London": ...}))
    result = await get_weather(ctx, "London")
    assert result.temp == 15.0
```

## Example

```bash
.venv/bin/python -m agentic_patterns.tool_use
```
