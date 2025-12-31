# Lessons Learned

Notes from implementing agentic design patterns with pydantic-ai.

## Testing pydantic-ai Agents

### Mock the dict, not just the agent

When agents are stored in a lookup dict (like `INTENT_HANDLERS`), patching the
module-level agent variable doesn't work because the dict already holds
references to the original agents.

```python
# WRONG - handler lookup still gets original agent
with patch("module.order_status_agent") as mock:
    ...

# RIGHT - patch the dict entry directly
with patch.dict("module.INTENT_HANDLERS", {Intent.ORDER_STATUS: mock_handler}):
    ...
```

### Mock result structure

pydantic-ai agent.run() returns a result object with `.output` attribute:

```python
mock_result = MagicMock()
mock_result.output = YourPydanticModel(...)
mock_agent.run = AsyncMock(return_value=mock_result)
```

## Code Style

### Union types

Use `X | Y` syntax instead of `Union[X, Y]`:

```python
# Preferred
RouteResponse = OrderResponse | ProductResponse | SupportResponse

# Avoid
RouteResponse = Union[OrderResponse, ProductResponse, SupportResponse]
```

### String concatenation for long prompts

System prompts exceeding 79 chars should use string concatenation:

```python
system_prompt=(
    "You are an intent classifier for a customer service system. "
    "Analyze the user's query and determine their intent.\n\n"
    "Classify into one of these categories:\n"
    "- order_status: Questions about order tracking\n"
)
```

## Project Structure

### Avoid deep nesting

Instead of `chapters/chapter1/prompt_chaining/prompt_chaining.py`, use a
flatter structure:

```
src/agentic_patterns/
    prompt_chaining.py
    routing.py
    parallelization.py
```

### Keep tests separate

Standard Python layout with tests outside src:

```
src/agentic_patterns/...
tests/
    conftest.py
    test_prompt_chaining.py
    test_routing.py
```

## pydantic-ai Specifics

### Model configuration

Use OpenAIChatModel with OpenAIProvider for Ollama compatibility:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    model_name="gpt-oss:20b",
    provider=OpenAIProvider(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
)
```

### Structured outputs

Use Pydantic models for type-safe agent outputs:

```python
class RouteDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)

agent = Agent(model, output_type=RouteDecision)
```

### Tool decorators

Use `@agent.tool` for async tools needing context, `@agent.tool_plain` for sync:

```python
@tool_agent.tool
async def get_weather(
    ctx: RunContext[ToolDependencies],
    location: str,
) -> WeatherResult:
    weather_data = ctx.deps.weather_data or DEFAULT_DATA
    ...

@tool_agent.tool_plain
def calculate(expression: str) -> CalculationResult:
    # No context needed, synchronous
    ...
```

### Testing decorated tool functions

Decorated `@tool` functions need direct testing with mock RunContext for full
coverage. Mocking only the agent doesn't cover the tool function bodies:

```python
@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.deps = ToolDependencies(weather_data={"berlin": {...}})
    return ctx

@pytest.mark.asyncio
async def test_get_weather_direct(mock_context):
    from module import get_weather
    result = await get_weather(mock_context, "Berlin")
    assert result.temperature == 5.0
```

### Dependency injection with dataclasses

Use `@dataclass` for agent dependencies, not Pydantic models:

```python
@dataclass
class ToolDependencies:
    weather_data: dict[str, dict] | None = None
    search_data: dict[str, list[str]] | None = None

agent = Agent(model, deps_type=ToolDependencies)
```

## Pattern-Specific Lessons

### Reflection Pattern (Producer-Critic)

Iterative refinement needs clear stopping criteria:

```python
class Critique(BaseModel):
    is_satisfactory: bool  # Clear boolean for loop control
    issues: list[str]
    suggestions: list[str]

async def run_reflection(prompt: str, max_iterations: int = 3):
    for i in range(max_iterations):
        critique = await critic_agent.run(...)
        if critique.output.is_satisfactory:
            break
        draft = await refiner_agent.run(...)
```

### Tool Use Pattern (Safe Eval)

For calculator tools, restrict eval to safe operations:

```python
allowed_names = {
    "abs": abs, "round": round, "min": min,
    "max": max, "pow": pow, "sum": sum,
}
result = eval(expression, {"__builtins__": {}}, allowed_names)
```

### Planning Pattern (Dependencies)

Track step dependencies and skip steps with unmet dependencies:

```python
class PlanStep(BaseModel):
    step_number: int
    dependencies: list[int] = Field(default_factory=list)
    status: StepStatus = Field(default=StepStatus.PENDING)

# During execution
deps_met = all(
    any(r.step_number == d and r.success for r in step_results)
    for d in step.dependencies
)
if not deps_met:
    step_results.append(StepResult(
        step_number=step.step_number,
        success=False,
        output="Skipped: dependencies not met",
    ))
```

### Multi-Agent Coordination

Use separate specialized agents rather than one complex agent:

```python
planner_agent = Agent(model, output_type=Plan, system_prompt="...")
executor_agent = Agent(model, output_type=StepResult, system_prompt="...")
replanner_agent = Agent(model, output_type=Plan, system_prompt="...")
synthesizer_agent = Agent(model, output_type=str, system_prompt="...")
```

### Exception Recovery (Deterministic Heuristics)

For 90% of failures (timeouts, rate limits, context overflow), a simple regex
check on the exception message is faster, cheaper, and more reliable than
sending the stack trace to an LLM "Clinic Agent".

**Pattern:**
1. Check exception type/message against known patterns.
2. If matched, apply hardcoded fix (backoff, truncate, retry).
3. Only invoke LLM Diagnosis if the error is truly `UNKNOWN`.

```python
# Fast path - no tokens used
if "rate limit" in str(exc).lower():
    return RecoveryAction(action="wait_and_retry", wait=60)

# Slow path - only for weird errors
return await clinic_agent.run(f"Diagnose: {exc}")
```

### MCP Integration (Native Support)

Don't over-engineer MCP connections. `pydantic-ai` handles the complexity of
connection management, tool discovery, and request forwarding natively.

**Anti-Pattern (Custom Connector):**
```python
class UniversalConnector:
    def connect(self): ...
    def list_tools(self): ...
```

**Idiomatic Pattern:**
```python
server = MCPServerStdio("python", args=["server.py"])
agent = Agent(model, toolsets=[server])
```

Use `tool_prefix` to namespace tools from different servers (e.g., `fs_read`
vs `db_read`) and `process_tool_call` to inject agent dependencies into the
MCP request context.

## Documentation

### Always build mkdocs locally before push

The CI runs `mkdocs build --strict` which fails on warnings. Always validate
locally before pushing:

```bash
uv run mkdocs build --strict
```

Common failures:
- **Broken source links**: Use absolute GitHub URLs for source file links, not
  relative paths like `../../src/...`. MkDocs treats relative `.py` links as
  documentation files.
- **Missing nav entries**: New pages in `docs/patterns/` must be added to the
  `nav:` section in `mkdocs.yml`.

```markdown
# WRONG - mkdocs can't resolve relative paths to source files
[Source Code](../../src/agentic_patterns/foo.py)

# RIGHT - absolute GitHub URL
Source: [`src/agentic_patterns/foo.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/foo.py)
```

## Coverage Strategies

### Test both integration and unit levels

- Integration: Mock agents, test orchestration logic
- Unit: Test Pydantic models directly, test tool functions with mock context

```python
# Integration - mock agent
with patch("module.executor_agent") as mock:
    mock.run = AsyncMock(return_value=mock_result)
    result = await execute_plan(plan)

# Unit - direct model test
def test_plan_step_valid():
    step = PlanStep(step_number=1, description="Test", expected_output="Out")
    assert step.status == StepStatus.PENDING
```

## Domain Exploration (The Cartographer)

### Dependency Weights vs. Capabilities
For a lightweight "Discovery Agent," avoid heavy scientific libraries unless
strictly necessary.

- **Issue:** `networkx.pagerank` defaults to `scipy`, which is a 100MB+ binary
  dependency.
- **Solution:** Use simpler centrality metrics like `degree_centrality` (which is
  pure Python) or catch the import error and fallback gracefully.
- **Lesson:** If you import a library for one function, check its transitive
  dependencies.

### Atomic Persistence
When building agents that run for minutes/hours (like crawlers), file corruption
on interruption is a major risk.

- **Anti-Pattern:** `open(file, 'w').write(json)` directly.
- **Idiomatic Pattern:** Write to temp, then rename.
  ```python
  temp = path.with_suffix(path.suffix + ".tmp")
  temp.write_text(content)
  temp.replace(path)  # Atomic on POSIX
  ```

### Hybrid Extraction (AST + LLM)
Combining static analysis with LLM inference requires a strict "Source of Truth"
hierarchy.

- **AST:** Provides the *Skeleton* (Nodes, Links, Locations). It is 100%
  accurate but semantically shallow.
- **LLM:** Provides the *Flesh* (Summaries, Concepts). It is semantically rich
  but structurally hallucination-prone.
- **Strategy:** Let AST build the graph. Let LLM *decorate* the nodes. Do not
  let LLM invent new code nodes unless they are purely conceptual.
