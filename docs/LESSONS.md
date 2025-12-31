# Lessons Learned

## Agent Marketplace (Ch 15)

### pydantic_graph End nodes
- `End[T]` returns data via `.data` attribute, not `.output`
- When testing graph nodes directly, assert on `result.data.field`
- The graph's `.run()` method returns `.output` (the unwrapped data)

### Parallel bid collection
- Use `asyncio.gather(*tasks)` for concurrent bidding
- Wrap each agent call in `asyncio.wait_for()` with timeout
- Filter `None` results after gather (timeouts/errors)

### Weighted scoring for selection
- Combine confidence (agent's self-assessment) with skill match
- Default weights: 60% confidence, 40% skill overlap
- Skill match = |intersection| / |required_skills|

### Test patterns for graph nodes
- Mock `GraphRunContext` with `MagicMock()`
- Set `ctx.state` to test state
- Call `await node.run(ctx)` directly
- Check return type (next node class or `End`)

### Selection Strategies (Ch 15b)
- Use `@runtime_checkable` for Protocol when `isinstance()` checks needed
- Strategy pattern: async `select()` method returns winner from bids
- Default strategy via `AgoraState.strategy` field with `None` default
- Fallback to default in node: `strategy = ctx.state.strategy or Default()`
- `@dataclass` for strategies with configurable weights (WeightedScoreStrategy)

### Protocol compliance testing
- `isinstance(strategy, SelectionStrategy)` works with `@runtime_checkable`
- Test each strategy's `select()` returns expected bid or `None`
- Mock LLM calls in AgentJudgmentStrategy tests

### Production Features (Ch 15c)

#### Callbacks for observability
- Define callback type aliases: `Callable[[T], Awaitable[None]]`
- Group in `@dataclass` with `None` defaults for optional callbacks
- Fire callbacks with `if callback: await callback(data)`
- Useful for logging, metrics, debugging without coupling

#### Load balancing
- Add `max_concurrent` and `current_load` to capability model
- Use `@property` for computed fields (`is_available`, `available_capacity`)
- Increment load before execution, decrement in `finally` block
- `CapacityAwareStrategy` factors available capacity into scoring

#### Output validators for realistic LLM responses
- LLMs ignore numeric rules in prompts (e.g., "confidence = skills/total")
- Use `@agent.output_validator` to enforce calculations server-side
- Guard against `ctx.deps is None` in validators
- `ModelRetry` to reject invalid outputs (e.g., bidding with 0 skill match)
- Cap confidence at actual skill match ratio for predictable behavior

#### Demo design for marketplace patterns
- Overlapping skills between agents creates competitive bidding
- Avoid "perfect fit" demos where one agent always wins
- Show skill match calculation in output to explain winner selection
- Longer timeouts (30s) for slower LLM servers

---

# TODOs

- [ ] Reimplement parallelization pattern as a graph
- [ ] Update knowledge_retrieval to use pydantic-ai embeddings API
- [ ] Update evaluation pattern to use pydantic-ai evaluations framework
- [x] ~~Implement Agora Milestone 2: SelectionStrategy protocol~~
- [x] ~~Implement Agora Milestone 3: Callbacks and load balancing~~
