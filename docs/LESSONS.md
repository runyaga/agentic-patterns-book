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

---

# TODOs

- [ ] Reimplement parallelization pattern as a graph
- [ ] Update knowledge_retrieval to use pydantic-ai embeddings API
- [ ] Update evaluation pattern to use pydantic-ai evaluations framework
- [x] ~~Implement Agora Milestone 2: SelectionStrategy protocol~~
- [ ] Implement Agora Milestone 3: Callbacks and load balancing
