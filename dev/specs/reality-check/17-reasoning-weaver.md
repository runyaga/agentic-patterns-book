# Production Reality Check: Reasoning Weaver (Tree of Thoughts)

**Target file**: `docs/patterns/17-reasoning-weaver.md`
**Replaces**: `## Production Reality Check` section (expand existing)

---

## Production Reality Check

### When to Use
- Problem has a verifiable solution condition (can check if answer is correct)
- Greedy decoding fails (choosing most likely next word leads to dead ends)
- Lookahead and backtracking are valuable (puzzles, multi-step planning)
- Example: "Game of 24" - use 4, 7, 8, 8 to make 24

### When NOT to Use
- Open-ended chat or creative writing (no way to score/prune branches)
- Simple queries that single-shot handles well
- Latency budget is tight (ToT is 10x-100x slower than standard calls)
- Token budget is constrained (branching factor × depth = token explosion)

**Key insight**: ToT generates *many* tokens. Reserve it for problems where
greedy approaches demonstrably fail and correctness can be verified.

### Production Considerations
- **Model selection**: Use small, fast models (e.g., `qwen3:4b`) for generation
  steps. Use stronger models (or pure code) for evaluation. Route between them.
- **Token budgeting**: Branching factor × depth = total tokens. Set strict
  limits. Default to low values (branch=3, depth=3) and increase only if needed.
- **Pruning aggressiveness**: Prune early to save cost. But too aggressive
  pruning may discard the winning path. Tune per problem type.
- **Evaluation quality**: Bad evaluators lead to bad pruning. Invest in robust
  scoring logic - consider deterministic checks where possible.
- **Observability**: Log the full tree structure for debugging. "Why did it
  choose this path?" is hard to answer without tree visualization.
- **Fallback**: If ToT times out or exceeds budget, fall back to single-shot.
  Don't fail completely just because exploration was expensive.
