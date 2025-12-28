# Production Reality Check: Prompt Chaining

**Target file**: `docs/patterns/01-prompt-chaining.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- Tasks naturally decompose into sequential steps with clear data dependencies
- Intermediate results need validation or structured formatting before proceeding
- Complex reasoning benefits from breaking down into smaller, verifiable steps
- Pipeline-style workflows where each step transforms the output

### When NOT to Use
- Single-shot queries that don't need intermediate processing
- When latency is critical (each chain link adds a full LLM round-trip)
- Simple transformations that could be done with string formatting or regex
- When steps don't have meaningful dependencies (use Parallelization instead)

### Production Considerations
- **Observability**: Log each chain step separately for debugging failed pipelines
- **Failure handling**: Decide retry strategy - per-step retry vs. full-chain restart
- **Cost**: Each step is a separate API call; multiply token costs by chain length
- **Latency**: Total latency = sum of all step latencies; consider caching for
  repeated inputs
- **State management**: Consider persisting intermediate results for long chains
  to enable resumption after failures

---

## Review

**Date**: 2025-12-27
**Reviewer**: Gemini Agent
**Status**: Verified
**Comments**: 
1. **Accuracy**: The "When to Use" criteria perfectly match the `run_prompt_chain` implementation which uses `ChainDeps` to maintain state across three distinct, sequential agents (Summarizer, Trend Analyzer, Email Drafter).
2. **Completeness**: The "When NOT to Use" section correctly identifies that Parallelization is better when dependencies are absent, which is a key distinction from this sequential pattern.
3. **Practicality**: The "Production Considerations" are realistic, especially the "State management" point, given that `ChainDeps` in the code acts as an in-memory version of exactly what's described.
4. **Consistency**: Matches the template in `documentation.md`.
