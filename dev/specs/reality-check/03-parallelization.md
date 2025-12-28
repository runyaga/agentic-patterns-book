# Production Reality Check: Parallelization

**Target file**: `docs/patterns/03-parallelization.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- **Sectioning**: Task divides into distinct, independent sub-topics that can be
  processed simultaneously
- **Voting**: High accuracy needed and models may hallucinate; consensus reduces
  error rates
- **Map-Reduce**: Large datasets where items can be processed individually then
  aggregated

### When NOT to Use
- Tasks with sequential dependencies (use Prompt Chaining instead)
- When API rate limits would throttle parallel requests anyway
- Small tasks where parallelization overhead exceeds time savings
- When results must be strictly ordered (parallel execution is non-deterministic)

### Production Considerations
- **Rate limits**: `asyncio.gather` can spike concurrent requests. Use
  `asyncio.Semaphore` to cap concurrency and avoid 429 errors.
- **Cost**: Voting (N parallel calls) multiplies cost by N. Budget accordingly
  and consider when consensus is truly necessary vs. single-call sufficiency.
- **Error handling**: One failed task can fail `gather()`. Use
  `return_exceptions=True` or `asyncio.TaskGroup` with proper exception handling.
- **Aggregation quality**: Map-Reduce aggregation step may lose nuance. Test
  aggregator prompts carefully with diverse inputs.
- **Memory**: Large parallel batches load all results into memory simultaneously.
  For very large datasets, use streaming or chunked processing.

---

## Review

**Date**: 2025-12-27
**Reviewer**: Gemini Agent
**Status**: Verified
**Comments**: 
1. **Accuracy**: The implementation demonstrates all three sub-patterns (Sectioning, Voting, Map-Reduce) using `asyncio.gather`, which validates the core premise of concurrent execution.
2. **Completeness**: The "When NOT to Use" section correctly highlights "sequential dependencies", which is the primary architectural boundary between this pattern and Prompt Chaining.
3. **Practicality**: The "Production Considerations" regarding `asyncio.Semaphore` and `return_exceptions=True` are critical additions for real-world reliability that go beyond the basic demo code.
4. **Consistency**: Matches the template in `documentation.md`.
