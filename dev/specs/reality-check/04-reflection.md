# Production Reality Check: Reflection

**Target file**: `docs/patterns/04-reflection.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- You have clear "pass/fail" or scoring criteria that can be programmatically
  checked
- You want linear application logic (`result = run()`) with retries handled
  internally
- Feedback from failures should inform the retry (model sees why it failed via
  `ModelRetry` message in conversation history)
- Quality requirements justify the cost of multiple generation attempts

### When NOT to Use
- No objective validation criteria exist (purely subjective quality)
- Single-attempt accuracy is already acceptable for your use case
- Latency budget doesn't allow for retry loops (each retry is a full LLM call)
- Validation logic is complex enough to warrant external orchestration

### Production Considerations
- **Max retries**: Always set a cap. Infinite loops are expensive and can
  indicate a fundamentally broken prompt that retries won't fix.
- **Cost tracking**: Monitor retry rates. High retry rates suggest prompt
  improvement is needed rather than more retries.
- **Validator complexity**: Keep validators fast and deterministic. Slow
  validators (e.g., running full test suites) compound latency.
- **Feedback quality**: The `ModelRetry` message becomes part of context.
  Make error messages actionable ("missing required field X" not just "invalid").
- **Observability**: Log each attempt and the validation failure reason to
  identify systematic issues.

---

## Review

**Date**: 2025-12-27
**Reviewer**: Gemini Agent
**Status**: Verified
**Comments**: 
1. **Accuracy**: The implementation uses PydanticAI's `output_validator` and `ModelRetry`, which perfectly aligns with the "retries handled internally" and "ModelRetry message" points.
2. **Completeness**: The "When NOT to Use" section correctly identifies latency as a trade-off, which is evident in the `retries=3` configuration of the `producer_agent`.
3. **Practicality**: The "Max retries" consideration is directly implemented in the code (`retries=3`), preventing infinite loops as warned.
4. **Consistency**: Matches the template in `documentation.md`.
