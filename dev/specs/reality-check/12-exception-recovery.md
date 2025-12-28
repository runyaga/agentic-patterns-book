# Production Reality Check: Exception Recovery

**Target file**: `docs/patterns/12-exception-recovery.md`
**Replaces**: `## When to Use` section (keep `## Streaming Limitation` separate)

---

## Production Reality Check

### When to Use

| Use Case | Recommended Approach |
| :--- | :--- |
| **Output Validation** | Use `@output_validator` with `ModelRetry`. For *semantic* errors where model ran but produced bad data. |
| **Exception Handling** | Use **Phoenix Protocol**. For *runtime* errors (crashes, timeouts, API failures). |
| **Complex Workflows** | Use `pydantic_graph`. If recovery requires complex state transitions. |

### When NOT to Use
- Errors are always fatal and shouldn't be retried (e.g., auth failures)
- Simple retry with backoff is sufficient (use `tenacity` or basic loop)
- Clinic Agent overhead exceeds value of intelligent diagnosis
- Streaming responses are required (see Streaming Limitation section)

### Production Considerations
- **Retry limits**: Always cap retries. Infinite retry loops are expensive and
  can indicate a fundamentally broken prompt/system.
- **Deterministic classification**: The heuristic classifier handles 90% of
  errors without LLM calls. Keep this path fast and cheap.
- **Clinic Agent costs**: The Clinic Agent is only called for UNKNOWN errors.
  Monitor how often this happens - high rates suggest missing heuristics.
- **Circuit breaking**: After N consecutive failures, consider circuit breaking
  to prevent cascade failures. Don't hammer a failing API.
- **Observability**: Log every exception, classification, and recovery action.
  Essential for understanding failure patterns and tuning heuristics.
- **Timeout handling**: Timeouts are common in LLM APIs. Ensure your recovery
  handles them gracefully with appropriate backoff.
