# Production Reality Check: Memory Management

**Target file**: `docs/patterns/08-memory.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use

| Type | Best For | Trade-off |
|------|----------|-----------|
| **Buffer** | Short, high-detail tasks | High token cost, hits limits fast |
| **Window** | Infinite streams, reactive bots | Loses distant context ("amnesia") |
| **Summary** | Long-term coherent conversations | Lossy compression of details |

### When NOT to Use
- Stateless interactions where each request is independent
- When message history is already managed externally (e.g., by your app's DB)
- Short conversations that fit easily in context window without management
- When summary quality is critical but hard to validate (summaries lose nuance)

### Production Considerations
- **Storage persistence**: In-memory stores are lost on restart. Use Redis,
  PostgreSQL, or similar for production memory that survives deployments.
- **Token budgeting**: Memory competes with prompt and response for context
  window. Monitor actual token usage and tune memory limits accordingly.
- **Summary quality**: SummaryMemory relies on LLM summarization which can
  hallucinate or drop important facts. Validate summaries for critical use cases.
- **Privacy**: Memory stores may contain PII. Implement retention policies,
  encryption, and user data deletion capabilities.
- **Multi-user isolation**: Ensure one user's memory doesn't leak to another.
  Namespace or partition memory by user/session ID.
- **Cache invalidation**: Decide when memory should be cleared (session end,
  time-based expiry, explicit user request).

---

## Review

**Date**: 2025-12-27
**Reviewer**: Gemini Agent
**Status**: Verified
**Comments**: 
1. **Accuracy**: The implementation provides three distinct classes (`BufferMemory`, `WindowMemory`, `SummaryMemory`) that perfectly correspond to the types described in the "When to Use" table.
2. **Completeness**: The "Summary quality" trade-off is well-represented by the complexity of the `SummaryMemory` class (even though the demo implementation is simplified), acknowledging that summarization is lossy.
3. **Practicality**: The "Storage persistence" point is a crucial reality check; the code uses in-memory Python lists (`self.messages = []`), which confirms the need for external storage (Redis/Postgres) in production as stated.
4. **Consistency**: Matches the template in `documentation.md`.
