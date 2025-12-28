# Production Reality Check: Routing

**Target file**: `docs/patterns/02-routing.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- Multiple specialized handlers exist with distinct capabilities
- Domain expertise varies significantly by query type
- Fallback handling is needed for ambiguous requests (using confidence scores)
- Request volume justifies the complexity of maintaining multiple handlers

### When NOT to Use
- Single-purpose applications where all queries go to the same handler
- When a system prompt can adequately cover all domains (simpler is better)
- Low-volume applications where a generalist agent is "good enough"
- When routing accuracy is critical but training data is limited

### Production Considerations
- **Classification errors**: Monitor misrouted queries; log router decisions for
  analysis. Consider human review for low-confidence classifications.
- **Handler availability**: Implement circuit breakers if handlers can fail
  independently. Fallback to a generalist handler when specialists are down.
- **Latency**: Router adds one LLM call before the actual handler. For latency-
  sensitive apps, consider caching common routes or using faster models for
  classification.
- **Observability**: Track routing distribution over time to detect drift or
  new query patterns that need new handlers.
- **Testing**: Test edge cases where queries span multiple intents. Confidence
  thresholds need tuning based on real traffic.

---

## Review

**Date**: 2025-12-27
**Reviewer**: Gemini Agent
**Status**: Verified
**Comments**: 
1. **Accuracy**: The implementation uses a dedicated `router_agent` with a `RouteDecision` model that includes `confidence`, directly supporting the "Fallback handling" and "confidence scores" criteria.
2. **Completeness**: The "When NOT to Use" section correctly warns about "generalist agent" sufficiency, which aligns with the trade-off of adding a routing step.
3. **Practicality**: The "Production Considerations" regarding classification errors and latency are highly relevant, as the `route_query` function explicitly performs two sequential LLM calls (router then handler).
4. **Consistency**: Matches the template in `documentation.md`.
