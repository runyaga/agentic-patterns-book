# Production Reality Check: Multi-Agent Collaboration

**Target file**: `docs/patterns/07-multi-agent.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- Task exceeds single context window or single agent's capability
- Distinct specialized skills are required (e.g., coding vs. writing vs. review)
- Parallel processing (multiple agents working simultaneously) provides speedup
- Sequential validation is needed (one agent reviews another's output)

### When NOT to Use
- A single well-prompted agent can handle the task (simpler is better)
- Communication overhead between agents exceeds task complexity
- Debugging and observability requirements are strict (multi-agent is harder
  to trace)
- Latency budget doesn't allow for multiple sequential agent calls

### Production Considerations
- **Coordination complexity**: More agents = more failure modes. Start with
  fewer agents and add only when single-agent approaches demonstrably fail.
- **Context sharing**: Agents need shared context. Design clear protocols for
  what information flows between agents and how.
- **Error propagation**: One agent's mistake can cascade. Implement validation
  between agent handoffs and clear error recovery strategies.
- **Cost multiplication**: N agents = roughly N times the token cost. Budget
  for the full pipeline, not just individual agents.
- **Observability**: Log each agent's inputs, outputs, and decision points.
  Multi-agent debugging without logs is nearly impossible.
- **Testing**: Test agents individually AND the full collaboration flow. Mock
  inter-agent communication for faster iteration.

---

## Review

**Date**: 2025-12-27
**Reviewer**: Gemini Agent
**Status**: Verified
**Comments**: 
1. **Accuracy**: The implementation demonstrates both "Supervisor" (via `pydantic_graph`) and "Network" (via `asyncio.gather`) collaboration styles, aligning with the "Collaboration structures" concept.
2. **Completeness**: The "Context sharing" consideration is addressed in the code by the `CollaborationContext` and `_get_previous` tools, which allow agents to see each other's work.
3. **Practicality**: The "Coordination complexity" warning is valid; the implementation shows how complex state management becomes (`CollaborationState`, `Graph`, `PlanNode`) compared to simpler patterns.
4. **Consistency**: Matches the template in `documentation.md`.
