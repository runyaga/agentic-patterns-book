# Production Reality Check: Planning

**Target file**: `docs/patterns/06-planning.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- Goal is too complex for a single prompt ("one-shot" doesn't work)
- Steps have strict dependencies (step B needs output from step A)
- Error recovery is needed (if step 2 fails, try alternative step 2b)
- Transparency in the process is required (users need to see/approve the plan)

### When NOT to Use
- Task is simple enough for prompt chaining with fixed steps
- Planning overhead exceeds execution time (small tasks don't need plans)
- Real-time latency requirements don't allow for plan generation
- The domain is well-understood with predictable steps (hardcode the workflow)

### Production Considerations
- **Plan quality**: LLM-generated plans can be unrealistic or miss steps.
  Consider human review for high-stakes tasks, or use constrained planning
  with predefined step templates.
- **Execution failures**: Plan for plan failures. What happens when step 3 of 5
  fails? Options: abort, retry step, replan from current state, skip to next.
- **State persistence**: Long-running plans need checkpointing. Store completed
  step results so execution can resume after crashes.
- **Cost explosion**: Replanning on every failure can spiral costs. Set limits
  on replan attempts and total steps.
- **Observability**: Log the generated plan, each step's execution, and any
  replanning events. Essential for debugging "why did the agent do that?"
- **Timeouts**: Set per-step and total-plan timeouts. Unbounded planning loops
  are expensive and confusing to users.

---

## Review

**Date**: 2025-12-27
**Reviewer**: Gemini Agent
**Status**: Verified
**Comments**: 
1. **Accuracy**: The implementation revolves around a `Plan` model with `PlanStep` items containing `dependencies`, which validates the "steps have strict dependencies" criterion.
2. **Completeness**: The "Error recovery" criterion is explicitly implemented via the `replan` function and `replanner_agent`, which is triggered when a step fails and `allow_replan` is True.
3. **Practicality**: The "Cost explosion" and "Timeouts" warnings are critical. The code implements `max_steps` and `max_retries` to mitigate these risks, demonstrating awareness of these production constraints.
4. **Consistency**: Matches the template in `documentation.md`.
