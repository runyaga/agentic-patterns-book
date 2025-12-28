# Production Reality Check: Tool Use

**Target file**: `docs/patterns/05-tool-use.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- Tasks require capabilities beyond text generation (math, current data, APIs)
- Real-time information is needed (weather, stock prices, database lookups)
- Interaction with external systems is required (sending emails, updating records)
- Computation must be exact (financial calculations, date math)

### When NOT to Use
- All required information can be provided in the prompt context
- Approximate answers from the model's training data are acceptable
- Tool execution has unacceptable latency for your use case
- Security constraints prevent giving the model access to external systems

### Production Considerations
- **Security**: Tools are code execution. Sanitize inputs, validate outputs,
  and use least-privilege principles. Never `eval()` LLM-generated code directly.
- **Error handling**: Tools can fail (network errors, invalid inputs). Return
  structured error messages so the model can recover or inform the user.
- **Rate limits**: External APIs have quotas. Implement caching and backoff.
  Consider whether the model might call tools excessively.
- **Idempotency**: For action tools (send email, update DB), consider what
  happens if the same tool is called twice. Guard against duplicate actions.
- **Observability**: Log all tool calls with inputs and outputs. This is your
  audit trail for debugging and compliance.
- **Testing**: Mock external services in tests. Test edge cases like timeouts,
  malformed responses, and rate limit errors.

---

## Review

**Date**: 2025-12-27
**Reviewer**: Gemini Agent
**Status**: Verified
**Comments**: 
1. **Accuracy**: The implementation demonstrates the use of tools (`get_weather`, `calculate`, `search_information`) via the `@tool_agent.tool` decorator, which matches the core "When to Use" criteria.
2. **Completeness**: The "When NOT to Use" section correctly warns about latency and security, which are addressed in the code by the `calculate` tool's restricted `eval` context (only allowing safe math functions).
3. **Practicality**: The "Production Considerations" regarding "Idempotency" and "Security" are well-founded warnings. The implementation's use of a restricted `eval` environment is a good example of security awareness.
4. **Consistency**: Matches the template in `documentation.md`.
