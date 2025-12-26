# Chapter 13: Human-in-the-Loop

Integrate human oversight for high-stakes decisions or low-confidence outputs.

## Implementation

Source: `src/agentic_patterns/human_in_loop.py`

### Escalation Policy

Rules defining when to stop for human review.

```python
@dataclass
class EscalationPolicy:
    confidence_threshold: float = 0.7
    sensitive_keywords: list[str] = field(default_factory=list)
    high_risk_actions: list[str] = field(default_factory=list)

    def should_escalate(self, output: AgentOutput) -> tuple[bool, str]:
        if output.confidence < self.confidence_threshold:
            return True, "low_confidence"
        
        if any(w in output.content for w in self.sensitive_keywords):
            return True, "sensitive_content"
            
        return False, None
```

### Execution with Oversight

```python
async def execute_with_oversight(task: str, policy: EscalationPolicy):
    # 1. Run Agent
    result = await task_agent.run(task)
    output = AgentOutput(content=result.output, confidence=...)

    # 2. Check Policy
    should_escalate, reason = policy.should_escalate(output)

    # 3. Escalate or Approve
    if should_escalate:
        # Halt and request review
        return EscalationRequest(reason=reason, output=output)
    else:
        return output.content
```

## Use Cases

- **Content Moderation**: Review flagged toxic/sensitive content.
- **Financial Ops**: Approve transactions > $10k.
- **Code Deployment**: Agent scaffolds code, Human reviews PR.
- **Medical/Legal**: AI drafts, Expert validates.

## When to Use

- Cost of error is high (financial loss, safety risk).
- Compliance requires human sign-off.
- AI is prone to "hallucination" in the domain.
- Confidence scores are reliable indicators of quality.

## Example

```bash
.venv/bin/python -m agentic_patterns.human_in_loop
```
