# Production Reality Check: Human-in-the-Loop

**Target file**: `docs/patterns/13-human-in-loop.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- Cost of error is high (financial loss, safety risk, legal liability)
- Compliance or regulations require human sign-off
- AI is prone to hallucination in the domain (medical, legal, financial)
- Confidence scores reliably indicate when human review is needed
- Building trust during initial deployment before full automation

### When NOT to Use
- Latency requirements don't allow waiting for human response
- Volume is too high for humans to review (need to improve AI or accept risk)
- Error cost is low and correction after-the-fact is acceptable
- No humans are available to review (24/7 systems without staff)

### Production Considerations
- **Queue management**: Human review requests need a queue. Handle backpressure
  when humans can't keep up. Consider SLAs and timeouts.
- **Fallback behavior**: What happens if no human responds within N minutes?
  Auto-approve, auto-reject, or escalate?
- **Reviewer fatigue**: Too many reviews leads to rubber-stamping. Monitor
  approval rates and review times for signs of fatigue.
- **Feedback loops**: Capture human decisions to improve the AI's confidence
  calibration and reduce future human reviews.
- **Audit trail**: Log every human decision with timestamp, reviewer ID, and
  rationale. Essential for compliance and debugging.
- **UI/UX**: Reviewers need context to make good decisions. Show relevant
  information, not just "approve/reject" buttons.
- **Testing**: Test both paths (auto-approve and human-review) thoroughly.
  Mock human responses in automated tests.
