# Production Reality Check: Goal Monitoring

**Target file**: `docs/patterns/11-goal-monitoring.md`
**Replaces**: `## When to Use (vs Cron)` section

---

## Production Reality Check

### When to Use

| Scenario | Use Cron / Systemd | Use Goal Monitor |
| :--- | :--- | :--- |
| **Logic** | Static thresholds (Disk > 90%) | Dynamic/Contextual reasoning |
| **Action** | Fixed scripts (`systemctl restart`) | Agentic remediation (read logs, plan fix) |
| **Trigger** | Time-based only | State-based (continuous) |
| **Cost** | Free (shell scripts) | Non-zero (LLM calls) |
| **Complexity** | Low | Medium (async loop) |

### When NOT to Use
- Static threshold monitoring (use Prometheus, Datadog, or cron)
- Remediation is a fixed script that doesn't need intelligence
- Cost of LLM calls for monitoring exceeds value of intelligent remediation
- System reliability requirements demand battle-tested monitoring tools

**Verdict:** For 90% of infrastructure monitoring, use existing tools. Use Goal
Monitoring when remediation requires *intelligence* (e.g., "Fix the typo in the
README" or "Refactor this function if it gets too complex").

### Production Considerations
- **Escalation**: Implement escalation for repeated failures. Don't let the
  agent retry forever - alert humans when remediation fails N times.
- **State persistence**: Save monitoring state to disk/DB so monitoring can
  resume after process restarts without losing context.
- **Resource limits**: Set timeouts on evaluators and remediation. Unbounded
  LLM calls can be expensive and slow.
- **Observability**: Log every check, every gap detected, every remediation
  attempt. This is your audit trail for "why did the system change X?"
- **Testing**: Mock evaluators in tests. Test the full WaitNode → CheckNode →
  RemediateNode cycle with simulated failures.
- **Graceful shutdown**: Handle SIGTERM properly - complete current check cycle
  before exiting.
