# Goal Monitoring

**Chapter 11: The Teleological Engine**

Most agents are reactive - they wait for prompts to act. The Goal Monitoring
pattern makes agents **proactive**: they continuously check if goals are met
and attempt remediation when gaps are detected.

## Overview

```mermaid
stateDiagram-v2
    [*] --> WaitNode: Start monitoring

    WaitNode --> CheckNode: interval elapsed
    WaitNode --> [*]: shutdown requested

    CheckNode --> WaitNode: all goals met
    CheckNode --> RemediateNode: gap detected

    RemediateNode --> CheckNode: retry check
```

## Key Components

### Goal

A measurable target with an async evaluator:

```python
--8<-- "src/agentic_patterns/goal_monitoring.py:models"
```

### GoalMonitor

Manages the monitoring lifecycle:

```python
--8<-- "src/agentic_patterns/goal_monitoring.py:monitor"
```

## Usage Example

```python
--8<-- "src/agentic_patterns/goal_monitoring.py:main"
```

## Comparators

| Comparator | Meaning | Example |
|------------|---------|---------|
| `>=` | Current should be >= target | Uptime >= 99.9% |
| `<=` | Current should be <= target | Disk usage <= 80% |
| `==` | Current should equal target | Active connections == 0 |
| `>` | Current should be > target | Revenue > 0 |
| `<` | Current should be < target | Error rate < 1% |

## How It Works

1. **WaitNode**: Sleeps for `check_interval`, then transitions to CheckNode
2. **CheckNode**: Evaluates all goals, populates status, transitions to
   RemediateNode on first gap or back to WaitNode if all pass
3. **RemediateNode**: Calls the remediation agent with the goal's hint,
   escalates on failure, then returns to CheckNode to re-verify

## Production Reality Check

### When to Use

| Scenario | Use Cron / Systemd | Use Goal Monitor |
| :--- | :--- | :--- |
| **Logic** | Static thresholds (Disk > 90%) | Dynamic/Contextual reasoning |
| **Action** | Fixed scripts (`systemctl restart`) | Agentic remediation (read logs, plan fix) |
| **Trigger** | Time-based only | State-based (continuous) |
| **Cost** | Free (shell scripts) | Non-zero (LLM calls) |
| **Complexity** | Low | Medium (async loop) |

- *Comparison*: For static thresholds and fixed scripts, use cron + alerts +
  runbooks instead

### When NOT to Use
- Static threshold monitoring (use Prometheus, Datadog, or cron)
- Remediation is a fixed script that doesn't need intelligence
- Cost of LLM calls for monitoring exceeds value of intelligent remediation
- System reliability requirements demand battle-tested monitoring tools
- *Anti-pattern*: Routine log rotation or disk cleanup tasks—deterministic
  scripts are more reliable and cheaper

**Verdict:** For 90% of infrastructure monitoring, use existing tools. Use Goal
Monitoring when remediation requires *intelligence* (e.g., "Fix the typo in the
README" or "Refactor this function if it gets too complex").

### Production Considerations
- **Escalation**: Implement escalation for repeated failures. Don't let the
  agent retry forever—alert humans when remediation fails N times.
- **State persistence**: Save monitoring state to disk/DB so monitoring can
  resume after process restarts without losing context.
- **Resource limits**: Set timeouts on evaluators and remediation. Unbounded
  LLM calls can be expensive and slow.
- **Change management**: Guard against agent-triggered changes without approvals.
  High-stakes remediations should require human sign-off.
- **Observability**: Log every check, every gap detected, every remediation
  attempt. This is your audit trail for "why did the system change X?"
- **Graceful shutdown**: Handle SIGTERM properly—complete current check cycle
  before exiting.

## Example

```bash
.venv/bin/python -m agentic_patterns.goal_monitoring
```

## Production TODOs

This V1 implementation is intentionally lean. See the spec for production
enhancements:

- **P1: Escalation** - Add EscalateNode for repeated failures, integrate
  alerting
- **P2: Persistence** - Save/load state to JSON for resume after restart
- **P3: Advanced Evaluators** - file_stat, agent_assessment evaluators
- **P4: OKR Hierarchy** - Objectives with multiple Key Results
- **P5: Observability** - Logfire integration for structured logging

## API Reference

::: agentic_patterns.goal_monitoring
    options:
      show_root_heading: true
      members:
        - Goal
        - GoalStatus
        - GoalMonitor
        - run_goal_monitor
        - on_escalate
