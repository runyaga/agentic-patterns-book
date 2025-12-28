# Production Reality Check Specs

This directory contains proposed "Production Reality Check" sections for each
pattern documentation page. These replace the existing "When to Use" sections.

## Purpose

Each file contains the **proposed content** that will be added to the
corresponding pattern doc in `docs/patterns/`. These specs are meant to be
reviewed by multiple AI agents (Claude, Codex, Gemini) before being applied.

## Review Instructions

For each file, verify:

1. **Accuracy**: Are the "When to Use" criteria correct for this pattern?
2. **Completeness**: Are important "When NOT to Use" cases covered?
3. **Practicality**: Are the "Production Considerations" realistic?
4. **Consistency**: Does the format match the template in `documentation.md`?

## Template

Each file follows this structure:

```markdown
# Production Reality Check: [Pattern Name]

**Target file**: `docs/patterns/XX-pattern-name.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- [Criteria for appropriate use]
- [Comparison to simpler alternatives]

### When NOT to Use
- [Anti-patterns]
- [Cost/complexity warnings]

### Production Considerations
- [Infrastructure requirements]
- [Observability needs]
- [Failure modes]
```

## Files

| File | Pattern | Chapter |
|------|---------|---------|
| `01-prompt-chaining.md` | Prompt Chaining | 1 |
| `02-routing.md` | Routing | 2 |
| `03-parallelization.md` | Parallelization | 3 |
| `04-reflection.md` | Reflection | 4 |
| `05-tool-use.md` | Tool Use | 5 |
| `06-planning.md` | Planning | 6 |
| `07-multi-agent.md` | Multi-Agent | 7 |
| `08-memory.md` | Memory | 8 |
| `09-learning.md` | Learning | 9 |
| `10-mcp-integration.md` | MCP Integration | 10 |
| `11-goal-monitoring.md` | Goal Monitoring | 11 |
| `12-exception-recovery.md` | Exception Recovery | 12 |
| `13-human-in-loop.md` | Human-in-the-Loop | 13 |
| `14-knowledge-retrieval.md` | Knowledge Retrieval (RAG) | 14 |
| `15-agent-marketplace.md` | Agent Marketplace | 15 |
| `17-reasoning-weaver.md` | Reasoning Weaver (ToT) | 17 |
| `21-domain-exploration.md` | Domain Exploration | 21 |

## Workflow

1. **Phase 1 (This PR)**: Create these spec files with proposed content
2. **Review**: AI agents and humans review each file
3. **Phase 2 (Next PR)**: Apply approved content to actual pattern docs
