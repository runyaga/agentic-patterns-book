# Documentation Specification

**Status**: FINAL v1
**Date**: 2025-12-26

## Objective

Maintain consistent, up-to-date documentation as patterns are implemented.
Documentation updates are a required step in the pattern implementation
workflow.

## Documentation Files

### README.md (root)

The main project entry point. Must include:

1. **Implemented Patterns Table** - Updated after each pattern completion
2. **Usage Examples** - Command to run each pattern
3. **Project Structure** - Current file layout

### docs/attribution.md

Credits and licensing. Updated when:
- New dependencies are added
- Source material references change

### dev/LESSONS.md

Development best practices. Updated when:
- New testing patterns are discovered
- Common pitfalls are encountered
- Useful techniques emerge during implementation

### docs/idioms.md

Framework idioms for pydantic-ai, pydantic_graph, and pydantic-evals.
Reference this when implementing patterns to ensure idiomatic code.
Updated when new framework patterns are discovered.

### dev/specs/pattern-implementation.md

Implementation workflow and status tracking. Contains:
- Pattern status table (DONE/pending/blocked)
- Quality gates checklist
- Error handling strategy

### dev/config/agents.md

Agent configuration reference. Updated when:
- New model configurations are tested
- Provider settings change

### docs/patterns/

Pattern-specific documentation. Each implemented pattern has its own page.

#### Structure

```
docs/patterns/
├── index.md                    # Pattern index and overview
├── 01-prompt-chaining.md       # Chapter 1
├── 02-routing.md               # Chapter 2
├── 03-parallelization.md       # Chapter 3
├── 04-reflection.md            # Chapter 4
├── 05-tool-use.md              # Chapter 5
├── 06-planning.md              # Chapter 6
├── 07-multi-agent.md           # Chapter 7
├── 08-memory.md                # Chapter 8
├── 09-learning.md              # Chapter 9
├── 10-mcp-integration.md       # Chapter 10
├── 11-goal-monitoring.md       # Chapter 11
├── 12-exception-recovery.md    # Chapter 12
├── 13-human-in-loop.md         # Chapter 13
├── 14-knowledge-retrieval.md   # Chapter 14
├── 15-agent-marketplace.md     # Chapter 15 (Spec)
├── 16-resource-optimization.md # Chapter 16
├── 17-reasoning-weaver.md      # Chapter 17 (Spec)
├── 18-guardrails.md            # Chapter 18
├── 19-evaluation-monitoring.md # Chapter 19
├── 20-prioritization.md        # Chapter 20
└── 21-domain-exploration.md    # Chapter 21 (Spec)
```

#### Naming Convention

Files use the format `{chapter}-{pattern-name}.md`:
- Chapter number with leading zero (01-21)
- Pattern name in kebab-case
- "(Spec)" suffix in index indicates specification/planned patterns

#### Content Structure

Each pattern page includes:
1. **Title and Overview**: Pattern name, chapter reference, summary
2. **Key Concepts**: Core ideas in bullet form
3. **Implementation**: Code snippets from the source module
4. **Use Cases**: When this pattern applies
5. **Production Reality Check**: Skeptical, practical guidance (replaces "When
   to Use") - see template below
6. **Example**: How to run the demo

#### Maintenance

- Create new pattern page when implementing a new chapter
- Update index.md when adding/removing patterns
- Keep code snippets in sync with source modules
- Pattern pages are reference docs, not tutorials

## Pattern Documentation Requirements

Each implemented pattern must have:

### 1. Module Docstring

At the top of the pattern file:

```python
"""
Pattern Name - Brief description.

This module implements the [Pattern Name] pattern from Chapter N.

Key concepts:
- Concept 1
- Concept 2

Example usage:
    result = await main_agent.run("input")
"""
```

### 2. Function Docstrings

All public functions require docstrings:

```python
async def process_input(data: InputModel) -> OutputModel:
    """
    Process input through the pattern pipeline.

    Args:
        data: The input to process.

    Returns:
        Processed output with results.

    Raises:
        ValueError: If input validation fails.
    """
```

### 3. README Entry

Add row to the Implemented Patterns table in README.md:

```markdown
| [Pattern Name](src/agentic_patterns/pattern_name.py) | N | Brief description |
```

Add usage example:

```markdown
# Pattern name: brief description
.venv/bin/python -m agentic_patterns.pattern_name
```

## README Update Procedure

After completing a pattern implementation:

1. **Update Implemented Patterns table**
   - Add new row with pattern name, chapter, description
   - Link to source file

2. **Add usage example**
   - Add command under "Run a pattern example" section
   - Include brief comment describing what it does

3. **Verify structure section**
   - Ensure file is listed in project structure
   - Update if new directories were created

## Pattern Implementation Spec Updates

After each pattern:

1. **Update status table**
   - Change pattern status from `pending` to `DONE`
   - Note any blocked patterns with reasons

2. **Update execution order**
   - Mark completed patterns
   - Adjust if dependencies changed

## Documentation Checklist

Before marking a pattern as complete, verify:

- [ ] Module docstring present
- [ ] Public function docstrings complete
- [ ] README.md Implemented Patterns table updated
- [ ] README.md usage example added
- [ ] pattern-implementation.md status updated to DONE
- [ ] Any new lessons added to LESSONS.md
- [ ] Production Reality Check section included (not just "When to Use")
- [ ] Example section with runnable command included
- [ ] `uv run mkdocs build --strict` passes locally

## Style Guidelines

### Markdown

- Line length: 79 characters (same as code)
- Use ATX headers (`#`, `##`, `###`)
- Code blocks with language specifier
- Tables aligned with pipes

### Docstrings

- Use Google style docstrings
- Keep first line under 79 characters
- Include Args, Returns, Raises sections as needed

### Consistency

- Pattern names: lowercase with underscores (file names)
- Chapter references: "Chapter N" format
- Status values: DONE, pending, blocked

## Automation

Documentation updates are manual but tracked in the implementation workflow.
The pattern-implementation.md spec includes a documentation phase that must
be completed before marking a pattern as DONE.

See Phase 6 in pattern-implementation.md for the complete documentation
workflow.

---

## MkDocs Configuration

### Mermaid Diagrams

Add flow diagrams to pattern docs using mermaid fenced code blocks:

```markdown
## Flow Diagram

` ` `mermaid
flowchart LR
    A[Input] --> B[Process]
    B --> C[Output]
` ` `
```

Setup in `mkdocs.yml`:

```yaml
markdown_extensions:
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

extra_javascript:
  - https://unpkg.com/mermaid@10/dist/mermaid.min.js
  - js/mermaid-init.js
```

The `docs/js/mermaid-init.js` handles initialization:

```javascript
document.addEventListener('DOMContentLoaded', function() {
    mermaid.initialize({ startOnLoad: true, theme: 'default' });
});
```

### Code Snippets from Source

Use pymdownx.snippets to include code directly from Python files (DRY):

In your Python file, add snippet markers:

```python
# --8<-- [start:models]
class MyModel(BaseModel):
    field: str
# --8<-- [end:models]
```

In your markdown docs:

```markdown
` ` `python
--8<-- "src/agentic_patterns/routing.py:models"
` ` `
```

This keeps docs in sync with source code automatically.

**Note**: Snippets inside mermaid fences don't work well due to how
pymdownx.superfences wraps content. Put mermaid diagrams inline in docs,
matching the diagram in the Python module docstring.

### Source Code Links

Link to source files using absolute GitHub URLs, not relative paths:

```markdown
# WRONG - mkdocs treats this as a doc file reference
[Source Code](../../src/agentic_patterns/foo.py)

# RIGHT - absolute GitHub URL
Source: [`src/agentic_patterns/foo.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/foo.py)
```

Relative paths to `.py` files fail `mkdocs build --strict` because mkdocs
looks for them in the docs directory.

---

## Production Reality Check Section

Every pattern page must include a "Production Reality Check" section that
replaces the simpler "When to Use" section. This provides honest, skeptical
guidance for practitioners.

### Required Subsections

1. **When to Use**: Clear criteria for appropriate use cases. Include
   comparison to simpler alternatives where applicable.

2. **When NOT to Use**: Anti-patterns, inappropriate use cases, and
   cost/complexity warnings. Be explicit about when simpler approaches
   (standard Python, cron jobs, basic retry loops) are better.

3. **Production Considerations**: What's missing for real deployment.
   May include:
   - Infrastructure requirements (message queues, databases)
   - Observability needs (logging, tracing, metrics)
   - Scaling concerns
   - Failure modes and recovery strategies

### Template

```markdown
## Production Reality Check

### When to Use
- [Clear criteria for when this pattern is appropriate]
- [Comparison to simpler alternatives]

### When NOT to Use
- [Anti-patterns and inappropriate use cases]
- [Cost/complexity warnings]

### Production Considerations
- **[Category]**: [What's needed for production]
- **[Category]**: [Scaling/failure concerns]

## Example

` ` `bash
.venv/bin/python -m agentic_patterns.pattern_name
` ` `
```

### Example (Prompt Chaining)

```markdown
## Production Reality Check

### When to Use
- Multiple LLM calls with clear data dependencies between steps
- When intermediate results need validation before proceeding
- Pipeline-style workflows where each step transforms the output

### When NOT to Use
- Single-shot queries that don't need intermediate processing
- When latency is critical (each chain link adds round-trip time)
- Simple transformations that could be done with string formatting

### Production Considerations
- **Observability**: Log each chain step separately for debugging
- **Failure handling**: Decide retry strategy per-step vs. full-chain
- **Cost**: Each step is a separate API call; budget accordingly

## Example

` ` `bash
.venv/bin/python -m agentic_patterns.prompt_chaining
` ` `
```

---

## Review Feedback (Codex)

- The spec says pattern names use underscores, but the docs naming convention
  is kebab-case; clarify which is authoritative and align filenames and table
  entries accordingly.
- The line length rule (79 chars) is not enforced in this file; either relax
  the rule for Markdown or add a formatter step, otherwise it is aspirational.
- The docs structure lists chapters 01-09, 13-14 and says 10-12, 15 are
  skipped; this conflicts with the reality-check specs that include 10-12 and
  15+ entries. Decide whether those chapters are truly skipped.
- README instructions link to `src/agentic_patterns/*.py` with a relative
  path, while later guidance says docs must link to absolute GitHub URLs to
  satisfy mkdocs. Make the scope of this rule explicit (README vs docs).
- The "FINAL v1" status and a future date (2025-12-26) are a red flag; if this
  is authoritative, move to a versioning scheme or update to a current date.

— Codex
