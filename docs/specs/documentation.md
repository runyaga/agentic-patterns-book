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

### docs/ATTRIBUTION.md

Credits and licensing. Updated when:
- New dependencies are added
- Source material references change

### docs/LESSONS.md

Development best practices. Updated when:
- New testing patterns are discovered
- Common pitfalls are encountered
- Useful techniques emerge during implementation

### docs/specs/pattern-implementation.md

Implementation workflow and status tracking. Contains:
- Pattern status table (DONE/pending/blocked)
- Quality gates checklist
- Error handling strategy

### docs/config/agents.md

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
├── 13-human-in-loop.md         # Chapter 13
└── 14-knowledge-retrieval.md   # Chapter 14
```

#### Naming Convention

Files use the format `{chapter}-{pattern-name}.md`:
- Chapter number with leading zero (01-14)
- Pattern name in kebab-case
- Gaps in numbering indicate skipped chapters (10-12, 15)

#### Content Structure

Each pattern page includes:
1. **Title and Overview**: Pattern name, chapter reference, summary
2. **Key Concepts**: Core ideas in bullet form
3. **Implementation**: Code snippets from the source module
4. **Use Cases**: When this pattern applies
5. **When to Use**: Decision guidance
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
