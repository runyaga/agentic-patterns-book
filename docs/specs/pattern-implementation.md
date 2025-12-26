# Pattern Implementation Specification

**Status**: FINAL v1
**Date**: 2025-12-26

## Objective

Systematically convert all agentic design patterns from the "Agentic Design Patterns" book into working pydantic-ai implementations with comprehensive tests, following project standards and Blacksmith validation.

## Book Chapters & Patterns

Chapters are implemented **sequentially** in order. Unknown chapters are skipped.

| Chapter | Pattern | Status | Depends On |
|---------|---------|--------|------------|
| 1 | Prompt Chaining | DONE | - |
| 2 | Routing | DONE | Ch 1 |
| 3 | Parallelization | DONE | Ch 1 |
| 4 | Reflection | DONE | Ch 1 |
| 5 | Tool Use | DONE | Ch 1-2 |
| 6 | Planning | DONE | Ch 1-5 |
| 7 | Multi-Agent Collaboration | DONE | Ch 1-6 |
| 8 | Memory Management | DONE | Ch 1 |
| 9 | Learning and Adaptation | DONE | Ch 4, 8 |
| 10-12 | SKIP | - | - |
| 13 | Human-in-the-Loop | DONE | Ch 1-7 |
| 14 | Knowledge Retrieval (RAG) | DONE | Ch 5 |
| 15 | SKIP | - | - |
| 16 | Resource-Aware Optimization | DONE | Ch 3, 6 |
| 17 | SKIP | - | - |
| 18 | Guardrails/Safety Patterns | DONE | Ch 1-7 |
| 19 | Evaluation and Monitoring | DONE | Ch 4, 18 |
| 20 | Prioritization | DONE | Ch 2, 6 |

## Folder Structure

```
src/agentic_patterns/
├── __init__.py
├── _models.py              # Shared model configuration
├── prompt_chaining.py      # Chapter 1
├── routing.py              # Chapter 2
├── parallelization.py      # Chapter 3
└── ...

tests/
├── conftest.py
├── test_prompt_chaining.py
├── test_routing.py
├── test_parallelization.py
└── ...
```

## Model Configuration

Use the shared `get_model()` function from `agentic_patterns._models`:

```python
from agentic_patterns import get_model

model = get_model()  # defaults: gpt-oss:20b, localhost:11434
model = get_model(model_name="other-model")  # override model
```

Patterns may override `model_name` or `base_url` as needed.

## Graph API

**IMPORTANT**: Any pydantic_graph usage MUST use the beta API:

```python
# Beta API (required)
from pydantic_graph.beta import GraphBuilder, StepContext

# NOT the stable API
# from pydantic_graph import BaseNode, End, Graph, GraphRunContext
```

The beta `GraphBuilder` API is cleaner and preferred over the stable `BaseNode` approach.

## Implementation Workflow

### Phase 1: Discovery
1. Query agent-book MCP for chapter content
2. Extract pattern description and use cases
3. Identify key concepts and data flows
4. Note any code examples (as reference only, not to copy)

### Phase 2: Design
1. Define Pydantic models for inputs/outputs
2. Design agent structure (single vs multi-agent)
3. Plan error handling strategy
4. Document in docstrings

### Phase 3: Implementation
1. Create `src/agentic_patterns/{pattern}.py`
2. Use `from agentic_patterns import get_model`
3. Follow code style from `CLAUDE.md`:
   - Line length: 79 chars
   - Type hints required
   - Docstrings for public functions

### Phase 4: Testing
1. Create `tests/test_{pattern}.py`
2. Unit tests for all Pydantic models
3. Integration tests with mocked agents
4. Edge case tests (empty inputs, failures)
5. Achieve >= 80% test coverage

### Phase 5: Validation
1. Run `uv run ruff check src/ tests/` - zero warnings
2. Run `uv run ruff format src/ tests/`
3. Run `uv run pytest` - all passing
4. Submit to Blacksmith agent for code review
5. Address Blacksmith feedback (max 2 cycles)
6. Mark as complete only after Blacksmith approval

### Phase 6: Documentation
1. Verify module has docstring with pattern description
2. Verify all public functions have docstrings
3. Update README.md:
   - Add row to Implemented Patterns table
   - Add usage example command
   - Update project structure if needed
4. Update this spec's status table (pattern → DONE)
5. Add any lessons learned to `docs/LESSONS.md`

See `docs/specs/documentation.md` for full documentation requirements.

## Error Handling Strategy

### Retry Logic
| Parameter | Value |
|-----------|-------|
| Max retries per agent call | 3 |
| Backoff strategy | Fixed 1s delay |
| Timeout per agent call | 90s |

### When to Give Up

Abandon implementation attempt when:
1. Agent returns malformed output after **3 retries**
2. Implementation fails after **3 complete attempts**
3. Blacksmith rejects after **2 revision cycles**

When giving up:
1. Log detailed error information
2. Mark pattern as `blocked`
3. Continue to next pattern
4. Report blocked patterns at end for human review

### Error Categories

| Error Type | Action | Max Attempts |
|------------|--------|--------------|
| LLM timeout | Retry after 1s | 3 |
| Malformed JSON | Retry with stricter prompt | 3 |
| Validation error | Log and retry with feedback | 3 |
| Blacksmith rejection | Revise based on feedback | 2 |
| Persistent failure | Log, mark blocked, skip | - |

## Blacksmith Integration

### When to Invoke
- After all tests pass
- After linter passes
- Before marking pattern as "complete"

### Blacksmith Review Criteria
1. Code correctness
2. Error handling completeness
3. Test coverage >= 80%
4. Documentation quality
5. Adherence to project standards (CLAUDE.md)

### Handling Blacksmith Feedback

```
Blacksmith Response
       │
       ├─── APPROVED ──────────> Mark Complete
       │
       ├─── MINOR ISSUES ──────> Auto-fix, Re-submit (cycle 1)
       │                              │
       │                              └──> If fails again ──> Human Review
       │
       └─── MAJOR ISSUES ──────> Revise, Re-submit (cycle 1)
                                      │
                                      └──> If fails again ──> Mark Blocked
```

Max Blacksmith cycles: **2**

## Quality Gates

Each pattern must pass ALL gates before completion:

- [x] `ruff check` with zero warnings
- [x] `ruff format` applied
- [x] `pytest` all tests passing
- [x] Test coverage >= 80%
- [x] Blacksmith approval
- [x] Documentation updated (README.md, status table)

## Execution Order

Patterns are implemented sequentially to:
1. Build foundational patterns first
2. Allow later patterns to import/reuse earlier ones
3. Make progress predictable for human review
4. Simplify debugging when issues arise

**Order**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 13 → 14 → 16 → 18 → 19 → 20

(Chapters 10-12, 15, 17 are skipped - unknown patterns)

## Progress Reporting

After each pattern:
```
Pattern: [name]
Status: [DONE | BLOCKED]
Tests: [X passed, Y failed]
Coverage: [XX%]
Blacksmith: [APPROVED | REJECTED (reason)]
Duration: [Xm Ys]
```

End of run summary:
```
Completed: X/Y patterns
Blocked: [list of blocked patterns with reasons]
Total Duration: [Xh Ym]
```

## Success Criteria

- All known chapters have working pydantic-ai implementations
- All tests passing (>= 80% coverage)
- All patterns Blacksmith-approved
- Blocked patterns documented with clear failure reasons
- Each pattern has runnable example in `if __name__ == "__main__"`
