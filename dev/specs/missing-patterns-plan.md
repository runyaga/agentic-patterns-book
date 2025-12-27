# Missing Patterns Implementation Strategy

**Status:** Draft
**Date:** 2025-12-27
**Context:** Plan for implementing remaining chapters from "Agentic Design Patterns" by Antonio Gulli.

## 1. Overview
This document serves as the master plan for implementing the missing chapters (10, 11, 12, 15, 17, 21).

**Core Philosophy:** "Library-First." All patterns will be implemented as pure Python modules (`src/agentic_patterns/`) usable within the existing single-process application. We will leverage `asyncio` for concurrency and embedded tools for state, avoiding requirements for external servers (Redis, Neo4j, etc.).

## 2. Documentation Standards
All new patterns must strictly adhere to `dev/specs/documentation.md`:
*   **Module Docstrings:** Complete Google-style docstrings.
*   **README Updates:** Add to "Implemented Patterns" table and "Usage Examples".
*   **Pattern Docs:** Create a new page in `docs/patterns/` with:
    *   Title & Overview
    *   Key Concepts
    *   Mermaid Flow Diagrams (inline)
    *   Use Cases & Decision Guidance
    *   Code Snippets
*   **Lessons Learned:** Update `dev/LESSONS.md`.

## 3. Implementation Roadmap & Infrastructure Strategy

### Phase 1: Robustness
*   **Chapter 12: Exception Handling -> "The Phoenix Protocol"**
    *   *Concept:* A "Clinic Agent" that diagnoses and heals failed agents.
    *   *Infrastructure:* Pure Python. Uses `pydantic-ai` models to wrap exceptions and state.
    *   *Spec File:* `dev/specs/missing-patterns/spec-12-phoenix-protocol.md`

### Phase 2: Extensibility
*   **Chapter 10: Model Context Protocol (MCP) -> "The Universal Connector"**
    *   *Concept:* An agent that dynamically discovers tools at runtime.
    *   *Infrastructure:* Python `mcp` SDK. connects to local subprocesses (stdio) or mock servers. No HTTP server required.
    *   *Spec File:* `dev/specs/missing-patterns/spec-10-universal-connector.md`

### Phase 3: Autonomy
*   **Chapter 11: Goal Setting -> "The Teleological Engine"**
    *   *Concept:* Background monitoring loop for agent objectives (OKRs).
    *   *Infrastructure:* `asyncio.create_task()` for background loops. In-memory state tracking.
    *   *Spec File:* `dev/specs/missing-patterns/spec-11-teleological-engine.md`
*   **Chapter 21: Exploration -> "The Cartographer"**
    *   *Concept:* Mapping unknown domains into a Knowledge Graph.
    *   *Infrastructure:* `networkx` for graph structure, `lancedb` (existing) for semantic storage.
    *   *Spec File:* `dev/specs/missing-patterns/spec-21-cartographer.md`

### Phase 4: Intelligence & Scale
*   **Chapter 17: Reasoning -> "The Cognitive Weaver"**
    *   *Concept:* Pluggable reasoning topologies (Tree of Thoughts).
    *   *Infrastructure:* Pure Python classes defining graph nodes and traversal logic.
    *   *Spec File:* `dev/specs/missing-patterns/spec-17-cognitive-weaver.md`
*   **Chapter 15: Inter-Agent Communication -> "The Agora"**
    *   *Concept:* Market-based task allocation via pub/sub.
    *   *Infrastructure:* **In-Memory Event Bus** using `asyncio.Queue` and `asyncio.Event`. No Redis/RabbitMQ.
    *   *Spec File:* `dev/specs/missing-patterns/spec-15-agora.md`

## 4. Next Steps
1.  Draft `spec-12-phoenix-protocol.md` (Phase 1).
2.  Implement `src/agentic_patterns/exception_recovery.py`.
3.  Add tests and docs.
