# Specification: PydanticAI Graph & Evaluation Refactoring

**Status**: DRAFT
**Date**: 2025-12-27
**Target Version**: pydantic-ai v1.39.0+, pydantic-graph v1.39.0+

## 1. Overview
This specification outlines the refactoring of existing agentic patterns to utilize the `pydantic_graph` framework and the streamlining of the evaluation system to leverage native Logfire integration. It also introduces a dedicated Human-in-the-Loop Graph implementation.

## 2. Refactoring Goals

### 2.1 Routing Pattern (`routing.py`)
- **From**: Functional orchestration with `if/else` dispatch logic.
- **To**: Class-based `pydantic_graph.Graph` implementation.
- **Why**: 
    - Moves control flow into type-safe node transitions.
    - Enables Mermaid diagram generation for documentation.
    - Simplifies addition of new handlers/intents.

### 2.2 Multi-Agent Pattern (`multi_agent.py`)
- **From**: Manual `while` loop managing the Supervisor-Worker cycle.
- **To**: Cyclic `pydantic_graph.Graph` implementation.
- **Why**: 
    - Replaces fragile loop logic with a formal state machine.
    - State is managed within `GraphRunContext[State]`, removing the need for mutable list manipulation in `deps`.
    - Better handles complex inter-agent handoffs.

### 2.3 Evaluation Pattern (`evaluation.py`)
- **Goal**: Remove redundant observability code.
- **Changes**: 
    - Delete `PerformanceMonitor` and `AgentMetrics`.
    - Delete in-memory metric tracking models.
    - Rely on **Logfire** (configured in `_models.py`) for latency, tokens, and success rates.
    - Retain high-value qualitative tools: `LLMJudge`, `DriftDetector`, `ABTestRunner`.

## 3. New Implementation: Human-in-the-Loop Graph

A new module `human_in_loop_graph.py` will be created to demonstrate the advanced capabilities of the Graph API for stateful, interactive workflows.

### 3.1 Design
- **State**: `ApprovalState` (Request, Decision, Feedback, Result).
- **Nodes**:
    - `AnalystNode`: Agent evaluates the initial request.
    - `HumanGateNode`: Represents the wait for human intervention.
    - `ActionNode`: Executed only upon approval.
    - `FeedbackNode`: Executed if the human requests changes (returns to `AnalystNode`).
- **Graph Flow**: 
    - `AnalystNode` -> `HumanGateNode`
    - `HumanGateNode` -> `ActionNode` (if Approved)
    - `HumanGateNode` -> `FeedbackNode` -> `AnalystNode` (if Modified)
    - `HumanGateNode` -> `End` (if Rejected)

## 4. Implementation Standards

- **Node Definition**: Use `@dataclass` for node classes.
- **Inheritance**: All nodes must inherit from `pydantic_graph.BaseNode`.
- **Transitions**: Use strict type hints in `run()` return values to define allowed edges.
- **State**: Use a dedicated `State` dataclass for each graph.
- **Visualization**: Include logic to export Mermaid diagrams if requested.

## 5. Verification Plan

1. **Unit Tests**: Update existing tests to invoke `Graph.run()` instead of functional wrappers.
2. **Coverage**: Maintain >= 80% coverage.
3. **Logfire**: Verify that refactored graphs correctly emit traces to the Logfire dashboard.
4. **Demo**: Ensure the `if __name__ == "__main__":` block in each file provides a clear, working demonstration of the new architecture.
