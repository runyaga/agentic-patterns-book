# Reproduce This Repository

This document explains how to recreate this repository from scratch using
the same vibe-coding workflow that produced it.

## Overview

This repository was built using:

1. **Claude Code** - Anthropic's CLI for Claude
2. **haiku-rag** - RAG framework for indexing the source book
3. **Ollama** - Local LLM inference
4. **pydantic-ai** - The target framework for implementations

The workflow: Index the book PDF with haiku-rag, expose it as an MCP server,
then use Claude Code to implement each pattern based on the book content.

## Prerequisites

- Python 3.14+
- Docker (for Docling server)
- Ollama installed and running
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)

## Step 1: Set Up Ollama

Install Ollama and pull the required models:

```bash
# Install Ollama (https://ollama.ai/)
# Then pull models:
ollama pull qwen3-embedding:4b   # For embeddings
ollama pull gpt-oss:20b          # For generation (or your preferred model)
```

## Step 2: Set Up Docling Server

haiku-rag uses Docling for PDF processing. Run the Docling server:

```bash
docker run -p 5001:5001 ds4sd/docling-serve:latest
```

Leave this running in a terminal.

## Step 3: Install haiku-rag

```bash
# Install with slim dependencies (remote Docling)
pip install haiku-rag[slim]

# Or install full (local Docling, larger download)
pip install haiku-rag
```

## Step 4: Initialize RAG Database

```bash
# Create config file
haiku-rag init-config rag.yaml

# Create database directory
mkdir db

# Initialize the database
haiku-rag init-db db/rag.lancedb
```

## Step 5: Index the Book

Place the book PDF in a directory (e.g., `source/`):

```bash
# Index the PDF (takes ~10 minutes, no progress output)
haiku-rag --config rag.yaml add-src source/ --db db/rag.lancedb
```

## Step 6: Launch MCP Server

Start haiku-rag as an MCP server:

```bash
haiku-rag --config rag.yaml serve --mcp --mcp-port 8888 --db db/rag.lancedb
```

## Step 7: Connect to Claude Code

In a new terminal, add the MCP server to Claude Code:

```bash
claude mcp add --transport http agent-book http://127.0.0.1:8888/mcp
```

Verify connection:

```bash
claude
# Then type: /mcp
# Should show agent-book server connected
```

## Step 8: Implement Patterns

Now use Claude Code to implement each pattern. Example workflow:

```bash
claude

# Ask Claude to implement a pattern:
> Implement the prompt chaining pattern from Chapter 1. Query the agent-book
> for the chapter content, then create src/agentic_patterns/prompt_chaining.py
> with idiomatic pydantic-ai code.
```

---

## Pattern Implementation Workflow

For each pattern, follow this 6-phase workflow:

### Phase 1: Discovery

Query the RAG system for book content:

```python
# In Claude Code, use:
mcp__agent-book__search_documents("prompt chaining chapter 1")
mcp__agent-book__ask_question("What are the key concepts in prompt chaining?")
```

### Phase 2: Design

Design Pydantic models for inputs/outputs:

- Define clear input/output types
- Use `@dataclass` for dependencies (not Pydantic)
- Plan error handling strategy

### Phase 3: Implementation

Write the pattern following idiomatic pydantic-ai:

```python
from pydantic_ai import Agent, RunContext, ModelRetry
from dataclasses import dataclass

@dataclass
class MyDeps:
    threshold: float = 0.8

agent = Agent(
    model,
    output_type=OutputType,
    deps_type=MyDeps,
)

@agent.system_prompt
def add_context(ctx: RunContext[MyDeps]) -> str:
    return f"Threshold: {ctx.deps.threshold}"

@agent.output_validator
async def validate(ctx: RunContext[MyDeps], output: OutputType) -> OutputType:
    if not meets_criteria(output):
        raise ModelRetry("Improve based on this feedback")
    return output
```

### Phase 4: Testing

Write tests with 80% branch coverage:

```bash
uv run pytest tests/test_{pattern}.py --cov=src/agentic_patterns/{pattern}
```

### Phase 5: Validation

Run quality checks:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest
```

### Phase 6: Documentation

Update pattern documentation in `docs/patterns/`.

---

## PydanticAI Idioms

Use these native features instead of manual Python control flow:

### Output Validation with ModelRetry

```python
@agent.output_validator
async def validate(ctx: RunContext[Deps], output: Output) -> Output:
    if not meets_criteria(output):
        raise ModelRetry("Feedback for improvement")
    return output
```

**Use when:** Quality criteria exist, failed outputs should retry with feedback.

### Dynamic System Prompts

```python
@agent.system_prompt
def add_context(ctx: RunContext[Deps]) -> str:
    return f"Context: {ctx.deps.some_field}"
```

**Use when:** Injecting persistent context (history, preferences, knowledge).

### Tools with RunContext

```python
@agent.tool
async def my_tool(ctx: RunContext[Deps], query: str) -> str:
    return ctx.deps.service.process(query)
```

**Use when:** Agent needs to dynamically fetch external data.

### Dependencies Pattern

```python
@dataclass
class MyDeps:
    threshold: float = 0.8
    service: SomeService | None = None

result = await agent.run("query", deps=MyDeps(threshold=0.9))
```

**Use when:** Runtime configuration, shared services, state across decorators.

---

## Decision Framework

When evaluating how to implement a pattern:

1. **Is there a retry loop?** → Use `@output_validator` + `ModelRetry`
2. **Is context injected into system prompt?** → Use `@system_prompt`
3. **Is there shared runtime state?** → Use `deps_type`
4. **Does the agent need to fetch data dynamically?** → Use `@tool`

If "no" to all, the pattern may not need these features.

---

## Quality Gates

Before marking a pattern complete:

- [ ] `uv run ruff check src/agentic_patterns/{pattern}.py` passes
- [ ] `uv run ruff format src/agentic_patterns/{pattern}.py` runs clean
- [ ] `uv run pytest tests/test_{pattern}.py` passes
- [ ] `.venv/bin/python -m agentic_patterns.{pattern}` demo runs
- [ ] 80%+ branch coverage
- [ ] No manual retry loops (use `@output_validator`)
- [ ] No f-string prompt construction for system context
- [ ] All runtime state passed via `deps`

---

## Pattern Status

| Chapter | Pattern | Status |
|---------|---------|--------|
| 1 | Prompt Chaining | Done |
| 2 | Routing | Done |
| 3 | Parallelization | Done |
| 4 | Reflection | Done |
| 5 | Tool Use | Done |
| 6 | Planning | Done |
| 7 | Multi-Agent | Done |
| 8 | Memory | Done |
| 9 | Learning | Done |
| 10-12 | (Skipped) | N/A |
| 13 | Human-in-Loop | Done |
| 14 | Knowledge Retrieval | Done |
| 15 | (Skipped) | N/A |
| 16 | Resource-Aware | Done |
| 17 | (Skipped) | N/A |
| 18 | Guardrails | Done |
| 19 | Evaluation | Done |
| 20 | Prioritization | Done |

---

## Troubleshooting

### haiku-rag indexing hangs

The indexing process provides no feedback. Wait ~10 minutes for a typical
book-length PDF. Check Docling server logs for progress.

### MCP connection fails

Ensure the haiku-rag server is running and the port matches:

```bash
# Check server is running
curl http://127.0.0.1:8888/health

# Re-add MCP server
claude mcp remove agent-book
claude mcp add --transport http agent-book http://127.0.0.1:8888/mcp
```

### Model not found

Pull the model first:

```bash
ollama pull gpt-oss:20b
ollama list  # Verify it's available
```

---

## Additional Resources

- [pydantic-ai Documentation](https://ai.pydantic.dev/)
- [haiku-rag Repository](https://github.com/anthropics/haiku-rag)
- [Ollama Documentation](https://ollama.ai/)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
