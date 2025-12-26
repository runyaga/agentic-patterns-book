# Chapter 14: Knowledge Retrieval (RAG)

Augment LLMs with external knowledge via embeddings and vector search.

## Implementation

Source: [`src/agentic_patterns/knowledge_retrieval.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/knowledge_retrieval.py)

### Data Models

```python
--8<-- "src/agentic_patterns/knowledge_retrieval.py:models"
```

### Utility Functions

```python
--8<-- "src/agentic_patterns/knowledge_retrieval.py:utils"
```

### Vector Store

```python
--8<-- "src/agentic_patterns/knowledge_retrieval.py:store"
```

### RAG Pipeline

```python
--8<-- "src/agentic_patterns/knowledge_retrieval.py:rag"
```

## Use Cases

- **Doc Q&A**: Chat with PDF manuals or API docs.
- **Company Wiki**: Search internal policies (Notion/Confluence).
- **Recent News**: Inject latest data that post-dates the model training.

## When to Use

- Information is proprietary, private, or too new for the model.
- Hallucination must be minimized (grounding).
- Source citations are required.

## Example

```bash
.venv/bin/python -m agentic_patterns.knowledge_retrieval
```
