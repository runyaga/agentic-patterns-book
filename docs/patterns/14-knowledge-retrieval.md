# Chapter 14: Knowledge Retrieval (RAG)

Augment LLMs with external knowledge via embeddings and vector search.

## Implementation

Source: `src/agentic_patterns/knowledge_retrieval.py`

### Vector Store

Embeds and stores document chunks.

```python
class Chunk(BaseModel):
    content: str
    doc_id: str
    embedding: list[float]

@dataclass
class VectorStore:
    chunks: list[Chunk] = field(default_factory=list)

    def add_document(self, content: str):
        # 1. Chunk Text
        text_chunks = chunk_text(content, size=500)
        
        # 2. Embed & Store
        for text in text_chunks:
            embedding = self.embedding_fn(text)
            self.chunks.append(Chunk(content=text, embedding=embedding))

    def search(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        query_vec = self.embedding_fn(query)
        # Cosine similarity search...
        return sorted_chunks[:k]
```

### RAG Pipeline

```python
@dataclass
class RAGPipeline:
    store: VectorStore

    async def query(self, question: str) -> RAGResponse:
        # 1. Retrieve
        chunks = self.store.search(question, k=3)
        context = "\n".join(f"- {c.content}" for c in chunks)

        # 2. Generate with Context
        result = await rag_agent.run(
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer using ONLY the context."
        )
        
        return RAGResponse(answer=result.output, sources=chunks)
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
