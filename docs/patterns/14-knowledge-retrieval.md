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

### Idiomatic Pattern (Tool-Based Retrieval)

The agent uses a `@tool` to dynamically search the knowledge base, deciding
when and what to search based on the question.

```python
@dataclass
class RAGDeps:
    """Dependencies for RAG agent with dynamic search."""
    store: VectorStore
    top_k: int = 3
    min_score: float = 0.1

# Agent with tool-based retrieval
rag_agent: Agent[RAGDeps, str] = Agent(
    model,
    system_prompt=(
        "Answer questions using the knowledge base. "
        "Use the search_knowledge tool to find relevant information."
    ),
    deps_type=RAGDeps,
)

@rag_agent.tool
async def search_knowledge(ctx: RunContext[RAGDeps], query: str) -> str:
    """Search the knowledge base for information relevant to the query."""
    retrieved = ctx.deps.store.search(query, k=ctx.deps.top_k)
    if not retrieved:
        return "No relevant information found."
    return "\n".join(f"[{r.rank}]: {r.chunk.content}" for r in retrieved)
```

### RAG Pipeline

```python
@dataclass
class RAGPipeline:
    store: VectorStore
    top_k: int = 3

    async def query(self, question: str) -> RAGResponse:
        # Create deps - agent uses tool to search as needed
        deps = RAGDeps(store=self.store, top_k=self.top_k)

        # Agent dynamically decides when to search
        result = await rag_agent.run(question, deps=deps)

        return RAGResponse(answer=result.output, ...)
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
