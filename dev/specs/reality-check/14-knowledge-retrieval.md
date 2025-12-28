# Production Reality Check: Knowledge Retrieval (RAG)

**Target file**: `docs/patterns/14-knowledge-retrieval.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use
- Information is proprietary, private, or too new for model's training data
- Hallucination must be minimized (grounding answers in source documents)
- Source citations are required (compliance, trust, verification)
- Knowledge base changes frequently and can't be baked into model weights

### When NOT to Use
- Questions are about general knowledge the model already has
- Document corpus is small enough to fit in context window directly
- Retrieval latency is unacceptable for your use case
- Query types don't match well with semantic similarity (e.g., exact lookups)

### Production Considerations
- **Chunking strategy**: Chunk size affects retrieval quality. Too small loses
  context; too large wastes tokens. Experiment with your corpus.
- **Embedding quality**: Embedding model choice matters. Domain-specific models
  may outperform general ones for specialized content.
- **Retrieval relevance**: Monitor retrieval quality. If retrieved chunks don't
  answer the question, the LLM will hallucinate anyway.
- **Index maintenance**: Documents change. Implement incremental indexing and
  handle document updates/deletions.
- **Latency budget**: Embedding query + vector search + LLM call. Each step
  adds latency. Consider caching frequent queries.
- **Cost tracking**: Embedding calls have costs. Cache embeddings for repeated
  queries. Monitor vector DB costs at scale.
- **Hybrid search**: Pure vector search misses exact matches. Consider hybrid
  (vector + keyword) for better recall.
- **Evaluation**: Regularly evaluate retrieval precision/recall and answer
  quality. RAG can degrade silently as corpus grows.
