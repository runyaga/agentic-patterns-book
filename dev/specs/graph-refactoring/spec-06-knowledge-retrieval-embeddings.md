# Spec 06: Knowledge Retrieval with pydantic-ai Embeddings

**Status**: DEFERRED
**Decision**: REFACTOR - replace simple_embedding with pydantic-ai Embedder API
**Priority**: P3 (punted - simple_embedding sufficient for demos)
**Complexity**: Medium

---

## 1. Overview

Refactor `knowledge_retrieval.py` to use the new pydantic-ai embeddings API instead of the custom `simple_embedding()` function. This provides:

- Real semantic embeddings (not character-based heuristics)
- Multiple provider support (Ollama, OpenAI, Cohere, Sentence Transformers)
- Built-in token counting and dimension control
- OpenTelemetry/Logfire instrumentation
- TestEmbeddingModel for unit tests

---

## 2. Current Implementation

### 2.1 What Exists

```python
# Current: Simple character-based embedding (not semantic)
def simple_embedding(text: str, dim: int = 64) -> list[float]:
    """Basic implementation for demonstration."""
    text_lower = text.lower()
    embedding = [0.0] * dim
    for i, char in enumerate(text_lower):
        idx = ord(char) % dim
        embedding[idx] += 1.0 / (i + 1)
    # Normalize...
    return embedding

# VectorStore uses this function
@dataclass
class VectorStore:
    embedding_fn: Callable[[str], list[float]] = field(default=simple_embedding)
```

**Problems**:
- Not semantic - "dog" and "canine" have no similarity
- Fixed 64 dimensions - not configurable
- No real model behind it
- No token counting or limits
- Hard to test with real embeddings

---

## 3. Target Implementation

### 3.1 New Embeddings Integration

```python
from pydantic_ai.embeddings import Embedder

# Default to Ollama for local development
def get_embedder(model: str | None = None) -> Embedder:
    """
    Get an embedder instance.

    Args:
        model: Model string (e.g., "ollama:nomic-embed-text",
               "openai:text-embedding-3-small")

    Returns:
        Configured Embedder instance.
    """
    if model is None:
        # Default to Ollama with nomic-embed-text
        model = os.getenv("EMBEDDING_MODEL", "ollama:nomic-embed-text")

    return Embedder(model)


async def embed_text(text: str, embedder: Embedder | None = None) -> list[float]:
    """Embed a single text string."""
    if embedder is None:
        embedder = get_embedder()
    result = await embedder.embed_query(text)
    return result[0]


async def embed_texts(texts: list[str], embedder: Embedder | None = None) -> list[list[float]]:
    """Embed multiple texts (for documents)."""
    if embedder is None:
        embedder = get_embedder()
    result = await embedder.embed_documents(texts)
    return result.embeddings
```

### 3.2 Updated VectorStore

```python
@dataclass
class VectorStore:
    """
    In-memory vector store for RAG with pydantic-ai embeddings.
    """

    chunks: list[Chunk] = field(default_factory=list)
    documents: dict[str, Document] = field(default_factory=dict)
    embedder: Embedder | None = None
    chunk_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    doc_counter: int = 0
    query_counter: int = 0

    def __post_init__(self):
        if self.embedder is None:
            self.embedder = get_embedder()

    async def add_document(
        self,
        content: str,
        metadata: dict | None = None,
        title: str = "",
        source: str = "",
    ) -> Document:
        """Add a document and embed its chunks."""
        doc_id = self._generate_doc_id()
        doc = Document(
            content=content,
            doc_id=doc_id,
            title=title,
            source=source,
            metadata=metadata or {},
        )
        self.documents[doc_id] = doc

        # Chunk the document
        text_chunks = chunk_text(
            content,
            chunk_size=self.chunk_config.chunk_size,
            chunk_overlap=self.chunk_config.chunk_overlap,
        )

        # Batch embed all chunks
        embeddings = await embed_texts(text_chunks, self.embedder)

        for i, (chunk_text_content, embedding) in enumerate(
            zip(text_chunks, embeddings, strict=True)
        ):
            chunk = Chunk(
                content=chunk_text_content,
                chunk_id=f"{doc_id}_chunk_{i:03d}",
                doc_id=doc_id,
                index=i,
                embedding=embedding,
                metadata={"title": title, "source": source},
            )
            self.chunks.append(chunk)

        return doc

    async def search(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Search for relevant chunks using semantic similarity."""
        self.query_counter += 1

        if not self.chunks:
            return []

        # Embed the query
        query_embedding = await embed_text(query, self.embedder)

        # Calculate similarities
        scored_chunks = []
        for chunk in self.chunks:
            score = cosine_similarity(query_embedding, chunk.embedding)
            if score >= threshold:
                scored_chunks.append((chunk, score))

        # Sort and return top k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(scored_chunks[:k], 1)
        ]
```

### 3.3 Sync Wrappers for Convenience

```python
def add_document_sync(
    store: VectorStore,
    content: str,
    **kwargs
) -> Document:
    """Synchronous document addition."""
    import asyncio
    return asyncio.run(store.add_document(content, **kwargs))


def search_sync(
    store: VectorStore,
    query: str,
    **kwargs
) -> list[RetrievedChunk]:
    """Synchronous search."""
    import asyncio
    return asyncio.run(store.search(query, **kwargs))
```

### 3.4 Configuration via Environment

```python
# Environment variables:
# EMBEDDING_MODEL - Model to use (default: ollama:nomic-embed-text)
# OLLAMA_URL - Ollama server URL (default: http://localhost:11434)
# OPENAI_API_KEY - For OpenAI embeddings

# Examples:
# EMBEDDING_MODEL=ollama:nomic-embed-text  # Local Ollama
# EMBEDDING_MODEL=openai:text-embedding-3-small  # OpenAI
# EMBEDDING_MODEL=cohere:embed-v4.0  # Cohere
```

---

## 4. Testing Strategy

### 4.1 Use TestEmbeddingModel

```python
# tests/test_knowledge_retrieval.py

import pytest
from pydantic_ai.embeddings import Embedder
from pydantic_ai.embeddings.test import TestEmbeddingModel

from agentic_patterns.knowledge_retrieval import VectorStore


@pytest.fixture
def test_embedder():
    """Create embedder with deterministic test model."""
    return Embedder(TestEmbeddingModel())


@pytest.fixture
def test_store(test_embedder):
    """Create VectorStore with test embedder."""
    return VectorStore(embedder=test_embedder)


async def test_add_document(test_store):
    """Test adding a document creates chunks with embeddings."""
    doc = await test_store.add_document(
        content="Python is a programming language.",
        title="Python Intro",
    )

    assert doc.doc_id is not None
    assert len(test_store.chunks) > 0
    assert all(len(c.embedding) > 0 for c in test_store.chunks)


async def test_search_returns_results(test_store):
    """Test search finds relevant chunks."""
    await test_store.add_document(
        content="Machine learning uses algorithms to learn from data.",
        title="ML Basics",
    )

    results = await test_store.search("algorithms", k=3)

    assert len(results) > 0
    assert results[0].score > 0
```

### 4.2 Integration Tests with Real Embeddings

```python
# tests/integration/test_knowledge_retrieval_integration.py

import pytest
from agentic_patterns.knowledge_retrieval import (
    VectorStore,
    build_knowledge_base,
)


@pytest.mark.integration
async def test_semantic_similarity():
    """Test that semantically similar texts have high similarity."""
    store = VectorStore()  # Uses real embedder

    await store.add_document("Dogs are loyal pets that love their owners.")
    await store.add_document("Cats are independent animals.")

    # Should find dog document for canine query
    results = await store.search("canine companions")

    assert len(results) > 0
    assert "dog" in results[0].chunk.content.lower()
```

---

## 5. Migration Path

### 5.1 Backwards Compatibility

Keep `simple_embedding` as fallback:

```python
def get_embedder(model: str | None = None) -> Embedder | None:
    """Get embedder, returns None if not available."""
    try:
        from pydantic_ai.embeddings import Embedder
        model = model or os.getenv("EMBEDDING_MODEL", "ollama:nomic-embed-text")
        return Embedder(model)
    except ImportError:
        return None


@dataclass
class VectorStore:
    embedder: Embedder | None = None
    _use_simple: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.embedder is None:
            self.embedder = get_embedder()
        if self.embedder is None:
            self._use_simple = True  # Fallback to simple_embedding
```

### 5.2 Implementation Tasks

1. [ ] Add `pydantic-ai[embeddings]` or ensure embeddings available
2. [ ] Create `get_embedder()` helper function
3. [ ] Update `VectorStore` to use async embeddings
4. [ ] Add sync wrappers for convenience
5. [ ] Update `RAGPipeline` for async embeddings
6. [ ] Update `build_knowledge_base()` factory
7. [ ] Update tests with `TestEmbeddingModel`
8. [ ] Add integration tests for semantic similarity
9. [ ] Update `__main__` demo
10. [ ] Run quality gates

---

## 6. Value vs Complexity Analysis

### 6.1 Benefits

| Benefit | Impact |
|---------|--------|
| Real semantic embeddings | High - actually works for RAG |
| Multiple provider support | High - flexibility |
| Ollama for local dev | High - no API keys needed |
| TestEmbeddingModel for tests | Medium - deterministic tests |
| Token counting | Medium - prevent errors |
| Logfire instrumentation | Medium - observability |

### 6.2 Complexity

| Factor | Rating |
|--------|--------|
| Code changes | Medium (~100 lines) |
| API change (sync -> async) | Medium |
| Test changes | Medium |
| New dependency | Low (already have pydantic-ai) |

### 6.3 Risks

| Risk | Mitigation |
|------|------------|
| Ollama not running | Fallback to simple_embedding |
| API breaking change | Pin pydantic-ai version |
| Async migration | Provide sync wrappers |

---

## 7. Supported Embedding Models

### 7.1 Local (Ollama)

```python
# Requires Ollama running with embedding model
# ollama pull nomic-embed-text

embedder = Embedder("ollama:nomic-embed-text")
```

### 7.2 OpenAI

```python
# Requires OPENAI_API_KEY
embedder = Embedder("openai:text-embedding-3-small")

# With dimension reduction
from pydantic_ai.embeddings import EmbeddingSettings
embedder = Embedder(
    "openai:text-embedding-3-large",
    settings=EmbeddingSettings(dimensions=256),
)
```

### 7.3 Sentence Transformers (Local)

```python
# Runs locally, downloads from HuggingFace
embedder = Embedder("sentence-transformers:all-MiniLM-L6-v2")
```

---

## 8. Files Changed

```
src/agentic_patterns/
├── knowledge_retrieval.py    # REFACTOR - use Embedder API

tests/
├── test_knowledge_retrieval.py  # UPDATE - use TestEmbeddingModel
```

---

## 9. Example Usage

```python
from agentic_patterns.knowledge_retrieval import (
    VectorStore,
    RAGPipeline,
    build_knowledge_base,
)

# Build knowledge base (async)
documents = [
    {"title": "Python", "content": "Python is a programming language..."},
    {"title": "ML", "content": "Machine learning enables..."},
]

store = await build_knowledge_base(documents)

# Query
pipeline = RAGPipeline(store=store)
response = await pipeline.query("What is Python?")
print(response.answer)

# Or use environment variable for model
# EMBEDDING_MODEL=openai:text-embedding-3-small python -m agentic_patterns.knowledge_retrieval
```
