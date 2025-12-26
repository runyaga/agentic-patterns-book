"""
Knowledge Retrieval (RAG) Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 14:
Retrieval-Augmented Generation connects LLMs to external knowledge.

Key concepts:
- Chunking: Break documents into manageable pieces
- Embeddings: Convert text to numerical vectors
- Vector Store: Store and search embeddings by similarity
- Retrieval: Find relevant chunks for a query
- Augmentation: Combine retrieved context with the query

This module implements:
- Document: A source document with metadata
- Chunk: A piece of a document with embedding
- VectorStore: In-memory store with similarity search
- RAGPipeline: End-to-end retrieval and generation

Example usage:
    store = VectorStore()
    store.add_document("AI is transforming...", {"source": "article"})
    pipeline = RAGPipeline(store)
    answer = await pipeline.query("What is AI doing?")
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai import RunContext

from agentic_patterns._models import get_model


# --8<-- [start:models]
class Document(BaseModel):
    """A source document for the knowledge base."""

    content: str = Field(description="Full document content")
    doc_id: str = Field(description="Unique document identifier")
    title: str = Field(default="", description="Document title")
    source: str = Field(default="", description="Document source")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When document was added",
    )


class Chunk(BaseModel):
    """A chunk of text from a document."""

    content: str = Field(description="Chunk text content")
    chunk_id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Parent document ID")
    index: int = Field(description="Position in document")
    embedding: list[float] = Field(
        default_factory=list,
        description="Vector embedding",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Chunk metadata",
    )


class RetrievedChunk(BaseModel):
    """A chunk with its similarity score."""

    chunk: Chunk = Field(description="The retrieved chunk")
    score: float = Field(description="Similarity score (0-1)")
    rank: int = Field(description="Rank in results")


class RAGResponse(BaseModel):
    """Response from the RAG pipeline."""

    answer: str = Field(description="Generated answer")
    query: str = Field(description="Original query")
    context_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks used for context",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in answer",
    )


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    chunk_size: int = Field(default=500, description="Max chunk size")
    chunk_overlap: int = Field(
        default=50, description="Overlap between chunks"
    )
    separator: str = Field(default="\n", description="Split point")


class RetrievalStats(BaseModel):
    """Statistics about retrieval operations."""

    total_documents: int = Field(description="Documents in store")
    total_chunks: int = Field(description="Chunks in store")
    avg_chunk_size: float = Field(description="Average chunk length")
    queries_processed: int = Field(description="Queries handled")
# --8<-- [end:models]


# --8<-- [start:utils]
def simple_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Create a simple text embedding.

    This is a basic implementation for demonstration. In production,
    use proper embedding models like sentence-transformers.

    Args:
        text: Text to embed.
        dim: Embedding dimension.

    Returns:
        List of floats representing the embedding.
    """
    # Simple character-based embedding
    text_lower = text.lower()
    embedding = [0.0] * dim

    for i, char in enumerate(text_lower):
        idx = ord(char) % dim
        embedding[idx] += 1.0 / (i + 1)

    # Normalize
    magnitude = math.sqrt(sum(x * x for x in embedding))
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Similarity score between 0 and 1.
    """
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return (dot_product / (mag1 * mag2) + 1) / 2  # Normalize to 0-1


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separator: str = "\n",
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.
        separator: Preferred split point.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to find a good break point
        if end < len(text):
            # Look for separator near the end
            break_point = text.rfind(separator, start, end)
            if break_point > start + chunk_size // 2:
                end = break_point + len(separator)
            else:
                # Look for space
                space_point = text.rfind(" ", start, end)
                if space_point > start + chunk_size // 2:
                    end = space_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break

    return chunks
# --8<-- [end:utils]


# --8<-- [start:store]
@dataclass
class VectorStore:
    """
    In-memory vector store for RAG.

    Stores document chunks with embeddings and supports
    similarity-based retrieval.
    """

    chunks: list[Chunk] = field(default_factory=list)
    documents: dict[str, Document] = field(default_factory=dict)
    embedding_fn: Callable[[str], list[float]] = field(
        default=simple_embedding
    )
    chunk_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    doc_counter: int = 0
    query_counter: int = 0

    def _generate_doc_id(self) -> str:
        """Generate a unique document ID."""
        self.doc_counter += 1
        return f"doc_{self.doc_counter:04d}"

    def add_document(
        self,
        content: str,
        metadata: dict | None = None,
        title: str = "",
        source: str = "",
    ) -> Document:
        """
        Add a document to the store.

        Args:
            content: Document text content.
            metadata: Optional metadata.
            title: Document title.
            source: Document source.

        Returns:
            The created Document.
        """
        doc_id = self._generate_doc_id()
        doc = Document(
            content=content,
            doc_id=doc_id,
            title=title,
            source=source,
            metadata=metadata or {},
        )
        self.documents[doc_id] = doc

        # Chunk and embed
        text_chunks = chunk_text(
            content,
            chunk_size=self.chunk_config.chunk_size,
            chunk_overlap=self.chunk_config.chunk_overlap,
            separator=self.chunk_config.separator,
        )

        for i, chunk_text_content in enumerate(text_chunks):
            chunk = Chunk(
                content=chunk_text_content,
                chunk_id=f"{doc_id}_chunk_{i:03d}",
                doc_id=doc_id,
                index=i,
                embedding=self.embedding_fn(chunk_text_content),
                metadata={"title": title, "source": source},
            )
            self.chunks.append(chunk)

        return doc

    def search(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        """
        Search for relevant chunks.

        Args:
            query: Search query.
            k: Number of results to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of RetrievedChunk with scores.
        """
        self.query_counter += 1

        if not self.chunks:
            return []

        query_embedding = self.embedding_fn(query)

        # Calculate similarities
        scored_chunks = []
        for chunk in self.chunks:
            score = cosine_similarity(query_embedding, chunk.embedding)
            if score >= threshold:
                scored_chunks.append((chunk, score))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for rank, (chunk, score) in enumerate(scored_chunks[:k], 1):
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                )
            )

        return results

    def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def get_stats(self) -> RetrievalStats:
        """Get store statistics."""
        avg_size = 0.0
        if self.chunks:
            avg_size = sum(len(c.content) for c in self.chunks) / len(
                self.chunks
            )

        return RetrievalStats(
            total_documents=len(self.documents),
            total_chunks=len(self.chunks),
            avg_chunk_size=avg_size,
            queries_processed=self.query_counter,
        )

    def clear(self) -> None:
        """Clear all documents and chunks."""
        self.chunks = []
        self.documents = {}
        self.doc_counter = 0
# --8<-- [end:store]


# --8<-- [start:rag]
# Initialize model
model = get_model()


@dataclass
class RAGDeps:
    """Dependencies for RAG agent with dynamic search."""

    store: VectorStore
    top_k: int = 3
    min_score: float = 0.1


# RAG generation agent with tool-based retrieval
rag_agent: Agent[RAGDeps, str] = Agent(
    model,
    system_prompt=(
        "You are a helpful assistant that answers questions using a "
        "knowledge base. Use the search_knowledge tool to find relevant "
        "information before answering. If the search doesn't return "
        "useful results, say so clearly. Always cite which parts of "
        "the retrieved context support your answer."
    ),
    deps_type=RAGDeps,
    output_type=str,
)


@rag_agent.tool
async def search_knowledge(
    ctx: RunContext[RAGDeps],
    query: str,
) -> str:
    """
    Search the knowledge base for information relevant to the query.

    Args:
        ctx: Run context with RAG dependencies.
        query: The search query to find relevant information.

    Returns:
        Formatted context from retrieved chunks, or message if none found.
    """
    retrieved = ctx.deps.store.search(
        query=query,
        k=ctx.deps.top_k,
        threshold=ctx.deps.min_score,
    )

    if not retrieved:
        return "No relevant information found in the knowledge base."

    # Format retrieved chunks as context
    context_parts = []
    for rc in retrieved:
        source_info = ""
        if rc.chunk.metadata.get("title"):
            source_info = f" (from: {rc.chunk.metadata['title']})"
        context_parts.append(f"[{rc.rank}]{source_info}: {rc.chunk.content}")

    return "\n\n".join(context_parts)


@dataclass
class RAGPipeline:
    """
    Complete RAG pipeline for retrieval-augmented generation.

    Uses tool-based retrieval where the agent decides when to search.
    """

    store: VectorStore
    top_k: int = 3
    min_score: float = 0.1

    async def query(
        self,
        question: str,
    ) -> RAGResponse:
        """
        Process a question through the RAG pipeline.

        The agent uses the search_knowledge tool to dynamically
        retrieve relevant context before answering.

        Args:
            question: User's question.

        Returns:
            RAGResponse with answer and sources.
        """
        # Create deps with store configuration
        deps = RAGDeps(
            store=self.store,
            top_k=self.top_k,
            min_score=self.min_score,
        )

        # Run agent - it will use search_knowledge tool as needed
        result = await rag_agent.run(question, deps=deps)
        answer = result.output

        # Get chunks that were retrieved during the query
        # (store tracks queries, we get the most recent search results)
        retrieved = self.store.search(
            query=question,
            k=self.top_k,
            threshold=self.min_score,
        )

        # Calculate confidence based on retrieval scores
        confidence = 0.5
        if retrieved:
            avg_score = sum(r.score for r in retrieved) / len(retrieved)
            confidence = min(0.95, avg_score)

        return RAGResponse(
            answer=answer,
            query=question,
            context_chunks=retrieved,
            confidence=confidence,
        )

    async def batch_query(
        self,
        questions: list[str],
    ) -> list[RAGResponse]:
        """
        Process multiple questions.

        Args:
            questions: List of questions.

        Returns:
            List of RAGResponse objects.
        """
        responses = []
        for question in questions:
            response = await self.query(question)
            responses.append(response)
        return responses


def build_knowledge_base(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> VectorStore:
    """
    Build a knowledge base from documents.

    Args:
        documents: List of dicts with 'content', 'title', 'source'.
        chunk_size: Size of chunks.
        chunk_overlap: Overlap between chunks.

    Returns:
        Populated VectorStore.
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    store = VectorStore(chunk_config=config)

    for doc in documents:
        store.add_document(
            content=doc.get("content", ""),
            title=doc.get("title", ""),
            source=doc.get("source", ""),
            metadata=doc.get("metadata", {}),
        )

    return store
# --8<-- [end:rag]


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Knowledge Retrieval (RAG) Pattern")
        print("=" * 60)

        # Create knowledge base
        documents = [
            {
                "title": "Python Basics",
                "content": (
                    "Python is a high-level programming language known for "
                    "its readability and versatility. It supports multiple "
                    "programming paradigms including procedural, "
                    "object-oriented, and functional programming. Python "
                    "uses dynamic typing and automatic memory management."
                ),
                "source": "tutorial",
            },
            {
                "title": "Machine Learning",
                "content": (
                    "Machine learning is a subset of artificial intelligence "
                    "that enables systems to learn from data. Common types "
                    "include supervised learning, unsupervised learning, "
                    "and reinforcement learning. Popular libraries include "
                    "scikit-learn, TensorFlow, and PyTorch."
                ),
                "source": "guide",
            },
            {
                "title": "Web Development",
                "content": (
                    "Modern web development involves both frontend and "
                    "backend technologies. Frontend uses HTML, CSS, and "
                    "JavaScript frameworks like React and Vue. Backend "
                    "commonly uses Python with Django or Flask, or Node.js "
                    "with Express. APIs often follow REST or GraphQL."
                ),
                "source": "reference",
            },
        ]

        print("\n--- Building Knowledge Base ---")
        store = build_knowledge_base(documents)
        stats = store.get_stats()
        print(f"Documents: {stats.total_documents}")
        print(f"Chunks: {stats.total_chunks}")
        print(f"Avg chunk size: {stats.avg_chunk_size:.0f} chars")

        # Create pipeline
        pipeline = RAGPipeline(store=store, top_k=2)

        # Query examples
        questions = [
            "What programming paradigms does Python support?",
            "What are common machine learning libraries?",
        ]

        for question in questions:
            print(f"\n--- Query: {question} ---")
            response = await pipeline.query(question)
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.0%}")
            print(f"Sources: {len(response.context_chunks)} chunks used")

    asyncio.run(main())
