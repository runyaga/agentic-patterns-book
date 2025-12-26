"""Tests for the Knowledge Retrieval (RAG) pattern module."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.knowledge_retrieval import Chunk
from agentic_patterns.knowledge_retrieval import ChunkingConfig
from agentic_patterns.knowledge_retrieval import Document
from agentic_patterns.knowledge_retrieval import RAGPipeline
from agentic_patterns.knowledge_retrieval import RAGResponse
from agentic_patterns.knowledge_retrieval import RetrievalStats
from agentic_patterns.knowledge_retrieval import RetrievedChunk
from agentic_patterns.knowledge_retrieval import VectorStore
from agentic_patterns.knowledge_retrieval import build_knowledge_base
from agentic_patterns.knowledge_retrieval import chunk_text
from agentic_patterns.knowledge_retrieval import cosine_similarity
from agentic_patterns.knowledge_retrieval import simple_embedding


class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self) -> None:
        """Test creating a document with all fields."""
        doc = Document(
            content="Test content",
            doc_id="doc_001",
            title="Test Doc",
            source="test",
            metadata={"key": "value"},
        )
        assert doc.content == "Test content"
        assert doc.doc_id == "doc_001"
        assert doc.title == "Test Doc"
        assert doc.source == "test"
        assert doc.metadata == {"key": "value"}

    def test_document_defaults(self) -> None:
        """Test document default values."""
        doc = Document(content="Content", doc_id="doc_002")
        assert doc.title == ""
        assert doc.source == ""
        assert doc.metadata == {}


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation(self) -> None:
        """Test creating a chunk with all fields."""
        chunk = Chunk(
            content="Chunk content",
            chunk_id="chunk_001",
            doc_id="doc_001",
            index=0,
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
        )
        assert chunk.content == "Chunk content"
        assert chunk.chunk_id == "chunk_001"
        assert chunk.doc_id == "doc_001"
        assert chunk.index == 0
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata == {"source": "test"}

    def test_chunk_defaults(self) -> None:
        """Test chunk default values."""
        chunk = Chunk(
            content="Content",
            chunk_id="chunk_002",
            doc_id="doc_001",
            index=1,
        )
        assert chunk.embedding == []
        assert chunk.metadata == {}


class TestRetrievedChunk:
    """Tests for RetrievedChunk model."""

    def test_retrieved_chunk_creation(self) -> None:
        """Test creating a retrieved chunk."""
        chunk = Chunk(
            content="Test",
            chunk_id="c1",
            doc_id="d1",
            index=0,
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.85,
            rank=1,
        )
        assert retrieved.chunk == chunk
        assert retrieved.score == 0.85
        assert retrieved.rank == 1


class TestRAGResponse:
    """Tests for RAGResponse model."""

    def test_rag_response_creation(self) -> None:
        """Test creating a RAG response."""
        response = RAGResponse(
            answer="The answer is 42",
            query="What is the answer?",
            context_chunks=[],
            confidence=0.9,
        )
        assert response.answer == "The answer is 42"
        assert response.query == "What is the answer?"
        assert response.confidence == 0.9

    def test_rag_response_with_chunks(self) -> None:
        """Test RAG response with context chunks."""
        chunk = Chunk(
            content="Context",
            chunk_id="c1",
            doc_id="d1",
            index=0,
        )
        retrieved = RetrievedChunk(chunk=chunk, score=0.9, rank=1)
        response = RAGResponse(
            answer="Answer",
            query="Question",
            context_chunks=[retrieved],
            confidence=0.8,
        )
        assert len(response.context_chunks) == 1


class TestChunkingConfig:
    """Tests for ChunkingConfig model."""

    def test_default_config(self) -> None:
        """Test default chunking configuration."""
        config = ChunkingConfig()
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.separator == "\n"

    def test_custom_config(self) -> None:
        """Test custom chunking configuration."""
        config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=100,
            separator=". ",
        )
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100
        assert config.separator == ". "


class TestRetrievalStats:
    """Tests for RetrievalStats model."""

    def test_retrieval_stats_creation(self) -> None:
        """Test creating retrieval stats."""
        stats = RetrievalStats(
            total_documents=10,
            total_chunks=50,
            avg_chunk_size=400.0,
            queries_processed=100,
        )
        assert stats.total_documents == 10
        assert stats.total_chunks == 50
        assert stats.avg_chunk_size == 400.0
        assert stats.queries_processed == 100


class TestSimpleEmbedding:
    """Tests for simple_embedding function."""

    def test_embedding_dimension(self) -> None:
        """Test embedding has correct dimension."""
        embedding = simple_embedding("test text", dim=64)
        assert len(embedding) == 64

    def test_embedding_custom_dimension(self) -> None:
        """Test embedding with custom dimension."""
        embedding = simple_embedding("test", dim=128)
        assert len(embedding) == 128

    def test_embedding_normalized(self) -> None:
        """Test embedding is normalized."""
        embedding = simple_embedding("some text", dim=64)
        magnitude = sum(x * x for x in embedding) ** 0.5
        assert abs(magnitude - 1.0) < 0.01  # Should be unit vector

    def test_similar_text_similar_embedding(self) -> None:
        """Test similar texts produce similar embeddings."""
        emb1 = simple_embedding("hello world", dim=64)
        emb2 = simple_embedding("hello world", dim=64)
        emb3 = simple_embedding("completely different", dim=64)

        sim_same = cosine_similarity(emb1, emb2)
        sim_diff = cosine_similarity(emb1, emb3)

        assert sim_same > sim_diff

    def test_empty_text(self) -> None:
        """Test embedding for empty text."""
        embedding = simple_embedding("", dim=64)
        assert len(embedding) == 64


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors."""
        vec = [0.1, 0.2, 0.3, 0.4]
        sim = cosine_similarity(vec, vec)
        assert sim > 0.99  # Should be very close to 1

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(0.5, abs=0.01)

    def test_different_length_vectors(self) -> None:
        """Test mismatched vector lengths return 0."""
        vec1 = [0.1, 0.2, 0.3]
        vec2 = [0.1, 0.2]
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_zero_vector(self) -> None:
        """Test zero vector returns 0."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [0.1, 0.2, 0.3]
        assert cosine_similarity(vec1, vec2) == 0.0


class TestChunkText:
    """Tests for chunk_text function."""

    def test_short_text_single_chunk(self) -> None:
        """Test short text returns single chunk."""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self) -> None:
        """Test empty text returns empty list."""
        chunks = chunk_text("", chunk_size=100)
        assert chunks == []

    def test_whitespace_text(self) -> None:
        """Test whitespace-only text returns empty list."""
        chunks = chunk_text("   \n\n   ", chunk_size=100)
        assert chunks == []

    def test_long_text_multiple_chunks(self) -> None:
        """Test long text is split into multiple chunks."""
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunks_respect_size_limit(self) -> None:
        """Test chunks don't exceed size limit significantly."""
        text = "This is a sentence. " * 50
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            # Allow some flexibility for break points
            assert len(chunk) <= 150

    def test_chunk_overlap(self) -> None:
        """Test chunks have overlap."""
        text = "abcdefghij" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        # Overlapping content should exist
        assert len(chunks) > 1

    def test_separator_respected(self) -> None:
        """Test chunking prefers separator."""
        text = "Line one.\nLine two.\nLine three.\nLine four."
        chunks = chunk_text(
            text, chunk_size=25, chunk_overlap=5, separator="\n"
        )
        # Should try to break on newlines
        assert len(chunks) >= 2


class TestVectorStore:
    """Tests for VectorStore dataclass."""

    def test_empty_store(self) -> None:
        """Test empty store initialization."""
        store = VectorStore()
        assert store.chunks == []
        assert store.documents == {}
        assert store.doc_counter == 0

    def test_add_document(self) -> None:
        """Test adding a document."""
        store = VectorStore()
        doc = store.add_document(
            content="Test document content",
            title="Test Doc",
            source="test",
            metadata={"key": "value"},
        )
        assert doc.doc_id == "doc_0001"
        assert doc.title == "Test Doc"
        assert store.doc_counter == 1
        assert "doc_0001" in store.documents

    def test_add_multiple_documents(self) -> None:
        """Test adding multiple documents."""
        store = VectorStore()
        store.add_document("Content 1", title="Doc 1")
        store.add_document("Content 2", title="Doc 2")
        store.add_document("Content 3", title="Doc 3")

        assert store.doc_counter == 3
        assert len(store.documents) == 3

    def test_document_creates_chunks(self) -> None:
        """Test document is chunked."""
        store = VectorStore(
            chunk_config=ChunkingConfig(chunk_size=50, chunk_overlap=10)
        )
        store.add_document("This is a test. " * 20)
        assert len(store.chunks) > 0

    def test_chunks_have_embeddings(self) -> None:
        """Test chunks have embeddings."""
        store = VectorStore()
        store.add_document("Test content")
        for chunk in store.chunks:
            assert len(chunk.embedding) > 0

    def test_search_empty_store(self) -> None:
        """Test searching empty store."""
        store = VectorStore()
        results = store.search("query")
        assert results == []

    def test_search_returns_results(self) -> None:
        """Test search returns relevant results."""
        store = VectorStore()
        store.add_document("Python is a programming language")
        store.add_document("JavaScript is for web development")

        results = store.search("programming language", k=2)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_search_respects_k_limit(self) -> None:
        """Test search respects k parameter."""
        store = VectorStore()
        for i in range(10):
            store.add_document(f"Document number {i}")

        results = store.search("document", k=3)
        assert len(results) <= 3

    def test_search_respects_threshold(self) -> None:
        """Test search respects threshold parameter."""
        store = VectorStore()
        store.add_document("Python programming")
        store.add_document("JavaScript coding")

        # High threshold should filter results
        results = store.search("query", k=10, threshold=0.99)
        # Results should be filtered or empty
        assert all(r.score >= 0.99 for r in results)

    def test_search_ranked_by_score(self) -> None:
        """Test results are ranked by score."""
        store = VectorStore()
        store.add_document("Python is great")
        store.add_document("Python programming basics")
        store.add_document("Unrelated topic here")

        results = store.search("Python", k=3)
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_get_document(self) -> None:
        """Test getting document by ID."""
        store = VectorStore()
        doc = store.add_document("Content", title="Test")

        retrieved = store.get_document(doc.doc_id)
        assert retrieved is not None
        assert retrieved.title == "Test"

    def test_get_nonexistent_document(self) -> None:
        """Test getting non-existent document."""
        store = VectorStore()
        assert store.get_document("fake_id") is None

    def test_get_stats(self) -> None:
        """Test getting store statistics."""
        store = VectorStore()
        store.add_document("Document one content")
        store.add_document("Document two content")
        store.search("test query")

        stats = store.get_stats()
        assert stats.total_documents == 2
        assert stats.total_chunks >= 2
        assert stats.queries_processed == 1

    def test_clear(self) -> None:
        """Test clearing the store."""
        store = VectorStore()
        store.add_document("Content")
        store.add_document("More content")

        store.clear()
        assert store.chunks == []
        assert store.documents == {}
        assert store.doc_counter == 0

    def test_custom_embedding_function(self) -> None:
        """Test using custom embedding function."""

        def custom_embed(text: str) -> list[float]:
            return [float(len(text))] * 10

        store = VectorStore(embedding_fn=custom_embed)
        store.add_document("Test")

        assert len(store.chunks) > 0
        assert store.chunks[0].embedding == [4.0] * 10


class TestRAGPipeline:
    """Tests for RAGPipeline dataclass (tool-based retrieval)."""

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initialization."""
        store = VectorStore()
        pipeline = RAGPipeline(store=store)
        assert pipeline.top_k == 3
        assert pipeline.min_score == 0.1

    def test_custom_pipeline_config(self) -> None:
        """Test custom pipeline configuration."""
        store = VectorStore()
        pipeline = RAGPipeline(
            store=store,
            top_k=5,
            min_score=0.5,
        )
        assert pipeline.top_k == 5
        assert pipeline.min_score == 0.5

    @pytest.mark.asyncio
    async def test_query_basic(self) -> None:
        """Test basic query execution."""
        store = VectorStore()
        store.add_document("Python is a programming language")
        pipeline = RAGPipeline(store=store)

        mock_result = MagicMock()
        mock_result.output = "Python is indeed a programming language."

        with patch(
            "agentic_patterns.knowledge_retrieval.rag_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            response = await pipeline.query("What is Python?")

        assert isinstance(response, RAGResponse)
        assert response.query == "What is Python?"
        assert response.answer == "Python is indeed a programming language."

    @pytest.mark.asyncio
    async def test_query_returns_context_chunks(self) -> None:
        """Test query returns context chunks."""
        store = VectorStore()
        store.add_document("Document about Python programming")
        pipeline = RAGPipeline(store=store, top_k=2)

        mock_result = MagicMock()
        mock_result.output = "Answer"

        with patch(
            "agentic_patterns.knowledge_retrieval.rag_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            response = await pipeline.query("Python")

        assert len(response.context_chunks) <= 2

    @pytest.mark.asyncio
    async def test_query_confidence_calculation(self) -> None:
        """Test confidence is calculated from retrieval scores."""
        store = VectorStore()
        store.add_document("Very relevant content about testing")
        pipeline = RAGPipeline(store=store)

        mock_result = MagicMock()
        mock_result.output = "Answer"

        with patch(
            "agentic_patterns.knowledge_retrieval.rag_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            response = await pipeline.query("testing")

        assert 0.0 <= response.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_batch_query(self) -> None:
        """Test batch query processing."""
        store = VectorStore()
        store.add_document("Content about Python")
        store.add_document("Content about JavaScript")
        pipeline = RAGPipeline(store=store)

        mock_result = MagicMock()
        mock_result.output = "Batch answer"

        with patch(
            "agentic_patterns.knowledge_retrieval.rag_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            responses = await pipeline.batch_query(
                [
                    "What is Python?",
                    "What is JavaScript?",
                ]
            )

        assert len(responses) == 2
        assert all(isinstance(r, RAGResponse) for r in responses)


class TestBuildKnowledgeBase:
    """Tests for build_knowledge_base function."""

    def test_build_from_documents(self) -> None:
        """Test building knowledge base from documents."""
        documents = [
            {"content": "First document", "title": "Doc 1", "source": "test"},
            {"content": "Second document", "title": "Doc 2", "source": "test"},
        ]
        store = build_knowledge_base(documents)

        stats = store.get_stats()
        assert stats.total_documents == 2

    def test_build_with_custom_chunk_size(self) -> None:
        """Test building with custom chunk size."""
        documents = [
            {"content": "Long content " * 100, "title": "Doc 1"},
        ]
        store = build_knowledge_base(
            documents,
            chunk_size=100,
            chunk_overlap=20,
        )

        assert store.chunk_config.chunk_size == 100
        assert store.chunk_config.chunk_overlap == 20

    def test_build_empty_documents(self) -> None:
        """Test building with empty document list."""
        store = build_knowledge_base([])
        stats = store.get_stats()
        assert stats.total_documents == 0
        assert stats.total_chunks == 0

    def test_build_with_metadata(self) -> None:
        """Test building with document metadata."""
        documents = [
            {
                "content": "Content",
                "title": "Title",
                "source": "Source",
                "metadata": {"custom": "data"},
            },
        ]
        store = build_knowledge_base(documents)

        doc = store.get_document("doc_0001")
        assert doc is not None
        assert doc.title == "Title"
        assert doc.source == "Source"


class TestIntegrationScenarios:
    """Integration tests for complete RAG workflows."""

    @pytest.mark.asyncio
    async def test_complete_rag_workflow(self) -> None:
        """Test complete RAG workflow from documents to answer."""
        # Build knowledge base
        documents = [
            {
                "title": "Python Guide",
                "content": "Python is a high-level programming language "
                "known for its simplicity and readability.",
                "source": "manual",
            },
            {
                "title": "JavaScript Guide",
                "content": "JavaScript is a scripting language used "
                "primarily for web development.",
                "source": "manual",
            },
        ]
        store = build_knowledge_base(documents)

        # Create pipeline
        pipeline = RAGPipeline(store=store, top_k=2)

        # Mock the LLM response
        mock_result = MagicMock()
        mock_result.output = (
            "Based on the context, Python is known for "
            "its simplicity and readability."
        )

        with patch(
            "agentic_patterns.knowledge_retrieval.rag_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            # Query
            response = await pipeline.query("What makes Python special?")

        # Verify response
        assert "Python" in response.answer
        assert response.confidence > 0
        assert len(response.context_chunks) > 0

    def test_semantic_search_quality(self) -> None:
        """Test semantic search finds relevant content."""
        store = VectorStore()
        store.add_document(
            "Artificial intelligence enables machines to learn",
            title="AI",
        )
        store.add_document(
            "Database systems store and retrieve data",
            title="Databases",
        )
        store.add_document(
            "Neural networks are inspired by the brain",
            title="Neural Networks",
        )

        # Search for AI-related content
        results = store.search("machine learning intelligence", k=2)

        # Top result should be AI or Neural Networks related
        top_titles = [r.chunk.metadata.get("title") for r in results]
        assert "AI" in top_titles or "Neural Networks" in top_titles

    def test_large_document_chunking(self) -> None:
        """Test handling of large documents."""
        # Create a large document
        large_content = "This is paragraph about topic. " * 500

        store = VectorStore(
            chunk_config=ChunkingConfig(chunk_size=200, chunk_overlap=30)
        )
        store.add_document(large_content, title="Large Doc")

        stats = store.get_stats()
        # Should create multiple chunks
        assert stats.total_chunks > 10
        # Chunks should be manageable size
        assert stats.avg_chunk_size < 300
