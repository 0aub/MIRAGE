"""
Test True REFRAG Implementation

Tests the Meta-style REFRAG with:
- Chunk embedding
- Fast retrieval
- Speedup calculations
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestREFRAGChunk:
    """Test REFRAG chunk dataclass"""

    def test_refrag_chunk_creation(self):
        """Test REFRAGChunk creation"""
        from core.refrag.chunk_embedder import REFRAGChunk

        chunk = REFRAGChunk(
            chunk_id="doc1_0_abc123",
            document_id="doc1",
            text="This is a test chunk",
            token_count=5,
            position=0
        )

        assert chunk.chunk_id == "doc1_0_abc123"
        assert chunk.token_count == 5
        assert chunk.embedding is None

    def test_refrag_chunk_to_dict(self):
        """Test REFRAGChunk serialization"""
        from core.refrag.chunk_embedder import REFRAGChunk

        chunk = REFRAGChunk(
            chunk_id="test_chunk",
            document_id="doc1",
            text="Test text",
            token_count=2,
            position=0
        )

        d = chunk.to_dict()
        assert "chunk_id" in d
        assert "has_embedding" in d
        assert d["has_embedding"] == False


class TestChunkEmbedder:
    """Test chunk embedder functionality"""

    def test_chunk_embedder_init(self):
        """Test ChunkEmbedder initialization"""
        from core.refrag.chunk_embedder import ChunkEmbedder

        embedder = ChunkEmbedder(chunk_size_tokens=16)
        assert embedder.chunk_size_tokens == 16

    def test_document_chunking(self):
        """Test document is split into chunks"""
        from core.refrag.chunk_embedder import ChunkEmbedder

        embedder = ChunkEmbedder(chunk_size_tokens=4)

        # 16 words should create 4 chunks of 4 tokens each
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"

        refrag_doc = embedder.embed_document(text, "test_doc")

        assert refrag_doc.document_id == "test_doc"
        assert len(refrag_doc.chunks) == 4
        assert refrag_doc.total_tokens == 16

    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        from core.refrag.chunk_embedder import ChunkEmbedder

        embedder = ChunkEmbedder(chunk_size_tokens=16)

        # 160 tokens should compress to 10 chunks
        tokens = ["word"] * 160
        text = " ".join(tokens)

        refrag_doc = embedder.embed_document(text, "test_doc")

        # Compression ratio should be ~16x
        assert refrag_doc.compression_ratio == 16.0

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        from core.refrag.chunk_embedder import ChunkEmbedder

        embedder = ChunkEmbedder()

        # Identical vectors should have similarity 1.0
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert embedder._cosine_similarity(a, b) == 1.0

        # Orthogonal vectors should have similarity 0.0
        c = np.array([0.0, 1.0, 0.0])
        assert embedder._cosine_similarity(a, c) == 0.0


class TestREFRAGRetriever:
    """Test REFRAG retriever"""

    def test_retriever_init(self):
        """Test REFRAGRetriever initialization"""
        from core.refrag.refrag_retriever import REFRAGRetriever

        retriever = REFRAGRetriever()
        assert retriever.max_context_tokens > 0

    def test_document_indexing(self):
        """Test document indexing"""
        from core.refrag.refrag_retriever import REFRAGRetriever

        retriever = REFRAGRetriever()

        text = "This is a test document with some content for testing."
        refrag_doc = retriever.index_document(text, "test_doc")

        assert refrag_doc.document_id == "test_doc"
        assert retriever.get_document("test_doc") is not None

    def test_retriever_stats(self):
        """Test retriever statistics"""
        from core.refrag.refrag_retriever import REFRAGRetriever

        retriever = REFRAGRetriever()

        # Index a document
        retriever.index_document("Test content", "doc1")

        stats = retriever.get_stats()
        assert stats["documents_indexed"] == 1
        assert "chunk_embedder_stats" in stats


class TestREFRAGRetrievalResult:
    """Test retrieval result dataclass"""

    def test_result_creation(self):
        """Test REFRAGRetrievalResult creation"""
        from core.refrag.refrag_retriever import REFRAGRetrievalResult

        result = REFRAGRetrievalResult(
            query="test query",
            context="Some context",
            chunks_retrieved=5,
            total_tokens=80,
            average_similarity=0.85,
            retrieval_time_ms=10.5,
            speedup_factor=16.0
        )

        assert result.speedup_factor == 16.0
        assert result.chunks_retrieved == 5

    def test_result_to_dict(self):
        """Test REFRAGRetrievalResult serialization"""
        from core.refrag.refrag_retriever import REFRAGRetrievalResult

        result = REFRAGRetrievalResult(
            query="test",
            context="context",
            chunks_retrieved=3,
            total_tokens=48,
            average_similarity=0.75,
            retrieval_time_ms=5.0,
            speedup_factor=10.0
        )

        d = result.to_dict()
        assert "speedup_factor" in d
        assert d["speedup_factor"] == 10.0


class TestREFRAGModule:
    """Test REFRAG module imports (all required, no fallbacks)"""

    def test_module_imports(self):
        """Test that all REFRAG components can be imported"""
        from core.refrag import (
            # Legacy
            REFRAGCompressor,
            CompressionPolicy,
            CompressionCache,
            # True REFRAG
            ChunkEmbedder,
            REFRAGRetriever,
            REFRAGRetrievalResult,
            # RL Policy
            RLExpansionPolicy,
            ExpansionDecision,
        )

        assert REFRAGCompressor is not None
        assert ChunkEmbedder is not None
        assert REFRAGRetriever is not None
        assert RLExpansionPolicy is not None

    def test_true_refrag_components(self):
        """Test that True REFRAG components are functional"""
        from core.refrag import (
            ChunkEmbedder,
            REFRAGRetriever,
        )

        # All components must be available (required, no fallback)
        assert ChunkEmbedder is not None
        assert REFRAGRetriever is not None


class TestSpeedupCalculation:
    """Test speedup calculation logic"""

    def test_speedup_formula(self):
        """Test that speedup is calculated correctly"""
        from core.refrag.refrag_retriever import REFRAGRetriever
        from core.refrag.chunk_embedder import REFRAGDocument, REFRAGChunk

        retriever = REFRAGRetriever()

        # Create mock document with known token count
        chunks = [
            REFRAGChunk(f"c{i}", "doc1", "test " * 16, 16, i)
            for i in range(10)
        ]
        doc = REFRAGDocument(
            document_id="doc1",
            chunks=chunks,
            total_tokens=160,  # 10 chunks * 16 tokens
            compression_ratio=16.0
        )

        # Estimate speedup
        retrieval_meta = {
            "total_tokens": 32,  # Retrieved 2 chunks worth
            "chunks_used": 2,
        }

        speedup = retriever._estimate_speedup([doc], retrieval_meta)

        # Should have significant speedup (160/32 * cache_benefit)
        assert speedup > 1.0
        assert speedup <= 50.0  # Capped at maximum


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
