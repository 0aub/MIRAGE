"""
Test RL Expansion Policy - Meta's REFRAG REINFORCE Implementation

Tests the REINFORCE-based expansion policy that decides which
chunks should be expanded from embeddings to full tokens.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestExpansionDecision:
    """Test ExpansionDecision dataclass"""

    def test_expansion_decision_creation(self):
        """Test ExpansionDecision creation"""
        from core.refrag.rl_expansion_policy import ExpansionDecision

        decision = ExpansionDecision(
            chunk_id="chunk_0",
            should_expand=True,
            expansion_probability=0.85,
            importance_score=0.9,
            features={"query_similarity": 0.9, "chunk_position": 0.0}
        )

        assert decision.chunk_id == "chunk_0"
        assert decision.should_expand == True
        assert decision.expansion_probability == 0.85
        assert decision.importance_score == 0.9

    def test_expansion_decision_to_dict(self):
        """Test ExpansionDecision serialization"""
        from core.refrag.rl_expansion_policy import ExpansionDecision

        decision = ExpansionDecision(
            chunk_id="test_chunk",
            should_expand=False,
            expansion_probability=0.3,
            importance_score=0.4
        )

        d = decision.to_dict()
        assert "chunk_id" in d
        assert "should_expand" in d
        assert d["should_expand"] == False
        assert d["expansion_probability"] == 0.3


class TestRLExpansionPolicy:
    """Test RLExpansionPolicy main class"""

    def test_policy_initialization(self):
        """Test RLExpansionPolicy initialization"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy(
            embedding_dim=768,
            budget_ratio=0.3
        )

        assert policy.budget_ratio == 0.3
        assert policy.embedding_dim == 768

    def test_policy_initialization_with_custom_params(self):
        """Test policy with custom hyperparameters"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy(
            embedding_dim=512,
            hidden_dim=64,
            learning_rate=1e-3,
            gamma=0.95,
            entropy_coef=0.02,
            budget_ratio=0.5
        )

        assert policy.budget_ratio == 0.5
        assert policy.gamma == 0.95
        assert policy.entropy_coef == 0.02

    def test_feature_names(self):
        """Test that feature names are defined"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        assert len(RLExpansionPolicy.FEATURE_NAMES) == 8
        assert "query_similarity" in RLExpansionPolicy.FEATURE_NAMES
        assert "chunk_position" in RLExpansionPolicy.FEATURE_NAMES
        assert "is_first_chunk" in RLExpansionPolicy.FEATURE_NAMES

    def test_decide_expansions_empty(self):
        """Test expansion decisions with empty chunks"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()
        decisions = policy.decide_expansions([], query_embedding=None)

        assert decisions == []

    def test_decide_expansions_basic(self):
        """Test basic expansion decisions"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy(budget_ratio=0.5)

        # Create mock chunks
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "text": f"Test chunk {i}",
                "token_count": 16,
                "embedding": np.random.randn(768),
                "importance_score": 0.5 + i * 0.1,
            }
            for i in range(10)
        ]

        query_embedding = np.random.randn(768)

        decisions = policy.decide_expansions(
            chunks=chunks,
            query_embedding=query_embedding
        )

        assert len(decisions) == 10
        assert all(hasattr(d, "should_expand") for d in decisions)
        assert all(hasattr(d, "expansion_probability") for d in decisions)

        # With budget_ratio=0.5, about 5 chunks should expand
        expanded = sum(1 for d in decisions if d.should_expand)
        assert expanded <= 5  # Should respect budget

    def test_decide_expansions_with_budget(self):
        """Test expansion with explicit budget"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        chunks = [
            {"chunk_id": f"c{i}", "text": f"text {i}", "token_count": 16}
            for i in range(10)
        ]

        decisions = policy.decide_expansions(
            chunks=chunks,
            query_embedding=None,
            budget=3
        )

        expanded = sum(1 for d in decisions if d.should_expand)
        assert expanded == 3

    def test_heuristic_fallback(self):
        """Test heuristic fallback when PyTorch not used"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        # Create chunks with varying importance
        chunks = [
            {
                "chunk_id": "high_importance",
                "text": "Important text about main topic",
                "token_count": 16,
                "embedding": np.random.randn(768),
                "importance_score": 0.95,
            },
            {
                "chunk_id": "low_importance",
                "text": "Less important text",
                "token_count": 16,
                "embedding": np.random.randn(768),
                "importance_score": 0.2,
            },
        ]

        decisions = policy.decide_expansions(
            chunks=chunks,
            query_embedding=np.random.randn(768),
            budget=1
        )

        # Higher importance chunk should be expanded
        high_imp = next(d for d in decisions if d.chunk_id == "high_importance")
        low_imp = next(d for d in decisions if d.chunk_id == "low_importance")

        assert high_imp.expansion_probability >= low_imp.expansion_probability


class TestFeatureExtraction:
    """Test feature extraction for policy network"""

    def test_feature_extraction(self):
        """Test that features are extracted correctly"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        query_emb = np.array([1.0, 0.0, 0.0])
        chunks = [
            {
                "chunk_id": "first",
                "text": "First chunk",
                "token_count": 16,
                "embedding": np.array([1.0, 0.0, 0.0]),  # Similar to query
                "entities": ["entity1", "entity2"],
            },
            {
                "chunk_id": "second",
                "text": "Second chunk",
                "token_count": 16,
                "embedding": np.array([0.0, 1.0, 0.0]),  # Orthogonal to query
            },
        ]

        features = policy._extract_features(chunks, query_emb)

        assert features.shape == (2, 8)

        # First chunk should have higher query similarity
        assert features[0, 0] > features[1, 0]

        # First chunk should have is_first_chunk = 1
        assert features[0, 4] == 1.0
        assert features[1, 4] == 0.0

        # Last chunk should have is_last_chunk = 1
        assert features[0, 5] == 0.0
        assert features[1, 5] == 1.0


class TestPerplexityReward:
    """Test perplexity-based reward calculation"""

    def test_perplexity_reward_calculation(self):
        """Test perplexity reward computation"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        # Mock logits and targets
        seq_len = 10
        vocab_size = 100

        # Create logits where correct token has high probability
        logits = np.zeros((seq_len, vocab_size))
        targets = np.arange(seq_len) % vocab_size

        for i in range(seq_len):
            logits[i, targets[i]] = 10.0  # High logit for correct token

        reward = policy.compute_perplexity_reward(logits, targets)

        # Low perplexity = higher (less negative) reward
        assert reward > -1.0  # Should be close to 0 for good predictions

    def test_perplexity_reward_bad_predictions(self):
        """Test reward for poor predictions"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        seq_len = 10
        vocab_size = 100

        # Random logits = bad predictions
        logits = np.random.randn(seq_len, vocab_size)
        targets = np.arange(seq_len) % vocab_size

        reward = policy.compute_perplexity_reward(logits, targets)

        # High perplexity = lower (more negative) reward
        assert reward < 0


class TestPolicyStats:
    """Test policy statistics tracking"""

    def test_get_stats(self):
        """Test stats retrieval"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy(budget_ratio=0.4)

        stats = policy.get_stats()

        assert "decisions_made" in stats
        assert "expansions_made" in stats
        assert "budget_ratio" in stats
        assert stats["budget_ratio"] == 0.4

    def test_stats_updated_after_decisions(self):
        """Test that stats are updated after decisions"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        initial_stats = policy.get_stats()
        initial_decisions = initial_stats["decisions_made"]

        chunks = [
            {"chunk_id": f"c{i}", "text": f"text {i}", "token_count": 16}
            for i in range(5)
        ]

        policy.decide_expansions(chunks, query_embedding=None, budget=2)

        updated_stats = policy.get_stats()
        assert updated_stats["decisions_made"] == initial_decisions + 5
        assert updated_stats["expansions_made"] >= 2


class TestCosineSimilarity:
    """Test cosine similarity helper"""

    def test_cosine_similarity_identical(self):
        """Test similarity of identical vectors"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        sim = policy._cosine_similarity(a, b)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Test similarity of orthogonal vectors"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])

        sim = policy._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_cosine_similarity_opposite(self):
        """Test similarity of opposite vectors"""
        from core.refrag.rl_expansion_policy import RLExpansionPolicy

        policy = RLExpansionPolicy()

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])

        sim = policy._cosine_similarity(a, b)
        assert abs(sim + 1.0) < 1e-6


class TestModuleImports:
    """Test module imports (all required, no fallbacks)"""

    def test_module_imports(self):
        """Test that all components can be imported"""
        from core.refrag import (
            RLExpansionPolicy,
            ExpansionDecision,
            get_rl_policy,
            train_expansion_policy,
        )

        assert RLExpansionPolicy is not None
        assert ExpansionDecision is not None
        assert get_rl_policy is not None
        assert train_expansion_policy is not None

    def test_get_rl_policy_factory(self):
        """Test get_rl_policy factory function"""
        from core.refrag import get_rl_policy

        policy = get_rl_policy()
        assert policy is not None
        assert hasattr(policy, "decide_expansions")


class TestIntegrationWithChunkEmbedder:
    """Test integration between RL policy and ChunkEmbedder"""

    def test_chunk_embedder_with_rl_policy(self):
        """Test ChunkEmbedder initializes with RL policy"""
        from core.refrag import ChunkEmbedder

        embedder = ChunkEmbedder(
            chunk_size_tokens=16,
            use_rl_policy=True,
            rl_budget_ratio=0.4
        )

        assert embedder.use_rl_policy == True
        assert embedder.rl_policy is not None

    def test_chunk_embedder_without_rl_policy(self):
        """Test ChunkEmbedder without RL policy"""
        from core.refrag import ChunkEmbedder

        embedder = ChunkEmbedder(
            chunk_size_tokens=16,
            use_rl_policy=False
        )

        assert embedder.use_rl_policy == False

    def test_rl_guided_selection(self):
        """Test RL-guided chunk selection in context building"""
        from core.refrag import ChunkEmbedder, REFRAGChunk

        embedder = ChunkEmbedder(
            chunk_size_tokens=16,
            use_rl_policy=True,
            rl_budget_ratio=0.5
        )

        # Create mock ranked chunks
        chunks_with_scores = []
        for i in range(10):
            chunk = REFRAGChunk(
                chunk_id=f"chunk_{i}",
                document_id="doc1",
                text=f"This is test chunk number {i} with content",
                token_count=8,
                embedding=np.random.randn(768),
                position=i
            )
            similarity = 0.9 - i * 0.05  # Decreasing similarity
            chunks_with_scores.append((chunk, similarity))

        query_embedding = np.random.randn(768)

        # Use RL-guided selection
        context, metadata = embedder._rl_guided_selection(
            query_embedding=query_embedding,
            ranked_chunks=chunks_with_scores,
            max_tokens=100
        )

        assert "selection_method" in metadata
        assert metadata["selection_method"] == "rl_policy"
        assert "rl_expansion_budget" in metadata
        assert metadata["chunks_used"] > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
