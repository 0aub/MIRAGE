"""
Test MIRAGE Centralized Configuration Constants

Tests that:
1. All constants are importable
2. Constants have valid types and values
3. Related constants are consistent
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfigConstants:
    """Test centralized configuration constants"""

    def test_import_all_constants(self):
        """Test that all constants can be imported"""
        from core.config.constants import (
            # Entity extraction
            ENTITY_SIMILARITY_THRESHOLD,
            ENTITY_CONFIDENCE_BOOST_MULTI,
            ENTITY_CONFIDENCE_BOOST_DUAL,
            ENTITY_FREQUENCY_BOOST,
            ENTITY_FREQUENCY_THRESHOLD,
            ENTITY_MAX_CONFIDENCE,
            ENTITY_WEIGHT_LLM,
            ENTITY_WEIGHT_CAMEL,
            ENTITY_WEIGHT_SPACY,
            ENTITY_WEIGHT_PATTERN,
            # Relationship extraction
            COOCCURRENCE_WINDOW,
            MIN_RELATIONSHIP_CONFIDENCE,
            MAX_RELATIONSHIPS_PER_PAIR,
            # Community detection
            LOUVAIN_RESOLUTION,
            MIN_COMMUNITY_SIZE,
            MAX_COMMUNITY_LEVELS,
            DEFAULT_COMMUNITY_LEVEL,
            # Community summarization
            MAX_ENTITIES_IN_SUMMARY,
            MAX_RELATIONSHIPS_IN_SUMMARY,
            MAX_SUMMARY_TOKENS,
            SUMMARIZATION_TEMPERATURE,
            # Retrieval configuration
            RETRIEVAL_TOP_K_DEFAULT,
            RETRIEVAL_TOP_K_MAX,
            RETRIEVAL_MIN_SCORE,
            VALIDATION_RELEVANCE_THRESHOLD,
            VALIDATION_MIN_RELEVANT_RATIO,
            VALIDATION_MAX_CORRECTIONS,
            # Fusion & ranking
            RRF_K,
            # REFRAG compression
            REFRAG_CHUNK_SIZE_TOKENS,
            REFRAG_COMPRESSION_RATIO_BASE,
            # Context limits
            MAX_CONTEXT_TOKENS,
            MAX_RESPONSE_TOKENS,
            MAX_ENTITIES_IN_CONTEXT,
            MAX_RELATIONSHIPS_IN_CONTEXT,
            MAX_COMMUNITIES_TO_SEARCH,
            GLOBAL_SEARCH_MIN_RELEVANCE,
            # Evaluation targets
            TARGET_TTFT_MS,
            TARGET_MRR,
            TARGET_FAITHFULNESS,
        )

        # If we get here, all imports succeeded
        assert True

    def test_entity_similarity_threshold_range(self):
        """Test that entity similarity threshold is in valid range [0, 1]"""
        from core.config.constants import ENTITY_SIMILARITY_THRESHOLD

        assert 0.0 <= ENTITY_SIMILARITY_THRESHOLD <= 1.0
        assert ENTITY_SIMILARITY_THRESHOLD >= 0.5, "Threshold should be high enough to prevent false positives"

    def test_confidence_values_range(self):
        """Test that confidence values are in valid range [0, 1]"""
        from core.config.constants import (
            ENTITY_CONFIDENCE_BOOST_MULTI,
            ENTITY_CONFIDENCE_BOOST_DUAL,
            ENTITY_FREQUENCY_BOOST,
            ENTITY_MAX_CONFIDENCE,
        )

        assert 0.0 <= ENTITY_CONFIDENCE_BOOST_MULTI <= 1.0
        assert 0.0 <= ENTITY_CONFIDENCE_BOOST_DUAL <= 1.0
        assert 0.0 <= ENTITY_FREQUENCY_BOOST <= 1.0
        assert 0.0 <= ENTITY_MAX_CONFIDENCE <= 1.0

        # Max confidence should not be 100%
        assert ENTITY_MAX_CONFIDENCE < 1.0, "Max confidence should not be 100%"

    def test_weight_ordering(self):
        """Test that extraction source weights follow expected hierarchy"""
        from core.config.constants import (
            ENTITY_WEIGHT_LLM,
            ENTITY_WEIGHT_CAMEL,
            ENTITY_WEIGHT_SPACY,
            ENTITY_WEIGHT_PATTERN,
        )

        # LLM should have highest weight (best contextual understanding)
        assert ENTITY_WEIGHT_LLM > ENTITY_WEIGHT_CAMEL
        assert ENTITY_WEIGHT_CAMEL > ENTITY_WEIGHT_SPACY
        assert ENTITY_WEIGHT_SPACY > ENTITY_WEIGHT_PATTERN

    def test_retrieval_config_consistency(self):
        """Test that retrieval configuration is consistent"""
        from core.config.constants import (
            RETRIEVAL_TOP_K_DEFAULT,
            RETRIEVAL_TOP_K_MAX,
            RETRIEVAL_MIN_SCORE,
        )

        # Default should be less than max
        assert RETRIEVAL_TOP_K_DEFAULT <= RETRIEVAL_TOP_K_MAX

        # Min score should be in [0, 1]
        assert 0.0 <= RETRIEVAL_MIN_SCORE <= 1.0

    def test_context_limits_consistency(self):
        """Test that context limits are consistent"""
        from core.config.constants import (
            MAX_CONTEXT_TOKENS,
            MAX_RESPONSE_TOKENS,
            MAX_ENTITIES_IN_CONTEXT,
            MAX_RELATIONSHIPS_IN_CONTEXT,
        )

        # Context + Response should fit in Allam's 2K limit
        assert MAX_CONTEXT_TOKENS + MAX_RESPONSE_TOKENS <= 2048

        # Entity and relationship limits should be reasonable
        assert MAX_ENTITIES_IN_CONTEXT > 0
        assert MAX_RELATIONSHIPS_IN_CONTEXT > 0

    def test_evaluation_targets_reasonable(self):
        """Test that evaluation targets are reasonable"""
        from core.config.constants import (
            TARGET_TTFT_MS,
            TARGET_E2E_LATENCY_MS,
            TARGET_MRR,
            TARGET_FAITHFULNESS,
        )

        # TTFT should be faster than E2E
        assert TARGET_TTFT_MS < TARGET_E2E_LATENCY_MS

        # Quality metrics should be in [0, 1]
        assert 0.0 <= TARGET_MRR <= 1.0
        assert 0.0 <= TARGET_FAITHFULNESS <= 1.0

        # Targets should be ambitious but achievable
        assert TARGET_MRR >= 0.5, "MRR target should be at least 0.5"
        assert TARGET_FAITHFULNESS >= 0.8, "Faithfulness target should be high"

    def test_llm_temperature_valid(self):
        """Test that LLM temperature values are valid"""
        from core.config.constants import (
            SUMMARIZATION_TEMPERATURE,
            LLM_TEMPERATURE_DEFAULT,
            LLM_TEMPERATURE_FACTUAL,
        )

        # Temperature should be in reasonable range
        for temp in [SUMMARIZATION_TEMPERATURE, LLM_TEMPERATURE_DEFAULT, LLM_TEMPERATURE_FACTUAL]:
            assert 0.0 <= temp <= 2.0, f"Temperature {temp} out of range"

        # Factual temperature should be lower (more deterministic)
        assert LLM_TEMPERATURE_FACTUAL <= LLM_TEMPERATURE_DEFAULT

    def test_rrf_k_standard(self):
        """Test that RRF k parameter follows standard"""
        from core.config.constants import RRF_K

        # Standard RRF k is 60 (from literature)
        assert RRF_K == 60, "RRF k should be 60 (standard from literature)"

    def test_embedding_config(self):
        """Test embedding configuration"""
        from core.config.constants import (
            EMBEDDING_DIM,
            EMBEDDING_MODEL,
            EMBEDDING_BATCH_SIZE,
        )

        # Jina v3 has 1024 dimensions
        assert EMBEDDING_DIM == 1024

        # Model should be Jina
        assert "jina" in EMBEDDING_MODEL.lower()

        # Batch size should be reasonable
        assert 1 <= EMBEDDING_BATCH_SIZE <= 128


class TestConfigInit:
    """Test config __init__.py exports"""

    def test_init_exports(self):
        """Test that __init__.py exports all constants"""
        from core.config import (
            ENTITY_SIMILARITY_THRESHOLD,
            MAX_CONTEXT_TOKENS,
            RETRIEVAL_TOP_K_DEFAULT,
            TARGET_TTFT_MS,
        )

        assert ENTITY_SIMILARITY_THRESHOLD is not None
        assert MAX_CONTEXT_TOKENS is not None
        assert RETRIEVAL_TOP_K_DEFAULT is not None
        assert TARGET_TTFT_MS is not None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
