"""
Test Entity Disambiguation Module

Tests the cross-encoder based entity disambiguation including:
- Exact matching
- Normalized matching
- Semantic matching
- Alias resolution
- Batch disambiguation
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestEntityDisambiguator:
    """Test entity disambiguation functionality"""

    def test_import_module(self):
        """Test that disambiguation module can be imported"""
        from core.graph_builder.entity_disambiguator import (
            EntityDisambiguator,
            DisambiguationResult,
            get_entity_disambiguator,
        )
        assert EntityDisambiguator is not None
        assert DisambiguationResult is not None

    def test_disambiguation_result_dataclass(self):
        """Test DisambiguationResult dataclass"""
        from core.graph_builder.entity_disambiguator import DisambiguationResult

        result = DisambiguationResult(
            query_entity="IBM",
            matched_entity="International Business Machines",
            similarity_score=0.95,
            match_type="semantic",
            candidates=[("IBM Corp", 0.90), ("Microsoft", 0.30)],
            context_used=True
        )

        assert result.query_entity == "IBM"
        assert result.matched_entity == "International Business Machines"
        assert result.similarity_score == 0.95
        assert result.match_type == "semantic"
        assert result.context_used == True

        # Test to_dict
        d = result.to_dict()
        assert d["query_entity"] == "IBM"
        assert d["match_type"] == "semantic"

    def test_known_aliases(self):
        """Test that known aliases are properly indexed"""
        from core.graph_builder.entity_disambiguator import EntityDisambiguator

        # Check that KNOWN_ALIASES exists
        assert hasattr(EntityDisambiguator, 'KNOWN_ALIASES')
        aliases = EntityDisambiguator.KNOWN_ALIASES

        # Check IBM alias
        assert "IBM" in aliases
        assert "International Business Machines" in aliases["IBM"]

        # Check Arabic alias
        assert "أرامكو" in aliases or "سابك" in aliases


class TestEntityNormalizer:
    """Test entity normalization (used by disambiguator)"""

    def test_normalize_person_name(self):
        """Test person name normalization"""
        from core.graph_builder.entity_normalizer import EntityNormalizer

        normalizer = EntityNormalizer()

        # Test title removal
        assert normalizer.normalize_entity_name("Dr. Ahmed Hassan", "Person") == "Ahmed Hassan"
        assert normalizer.normalize_entity_name("Prof. John Smith", "Person") == "John Smith"

    def test_normalize_organization_name(self):
        """Test organization name normalization"""
        from core.graph_builder.entity_normalizer import EntityNormalizer

        normalizer = EntityNormalizer()

        # Test suffix removal
        result = normalizer.normalize_entity_name("IBM Corp.", "Organization")
        assert "corp" not in result.lower()

    def test_normalize_arabic_name(self):
        """Test Arabic name normalization"""
        from core.graph_builder.entity_normalizer import EntityNormalizer

        normalizer = EntityNormalizer()

        # Test Arabic title removal
        result = normalizer.normalize_entity_name("الدكتور محمد علي", "Person")
        assert "الدكتور" not in result

    def test_arabic_character_normalization(self):
        """Test Arabic character variations"""
        from core.graph_builder.entity_normalizer import EntityNormalizer

        normalizer = EntityNormalizer()

        # Test Alef normalization
        result1 = normalizer.normalize_entity_name("أحمد", "Person")
        result2 = normalizer.normalize_entity_name("احمد", "Person")
        # Both should normalize to same form
        assert result1 == result2


class TestFuzzySimilarity:
    """Test fuzzy similarity matching"""

    def test_substring_match(self):
        """Test substring matching bonus"""
        from core.graph_builder.entity_disambiguator import EntityDisambiguator

        class MockClient:
            def execute_query(self, *args, **kwargs):
                return []

        disambiguator = EntityDisambiguator(MockClient())

        # Test fuzzy similarity
        similarities = disambiguator._fuzzy_similarity(
            "IBM",
            ["IBM Corporation", "Microsoft", "Apple"]
        )

        # IBM should be highest (substring match)
        assert similarities[0] > similarities[1]
        assert similarities[0] > similarities[2]

    def test_jaccard_similarity(self):
        """Test Jaccard-based character similarity"""
        from core.graph_builder.entity_disambiguator import EntityDisambiguator

        class MockClient:
            def execute_query(self, *args, **kwargs):
                return []

        disambiguator = EntityDisambiguator(MockClient())

        # Test similar names
        similarities = disambiguator._fuzzy_similarity(
            "Ahmed Hassan",
            ["Ahmed Hasan", "John Smith", "Ahmed H."]
        )

        # First two should be higher than John Smith
        assert similarities[0] > similarities[1]
        assert similarities[2] > similarities[1]


class TestDisambiguationIntegration:
    """Test disambiguation integration with retrieval"""

    def test_retrieval_engine_has_disambiguator(self):
        """Test that RetrievalEngine has disambiguator property"""
        from core.retrieval.retrieval_engine import RetrievalEngine

        engine = RetrievalEngine()

        # Should have property (may be None if not initialized)
        assert hasattr(engine, 'entity_disambiguator')

    def test_disambiguation_available_flag(self):
        """Test DISAMBIGUATOR_AVAILABLE flag"""
        from core.retrieval import retrieval_engine

        # Should have the flag defined
        assert hasattr(retrieval_engine, 'DISAMBIGUATOR_AVAILABLE')


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
