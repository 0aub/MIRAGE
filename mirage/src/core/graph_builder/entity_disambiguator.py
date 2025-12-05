"""
Entity Disambiguation Module for GraphRAG

Implements cross-encoder based entity linking to match query entities
to graph entities. This solves the problem of:
- "IBM" vs "International Business Machines"
- "Dr. Ahmed" vs "Ahmed Hassan"
- "Ministry of Health" vs "Health Ministry"

Uses sentence-transformers cross-encoder for semantic similarity,
combined with rule-based normalization for hybrid matching.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

# Import centralized constants (relative import for package compatibility)
from ..config.constants import (
    ENTITY_SIMILARITY_THRESHOLD,
    ENTITY_MAX_CONFIDENCE,
    MAX_ENTITIES_IN_CONTEXT,
)

# Required imports - no fallback (enforced in Docker)
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np


@dataclass
class DisambiguationResult:
    """Result of entity disambiguation."""
    query_entity: str
    matched_entity: Optional[str]
    similarity_score: float
    match_type: str  # "exact", "normalized", "semantic", "none"
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    context_used: bool = False

    def to_dict(self) -> Dict:
        return {
            "query_entity": self.query_entity,
            "matched_entity": self.matched_entity,
            "similarity_score": self.similarity_score,
            "match_type": self.match_type,
            "candidates": self.candidates[:5],  # Top 5
            "context_used": self.context_used
        }


@dataclass
class EntityCandidate:
    """A candidate entity from the graph."""
    name: str
    type: str
    aliases: List[str] = field(default_factory=list)
    mention_count: int = 0
    embedding: Optional[List[float]] = None


class EntityDisambiguator:
    """
    Entity disambiguation using cross-encoder semantic similarity.

    Strategy (in order of preference):
    1. Exact match - Query entity exactly matches graph entity
    2. Normalized match - After rule-based normalization
    3. Semantic match - Cross-encoder similarity above threshold

    Features:
    - Cross-encoder based semantic similarity
    - Context-aware disambiguation (uses surrounding text)
    - Alias support (IBM -> International Business Machines)
    - Caching for performance
    - Bilingual support (Arabic + English)

    Usage:
        disambiguator = EntityDisambiguator(neo4j_client)
        result = disambiguator.disambiguate(
            query_entity="IBM",
            entity_type="Organization",
            context="IBM announced new AI products"
        )
        print(f"Matched: {result.matched_entity} ({result.match_type})")
    """

    # Common aliases for well-known entities
    KNOWN_ALIASES = {
        # Organizations
        "IBM": ["International Business Machines", "IBM Corp", "IBM Corporation"],
        "MS": ["Microsoft", "Microsoft Corporation"],
        "MSFT": ["Microsoft"],
        "Google": ["Alphabet", "Google Inc", "Google LLC"],
        "Meta": ["Facebook", "Meta Platforms"],
        # Arabic organizations
        "أرامكو": ["أرامكو السعودية", "شركة أرامكو", "Saudi Aramco", "Aramco"],
        "سابك": ["شركة سابك", "SABIC"],
        # Saudi ministries
        "وزارة الصحة": ["Ministry of Health", "MOH"],
        "وزارة التعليم": ["Ministry of Education", "MOE"],
        "وزارة الداخلية": ["Ministry of Interior", "MOI"],
    }

    # Cross-encoder model for semantic similarity
    DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        neo4j_client,
        cross_encoder_model: str = None,
        similarity_threshold: float = None,
        use_caching: bool = True
    ):
        """
        Initialize entity disambiguator.

        Args:
            neo4j_client: Neo4j client for graph access
            cross_encoder_model: Model name for cross-encoder (default: ms-marco-MiniLM)
            similarity_threshold: Minimum similarity for semantic match (default: from constants)
            use_caching: Whether to cache entity lookups
        """
        self.neo4j_client = neo4j_client
        self.similarity_threshold = similarity_threshold or ENTITY_SIMILARITY_THRESHOLD
        self.use_caching = use_caching

        # Initialize cross-encoder (required, no fallback)
        model_name = cross_encoder_model or self.DEFAULT_CROSS_ENCODER
        self.cross_encoder = CrossEncoder(model_name)
        logger.info(f"Cross-encoder loaded: {model_name}")

        # Sentence transformer for additional embedding capabilities
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer loaded")

        # Entity normalizer for rule-based matching
        from .entity_normalizer import EntityNormalizer
        self.normalizer = EntityNormalizer()

        # Caches
        self._entity_cache: Dict[str, List[EntityCandidate]] = {}
        self._disambiguation_cache: Dict[Tuple[str, str], DisambiguationResult] = {}
        self._alias_index: Dict[str, str] = self._build_alias_index()

    def _build_alias_index(self) -> Dict[str, str]:
        """Build reverse alias index."""
        index = {}
        for canonical, aliases in self.KNOWN_ALIASES.items():
            for alias in aliases:
                index[alias.lower()] = canonical
            # Also map canonical to itself
            index[canonical.lower()] = canonical
        return index

    def disambiguate(
        self,
        query_entity: str,
        entity_type: Optional[str] = None,
        context: Optional[str] = None,
        top_k: int = 5
    ) -> DisambiguationResult:
        """
        Disambiguate a query entity against graph entities.

        Args:
            query_entity: Entity name from query
            entity_type: Optional entity type filter (Person, Organization, etc.)
            context: Optional surrounding context for better disambiguation
            top_k: Number of candidates to consider

        Returns:
            DisambiguationResult with matched entity and metadata

        Example:
            >>> result = disambiguator.disambiguate("IBM", "Organization")
            >>> print(f"Matched: {result.matched_entity}")
        """
        if not query_entity:
            return DisambiguationResult(
                query_entity="",
                matched_entity=None,
                similarity_score=0.0,
                match_type="none"
            )

        # Check cache first
        cache_key = (query_entity.lower(), entity_type or "")
        if self.use_caching and cache_key in self._disambiguation_cache:
            return self._disambiguation_cache[cache_key]

        # Step 1: Check known aliases
        alias_match = self._check_alias(query_entity)
        if alias_match:
            result = DisambiguationResult(
                query_entity=query_entity,
                matched_entity=alias_match,
                similarity_score=1.0,
                match_type="exact"
            )
            self._cache_result(cache_key, result)
            return result

        # Step 2: Get candidate entities from graph
        candidates = self._get_candidates(entity_type, top_k * 3)

        if not candidates:
            result = DisambiguationResult(
                query_entity=query_entity,
                matched_entity=None,
                similarity_score=0.0,
                match_type="none"
            )
            return result

        # Step 3: Try exact match
        exact_match = self._exact_match(query_entity, candidates)
        if exact_match:
            result = DisambiguationResult(
                query_entity=query_entity,
                matched_entity=exact_match.name,
                similarity_score=1.0,
                match_type="exact",
                candidates=[(exact_match.name, 1.0)]
            )
            self._cache_result(cache_key, result)
            return result

        # Step 4: Try normalized match
        normalized_query = self.normalizer.normalize_entity_name(
            query_entity, entity_type or "Unknown"
        )
        normalized_match = self._normalized_match(normalized_query, candidates)
        if normalized_match:
            result = DisambiguationResult(
                query_entity=query_entity,
                matched_entity=normalized_match.name,
                similarity_score=0.95,
                match_type="normalized",
                candidates=[(normalized_match.name, 0.95)]
            )
            self._cache_result(cache_key, result)
            return result

        # Step 5: Semantic match using cross-encoder
        semantic_result = self._semantic_match(
            query_entity, candidates, context, top_k
        )

        self._cache_result(cache_key, semantic_result)
        return semantic_result

    def _check_alias(self, query_entity: str) -> Optional[str]:
        """Check if query matches a known alias."""
        return self._alias_index.get(query_entity.lower())

    def _get_candidates(
        self,
        entity_type: Optional[str],
        limit: int
    ) -> List[EntityCandidate]:
        """Get candidate entities from graph."""
        cache_key = entity_type or "all"

        if self.use_caching and cache_key in self._entity_cache:
            return self._entity_cache[cache_key][:limit]

        # Query Neo4j for entities
        query = """
        MATCH (e:Entity)
        WHERE $entity_type IS NULL OR e.type = $entity_type
        RETURN e.name as name, e.type as type,
               e.aliases as aliases, e.mention_count as mention_count
        ORDER BY e.mention_count DESC
        LIMIT $limit
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'entity_type': entity_type,
                'limit': limit * 2  # Get more for caching
            })

            candidates = []
            for r in results:
                candidate = EntityCandidate(
                    name=r.get('name', ''),
                    type=r.get('type', 'Unknown'),
                    aliases=r.get('aliases', []) or [],
                    mention_count=r.get('mention_count', 0) or 0
                )
                candidates.append(candidate)

            # Cache results
            if self.use_caching:
                self._entity_cache[cache_key] = candidates

            return candidates[:limit]

        except Exception as e:
            logger.error(f"Error getting entity candidates: {e}")
            return []

    def _exact_match(
        self,
        query_entity: str,
        candidates: List[EntityCandidate]
    ) -> Optional[EntityCandidate]:
        """Find exact match in candidates."""
        query_lower = query_entity.lower()

        for candidate in candidates:
            if candidate.name.lower() == query_lower:
                return candidate

            # Check aliases
            for alias in candidate.aliases:
                if alias.lower() == query_lower:
                    return candidate

        return None

    def _normalized_match(
        self,
        normalized_query: str,
        candidates: List[EntityCandidate]
    ) -> Optional[EntityCandidate]:
        """Find match after normalization."""
        normalized_lower = normalized_query.lower()

        for candidate in candidates:
            normalized_candidate = self.normalizer.normalize_entity_name(
                candidate.name, candidate.type
            )
            if normalized_candidate.lower() == normalized_lower:
                return candidate

        return None

    def _semantic_match(
        self,
        query_entity: str,
        candidates: List[EntityCandidate],
        context: Optional[str],
        top_k: int
    ) -> DisambiguationResult:
        """
        Find semantic match using cross-encoder or embedding similarity.
        """
        if not candidates:
            return DisambiguationResult(
                query_entity=query_entity,
                matched_entity=None,
                similarity_score=0.0,
                match_type="none"
            )

        # Prepare query (with context if available)
        if context:
            query_text = f"{query_entity}: {context[:200]}"
            context_used = True
        else:
            query_text = query_entity
            context_used = False

        # Prepare candidate texts
        candidate_texts = [c.name for c in candidates]

        # Calculate similarities using cross-encoder (primary method)
        similarities = self._cross_encoder_similarity(
            query_text, candidate_texts
        )

        # Rank candidates
        scored_candidates = list(zip(candidates, similarities))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Get top candidates
        top_candidates = [(c.name, s) for c, s in scored_candidates[:top_k]]

        # Check if best match exceeds threshold
        best_candidate, best_score = scored_candidates[0]

        if best_score >= self.similarity_threshold:
            return DisambiguationResult(
                query_entity=query_entity,
                matched_entity=best_candidate.name,
                similarity_score=best_score,
                match_type="semantic",
                candidates=top_candidates,
                context_used=context_used
            )
        else:
            return DisambiguationResult(
                query_entity=query_entity,
                matched_entity=None,
                similarity_score=best_score,
                match_type="none",
                candidates=top_candidates,
                context_used=context_used
            )

    def _cross_encoder_similarity(
        self,
        query: str,
        candidates: List[str]
    ) -> List[float]:
        """Calculate similarity using cross-encoder."""
        try:
            # Create query-candidate pairs
            pairs = [[query, candidate] for candidate in candidates]

            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(pairs)

            # Normalize to [0, 1]
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                normalized = [
                    (s - min_score) / (max_score - min_score)
                    for s in scores
                ]
            else:
                normalized = [0.5] * len(scores)

            return normalized

        except Exception as e:
            logger.error(f"Cross-encoder error: {e}")
            return [0.0] * len(candidates)

    def _embedding_similarity(
        self,
        query: str,
        candidates: List[str]
    ) -> List[float]:
        """Calculate similarity using sentence embeddings."""
        try:
            # Encode query and candidates
            query_embedding = self.sentence_transformer.encode([query])[0]
            candidate_embeddings = self.sentence_transformer.encode(candidates)

            # Calculate cosine similarity
            similarities = []
            for candidate_emb in candidate_embeddings:
                similarity = np.dot(query_embedding, candidate_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_emb)
                )
                similarities.append(float(similarity))

            return similarities

        except Exception as e:
            logger.error(f"Embedding similarity error: {e}")
            return [0.0] * len(candidates)

    def _fuzzy_similarity(
        self,
        query: str,
        candidates: List[str]
    ) -> List[float]:
        """
        Fallback fuzzy string similarity (no ML required).
        Uses Levenshtein-like ratio.
        """
        similarities = []
        query_lower = query.lower()

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Calculate character overlap ratio
            query_chars = set(query_lower)
            candidate_chars = set(candidate_lower)

            if not query_chars or not candidate_chars:
                similarities.append(0.0)
                continue

            intersection = len(query_chars & candidate_chars)
            union = len(query_chars | candidate_chars)
            jaccard = intersection / union if union > 0 else 0

            # Bonus for substring match
            substring_bonus = 0.0
            if query_lower in candidate_lower or candidate_lower in query_lower:
                substring_bonus = 0.3

            # Combined score
            score = min(1.0, jaccard + substring_bonus)
            similarities.append(score)

        return similarities

    def _cache_result(
        self,
        key: Tuple[str, str],
        result: DisambiguationResult
    ):
        """Cache disambiguation result."""
        if self.use_caching:
            self._disambiguation_cache[key] = result

    def disambiguate_batch(
        self,
        entities: List[Dict[str, str]],
        context: Optional[str] = None
    ) -> List[DisambiguationResult]:
        """
        Disambiguate multiple entities efficiently.

        Args:
            entities: List of dicts with 'name' and optional 'type'
            context: Shared context for all entities

        Returns:
            List of DisambiguationResult for each entity
        """
        results = []
        for entity in entities:
            name = entity.get('name', '')
            entity_type = entity.get('type')

            result = self.disambiguate(
                query_entity=name,
                entity_type=entity_type,
                context=context
            )
            results.append(result)

        return results

    def link_query_entities(
        self,
        query: str,
        extracted_entities: List[Dict[str, str]]
    ) -> Dict[str, DisambiguationResult]:
        """
        Link extracted entities from a query to graph entities.

        Args:
            query: Original query text (used as context)
            extracted_entities: Entities extracted from query

        Returns:
            Dict mapping entity names to disambiguation results
        """
        results = {}

        for entity in extracted_entities:
            name = entity.get('name', entity.get('text', ''))
            entity_type = entity.get('type')

            if not name:
                continue

            result = self.disambiguate(
                query_entity=name,
                entity_type=entity_type,
                context=query
            )
            results[name] = result

        return results

    def clear_cache(self):
        """Clear all caches."""
        self._entity_cache.clear()
        self._disambiguation_cache.clear()
        logger.info("Disambiguation caches cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'entity_cache_size': len(self._entity_cache),
            'disambiguation_cache_size': len(self._disambiguation_cache),
            'alias_index_size': len(self._alias_index)
        }


# Singleton instance
_disambiguator_instance: Optional[EntityDisambiguator] = None


def get_entity_disambiguator(neo4j_client=None) -> EntityDisambiguator:
    """
    Get or create singleton EntityDisambiguator instance.

    Args:
        neo4j_client: Neo4j client (required on first call)

    Returns:
        EntityDisambiguator instance
    """
    global _disambiguator_instance

    if _disambiguator_instance is None:
        if neo4j_client is None:
            raise ValueError("neo4j_client required for first initialization")
        _disambiguator_instance = EntityDisambiguator(neo4j_client)

    return _disambiguator_instance
