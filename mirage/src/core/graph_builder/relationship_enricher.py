"""
LLM-Based Relationship Enricher
Enriches entity pairs with semantic relationships using LLM inference.

Features:
- Infers relationship types from context
- Supports Arabic and English
- Batch processing for efficiency
- Relationship confidence scoring
- Integration with Neo4j for enrichment
"""

import os
import json
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

# Required imports - no fallback (enforced in Docker)
import redis
import tiktoken

from ...config import settings
from .relationship_normalizer import get_relationship_normalizer, NormalizedRelationship


@dataclass
class EnrichedRelationship:
    """Enriched relationship with inferred type and metadata"""
    source_entity: str
    target_entity: str
    relationship_type: str
    normalized_type: str
    cypher_type: str
    confidence: float
    description: Optional[str] = None
    evidence: Optional[str] = None
    category: str = "other"
    bidirectional: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relationship_type,
            "normalized_type": self.normalized_type,
            "cypher_type": self.cypher_type,
            "confidence": self.confidence,
            "description": self.description,
            "evidence": self.evidence,
            "category": self.category,
            "bidirectional": self.bidirectional,
            "attributes": self.attributes,
            "source_type": "llm_enriched",
        }


class RelationshipEnricher:
    """
    Enriches entity pairs with semantic relationships using LLM.

    Can be used to:
    1. Infer relationships between newly extracted entities
    2. Enrich existing graph with more specific relationship types
    3. Validate and refine co-occurrence relationships
    """

    # Relationship inference prompt templates
    ARABIC_SYSTEM_PROMPT = """أنت متخصص في تحليل العلاقات بين الكيانات. مهمتك تحديد نوع العلاقة الدقيق بين كيانين بناءً على السياق المعطى.

**تعليمات:**
1. حلل العلاقة بين الكيانين في السياق
2. اختر نوع العلاقة الأكثر تحديداً ودقة
3. تجنب العلاقات العامة مثل "مرتبط_بـ" أو "له_علاقة_مع"
4. قدم درجة الثقة (0.0-1.0) بناءً على وضوح العلاقة في السياق

**أنواع العلاقات المفضلة:**
- قيادة: يرأس، يدير، يشرف_على، يقود
- إنشاء: أسس، أطلق، أنشأ، طور، صمم
- عمل: يعمل_في، موظف_في، عضو_في
- موقع: يقع_في، مقره_في، عُقد_في، استضاف
- مشاركة: شارك_في، حضر، نظم، دعم
- جوائز: حصل_على، فاز_بـ، نال
- تعاون: يتعاون_مع، شريك
- استخدام: يستخدم، يطبق

**تنسيق الرد (JSON فقط):**
{"relationship": "نوع_العلاقة", "confidence": 0.85, "description": "وصف مختصر", "evidence": "الدليل من السياق"}"""

    ENGLISH_SYSTEM_PROMPT = """You are an expert in analyzing relationships between entities. Your task is to determine the precise relationship type between two entities based on the given context.

**Instructions:**
1. Analyze the relationship between the two entities in the context
2. Choose the most specific and accurate relationship type
3. Avoid generic relationships like "related_to" or "associated_with"
4. Provide a confidence score (0.0-1.0) based on how clear the relationship is in the context

**Preferred Relationship Types:**
- Leadership: leads, manages, supervises, heads
- Creation: founded, launched, created, developed, designed
- Employment: works_at, employed_by, member_of
- Location: located_in, headquartered_in, held_in, hosted
- Participation: participated_in, attended, organized, supported
- Awards: received, won, awarded
- Collaboration: collaborates_with, partner_of
- Usage: uses, applies, employs

**Response Format (JSON only):**
{"relationship": "relationship_type", "confidence": 0.85, "description": "brief description", "evidence": "evidence from context"}"""

    def __init__(
        self,
        min_confidence: float = 0.5,
        use_cache: bool = True,
        cache_ttl: int = 86400  # 24 hours
    ):
        """
        Initialize the relationship enricher.

        Args:
            min_confidence: Minimum confidence to accept a relationship
            use_cache: Whether to cache LLM results
            cache_ttl: Cache TTL in seconds
        """
        self.min_confidence = min_confidence
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl

        # Initialize normalizer
        self.normalizer = get_relationship_normalizer()

        # Initialize provider
        self.provider = None
        self.model = None
        self._detect_provider()

        # Initialize token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize Redis for caching
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.redis_client.ping()

        # Statistics
        self._stats = {
            "enrichments_attempted": 0,
            "enrichments_successful": 0,
            "cache_hits": 0,
            "low_confidence_filtered": 0,
        }

        logger.info(
            f"RelationshipEnricher initialized: provider={self.provider}, "
            f"min_confidence={min_confidence}"
        )

    def _detect_provider(self) -> None:
        """Auto-detect LLM provider"""
        if settings.use_tgi and settings.tgi_endpoint:
            self.provider = "tgi"
            self.model = "tgi"
            self.tgi_endpoint = settings.tgi_endpoint
            logger.info(f"Using TGI for relationship enrichment: {settings.tgi_endpoint}")
        elif settings.openai_api_key:
            self.provider = "openai"
            self.model = "gpt-4o-mini"
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        elif settings.anthropic_api_key:
            self.provider = "anthropic"
            self.model = "claude-3-haiku-20240307"
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        elif settings.google_api_key:
            self.provider = "google"
            self.model = "gemini/gemini-2.0-flash-exp"
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        else:
            logger.warning("No LLM provider available for relationship enrichment")

    def _get_cache_key(
        self,
        entity1: str,
        entity2: str,
        context_hash: str
    ) -> str:
        """Generate cache key for relationship"""
        # Sort entities for consistent key
        entities = sorted([entity1.lower(), entity2.lower()])
        return f"rel_enrich:{entities[0]}:{entities[1]}:{context_hash}"

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if result is cached"""
        if not self.use_cache:
            return None

        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")

        return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache enrichment result"""
        if not self.use_cache:
            return

        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result)
            )
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def enrich_relationship(
        self,
        entity1: str,
        entity1_type: str,
        entity2: str,
        entity2_type: str,
        context: str,
        language: str = "auto"
    ) -> Optional[EnrichedRelationship]:
        """
        Infer the relationship type between two entities using LLM.

        Args:
            entity1: First entity name
            entity1_type: First entity type (e.g., "Person", "Organization")
            entity2: Second entity name
            entity2_type: Second entity type
            context: Text context containing both entities
            language: Language of the context ("ar", "en", or "auto")

        Returns:
            EnrichedRelationship or None if no valid relationship found
        """
        if not self.provider:
            logger.warning("No LLM provider available")
            return None

        self._stats["enrichments_attempted"] += 1

        # Detect language if needed
        if language == "auto":
            arabic_ratio = len([c for c in context if '\u0600' <= c <= '\u06FF']) / max(len(context), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        # Check cache
        import hashlib
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        cache_key = self._get_cache_key(entity1, entity2, context_hash)

        cached = self._check_cache(cache_key)
        if cached:
            return self._parse_cached_result(entity1, entity2, cached)

        # Build prompt
        system_prompt = self.ARABIC_SYSTEM_PROMPT if language == "ar" else self.ENGLISH_SYSTEM_PROMPT

        if language == "ar":
            user_prompt = f"""حدد العلاقة بين الكيانين التاليين:

الكيان الأول: {entity1} (نوع: {entity1_type})
الكيان الثاني: {entity2} (نوع: {entity2_type})

السياق:
{context[:2000]}

JSON:"""
        else:
            user_prompt = f"""Determine the relationship between these two entities:

Entity 1: {entity1} (type: {entity1_type})
Entity 2: {entity2} (type: {entity2_type})

Context:
{context[:2000]}

JSON:"""

        # Call LLM
        try:
            result = self._call_llm(system_prompt, user_prompt)
            if not result:
                return None

            # Parse response
            relationship_type = result.get("relationship", "")
            confidence = float(result.get("confidence", 0.5))
            description = result.get("description", "")
            evidence = result.get("evidence", "")

            # Filter low confidence
            if confidence < self.min_confidence:
                self._stats["low_confidence_filtered"] += 1
                logger.debug(
                    f"Low confidence relationship filtered: "
                    f"{entity1} -> {entity2}: {relationship_type} ({confidence})"
                )
                return None

            # Normalize relationship type
            normalized = self.normalizer.normalize(relationship_type)
            cypher_type = self.normalizer.to_cypher_type(relationship_type)

            # Create enriched relationship
            enriched = EnrichedRelationship(
                source_entity=entity1,
                target_entity=entity2,
                relationship_type=relationship_type,
                normalized_type=normalized.normalized_type,
                cypher_type=cypher_type,
                confidence=confidence * normalized.confidence,  # Combine confidences
                description=description,
                evidence=evidence,
                category=normalized.category,
                bidirectional=normalized.bidirectional,
            )

            # Cache result
            self._cache_result(cache_key, {
                "relationship_type": relationship_type,
                "confidence": confidence,
                "description": description,
                "evidence": evidence,
            })

            self._stats["enrichments_successful"] += 1

            logger.debug(
                f"Enriched relationship: {entity1} -[{cypher_type}]-> {entity2} "
                f"(confidence: {enriched.confidence:.2f})"
            )

            return enriched

        except Exception as e:
            logger.error(f"Error enriching relationship: {e}")
            return None

    def _parse_cached_result(
        self,
        entity1: str,
        entity2: str,
        cached: Dict[str, Any]
    ) -> Optional[EnrichedRelationship]:
        """Parse cached result into EnrichedRelationship"""
        relationship_type = cached.get("relationship_type", "")
        confidence = float(cached.get("confidence", 0.5))

        if confidence < self.min_confidence:
            return None

        normalized = self.normalizer.normalize(relationship_type)
        cypher_type = self.normalizer.to_cypher_type(relationship_type)

        return EnrichedRelationship(
            source_entity=entity1,
            target_entity=entity2,
            relationship_type=relationship_type,
            normalized_type=normalized.normalized_type,
            cypher_type=cypher_type,
            confidence=confidence * normalized.confidence,
            description=cached.get("description", ""),
            evidence=cached.get("evidence", ""),
            category=normalized.category,
            bidirectional=normalized.bidirectional,
        )

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Optional[Dict[str, Any]]:
        """Call LLM and parse JSON response"""
        try:
            if self.provider == "tgi":
                response = requests.post(
                    f"{self.tgi_endpoint}/v1/chat/completions",
                    json={
                        "model": "tgi",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 256,
                        "top_p": 0.9,
                    },
                    timeout=15
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            else:
                # Use LiteLLM for cloud providers
                from litellm import completion
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def enrich_batch(
        self,
        entity_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], str]],
        language: str = "auto"
    ) -> List[EnrichedRelationship]:
        """
        Enrich multiple entity pairs in batch.

        Args:
            entity_pairs: List of (entity1, entity2, context) tuples
                          where entity is {"name": str, "type": str}
            language: Language of the context

        Returns:
            List of EnrichedRelationship objects
        """
        results = []

        for entity1, entity2, context in entity_pairs:
            enriched = self.enrich_relationship(
                entity1=entity1["name"],
                entity1_type=entity1.get("type", "Unknown"),
                entity2=entity2["name"],
                entity2_type=entity2.get("type", "Unknown"),
                context=context,
                language=language
            )

            if enriched:
                results.append(enriched)

        logger.info(
            f"Batch enrichment: {len(results)}/{len(entity_pairs)} relationships enriched"
        )

        return results

    def enrich_cooccurrence_relationships(
        self,
        cooccurrence_rels: List[Dict[str, Any]],
        chunk_texts: Dict[str, str],
        language: str = "auto"
    ) -> List[EnrichedRelationship]:
        """
        Enrich co-occurrence relationships with semantic types.

        Takes COOCCURS_WITH relationships and infers specific relationship types.

        Args:
            cooccurrence_rels: List of co-occurrence relationships
                               with "source", "target", "chunks"
            chunk_texts: Mapping of chunk_id -> chunk text
            language: Language of the text

        Returns:
            List of enriched relationships
        """
        results = []

        for rel in cooccurrence_rels:
            source = rel.get("source", "")
            target = rel.get("target", "")
            chunk_ids = rel.get("chunks", [])

            # Get context from first available chunk
            context = ""
            for chunk_id in chunk_ids:
                if chunk_id in chunk_texts:
                    context = chunk_texts[chunk_id]
                    break

            if not context:
                logger.debug(f"No context found for {source} -> {target}")
                continue

            # Enrich with LLM
            enriched = self.enrich_relationship(
                entity1=source,
                entity1_type=rel.get("source_type", "Unknown"),
                entity2=target,
                entity2_type=rel.get("target_type", "Unknown"),
                context=context,
                language=language
            )

            if enriched:
                # Preserve co-occurrence metadata
                enriched.attributes["cooccurrence_frequency"] = rel.get("frequency", 1)
                enriched.attributes["cooccurrence_chunks"] = chunk_ids
                results.append(enriched)

        logger.info(
            f"Co-occurrence enrichment: {len(results)}/{len(cooccurrence_rels)} "
            "relationships enriched with semantic types"
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get enrichment statistics"""
        return {
            **self._stats,
            "provider": self.provider,
            "min_confidence": self.min_confidence,
            "success_rate": (
                self._stats["enrichments_successful"] /
                max(self._stats["enrichments_attempted"], 1)
            ),
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_relationship_enricher: Optional[RelationshipEnricher] = None


def get_relationship_enricher() -> RelationshipEnricher:
    """Get or create global RelationshipEnricher instance"""
    global _relationship_enricher

    if _relationship_enricher is None:
        _relationship_enricher = RelationshipEnricher()

    return _relationship_enricher


def enrich_relationship(
    entity1: str,
    entity1_type: str,
    entity2: str,
    entity2_type: str,
    context: str,
    language: str = "auto"
) -> Optional[EnrichedRelationship]:
    """Convenience function to enrich a single relationship"""
    return get_relationship_enricher().enrich_relationship(
        entity1, entity1_type, entity2, entity2_type, context, language
    )
