"""
Coreference Resolution - Resolve Entity Mentions to Canonical Forms

Problem:
- Same entity appears with different names:
  - "الملك سلمان" = "خادم الحرمين الشريفين" = "الملك سلمان بن عبدالعزيز"
  - "IBM" = "International Business Machines" = "آي بي إم"
  - "Saudi Arabia" = "KSA" = "المملكة العربية السعودية"

Solution:
- Resolve all mentions to canonical form before storage
- Maintain alias mappings
- Use context for disambiguation when needed

Benefits:
- No duplicate entities
- Better graph connectivity
- More accurate retrieval
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import requests


@dataclass
class ResolvedEntity:
    """Result of coreference resolution."""
    original: str
    canonical: str
    confidence: float
    method: str  # "cache", "rule", "context", "llm"
    aliases: List[str] = field(default_factory=list)
    entity_type: Optional[str] = None


class CoreferenceResolver:
    """
    Resolve entity mentions to canonical forms.

    Strategies (in order of preference):
    1. Cache lookup (fastest)
    2. Rule-based matching (fast, reliable)
    3. Context-based disambiguation (medium)
    4. LLM-based resolution (slow, last resort)

    Usage:
        resolver = CoreferenceResolver()
        result = resolver.resolve("الملك سلمان", context="زار الملك سلمان...")
        # result.canonical = "الملك سلمان بن عبدالعزيز"
    """

    def __init__(
        self,
        neo4j_client=None,
        embedder=None,
        llm_endpoint: str = "http://tgi:80",
        use_llm: bool = False
    ):
        """
        Initialize coreference resolver.

        Args:
            neo4j_client: Neo4j client for graph lookups
            embedder: Embedding manager for similarity
            llm_endpoint: TGI endpoint for LLM resolution
            use_llm: Whether to use LLM for resolution (slower)
        """
        self.neo4j_client = neo4j_client
        self.embedder = embedder
        self.llm_endpoint = llm_endpoint
        self.use_llm = use_llm

        # Cache: alias -> canonical
        self.alias_cache: Dict[str, str] = {}

        # Canonical forms database: canonical -> {aliases, type}
        self.canonical_db: Dict[str, Dict] = {}

        # Load built-in rules
        self._load_builtin_rules()

        logger.info("CoreferenceResolver initialized")

    def _load_builtin_rules(self):
        """Load built-in alias rules for common entities."""
        # Arabic/English entity mappings
        builtin_aliases = {
            # Countries
            "المملكة العربية السعودية": {
                "aliases": ["السعودية", "KSA", "Saudi Arabia", "Kingdom of Saudi Arabia"],
                "type": "Location"
            },
            "الإمارات العربية المتحدة": {
                "aliases": ["الإمارات", "UAE", "United Arab Emirates"],
                "type": "Location"
            },

            # Organizations (Saudi)
            "هيئة الحكومة الرقمية": {
                "aliases": ["الحكومة الرقمية", "DGA", "Digital Government Authority"],
                "type": "Organization"
            },
            "وزارة الموارد البشرية والتنمية الاجتماعية": {
                "aliases": ["وزارة الموارد البشرية", "HRSD", "وزارة العمل"],
                "type": "Organization"
            },

            # Common abbreviations
            "الذكاء الاصطناعي": {
                "aliases": ["AI", "Artificial Intelligence", "ذكاء اصطناعي"],
                "type": "Concept"
            },
            "التحول الرقمي": {
                "aliases": ["Digital Transformation", "الرقمنة"],
                "type": "Concept"
            },
        }

        for canonical, data in builtin_aliases.items():
            self.canonical_db[canonical] = data
            for alias in data.get("aliases", []):
                self.alias_cache[alias.lower()] = canonical
            # Self-reference
            self.alias_cache[canonical.lower()] = canonical

    def resolve(
        self,
        entity: str,
        context: str = "",
        entity_type: Optional[str] = None
    ) -> ResolvedEntity:
        """
        Resolve entity mention to canonical form.

        Args:
            entity: Entity mention to resolve
            context: Surrounding text for disambiguation
            entity_type: Optional type hint

        Returns:
            ResolvedEntity with canonical form
        """
        if not entity:
            return ResolvedEntity(
                original=entity,
                canonical=entity,
                confidence=0.0,
                method="empty"
            )

        # Strategy 1: Cache lookup
        cached = self._cache_lookup(entity)
        if cached:
            return cached

        # Strategy 2: Rule-based matching
        rule_match = self._rule_based_match(entity, entity_type)
        if rule_match:
            return rule_match

        # Strategy 3: Graph-based matching
        if self.neo4j_client:
            graph_match = self._graph_based_match(entity, context)
            if graph_match:
                return graph_match

        # Strategy 4: Context-based disambiguation
        if context:
            context_match = self._context_based_match(entity, context)
            if context_match:
                return context_match

        # Strategy 5: LLM-based resolution (if enabled)
        if self.use_llm:
            llm_match = self._llm_resolve(entity, context, entity_type)
            if llm_match:
                # Cache the result
                self.alias_cache[entity.lower()] = llm_match.canonical
                return llm_match

        # No match found - use normalized form as canonical
        canonical = self._normalize(entity)
        return ResolvedEntity(
            original=entity,
            canonical=canonical,
            confidence=0.5,
            method="normalized",
            entity_type=entity_type
        )

    def _cache_lookup(self, entity: str) -> Optional[ResolvedEntity]:
        """Look up entity in alias cache."""
        normalized = entity.lower().strip()

        if normalized in self.alias_cache:
            canonical = self.alias_cache[normalized]
            db_entry = self.canonical_db.get(canonical, {})

            return ResolvedEntity(
                original=entity,
                canonical=canonical,
                confidence=0.95,
                method="cache",
                aliases=db_entry.get("aliases", []),
                entity_type=db_entry.get("type")
            )

        return None

    def _rule_based_match(
        self,
        entity: str,
        entity_type: Optional[str]
    ) -> Optional[ResolvedEntity]:
        """Apply rule-based matching patterns."""
        normalized = entity.lower().strip()

        # Pattern 1: Title removal for person names
        # "الدكتور أحمد" -> "أحمد"
        title_patterns = [
            (r'^(الدكتور|الأستاذ|المهندس|السيد|الشيخ)\s+', ''),
            (r'^(Dr\.\s*|Prof\.\s*|Mr\.\s*|Ms\.\s*)', ''),
        ]

        for pattern, replacement in title_patterns:
            if re.match(pattern, normalized, re.IGNORECASE):
                cleaned = re.sub(pattern, replacement, entity, flags=re.IGNORECASE)
                return ResolvedEntity(
                    original=entity,
                    canonical=cleaned.strip(),
                    confidence=0.8,
                    method="rule_title",
                    entity_type=entity_type or "Person"
                )

        # Pattern 2: Acronym expansion
        # Check if we have an expansion in our DB
        for canonical, data in self.canonical_db.items():
            for alias in data.get("aliases", []):
                if alias.upper() == entity.upper():
                    return ResolvedEntity(
                        original=entity,
                        canonical=canonical,
                        confidence=0.9,
                        method="rule_acronym",
                        aliases=data.get("aliases", []),
                        entity_type=data.get("type")
                    )

        # Pattern 3: Common suffixes/prefixes
        common_variations = [
            (r'(شركة|مؤسسة|منظمة)\s*', ''),  # Company/institution prefix
            (r'\s*(المحدودة|القابضة)$', ''),  # Suffix
        ]

        for pattern, replacement in common_variations:
            if re.search(pattern, normalized):
                cleaned = re.sub(pattern, replacement, entity).strip()
                # Check if cleaned version exists
                if cleaned.lower() in self.alias_cache:
                    return self._cache_lookup(cleaned)

        return None

    def _graph_based_match(
        self,
        entity: str,
        context: str
    ) -> Optional[ResolvedEntity]:
        """Match against existing graph entities."""
        try:
            # Find similar entities in graph
            query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $term OR e.normalized_name CONTAINS $term
            RETURN e.name as name, e.type as type, e.aliases as aliases
            ORDER BY size(e.name)
            LIMIT 5
            """

            normalized = self._normalize(entity)
            results = self.neo4j_client.execute_query(query, {'term': normalized})

            for r in results:
                existing_name = r.get('name', '')

                # Check similarity
                similarity = self._string_similarity(entity, existing_name)
                if similarity >= 0.8:
                    return ResolvedEntity(
                        original=entity,
                        canonical=existing_name,
                        confidence=similarity,
                        method="graph",
                        aliases=r.get('aliases', []) or [],
                        entity_type=r.get('type')
                    )

        except Exception as e:
            logger.debug(f"Graph match failed: {e}")

        return None

    def _context_based_match(
        self,
        entity: str,
        context: str
    ) -> Optional[ResolvedEntity]:
        """Use context to disambiguate entity."""
        if not self.embedder:
            return None

        try:
            # Get entity + context embedding
            full_text = f"{entity}: {context[:200]}"
            query_embedding = self.embedder.embed(full_text)

            # Find candidates in graph with embeddings
            if self.neo4j_client:
                query = """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                RETURN e.name as name, e.type as type, e.embedding as embedding
                LIMIT 50
                """

                results = self.neo4j_client.execute_query(query)

                best_match = None
                best_similarity = 0.0

                import numpy as np

                for r in results:
                    embedding = r.get('embedding')
                    if embedding:
                        if isinstance(embedding, list):
                            embedding = np.array(embedding)

                        similarity = np.dot(query_embedding, embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                        )

                        if similarity > best_similarity and similarity >= 0.7:
                            best_similarity = similarity
                            best_match = r

                if best_match:
                    return ResolvedEntity(
                        original=entity,
                        canonical=best_match.get('name', entity),
                        confidence=float(best_similarity),
                        method="context",
                        entity_type=best_match.get('type')
                    )

        except Exception as e:
            logger.debug(f"Context match failed: {e}")

        return None

    def _llm_resolve(
        self,
        entity: str,
        context: str,
        entity_type: Optional[str]
    ) -> Optional[ResolvedEntity]:
        """Use LLM to resolve entity coreference."""
        prompt = f"""حدد الشكل الكامل الرسمي للكيان التالي:

الكيان: {entity}
السياق: {context[:300]}
النوع: {entity_type or "غير محدد"}

الشكل الرسمي الكامل:"""

        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 50,
                        "temperature": 0.3,
                        "do_sample": False
                    }
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                canonical = result.get('generated_text', '').strip()

                # Clean up response
                canonical = canonical.split('\n')[0].strip()
                canonical = re.sub(r'^(الشكل الرسمي:|الإجابة:)\s*', '', canonical)

                if canonical and len(canonical) > 2:
                    return ResolvedEntity(
                        original=entity,
                        canonical=canonical,
                        confidence=0.75,
                        method="llm",
                        entity_type=entity_type
                    )

        except Exception as e:
            logger.debug(f"LLM resolution failed: {e}")

        return None

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.strip())

        # Arabic-specific normalization
        # Normalize alef variants
        normalized = re.sub(r'[إأآا]', 'ا', normalized)
        # Normalize teh marbuta
        normalized = re.sub(r'ة', 'ه', normalized)

        return normalized

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Jaccard on character trigrams."""
        def get_trigrams(s: str) -> Set[str]:
            s = self._normalize(s.lower())
            s = f"  {s}  "
            return set(s[i:i+3] for i in range(len(s) - 2))

        t1 = get_trigrams(s1)
        t2 = get_trigrams(s2)

        if not t1 or not t2:
            return 0.0

        intersection = len(t1 & t2)
        union = len(t1 | t2)

        return intersection / union if union > 0 else 0.0

    def add_alias(
        self,
        canonical: str,
        alias: str,
        entity_type: Optional[str] = None
    ):
        """Add a new alias mapping."""
        normalized_alias = alias.lower().strip()
        self.alias_cache[normalized_alias] = canonical

        if canonical not in self.canonical_db:
            self.canonical_db[canonical] = {
                "aliases": [alias],
                "type": entity_type
            }
        else:
            if alias not in self.canonical_db[canonical].get("aliases", []):
                self.canonical_db[canonical]["aliases"].append(alias)

        logger.debug(f"Added alias: '{alias}' -> '{canonical}'")

    def get_aliases(self, canonical: str) -> List[str]:
        """Get all known aliases for a canonical form."""
        if canonical in self.canonical_db:
            return self.canonical_db[canonical].get("aliases", [])
        return []

    def merge_entities(
        self,
        entity1: str,
        entity2: str,
        canonical: Optional[str] = None
    ) -> str:
        """
        Merge two entities that refer to the same thing.

        Args:
            entity1: First entity name
            entity2: Second entity name
            canonical: Optional canonical form (defaults to entity1)

        Returns:
            The canonical form
        """
        canonical = canonical or entity1

        # Add both as aliases
        self.add_alias(canonical, entity1)
        self.add_alias(canonical, entity2)

        # Merge alias lists
        aliases1 = self.get_aliases(entity1)
        aliases2 = self.get_aliases(entity2)

        for alias in aliases1 + aliases2:
            self.add_alias(canonical, alias)

        return canonical


# Singleton
_coreference_resolver: Optional[CoreferenceResolver] = None


def get_coreference_resolver(**kwargs) -> CoreferenceResolver:
    """Get or create coreference resolver singleton."""
    global _coreference_resolver

    if _coreference_resolver is None:
        _coreference_resolver = CoreferenceResolver(**kwargs)

    return _coreference_resolver
