"""
Relationship Type Normalizer
Normalizes and standardizes relationship types across Arabic and English.

Features:
- Maps Arabic relationship types to English equivalents
- Standardizes relationship type naming (lowercase, underscores)
- Groups semantically similar relationships
- Provides confidence scoring for relationship types
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class NormalizedRelationship:
    """Normalized relationship with metadata"""
    original_type: str
    normalized_type: str
    english_type: str
    arabic_type: str
    category: str
    confidence: float
    bidirectional: bool = False


class RelationshipTypeNormalizer:
    """
    Normalizes relationship types across languages and variations.

    Maps semantic relationship types to standardized forms for:
    - Consistent Neo4j relationship type naming
    - Cross-language queries (Arabic ↔ English)
    - Relationship category grouping
    """

    # Relationship type mappings: Arabic -> English
    ARABIC_TO_ENGLISH: Dict[str, str] = {
        # Leadership & Management
        "يرأس": "leads",
        "يدير": "manages",
        "يشرف_على": "supervises",
        "يقود": "leads",
        "رئيس": "leads",
        "مدير": "manages",

        # Founding & Creation
        "أسس": "founded",
        "أطلق": "launched",
        "أنشأ": "created",
        "طور": "developed",
        "صمم": "designed",
        "بنى": "built",

        # Employment & Affiliation
        "يعمل_في": "works_at",
        "يعمل_لدى": "works_for",
        "عمل_في": "worked_at",
        "موظف_في": "employed_by",
        "عضو_في": "member_of",
        "ينتمي_إلى": "belongs_to",

        # Location
        "يقع_في": "located_in",
        "مقره_في": "headquartered_in",
        "عُقد_في": "held_in",
        "استضاف": "hosted",
        "في": "in",

        # Participation & Involvement
        "شارك_في": "participated_in",
        "حضر": "attended",
        "نظم": "organized",
        "أقام": "hosted",
        "دعم": "supported",
        "رعى": "sponsored",

        # Awards & Recognition
        "حصل_على": "received",
        "فاز_بـ": "won",
        "نال": "received",
        "منح": "awarded",

        # Communication & Announcement
        "أعلن_عن": "announced",
        "أعلن": "announced",
        "تحدث_عن": "spoke_about",
        "صرح_عن": "stated",
        "نشر": "published",

        # Ownership & Control
        "يملك": "owns",
        "يمتلك": "owns",
        "يسيطر_على": "controls",
        "يتحكم_في": "controls",

        # Collaboration
        "يتعاون_مع": "collaborates_with",
        "شريك": "partner_of",
        "تعاون_مع": "collaborated_with",

        # Usage & Application
        "يستخدم": "uses",
        "تستخدم": "uses",
        "يطبق": "applies",
        "يوظف": "employs",

        # Education & Learning
        "درس_في": "studied_at",
        "تخرج_من": "graduated_from",
        "حاصل_على": "holds_degree_from",

        # Investment & Funding
        "استثمر_في": "invested_in",
        "مول": "funded",
        "دعم_ماليا": "financially_supported",

        # Representation
        "يمثل": "represents",
        "ممثل": "represents",
        "سفير": "ambassador_for",
    }

    # English type standardization (handle variations)
    ENGLISH_STANDARDIZATION: Dict[str, str] = {
        # Variations -> Standard form
        "works_for": "works_at",
        "employed_by": "works_at",
        "employed_at": "works_at",
        "working_at": "works_at",

        "leads": "leads",
        "leading": "leads",
        "led": "leads",
        "head_of": "leads",
        "heads": "leads",

        "manages": "manages",
        "managing": "manages",
        "managed": "manages",
        "manager_of": "manages",

        "founded": "founded",
        "founder_of": "founded",
        "co-founded": "founded",
        "co_founded": "founded",

        "launched": "launched",
        "launching": "launched",
        "launches": "launched",

        "created": "created",
        "creates": "created",
        "creating": "created",

        "developed": "developed",
        "develops": "developed",
        "developing": "developed",

        "located_in": "located_in",
        "based_in": "located_in",
        "headquartered_in": "located_in",
        "situated_in": "located_in",

        "held_in": "held_in",
        "held_at": "held_in",
        "took_place_in": "held_in",

        "participated_in": "participated_in",
        "participated": "participated_in",
        "participates_in": "participated_in",

        "organized": "organized",
        "organizes": "organized",
        "organizing": "organized",
        "organizer_of": "organized",

        "received": "received",
        "receives": "received",
        "receiving": "received",

        "won": "won",
        "wins": "won",
        "winning": "won",
        "winner_of": "won",

        "announced": "announced",
        "announces": "announced",
        "announcing": "announced",

        "owns": "owns",
        "owned": "owns",
        "owner_of": "owns",

        "uses": "uses",
        "used": "uses",
        "using": "uses",
        "utilizes": "uses",

        "collaborates_with": "collaborates_with",
        "collaborated_with": "collaborates_with",
        "collaboration_with": "collaborates_with",
        "partner_of": "collaborates_with",
        "partners_with": "collaborates_with",

        "member_of": "member_of",
        "belongs_to": "member_of",
        "part_of": "member_of",

        "invested_in": "invested_in",
        "invests_in": "invested_in",
        "investor_in": "invested_in",

        "funded": "funded",
        "funds": "funded",
        "funding": "funded",
        "funder_of": "funded",

        "represents": "represents",
        "representing": "represents",
        "represented": "represents",

        "spoke_about": "spoke_about",
        "speaks_about": "spoke_about",
        "discussed": "spoke_about",

        "hosted": "hosted",
        "hosts": "hosted",
        "hosting": "hosted",

        "sponsored": "sponsored",
        "sponsors": "sponsored",
        "sponsoring": "sponsored",
        "sponsor_of": "sponsored",

        "supported": "supported",
        "supports": "supported",
        "supporting": "supported",

        "studied_at": "studied_at",
        "studied_in": "studied_at",
        "studies_at": "studied_at",

        "graduated_from": "graduated_from",
        "graduates_from": "graduated_from",
        "alumnus_of": "graduated_from",
    }

    # Relationship categories for semantic grouping
    RELATIONSHIP_CATEGORIES: Dict[str, List[str]] = {
        "leadership": ["leads", "manages", "supervises", "heads"],
        "creation": ["founded", "launched", "created", "developed", "designed", "built"],
        "employment": ["works_at", "employed_by", "member_of", "belongs_to"],
        "location": ["located_in", "headquartered_in", "held_in", "hosted", "in"],
        "participation": ["participated_in", "attended", "organized", "supported", "sponsored"],
        "recognition": ["received", "won", "awarded"],
        "communication": ["announced", "spoke_about", "stated", "published"],
        "ownership": ["owns", "controls"],
        "collaboration": ["collaborates_with", "partner_of"],
        "usage": ["uses", "applies", "employs"],
        "education": ["studied_at", "graduated_from", "holds_degree_from"],
        "investment": ["invested_in", "funded", "financially_supported"],
        "representation": ["represents", "ambassador_for"],
    }

    # Bidirectional relationships (symmetric)
    BIDIRECTIONAL_TYPES: set = {
        "collaborates_with",
        "partner_of",
        "related_to",
        "similar_to",
        "cooccurs_with",
    }

    # Confidence modifiers based on relationship specificity
    CONFIDENCE_MODIFIERS: Dict[str, float] = {
        # High confidence - very specific relationships
        "founded": 1.0,
        "leads": 0.95,
        "manages": 0.95,
        "works_at": 0.9,
        "located_in": 0.9,
        "won": 0.95,
        "graduated_from": 0.95,

        # Medium confidence - moderately specific
        "participated_in": 0.8,
        "organized": 0.85,
        "announced": 0.85,
        "developed": 0.85,
        "uses": 0.8,

        # Lower confidence - somewhat generic
        "collaborates_with": 0.7,
        "supported": 0.7,
        "member_of": 0.75,

        # Default for unknown types
        "_default": 0.6,
    }

    def __init__(self):
        """Initialize the normalizer"""
        # Build reverse mapping: English -> Arabic
        self.english_to_arabic: Dict[str, str] = {
            v: k for k, v in self.ARABIC_TO_ENGLISH.items()
        }

        # Build category lookup: type -> category
        self.type_to_category: Dict[str, str] = {}
        for category, types in self.RELATIONSHIP_CATEGORIES.items():
            for rel_type in types:
                self.type_to_category[rel_type] = category

        logger.info(
            f"RelationshipTypeNormalizer initialized: "
            f"{len(self.ARABIC_TO_ENGLISH)} Arabic types, "
            f"{len(self.ENGLISH_STANDARDIZATION)} English variations, "
            f"{len(self.RELATIONSHIP_CATEGORIES)} categories"
        )

    def normalize(self, relationship_type: str) -> NormalizedRelationship:
        """
        Normalize a relationship type to standard form.

        Args:
            relationship_type: Original relationship type (Arabic or English)

        Returns:
            NormalizedRelationship with all metadata
        """
        original = relationship_type.strip()

        # Clean up the type: lowercase, replace spaces/hyphens with underscores
        cleaned = original.lower().replace(" ", "_").replace("-", "_")

        # Check if it's Arabic
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in original)

        if is_arabic:
            # Arabic -> English mapping
            english_type = self.ARABIC_TO_ENGLISH.get(cleaned, cleaned)
            arabic_type = original
        else:
            # English standardization
            english_type = self.ENGLISH_STANDARDIZATION.get(cleaned, cleaned)
            arabic_type = self.english_to_arabic.get(english_type, "")

        # Further standardize the English type
        normalized_type = self.ENGLISH_STANDARDIZATION.get(english_type, english_type)

        # Get category
        category = self.type_to_category.get(normalized_type, "other")

        # Calculate confidence modifier
        confidence = self.CONFIDENCE_MODIFIERS.get(
            normalized_type,
            self.CONFIDENCE_MODIFIERS["_default"]
        )

        # Check if bidirectional
        bidirectional = normalized_type in self.BIDIRECTIONAL_TYPES

        return NormalizedRelationship(
            original_type=original,
            normalized_type=normalized_type,
            english_type=english_type,
            arabic_type=arabic_type,
            category=category,
            confidence=confidence,
            bidirectional=bidirectional
        )

    def to_cypher_type(self, relationship_type: str) -> str:
        """
        Convert relationship type to valid Cypher relationship type.

        Neo4j relationship types must be uppercase with underscores.

        Args:
            relationship_type: Original type

        Returns:
            Valid Cypher relationship type (e.g., "FOUNDED", "WORKS_AT")
        """
        normalized = self.normalize(relationship_type)

        # Convert to uppercase for Cypher
        cypher_type = normalized.normalized_type.upper()

        # Ensure it's a valid identifier (letters, numbers, underscores)
        import re
        cypher_type = re.sub(r'[^A-Z0-9_]', '_', cypher_type)

        # Remove consecutive underscores
        cypher_type = re.sub(r'_+', '_', cypher_type)

        # Remove leading/trailing underscores
        cypher_type = cypher_type.strip('_')

        # Fallback for empty result
        if not cypher_type:
            cypher_type = "RELATED_TO"

        return cypher_type

    def get_category(self, relationship_type: str) -> str:
        """
        Get the semantic category for a relationship type.

        Args:
            relationship_type: Relationship type

        Returns:
            Category name (e.g., "leadership", "creation", "location")
        """
        normalized = self.normalize(relationship_type)
        return normalized.category

    def get_confidence_modifier(self, relationship_type: str) -> float:
        """
        Get confidence modifier for a relationship type.

        More specific relationships get higher confidence.

        Args:
            relationship_type: Relationship type

        Returns:
            Confidence modifier (0.0-1.0)
        """
        normalized = self.normalize(relationship_type)
        return normalized.confidence

    def is_bidirectional(self, relationship_type: str) -> bool:
        """
        Check if a relationship type is bidirectional (symmetric).

        Args:
            relationship_type: Relationship type

        Returns:
            True if relationship is bidirectional
        """
        normalized = self.normalize(relationship_type)
        return normalized.bidirectional

    def translate_type(
        self,
        relationship_type: str,
        to_language: str = "en"
    ) -> str:
        """
        Translate relationship type to specified language.

        Args:
            relationship_type: Original type
            to_language: Target language ("en" or "ar")

        Returns:
            Translated type
        """
        normalized = self.normalize(relationship_type)

        if to_language == "ar":
            return normalized.arabic_type or normalized.normalized_type
        else:
            return normalized.english_type or normalized.normalized_type

    def get_related_types(self, relationship_type: str) -> List[str]:
        """
        Get semantically related relationship types.

        Args:
            relationship_type: Relationship type

        Returns:
            List of related types in the same category
        """
        category = self.get_category(relationship_type)
        return self.RELATIONSHIP_CATEGORIES.get(category, [])

    def get_stats(self) -> Dict[str, int]:
        """Get normalizer statistics"""
        return {
            "arabic_types": len(self.ARABIC_TO_ENGLISH),
            "english_variations": len(self.ENGLISH_STANDARDIZATION),
            "categories": len(self.RELATIONSHIP_CATEGORIES),
            "bidirectional_types": len(self.BIDIRECTIONAL_TYPES),
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_relationship_normalizer: Optional[RelationshipTypeNormalizer] = None


def get_relationship_normalizer() -> RelationshipTypeNormalizer:
    """Get or create global RelationshipTypeNormalizer instance"""
    global _relationship_normalizer

    if _relationship_normalizer is None:
        _relationship_normalizer = RelationshipTypeNormalizer()

    return _relationship_normalizer


def normalize_relationship_type(relationship_type: str) -> str:
    """Convenience function to normalize a relationship type"""
    return get_relationship_normalizer().normalize(relationship_type).normalized_type


def to_cypher_relationship_type(relationship_type: str) -> str:
    """Convenience function to get Cypher-compatible relationship type"""
    return get_relationship_normalizer().to_cypher_type(relationship_type)
