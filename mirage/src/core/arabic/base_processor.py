"""
MIRAGE V2 Arabic Processing - Base Classes
Defines the interface for all Arabic text processors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class ArabicDialect(Enum):
    """Arabic dialect types"""
    MSA = "MSA"      # Modern Standard Arabic
    EGY = "EGY"      # Egyptian
    GLF = "GLF"      # Gulf
    LEV = "LEV"      # Levantine
    MGR = "MGR"      # Maghrebi
    UNKNOWN = "UNKNOWN"


class EntityType(Enum):
    """Named entity types"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    MISC = "MISC"


@dataclass
class Token:
    """Tokenized word with linguistic annotations"""
    text: str
    lemma: Optional[str] = None
    pos: Optional[str] = None  # Part-of-speech tag
    stem: Optional[str] = None
    is_arabic: bool = True
    start_char: int = 0
    end_char: int = 0


@dataclass
class NamedEntity:
    """Named entity with type and confidence"""
    text: str
    entity_type: EntityType
    start_char: int
    end_char: int
    confidence: float = 1.0
    normalized_text: Optional[str] = None  # After normalization


@dataclass
class ProcessedText:
    """Result of Arabic text processing"""
    original_text: str
    normalized_text: str
    tokens: List[Token] = field(default_factory=list)
    entities: List[NamedEntity] = field(default_factory=list)
    dialect: ArabicDialect = ArabicDialect.UNKNOWN
    is_arabic: bool = True
    language_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseArabicProcessor(ABC):
    """
    Abstract base class for Arabic text processors.

    All Arabic processors (CAMeL, Stanza, Disabled) must implement this interface.
    This enables easy swapping of processors via configuration for A/B testing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor with configuration.

        Args:
            config: Processor-specific configuration
        """
        self.config = config or {}
        self._initialized = False

    @property
    def name(self) -> str:
        """Processor name for logging and identification"""
        return self.__class__.__name__

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize processor resources (models, etc.).
        Called lazily on first use.
        """
        pass

    def ensure_initialized(self) -> None:
        """Ensure processor is initialized before use"""
        if not self._initialized:
            self.initialize()
            self._initialized = True

    @abstractmethod
    def process(self, text: str) -> ProcessedText:
        """
        Process Arabic text with full pipeline.

        Args:
            text: Input text (Arabic or mixed)

        Returns:
            ProcessedText with all annotations
        """
        pass

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text.

        Normalization includes:
        - Alef normalization (أ إ آ → ا)
        - Yaa normalization (ى → ي)
        - Remove diacritics (tashkeel)
        - Remove tatweel (ـــ)

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize text into words with annotations.

        Args:
            text: Input text

        Returns:
            List of Token objects
        """
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[NamedEntity]:
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            List of NamedEntity objects
        """
        pass

    def detect_dialect(self, text: str) -> Tuple[ArabicDialect, float]:
        """
        Detect Arabic dialect.

        Args:
            text: Input text

        Returns:
            Tuple of (dialect, confidence)
        """
        # Default implementation returns MSA with low confidence
        return ArabicDialect.MSA, 0.5

    def is_arabic(self, text: str) -> bool:
        """
        Check if text contains Arabic characters.

        Args:
            text: Input text

        Returns:
            True if text contains Arabic
        """
        arabic_ranges = [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ]

        for char in text:
            code = ord(char)
            for start, end in arabic_ranges:
                if start <= code <= end:
                    return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "config": self.config
        }

    def close(self) -> None:
        """Release processor resources"""
        self._initialized = False


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================

class ArabicNormalizer:
    """
    Rule-based Arabic text normalizer.
    Used by all processors as a common normalization layer.
    """

    # Character mappings for normalization
    ALEF_VARIANTS = {
        '\u0622': '\u0627',  # آ → ا (Alef with Madda)
        '\u0623': '\u0627',  # أ → ا (Alef with Hamza Above)
        '\u0625': '\u0627',  # إ → ا (Alef with Hamza Below)
        '\u0671': '\u0627',  # ٱ → ا (Alef Wasla)
    }

    YAA_VARIANTS = {
        '\u0649': '\u064A',  # ى → ي (Alef Maksura → Yaa)
    }

    TAA_MARBUTA = {
        '\u0629': '\u0647',  # ة → ه (Taa Marbuta → Haa)
    }

    # Diacritics (tashkeel) to remove
    DIACRITICS = [
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0653',  # Maddah
        '\u0654',  # Hamza Above
        '\u0655',  # Hamza Below
        '\u0656',  # Subscript Alef
        '\u0670',  # Superscript Alef
    ]

    # Tatweel (kashida)
    TATWEEL = '\u0640'  # ـ

    def __init__(
        self,
        normalize_alef: bool = True,
        normalize_yaa: bool = True,
        normalize_taa_marbuta: bool = False,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        remove_non_arabic: bool = False
    ):
        """
        Initialize normalizer with options.

        Args:
            normalize_alef: Normalize Alef variants to bare Alef
            normalize_yaa: Normalize Alef Maksura to Yaa
            normalize_taa_marbuta: Normalize Taa Marbuta to Haa (usually disabled)
            remove_diacritics: Remove tashkeel marks
            remove_tatweel: Remove tatweel/kashida
            remove_non_arabic: Remove non-Arabic characters (aggressive)
        """
        self.normalize_alef = normalize_alef
        self.normalize_yaa = normalize_yaa
        self.normalize_taa_marbuta = normalize_taa_marbuta
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.remove_non_arabic = remove_non_arabic

        # Build translation table
        self._build_translation_table()

    def _build_translation_table(self) -> None:
        """Build character translation table"""
        trans_dict = {}

        if self.normalize_alef:
            trans_dict.update(self.ALEF_VARIANTS)

        if self.normalize_yaa:
            trans_dict.update(self.YAA_VARIANTS)

        if self.normalize_taa_marbuta:
            trans_dict.update(self.TAA_MARBUTA)

        if self.remove_diacritics:
            for d in self.DIACRITICS:
                trans_dict[d] = ''

        if self.remove_tatweel:
            trans_dict[self.TATWEEL] = ''

        self._trans_table = str.maketrans(trans_dict)

    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Apply character translations
        text = text.translate(self._trans_table)

        # Remove non-Arabic if requested
        if self.remove_non_arabic:
            text = self._keep_arabic_only(text)

        return text

    def _keep_arabic_only(self, text: str) -> str:
        """Keep only Arabic characters and spaces"""
        result = []
        for char in text:
            if self._is_arabic_char(char) or char.isspace():
                result.append(char)
        return ''.join(result)

    def _is_arabic_char(self, char: str) -> bool:
        """Check if character is Arabic"""
        code = ord(char)
        return (
            0x0600 <= code <= 0x06FF or  # Arabic
            0x0750 <= code <= 0x077F or  # Arabic Supplement
            0x08A0 <= code <= 0x08FF or  # Arabic Extended-A
            0xFB50 <= code <= 0xFDFF or  # Arabic Presentation Forms-A
            0xFE70 <= code <= 0xFEFF     # Arabic Presentation Forms-B
        )


# =============================================================================
# ENTITY NORMALIZER
# =============================================================================

class EntityNormalizer:
    """
    Normalize extracted entities for better matching.
    Removes titles, suffixes, and applies Arabic normalization.
    """

    # Common Arabic titles to remove
    DEFAULT_TITLES = [
        "الدكتور", "دكتور", "د.",
        "الأستاذ", "أستاذ",
        "المهندس", "مهندس", "م.",
        "الشيخ", "شيخ",
        "السيد", "سيد",
        "الحاج", "حاج",
        "الأمير", "أمير",
        "الملك", "ملك",
        "الرئيس", "رئيس",
        "الوزير", "وزير",
        "السفير", "سفير",
    ]

    # Common organization suffixes
    DEFAULT_ORG_SUFFIXES = [
        "المحدودة", "محدودة",
        "للتجارة", "التجارية",
        "القابضة",
        "المساهمة",
        "الخاصة",
        "العامة",
    ]

    def __init__(
        self,
        titles: Optional[List[str]] = None,
        org_suffixes: Optional[List[str]] = None,
        normalizer: Optional[ArabicNormalizer] = None
    ):
        """
        Initialize entity normalizer.

        Args:
            titles: List of titles to remove (uses defaults if None)
            org_suffixes: List of organization suffixes to remove
            normalizer: Arabic normalizer instance
        """
        self.titles = titles or self.DEFAULT_TITLES
        self.org_suffixes = org_suffixes or self.DEFAULT_ORG_SUFFIXES
        self.normalizer = normalizer or ArabicNormalizer()

    def normalize_entity(
        self,
        entity: str,
        entity_type: Optional[EntityType] = None
    ) -> str:
        """
        Normalize an entity string.

        Args:
            entity: Entity text
            entity_type: Entity type (for type-specific normalization)

        Returns:
            Normalized entity
        """
        # Apply Arabic normalization
        normalized = self.normalizer.normalize(entity)

        # Remove titles for PERSON entities
        if entity_type in (EntityType.PERSON, None):
            normalized = self._remove_titles(normalized)

        # Remove suffixes for ORGANIZATION entities
        if entity_type in (EntityType.ORGANIZATION, None):
            normalized = self._remove_org_suffixes(normalized)

        # Clean up whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def _remove_titles(self, text: str) -> str:
        """Remove titles from beginning of text"""
        for title in self.titles:
            if text.startswith(title + ' '):
                text = text[len(title):].strip()
        return text

    def _remove_org_suffixes(self, text: str) -> str:
        """Remove organization suffixes from end of text"""
        for suffix in self.org_suffixes:
            if text.endswith(' ' + suffix):
                text = text[:-len(suffix)].strip()
        return text
