"""
MIRAGE V2 Arabic Processing - CAMeL Tools Processor
Uses CAMeL Tools for Arabic NLP: morphological analysis, NER, dialect detection.

CAMeL Tools: https://github.com/CAMeL-Lab/camel_tools
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from .base_processor import (
    BaseArabicProcessor,
    ProcessedText,
    Token,
    NamedEntity,
    EntityType,
    ArabicDialect,
    ArabicNormalizer,
)


class CAMeLProcessor(BaseArabicProcessor):
    """
    Arabic text processor using CAMeL Tools.

    Features:
    - Morphological analysis (tokenization, POS, lemmatization)
    - Named Entity Recognition
    - Dialect identification
    - Arabic text normalization

    Requires: pip install camel-tools
    """

    # Dialect mapping from CAMeL to our enum
    DIALECT_MAP = {
        "MSA": ArabicDialect.MSA,
        "EGY": ArabicDialect.EGY,
        "GLF": ArabicDialect.GLF,
        "LEV": ArabicDialect.LEV,
        "MGR": ArabicDialect.MGR,
    }

    # NER label mapping
    NER_TYPE_MAP = {
        "B-PER": EntityType.PERSON,
        "I-PER": EntityType.PERSON,
        "B-ORG": EntityType.ORGANIZATION,
        "I-ORG": EntityType.ORGANIZATION,
        "B-LOC": EntityType.LOCATION,
        "I-LOC": EntityType.LOCATION,
        "B-MISC": EntityType.MISC,
        "I-MISC": EntityType.MISC,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CAMeL processor.

        Args:
            config: Configuration dict with options:
                - morphological_analysis: bool (default True)
                - diacritization: bool (default False)
                - dialect_identification: bool (default True)
                - ner.enabled: bool (default True)
                - normalization.*: normalization options
        """
        super().__init__(config)

        # Configuration defaults
        self.morph_enabled = self.config.get("morphological_analysis", True)
        self.dialect_enabled = self.config.get("dialect_identification", True)
        self.ner_enabled = self.config.get("ner", {}).get("enabled", True)

        # Normalization config
        norm_config = self.config.get("normalization", {})
        self.normalizer = ArabicNormalizer(
            normalize_alef=norm_config.get("alef", True),
            normalize_yaa=norm_config.get("yaa", True),
            normalize_taa_marbuta=norm_config.get("taa_marbuta", False),
            remove_diacritics=norm_config.get("remove_diacritics", True),
            remove_tatweel=norm_config.get("remove_tatweel", True),
        )

        # CAMeL components (lazy loaded)
        self._morphological_analyzer = None
        self._disambiguator = None
        self._ner_tagger = None
        self._dialect_identifier = None
        self._simple_tokenizer = None

    def initialize(self) -> None:
        """Initialize CAMeL Tools components"""
        logger.info("Initializing CAMeL Tools processor...")

        try:
            # Import CAMeL Tools
            from camel_tools.tokenizers.word import simple_word_tokenize
            from camel_tools.utils.normalize import normalize_unicode

            self._simple_tokenizer = simple_word_tokenize
            self._normalize_unicode = normalize_unicode

            # Load morphological analyzer
            if self.morph_enabled:
                try:
                    from camel_tools.morphology.database import MorphologyDB
                    from camel_tools.morphology.analyzer import Analyzer
                    from camel_tools.disambig.mle import MLEDisambiguator

                    db = MorphologyDB.builtin_db()
                    self._morphological_analyzer = Analyzer(db)
                    self._disambiguator = MLEDisambiguator.pretrained()
                    logger.info("CAMeL morphological analyzer loaded")
                except Exception as e:
                    logger.warning(f"Failed to load CAMeL morphological analyzer: {e}")
                    self.morph_enabled = False

            # Load NER tagger
            if self.ner_enabled:
                try:
                    from camel_tools.ner import NERecognizer
                    self._ner_tagger = NERecognizer.pretrained()
                    logger.info("CAMeL NER tagger loaded")
                except Exception as e:
                    logger.warning(f"Failed to load CAMeL NER: {e}")
                    self.ner_enabled = False

            # Load dialect identifier
            if self.dialect_enabled:
                try:
                    from camel_tools.dialectid import DialectIdentifier
                    self._dialect_identifier = DialectIdentifier.pretrained()
                    logger.info("CAMeL dialect identifier loaded")
                except Exception as e:
                    logger.warning(f"Failed to load CAMeL dialect identifier: {e}")
                    self.dialect_enabled = False

            self._initialized = True
            logger.info("CAMeL Tools processor initialized successfully")

        except ImportError as e:
            logger.error(f"CAMeL Tools not installed: {e}")
            logger.error("Install with: pip install camel-tools")
            raise

    def process(self, text: str) -> ProcessedText:
        """
        Process Arabic text with full CAMeL pipeline.

        Args:
            text: Input text

        Returns:
            ProcessedText with all annotations
        """
        self.ensure_initialized()

        # Check if text contains Arabic
        has_arabic = self.is_arabic(text)

        # Normalize text
        normalized = self.normalize(text)

        # Tokenize
        tokens = self.tokenize(text) if has_arabic else []

        # Extract entities
        entities = self.extract_entities(text) if has_arabic and self.ner_enabled else []

        # Detect dialect
        dialect = ArabicDialect.UNKNOWN
        dialect_conf = 0.0
        if has_arabic and self.dialect_enabled:
            dialect, dialect_conf = self.detect_dialect(text)

        return ProcessedText(
            original_text=text,
            normalized_text=normalized,
            tokens=tokens,
            entities=entities,
            dialect=dialect,
            is_arabic=has_arabic,
            language_confidence=dialect_conf if has_arabic else 0.0,
            metadata={
                "processor": "camel",
                "morph_enabled": self.morph_enabled,
                "ner_enabled": self.ner_enabled,
                "dialect_enabled": self.dialect_enabled,
            }
        )

    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text using CAMeL and custom rules.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        self.ensure_initialized()

        # First apply CAMeL unicode normalization
        if self._normalize_unicode:
            text = self._normalize_unicode(text)

        # Then apply our normalization rules
        return self.normalizer.normalize(text)

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize text with morphological analysis.

        Args:
            text: Input text

        Returns:
            List of Token objects with POS and lemma
        """
        self.ensure_initialized()

        if not self._simple_tokenizer:
            return []

        # Simple tokenization
        words = self._simple_tokenizer(text)
        tokens = []

        # Add morphological analysis if available
        if self._disambiguator and self.morph_enabled:
            try:
                disambiguated = self._disambiguator.disambiguate(words)

                for i, (word, analysis) in enumerate(zip(words, disambiguated)):
                    # Get best analysis
                    if analysis.analyses:
                        best = analysis.analyses[0]
                        token = Token(
                            text=word,
                            lemma=best.get("lex", word),
                            pos=best.get("pos", None),
                            stem=best.get("stem", None),
                            is_arabic=self.is_arabic(word),
                        )
                    else:
                        token = Token(
                            text=word,
                            is_arabic=self.is_arabic(word),
                        )
                    tokens.append(token)

            except Exception as e:
                logger.warning(f"Morphological analysis failed: {e}")
                # Fallback to simple tokens
                for word in words:
                    tokens.append(Token(text=word, is_arabic=self.is_arabic(word)))
        else:
            # Simple tokenization only
            for word in words:
                tokens.append(Token(text=word, is_arabic=self.is_arabic(word)))

        return tokens

    def extract_entities(self, text: str) -> List[NamedEntity]:
        """
        Extract named entities using CAMeL NER.

        Args:
            text: Input text

        Returns:
            List of NamedEntity objects
        """
        self.ensure_initialized()

        if not self._ner_tagger or not self.ner_enabled:
            return []

        try:
            # Tokenize for NER
            words = self._simple_tokenizer(text)

            # Run NER
            labels = self._ner_tagger.predict_sentence(words)

            # Convert to entities
            entities = []
            current_entity = None
            current_type = None
            char_offset = 0

            for word, label in zip(words, labels):
                # Find word position in original text
                try:
                    start = text.index(word, char_offset)
                    end = start + len(word)
                    char_offset = end
                except ValueError:
                    start = end = char_offset

                if label.startswith("B-"):
                    # Start of new entity
                    if current_entity:
                        entities.append(current_entity)

                    entity_type = self.NER_TYPE_MAP.get(label, EntityType.MISC)
                    current_entity = NamedEntity(
                        text=word,
                        entity_type=entity_type,
                        start_char=start,
                        end_char=end,
                    )
                    current_type = label[2:]

                elif label.startswith("I-") and current_entity:
                    # Continuation of entity
                    if label[2:] == current_type:
                        current_entity.text += " " + word
                        current_entity.end_char = end

                else:
                    # Outside entity
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                        current_type = None

            # Don't forget last entity
            if current_entity:
                entities.append(current_entity)

            return entities

        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            return []

    def detect_dialect(self, text: str) -> Tuple[ArabicDialect, float]:
        """
        Detect Arabic dialect using CAMeL.

        Args:
            text: Input text

        Returns:
            Tuple of (dialect, confidence)
        """
        self.ensure_initialized()

        if not self._dialect_identifier or not self.dialect_enabled:
            return ArabicDialect.UNKNOWN, 0.0

        try:
            # Get dialect predictions
            predictions = self._dialect_identifier.predict([text])

            if predictions:
                pred = predictions[0]
                # CAMeL returns a dict with dialect scores
                if isinstance(pred, dict):
                    best_dialect = max(pred.items(), key=lambda x: x[1])
                    dialect = self.DIALECT_MAP.get(best_dialect[0], ArabicDialect.UNKNOWN)
                    return dialect, best_dialect[1]
                else:
                    # Simple label
                    dialect = self.DIALECT_MAP.get(str(pred), ArabicDialect.UNKNOWN)
                    return dialect, 0.8

        except Exception as e:
            logger.warning(f"Dialect detection failed: {e}")

        return ArabicDialect.UNKNOWN, 0.0

    def close(self) -> None:
        """Release CAMeL resources"""
        self._morphological_analyzer = None
        self._disambiguator = None
        self._ner_tagger = None
        self._dialect_identifier = None
        self._simple_tokenizer = None
        super().close()
