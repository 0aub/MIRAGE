"""
MIRAGE V2 Arabic Processing - Stanza Processor
Uses Stanford Stanza for Arabic NLP.

Stanza: https://stanfordnlp.github.io/stanza/
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


class StanzaProcessor(BaseArabicProcessor):
    """
    Arabic text processor using Stanford Stanza.

    Features:
    - Tokenization with multi-word token expansion
    - POS tagging and lemmatization
    - Named Entity Recognition
    - Dependency parsing (optional)

    Requires: pip install stanza
    """

    # NER label mapping from Stanza to our types
    NER_TYPE_MAP = {
        "PER": EntityType.PERSON,
        "PERSON": EntityType.PERSON,
        "ORG": EntityType.ORGANIZATION,
        "LOC": EntityType.LOCATION,
        "GPE": EntityType.LOCATION,  # Geopolitical entity
        "DATE": EntityType.DATE,
        "TIME": EntityType.TIME,
        "MONEY": EntityType.MONEY,
        "MISC": EntityType.MISC,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Stanza processor.

        Args:
            config: Configuration dict with options:
                - language: str (default "ar")
                - processors: str (default "tokenize,mwt,pos,lemma,ner")
                - normalization.*: normalization options
        """
        super().__init__(config)

        # Configuration
        self.language = self.config.get("language", "ar")
        self.processors = self.config.get("processors", "tokenize,mwt,pos,lemma,ner")

        # Normalization config
        norm_config = self.config.get("normalization", {})
        self.normalizer = ArabicNormalizer(
            normalize_alef=norm_config.get("alef", True),
            normalize_yaa=norm_config.get("yaa", True),
            normalize_taa_marbuta=norm_config.get("taa_marbuta", False),
            remove_diacritics=norm_config.get("remove_diacritics", True),
            remove_tatweel=norm_config.get("remove_tatweel", True),
        )

        # Stanza pipeline (lazy loaded)
        self._pipeline = None

    def initialize(self) -> None:
        """Initialize Stanza pipeline"""
        logger.info(f"Initializing Stanza processor for {self.language}...")

        try:
            import stanza

            # Download model if not present
            try:
                stanza.download(self.language, processors=self.processors, verbose=False)
            except Exception as e:
                logger.debug(f"Model download check: {e}")

            # Initialize pipeline
            self._pipeline = stanza.Pipeline(
                lang=self.language,
                processors=self.processors,
                verbose=False,
                use_gpu=False,  # CPU for consistency
            )

            self._initialized = True
            logger.info(f"Stanza processor initialized: lang={self.language}, processors={self.processors}")

        except ImportError:
            logger.error("Stanza not installed. Install with: pip install stanza")
            raise

    def process(self, text: str) -> ProcessedText:
        """
        Process Arabic text with Stanza pipeline.

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

        # Process with Stanza if Arabic
        tokens = []
        entities = []

        if has_arabic and self._pipeline:
            try:
                doc = self._pipeline(text)

                # Extract tokens
                tokens = self._extract_tokens(doc)

                # Extract entities
                if "ner" in self.processors:
                    entities = self._extract_entities(doc)

            except Exception as e:
                logger.warning(f"Stanza processing failed: {e}")

        return ProcessedText(
            original_text=text,
            normalized_text=normalized,
            tokens=tokens,
            entities=entities,
            dialect=ArabicDialect.MSA,  # Stanza doesn't do dialect detection
            is_arabic=has_arabic,
            language_confidence=1.0 if has_arabic else 0.0,
            metadata={
                "processor": "stanza",
                "language": self.language,
                "processors": self.processors,
            }
        )

    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return self.normalizer.normalize(text)

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize text using Stanza.

        Args:
            text: Input text

        Returns:
            List of Token objects
        """
        self.ensure_initialized()

        if not self._pipeline:
            return []

        try:
            doc = self._pipeline(text)
            return self._extract_tokens(doc)
        except Exception as e:
            logger.warning(f"Stanza tokenization failed: {e}")
            return []

    def _extract_tokens(self, doc) -> List[Token]:
        """Extract tokens from Stanza document"""
        tokens = []

        for sentence in doc.sentences:
            for word in sentence.words:
                token = Token(
                    text=word.text,
                    lemma=word.lemma,
                    pos=word.upos,  # Universal POS tag
                    is_arabic=self.is_arabic(word.text),
                    start_char=word.start_char if hasattr(word, 'start_char') else 0,
                    end_char=word.end_char if hasattr(word, 'end_char') else 0,
                )
                tokens.append(token)

        return tokens

    def extract_entities(self, text: str) -> List[NamedEntity]:
        """
        Extract named entities using Stanza NER.

        Args:
            text: Input text

        Returns:
            List of NamedEntity objects
        """
        self.ensure_initialized()

        if not self._pipeline or "ner" not in self.processors:
            return []

        try:
            doc = self._pipeline(text)
            return self._extract_entities(doc)
        except Exception as e:
            logger.warning(f"Stanza NER failed: {e}")
            return []

    def _extract_entities(self, doc) -> List[NamedEntity]:
        """Extract entities from Stanza document"""
        entities = []

        for sentence in doc.sentences:
            for ent in sentence.ents:
                entity_type = self.NER_TYPE_MAP.get(ent.type, EntityType.MISC)

                entity = NamedEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_char=ent.start_char if hasattr(ent, 'start_char') else 0,
                    end_char=ent.end_char if hasattr(ent, 'end_char') else 0,
                )
                entities.append(entity)

        return entities

    def detect_dialect(self, text: str) -> Tuple[ArabicDialect, float]:
        """
        Stanza doesn't support dialect detection.
        Returns MSA with medium confidence as default.

        Args:
            text: Input text

        Returns:
            Tuple of (MSA, 0.6)
        """
        return ArabicDialect.MSA, 0.6

    def close(self) -> None:
        """Release Stanza resources"""
        self._pipeline = None
        super().close()
