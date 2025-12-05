"""
Ensemble Entity Extractor - Phase 2 GraphRAG Enhancement

Combines multiple extraction methods for higher quality:
1. Rule-based: CAMeL Tools NER for Arabic, spaCy for English
2. Pattern-based: Regex patterns for common entity types
3. LLM-based: Allam/TGI for context-aware extraction

Confidence calibration based on:
- Agreement between extractors (boosted if multiple agree)
- Entity frequency in document
- Extraction source reliability
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

# Required imports - no fallback (enforced in Docker)
from camel_tools.ner import NERecognizer
from camel_tools.tokenizers.word import simple_word_tokenize
import spacy


@dataclass
class ExtractedEntity:
    """Entity with confidence and source tracking."""
    text: str
    type: str
    confidence: float
    sources: List[str] = field(default_factory=list)  # Which extractors found it
    frequency: int = 1  # How many times found in document
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "confidence": self.confidence,
            "sources": self.sources,
            "frequency": self.frequency,
            "attributes": self.attributes
        }


class EnsembleEntityExtractor:
    """
    Ensemble extractor combining rule-based, pattern-based, and LLM methods.

    This implements the hybrid approach recommended for smaller LLMs:
    - 60% rule-based (CAMeL/spaCy) for reliable base extraction
    - 20% pattern-based for domain-specific entities
    - 20% LLM validation and relationship extraction

    Benefits:
    - Higher recall from rule-based methods
    - Higher precision from LLM validation
    - Confidence calibration from agreement
    """

    # Arabic entity patterns
    ARABIC_PATTERNS = {
        "Organization": [
            r"وزارة\s+[\u0600-\u06FF\s]+",           # Ministry of...
            r"هيئة\s+[\u0600-\u06FF\s]+",            # Authority of...
            r"مؤسسة\s+[\u0600-\u06FF\s]+",           # Foundation of...
            r"شركة\s+[\u0600-\u06FF\s]+",            # Company of...
            r"جامعة\s+[\u0600-\u06FF\s]+",           # University of...
            r"بنك\s+[\u0600-\u06FF\s]+",             # Bank of...
            r"مركز\s+[\u0600-\u06FF\s]+",            # Center of...
            r"معهد\s+[\u0600-\u06FF\s]+",            # Institute of...
            r"المركز\s+[\u0600-\u06FF]+\s+[\u0600-\u06FF]+",  # The X Center
            r"الهيئة\s+[\u0600-\u06FF]+\s+[\u0600-\u06FF]+",  # The X Authority
        ],
        "Program": [
            r"برنامج\s+[\u0600-\u06FF\s]+",          # Program...
            r"مبادرة\s+[\u0600-\u06FF\s]+",          # Initiative...
            r"مشروع\s+[\u0600-\u06FF\s]+",           # Project...
            r"خطة\s+[\u0600-\u06FF\s]+",             # Plan...
            r"استراتيجية\s+[\u0600-\u06FF\s]+",      # Strategy...
            r"رؤية\s+\d{4}",                          # Vision 2030, etc.
        ],
        "Award": [
            r"جائزة\s+[\u0600-\u06FF\s]+",           # Award...
            r"وسام\s+[\u0600-\u06FF\s]+",            # Medal...
            r"تكريم\s+[\u0600-\u06FF\s]+",           # Honor...
        ],
        "Event": [
            r"مؤتمر\s+[\u0600-\u06FF\s]+",           # Conference...
            r"قمة\s+[\u0600-\u06FF\s]+",             # Summit...
            r"منتدى\s+[\u0600-\u06FF\s]+",           # Forum...
            r"ملتقى\s+[\u0600-\u06FF\s]+",           # Meeting...
            r"معرض\s+[\u0600-\u06FF\s]+",            # Exhibition...
        ],
        "Location": [
            r"مدينة\s+[\u0600-\u06FF]+",             # City of...
            r"منطقة\s+[\u0600-\u06FF]+",             # Region of...
            r"محافظة\s+[\u0600-\u06FF]+",            # Province of...
        ],
        "Technology": [
            r"الذكاء\s+الاصطناعي",                   # Artificial Intelligence
            r"التعلم\s+الآلي",                       # Machine Learning
            r"البيانات\s+الضخمة",                    # Big Data
            r"الحوسبة\s+السحابية",                   # Cloud Computing
            r"سلسلة\s+الكتل",                        # Blockchain
            r"إنترنت\s+الأشياء",                     # IoT
        ],
        "Person": [
            r"الأمير\s+[\u0600-\u06FF\s]+",          # Prince...
            r"الشيخ\s+[\u0600-\u06FF\s]+",           # Sheikh...
            r"الدكتور\s+[\u0600-\u06FF\s]+",         # Doctor...
            r"المهندس\s+[\u0600-\u06FF\s]+",         # Engineer...
            r"الأستاذ\s+[\u0600-\u06FF\s]+",         # Professor...
            r"سمو\s+[\u0600-\u06FF\s]+",             # Highness...
            r"معالي\s+[\u0600-\u06FF\s]+",           # Excellency...
        ],
    }

    # English entity patterns
    ENGLISH_PATTERNS = {
        "Organization": [
            r"Ministry of \w+",
            r"Department of \w+",
            r"\w+ Authority",
            r"\w+ Corporation",
            r"\w+ Foundation",
            r"\w+ Institute",
            r"\w+ University",
            r"\w+ Center",
            r"\w+ Bank",
        ],
        "Program": [
            r"\w+ Program",
            r"\w+ Initiative",
            r"\w+ Project",
            r"\w+ Strategy",
            r"Vision \d{4}",
        ],
        "Event": [
            r"\w+ Conference",
            r"\w+ Summit",
            r"\w+ Forum",
            r"\w+ Exhibition",
        ],
        "Technology": [
            r"Artificial Intelligence|AI",
            r"Machine Learning|ML",
            r"Big Data",
            r"Cloud Computing",
            r"Blockchain",
            r"Internet of Things|IoT",
            r"Digital Transformation",
        ],
    }

    def __init__(
        self,
        use_camel: bool = True,
        use_spacy: bool = True,
        use_patterns: bool = True,
        use_llm: bool = True,
        llm_extractor=None,
        confidence_threshold: float = 0.4
    ):
        """
        Initialize ensemble extractor.

        Args:
            use_camel: Enable CAMeL Tools NER
            use_spacy: Enable spaCy NER
            use_patterns: Enable pattern-based extraction
            use_llm: Enable LLM-based extraction
            llm_extractor: Optional pre-initialized LLM extractor
            confidence_threshold: Minimum confidence to keep entity
        """
        self.use_camel = use_camel  # CAMeL is required
        self.use_spacy = use_spacy  # spaCy is required
        self.use_patterns = use_patterns
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold

        # Initialize extractors
        self.camel_ner = None
        self.spacy_model = None
        self.llm_extractor = llm_extractor

        # Compile patterns
        self._arabic_patterns = self._compile_patterns(self.ARABIC_PATTERNS)
        self._english_patterns = self._compile_patterns(self.ENGLISH_PATTERNS)

        # Entity type mapping for CAMeL
        self.camel_type_map = {
            "per": "Person",
            "org": "Organization",
            "loc": "Location",
            "misc": "Miscellaneous",
        }

        # Confidence weights for each extractor source
        self.source_weights = {
            "camel": 0.75,    # Good for Arabic NER
            "spacy": 0.70,    # Good for English NER
            "pattern": 0.65,  # Reliable but limited
            "llm": 0.85,      # Best quality but may hallucinate
        }

        logger.info(
            f"EnsembleExtractor initialized: "
            f"camel={self.use_camel}, spacy={self.use_spacy}, "
            f"patterns={self.use_patterns}, llm={self.use_llm}"
        )

    def _compile_patterns(
        self,
        pattern_dict: Dict[str, List[str]]
    ) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns."""
        compiled = {}
        for entity_type, patterns in pattern_dict.items():
            compiled[entity_type] = [
                re.compile(p, re.UNICODE | re.IGNORECASE)
                for p in patterns
            ]
        return compiled

    def _load_camel(self):
        """Load CAMeL NER model lazily."""
        if self.camel_ner is not None or not self.use_camel:
            return

        try:
            self.camel_ner = NERecognizer.pretrained()
            logger.info("CAMeL NER model loaded")
        except Exception as e:
            logger.error(f"Failed to load CAMeL NER: {e}")
            self.use_camel = False

    def _load_spacy(self):
        """Load spaCy model lazily."""
        if self.spacy_model is not None or not self.use_spacy:
            return

        try:
            self.spacy_model = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except OSError:
            try:
                self.spacy_model = spacy.blank("en")
                logger.warning("spaCy model not found, using blank model")
            except Exception as e:
                logger.error(f"Failed to load spaCy: {e}")
                self.use_spacy = False

    def detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English."""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        return 'ar' if arabic_chars > latin_chars else 'en'

    def extract(
        self,
        text: str,
        language: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities using ensemble approach.

        Args:
            text: Text to extract from
            language: Language code ('ar' or 'en'), auto-detected if None
            document_id: Optional document ID

        Returns:
            Dict with 'entities', 'relationships', 'extraction_stats'
        """
        if not text or not text.strip():
            return {"entities": [], "relationships": [], "extraction_stats": {}}

        language = language or self.detect_language(text)
        logger.info(f"Ensemble extraction: {len(text)} chars, language={language}")

        # Collect entities from all sources
        all_entities: Dict[str, ExtractedEntity] = {}  # key: normalized_text

        # 1. Pattern-based extraction (fast, high precision)
        if self.use_patterns:
            pattern_entities = self._extract_with_patterns(text, language)
            self._merge_entities(all_entities, pattern_entities, "pattern")
            logger.debug(f"Pattern extraction: {len(pattern_entities)} entities")

        # 2. Rule-based NER extraction
        if language == 'ar' and self.use_camel:
            self._load_camel()
            if self.camel_ner:
                camel_entities = self._extract_with_camel(text)
                self._merge_entities(all_entities, camel_entities, "camel")
                logger.debug(f"CAMeL extraction: {len(camel_entities)} entities")

        elif language == 'en' and self.use_spacy:
            self._load_spacy()
            if self.spacy_model:
                spacy_entities = self._extract_with_spacy(text)
                self._merge_entities(all_entities, spacy_entities, "spacy")
                logger.debug(f"spaCy extraction: {len(spacy_entities)} entities")

        # 3. LLM extraction (optional, for validation and relationships)
        relationships = []
        if self.use_llm and self.llm_extractor:
            try:
                llm_result = self.llm_extractor.extract_entities_and_relationships(
                    text, language, document_id
                )
                llm_entities = llm_result.get("entities", [])
                relationships = llm_result.get("relationships", [])

                # Convert to ExtractedEntity format
                llm_extracted = []
                for e in llm_entities:
                    llm_extracted.append(ExtractedEntity(
                        text=e.get("text", ""),
                        type=e.get("type", "Unknown"),
                        confidence=e.get("confidence", 0.85),
                        attributes=e.get("attributes", {})
                    ))

                self._merge_entities(all_entities, llm_extracted, "llm")
                logger.debug(f"LLM extraction: {len(llm_extracted)} entities, {len(relationships)} relationships")

            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

        # 4. Calibrate confidence based on agreement
        calibrated_entities = self._calibrate_confidence(all_entities)

        # 5. Filter by confidence threshold
        filtered_entities = [
            e for e in calibrated_entities
            if e.confidence >= self.confidence_threshold
        ]

        # Convert to output format
        entities_out = [e.to_dict() for e in filtered_entities]

        # Statistics
        stats = {
            "total_raw": len(all_entities),
            "total_filtered": len(filtered_entities),
            "by_source": self._count_by_source(filtered_entities),
            "by_type": self._count_by_type(filtered_entities),
            "avg_confidence": sum(e.confidence for e in filtered_entities) / len(filtered_entities) if filtered_entities else 0,
        }

        logger.info(
            f"Ensemble extraction complete: {len(filtered_entities)} entities, "
            f"{len(relationships)} relationships (avg conf: {stats['avg_confidence']:.2f})"
        )

        return {
            "entities": entities_out,
            "relationships": relationships,
            "extraction_stats": stats
        }

    def _extract_with_patterns(
        self,
        text: str,
        language: str
    ) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        patterns = self._arabic_patterns if language == 'ar' else self._english_patterns

        for entity_type, compiled_patterns in patterns.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    entity_text = match.group().strip()
                    # Clean up extracted text
                    entity_text = self._clean_entity_text(entity_text)

                    if len(entity_text) >= 2:  # Minimum length
                        entities.append(ExtractedEntity(
                            text=entity_text,
                            type=entity_type,
                            confidence=self.source_weights["pattern"]
                        ))

        return entities

    def _extract_with_camel(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using CAMeL Tools NER."""
        entities = []

        try:
            # Tokenize Arabic text
            sentences = text.split('.')

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Simple word tokenization
                try:
                    words = simple_word_tokenize(sentence)
                except:
                    words = sentence.split()

                if not words:
                    continue

                # Get NER predictions
                ner_result = self.camel_ner.predict_sentence(words)

                # Parse IOB tags
                current_entity = None
                current_words = []

                for word, tag in zip(words, ner_result):
                    if tag.startswith('B-'):
                        # Save previous entity
                        if current_entity:
                            entities.append(current_entity)

                        entity_type = self.camel_type_map.get(
                            tag[2:].lower(),
                            tag[2:]
                        )
                        current_entity = ExtractedEntity(
                            text=word,
                            type=entity_type,
                            confidence=self.source_weights["camel"]
                        )
                        current_words = [word]

                    elif tag.startswith('I-') and current_entity:
                        current_words.append(word)
                        current_entity.text = " ".join(current_words)

                    else:
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None
                            current_words = []

                if current_entity:
                    entities.append(current_entity)

        except Exception as e:
            logger.error(f"CAMeL extraction error: {e}")

        return entities

    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        entities = []

        try:
            doc = self.spacy_model(text)

            # spaCy type mapping
            type_map = {
                "PERSON": "Person",
                "ORG": "Organization",
                "GPE": "Location",
                "LOC": "Location",
                "DATE": "Date",
                "EVENT": "Event",
                "PRODUCT": "Product",
                "WORK_OF_ART": "WorkOfArt",
                "LAW": "Law",
            }

            for ent in doc.ents:
                entity_type = type_map.get(ent.label_, ent.label_)
                entities.append(ExtractedEntity(
                    text=ent.text,
                    type=entity_type,
                    confidence=self.source_weights["spacy"]
                ))

        except Exception as e:
            logger.error(f"spaCy extraction error: {e}")

        return entities

    def _merge_entities(
        self,
        all_entities: Dict[str, ExtractedEntity],
        new_entities: List[ExtractedEntity],
        source: str
    ):
        """
        Merge new entities into the collection.

        Key: normalized entity text
        Updates confidence and sources for existing entities.
        """
        for entity in new_entities:
            key = self._normalize_key(entity.text)

            if key in all_entities:
                # Entity already exists - merge
                existing = all_entities[key]
                if source not in existing.sources:
                    existing.sources.append(source)
                existing.frequency += 1

                # Keep higher confidence
                if entity.confidence > existing.confidence:
                    existing.confidence = entity.confidence

                # Merge attributes
                existing.attributes.update(entity.attributes)
            else:
                # New entity
                entity.sources = [source]
                all_entities[key] = entity

    def _normalize_key(self, text: str) -> str:
        """Normalize entity text for deduplication key."""
        # Lowercase
        text = text.lower().strip()

        # Remove diacritics for Arabic
        diacritics = re.compile(r'[\u064B-\u0652\u0670]')
        text = diacritics.sub('', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    def _clean_entity_text(self, text: str) -> str:
        """Clean extracted entity text."""
        # Remove leading/trailing punctuation and whitespace
        text = re.sub(r'^[\s\u060C\u061B\u061F\.,;:!?]+', '', text)
        text = re.sub(r'[\s\u060C\u061B\u061F\.,;:!?]+$', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _calibrate_confidence(
        self,
        all_entities: Dict[str, ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """
        Calibrate confidence scores based on:
        - Number of sources that found the entity
        - Entity frequency
        - Source reliability
        """
        calibrated = []

        for entity in all_entities.values():
            base_confidence = entity.confidence

            # Boost for multiple sources agreeing
            source_count = len(entity.sources)
            if source_count >= 3:
                agreement_boost = 0.15
            elif source_count >= 2:
                agreement_boost = 0.10
            else:
                agreement_boost = 0.0

            # Boost for high frequency (mentioned multiple times)
            if entity.frequency >= 3:
                frequency_boost = 0.10
            elif entity.frequency >= 2:
                frequency_boost = 0.05
            else:
                frequency_boost = 0.0

            # LLM validation boost
            llm_boost = 0.05 if "llm" in entity.sources else 0.0

            # Calculate final confidence (cap at 0.98)
            final_confidence = min(
                base_confidence + agreement_boost + frequency_boost + llm_boost,
                0.98
            )

            entity.confidence = final_confidence
            calibrated.append(entity)

        return calibrated

    def _count_by_source(
        self,
        entities: List[ExtractedEntity]
    ) -> Dict[str, int]:
        """Count entities by extraction source."""
        counts = defaultdict(int)
        for e in entities:
            for source in e.sources:
                counts[source] += 1
        return dict(counts)

    def _count_by_type(
        self,
        entities: List[ExtractedEntity]
    ) -> Dict[str, int]:
        """Count entities by type."""
        counts = defaultdict(int)
        for e in entities:
            counts[e.type] += 1
        return dict(counts)

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        language: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from document chunks.

        Args:
            chunks: List of chunk dicts with 'text' field
            language: Optional language code
            document_id: Optional document ID

        Returns:
            Combined extraction results
        """
        all_entities: Dict[str, ExtractedEntity] = {}
        all_relationships = []

        # Detect language from first chunk
        if not language and chunks:
            first_text = chunks[0].get("text", "")
            language = self.detect_language(first_text)

        logger.info(f"Processing {len(chunks)} chunks with ensemble extraction")

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            if not chunk_text.strip():
                continue

            # Extract from this chunk
            result = self.extract(chunk_text, language, document_id)

            # Merge entities
            for e_dict in result.get("entities", []):
                entity = ExtractedEntity(
                    text=e_dict["text"],
                    type=e_dict["type"],
                    confidence=e_dict["confidence"],
                    sources=e_dict.get("sources", []),
                    frequency=e_dict.get("frequency", 1),
                    attributes=e_dict.get("attributes", {})
                )
                self._merge_entities(all_entities, [entity], "chunk")

            # Collect relationships
            all_relationships.extend(result.get("relationships", []))

        # Final calibration
        calibrated = self._calibrate_confidence(all_entities)
        filtered = [e for e in calibrated if e.confidence >= self.confidence_threshold]

        # Deduplicate relationships
        unique_rels = self._deduplicate_relationships(all_relationships)

        return {
            "status": "success",
            "entities": [e.to_dict() for e in filtered],
            "relationships": unique_rels,
            "total_entities": len(filtered),
            "extraction_method": "ensemble"
        }

    def _deduplicate_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate relationships."""
        seen = set()
        unique = []

        for rel in relationships:
            source = rel.get("source", "").lower().strip()
            target = rel.get("target", "").lower().strip()
            rel_type = rel.get("type", "").lower().strip()

            key = (source, target, rel_type)
            if key not in seen:
                seen.add(key)
                unique.append(rel)

        return unique


# Singleton instance
_ensemble_extractor: Optional[EnsembleEntityExtractor] = None


def get_ensemble_extractor(**kwargs) -> EnsembleEntityExtractor:
    """Get or create ensemble extractor singleton."""
    global _ensemble_extractor

    if _ensemble_extractor is None:
        # Try to get LLM extractor
        llm_extractor = None
        try:
            from .llm_entity_extractor import LLMEntityExtractor
            llm_extractor = LLMEntityExtractor()
        except Exception as e:
            logger.warning(f"Could not initialize LLM extractor: {e}")

        _ensemble_extractor = EnsembleEntityExtractor(
            llm_extractor=llm_extractor,
            **kwargs
        )

    return _ensemble_extractor
