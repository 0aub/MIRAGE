"""
Entity Extractor
Extracts named entities from text using:
- CAMeLTools for Arabic
- spaCy for English
Supports dynamic entity types that emerge from data
"""

import re
from typing import List, Dict, Any, Optional
from loguru import logger

# Language detection
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available")

try:
    from camel_tools.ner import NERecognizer
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    logger.warning("CAMeLTools not available")

try:
    from .llm_entity_extractor import LLMEntityExtractor
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM entity extractor not available")

try:
    from .entity_resolver import EntityResolver
    RESOLVER_AVAILABLE = True
except ImportError:
    RESOLVER_AVAILABLE = False
    logger.warning("Entity resolver not available")


class EntityExtractor:
    """Extract named entities from text in multiple languages"""

    def __init__(self, use_llm: bool = True, resolve_duplicates: bool = True, similarity_threshold: float = 0.85):
        """
        Initialize entity extractor

        Args:
            use_llm: If True, try to use LLM-based extraction first (recommended)
            resolve_duplicates: If True, resolve duplicate entities using embedding similarity
            similarity_threshold: Cosine similarity threshold for merging entities (0.85 recommended)
        """
        self.spacy_model = None
        self.camel_ner = None
        self.llm_extractor = None
        self.entity_resolver = None

        # Initialize models lazily
        self._spacy_loaded = False
        self._camel_loaded = False
        self._llm_loaded = False

        # Try to initialize LLM extractor if requested
        self.use_llm = use_llm
        if use_llm and LLM_AVAILABLE:
            try:
                self.llm_extractor = LLMEntityExtractor()
                self._llm_loaded = True
                logger.info("LLM entity extractor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM extractor: {e}")
                self._llm_loaded = False

        # Initialize entity resolver for deduplication
        self.resolve_duplicates = resolve_duplicates
        if resolve_duplicates and RESOLVER_AVAILABLE:
            try:
                self.entity_resolver = EntityResolver(similarity_threshold=similarity_threshold)
                logger.info(f"Entity resolver initialized (threshold: {similarity_threshold})")
            except Exception as e:
                logger.warning(f"Failed to initialize entity resolver: {e}")
                self.entity_resolver = None

        # Entity type mapping
        self.entity_type_map = {
            # spaCy types
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Money",
            "PERCENT": "Percentage",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "WORK_OF_ART": "WorkOfArt",
            "LAW": "Law",
            "LANGUAGE": "Language",
            "NORP": "Group",
            "FAC": "Facility",

            # CAMeL types (will be mapped similarly)
            "per": "Person",
            "org": "Organization",
            "loc": "Location",
            "misc": "Miscellaneous",
        }

    def _load_spacy_model(self):
        """Load spaCy model for English"""
        if not SPACY_AVAILABLE or self._spacy_loaded:
            return

        try:
            # Try to load the model
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                # Model not installed, use blank model
                logger.warning("spaCy model not found. Using blank model. Install with: python -m spacy download en_core_web_sm")
                self.spacy_model = spacy.blank("en")

            self._spacy_loaded = True

        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            self.spacy_model = None

    def _load_camel_model(self):
        """Load CAMeLTools NER for Arabic"""
        if not CAMEL_AVAILABLE or self._camel_loaded:
            return

        try:
            self.camel_ner = NERecognizer.pretrained()
            self._camel_loaded = True
            logger.info("Loaded CAMeL NER model")
        except Exception as e:
            logger.error(f"Error loading CAMeL NER: {e}")
            self.camel_ner = None

    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Arabic or English

        Args:
            text: Text to analyze

        Returns:
            'ar' for Arabic, 'en' for English
        """
        # Count Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        # Count Latin characters
        latin_chars = len(re.findall(r'[a-zA-Z]', text))

        # Determine language
        if arabic_chars > latin_chars:
            return 'ar'
        return 'en'

    def extract_entities(
        self,
        text: str,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text

        Args:
            text: Text to extract entities from
            language: Optional language code ('ar' or 'en'). Auto-detected if None.

        Returns:
            List of entity dicts with: text, type, start, end, confidence
        """
        if not text or not text.strip():
            return []

        # Detect language if not provided
        if language is None:
            language = self.detect_language(text)

        logger.debug(f"Extracting entities from {len(text)} chars ({language})")

        entities = []

        if language == 'ar':
            entities = self._extract_arabic_entities(text)
        else:
            entities = self._extract_english_entities(text)

        # Deduplicate and normalize
        entities = self._deduplicate_entities(entities)

        logger.info(f"Extracted {len(entities)} entities ({language})")

        return entities

    def _extract_english_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from English text using spaCy"""
        if not self._spacy_loaded:
            self._load_spacy_model()

        if self.spacy_model is None:
            logger.warning("spaCy model not available, returning empty entities")
            return []

        entities = []

        try:
            doc = self.spacy_model(text)

            for ent in doc.ents:
                entity_type = self.entity_type_map.get(ent.label_, ent.label_)

                entities.append({
                    "text": ent.text,
                    "type": entity_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                    "confidence": 0.8,  # spaCy doesn't provide confidence scores
                })

        except Exception as e:
            logger.error(f"Error extracting English entities: {e}")

        return entities

    def _extract_arabic_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from Arabic text using CAMeLTools"""
        if not self._camel_loaded:
            self._load_camel_model()

        if self.camel_ner is None:
            logger.error("CAMeL NER not available - entity extraction requires proper NER models")
            return []

        entities = []

        try:
            # Split into sentences (simple approach)
            sentences = text.split('.')

            for sentence in sentences:
                if not sentence.strip():
                    continue

                # CAMeL expects tokenized input
                words = sentence.split()

                if not words:
                    continue

                # Get NER tags
                ner_result = self.camel_ner.predict_sentence(words)

                # Extract entities from IOB tags
                current_entity = None
                current_type = None
                current_words = []

                for word, tag in zip(words, ner_result):
                    if tag.startswith('B-'):
                        # Beginning of entity
                        if current_entity:
                            entities.append(current_entity)

                        entity_type = self.entity_type_map.get(
                            tag[2:].lower(),
                            tag[2:]
                        )

                        current_entity = {
                            "text": word,
                            "type": entity_type,
                            "label": tag[2:],
                            "confidence": 0.75,
                        }
                        current_words = [word]

                    elif tag.startswith('I-') and current_entity:
                        # Inside entity
                        current_words.append(word)
                        current_entity["text"] = " ".join(current_words)

                    elif current_entity:
                        # End of entity
                        entities.append(current_entity)
                        current_entity = None
                        current_words = []

                # Add last entity if exists
                if current_entity:
                    entities.append(current_entity)

        except Exception as e:
            logger.error(f"Error extracting Arabic entities with CAMeL: {e}")
            return []

        return entities

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []

        for entity in entities:
            # Create key from text and type
            key = (entity["text"].lower().strip(), entity["type"])

            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        language: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from document chunks
        REQUIRES LLM extraction - no fallback to ensure quality

        Args:
            chunks: List of text chunks
            language: Optional language code
            document_id: Optional document ID for progress tracking

        Returns:
            Dict with entities, relationships, and status information
            Returns status='unavailable' if LLM is not ready instead of raising errors
        """
        # Check if LLM extraction is available
        if not self._llm_loaded or not self.llm_extractor:
            logger.warning("LLM extraction requested but not available - returning status info")
            return {
                "status": "unavailable",
                "message": "LLM entity extraction is currently unavailable. The Llama 3.1 70B model may still be downloading. Please wait a few minutes and try again.",
                "entities": [],
                "relationships": [],
                "total_entities": 0,
                "extraction_method": "none"
            }

        logger.info("Using LLM-based entity extraction")
        try:
            result = self.llm_extractor.extract_from_chunks(chunks, language, document_id)

            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            # Resolve duplicate entities using embedding-based clustering
            if self.resolve_duplicates and self.entity_resolver and entities:
                logger.info(f"Resolving {len(entities)} entities using embedding similarity")

                # Import embedder here to avoid circular dependency
                from ..embeddings import JinaEmbedder
                embedder = JinaEmbedder()

                # Compute embeddings for all entity texts
                entity_texts = [e["text"] for e in entities]
                embeddings = embedder.embed(entity_texts)

                # Resolve duplicates
                entities_before = len(entities)
                entities = self.entity_resolver.resolve_entities(entities, embeddings)
                entities_after = len(entities)

                reduction_pct = (1 - entities_after / entities_before) * 100 if entities_before > 0 else 0
                logger.info(
                    f"Entity resolution: {entities_before} → {entities_after} entities "
                    f"({reduction_pct:.1f}% reduction)"
                )

            # LLM extractor returns both entities and relationships
            return {
                "status": "success",
                "entities": entities,
                "relationships": relationships,
                "total_entities": len(entities),
                "extraction_method": "llm_with_resolution" if self.resolve_duplicates else "llm"
            }
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Check if it's a rate limit error
            error_str = str(e).lower()
            if any(term in error_str for term in ["rate limit", "quota", "429", "resource exhausted"]):
                return {
                    "status": "rate_limited",
                    "message": f"API rate limit exceeded: {str(e)}. Consider using local TGI for unlimited processing.",
                    "entities": [],
                    "relationships": [],
                    "total_entities": 0,
                    "extraction_method": "none"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Entity extraction failed: {str(e)}",
                    "entities": [],
                    "relationships": [],
                    "total_entities": 0,
                    "extraction_method": "none"
                }
