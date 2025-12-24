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

# Required imports - no fallback (enforced in Docker)
import spacy
from camel_tools.ner import NERecognizer
from .extraction import LLMEntityExtractor
from .entity_resolver import EntityResolver
from .ensemble_extractor import EnsembleEntityExtractor, get_ensemble_extractor


class EntityExtractor:
    """Extract named entities from text in multiple languages"""

    def __init__(
        self,
        use_llm: bool = True,
        use_ensemble: bool = True,
        resolve_duplicates: bool = True,
        similarity_threshold: float = 0.85,
        embedder=None
    ):
        """
        Initialize entity extractor

        Args:
            use_llm: If True, try to use LLM-based extraction first (recommended)
            use_ensemble: If True, use ensemble extraction (CAMeL + patterns + LLM)
            resolve_duplicates: If True, resolve duplicate entities using embedding similarity
            similarity_threshold: Cosine similarity threshold for merging entities (0.85 recommended)
            embedder: Optional embedder for generating entity embeddings (for semantic relationships)
        """
        self.spacy_model = None
        self.camel_ner = None
        self.llm_extractor = None
        self.entity_resolver = None
        self.ensemble_extractor = None
        self.embedder = embedder  # Store embedder for entity embedding generation

        # Initialize models lazily
        self._spacy_loaded = False
        self._camel_loaded = False
        self._llm_loaded = False
        self._ensemble_loaded = False

        # Phase 2: Try ensemble extraction first (combines CAMeL + patterns + LLM)
        self.use_ensemble = use_ensemble
        if use_ensemble:
            self.ensemble_extractor = get_ensemble_extractor()
            self._ensemble_loaded = True
            logger.info("Ensemble entity extractor initialized (CAMeL + patterns + LLM)")

        # Initialize LLM extractor if requested and ensemble not loaded
        self.use_llm = use_llm
        if use_llm and not self._ensemble_loaded:
            self.llm_extractor = LLMEntityExtractor()
            self._llm_loaded = True
            logger.info("LLM entity extractor initialized successfully")

        # Initialize entity resolver for deduplication (required)
        self.resolve_duplicates = resolve_duplicates
        if resolve_duplicates:
            self.entity_resolver = EntityResolver(similarity_threshold=similarity_threshold)
            logger.info(f"Entity resolver initialized (threshold: {similarity_threshold})")

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
        if self._spacy_loaded:
            return

        try:
            self.spacy_model = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            # Model not installed, use blank model
            logger.warning("spaCy model not found. Using blank model. Install with: python -m spacy download en_core_web_sm")
            self.spacy_model = spacy.blank("en")

        self._spacy_loaded = True

    def _load_camel_model(self):
        """Load CAMeLTools NER for Arabic"""
        if self._camel_loaded:
            return

        self.camel_ner = NERecognizer.pretrained()
        self._camel_loaded = True
        logger.info("Loaded CAMeL NER model")

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
        Uses ensemble extraction (Phase 2) or LLM extraction as fallback

        Args:
            chunks: List of text chunks
            language: Optional language code
            document_id: Optional document ID for progress tracking

        Returns:
            Dict with entities, relationships, and status information
            Returns status='unavailable' if extraction is not ready
        """
        # Phase 2: Try ensemble extraction first (CAMeL + patterns + LLM)
        if self._ensemble_loaded and self.ensemble_extractor:
            logger.info("Using ensemble entity extraction (CAMeL + patterns + LLM)")
            try:
                result = self.ensemble_extractor.extract_from_chunks(chunks, language, document_id)

                entities = result.get("entities", [])
                relationships = result.get("relationships", [])

                # Apply entity resolution if enabled
                if self.resolve_duplicates and self.entity_resolver and entities:
                    logger.info(f"Resolving {len(entities)} entities using embedding similarity")

                    from ..embeddings import JinaEmbedder
                    embedder = JinaEmbedder()

                    entity_texts = [e.get("text", e.get("name", "")) for e in entities]
                    embeddings = embedder.embed(entity_texts)

                    entities_before = len(entities)
                    entities = self.entity_resolver.resolve_entities(entities, embeddings)
                    entities_after = len(entities)

                    reduction_pct = (1 - entities_after / entities_before) * 100 if entities_before > 0 else 0
                    logger.info(
                        f"Entity resolution: {entities_before} → {entities_after} entities "
                        f"({reduction_pct:.1f}% reduction)"
                    )

                return {
                    "status": "success",
                    "entities": entities,
                    "relationships": relationships,
                    "total_entities": len(entities),
                    "extraction_method": "ensemble",
                    "extraction_stats": result.get("extraction_stats", {})
                }
            except Exception as e:
                logger.warning(f"Ensemble extraction failed, falling back to LLM: {e}")

        # Fallback: Check if LLM extraction is available
        if not self._llm_loaded or not self.llm_extractor:
            logger.warning("No extraction method available - returning status info")
            return {
                "status": "unavailable",
                "message": "Entity extraction is currently unavailable. Please ensure TGI is running or API keys are configured.",
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

    def extract_with_chunk_references(
        self,
        chunks: List[Dict[str, Any]],
        language: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and track which chunks mention them (for Hybrid Vector-Graph)

        This is a wrapper around extract_from_chunks that adds chunk references to each entity.
        Required format: {name, type, chunks: [chunk_ids], confidence}

        Args:
            chunks: List of chunks with 'id' and 'text' fields
            language: Language code
            document_id: Document ID

        Returns:
            Dict with:
            - entities_by_name: {entity_name: {name, type, chunks: [ids], confidence}}
            - all_entities: Original flat list
            - relationships: Entity relationships
        """
        # First, extract entities normally
        result = self.extract_from_chunks(chunks, language, document_id)

        if result.get("status") != "success":
            return result  # Return error/unavailable status as-is

        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        # Build mapping: entity_name -> [chunk_ids where it appears]
        # Initialize all entities first to avoid filtering
        entity_chunk_map = {}
        for entity in entities:
            entity_name = entity.get("name") or entity.get("text")
            if entity_name not in entity_chunk_map:
                entity_chunk_map[entity_name] = {
                    "name": entity_name,
                    "type": entity.get("type", "Unknown"),
                    "chunks": [],
                    "confidence_scores": [entity.get("confidence", 1.0)]
                }

        # Map entities to chunks using improved matching
        for chunk in chunks:
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            chunk_text = chunk.get("text", "").lower()

            # Find which entities appear in this chunk with better matching
            for entity in entities:
                entity_name = entity.get("name") or entity.get("text")
                entity_lower = entity_name.lower()

                # Use word boundary matching to avoid false positives like "Plan" in "Planning"
                # Check if entity appears as a complete word/phrase
                import re
                # Escape special regex characters in entity name
                escaped_entity = re.escape(entity_lower)
                # Match with word boundaries (or surrounded by punctuation/whitespace)
                pattern = r'(?:^|[\s\.,;:!?\(\)\[\]\{\}"\'`])'  + escaped_entity + r'(?:$|[\s\.,;:!?\(\)\[\]\{\}"\'`])'

                if re.search(pattern, chunk_text):
                    if chunk_id not in entity_chunk_map[entity_name]["chunks"]:
                        entity_chunk_map[entity_name]["chunks"].append(chunk_id)

        # Calculate average confidence per entity and generate embeddings
        entities_with_chunks = []
        entity_names = list(entity_chunk_map.keys())

        # Batch generate embeddings for all entity names (more efficient)
        entity_embeddings = []
        if entity_names and hasattr(self, 'embedder') and self.embedder:
            try:
                logger.info(f"Generating embeddings for {len(entity_names)} entities...")
                entity_embeddings = self.embedder.embed(entity_names, task="retrieval.query")
                logger.info(f"Generated {len(entity_embeddings)} entity embeddings")
            except Exception as e:
                logger.warning(f"Failed to generate entity embeddings: {e}")
                entity_embeddings = []

        for idx, (entity_name, data) in enumerate(entity_chunk_map.items()):
            avg_confidence = (
                sum(data["confidence_scores"]) / len(data["confidence_scores"])
                if data["confidence_scores"] else 0.5
            )

            entity_dict = {
                "name": entity_name,
                "type": data["type"],
                "chunks": list(set(data["chunks"])),  # Deduplicate
                "confidence": avg_confidence
            }

            # Add embedding if available
            if idx < len(entity_embeddings):
                entity_dict["embedding"] = entity_embeddings[idx]

            entities_with_chunks.append(entity_dict)

        logger.info(
            f"Mapped {len(entities_with_chunks)} unique entities to chunks "
            f"(avg {sum(len(e['chunks']) for e in entities_with_chunks) / len(entities_with_chunks):.1f} chunks per entity)"
            if entities_with_chunks else "No entities found"
        )

        return {
            "status": "success",
            "entities_by_name": {e["name"]: e for e in entities_with_chunks},
            "entities_with_chunks": entities_with_chunks,  # For Neo4j storage
            "all_entities": entities,  # Original flat list
            "relationships": relationships,
            "total_entities": len(entities_with_chunks),
            "extraction_method": result.get("extraction_method")
        }
