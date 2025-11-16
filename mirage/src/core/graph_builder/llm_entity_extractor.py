"""
LLM-Based Entity and Relationship Extractor
Uses LiteLLM to support multiple providers (OpenAI, Claude, Gemini)
with automatic provider detection and token-aware chunking
"""

import os
import json
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis-py not available - chunk progress tracking disabled")

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("LiteLLM not available")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using rough estimation for token counting")

from ...config import settings


class LLMEntityExtractor:
    """
    Extract entities and relationships using LLMs
    Supports: OpenAI, Claude (Anthropic), Gemini (Google)
    """

    def __init__(self):
        """Initialize with auto-detected LLM provider"""
        self.provider = None
        self.model = None
        # Reduced chunk size for better quality - prioritize thoroughness over speed
        # Smaller chunks = more detailed extraction, better handling of complex text
        self.max_tokens_per_request = 1200  # Conservative for quality extraction
        self.encoding = None
        self.redis_client = None

        # Auto-detect available provider
        self._detect_provider()

        # Initialize token counter
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")

        # Initialize Redis for progress tracking
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.debug("Redis client initialized for chunk progress tracking")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis for progress tracking: {e}")
                self.redis_client = None

    def _detect_provider(self) -> None:
        """Auto-detect which LLM provider to use based on available API keys"""
        if not LITELLM_AVAILABLE:
            logger.error("LiteLLM not installed. Install with: pip install litellm")
            return

        # Check for TGI (local GPU) first - highest priority if enabled
        if settings.use_tgi and settings.tgi_endpoint:
            self.provider = "tgi"
            self.model = "tgi"  # TGI doesn't need model specification
            self.tgi_endpoint = settings.tgi_endpoint
            logger.info(f"Using local TGI endpoint at {settings.tgi_endpoint} for entity extraction (NO RATE LIMITS!)")
            return

        # Check for API keys in order of preference
        if settings.openai_api_key:
            self.provider = "openai"
            self.model = "gpt-4o-mini"  # Fast and cheap
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            logger.info("Using OpenAI (gpt-4o-mini) for entity extraction")

        elif settings.anthropic_api_key:
            self.provider = "anthropic"
            self.model = "claude-3-haiku-20240307"  # Fast and cheap
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
            logger.info("Using Anthropic Claude (Haiku) for entity extraction")

        elif settings.google_api_key:
            self.provider = "google"
            self.model = "gemini/gemini-2.0-flash-exp"  # Latest Gemini 2.0
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
            logger.info("Using Google Gemini 2.0 (Flash) for entity extraction")

        else:
            logger.error("No LLM provider available! Please enable TGI or set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
            self.provider = None

    def _update_progress(self, document_id: str, current_chunk: int, total_chunks: int):
        """Update chunk progress in Redis"""
        if not self.redis_client or not document_id:
            return
        try:
            key = f"processing:{document_id}"
            data = self.redis_client.get(key)
            if data:
                status_data = json.loads(data)
                status_data["current_chunk"] = current_chunk
                status_data["total_chunks"] = total_chunks
                status_data["phase"] = "extraction"  # Entity extraction phase
                # Update with same TTL
                self.redis_client.setex(key, 3600, json.dumps(status_data))
        except Exception as e:
            logger.debug(f"Failed to update chunk progress: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough estimation: ~4 chars per token
            return len(text) // 4

    def chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """
        Split text into chunks that fit within token limits
        Uses overlapping windows to maintain context
        """
        if max_tokens is None:
            max_tokens = self.max_tokens_per_request

        # If text fits, return as single chunk
        total_tokens = self.count_tokens(text)
        if total_tokens <= max_tokens:
            return [text]

        # Split into sentences (support both English and Arabic)
        # Arabic: . ، ؛ (period, comma, semicolon)
        # English: . ! ?
        text_marked = text
        text_marked = text_marked.replace('. ', '.|')
        text_marked = text_marked.replace('.\n', '.|')
        text_marked = text_marked.replace('! ', '!|')
        text_marked = text_marked.replace('? ', '?|')
        text_marked = text_marked.replace('، ', '،|')  # Arabic comma
        text_marked = text_marked.replace('؛ ', '؛|')  # Arabic semicolon
        sentences = [s.strip() for s in text_marked.split('|') if s.strip()]

        # Fallback: if no sentences found (continuous text), split by words
        if len(sentences) <= 1:
            logger.warning("No sentence boundaries found, splitting by token count")
            words = text.split()
            chunks = []
            current_chunk = []
            current_tokens = 0

            for word in words:
                word_tokens = self.count_tokens(word + ' ')

                if current_tokens + word_tokens > max_tokens and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Overlap last 50 words
                    current_chunk = current_chunk[-50:] if len(current_chunk) > 50 else []
                    current_tokens = self.count_tokens(' '.join(current_chunk))

                current_chunk.append(word)
                current_tokens += word_tokens

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            logger.info(f"Split text into {len(chunks)} chunks (total tokens: {total_tokens})")
            return chunks

        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_sentences = 2  # Number of sentences to overlap

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                current_chunk = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"Split text into {len(chunks)} chunks (total tokens: {total_tokens})")
        return chunks

    def extract_entities_and_relationships(
        self,
        text: str,
        language: str = "auto",
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from text using LLM

        Args:
            text: Text to extract from
            language: Language code ("ar", "en", or "auto")
            document_id: Optional document ID for progress tracking

        Returns:
            Dict with 'entities' and 'relationships' lists
        """
        if not self.provider:
            logger.error("No LLM provider available")
            return {"entities": [], "relationships": []}

        # Detect language if needed
        if language == "auto":
            arabic_ratio = len([c for c in text if '\u0600' <= c <= '\u06FF']) / max(len(text), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        # Chunk text if needed
        chunks = self.chunk_text(text)
        total_chunks = len(chunks)

        all_entities = []
        all_relationships = []

        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            logger.info(f"Processing chunk {current_chunk}/{total_chunks}")

            # Update progress in Redis
            self._update_progress(document_id, current_chunk, total_chunks)

            try:
                result = self._extract_from_chunk(chunk, language)
                all_entities.extend(result.get("entities", []))
                all_relationships.extend(result.get("relationships", []))
            except Exception as e:
                # Check if it's a rate limit error
                error_str = str(e).lower()
                if any(term in error_str for term in ["rate limit", "quota", "429", "resource exhausted", "requests per"]):
                    logger.error(f"Rate limit exceeded: {e}")
                    raise Exception(f"Rate limit exceeded: {str(e)}") from e
                logger.error(f"Error processing chunk {current_chunk}: {e}")
                continue

        # Deduplicate entities
        entities = self._deduplicate_entities(all_entities)
        relationships = self._deduplicate_relationships(all_relationships)

        logger.info(f"Extracted {len(entities)} unique entities and {len(relationships)} relationships (before quality filtering)")

        # Apply quality filters to remove generic entities and weak relationships
        entities, relationships = self._filter_quality(entities, relationships)

        logger.info(f"After quality filtering: {len(entities)} entities and {len(relationships)} relationships")

        return {
            "entities": entities,
            "relationships": relationships
        }

    def _extract_from_chunk(self, chunk: str, language: str) -> Dict[str, Any]:
        """Extract entities and relationships from a single chunk"""

        # Create language-specific prompt
        if language == "ar":
            system_prompt = """أنت مستخرج معلومات متخصص في بناء الرسوم البيانية المعرفية الشاملة والغنية بالمعلومات.

**تعليمات استخراج الكيانات:**
- استخرج فقط الكيانات المحددة والملموسة (أسماء أشخاص، منظمات، أماكن، برامج، مشاريع، منتجات، إلخ)
- **تجنب تماماً** الكيانات العامة مثل: "فرصة"، "قطاع"، "مجال"، "جانب"، "نقطة"، "شيء"، "موضوع"، "تحدي"
- لكل كيان، حدد: الاسم، النوع، درجة الأهمية، وصف مختصر (جملة واحدة)، والسمات الرئيسية
- السمات يمكن أن تشمل: التواريخ، الأرقام، المناصب، الأدوار، الإنجازات، المواقع، إلخ

**تعليمات استخراج العلاقات:**
- استخدم أنواع علاقات محددة ووصفية (مثل: يرأس، يدير، شارك_في، أطلق، حصل_على، أسس، يتعاون_مع، يقع_في، عمل_في، فاز_بـ، نظم، استضاف، طور، صمم)
- **ممنوع منعاً باتاً** استخدام العلاقات العامة مثل: "مرتبط_بـ"، "له_علاقة_مع"، "متصل_بـ"، "يتعلق_بـ"
- تأكد أن العلاقة منطقية وواضحة بين الكيانين
- أضف وصف للعلاقة إذا كان هناك سياق مهم (اختياري)
- أضف السمات الزمنية أو الكمية إذا توفرت

**أمثلة:**
1. {"entities": [{"text": "وزارة التعليم", "type": "Organization", "importance": "high", "description": "الجهة الحكومية المسؤولة عن التعليم في المملكة", "attributes": {"sector": "government", "domain": "education"}}, {"text": "الرياض", "type": "Location", "importance": "medium", "description": "عاصمة المملكة العربية السعودية", "attributes": {"type": "capital_city"}}], "relationships": [{"source": "وزارة التعليم", "target": "الرياض", "type": "يقع_في", "description": "المقر الرئيسي"}]}

2. {"entities": [{"text": "محمد بن سلمان", "type": "Person", "importance": "high", "description": "ولي العهد رئيس مجلس الوزراء", "attributes": {"role": "Crown Prince", "position": "Prime Minister"}}, {"text": "رؤية 2030", "type": "Program", "importance": "high", "description": "خطة استراتيجية لتنويع الاقتصاد السعودي", "attributes": {"launch_year": "2016", "target_year": "2030"}}], "relationships": [{"source": "محمد بن سلمان", "target": "رؤية 2030", "type": "أطلق", "attributes": {"year": "2016"}}]}

3. {"entities": [{"text": "جائزة التحول الرقمي", "type": "Award", "importance": "high", "description": "جائزة سنوية للابتكار في التحول الرقمي", "attributes": {"frequency": "annual", "category": "digital_innovation"}}, {"text": "وزارة الاتصالات", "type": "Organization", "importance": "high", "description": "وزارة الاتصالات وتقنية المعلومات", "attributes": {"sector": "government", "domain": "telecommunications"}}], "relationships": [{"source": "وزارة الاتصالات", "target": "جائزة التحول الرقمي", "type": "نظم", "description": "تنظيم الجائزة السنوية"}]}"""

            user_prompt = f"""استخرج الكيانات والعلاقات من النص التالي مع معلومات شاملة ومفصلة. تأكد من تضمين الوصف والسمات لكل كيان.

النص: {chunk}

JSON:"""

        else:  # English
            system_prompt = """You are an expert information extractor specialized in building comprehensive and information-rich knowledge graphs.

**Entity Extraction Instructions:**
- Extract only specific, concrete entities (person names, organizations, places, programs, projects, products, etc.)
- **Completely avoid** generic entities like: "opportunity", "sector", "field", "aspect", "point", "thing", "topic", "challenge", "area", "way"
- For each entity, specify: name, type, importance level, brief description (one sentence), and key attributes
- Attributes can include: dates, numbers, positions, roles, achievements, locations, etc.

**Relationship Extraction Instructions:**
- Use specific, descriptive relationship types (e.g., "leads", "manages", "participates_in", "launched", "won", "founded", "collaborates_with", "located_in", "works_for", "organized", "hosted", "developed", "designed")
- **Strictly forbidden** to use generic relationships like: "related_to", "has_relationship_with", "connected_to", "associated_with", "linked_to"
- Ensure the relationship is logical and clear between the two entities
- Add relationship description if there's important context (optional)
- Add temporal or quantitative attributes when available

**Examples:**
1. {"entities": [{"text": "Ministry of Education", "type": "Organization", "importance": "high", "description": "Government body responsible for education in the Kingdom", "attributes": {"sector": "government", "domain": "education"}}, {"text": "Riyadh", "type": "Location", "importance": "medium", "description": "Capital city of Saudi Arabia", "attributes": {"type": "capital_city"}}], "relationships": [{"source": "Ministry of Education", "target": "Riyadh", "type": "located_in", "description": "Main headquarters"}]}

2. {"entities": [{"text": "Bill Gates", "type": "Person", "importance": "high", "description": "Co-founder of Microsoft and philanthropist", "attributes": {"role": "co-founder", "known_for": "Microsoft"}}, {"text": "Microsoft", "type": "Organization", "importance": "high", "description": "Global technology company", "attributes": {"founded": "1975", "industry": "technology"}}], "relationships": [{"source": "Bill Gates", "target": "Microsoft", "type": "founded", "attributes": {"year": "1975"}}]}

3. {"entities": [{"text": "Digital Transformation Award", "type": "Award", "importance": "high", "description": "Annual award for digital innovation excellence", "attributes": {"frequency": "annual", "category": "digital_innovation"}}, {"text": "Ministry of Communications", "type": "Organization", "importance": "high", "description": "Ministry of Communications and Information Technology", "attributes": {"sector": "government", "domain": "telecommunications"}}], "relationships": [{"source": "Ministry of Communications", "target": "Digital Transformation Award", "type": "organized", "description": "Annual award organization"}]}"""

            user_prompt = f"""Extract entities and relationships from the following text with comprehensive and detailed information. Make sure to include description and attributes for each entity.

Text: {chunk}

JSON:"""

        try:
            # Handle TGI separately (uses /v1/chat/completions for automatic chat template)
            if self.provider == "tgi":
                # Use OpenAI-compatible endpoint which auto-applies correct chat template
                # Retry logic with exponential backoff for 422 errors (chunk too large)
                max_retries = 3
                retry_delay = 2

                for attempt in range(max_retries):
                    try:
                        tgi_response = requests.post(
                            f"{self.tgi_endpoint}/v1/chat/completions",
                            json={
                                "model": "tgi",  # Required but ignored by TGI
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                "temperature": 0.2,  # Low for consistency
                                "max_tokens": 2000,  # Reduced for concise JSON
                                # Note: response_format not supported by TGI v3.3.6
                            },
                            timeout=180  # 3 minute timeout for long extractions
                        )
                        tgi_response.raise_for_status()
                        content = tgi_response.json()["choices"][0]["message"]["content"]
                        break  # Success!

                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 422 and attempt < max_retries - 1:
                            # Chunk too large - split it smaller and return partial results
                            logger.warning(f"Chunk too large (422 error), attempt {attempt + 1}/{max_retries}. Chunk has ~{self.count_tokens(chunk)} tokens")
                            # If this is failing, the chunk is still too large even after our chunking
                            # Return empty to skip this problematic chunk
                            if attempt == max_retries - 2:
                                logger.error(f"Giving up on chunk after {max_retries} attempts - chunk may be too large or malformed")
                                return {"entities": [], "relationships": []}
                            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        else:
                            raise  # Re-raise other errors or final 422
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"TGI request failed (attempt {attempt + 1}/{max_retries}): {e}")
                            time.sleep(retry_delay * (attempt + 1))
                        else:
                            raise
            else:
                # Use LiteLLM for cloud providers
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=4000,  # Increased for longer responses
                    response_format={"type": "json_object"},  # Force JSON output
                )
                content = response.choices[0].message.content

            # Parse JSON from response
            # Sometimes LLM wraps in ```json ... ```, so clean it
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())

            # Add confidence scores
            for entity in result.get("entities", []):
                entity["confidence"] = 0.9 if entity.get("importance") == "high" else 0.7

            for rel in result.get("relationships", []):
                rel["confidence"] = 0.8

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Log first 500 chars of response for debugging
            logger.error(f"Response preview: {content[:500] if content else 'None'}...")
            return {"entities": [], "relationships": []}

        except Exception as e:
            # Check if it's a rate limit error
            error_str = str(e).lower()
            if any(term in error_str for term in ["rate limit", "quota", "429", "resource exhausted", "requests per"]):
                logger.error(f"Rate limit exceeded: {e}")
                raise  # Re-raise rate limit errors
            logger.error(f"Error calling LLM: {e}")
            return {"entities": [], "relationships": []}

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities, keeping highest confidence"""
        seen = {}
        for entity in entities:
            key = (entity["text"].lower().strip(), entity["type"])
            if key not in seen or entity.get("confidence", 0) > seen[key].get("confidence", 0):
                seen[key] = entity

        return list(seen.values())

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships"""
        seen = set()
        unique = []

        for rel in relationships:
            # Validate that relationship has required fields
            if not all(key in rel for key in ["source", "target", "type"]):
                logger.warning(f"Skipping invalid relationship (missing required fields): {rel}")
                continue

            key = (rel["source"].lower().strip(), rel["target"].lower().strip(), rel["type"])
            if key not in seen:
                seen.add(key)
                unique.append(rel)

        return unique

    def _is_generic_entity(self, entity: Dict[str, Any]) -> bool:
        """
        Check if entity is too generic/vague to be useful

        Returns:
            True if entity should be filtered out
        """
        text = entity.get("text", "").strip()
        entity_type = entity.get("type", "")

        # Filter very short entities (likely incomplete)
        if len(text) < 2:
            return True

        # Filter overly long entity names (likely contaminated with description)
        # Entity names should typically be short (< 100 characters for most, < 200 max)
        # This catches cases where the LLM put the description in the text field
        if len(text) > 150:
            logger.warning(f"Filtering overly long entity name ({len(text)} chars): {text[:50]}...")
            return True

        # Blacklist of generic terms in Arabic
        arabic_generic_terms = {
            "فرصة", "فرص", "فرصة كبيرة", "فرصة جيدة",  # opportunity
            "قطاع", "قطاعات",  # sector
            "مجال", "مجالات",  # field
            "جانب", "جوانب",  # aspect
            "نقطة", "نقاط",  # point
            "عنصر", "عناصر",  # element
            "جزء", "أجزاء",  # part
            "شيء", "أشياء",  # thing
            "موضوع", "مواضيع",  # topic
            "حالة", "حالات",  # case
            "مثال", "أمثلة",  # example
            "طريقة", "طرق",  # method
            "نوع", "أنواع",  # type
            "مستوى", "مستويات",  # level
            "درجة",  # degree
            "إمكانية", "إمكانيات",  # possibility
            "هدف", "أهداف",  # goal (unless it's a specific program name)
            "نتيجة", "نتائج",  # result
            "تحدي", "تحديات",  # challenge
        }

        # Blacklist of generic terms in English
        english_generic_terms = {
            "opportunity", "opportunities", "big opportunity", "good opportunity",
            "sector", "sectors",
            "field", "fields",
            "aspect", "aspects",
            "point", "points",
            "element", "elements",
            "part", "parts",
            "thing", "things",
            "topic", "topics",
            "case", "cases",
            "example", "examples",
            "method", "methods",
            "type", "types",
            "level", "levels",
            "degree",
            "possibility", "possibilities",
            "goal", "goals",  # unless specific program
            "result", "results",
            "challenge", "challenges",
            "item", "items",
            "area", "areas",
            "way", "ways",
        }

        text_lower = text.lower()

        # Check against blacklists
        if text_lower in arabic_generic_terms or text_lower in english_generic_terms:
            logger.debug(f"Filtering generic entity: {text}")
            return True

        # Filter entities that are just numbers or dates without context
        if entity_type in ["Date", "Time", "Money", "Percentage"] and len(text.split()) <= 2:
            # These are too generic without entity context
            return True

        # Filter single-word entities with type "Miscellaneous" (usually noise)
        if entity_type == "Miscellaneous" and len(text.split()) <= 1:
            return True

        return False

    def _is_weak_relationship(self, relationship: Dict[str, Any]) -> bool:
        """
        Check if relationship type is too weak/generic to be useful

        Returns:
            True if relationship should be filtered out
        """
        rel_type = relationship.get("type", "").lower().strip()

        # Blacklist of weak/generic relationship types
        weak_types = {
            # English
            "related_to", "related to",
            "has_relationship_with", "has relationship with",
            "connected_to", "connected to",
            "associated_with", "associated with",
            "linked_to", "linked to",
            "has_connection_with", "has connection with",
            "relates_to", "relates to",

            # Arabic
            "مرتبط_بـ", "مرتبط ب", "مرتبط",
            "له_علاقة_مع", "له علاقة مع", "له علاقة",
            "متصل_بـ", "متصل ب",
            "مقترن_بـ", "مقترن ب",
            "يتعلق_بـ", "يتعلق ب", "يتعلق",
        }

        if rel_type in weak_types:
            logger.debug(f"Filtering weak relationship type: {rel_type}")
            return True

        # Filter relationships where source and target are identical
        source = relationship.get("source", "").lower().strip()
        target = relationship.get("target", "").lower().strip()
        if source == target:
            logger.debug(f"Filtering self-referential relationship: {source} -> {target}")
            return True

        return False

    def _filter_quality(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Apply quality filters to entities and relationships

        Args:
            entities: List of entities
            relationships: List of relationships

        Returns:
            Tuple of (filtered_entities, filtered_relationships)
        """
        # Filter entities
        quality_entities = [
            e for e in entities
            if not self._is_generic_entity(e)
        ]

        # Create set of valid entity names for relationship filtering
        valid_entity_names = {e["text"].lower().strip() for e in quality_entities}

        # Filter relationships:
        # 1. Remove weak relationship types
        # 2. Remove relationships referencing filtered entities
        quality_relationships = []
        for rel in relationships:
            # Check if relationship type is weak
            if self._is_weak_relationship(rel):
                continue

            # Check if both entities still exist after entity filtering
            source = rel.get("source", "").lower().strip()
            target = rel.get("target", "").lower().strip()

            if source not in valid_entity_names or target not in valid_entity_names:
                logger.debug(f"Filtering relationship with missing entity: {source} -> {target}")
                continue

            quality_relationships.append(rel)

        filtered_entity_count = len(entities) - len(quality_entities)
        filtered_rel_count = len(relationships) - len(quality_relationships)

        if filtered_entity_count > 0 or filtered_rel_count > 0:
            logger.info(
                f"Quality filter: removed {filtered_entity_count} generic entities "
                f"and {filtered_rel_count} weak relationships"
            )

        return quality_entities, quality_relationships

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        language: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from document chunks with per-chunk processing
        This avoids token limit errors by processing each semantic chunk independently

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            language: Optional language override
            document_id: Optional document ID for progress tracking

        Returns:
            Dict with 'entities' and 'relationships'
        """
        if not self.provider:
            logger.error("No LLM provider available")
            return {"entities": [], "relationships": []}

        # Detect language if needed (check first chunk)
        if language is None or language == "auto":
            first_text = chunks[0].get("text", "") if chunks else ""
            arabic_ratio = len([c for c in first_text if '\u0600' <= c <= '\u06FF']) / max(len(first_text), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        logger.info(f"Processing {len(chunks)} semantic chunks for entity extraction (language: {language})")

        # Process each semantic chunk individually
        all_entities = []
        all_relationships = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            chunk_text = chunk.get("text", "")

            if not chunk_text.strip():
                continue

            logger.info(f"Extracting from chunk {current_chunk}/{total_chunks} ({len(chunk_text)} chars)")

            # Update progress in Redis
            self._update_progress(document_id, current_chunk, total_chunks)

            try:
                # Extract from this semantic chunk
                result = self._extract_from_chunk(chunk_text, language)

                entities_found = len(result.get("entities", []))
                relationships_found = len(result.get("relationships", []))

                logger.info(f"Chunk {current_chunk}: Found {entities_found} entities, {relationships_found} relationships")

                all_entities.extend(result.get("entities", []))
                all_relationships.extend(result.get("relationships", []))

            except Exception as e:
                # Check if it's a rate limit error
                error_str = str(e).lower()
                if any(term in error_str for term in ["rate limit", "quota", "429", "resource exhausted", "requests per"]):
                    logger.error(f"Rate limit exceeded: {e}")
                    raise Exception(f"Rate limit exceeded: {str(e)}") from e
                logger.error(f"Error processing chunk {current_chunk}: {e}")
                continue

        logger.info(f"Raw extraction complete: {len(all_entities)} entities, {len(all_relationships)} relationships (before deduplication)")

        # Deduplicate entities and relationships
        entities = self._deduplicate_entities(all_entities)
        relationships = self._deduplicate_relationships(all_relationships)

        logger.info(f"After deduplication: {len(entities)} unique entities, {len(relationships)} unique relationships")

        # Apply quality filters to remove generic entities and weak relationships
        entities, relationships = self._filter_quality(entities, relationships)

        logger.info(f"After quality filtering: {len(entities)} entities, {len(relationships)} relationships")

        return {
            "entities": entities,
            "relationships": relationships
        }
