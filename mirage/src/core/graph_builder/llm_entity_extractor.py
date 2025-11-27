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
from .entity_normalizer import EntityNormalizer


class LLMEntityExtractor:
    """
    Extract entities and relationships using LLMs
    Supports: OpenAI, Claude (Anthropic), Gemini (Google)
    """

    def __init__(self):
        """Initialize with auto-detected LLM provider"""
        self.provider = None
        self.model = None
        # GraphRAG best practice: 400-600 tokens per chunk for optimal entity extraction
        # Research shows smaller chunks detect 50x more entities than 1200+ token chunks
        # Microsoft GraphRAG: "600 tokens = more precise extraction, higher entity detection"
        self.max_tokens_per_request = 600  # GraphRAG recommended size
        self.encoding = None
        self.redis_client = None

        # Initialize entity normalizer for deduplication
        self.normalizer = EntityNormalizer()
        logger.info("Entity normalizer initialized for duplicate prevention")

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
- **مهم جداً**: استخرج **جميع** الكيانات المحددة والملموسة من النص، وليس فقط الكيانات الرئيسية أو المهمة
- **لا تتجاهل أي كيان محدد** حتى لو ذُكر مرة واحدة فقط
- الكيانات المطلوبة: أسماء أشخاص، منظمات، أماكن، برامج، مشاريع، منتجات، فعاليات، جوائز، سياسات، تقنيات، مقاييس، إلخ
- **تجنب تماماً** الكيانات العامة مثل: "فرصة"، "قطاع"، "مجال"، "جانب"، "نقطة"، "شيء"، "موضوع"، "تحدي"، "أمر"، "حالة"
- لكل كيان، حدد: الاسم، النوع، درجة الأهمية، وصف مختصر (جملة واحدة)، والسمات الرئيسية
- السمات يمكن أن تشمل: التواريخ، الأرقام، المناصب، الأدوار، الإنجازات، المواقع، إلخ

**تعليمات استخراج العلاقات:**
- استخرج **جميع** العلاقات الواضحة بين الكيانات
- استخدم أنواع علاقات محددة ووصفية (مثل: يرأس، يدير، شارك_في، أطلق، حصل_على، أسس، يتعاون_مع، يقع_في، عمل_في، فاز_بـ، نظم، استضاف، طور، صمم، تحدث_عن، أعلن_عن)
- **ممنوع منعاً باتاً** استخدام العلاقات العامة مثل: "مرتبط_بـ"، "له_علاقة_مع"، "متصل_بـ"، "يتعلق_بـ"
- تأكد أن العلاقة منطقية وواضحة بين الكيانين
- أضف وصف للعلاقة إذا كان هناك سياق مهم (اختياري)
- أضف السمات الزمنية أو الكمية إذا توفرت

**أمثلة:**
1. {"entities": [{"text": "وزارة التعليم", "type": "Organization", "importance": "high", "description": "الجهة الحكومية المسؤولة عن التعليم", "attributes": {"sector": "government", "domain": "education"}}, {"text": "الرياض", "type": "Location", "importance": "medium", "description": "عاصمة المملكة", "attributes": {"type": "capital"}}], "relationships": [{"source": "وزارة التعليم", "target": "الرياض", "type": "يقع_في", "description": "المقر الرئيسي"}]}

2. {"entities": [{"text": "محمد بن سلمان", "type": "Person", "importance": "high", "description": "ولي العهد رئيس مجلس الوزراء", "attributes": {"role": "Crown Prince"}}, {"text": "رؤية 2030", "type": "Program", "importance": "high", "description": "خطة استراتيجية لتنويع الاقتصاد", "attributes": {"year": "2030"}}], "relationships": [{"source": "محمد بن سلمان", "target": "رؤية 2030", "type": "أطلق", "attributes": {"year": "2016"}}]}

3. {"entities": [{"text": "جائزة التحول الرقمي", "type": "Award", "importance": "high", "description": "جائزة سنوية للابتكار الرقمي", "attributes": {"frequency": "annual"}}, {"text": "وزارة الاتصالات", "type": "Organization", "importance": "high", "description": "وزارة الاتصالات وتقنية المعلومات", "attributes": {"sector": "government"}}], "relationships": [{"source": "وزارة الاتصالات", "target": "جائزة التحول الرقمي", "type": "نظم"}]}

4. {"entities": [{"text": "منصة إحسان", "type": "Product", "importance": "high", "description": "منصة وطنية للتبرعات الخيرية", "attributes": {"type": "digital_platform"}}, {"text": "المركز الوطني للتنمية", "type": "Organization", "importance": "medium", "description": "مركز حكومي للتنمية"}], "relationships": [{"source": "المركز الوطني للتنمية", "target": "منصة إحسان", "type": "طور"}]}

5. {"entities": [{"text": "د. أحمد العيسى", "type": "Person", "importance": "high", "description": "وزير التعليم السابق", "attributes": {"position": "former_minister"}}, {"text": "برنامج نافس", "type": "Program", "importance": "medium", "description": "برنامج لدعم القطاع غير الربحي", "attributes": {"sector": "nonprofit"}}], "relationships": [{"source": "د. أحمد العيسى", "target": "برنامج نافس", "type": "أطلق"}]}

6. {"entities": [{"text": "قمة المعرفة 2024", "type": "Event", "importance": "high", "description": "مؤتمر سنوي للمعرفة والابتكار", "attributes": {"year": "2024"}}, {"text": "دبي", "type": "Location", "importance": "medium", "description": "إمارة دبي", "attributes": {"country": "UAE"}}], "relationships": [{"source": "قمة المعرفة 2024", "target": "دبي", "type": "عُقد_في"}]}

7. {"entities": [{"text": "مؤسسة مسك الخيرية", "type": "Organization", "importance": "high", "description": "مؤسسة خيرية غير ربحية", "attributes": {"type": "nonprofit"}}, {"text": "مبادرة مسك", "type": "Program", "importance": "medium", "description": "مبادرة لتمكين الشباب"}], "relationships": [{"source": "مؤسسة مسك الخيرية", "target": "مبادرة مسك", "type": "أطلق"}]}

8. {"entities": [{"text": "الهيئة السعودية للبيانات", "type": "Organization", "importance": "high", "description": "هيئة حكومية لإدارة البيانات", "attributes": {"sector": "government"}}, {"text": "استراتيجية البيانات الوطنية", "type": "Policy", "importance": "high", "description": "استراتيجية وطنية للبيانات"}], "relationships": [{"source": "الهيئة السعودية للبيانات", "target": "استراتيجية البيانات الوطنية", "type": "أعلن_عن"}]}

9. {"entities": [{"text": "نيوم", "type": "Project", "importance": "high", "description": "مشروع مدينة مستقبلية", "attributes": {"location": "تبوك", "size": "26500_km2"}}, {"text": "تبوك", "type": "Location", "importance": "medium", "description": "منطقة في شمال غرب السعودية"}], "relationships": [{"source": "نيوم", "target": "تبوك", "type": "يقع_في"}]}

10. {"entities": [{"text": "برنامج سكني", "type": "Program", "importance": "high", "description": "برنامج إسكان حكومي", "attributes": {"sector": "housing"}}, {"text": "وزارة الإسكان", "type": "Organization", "importance": "high", "description": "وزارة الإسكان السعودية", "attributes": {"sector": "government"}}], "relationships": [{"source": "وزارة الإسكان", "target": "برنامج سكني", "type": "يدير"}]}

11. {"entities": [{"text": "الذكاء الاصطناعي", "type": "Technology", "importance": "high", "description": "تقنية الذكاء الاصطناعي"}, {"text": "الهيئة السعودية للبيانات", "type": "Organization", "importance": "high", "description": "هيئة حكومية"}], "relationships": [{"source": "الهيئة السعودية للبيانات", "target": "الذكاء الاصطناعي", "type": "تستخدم"}]}

12. {"entities": [{"text": "منتدى الاستثمار", "type": "Event", "importance": "high", "description": "منتدى سنوي للاستثمار", "attributes": {"frequency": "annual"}}, {"text": "هيئة الاستثمار", "type": "Organization", "importance": "high", "description": "هيئة حكومية للاستثمار"}], "relationships": [{"source": "هيئة الاستثمار", "target": "منتدى الاستثمار", "type": "نظم"}]}"""

            user_prompt = f"""استخرج **جميع** الكيانات والعلاقات المحددة من النص التالي. لا تتخطى أي كيان محدد حتى لو ذُكر مرة واحدة.

**مهم جداً**: هذا النص جزء من محادثة أو نص أطول وقد يحتوي على جمل غير مكتملة أو متقطعة - هذا طبيعي تماماً! استخرج الكيانات والعلاقات الموجودة حتى لو كان النص غير مكتمل. لا تعتذر أو ترفض الاستخراج - فقط استخرج ما تجده.

النص: {chunk}

JSON:"""

        else:  # English
            system_prompt = """You are an expert information extractor specialized in building comprehensive and information-rich knowledge graphs.

**Entity Extraction Instructions:**
- **VERY IMPORTANT**: Extract **ALL** specific, concrete entities from the text, not just the main or important ones
- **Do not skip any specific entity** even if mentioned only once
- Required entities: person names, organizations, places, programs, projects, products, events, awards, policies, technologies, metrics, etc.
- **Completely avoid** generic entities like: "opportunity", "sector", "field", "aspect", "point", "thing", "topic", "challenge", "area", "way", "issue", "matter"
- For each entity, specify: name, type, importance level, brief description (one sentence), and key attributes
- Attributes can include: dates, numbers, positions, roles, achievements, locations, etc.

**Relationship Extraction Instructions:**
- Extract **ALL** clear relationships between entities
- Use specific, descriptive relationship types (e.g., "leads", "manages", "participates_in", "launched", "won", "founded", "collaborates_with", "located_in", "works_for", "organized", "hosted", "developed", "designed", "spoke_about", "announced")
- **Strictly forbidden** to use generic relationships like: "related_to", "has_relationship_with", "connected_to", "associated_with", "linked_to"
- Ensure the relationship is logical and clear between the two entities
- Add relationship description if there's important context (optional)
- Add temporal or quantitative attributes when available

**Examples:**
1. {"entities": [{"text": "Ministry of Education", "type": "Organization", "importance": "high", "description": "Government body responsible for education", "attributes": {"sector": "government"}}, {"text": "Riyadh", "type": "Location", "importance": "medium", "description": "Capital city", "attributes": {"type": "capital"}}], "relationships": [{"source": "Ministry of Education", "target": "Riyadh", "type": "located_in"}]}

2. {"entities": [{"text": "Bill Gates", "type": "Person", "importance": "high", "description": "Co-founder of Microsoft", "attributes": {"role": "co-founder"}}, {"text": "Microsoft", "type": "Organization", "importance": "high", "description": "Technology company", "attributes": {"founded": "1975"}}], "relationships": [{"source": "Bill Gates", "target": "Microsoft", "type": "founded", "attributes": {"year": "1975"}}]}

3. {"entities": [{"text": "Digital Transformation Award", "type": "Award", "importance": "high", "description": "Annual award for innovation", "attributes": {"frequency": "annual"}}, {"text": "Ministry of Communications", "type": "Organization", "importance": "high", "description": "Government ministry", "attributes": {"sector": "government"}}], "relationships": [{"source": "Ministry of Communications", "target": "Digital Transformation Award", "type": "organized"}]}

4. {"entities": [{"text": "Ihsan Platform", "type": "Product", "importance": "high", "description": "National charity platform", "attributes": {"type": "digital"}}, {"text": "National Development Center", "type": "Organization", "importance": "medium", "description": "Government development center"}], "relationships": [{"source": "National Development Center", "target": "Ihsan Platform", "type": "developed"}]}

5. {"entities": [{"text": "Dr. Ahmed", "type": "Person", "importance": "high", "description": "Former minister", "attributes": {"position": "minister"}}, {"text": "Nafis Program", "type": "Program", "importance": "medium", "description": "Nonprofit support program"}], "relationships": [{"source": "Dr. Ahmed", "target": "Nafis Program", "type": "launched"}]}

6. {"entities": [{"text": "Knowledge Summit 2024", "type": "Event", "importance": "high", "description": "Annual knowledge conference", "attributes": {"year": "2024"}}, {"text": "Dubai", "type": "Location", "importance": "medium", "description": "UAE emirate"}], "relationships": [{"source": "Knowledge Summit 2024", "target": "Dubai", "type": "held_in"}]}

7. {"entities": [{"text": "Misk Foundation", "type": "Organization", "importance": "high", "description": "Charitable nonprofit", "attributes": {"type": "nonprofit"}}, {"text": "Misk Initiative", "type": "Program", "importance": "medium", "description": "Youth empowerment program"}], "relationships": [{"source": "Misk Foundation", "target": "Misk Initiative", "type": "launched"}]}

8. {"entities": [{"text": "Data Authority", "type": "Organization", "importance": "high", "description": "Government data agency", "attributes": {"sector": "government"}}, {"text": "National Data Strategy", "type": "Policy", "importance": "high", "description": "National data strategy"}], "relationships": [{"source": "Data Authority", "target": "National Data Strategy", "type": "announced"}]}

9. {"entities": [{"text": "NEOM", "type": "Project", "importance": "high", "description": "Future city project", "attributes": {"size": "26500_km2"}}, {"text": "Tabuk", "type": "Location", "importance": "medium", "description": "Northwestern region"}], "relationships": [{"source": "NEOM", "target": "Tabuk", "type": "located_in"}]}

10. {"entities": [{"text": "Sakani Program", "type": "Program", "importance": "high", "description": "Housing program", "attributes": {"sector": "housing"}}, {"text": "Ministry of Housing", "type": "Organization", "importance": "high", "description": "Government ministry", "attributes": {"sector": "government"}}], "relationships": [{"source": "Ministry of Housing", "target": "Sakani Program", "type": "manages"}]}

11. {"entities": [{"text": "Artificial Intelligence", "type": "Technology", "importance": "high", "description": "AI technology"}, {"text": "Data Authority", "type": "Organization", "importance": "high", "description": "Government agency"}], "relationships": [{"source": "Data Authority", "target": "Artificial Intelligence", "type": "uses"}]}

12. {"entities": [{"text": "Investment Forum", "type": "Event", "importance": "high", "description": "Annual investment conference", "attributes": {"frequency": "annual"}}, {"text": "Investment Authority", "type": "Organization", "importance": "high", "description": "Government investment agency"}], "relationships": [{"source": "Investment Authority", "target": "Investment Forum", "type": "organized"}]}"""

            user_prompt = f"""Extract **ALL** specific entities and relationships from the following text. Do not skip any specific entity even if mentioned only once.

**VERY IMPORTANT**: This text is part of a conversation or longer document and may contain incomplete or fragmented sentences - this is completely normal! Extract whatever entities and relationships ARE present, even if the text seems incomplete. Do not apologize or refuse to extract - just extract what you find.

Text: {chunk}

JSON:"""

        try:
            # Handle TGI separately (uses /v1/chat/completions for automatic chat template)
            if self.provider == "tgi":
                # Use OpenAI-compatible endpoint which auto-applies correct chat template
                # FAIL FAST: No retries for connection errors - only retry 422 (chunk too large)
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
                                "temperature": 0.1,  # Very low for consistency and valid JSON
                                "max_tokens": 4096,  # Increased to prevent truncation (Qwen3 handles long responses well)
                                "top_p": 0.9,  # Nucleus sampling for better quality
                                # Note: response_format not supported by TGI v3.3.6
                            },
                            timeout=30  # Reduced timeout: 30 seconds (fail fast!)
                        )
                        tgi_response.raise_for_status()
                        content = tgi_response.json()["choices"][0]["message"]["content"]
                        break  # Success!

                    except requests.exceptions.ConnectionError as e:
                        # TGI is not running or not reachable - FAIL IMMEDIATELY
                        logger.error(f"TGI connection failed - TGI container may not be running: {e}")
                        raise ConnectionError(
                            f"TGI is not available at {self.tgi_endpoint}. "
                            "Please start the TGI container with: docker compose up -d tgi"
                        )
                    except requests.exceptions.Timeout as e:
                        # TGI is too slow or overloaded - FAIL IMMEDIATELY
                        logger.error(f"TGI request timed out after 30 seconds: {e}")
                        raise TimeoutError(
                            f"TGI request timed out. The model may be loading or overloaded. "
                            "Check TGI logs: docker logs mirage-tgi"
                        )
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 422 and attempt < max_retries - 1:
                            # Chunk too large - retry with smaller chunk
                            logger.warning(f"Chunk too large (422 error), attempt {attempt + 1}/{max_retries}. Chunk has ~{self.count_tokens(chunk)} tokens")
                            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        elif e.response.status_code == 422:
                            # Final 422 attempt - give up on this chunk
                            logger.error(f"Chunk still too large after {max_retries} attempts - skipping")
                            return {"entities": [], "relationships": []}
                        else:
                            # Other HTTP errors (500, 503, etc.) - FAIL IMMEDIATELY
                            logger.error(f"TGI HTTP error {e.response.status_code}: {e}")
                            raise
                    except Exception as e:
                        # Unexpected errors - FAIL IMMEDIATELY
                        logger.error(f"Unexpected TGI error: {type(e).__name__}: {e}")
                        raise
            else:
                # Use LiteLLM for cloud providers
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Very low for consistency and valid JSON
                    max_tokens=4096,  # Increased to prevent truncation
                    top_p=0.9,  # Nucleus sampling for better quality
                    response_format={"type": "json_object"},  # Force JSON output
                )
                content = response.choices[0].message.content

            # Parse JSON from response
            # Sometimes LLM wraps in ```json ... ```, so clean it
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            content = content.strip()

            # Try to parse JSON with automatic repair
            result = self._parse_json_with_repair(content)

            if result is None:
                logger.error("All JSON repair attempts failed")
                return {"entities": [], "relationships": []}

            # Add confidence scores
            for entity in result.get("entities", []):
                entity["confidence"] = 0.9 if entity.get("importance") == "high" else 0.7

            for rel in result.get("relationships", []):
                rel["confidence"] = 0.8

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON after repair: {e}")
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

    def _parse_json_with_repair(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON with automatic repair for common LLM output errors.

        Tries multiple repair strategies for malformed JSON from Qwen3-4B:
        - Unterminated strings
        - Missing/extra commas
        - Truncated output

        Returns:
            Parsed JSON dict or None if all repair attempts fail
        """
        import re

        # Strategy 1: Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON object boundaries (in case there's extra text)
        try:
            # Find first { and last }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_content = content[start:end+1]
                return json.loads(json_content)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Fix common issues
        try:
            repaired = content

            # Remove trailing commas before } or ]
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)

            # Fix unterminated strings at end (add closing quote if missing)
            if repaired.count('"') % 2 != 0:
                # Odd number of quotes - add closing quote before last }
                last_brace = repaired.rfind('}')
                if last_brace != -1:
                    repaired = repaired[:last_brace] + '"' + repaired[last_brace:]

            # Try to parse again
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Strategy 4: Truncate to last complete entity/relationship and close JSON
        try:
            # Find last complete entity or relationship closing brace before truncation
            last_complete = max(
                content.rfind('  }'),  # Last complete item with proper indentation
                content.rfind('}\n'),  # Last item followed by newline
            )

            if last_complete != -1:
                truncated = content[:last_complete + 3]  # Include the closing brace

                # Close any open arrays/objects
                open_braces = truncated.count('{') - truncated.count('}')
                open_brackets = truncated.count('[') - truncated.count(']')

                for _ in range(open_brackets):
                    truncated += '\n  ]'
                for _ in range(open_braces):
                    truncated += '\n}'

                return json.loads(truncated)
        except json.JSONDecodeError:
            pass

        # All repair attempts failed
        logger.warning("JSON repair failed after 4 strategies")
        return None

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities using entity normalizer.

        Prevents duplicates like "Dr. Ahmed Hassan" and "Ahmed Hassan, PhD"
        by normalizing entity names before deduplication.
        """
        seen = {}
        original_count = len(entities)

        for entity in entities:
            entity_text = entity.get("text", "")
            entity_type = entity.get("type", "Unknown")

            # Normalize the entity name
            normalized_name = self.normalizer.normalize_entity_name(entity_text, entity_type)

            # Use normalized name for deduplication key
            key = (normalized_name.lower().strip(), entity_type)

            # Keep entity with highest confidence or update text to normalized form
            if key not in seen or entity.get("confidence", 0) > seen[key].get("confidence", 0):
                # Update entity text to normalized form
                entity["text"] = normalized_name
                entity["original_text"] = entity_text  # Keep original for reference
                seen[key] = entity

        deduplicated = list(seen.values())
        removed_count = original_count - len(deduplicated)

        if removed_count > 0:
            logger.info(f"Entity normalizer removed {removed_count} duplicate entities")

        return deduplicated

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships and normalize entity names.

        Also normalizes source/target entity names to match normalized entity names.
        """
        seen = set()
        unique = []

        for rel in relationships:
            # Validate that relationship has required fields
            if not all(key in rel for key in ["source", "target", "type"]):
                logger.warning(f"Skipping invalid relationship (missing required fields): {rel}")
                continue

            # Normalize source and target names (assume Person type for safety)
            # The exact type doesn't matter much for relationship normalization
            source_normalized = self.normalizer.normalize_entity_name(rel["source"], "Person")
            target_normalized = self.normalizer.normalize_entity_name(rel["target"], "Person")

            # Update relationship with normalized names
            rel["source"] = source_normalized
            rel["target"] = target_normalized

            # Deduplicate using normalized names
            key = (source_normalized.lower().strip(), target_normalized.lower().strip(), rel["type"])
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

            # Check chunk size - if too large, subdivide into 600-token pieces (GraphRAG best practice)
            chunk_tokens = self.count_tokens(chunk_text)

            if chunk_tokens > self.max_tokens_per_request:
                logger.info(f"Chunk {current_chunk}/{total_chunks} is large ({chunk_tokens} tokens, {len(chunk_text)} chars) - subdividing into {self.max_tokens_per_request}-token pieces")
                # Subdivide large chunk into smaller pieces for better extraction
                sub_chunks = self.chunk_text(chunk_text, max_tokens=self.max_tokens_per_request)
                logger.info(f"Subdivided into {len(sub_chunks)} sub-chunks for detailed extraction")
            else:
                logger.info(f"Extracting from chunk {current_chunk}/{total_chunks} ({chunk_tokens} tokens, {len(chunk_text)} chars)")
                sub_chunks = [chunk_text]

            # Update progress in Redis
            self._update_progress(document_id, current_chunk, total_chunks)

            # Extract from each sub-chunk
            chunk_entities = []
            chunk_relationships = []

            for sub_idx, sub_chunk in enumerate(sub_chunks):
                if not sub_chunk.strip():
                    continue

                try:
                    # Extract from this sub-chunk
                    result = self._extract_from_chunk(sub_chunk, language)

                    entities_found = len(result.get("entities", []))
                    relationships_found = len(result.get("relationships", []))

                    if len(sub_chunks) > 1:
                        logger.info(f"  Sub-chunk {sub_idx+1}/{len(sub_chunks)}: Found {entities_found} entities, {relationships_found} relationships")
                    else:
                        logger.info(f"Chunk {current_chunk}: Found {entities_found} entities, {relationships_found} relationships")

                    chunk_entities.extend(result.get("entities", []))
                    chunk_relationships.extend(result.get("relationships", []))

                except (ConnectionError, TimeoutError) as e:
                    # TGI connection/timeout errors - FAIL IMMEDIATELY, don't continue processing
                    logger.error(f"TGI unavailable on chunk {current_chunk}/{total_chunks}: {e}")
                    raise  # Re-raise to stop processing immediately
                except Exception as e:
                    # Check if it's a rate limit error
                    error_str = str(e).lower()
                    if any(term in error_str for term in ["rate limit", "quota", "429", "resource exhausted", "requests per"]):
                        logger.error(f"Rate limit exceeded: {e}")
                        raise Exception(f"Rate limit exceeded: {str(e)}") from e
                    logger.error(f"Error processing sub-chunk {sub_idx+1}/{len(sub_chunks)}: {e}")
                    continue

            # Add chunk results to overall results
            all_entities.extend(chunk_entities)
            all_relationships.extend(chunk_relationships)

            if len(sub_chunks) > 1:
                logger.info(f"Chunk {current_chunk} total: {len(chunk_entities)} entities, {len(chunk_relationships)} relationships from {len(sub_chunks)} sub-chunks")

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
