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

# Required imports - no fallback (enforced in Docker)
import redis
import litellm
from litellm import completion
import tiktoken

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

        # Initialize token counter (required)
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize Redis for progress tracking (required)
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.redis_client.ping()
        logger.debug("Redis client initialized for chunk progress tracking")

    def _detect_provider(self) -> None:
        """Auto-detect which LLM provider to use based on available API keys"""

        # Check for Ollama (local) first - highest priority if enabled
        if settings.use_ollama:
            self.ollama_endpoint = settings.ollama_endpoint
            self.ollama_model = settings.ollama_model
            self.provider = "ollama"
            self.model = settings.ollama_model
            logger.info(f"Using Ollama at {self.ollama_endpoint} with model {self.ollama_model} for entity extraction")
            return

        # Check for TGI (local GPU) second - high priority if enabled
        if settings.use_tgi:
            # Prefer dedicated extraction endpoint if available
            if hasattr(settings, "entity_extraction_endpoint") and settings.entity_extraction_endpoint:
                self.tgi_endpoint = settings.entity_extraction_endpoint
                logger.info(f"Using dedicated TGI endpoint at {self.tgi_endpoint} for entity extraction (Qwen)")
            elif settings.tgi_endpoint:
                self.tgi_endpoint = settings.tgi_endpoint
                logger.info(f"Using shared TGI endpoint at {self.tgi_endpoint} for entity extraction")

            self.provider = "tgi"
            self.model = "tgi"  # TGI doesn't need model specification
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
        """Update chunk progress in Redis and job manager"""
        if not self.redis_client or not document_id:
            return
        try:
            # Find and update the job associated with this document
            job_keys = self.redis_client.keys("job:*")
            for job_key in job_keys:
                try:
                    job_data = self.redis_client.get(job_key)
                    if job_data:
                        job_dict = json.loads(job_data)
                        if job_dict.get("document_id") == document_id:
                            # Found the job! Update it with chunk progress
                            # Set total_chunks only if not already set (to prevent it from changing)
                            if "total_chunks" not in job_dict or job_dict["total_chunks"] == 0:
                                job_dict["total_chunks"] = total_chunks

                            # Use the stored total_chunks for consistent progress calculation
                            stored_total = job_dict["total_chunks"]
                            chunk_progress = (current_chunk / stored_total) * 30 + 40  # 40-70%

                            job_dict["current_chunk"] = current_chunk
                            job_dict["progress"] = int(chunk_progress)

                            # Use "page" for PDFs, "chunk" for other content types
                            content_type = job_dict.get("content_type", "")
                            unit = "pages" if content_type == "pdf" else "chunks"
                            job_dict["current_phase"] = f"Extracting entities ({current_chunk}/{stored_total} {unit})"

                            # Save back to Redis with TTL
                            self.redis_client.setex(job_key, 3600, json.dumps(job_dict))
                            logger.debug(f"Updated job progress: {unit} {current_chunk}/{total_chunks} ({int(chunk_progress)}%)")
                            break
                except Exception:
                    continue
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
        # Domain-agnostic prompt for multi-domain knowledge graph extraction
        if language == "ar":
            system_prompt = """أنت مستخرج معرفة متخصص لبناء رسم بياني معرفي (Knowledge Graph).
مهمتك: استخراج الكيانات المسماة والعلاقات بينها. العلاقات مهمة جداً!

**أنواع الكيانات:**
- Organization: منظمات، شركات، مؤسسات، جهات حكومية
- Person: أشخاص، مناصب، أدوار
- Location: أماكن، مدن، دول، مناطق
- Concept: مفاهيم، مصطلحات، تعريفات
- Event: أحداث، مناسبات، تواريخ مهمة
- Product: منتجات، خدمات، أنظمة
- Document: وثائق، سياسات، قوانين، تقارير

**أنواع العلاقات (مهم جداً!):**
- RELATED_TO: علاقة عامة بين كيانين
- PART_OF: كيان جزء من كيان أكبر
- CONTAINS: كيان يحتوي على كيان آخر
- CAUSES: كيان يسبب شيء آخر
- USES: كيان يستخدم كيان آخر
- CREATED_BY: كيان أنشأه كيان آخر
- LOCATED_IN: كيان موجود في مكان
- BELONGS_TO: كيان ينتمي لكيان آخر
- DEPENDS_ON: كيان يعتمد على كيان آخر
- AFFECTS: كيان يؤثر على كيان آخر
- أو أي علاقة وصفية مناسبة للسياق

**صيغة JSON:**
{
  "entities": [{"text": "اسم", "type": "النوع", "importance": "high/medium/low"}],
  "relationships": [{"source": "المصدر", "target": "الهدف", "type": "RELATIONSHIP_TYPE", "weight": 0.8}]
}

مهم: استخرج علاقات بين الكيانات! لا تستخرج كيانات بدون علاقات."""

            user_prompt = f"""استخرج الكيانات والعلاقات بينها:

{chunk}

JSON:"""

        else:  # English
            system_prompt = """You are a knowledge extractor for building a Knowledge Graph.
Extract named entities AND relationships between them. Relationships are critical!

**Entity Types:**
- Organization: Companies, institutions, agencies, departments
- Person: People, roles, positions, titles
- Location: Places, cities, countries, regions
- Concept: Ideas, terms, definitions, topics
- Event: Events, dates, milestones, occurrences
- Product: Products, services, systems, tools
- Document: Documents, policies, laws, reports, papers

**Relationship Types (CRITICAL!):**
- RELATED_TO: General association between entities
- PART_OF: Entity is component/member of another
- CONTAINS: Entity contains another entity
- CAUSES: Entity causes/leads to something
- USES: Entity uses/utilizes another
- CREATED_BY: Entity was created by another
- LOCATED_IN: Entity is located in a place
- BELONGS_TO: Entity belongs to another
- DEPENDS_ON: Entity depends on another
- AFFECTS: Entity affects/impacts another
- WORKS_FOR: Person works for organization
- OWNS: Entity owns another entity
- PRODUCES: Entity produces/creates something
- Or any descriptive relationship fitting the context

**JSON format:**
{
  "entities": [{"text": "Name", "type": "Type", "importance": "high/medium/low"}],
  "relationships": [{"source": "Source", "target": "Target", "type": "RELATIONSHIP_TYPE", "weight": 0.8}]
}

IMPORTANT: Extract relationships between entities! Do not return entities without relationships."""

            user_prompt = f"""Extract entities AND relationships:

{chunk}

JSON:"""

        try:
            # Handle Ollama and TGI separately (both use OpenAI-compatible /v1/chat/completions)
            if self.provider == "ollama":
                # Ollama uses OpenAI-compatible API
                max_retries = 3
                retry_delay = 2

                for attempt in range(max_retries):
                    try:
                        ollama_response = requests.post(
                            f"{self.ollama_endpoint}/v1/chat/completions",
                            json={
                                "model": self.ollama_model,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                "temperature": 0.1,  # Very low for consistency and valid JSON
                                "max_tokens": 4096,  # Increased to prevent truncation
                                "top_p": 0.9,  # Nucleus sampling for better quality
                            },
                            timeout=60  # Ollama can be slower on first request (model loading)
                        )
                        ollama_response.raise_for_status()
                        content = ollama_response.json()["choices"][0]["message"]["content"]
                        break  # Success!

                    except requests.exceptions.ConnectionError as e:
                        # Ollama is not running or not reachable - FAIL IMMEDIATELY
                        logger.error(f"Ollama connection failed - Ollama may not be running: {e}")
                        raise ConnectionError(
                            f"Ollama is not available at {self.ollama_endpoint}. "
                            "Please start Ollama with: docker compose up -d ollama"
                        )
                    except requests.exceptions.Timeout as e:
                        # Ollama is too slow or overloaded - FAIL IMMEDIATELY
                        logger.error(f"Ollama request timed out after 60 seconds: {e}")
                        raise TimeoutError(
                            f"Ollama request timed out. The model may be loading or overloaded. "
                            "Check Ollama logs: docker logs mirage-ollama"
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
                            logger.error(f"Ollama HTTP error {e.response.status_code}: {e}")
                            raise
                    except Exception as e:
                        # Unexpected errors - FAIL IMMEDIATELY
                        logger.error(f"Unexpected Ollama error: {type(e).__name__}: {e}")
                        raise

            elif self.provider == "tgi":
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

            # Handle case where LLM returns a list instead of dict
            # This can happen with some models returning just the entities array
            if isinstance(result, list):
                # Check if it looks like entities (list of dicts with text/type)
                if all(isinstance(item, dict) for item in result):
                    result = {"entities": result, "relationships": []}
                else:
                    logger.warning(f"LLM returned unexpected list format: {result[:100]}")
                    return {"entities": [], "relationships": []}

            # Ensure result is a dict with expected keys
            if not isinstance(result, dict):
                logger.warning(f"LLM returned unexpected type: {type(result)}")
                return {"entities": [], "relationships": []}

            # Handle nested entities (some models return {"entities": {"entities": [...]}})
            entities = result.get("entities", [])
            if isinstance(entities, dict) and "entities" in entities:
                entities = entities.get("entities", [])
            elif not isinstance(entities, list):
                entities = []

            relationships = result.get("relationships", [])
            if not isinstance(relationships, list):
                relationships = []

            result = {"entities": entities, "relationships": relationships}

            # Add confidence scores
            for entity in result.get("entities", []):
                if isinstance(entity, dict):
                    entity["confidence"] = 0.9 if entity.get("importance") == "high" else 0.7

            for rel in result.get("relationships", []):
                if isinstance(rel, dict):
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

        GraphRAG Enhancement: Merges descriptions from duplicates to create richer entity profiles.
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

            if key not in seen:
                # First occurrence - store it
                entity["text"] = normalized_name
                entity["original_text"] = entity_text  # Keep original for reference
                seen[key] = entity
            else:
                # Duplicate found - merge information
                existing = seen[key]

                # Keep higher confidence
                if entity.get("confidence", 0) > existing.get("confidence", 0):
                    existing["confidence"] = entity.get("confidence", 0)

                # Merge descriptions (GraphRAG: combine descriptions for richer context)
                new_desc = entity.get("description", "")
                existing_desc = existing.get("description", "")
                if new_desc and new_desc not in existing_desc:
                    if existing_desc:
                        # Append new description if it adds information
                        if len(existing_desc) < 500:  # Avoid overly long descriptions
                            existing["description"] = f"{existing_desc} {new_desc}"
                    else:
                        existing["description"] = new_desc

                # Merge attributes
                new_attrs = entity.get("attributes", {})
                existing_attrs = existing.get("attributes", {})
                existing["attributes"] = {**new_attrs, **existing_attrs}

                # Update importance if higher
                importance_order = {"high": 3, "medium": 2, "low": 1}
                new_importance = importance_order.get(entity.get("importance", "low"), 1)
                existing_importance = importance_order.get(existing.get("importance", "low"), 1)
                if new_importance > existing_importance:
                    existing["importance"] = entity.get("importance")

        deduplicated = list(seen.values())
        removed_count = original_count - len(deduplicated)

        if removed_count > 0:
            logger.info(f"Entity normalizer removed {removed_count} duplicate entities (merged descriptions)")

        return deduplicated

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships and normalize entity names.

        GraphRAG Enhancement: Merges descriptions and weights from duplicate relationships.
        """
        seen = {}
        unique = []

        for rel in relationships:
            # Validate that relationship has required fields
            if not all(key in rel for key in ["source", "target", "type"]):
                logger.warning(f"Skipping invalid relationship (missing required fields): {rel}")
                continue

            # Normalize source and target names (assume Person type for safety)
            source_normalized = self.normalizer.normalize_entity_name(rel["source"], "Person")
            target_normalized = self.normalizer.normalize_entity_name(rel["target"], "Person")

            # Update relationship with normalized names
            rel["source"] = source_normalized
            rel["target"] = target_normalized

            # Deduplicate using normalized names
            key = (source_normalized.lower().strip(), target_normalized.lower().strip(), rel["type"].lower())

            if key not in seen:
                # First occurrence
                seen[key] = rel
                unique.append(rel)
            else:
                # Duplicate found - merge information
                existing = seen[key]

                # Merge descriptions
                new_desc = rel.get("description", "")
                existing_desc = existing.get("description", "")
                if new_desc and new_desc not in existing_desc:
                    if existing_desc:
                        if len(existing_desc) < 300:
                            existing["description"] = f"{existing_desc} {new_desc}"
                    else:
                        existing["description"] = new_desc

                # Keep higher weight
                new_weight = rel.get("weight", 0.5)
                existing_weight = existing.get("weight", 0.5)
                if new_weight > existing_weight:
                    existing["weight"] = new_weight

                # Merge attributes
                new_attrs = rel.get("attributes", {})
                existing_attrs = existing.get("attributes", {})
                existing["attributes"] = {**new_attrs, **existing_attrs}

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

        # NO static blacklists - the LLM decides dynamically what is a valid entity
        # The extraction prompt instructs the LLM to only extract specific named entities
        # This approach is more flexible and doesn't require manual maintenance

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
        # NO static blacklists - the LLM decides dynamically what relationships to extract
        # The extraction prompt instructs the LLM to only extract specific relationship types

        # Only filter self-referential relationships (source == target)
        source = relationship.get("source", "").lower().strip()
        target = relationship.get("target", "").lower().strip()
        if source == target:
            logger.debug(f"Filtering self-referential relationship: {source} -> {target}")
            return True

        return False

    def validate_entities_with_llm(
        self,
        entities: List[Dict[str, Any]],
        language: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to dynamically validate entities and filter false positives.
        This provides smarter filtering than static blacklists.

        Args:
            entities: List of entities to validate
            language: Language code ("ar", "en", or "auto")

        Returns:
            List of validated entities (false positives removed)
        """
        if not entities or not self.provider:
            return entities

        # Detect language
        if language == "auto":
            sample_text = " ".join([e.get("text", "") for e in entities[:5]])
            arabic_ratio = len([c for c in sample_text if '\u0600' <= c <= '\u06FF']) / max(len(sample_text), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        # Batch entities for validation (max 20 at a time)
        batch_size = 20
        validated_entities = []

        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            entity_names = [e.get("text", "") for e in batch]

            # Create validation prompt
            if language == "ar":
                system_prompt = """أنت مدقق كيانات. راجع قائمة الكيانات وحدد أيها كيانات حقيقية (أسماء، منظمات، أماكن، برامج) وأيها مصطلحات عامة أو عبارات شائعة.

أجب بـ JSON فقط: {"valid": ["كيان1", "كيان2"], "invalid": ["مصطلح عام1"]}

الكيانات العامة التي يجب رفضها:
- العبارات الدينية (باذن الله، إن شاء الله)
- المصطلحات العامة (البيانات، التشغيل، السياسات)
- الكلمات الوصفية المجردة"""

                user_prompt = f"""راجع هذه الكيانات وصنفها:

{json.dumps(entity_names, ensure_ascii=False)}

JSON:"""
            else:
                system_prompt = """You are an entity validator. Review the list of entities and identify which are real named entities (names, organizations, places, programs) and which are generic terms or common phrases.

Respond with JSON only: {"valid": ["entity1", "entity2"], "invalid": ["generic term1"]}

Generic entities to reject:
- Religious/cultural phrases
- Generic operational terms (data, system, process, policy)
- Abstract descriptive words"""

                user_prompt = f"""Review these entities and classify them:

{json.dumps(entity_names, ensure_ascii=False)}

JSON:"""

            try:
                if self.provider == "ollama":
                    response = requests.post(
                        f"{self.ollama_endpoint}/v1/chat/completions",
                        json={
                            "model": self.ollama_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "temperature": 0.1,
                            "max_tokens": 1024,
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                elif self.provider == "tgi":
                    response = requests.post(
                        f"{self.tgi_endpoint}/v1/chat/completions",
                        json={
                            "model": "tgi",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "temperature": 0.1,
                            "max_tokens": 1024,
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                else:
                    response = completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1024,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content

                # Parse response
                result = self._parse_json_with_repair(content)
                if result and "valid" in result:
                    valid_names = set(result["valid"])
                    for entity in batch:
                        if entity.get("text", "") in valid_names:
                            validated_entities.append(entity)
                        else:
                            logger.debug(f"LLM filtered entity: {entity.get('text')}")
                else:
                    # If parsing fails, keep all entities from this batch
                    validated_entities.extend(batch)

            except Exception as e:
                logger.warning(f"LLM validation failed, keeping batch: {e}")
                validated_entities.extend(batch)

        logger.info(f"LLM validation: {len(entities)} -> {len(validated_entities)} entities")
        return validated_entities

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
