"""
LLM-Based Entity and Relationship Extractor
Uses LiteLLM to support multiple providers (OpenAI, Claude, Gemini)
with automatic provider detection and token-aware chunking
"""

import os
import json
import requests
import time
from typing import List, Dict, Any, Optional
from loguru import logger

import redis
import tiktoken
from litellm import completion

from ....config import settings
from ..entity_normalizer import EntityNormalizer
from .prompts import get_extraction_prompt, get_validation_prompt
from .json_parser import parse_json_with_repair, clean_json_wrapper, normalize_extraction_result
from .quality_filters import (
    deduplicate_entities,
    deduplicate_relationships,
    filter_quality,
)


class LLMEntityExtractor:
    """
    Extract entities and relationships using LLMs
    Supports: OpenAI, Claude (Anthropic), Gemini (Google), Ollama, TGI
    """

    def __init__(self):
        """Initialize with auto-detected LLM provider"""
        self.provider = None
        self.model = None
        self.max_tokens_per_request = 600  # GraphRAG recommended size
        self.encoding = None
        self.redis_client = None

        self.normalizer = EntityNormalizer()
        logger.info("Entity normalizer initialized")

        self._detect_provider()
        self.encoding = tiktoken.get_encoding("cl100k_base")

        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.redis_client.ping()
        logger.debug("Redis client initialized for progress tracking")

    def _detect_provider(self) -> None:
        """Auto-detect which LLM provider to use"""
        if settings.use_ollama:
            self.ollama_endpoint = settings.ollama_endpoint
            self.ollama_model = settings.ollama_model
            self.provider = "ollama"
            self.model = settings.ollama_model
            logger.info(f"Using Ollama at {self.ollama_endpoint}")
            return

        if settings.use_tgi:
            self.tgi_endpoint = getattr(settings, "entity_extraction_endpoint", None) or settings.tgi_endpoint
            self.provider = "tgi"
            self.model = "tgi"
            logger.info(f"Using TGI at {self.tgi_endpoint}")
            return

        if settings.openai_api_key:
            self.provider = "openai"
            self.model = "gpt-4o-mini"
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            logger.info("Using OpenAI (gpt-4o-mini)")
        elif settings.anthropic_api_key:
            self.provider = "anthropic"
            self.model = "claude-3-haiku-20240307"
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
            logger.info("Using Anthropic Claude (Haiku)")
        elif settings.google_api_key:
            self.provider = "google"
            self.model = "gemini/gemini-2.0-flash-exp"
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
            logger.info("Using Google Gemini 2.0")
        else:
            logger.error("No LLM provider available!")
            self.provider = None

    def _update_progress(self, document_id: str, current_chunk: int, total_chunks: int):
        """Update chunk progress in Redis"""
        if not self.redis_client or not document_id:
            return
        try:
            job_keys = self.redis_client.keys("job:*")
            for job_key in job_keys:
                try:
                    job_data = self.redis_client.get(job_key)
                    if job_data:
                        job_dict = json.loads(job_data)
                        if job_dict.get("document_id") == document_id:
                            if "total_chunks" not in job_dict or job_dict["total_chunks"] == 0:
                                job_dict["total_chunks"] = total_chunks

                            stored_total = job_dict["total_chunks"]
                            chunk_progress = (current_chunk / stored_total) * 30 + 40

                            job_dict["current_chunk"] = current_chunk
                            job_dict["progress"] = int(chunk_progress)

                            content_type = job_dict.get("content_type", "")
                            unit = "pages" if content_type == "pdf" else "chunks"
                            job_dict["current_phase"] = f"Extracting entities ({current_chunk}/{stored_total} {unit})"

                            self.redis_client.setex(job_key, 3600, json.dumps(job_dict))
                            break
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Failed to update chunk progress: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text) // 4

    def chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """Split text into chunks that fit within token limits"""
        if max_tokens is None:
            max_tokens = self.max_tokens_per_request

        total_tokens = self.count_tokens(text)
        if total_tokens <= max_tokens:
            return [text]

        # Split by sentences
        text_marked = text
        for sep in ['. ', '.\n', '! ', '? ', '، ', '؛ ']:
            text_marked = text_marked.replace(sep, sep.rstrip() + '|')
        sentences = [s.strip() for s in text_marked.split('|') if s.strip()]

        if len(sentences) <= 1:
            # No sentence boundaries - split by words
            words = text.split()
            chunks = []
            current_chunk = []
            current_tokens = 0

            for word in words:
                word_tokens = self.count_tokens(word + ' ')
                if current_tokens + word_tokens > max_tokens and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = current_chunk[-50:] if len(current_chunk) > 50 else []
                    current_tokens = self.count_tokens(' '.join(current_chunk))
                current_chunk.append(word)
                current_tokens += word_tokens

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks

        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_sentences = 2

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"Split text into {len(chunks)} chunks (total tokens: {total_tokens})")
        return chunks

    def extract_entities_and_relationships(
        self, text: str, language: str = "auto", document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract entities and relationships from text"""
        if not self.provider:
            logger.error("No LLM provider available")
            return {"entities": [], "relationships": []}

        if language == "auto":
            arabic_ratio = len([c for c in text if '\u0600' <= c <= '\u06FF']) / max(len(text), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        chunks = self.chunk_text(text)
        total_chunks = len(chunks)

        all_entities = []
        all_relationships = []

        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            logger.info(f"Processing chunk {current_chunk}/{total_chunks}")
            self._update_progress(document_id, current_chunk, total_chunks)

            try:
                result = self._extract_from_chunk(chunk, language)
                all_entities.extend(result.get("entities", []))
                all_relationships.extend(result.get("relationships", []))
            except Exception as e:
                error_str = str(e).lower()
                if any(term in error_str for term in ["rate limit", "quota", "429"]):
                    raise
                logger.error(f"Error processing chunk {current_chunk}: {e}")
                continue

        entities = deduplicate_entities(all_entities, self.normalizer)
        relationships = deduplicate_relationships(all_relationships, self.normalizer)
        entities, relationships = filter_quality(entities, relationships)

        logger.info(f"Final: {len(entities)} entities, {len(relationships)} relationships")
        return {"entities": entities, "relationships": relationships}

    def _extract_from_chunk(self, chunk: str, language: str) -> Dict[str, Any]:
        """Extract entities and relationships from a single chunk"""
        system_prompt, user_template = get_extraction_prompt(language)
        user_prompt = user_template.format(chunk=chunk)

        try:
            content = self._call_llm(system_prompt, user_prompt)
            content = clean_json_wrapper(content)
            result = parse_json_with_repair(content)
            result = normalize_extraction_result(result)

            # Add confidence scores
            for entity in result.get("entities", []):
                if isinstance(entity, dict):
                    entity["confidence"] = 0.9 if entity.get("importance") == "high" else 0.7
            for rel in result.get("relationships", []):
                if isinstance(rel, dict):
                    rel["confidence"] = 0.8

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            error_str = str(e).lower()
            if any(term in error_str for term in ["rate limit", "quota", "429"]):
                raise
            logger.error(f"Error calling LLM: {e}")
            return {"entities": [], "relationships": []}

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM provider and return response content"""
        if self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        elif self.provider == "tgi":
            return self._call_tgi(system_prompt, user_prompt)
        else:
            return self._call_litellm(system_prompt, user_prompt)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama API"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_endpoint}/v1/chat/completions",
                    json={
                        "model": self.ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    },
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError(f"Ollama not available at {self.ollama_endpoint}")
            except requests.exceptions.Timeout:
                raise TimeoutError("Ollama request timed out")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 422 and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise

    def _call_tgi(self, system_prompt: str, user_prompt: str) -> str:
        """Call TGI API"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.tgi_endpoint}/v1/chat/completions",
                    json={
                        "model": "tgi",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.ConnectionError:
                raise ConnectionError(f"TGI not available at {self.tgi_endpoint}")
            except requests.exceptions.Timeout:
                raise TimeoutError("TGI request timed out")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 422 and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise

    def _call_litellm(self, system_prompt: str, user_prompt: str) -> str:
        """Call cloud LLM via LiteLLM"""
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def validate_entities_with_llm(
        self, entities: List[Dict[str, Any]], language: str = "auto"
    ) -> List[Dict[str, Any]]:
        """Use LLM to validate entities and filter false positives"""
        if not entities or not self.provider:
            return entities

        if language == "auto":
            sample_text = " ".join([e.get("text", "") for e in entities[:5]])
            arabic_ratio = len([c for c in sample_text if '\u0600' <= c <= '\u06FF']) / max(len(sample_text), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        batch_size = 20
        validated_entities = []

        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            entity_names = [e.get("text", "") for e in batch]

            system_prompt, user_template = get_validation_prompt(language)
            user_prompt = user_template.format(entities=json.dumps(entity_names, ensure_ascii=False))

            try:
                content = self._call_llm(system_prompt, user_prompt)
                result = parse_json_with_repair(content)

                if result and "valid" in result:
                    valid_names = set(result["valid"])
                    for entity in batch:
                        if entity.get("text", "") in valid_names:
                            validated_entities.append(entity)
                else:
                    validated_entities.extend(batch)

            except Exception as e:
                logger.warning(f"LLM validation failed, keeping batch: {e}")
                validated_entities.extend(batch)

        logger.info(f"LLM validation: {len(entities)} -> {len(validated_entities)} entities")
        return validated_entities

    def extract_from_chunks(
        self, chunks: List[Dict[str, Any]], language: Optional[str] = None, document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract entities from document chunks"""
        if not self.provider:
            logger.error("No LLM provider available")
            return {"entities": [], "relationships": []}

        if language is None or language == "auto":
            first_text = chunks[0].get("text", "") if chunks else ""
            arabic_ratio = len([c for c in first_text if '\u0600' <= c <= '\u06FF']) / max(len(first_text), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        logger.info(f"Processing {len(chunks)} chunks (language: {language})")

        all_entities = []
        all_relationships = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            chunk_text = chunk.get("text", "")
            if not chunk_text.strip():
                continue

            chunk_tokens = self.count_tokens(chunk_text)
            if chunk_tokens > self.max_tokens_per_request:
                sub_chunks = self.chunk_text(chunk_text, max_tokens=self.max_tokens_per_request)
            else:
                sub_chunks = [chunk_text]

            self._update_progress(document_id, current_chunk, total_chunks)

            for sub_idx, sub_chunk in enumerate(sub_chunks):
                if not sub_chunk.strip():
                    continue
                try:
                    result = self._extract_from_chunk(sub_chunk, language)
                    all_entities.extend(result.get("entities", []))
                    all_relationships.extend(result.get("relationships", []))
                except (ConnectionError, TimeoutError):
                    raise
                except Exception as e:
                    error_str = str(e).lower()
                    if any(term in error_str for term in ["rate limit", "quota", "429"]):
                        raise
                    logger.error(f"Error processing sub-chunk: {e}")
                    continue

        entities = deduplicate_entities(all_entities, self.normalizer)
        relationships = deduplicate_relationships(all_relationships, self.normalizer)
        entities, relationships = filter_quality(entities, relationships)

        logger.info(f"Final: {len(entities)} entities, {len(relationships)} relationships")
        return {"entities": entities, "relationships": relationships}
