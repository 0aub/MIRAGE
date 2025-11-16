"""
Content Rewriter Service
Rewrites and cleans content chunks using LLM with progress tracking
"""

import os
import json
import requests
import time
from typing import List, Dict, Any, Optional
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

from ...config import settings
from ...config.config_loader import get_config_loader


class ContentRewriter:
    """
    Rewrite and clean content chunks using LLM
    Tracks progress in Redis for UI updates
    """

    def __init__(self):
        """Initialize with auto-detected LLM provider"""
        # Load configuration
        config_loader = get_config_loader()

        self.provider = None
        self.model = None
        self.redis_client = None

        # Load LLM parameters from config
        llm_params = config_loader.get_llm_parameters()
        self.temperature = llm_params.get("temperature", 0.3)
        self.max_tokens = llm_params.get("max_tokens", 2000)

        # Auto-detect available provider from config
        self._detect_provider()

        # Initialize Redis for progress tracking
        if REDIS_AVAILABLE:
            try:
                redis_config = config_loader.get_redis_config()
                redis_url = redis_config.get("url", settings.redis_url)
                self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.debug("Redis client initialized for rewriting progress tracking")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis for progress tracking: {e}")
                self.redis_client = None

    def _detect_provider(self) -> None:
        """Auto-detect which LLM provider to use based on configuration and available API keys"""
        if not LITELLM_AVAILABLE:
            logger.error("LiteLLM not installed. Install with: pip install litellm")
            return

        # Load configuration
        config_loader = get_config_loader()
        enabled_provider = config_loader.get_enabled_llm_provider()

        if not enabled_provider:
            logger.error("No LLM provider enabled! Please enable a provider in models.yaml")
            self.provider = None
            return

        provider_name = enabled_provider["name"]
        provider_config = enabled_provider

        # Handle TGI provider
        if provider_name == "tgi":
            if settings.use_tgi and settings.tgi_endpoint:
                self.provider = "tgi"
                self.model = provider_config.get("model", "tgi")
                self.tgi_endpoint = provider_config.get("endpoint", settings.tgi_endpoint)
                logger.info(f"Using TGI at {self.tgi_endpoint} for content rewriting (from config)")
                return
            else:
                logger.warning("TGI provider enabled in config but settings.use_tgi=False or no endpoint")

        # Handle cloud providers
        elif provider_name == "openai":
            if settings.openai_api_key:
                self.provider = "openai"
                self.model = provider_config.get("model", "gpt-4o-mini")
                os.environ["OPENAI_API_KEY"] = settings.openai_api_key
                logger.info(f"Using OpenAI ({self.model}) for content rewriting (from config)")
                return
            else:
                logger.warning("OpenAI provider enabled in config but no OPENAI_API_KEY set")

        elif provider_name == "anthropic":
            if settings.anthropic_api_key:
                self.provider = "anthropic"
                self.model = provider_config.get("model", "claude-3-haiku-20240307")
                os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
                logger.info(f"Using Anthropic ({self.model}) for content rewriting (from config)")
                return
            else:
                logger.warning("Anthropic provider enabled in config but no ANTHROPIC_API_KEY set")

        elif provider_name == "google":
            if settings.google_api_key:
                self.provider = "google"
                self.model = provider_config.get("model", "gemini/gemini-2.0-flash-exp")
                os.environ["GOOGLE_API_KEY"] = settings.google_api_key
                logger.info(f"Using Google ({self.model}) for content rewriting (from config)")
                return
            else:
                logger.warning("Google provider enabled in config but no GOOGLE_API_KEY set")

        logger.error(f"Enabled provider '{provider_name}' not properly configured with API keys")
        self.provider = None

    def _update_progress(self, document_id: str, current_chunk: int, total_chunks: int, phase: str = "rewriting"):
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
                status_data["phase"] = phase
                # Update with same TTL
                self.redis_client.setex(key, 3600, json.dumps(status_data))
        except Exception as e:
            logger.debug(f"Failed to update rewriting progress: {e}")

    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for sub-chunking"""
        # Simple sentence splitting - handles both Arabic and English
        import re
        # Split on common sentence terminators
        sentences = re.split(r'([.!?؟۔।]+[\s\n])', text)
        # Recombine terminators with sentences
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            if sentence.strip():
                result.append(sentence)
        # Add last sentence if no terminator
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return result

    def _rewrite_single_piece(self, text: str, language: str, system_prompt: str) -> str:
        """Rewrite a single piece of text (internal helper)"""
        user_prompt = f"""النص: {text}

النص المُحسّن:""" if language == "ar" else f"""Text: {text}

Improved text:"""

        try:
            # Handle TGI separately
            if self.provider == "tgi":
                tgi_response = requests.post(
                    f"{self.tgi_endpoint}/v1/chat/completions",
                    json={
                        "model": "tgi",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=180
                )

                # Check response status and log detailed error
                if tgi_response.status_code != 200:
                    error_body = tgi_response.text
                    logger.error(f"TGI returned {tgi_response.status_code}: {error_body}")
                    raise requests.HTTPError(f"{tgi_response.status_code} {tgi_response.reason} - {error_body}")

                rewritten_text = tgi_response.json()["choices"][0]["message"]["content"]
            else:
                # Use LiteLLM for cloud providers
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                rewritten_text = response.choices[0].message.content

            return rewritten_text.strip()

        except Exception as e:
            logger.error(f"Error rewriting piece: {e}")
            raise RuntimeError(f"Content rewriting failed: {str(e)}") from e

    def rewrite_chunk(self, chunk_text: str, language: str = "auto") -> str:
        """
        Rewrite a single chunk using LLM
        Automatically splits large chunks to fit within token limits

        Args:
            chunk_text: Text to rewrite
            language: Language code ("ar", "en", or "auto")

        Returns:
            Rewritten text
        """
        if not self.provider:
            raise RuntimeError("No LLM provider available for content rewriting. Please configure TGI or set an API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)")

        # Detect language if needed
        if language == "auto":
            arabic_ratio = len([c for c in chunk_text if '\u0600' <= c <= '\u06FF']) / max(len(chunk_text), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        # Create language-specific prompt
        if language == "ar":
            system_prompt = """أنت خبير في تحسين وتنظيف المحتوى النصي.

مهمتك:
- تحسين صياغة النص وتنظيفه
- إزالة الكلام المتكرر والحشو
- تصحيح الأخطاء النحوية والإملائية
- الحفاظ على جميع المعلومات والحقائق الواردة
- عدم إضافة معلومات جديدة غير موجودة
- إعادة كتابة النص بشكل واضح ومفهوم

أعد كتابة النص المعطى فقط، بدون أي تعليقات أو شرح."""

        else:  # English
            system_prompt = """You are an expert in content improvement and cleaning.

Your task:
- Improve text clarity and readability
- Remove repetition and filler words
- Fix grammar and spelling errors
- Preserve all information and facts
- Do not add new information not present in the original
- Rewrite the text clearly and concisely

Return only the rewritten text, without any comments or explanations."""

        # Check token count for TGI (ALLaM-7B has 2048 token limit)
        if self.provider == "tgi":
            # Load model constraints from config
            config_loader = get_config_loader()
            constraints = config_loader.get_model_constraints("tgi_allam_7b")

            # Calculate max allowed text size based on prompt overhead and constraints
            prompt_template_length = len(system_prompt) + 100  # Add buffer for user prompt template
            max_input_tokens = constraints.get("max_input_tokens", 2048)
            reserved_response_tokens = constraints.get("reserved_response_tokens", 500)
            chars_per_token = constraints.get("chars_per_token", 4)

            max_tokens = max_input_tokens - reserved_response_tokens
            max_text_chars = (max_tokens * chars_per_token) - prompt_template_length

            # If chunk exceeds limit, split into sub-chunks and rewrite each
            if len(chunk_text) > max_text_chars:
                logger.info(f"Chunk too large ({len(chunk_text)} chars, max {max_text_chars}). Splitting into sub-chunks for rewriting.")

                # Split text into sentences
                sentences = self._split_text_into_sentences(chunk_text)

                # Group sentences into sub-chunks that fit within limit
                sub_chunks = []
                current_sub_chunk = []
                current_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence)
                    if current_length + sentence_length > max_text_chars and current_sub_chunk:
                        # Current sub-chunk is full, save it and start new one
                        sub_chunks.append(" ".join(current_sub_chunk))
                        current_sub_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        # Add sentence to current sub-chunk
                        current_sub_chunk.append(sentence)
                        current_length += sentence_length

                # Add last sub-chunk
                if current_sub_chunk:
                    sub_chunks.append(" ".join(current_sub_chunk))

                logger.info(f"Split into {len(sub_chunks)} sub-chunks for rewriting")

                # Rewrite each sub-chunk and combine
                rewritten_sub_chunks = []
                for i, sub_chunk in enumerate(sub_chunks):
                    logger.info(f"Rewriting sub-chunk {i+1}/{len(sub_chunks)}")
                    rewritten = self._rewrite_single_piece(sub_chunk, language, system_prompt)
                    rewritten_sub_chunks.append(rewritten)

                # Combine rewritten sub-chunks
                return "\n\n".join(rewritten_sub_chunks)

        # Chunk fits within limit - rewrite normally
        return self._rewrite_single_piece(chunk_text, language, system_prompt)

    def rewrite_chunks(
        self,
        chunks: List[Dict[str, Any]],
        language: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Rewrite document chunks one by one with progress tracking

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            language: Optional language override
            document_id: Optional document ID for progress tracking

        Returns:
            List of rewritten chunks with same structure
        """
        if not self.provider:
            raise RuntimeError("No LLM provider available for content rewriting. Please configure TGI or set an API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)")

        total_chunks = len(chunks)
        logger.info(f"Starting rewriting of {total_chunks} chunks")

        rewritten_chunks = []

        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            logger.info(f"Rewriting chunk {current_chunk}/{total_chunks}")

            # Update progress in Redis
            self._update_progress(document_id, current_chunk, total_chunks, phase="rewriting")

            # Rewrite the chunk text
            original_text = chunk.get("text", "")
            rewritten_text = self.rewrite_chunk(original_text, language or "auto")

            # DEBUG: Check if text actually changed
            if rewritten_text == original_text:
                logger.warning(f"Chunk {current_chunk}: Rewritten text is IDENTICAL to original (length={len(original_text)})")
            else:
                logger.info(f"Chunk {current_chunk}: Text changed (orig={len(original_text)} chars, new={len(rewritten_text)} chars)")

            # Create new chunk with rewritten text but same metadata
            rewritten_chunk = {
                "text": rewritten_text,
                "metadata": chunk.get("metadata", {})
            }
            rewritten_chunks.append(rewritten_chunk)

        logger.info(f"Completed rewriting {total_chunks} chunks")
        return rewritten_chunks
