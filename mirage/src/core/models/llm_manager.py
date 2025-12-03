"""
MIRAGE V2 LLM Manager
Handles LLM inference through Text Generation Inference (TGI).

Features:
- TGI client for local LLM inference
- Configurable generation parameters
- Streaming support
- Token counting and context management
- Error handling and retries
"""

import os
import time
import asyncio
from typing import Optional, Dict, Any, List, Generator, AsyncGenerator, Union
from dataclasses import dataclass
from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: Optional[List[str]] = None
    do_sample: bool = True


@dataclass
class GenerationResult:
    """Result from text generation"""
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_ms: float
    model_id: str


class LLMManager:
    """
    Manages LLM inference through TGI.

    Responsibilities:
    - Connect to TGI endpoint
    - Generate text with configurable parameters
    - Handle streaming responses
    - Manage token limits
    - Provide retry logic
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        default_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize LLM manager.

        Args:
            endpoint: TGI endpoint URL
            model_id: Model identifier (for logging)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            default_config: Default generation configuration
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not installed. Install with: pip install httpx")

        self.endpoint = endpoint or os.environ.get("TGI_ENDPOINT", "http://tgi:8765")
        self.model_id = model_id or os.environ.get("LLM_MODEL_ID", "ALLaM-7B")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_config = default_config or GenerationConfig()

        # HTTP client
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

        # Token counting
        self._tokenizer = None
        self._chars_per_token = 4  # Fallback estimate

        # Statistics
        self._total_requests = 0
        self._total_errors = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        logger.info(
            f"LLMManager initialized: endpoint={self.endpoint}, "
            f"model={self.model_id}, timeout={timeout}s"
        )

    def _get_client(self) -> httpx.Client:
        """Get or create sync HTTP client"""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.endpoint,
                timeout=self.timeout
            )
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.endpoint,
                timeout=self.timeout
            )
        return self._async_client

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        if TIKTOKEN_AVAILABLE:
            try:
                if self._tokenizer is None:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                return len(self._tokenizer.encode(text))
            except Exception:
                pass

        # Fallback: character-based estimate
        return len(text) // self._chars_per_token

    def _prepare_request(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Prepare TGI request payload.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Request payload dictionary
        """
        cfg = config or self.default_config

        # TGI generate endpoint format
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repetition_penalty": cfg.repetition_penalty,
                "do_sample": cfg.do_sample,
                "return_full_text": False,
            }
        }

        if cfg.stop_sequences:
            payload["parameters"]["stop"] = cfg.stop_sequences

        return payload

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        retry: bool = True
    ) -> GenerationResult:
        """
        Generate text from prompt (synchronous).

        Args:
            prompt: Input prompt
            config: Generation configuration (uses default if not provided)
            retry: Enable automatic retries

        Returns:
            GenerationResult with generated text and metadata
        """
        client = self._get_client()
        payload = self._prepare_request(prompt, config)

        input_tokens = self._estimate_tokens(prompt)
        start_time = time.time()

        last_error = None
        attempts = self.max_retries if retry else 1

        for attempt in range(attempts):
            try:
                self._total_requests += 1

                response = client.post("/generate", json=payload)
                response.raise_for_status()

                data = response.json()

                # Parse TGI response
                generated_text = data.get("generated_text", "")
                output_tokens = self._estimate_tokens(generated_text)

                latency_ms = (time.time() - start_time) * 1000

                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                return GenerationResult(
                    text=generated_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason=data.get("details", {}).get("finish_reason", "stop"),
                    latency_ms=latency_ms,
                    model_id=self.model_id
                )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"TGI timeout (attempt {attempt + 1}/{attempts})")
                self._total_errors += 1

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"TGI HTTP error: {e.response.status_code}")
                self._total_errors += 1

                # Don't retry on 4xx errors
                if 400 <= e.response.status_code < 500:
                    break

            except Exception as e:
                last_error = e
                logger.error(f"TGI error: {e}")
                self._total_errors += 1

            # Wait before retry
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"Failed to generate after {attempts} attempts: {last_error}")

    async def agenerate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        retry: bool = True
    ) -> GenerationResult:
        """
        Generate text from prompt (asynchronous).

        Args:
            prompt: Input prompt
            config: Generation configuration
            retry: Enable automatic retries

        Returns:
            GenerationResult with generated text and metadata
        """
        client = self._get_async_client()
        payload = self._prepare_request(prompt, config)

        input_tokens = self._estimate_tokens(prompt)
        start_time = time.time()

        last_error = None
        attempts = self.max_retries if retry else 1

        for attempt in range(attempts):
            try:
                self._total_requests += 1

                response = await client.post("/generate", json=payload)
                response.raise_for_status()

                data = response.json()

                generated_text = data.get("generated_text", "")
                output_tokens = self._estimate_tokens(generated_text)

                latency_ms = (time.time() - start_time) * 1000

                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                return GenerationResult(
                    text=generated_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason=data.get("details", {}).get("finish_reason", "stop"),
                    latency_ms=latency_ms,
                    model_id=self.model_id
                )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"TGI timeout (attempt {attempt + 1}/{attempts})")
                self._total_errors += 1

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"TGI HTTP error: {e.response.status_code}")
                self._total_errors += 1

                if 400 <= e.response.status_code < 500:
                    break

            except Exception as e:
                last_error = e
                logger.error(f"TGI error: {e}")
                self._total_errors += 1

            if attempt < attempts - 1:
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to generate after {attempts} attempts: {last_error}")

    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Generator[str, None, None]:
        """
        Stream generated text (synchronous).

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            Text chunks as they're generated
        """
        client = self._get_client()
        payload = self._prepare_request(prompt, config)
        payload["stream"] = True

        try:
            self._total_requests += 1

            with client.stream("POST", "/generate_stream", json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        # Parse SSE format
                        if line.startswith("data:"):
                            import json
                            data = json.loads(line[5:])
                            if "token" in data:
                                yield data["token"]["text"]

        except Exception as e:
            logger.error(f"TGI streaming error: {e}")
            self._total_errors += 1
            raise

    async def astream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated text (asynchronous).

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            Text chunks as they're generated
        """
        client = self._get_async_client()
        payload = self._prepare_request(prompt, config)
        payload["stream"] = True

        try:
            self._total_requests += 1

            async with client.stream("POST", "/generate_stream", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        if line.startswith("data:"):
                            import json
                            data = json.loads(line[5:])
                            if "token" in data:
                                yield data["token"]["text"]

        except Exception as e:
            logger.error(f"TGI streaming error: {e}")
            self._total_errors += 1
            raise

    def health_check(self) -> bool:
        """
        Check if TGI endpoint is healthy.

        Returns:
            True if healthy
        """
        try:
            client = self._get_client()
            response = client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"TGI health check failed: {e}")
            return False

    async def ahealth_check(self) -> bool:
        """Check if TGI endpoint is healthy (async)"""
        try:
            client = self._get_async_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"TGI health check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information from TGI.

        Returns:
            Model information dictionary
        """
        try:
            client = self._get_client()
            response = client.get("/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {"model_id": self.model_id}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "error_rate": self._total_errors / self._total_requests if self._total_requests > 0 else 0
        }

    def close(self) -> None:
        """Close HTTP clients"""
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._async_client is not None:
            # Note: For proper async cleanup, use aclose()
            try:
                asyncio.get_event_loop().run_until_complete(self._async_client.aclose())
            except Exception:
                pass
            self._async_client = None

        logger.info("LLMManager closed")

    async def aclose(self) -> None:
        """Close HTTP clients (async)"""
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

        logger.info("LLMManager closed")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_llm_manager: Optional[LLMManager] = None


def get_llm_manager(
    endpoint: Optional[str] = None,
    **kwargs
) -> LLMManager:
    """
    Get or create global LLMManager instance.

    Args:
        endpoint: TGI endpoint (only used for initial creation)
        **kwargs: Additional arguments for LLMManager

    Returns:
        Global LLMManager instance
    """
    global _llm_manager

    if _llm_manager is None:
        _llm_manager = LLMManager(endpoint=endpoint, **kwargs)
    elif endpoint and endpoint != _llm_manager.endpoint:
        logger.warning(
            f"Requested endpoint={endpoint} differs from current. "
            f"Call reset_llm_manager() first to change endpoint."
        )

    return _llm_manager


def reset_llm_manager() -> None:
    """Reset global LLM manager"""
    global _llm_manager

    if _llm_manager is not None:
        _llm_manager.close()
        _llm_manager = None
