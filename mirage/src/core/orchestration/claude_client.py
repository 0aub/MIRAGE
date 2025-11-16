"""
Multi-Provider LLM Client
Handles response generation with citation extraction
Supports: TGI (local), OpenAI, Anthropic (Claude), Google (Gemini)
"""

import os
import re
import requests
from typing import List, Dict, Any, Optional
from loguru import logger

from ...config import settings

# Optional imports
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not available")

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("litellm package not available")


class ClaudeClient:
    """Multi-provider LLM client with citation support"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client with auto-detected provider

        Args:
            api_key: API key (uses settings auto-detection if not provided)
        """
        self.provider = None
        self.model = None
        self.api_key = api_key
        self.client = None
        self.tgi_endpoint = None

        # Auto-detect available provider
        self._detect_provider()

        logger.info(f"Initialized LLM client: provider={self.provider}, model={self.model}")

    def _detect_provider(self) -> None:
        """Auto-detect which LLM provider to use based on available API keys"""

        # Check for TGI (local GPU) first - highest priority if enabled
        if settings.use_tgi and settings.tgi_endpoint:
            self.provider = "tgi"
            self.model = "tgi"
            self.tgi_endpoint = settings.tgi_endpoint
            logger.info(f"Using local TGI endpoint at {settings.tgi_endpoint} for chat (NO RATE LIMITS!)")
            return

        # Check for API keys in order of preference
        if settings.openai_api_key and LITELLM_AVAILABLE:
            self.provider = "openai"
            self.model = "gpt-4o-mini"  # Fast and cheap
            self.api_key = settings.openai_api_key
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            logger.info("Using OpenAI (gpt-4o-mini) for chat")

        elif settings.anthropic_api_key:
            self.provider = "anthropic"
            self.model = "claude-3-5-sonnet-20241022"  # Best quality
            self.api_key = settings.anthropic_api_key
            if ANTHROPIC_AVAILABLE:
                self.client = Anthropic(api_key=self.api_key)
                logger.info("Using Anthropic Claude 3.5 Sonnet for chat")
            else:
                logger.error("Anthropic package not installed but API key provided")
                self.provider = None

        elif settings.google_api_key and LITELLM_AVAILABLE:
            self.provider = "google"
            self.model = "gemini/gemini-2.0-flash-exp"  # Latest Gemini 2.0
            self.api_key = settings.google_api_key
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
            logger.info("Using Google Gemini 2.0 (Flash) for chat")

        else:
            logger.error("No LLM provider available! Please enable TGI or set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
            self.provider = None
    
    def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """
        Generate response using any available LLM provider with context

        Args:
            query: User query
            context: Retrieved and compressed context
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response

        Returns:
            Dict with response, citations, and metadata
        """
        if not self.provider:
            logger.error("No LLM provider available")
            return {
                "response": "Error: No LLM provider configured. Please set up TGI or an API key.",
                "citations": [],
                "model": "none",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        # Default system prompt
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        # Construct user message with context
        user_message = self._construct_message(query, context)

        try:
            logger.debug(f"Calling {self.provider} with query length={len(query)}, context length={len(context)}")

            # Handle TGI separately (uses /v1/chat/completions)
            if self.provider == "tgi":
                tgi_response = requests.post(
                    f"{self.tgi_endpoint}/v1/chat/completions",
                    json={
                        "model": "tgi",  # Required but ignored by TGI
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.3,
                        "max_tokens": max_tokens,
                    },
                    timeout=120
                )
                tgi_response.raise_for_status()
                response_data = tgi_response.json()
                response_text = response_data["choices"][0]["message"]["content"]

                # TGI doesn't provide token counts in v3.3.6
                input_tokens = 0
                output_tokens = 0

            # Handle Anthropic directly (native client)
            elif self.provider == "anthropic" and self.client:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                response_text = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

            # Handle OpenAI and Gemini via LiteLLM
            elif self.provider in ["openai", "google"] and LITELLM_AVAILABLE:
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens,
                )
                response_text = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            else:
                raise Exception(f"Provider {self.provider} not properly configured")

            # Extract citations from response
            citations = self._extract_citations(response_text, context)

            logger.info(
                f"Generated response: {len(response_text)} chars, "
                f"{len(citations)} citations, "
                f"tokens={input_tokens}+{output_tokens}"
            )

            return {
                "response": response_text,
                "citations": citations,
                "model": self.model,
                "provider": self.provider,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        except Exception as e:
            logger.error(f"Error calling {self.provider} API: {e}")
            raise
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG"""
        return """You are an intelligent assistant helping users answer questions based on a knowledge graph.

You will be provided with:
1. A user question
2. Relevant context retrieved from a knowledge graph

Your task:
- Answer the question accurately using the provided context
- Be concise but comprehensive
- If the context doesn't contain enough information, say so
- Include citations by referencing the context sections
- Maintain a helpful and professional tone

When citing information:
- Reference specific parts of the context
- Use inline citations like [1], [2], etc.
- Be precise about which facts come from which sources"""
    
    def _construct_message(self, query: str, context: str) -> str:
        """Construct user message with context"""
        return f"""Context from knowledge graph:
---
{context}
---

Question: {query}

Please answer the question using the context provided above. Include citations where appropriate."""
    
    def _extract_citations(
        self,
        response: str,
        context: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract citations from response
        
        Args:
            response: Generated response text
            context: Original context
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Find citation markers like [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        matches = re.finditer(citation_pattern, response)
        
        seen_citations = set()
        for match in matches:
            citation_num = match.group(1)
            
            if citation_num not in seen_citations:
                seen_citations.add(citation_num)
                
                # Try to find what this citation refers to in context
                # This is a simple heuristic - could be improved
                citations.append({
                    "id": citation_num,
                    "text": self._find_citation_text(int(citation_num), context),
                    "position": match.start(),
                })
        
        # If no explicit citations found, try to extract quoted text
        if not citations:
            quotes = re.findall(r'"([^"]+)"', response)
            for i, quote in enumerate(quotes[:5], 1):  # Limit to 5
                if quote in context:
                    citations.append({
                        "id": str(i),
                        "text": quote,
                        "type": "quote",
                    })
        
        return citations
    
    def _find_citation_text(self, citation_num: int, context: str) -> str:
        """
        Find text corresponding to citation number
        
        Simple heuristic: split context into chunks and return the nth chunk
        """
        # Split context into paragraphs or chunks
        chunks = [c.strip() for c in context.split('\n\n') if c.strip()]
        
        if not chunks:
            chunks = [c.strip() for c in context.split('\n') if c.strip()]
        
        # Return corresponding chunk (1-indexed)
        if 0 < citation_num <= len(chunks):
            chunk = chunks[citation_num - 1]
            # Truncate if too long
            if len(chunk) > 300:
                chunk = chunk[:300] + "..."
            return chunk
        
        return "Context not found"
    
    def stream_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
    ):
        """
        Stream response from any LLM provider (for real-time responses)

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens

        Yields:
            Response chunks as they arrive
        """
        if not self.provider:
            yield "Error: No LLM provider configured"
            return

        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        user_message = self._construct_message(query, context)

        try:
            # TGI streaming support (via SSE)
            if self.provider == "tgi":
                response = requests.post(
                    f"{self.tgi_endpoint}/v1/chat/completions",
                    json={
                        "model": "tgi",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.3,
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                    stream=True,
                    timeout=120
                )
                response.raise_for_status()

                # Parse SSE stream
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str != '[DONE]':
                                try:
                                    import json
                                    data = json.loads(data_str)
                                    delta = data.get('choices', [{}])[0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                                except:
                                    pass

            # Anthropic native streaming
            elif self.provider == "anthropic" and self.client:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                ) as stream:
                    for text in stream.text_stream:
                        yield text

            # LiteLLM streaming for OpenAI and Gemini
            elif self.provider in ["openai", "google"] and LITELLM_AVAILABLE:
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            else:
                yield f"Error: Provider {self.provider} not configured for streaming"

        except Exception as e:
            logger.error(f"Error streaming from {self.provider}: {e}")
            yield f"Error: {str(e)}"
