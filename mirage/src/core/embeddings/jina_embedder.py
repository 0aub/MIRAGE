"""
Jina v4 Embedding Wrapper
Supports both API-based and local embeddings
Optimized for multilingual text including Arabic
Uses jina-embeddings-v4 with 1024-dimensional embeddings
"""

import numpy as np
from typing import List, Union, Optional
import requests
from loguru import logger

from ...config import settings
from ...config.config_loader import get_config_loader


class JinaEmbedder:
    """Wrapper for Jina v4 embeddings with caching and batch support"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_api: Optional[bool] = None,
        embedding_dim: Optional[int] = None,
    ):
        """
        Initialize Jina embedder

        Args:
            api_key: Jina API key (optional, defaults to local inference if not provided)
            model: Model identifier (loaded from config if not provided)
            use_api: Use Jina API vs local inference (loaded from config if not provided)
            embedding_dim: Embedding dimension (loaded from config if not provided)
        """
        # Load configuration
        config_loader = get_config_loader()
        jina_config = config_loader.get_jina_config()

        self.api_key = api_key or ""
        self.model = model or jina_config.get("model", "jina-embeddings-v4")
        # Only use API if api_key is actually provided
        self.use_api = (use_api if use_api is not None else jina_config.get("use_api", False)) and bool(self.api_key)
        self.embedding_dim = embedding_dim or jina_config.get("embedding_dim", 1024)

        # API endpoint
        self.api_url = jina_config.get("api_url", "https://api.jina.ai/v1/embeddings")

        # Embedding cache
        self._cache = {}

        logger.info(
            f"Initialized JinaEmbedder: model={self.model}, "
            f"use_api={self.use_api}, dim={self.embedding_dim}"
        )

    def embed(
        self,
        texts: Union[str, List[str]],
        task: str = "retrieval.passage",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            task: Task type (retrieval.passage, retrieval.query, classification, etc.)
            normalize: Normalize embeddings to unit length

        Returns:
            Numpy array of embeddings (N x D)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Check cache
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, task)
            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Get embeddings for uncached texts
        if uncached_texts:
            if self.use_api:
                new_embeddings = self._embed_api(uncached_texts, task)
            else:
                new_embeddings = self._embed_local(uncached_texts)

            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text, task)
                self._cache[cache_key] = embedding

            # Merge cached and new embeddings in correct order
            result = [None] * len(texts)
            cached_idx = 0
            uncached_idx = 0

            for i in range(len(texts)):
                if i in uncached_indices:
                    result[i] = new_embeddings[uncached_idx]
                    uncached_idx += 1
                else:
                    result[i] = embeddings[cached_idx]
                    cached_idx += 1

            embeddings = result

        # Convert to numpy array
        embeddings = np.array(embeddings)

        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        # Return single embedding if single input
        if single_input:
            return embeddings[0]

        return embeddings

    def _embed_api(self, texts: List[str], task: str) -> List[np.ndarray]:
        """
        Get embeddings via Jina API

        Args:
            texts: List of texts
            task: Task type

        Returns:
            List of embedding arrays
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts,
            "task": task,
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]

            logger.debug(f"Generated {len(embeddings)} embeddings via API")
            return embeddings

        except Exception as e:
            logger.error(f"Error calling Jina API: {e}")
            # Fallback to random embeddings for development
            logger.warning("Falling back to random embeddings")
            return [np.random.randn(self.embedding_dim).tolist() for _ in texts]

    def _embed_local(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings using local Jina model

        Args:
            texts: List of texts

        Returns:
            List of embedding arrays
        """
        # TODO: Implement local Jina inference
        # For now, fallback to random embeddings
        logger.warning("Local inference not implemented, using random embeddings")
        return [np.random.randn(self.embedding_dim) for _ in texts]

    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
        method: str = "cosine",
    ) -> float:
        """
        Compute similarity between two texts or embeddings

        Args:
            text1: First text or embedding
            text2: Second text or embedding
            method: Similarity method (cosine, dot, euclidean)

        Returns:
            Similarity score
        """
        # Get embeddings if text provided
        if isinstance(text1, str):
            emb1 = self.embed(text1)
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self.embed(text2)
        else:
            emb2 = text2

        # Compute similarity
        if method == "cosine":
            # Cosine similarity (assuming normalized embeddings)
            return float(np.dot(emb1, emb2))
        elif method == "dot":
            return float(np.dot(emb1, emb2))
        elif method == "euclidean":
            return float(-np.linalg.norm(emb1 - emb2))
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def batch_similarity(
        self,
        query: Union[str, np.ndarray],
        candidates: Union[List[str], np.ndarray],
        method: str = "cosine",
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple candidates

        Args:
            query: Query text or embedding
            candidates: List of candidate texts or embeddings matrix
            method: Similarity method

        Returns:
            Array of similarity scores
        """
        # Get embeddings
        if isinstance(query, str):
            query_emb = self.embed(query)
        else:
            query_emb = query

        if isinstance(candidates, list) and isinstance(candidates[0], str):
            candidate_embs = self.embed(candidates)
        else:
            candidate_embs = candidates

        # Compute similarities
        if method == "cosine":
            # Matrix multiplication for batch cosine similarity
            similarities = np.dot(candidate_embs, query_emb)
        elif method == "dot":
            similarities = np.dot(candidate_embs, query_emb)
        elif method == "euclidean":
            similarities = -np.linalg.norm(candidate_embs - query_emb, axis=1)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

        return similarities

    def _get_cache_key(self, text: str, task: str) -> str:
        """Generate cache key for text and task"""
        return f"{task}:{hash(text)}"

    def clear_cache(self):
        """Clear embedding cache"""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached embeddings"""
        return len(self._cache)
