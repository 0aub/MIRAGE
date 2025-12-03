"""
MIRAGE V2 Embedding Manager
Handles embedding model loading, inference, and batch processing.

Features:
- Automatic model loading from registry
- Batch embedding with GPU optimization
- Task-specific embeddings (passage, query, entity)
- Caching for repeated embeddings
- Fallback to CPU if GPU unavailable
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from .registry import get_model_registry, ModelRegistry


class EmbeddingManager:
    """
    Manages embedding model loading and inference.

    Responsibilities:
    - Load embedding models from HuggingFace or local cache
    - Generate embeddings with batch optimization
    - Handle task-specific embedding prefixes
    - Manage model lifecycle (load, unload)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "auto",
        dimensions: Optional[int] = None,
        normalize: bool = True,
        batch_size: int = 32,
        cache_embeddings: bool = True,
        registry: Optional[ModelRegistry] = None
    ):
        """
        Initialize embedding manager.

        Args:
            model_id: HuggingFace model ID or local path
            device: Target device ("cuda", "cpu", "auto")
            dimensions: Expected embedding dimensions (for validation)
            normalize: Normalize embeddings to unit length
            batch_size: Batch size for inference
            cache_embeddings: Cache embeddings in memory
            registry: Model registry to use (default: global)
        """
        self.registry = registry or get_model_registry()

        # Default model if not specified
        self.model_id = model_id or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

        # Resolve device
        if device == "auto":
            device = self.registry.default_device
        self.device = device

        self.expected_dimensions = dimensions
        self.normalize = normalize
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings

        # Model state
        self._model = None
        self._tokenizer = None
        self._actual_dimensions: Optional[int] = None

        # Embedding cache
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Task prefixes for models that support them
        self._task_prefixes = {
            "passage": "passage: ",
            "query": "query: ",
            "entity": "entity: ",
            "classification": "classify: ",
        }

        logger.info(
            f"EmbeddingManager initialized: model={self.model_id}, "
            f"device={self.device}, normalize={normalize}"
        )

    def _load_model(self) -> None:
        """Load the embedding model"""
        if self._model is not None:
            return

        logger.info(f"Loading embedding model: {self.model_id}")

        try:
            from sentence_transformers import SentenceTransformer

            # For sentence-transformers models, let it handle its own caching
            # This is more efficient than going through HuggingFace Hub
            if self.model_id.startswith("sentence-transformers/"):
                # Extract the model name after "sentence-transformers/"
                model_name = self.model_id.split("/", 1)[1]
                model_path = model_name
            else:
                # For other models, check registry first
                local_path = self.registry.get_local_path(self.model_id, auto_download=False)
                if local_path and local_path.exists():
                    model_path = str(local_path)
                else:
                    # Use model_id directly - sentence-transformers will handle download
                    model_path = self.model_id

            # Load model
            self._model = SentenceTransformer(
                model_path,
                device=self.device
            )

            # Register in registry for tracking
            self.registry.register(
                self.model_id,
                model_type="embedding",
                device=self.device
            )

            # Get actual dimensions
            self._actual_dimensions = self._model.get_sentence_embedding_dimension()

            if self.expected_dimensions and self._actual_dimensions != self.expected_dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: expected={self.expected_dimensions}, "
                    f"actual={self._actual_dimensions}. Using actual dimensions."
                )

            logger.info(
                f"Embedding model loaded: {self.model_id} "
                f"(dim={self._actual_dimensions}, device={self.device})"
            )

        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before use"""
        if self._model is None:
            self._load_model()

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions"""
        self._ensure_loaded()
        return self._actual_dimensions

    def _get_cache_key(self, text: str, task: Optional[str] = None) -> str:
        """Generate cache key for text and task"""
        import hashlib
        content = f"{task or 'default'}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _apply_task_prefix(self, texts: List[str], task: Optional[str]) -> List[str]:
        """Apply task prefix to texts if model supports it"""
        if not task or task not in self._task_prefixes:
            return texts

        # Check if model supports task prefixes
        # (some models like Jina, E5, etc. use prefixes)
        prefix = self._task_prefixes.get(task, "")

        # Only apply for known prefixed models
        prefixed_models = ["e5", "jina", "instructor"]
        model_lower = self.model_id.lower()

        if any(pm in model_lower for pm in prefixed_models):
            return [f"{prefix}{text}" for text in texts]

        return texts

    def embed(
        self,
        texts: Union[str, List[str]],
        task: Optional[str] = None,
        normalize: Optional[bool] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            task: Task type ("passage", "query", "entity", etc.)
            normalize: Override default normalization
            show_progress: Show progress bar for large batches

        Returns:
            Numpy array of embeddings (N x D)
        """
        self._ensure_loaded()

        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        if normalize is None:
            normalize = self.normalize

        # Check cache
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        if self.cache_embeddings:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, task)
                if cache_key in self._cache:
                    embeddings.append((i, self._cache[cache_key]))
                    self._cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self._cache_misses += 1
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Generate embeddings for uncached texts
        if uncached_texts:
            # Apply task prefixes
            processed_texts = self._apply_task_prefix(uncached_texts, task)

            # Generate embeddings in batches
            new_embeddings = self._model.encode(
                processed_texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                batch_size=self.batch_size
            )

            # Cache new embeddings
            if self.cache_embeddings:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    cache_key = self._get_cache_key(text, task)
                    self._cache[cache_key] = embedding

            # Merge with cached embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])

        # Return single embedding if single input
        if single_input:
            return result[0]

        return result

    def embed_passages(
        self,
        passages: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for document passages.

        Args:
            passages: List of document passages
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        return self.embed(passages, task="passage", show_progress=show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.

        Args:
            query: Search query

        Returns:
            Query embedding vector
        """
        return self.embed(query, task="query")

    def embed_entities(
        self,
        entities: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for entities.

        Args:
            entities: List of entity names/descriptions
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        return self.embed(entities, task="entity", show_progress=show_progress)

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            method: Similarity method ("cosine", "dot", "euclidean")

        Returns:
            Similarity score
        """
        if method == "cosine":
            # Cosine similarity (assuming normalized)
            return float(np.dot(embedding1, embedding2))
        elif method == "dot":
            return float(np.dot(embedding1, embedding2))
        elif method == "euclidean":
            return float(-np.linalg.norm(embedding1 - embedding2))
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        method: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple candidates.

        Args:
            query_embedding: Query embedding
            candidate_embeddings: Matrix of candidate embeddings (N x D)
            method: Similarity method

        Returns:
            Array of similarity scores
        """
        if method in ("cosine", "dot"):
            return np.dot(candidate_embeddings, query_embedding)
        elif method == "euclidean":
            return -np.linalg.norm(candidate_embeddings - query_embedding, axis=1)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate
        }

    def unload(self) -> None:
        """Unload model to free memory"""
        if self._model is not None:
            del self._model
            self._model = None

            # Clear GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info(f"Unloaded embedding model: {self.model_id}")

    def __del__(self):
        """Cleanup on deletion"""
        self.unload()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(
    model_id: Optional[str] = None,
    **kwargs
) -> EmbeddingManager:
    """
    Get or create global EmbeddingManager instance.

    Args:
        model_id: Model ID (only used for initial creation)
        **kwargs: Additional arguments for EmbeddingManager

    Returns:
        Global EmbeddingManager instance
    """
    global _embedding_manager

    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(model_id=model_id, **kwargs)
    elif model_id and model_id != _embedding_manager.model_id:
        logger.warning(
            f"Requested model_id={model_id} differs from loaded model. "
            f"Call reset_embedding_manager() first to change models."
        )

    return _embedding_manager


def reset_embedding_manager() -> None:
    """Reset global embedding manager (unload model)"""
    global _embedding_manager

    if _embedding_manager is not None:
        _embedding_manager.unload()
        _embedding_manager = None
