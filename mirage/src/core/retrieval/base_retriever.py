"""
MIRAGE V2 Base Retriever
Abstract base class for all retrieval modes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import numpy as np


class RetrievalMode(Enum):
    """Available retrieval modes"""
    NAIVE = "naive"          # Basic vector search only
    LOCAL = "local"          # Entity-focused (entity → chunks)
    GLOBAL = "global"        # Relationship-focused (relationships → entities → chunks)
    HYBRID = "hybrid"        # Combines local + global
    MIX = "mix"             # All modes combined with RRF
    SEMANTIC = "semantic"    # Deep semantic matching
    BYPASS = "bypass"        # No retrieval (for testing generation)


@dataclass
class RetrievalResult:
    """Result from a retrieval operation"""
    chunk_id: str
    document_id: str
    text: str
    score: float
    retrieval_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For debugging/explanation
    via_entity: Optional[str] = None
    via_relationship: Optional[str] = None
    hop_distance: int = 0


@dataclass
class RetrievalResponse:
    """Full response from retrieval"""
    results: List[RetrievalResult]
    query: str
    mode: RetrievalMode
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.

    Each retriever implements a different retrieval strategy
    but provides a consistent interface.
    """

    def __init__(
        self,
        top_k: int = 10,
        score_threshold: float = 0.0,
        **kwargs
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.kwargs = kwargs

    @property
    @abstractmethod
    def mode(self) -> RetrievalMode:
        """Return the retrieval mode this retriever implements"""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RetrievalResponse:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query string
            query_embedding: Pre-computed query embedding (optional)
            top_k: Override default top_k
            **kwargs: Additional parameters

        Returns:
            RetrievalResponse with results
        """
        pass

    def _create_result(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        score: float,
        **kwargs
    ) -> RetrievalResult:
        """Helper to create a RetrievalResult"""
        return RetrievalResult(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            score=score,
            retrieval_mode=self.mode.value,
            metadata=kwargs.get("metadata", {}),
            via_entity=kwargs.get("via_entity"),
            via_relationship=kwargs.get("via_relationship"),
            hop_distance=kwargs.get("hop_distance", 0)
        )

    def _filter_by_threshold(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Filter results below score threshold"""
        return [r for r in results if r.score >= self.score_threshold]

    def _deduplicate(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Remove duplicate chunks, keeping highest score"""
        seen = {}
        for r in results:
            if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                seen[r.chunk_id] = r
        return list(seen.values())
