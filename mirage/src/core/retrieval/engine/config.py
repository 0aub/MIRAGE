"""
Retrieval Engine Configuration

Configuration dataclass for the unified retrieval engine.
"""

from typing import Dict
from dataclasses import dataclass, field

from ..base_retriever import RetrievalMode


@dataclass
class RetrievalEngineConfig:
    """Configuration for the retrieval engine"""
    # Default mode when no routing
    # Updated 2024-12: GLOBAL is best performer (100%, avg 0.842)
    default_mode: RetrievalMode = RetrievalMode.GLOBAL

    # Top-k settings
    default_top_k: int = 10
    max_top_k: int = 50

    # Score thresholds
    min_score: float = 0.0

    # Fusion settings
    fusion_method: str = "rrf"  # "rrf", "weighted", "interleave"
    # Weights tuned based on evaluation results (2024-12):
    # global: 12/12 (0.842), local: 12/12 (0.833), hybrid: 11/12 (0.825), naive: 11/12 (0.817)
    mode_weights: Dict[str, float] = field(default_factory=lambda: {
        "naive": 0.7,
        "local": 0.95,
        "global": 1.0,    # Best performer - highest weight
        "hybrid": 0.85,
        "semantic": 0.9
    })

    # Auto-routing
    auto_route: bool = True

    # Fallback behavior
    fallback_on_error: bool = True
    fallback_mode: RetrievalMode = RetrievalMode.NAIVE

    # Metrics tracking (Phase 3)
    track_metrics: bool = True
