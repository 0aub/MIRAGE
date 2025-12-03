"""
MIRAGE V2 Graph Module
Provides temporal management and graph utilities.

Note: Most graph functionality is in graph_builder module.
This module adds V2-specific features like temporal management.
"""

from .temporal_manager import (
    TemporalManager,
    TemporalConfig,
    TemporalEntity,
    TemporalRelationship,
    DecayFunction,
    get_temporal_manager,
)

__all__ = [
    "TemporalManager",
    "TemporalConfig",
    "TemporalEntity",
    "TemporalRelationship",
    "DecayFunction",
    "get_temporal_manager",
]
