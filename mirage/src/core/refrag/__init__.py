"""
REFRAG Module
Phase 3: REFRAG Integration

Reinforcement Learning-based Fragment Compression
Selective compression based on importance scoring
Achieves significant speedup while maintaining quality

Handles:
- Fragment importance scoring
- RL-based compression policy
- Selective compression
- Compression caching
"""

from .compressor import REFRAGCompressor
from .policy import CompressionPolicy
from .cache import CompressionCache

__all__ = [
    "REFRAGCompressor",
    "CompressionPolicy",
    "CompressionCache",
]
