"""
Community Detection Models

Dataclasses for community detection results.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Community:
    """Represents a community at a specific hierarchy level."""
    id: str
    level: int
    entities: List[str]
    size: int
    parent_community: Optional[str] = None
    child_communities: List[str] = field(default_factory=list)
    # GraphRAG Enhancement: Store descriptions and summaries
    resolution: float = 1.0
    summary: Optional[str] = None
    key_entities: List[str] = field(default_factory=list)
    entity_descriptions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "level": self.level,
            "entities": self.entities,
            "size": self.size,
            "parent_community": self.parent_community,
            "child_communities": self.child_communities,
            "resolution": self.resolution,
            "summary": self.summary,
            "key_entities": self.key_entities
        }


@dataclass
class CommunityDetectionResult:
    """Result from community detection."""
    communities: List[Community]
    hierarchy_levels: int
    total_communities: int
    modularity: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "communities": [c.to_dict() for c in self.communities],
            "hierarchy_levels": self.hierarchy_levels,
            "total_communities": self.total_communities,
            "modularity": self.modularity
        }
