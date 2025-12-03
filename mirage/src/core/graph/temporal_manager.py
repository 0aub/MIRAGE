"""
MIRAGE V2 Temporal Manager
Manages temporal aspects of graph data:
- Temporal filtering (filter by date ranges)
- Confidence decay (older information has lower confidence)
- Version tracking for entities and relationships
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
from loguru import logger


class DecayFunction(Enum):
    """Confidence decay functions"""
    EXPONENTIAL = "exponential"  # Fast initial decay, slow later
    LINEAR = "linear"            # Constant decay rate
    STEP = "step"               # No decay until threshold, then drop
    LOGARITHMIC = "logarithmic"  # Slow initial decay, faster later


@dataclass
class TemporalConfig:
    """Configuration for temporal management"""
    # Decay settings
    decay_function: DecayFunction = DecayFunction.EXPONENTIAL
    half_life_days: int = 180  # Days until confidence reaches 50%
    min_confidence: float = 0.1  # Minimum confidence floor

    # Time windows for queries
    default_window_days: int = 365  # Default query window

    # Freshness boost
    freshness_boost_days: int = 7  # Recent items get boost
    freshness_boost_factor: float = 1.2  # Boost multiplier

    # Version tracking
    track_versions: bool = True
    max_versions: int = 10


@dataclass
class TemporalEntity:
    """Entity with temporal metadata"""
    entity_id: str
    name: str
    entity_type: str
    created_at: datetime
    updated_at: datetime
    version: int = 1
    original_confidence: float = 1.0
    current_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalRelationship:
    """Relationship with temporal metadata"""
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: str
    created_at: datetime
    updated_at: datetime
    version: int = 1
    original_confidence: float = 1.0
    current_confidence: float = 1.0
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalManager:
    """
    Manages temporal aspects of knowledge graph.

    Features:
    1. Confidence Decay - Information becomes less reliable over time
    2. Temporal Filtering - Query data within date ranges
    3. Freshness Boost - Recent updates get priority
    4. Version Tracking - Track changes over time
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        logger.info(
            f"TemporalManager initialized: decay={self.config.decay_function.value}, "
            f"half_life={self.config.half_life_days} days"
        )

    # =========================================================================
    # CONFIDENCE DECAY
    # =========================================================================

    def calculate_decay(
        self,
        original_confidence: float,
        created_at: datetime,
        reference_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate decayed confidence based on age.

        Args:
            original_confidence: Initial confidence value
            created_at: When the information was created/updated
            reference_time: Time to calculate decay from (default: now)

        Returns:
            Decayed confidence value
        """
        if reference_time is None:
            reference_time = datetime.now()

        # Calculate age in days
        age_days = (reference_time - created_at).days
        if age_days <= 0:
            return original_confidence

        # Apply decay function
        if self.config.decay_function == DecayFunction.EXPONENTIAL:
            decay_factor = self._exponential_decay(age_days)
        elif self.config.decay_function == DecayFunction.LINEAR:
            decay_factor = self._linear_decay(age_days)
        elif self.config.decay_function == DecayFunction.STEP:
            decay_factor = self._step_decay(age_days)
        else:  # LOGARITHMIC
            decay_factor = self._logarithmic_decay(age_days)

        # Apply decay and clamp to minimum
        decayed = original_confidence * decay_factor
        return max(decayed, self.config.min_confidence)

    def _exponential_decay(self, age_days: int) -> float:
        """Exponential decay: f(t) = 0.5^(t/half_life)"""
        return math.pow(0.5, age_days / self.config.half_life_days)

    def _linear_decay(self, age_days: int) -> float:
        """Linear decay over twice the half-life"""
        max_days = self.config.half_life_days * 2
        return max(0, 1 - (age_days / max_days))

    def _step_decay(self, age_days: int) -> float:
        """No decay until half-life, then drop to 50%"""
        if age_days < self.config.half_life_days:
            return 1.0
        elif age_days < self.config.half_life_days * 2:
            return 0.5
        else:
            return 0.25

    def _logarithmic_decay(self, age_days: int) -> float:
        """Logarithmic decay: slow at first, faster later"""
        # f(t) = 1 - log(1 + t) / log(1 + 2*half_life)
        max_days = self.config.half_life_days * 2
        decay = 1 - math.log1p(age_days) / math.log1p(max_days)
        return max(0, decay)

    def apply_freshness_boost(
        self,
        score: float,
        updated_at: datetime,
        reference_time: Optional[datetime] = None
    ) -> float:
        """
        Apply freshness boost to recently updated items.

        Args:
            score: Original score
            updated_at: When the item was last updated
            reference_time: Time to calculate from (default: now)

        Returns:
            Boosted score
        """
        if reference_time is None:
            reference_time = datetime.now()

        age_days = (reference_time - updated_at).days

        if age_days <= self.config.freshness_boost_days:
            # Apply graduated boost (more recent = higher boost)
            boost_factor = self.config.freshness_boost_factor - (
                (self.config.freshness_boost_factor - 1.0) *
                (age_days / self.config.freshness_boost_days)
            )
            return score * boost_factor

        return score

    # =========================================================================
    # TEMPORAL FILTERING
    # =========================================================================

    def filter_by_time_window(
        self,
        items: List[Dict[str, Any]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        date_field: str = "created_at"
    ) -> List[Dict[str, Any]]:
        """
        Filter items by time window.

        Args:
            items: List of items with date field
            start_date: Start of window (default: end_date - default_window)
            end_date: End of window (default: now)
            date_field: Name of date field in items

        Returns:
            Filtered items within time window
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=self.config.default_window_days)

        filtered = []
        for item in items:
            item_date = item.get(date_field)

            # Handle string dates
            if isinstance(item_date, str):
                try:
                    item_date = datetime.fromisoformat(item_date.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

            if item_date is None:
                continue

            # Remove timezone info for comparison if needed
            if hasattr(item_date, 'tzinfo') and item_date.tzinfo is not None:
                item_date = item_date.replace(tzinfo=None)

            if start_date <= item_date <= end_date:
                filtered.append(item)

        return filtered

    def get_recent_entities(
        self,
        entities: List[Dict[str, Any]],
        days: int = 30,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get entities updated within recent days with minimum confidence"""
        cutoff = datetime.now() - timedelta(days=days)

        recent = []
        for entity in entities:
            # Check date
            updated_at = entity.get("updated_at") or entity.get("created_at")
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

            if updated_at is None or updated_at < cutoff:
                continue

            # Check confidence
            confidence = entity.get("confidence", 1.0)
            if confidence < min_confidence:
                continue

            recent.append(entity)

        return recent

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def apply_decay_to_entities(
        self,
        entities: List[Dict[str, Any]],
        reference_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Apply confidence decay to a list of entities"""
        if reference_time is None:
            reference_time = datetime.now()

        for entity in entities:
            created_at = entity.get("updated_at") or entity.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

            if created_at is None:
                continue

            # Remove timezone for calculation
            if hasattr(created_at, 'tzinfo') and created_at.tzinfo is not None:
                created_at = created_at.replace(tzinfo=None)

            original_conf = entity.get("original_confidence", entity.get("confidence", 1.0))
            entity["decayed_confidence"] = self.calculate_decay(
                original_conf, created_at, reference_time
            )

        return entities

    def apply_decay_to_relationships(
        self,
        relationships: List[Dict[str, Any]],
        reference_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Apply confidence decay to a list of relationships"""
        if reference_time is None:
            reference_time = datetime.now()

        for rel in relationships:
            created_at = rel.get("updated_at") or rel.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

            if created_at is None:
                continue

            # Remove timezone for calculation
            if hasattr(created_at, 'tzinfo') and created_at.tzinfo is not None:
                created_at = created_at.replace(tzinfo=None)

            # Check validity period
            valid_until = rel.get("valid_until")
            if valid_until:
                if isinstance(valid_until, str):
                    try:
                        valid_until = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass

                if isinstance(valid_until, datetime):
                    if hasattr(valid_until, 'tzinfo') and valid_until.tzinfo is not None:
                        valid_until = valid_until.replace(tzinfo=None)
                    if reference_time > valid_until:
                        rel["decayed_confidence"] = self.config.min_confidence
                        continue

            original_conf = rel.get("original_confidence", rel.get("confidence", 1.0))
            rel["decayed_confidence"] = self.calculate_decay(
                original_conf, created_at, reference_time
            )

        return relationships

    def rank_by_temporal_score(
        self,
        items: List[Dict[str, Any]],
        base_score_field: str = "score",
        temporal_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Rank items combining base score with temporal factors.

        Final score = base_score * (1 - temporal_weight) + temporal_score * temporal_weight

        Temporal score considers:
        - Decayed confidence
        - Freshness boost
        """
        reference_time = datetime.now()

        for item in items:
            base_score = item.get(base_score_field, 0.5)

            # Get temporal factors
            created_at = item.get("updated_at") or item.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    created_at = None

            if created_at is None:
                item["temporal_score"] = base_score
                continue

            # Remove timezone for calculation
            if hasattr(created_at, 'tzinfo') and created_at.tzinfo is not None:
                created_at = created_at.replace(tzinfo=None)

            # Calculate decayed confidence
            original_conf = item.get("confidence", 1.0)
            decayed_conf = self.calculate_decay(original_conf, created_at, reference_time)

            # Apply freshness boost
            freshness_score = self.apply_freshness_boost(decayed_conf, created_at, reference_time)

            # Combine scores
            temporal_score = freshness_score
            final_score = base_score * (1 - temporal_weight) + temporal_score * temporal_weight

            item["temporal_score"] = temporal_score
            item["final_score"] = final_score

        # Sort by final score
        return sorted(items, key=lambda x: x.get("final_score", 0), reverse=True)

    # =========================================================================
    # NEO4J INTEGRATION
    # =========================================================================

    def get_temporal_filter_cypher(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        node_alias: str = "n"
    ) -> str:
        """
        Generate Cypher WHERE clause for temporal filtering.

        Returns clause like: "AND n.created_at >= $start_date AND n.created_at <= $end_date"
        """
        clauses = []

        if start_date:
            clauses.append(f"{node_alias}.created_at >= datetime($start_date)")

        if end_date:
            clauses.append(f"{node_alias}.created_at <= datetime($end_date)")

        if clauses:
            return " AND " + " AND ".join(clauses)
        return ""

    def get_decay_cypher(self, node_alias: str = "n") -> str:
        """
        Generate Cypher expression for confidence decay.

        Uses exponential decay formula in Cypher.
        """
        half_life = self.config.half_life_days

        return f"""
        CASE
            WHEN {node_alias}.updated_at IS NULL THEN {node_alias}.confidence
            ELSE {node_alias}.confidence * exp(
                -0.693 * duration.inDays(
                    datetime({node_alias}.updated_at), datetime()
                ).days / {half_life}
            )
        END
        """


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_temporal_manager: Optional[TemporalManager] = None


def get_temporal_manager(config: Optional[TemporalConfig] = None) -> TemporalManager:
    """Get or create global TemporalManager"""
    global _temporal_manager

    if _temporal_manager is None or config is not None:
        _temporal_manager = TemporalManager(config)

    return _temporal_manager
