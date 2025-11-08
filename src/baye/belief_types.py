"""
Core data types for the belief tracking system.

This module defines the fundamental structures used throughout the justification
graph: Belief nodes, propagation events, and relationship types.
"""

from typing import List, Set, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import uuid


# Type aliases for clarity
BeliefID = str
Confidence = float  # Range: [-1, 1]
Delta = float  # Range: [-2, 2]


class RelationType(Enum):
    """Types of relationships between beliefs."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"


@dataclass
class Belief:
    """
    A single belief with Beta distribution tracking and justification links.

    Uses Beta distribution (a, b) for Bayesian belief updating:
    - confidence = a / (a + b)  [mean of Beta]
    - uncertainty = variance of Beta distribution

    Attributes:
        id: Unique identifier
        content: Natural language statement
        a: Beta distribution alpha parameter (positive evidence)
        b: Beta distribution beta parameter (negative evidence)
        context: Domain category (e.g., "api_reliability", "security")
        source_task: Task that generated this belief
        belief_class: Class/type of belief (e.g., "normal", "scratch")
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
        last_decay_at: Timestamp of last temporal decay
        supported_by: Parent beliefs that justify this one
        contradicts: Beliefs that contradict this one
        supports: Child beliefs that this one justifies
        embedding: Optional semantic embedding vector
        metadata: Additional metadata (reliability, tags, etc.)
    """
    content: str
    a: float = 1.0  # Beta alpha (positive evidence)
    b: float = 1.0  # Beta beta (negative evidence)
    context: str = "general"
    source_task: str = "unknown"
    belief_class: str = "normal"  # "normal", "scratch", "foundational"
    id: BeliefID = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_decay_at: datetime = field(default_factory=datetime.now)
    supported_by: List[BeliefID] = field(default_factory=list)
    contradicts: List[BeliefID] = field(default_factory=list)
    supports: List[BeliefID] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate Beta parameters."""
        if self.a <= 0 or self.b <= 0:
            raise ValueError(f"Beta parameters must be positive, got a={self.a}, b={self.b}")

        # Backward compatibility: if 'confidence' was passed, convert to Beta
        # This allows old code to still work
        if hasattr(self, '_legacy_confidence'):
            self.set_confidence(self._legacy_confidence)

    @classmethod
    def from_confidence(cls, content: str, confidence: float, **kwargs):
        """
        Create belief from legacy confidence value.

        Converts confidence ∈ [-1, 1] to Beta(a, b) using method of moments.
        For |conf| close to extremes, uses stronger priors.

        Args:
            content: Belief content
            confidence: Legacy confidence in [-1, 1]
            **kwargs: Other Belief parameters
        """
        # Convert confidence to Beta parameters
        # Map [-1, 1] → [0, 1] for Beta distribution
        p = (confidence + 1.0) / 2.0  # Map to [0, 1]

        # Strength of prior (how many "observations" this represents)
        # Higher confidence → stronger prior
        strength = 10.0 * abs(confidence)
        strength = max(2.0, strength)  # At least 2 observations

        a = p * strength
        b = (1 - p) * strength

        return cls(content=content, a=a, b=b, **kwargs)

    @property
    def confidence(self) -> Confidence:
        """
        Compute confidence from Beta distribution.

        Returns mean of Beta mapped to [-1, 1]:
        - Beta mean = a / (a + b) ∈ [0, 1]
        - Confidence = 2 * mean - 1 ∈ [-1, 1]
        """
        mean = self.a / (self.a + self.b)
        return 2.0 * mean - 1.0

    @property
    def uncertainty(self) -> float:
        """
        Compute uncertainty (variance) from Beta distribution.

        Returns:
            Variance of Beta distribution in [0, 1]
        """
        ab_sum = self.a + self.b
        return (self.a * self.b) / (ab_sum * ab_sum * (ab_sum + 1))

    @property
    def total_evidence(self) -> float:
        """Total amount of evidence (a + b)."""
        return self.a + self.b

    def set_confidence(self, confidence: float, strength: float = 10.0):
        """
        Set confidence by updating Beta parameters.

        Args:
            confidence: Target confidence in [-1, 1]
            strength: Strength of belief (total evidence count)
        """
        if not -1.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [-1, 1], got {confidence}")

        p = (confidence + 1.0) / 2.0
        self.a = p * strength
        self.b = (1 - p) * strength
        self.updated_at = datetime.now()

    def update_confidence(self, delta: Delta) -> float:
        """
        Update confidence by delta (backward compatibility).

        Converts delta to Beta update approximately.

        Args:
            delta: Change in confidence

        Returns:
            Actual delta applied
        """
        old_conf = self.confidence
        new_conf = max(-1.0, min(1.0, old_conf + delta))
        self.set_confidence(new_conf, strength=self.total_evidence)
        return self.confidence - old_conf

    def update_beta(self, delta_a: float, delta_b: float):
        """
        Update Beta parameters directly (for Bayesian updates).

        Args:
            delta_a: Change in alpha (positive evidence)
            delta_b: Change in beta (negative evidence)
        """
        self.a = max(0.001, self.a + delta_a)  # Keep positive
        self.b = max(0.001, self.b + delta_b)
        self.updated_at = datetime.now()

    def apply_decay(self, lambda_decay: float):
        """
        Apply temporal decay to Beta parameters.

        Shrinks (a, b) toward prior (1, 1) by factor λ ∈ [0, 1].

        Args:
            lambda_decay: Decay rate (0 = no decay, 1 = full reset)
        """
        # Decay toward uniform prior Beta(1, 1)
        self.a = 1.0 + (self.a - 1.0) * (1.0 - lambda_decay)
        self.b = 1.0 + (self.b - 1.0) * (1.0 - lambda_decay)
        self.last_decay_at = datetime.now()

    def add_supporter(self, belief_id: BeliefID) -> None:
        """Add a belief that supports this one."""
        if belief_id not in self.supported_by:
            self.supported_by.append(belief_id)

    def add_contradiction(self, belief_id: BeliefID) -> None:
        """Add a belief that contradicts this one."""
        if belief_id not in self.contradicts:
            self.contradicts.append(belief_id)

    def add_dependent(self, belief_id: BeliefID) -> None:
        """Add a belief that depends on this one."""
        if belief_id not in self.supports:
            self.supports.append(belief_id)

    def __repr__(self) -> str:
        return (f"Belief(id={self.id}, conf={self.confidence:.2f}, "
                f"content='{self.content[:40]}...')")


@dataclass
class PropagationEvent:
    """
    Record of a single belief update during propagation.

    Attributes:
        belief_id: ID of the updated belief
        old_confidence: Confidence before update
        new_confidence: Confidence after update
        delta: Change in confidence
        depth: Depth in propagation cascade
        source_belief_id: ID of belief that triggered this update
        mechanism: Type of propagation ("causal" or "semantic")
    """
    belief_id: BeliefID
    old_confidence: Confidence
    new_confidence: Confidence
    delta: Delta
    depth: int
    source_belief_id: Optional[BeliefID] = None
    mechanism: str = "causal"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def magnitude(self) -> float:
        """Absolute magnitude of the change."""
        return abs(self.delta)


@dataclass
class PropagationResult:
    """
    Result of a complete propagation cascade.

    Attributes:
        origin_belief_id: ID of belief that started the cascade
        events: List of all update events
        total_beliefs_updated: Count of unique beliefs affected
        max_depth_reached: Maximum depth of propagation
        duration_ms: Time taken in milliseconds
        cycles_detected: Number of cycles encountered
    """
    origin_belief_id: BeliefID
    events: List[PropagationEvent] = field(default_factory=list)
    total_beliefs_updated: int = 0
    max_depth_reached: int = 0
    duration_ms: float = 0.0
    cycles_detected: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def add_event(self, event: PropagationEvent) -> None:
        """Add an event and update statistics."""
        self.events.append(event)
        self.total_beliefs_updated = len(set(e.belief_id for e in self.events))
        self.max_depth_reached = max(self.max_depth_reached, event.depth)

    def get_events_by_depth(self, depth: int) -> List[PropagationEvent]:
        """Get all events at a specific depth."""
        return [e for e in self.events if e.depth == depth]

    def get_total_delta(self) -> float:
        """Sum of all absolute deltas."""
        return sum(e.magnitude for e in self.events)
