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
    A single belief with confidence tracking and justification links.

    Attributes:
        id: Unique identifier
        content: Natural language statement
        confidence: Confidence level in [-1, 1]
        context: Domain category (e.g., "api_reliability", "security")
        source_task: Task that generated this belief
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
        supported_by: Parent beliefs that justify this one
        contradicted_by: Beliefs that contradict this one
        dependents: Child beliefs that depend on this one
        embedding: Optional semantic embedding vector
    """
    content: str
    confidence: Confidence
    context: str = "general"
    source_task: str = "unknown"
    id: BeliefID = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    supported_by: List[BeliefID] = field(default_factory=list)
    contradicted_by: List[BeliefID] = field(default_factory=list)
    dependents: List[BeliefID] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        """Validate confidence range."""
        if not -1.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [-1, 1], got {self.confidence}")

    def update_confidence(self, delta: Delta) -> None:
        """
        Update confidence by delta, clamping to [-1, 1].

        Args:
            delta: Change in confidence
        """
        self.confidence = max(-1.0, min(1.0, self.confidence + delta))
        self.updated_at = datetime.now()

    def add_supporter(self, belief_id: BeliefID) -> None:
        """Add a belief that supports this one."""
        if belief_id not in self.supported_by:
            self.supported_by.append(belief_id)

    def add_contradiction(self, belief_id: BeliefID) -> None:
        """Add a belief that contradicts this one."""
        if belief_id not in self.contradicted_by:
            self.contradicted_by.append(belief_id)

    def add_dependent(self, belief_id: BeliefID) -> None:
        """Add a belief that depends on this one."""
        if belief_id not in self.dependents:
            self.dependents.append(belief_id)

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
