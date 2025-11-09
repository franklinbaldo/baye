"""
Evidence system for Update-on-Use belief tracking.

This module implements US-01: recording evidence from tool calls and updating
beliefs using Bayesian Beta distribution updates.

Core formula: w = s * r * n * q * α
- s: Sentiment/polarity (+1 supports, -1 contradicts)
- r: Reliability of source/tool
- n: Novelty (1 - max similarity to existing evidence)
- q: Quality score
- α: Learning rate for belief class
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

from .belief_types import BeliefID


# ============================================================================
# Evidence Data Structures
# ============================================================================

@dataclass
class Evidence:
    """
    A piece of evidence supporting or contradicting a belief.

    Attributes:
        id: Unique identifier
        belief_id: Belief this evidence affects
        content: Text content of the evidence
        source: Source identifier (e.g., "tool:web_search", "human:review")
        source_type: Category of source (e.g., "api", "database", "llm")
        sentiment: +1 (supports), -1 (contradicts), 0 (neutral)
        quality: Quality score in [0, 1]
        hash: Content hash for deduplication
        created_at: Timestamp
        metadata: Additional context (tool params, etc.)
    """
    belief_id: BeliefID
    content: str
    source: str
    source_type: str = "unknown"
    sentiment: float = 1.0  # +1, -1, or 0
    quality: float = 1.0
    id: str = field(default_factory=lambda: hashlib.sha256(
        f"{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16])
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute content hash for deduplication."""
        # Compute stable hash from content + source
        content_str = f"{self.content}|{self.source}"
        self.hash = hashlib.sha256(content_str.encode()).hexdigest()

    @property
    def hash(self) -> str:
        """Content hash for deduplication."""
        if not hasattr(self, '_hash'):
            content_str = f"{self.content}|{self.source}"
            self._hash = hashlib.sha256(content_str.encode()).hexdigest()
        return self._hash

    @hash.setter
    def hash(self, value: str):
        """Set hash."""
        self._hash = value


@dataclass
class EvidenceUpdate:
    """
    Record of a belief update from evidence.

    Tracks the full calculation for observability (US-12).

    Attributes:
        evidence_id: Evidence that triggered update
        belief_id: Belief that was updated
        weight_w: Final computed weight
        sentiment_s: Sentiment component
        reliability_r: Reliability component
        novelty_n: Novelty component
        quality_q: Quality component
        alpha: Learning rate
        delta_a: Change in Beta alpha
        delta_b: Change in Beta beta
        a_before: Beta alpha before update
        b_before: Beta beta before update
        a_after: Beta alpha after update
        b_after: Beta beta after update
        was_duplicate: Whether evidence was duplicate
        timestamp: When update occurred
    """
    evidence_id: str
    belief_id: BeliefID
    weight_w: float
    sentiment_s: float
    reliability_r: float
    novelty_n: float
    quality_q: float
    alpha: float
    delta_a: float
    delta_b: float
    a_before: float
    b_before: float
    a_after: float
    b_after: float
    was_duplicate: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            'evidence_id': self.evidence_id,
            'belief_id': self.belief_id,
            'weight': self.weight_w,
            'components': {
                's': self.sentiment_s,
                'r': self.reliability_r,
                'n': self.novelty_n,
                'q': self.quality_q,
                'alpha': self.alpha,
            },
            'beta_update': {
                'before': {'a': self.a_before, 'b': self.b_before},
                'delta': {'a': self.delta_a, 'b': self.delta_b},
                'after': {'a': self.a_after, 'b': self.b_after},
            },
            'was_duplicate': self.was_duplicate,
            'timestamp': self.timestamp.isoformat(),
        }


# ============================================================================
# Evidence Store
# ============================================================================

class EvidenceStore:
    """
    Storage and deduplication for evidence.

    Features:
    - Hash-based deduplication (US-01)
    - Novelty calculation (US-03)
    - Audit trail (US-12)
    """

    def __init__(self):
        """Initialize evidence store."""
        self.evidence_by_id: Dict[str, Evidence] = {}
        self.evidence_by_belief: Dict[BeliefID, List[str]] = {}
        self.evidence_hashes: Dict[str, str] = {}  # hash -> evidence_id
        self.update_log: List[EvidenceUpdate] = []

    def add_evidence(self, evidence: Evidence) -> tuple[bool, Optional[str]]:
        """
        Add evidence to store with deduplication.

        Args:
            evidence: Evidence to add

        Returns:
            (is_new, existing_id): is_new=True if novel, existing_id if duplicate
        """
        # Check for duplicate
        if evidence.hash in self.evidence_hashes:
            existing_id = self.evidence_hashes[evidence.hash]
            return False, existing_id

        # Store evidence
        self.evidence_by_id[evidence.id] = evidence
        self.evidence_hashes[evidence.hash] = evidence.id

        # Index by belief
        if evidence.belief_id not in self.evidence_by_belief:
            self.evidence_by_belief[evidence.belief_id] = []
        self.evidence_by_belief[evidence.belief_id].append(evidence.id)

        return True, None

    def get_evidence_for_belief(self, belief_id: BeliefID) -> List[Evidence]:
        """Get all evidence for a belief."""
        if belief_id not in self.evidence_by_belief:
            return []

        evidence_ids = self.evidence_by_belief[belief_id]
        return [self.evidence_by_id[eid] for eid in evidence_ids]

    def calculate_novelty(self, new_evidence: Evidence,
                         similarity_fn=None) -> float:
        """
        Calculate novelty of evidence (US-03).

        Novelty n = 1 - max_similarity to existing evidence for same belief.

        Args:
            new_evidence: New evidence to check
            similarity_fn: Function(text1, text2) -> similarity in [0, 1]
                          Defaults to Jaccard similarity

        Returns:
            Novelty score in [0, 1] where 1 = completely novel
        """
        if similarity_fn is None:
            from .belief_estimation import SemanticEstimator
            estimator = SemanticEstimator()
            similarity_fn = estimator._calculate_similarity

        # Get existing evidence for this belief
        existing = self.get_evidence_for_belief(new_evidence.belief_id)

        if not existing:
            return 1.0  # Completely novel if first evidence

        # Compute similarity to each existing evidence
        similarities = []
        for ev in existing:
            sim = similarity_fn(new_evidence.content, ev.content)
            similarities.append(sim)

        # Novelty = 1 - max similarity
        max_sim = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_sim

        return max(0.0, min(1.0, novelty))

    def log_update(self, update: EvidenceUpdate):
        """Log an evidence update for observability (US-12)."""
        self.update_log.append(update)

    def get_updates_for_belief(self, belief_id: BeliefID) -> List[EvidenceUpdate]:
        """Get all updates for a belief."""
        return [u for u in self.update_log if u.belief_id == belief_id]

    def get_duplicate_rate(self) -> float:
        """
        Calculate rate of duplicate evidence (US-12 metric).

        Returns:
            Fraction of updates that were duplicates
        """
        if not self.update_log:
            return 0.0

        duplicates = sum(1 for u in self.update_log if u.was_duplicate)
        return duplicates / len(self.update_log)

    def export_audit_trail(self, belief_id: Optional[BeliefID] = None) -> List[Dict]:
        """
        Export audit trail for observability (US-12).

        Args:
            belief_id: Optional filter by belief

        Returns:
            List of update records as dicts
        """
        updates = self.update_log
        if belief_id:
            updates = [u for u in updates if u.belief_id == belief_id]

        return [u.to_dict() for u in updates]


# ============================================================================
# Update-on-Use Engine
# ============================================================================

class UpdateOnUseEngine:
    """
    Core engine for Update-on-Use belief updating (US-01).

    Implements the formula: w = s * r * n * q * α

    Then updates Beta distribution:
    - If s > 0: delta_a = w, delta_b = 0
    - If s < 0: delta_a = 0, delta_b = |w|
    """

    def __init__(self,
                 evidence_store: EvidenceStore,
                 alpha_by_class: Optional[Dict[str, float]] = None):
        """
        Initialize Update-on-Use engine.

        Args:
            evidence_store: Evidence storage
            alpha_by_class: Learning rates by belief class
                           Default: {"normal": 1.0, "scratch": 0.1}
        """
        self.evidence_store = evidence_store

        # Default learning rates (US-10)
        self.alpha_by_class = alpha_by_class or {
            "normal": 1.0,
            "scratch": 0.1,  # Low impact for scratch beliefs
            "foundational": 0.5,  # Moderate for foundational
        }

    def calculate_weight(self,
                        sentiment: float,
                        reliability: float,
                        novelty: float,
                        quality: float,
                        alpha: float) -> float:
        """
        Calculate evidence weight: w = s * r * n * q * α

        Args:
            sentiment: +1 (supports), -1 (contradicts)
            reliability: Reliability score in [0, 1]
            novelty: Novelty score in [0, 1]
            quality: Quality score in [0, 1]
            alpha: Learning rate

        Returns:
            Weight (can be positive or negative)
        """
        return sentiment * reliability * novelty * quality * alpha

    def update_belief_from_evidence(self,
                                   belief,
                                   evidence: Evidence,
                                   reliability: float,
                                   min_weight_threshold: float = 0.01) -> EvidenceUpdate:
        """
        Update belief from evidence using UoU formula (US-01).

        Args:
            belief: Belief to update (modified in place)
            evidence: Evidence to apply
            reliability: Reliability of evidence source (from catalog, US-02)
            min_weight_threshold: Minimum weight to apply update (US-10)

        Returns:
            EvidenceUpdate record for observability
        """
        # Calculate novelty (US-03)
        novelty = self.evidence_store.calculate_novelty(evidence)

        # Get learning rate for belief class
        alpha = self.alpha_by_class.get(belief.belief_class, 1.0)

        # Calculate weight
        weight = self.calculate_weight(
            sentiment=evidence.sentiment,
            reliability=reliability,
            novelty=novelty,
            quality=evidence.quality,
            alpha=alpha
        )

        # Record before state
        a_before = belief.a
        b_before = belief.b

        # Check if weight is below threshold (US-10 abstention)
        if abs(weight) < min_weight_threshold:
            # Create update record but don't modify belief
            update = EvidenceUpdate(
                evidence_id=evidence.id,
                belief_id=belief.id,
                weight_w=weight,
                sentiment_s=evidence.sentiment,
                reliability_r=reliability,
                novelty_n=novelty,
                quality_q=evidence.quality,
                alpha=alpha,
                delta_a=0.0,
                delta_b=0.0,
                a_before=a_before,
                b_before=b_before,
                a_after=a_before,
                b_after=b_before,
                was_duplicate=False
            )

            # Log reason for no update
            update.metadata = {"reason": "weight_below_threshold", "threshold": min_weight_threshold}

            return update

        # Apply Beta update based on sentiment
        if evidence.sentiment > 0:
            # Positive evidence: increase alpha
            delta_a = abs(weight)
            delta_b = 0.0
        else:
            # Negative evidence: increase beta
            delta_a = 0.0
            delta_b = abs(weight)

        # Update belief
        belief.update_beta(delta_a, delta_b)

        # Create update record
        update = EvidenceUpdate(
            evidence_id=evidence.id,
            belief_id=belief.id,
            weight_w=weight,
            sentiment_s=evidence.sentiment,
            reliability_r=reliability,
            novelty_n=novelty,
            quality_q=evidence.quality,
            alpha=alpha,
            delta_a=delta_a,
            delta_b=delta_b,
            a_before=a_before,
            b_before=b_before,
            a_after=belief.a,
            b_after=belief.b
        )

        return update

    def process_tool_call(self,
                         belief_id: BeliefID,
                         belief,
                         tool_result: str,
                         tool_name: str,
                         reliability: float,
                         sentiment: float = 1.0,
                         quality: float = 1.0,
                         metadata: Optional[Dict] = None) -> tuple[Evidence, EvidenceUpdate]:
        """
        Process a tool call and update belief (US-01 main entry point).

        This is the primary integration point for agentic tool usage.

        Args:
            belief_id: ID of belief to update
            belief: Belief object (will be modified)
            tool_result: Result text from tool
            tool_name: Name of tool (e.g., "web_search", "code_execution")
            reliability: Reliability of this tool (from catalog)
            sentiment: +1 (supports), -1 (contradicts), 0 (neutral)
            quality: Quality assessment of result
            metadata: Additional context

        Returns:
            (evidence, update): Created evidence and update record

        Example:
            >>> engine = UpdateOnUseEngine(store)
            >>> belief = graph.beliefs["belief123"]
            >>> evidence, update = engine.process_tool_call(
            ...     belief_id="belief123",
            ...     belief=belief,
            ...     tool_result="API returned 500 error",
            ...     tool_name="api_call",
            ...     reliability=0.9,
            ...     sentiment=-1.0  # Contradicts reliability assumption
            ... )
        """
        # Create evidence
        evidence = Evidence(
            belief_id=belief_id,
            content=tool_result,
            source=f"tool:{tool_name}",
            source_type="tool",
            sentiment=sentiment,
            quality=quality,
            metadata=metadata or {}
        )

        # Check for duplicate
        is_new, existing_id = self.evidence_store.add_evidence(evidence)

        if not is_new:
            # Duplicate evidence: create no-op update
            update = EvidenceUpdate(
                evidence_id=existing_id,
                belief_id=belief_id,
                weight_w=0.0,
                sentiment_s=sentiment,
                reliability_r=reliability,
                novelty_n=0.0,  # Zero novelty for duplicate
                quality_q=quality,
                alpha=self.alpha_by_class.get(belief.belief_class, 1.0),
                delta_a=0.0,
                delta_b=0.0,
                a_before=belief.a,
                b_before=belief.b,
                a_after=belief.a,
                b_after=belief.b,
                was_duplicate=True
            )
        else:
            # New evidence: perform update
            update = self.update_belief_from_evidence(
                belief=belief,
                evidence=evidence,
                reliability=reliability
            )

        # Log update
        self.evidence_store.log_update(update)

        return evidence, update
