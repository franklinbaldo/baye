"""
Belief update policies (US-10).

Features:
- Abstention policy (don't update if weight too low)
- Scratch beliefs (temporary, low-impact investigation)
- Policy-based belief classes
"""

from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

from .belief_types import Belief


# ============================================================================
# Policy Configuration
# ============================================================================

@dataclass
class BeliefClassPolicy:
    """
    Policy configuration for a belief class.

    Attributes:
        class_name: Belief class identifier
        alpha: Learning rate (weight multiplier)
        min_weight_threshold: Minimum weight to apply update
        max_lifetime_hours: Max lifetime before auto-expiry (None = infinite)
        decay_enabled: Whether temporal decay is active
        allow_propagation: Whether updates propagate to dependencies
        description: Human-readable description
    """
    class_name: str
    alpha: float = 1.0
    min_weight_threshold: float = 0.01
    max_lifetime_hours: Optional[float] = None
    decay_enabled: bool = True
    allow_propagation: bool = True
    description: str = ""


# ============================================================================
# Default Policies
# ============================================================================

DEFAULT_POLICIES = {
    "normal": BeliefClassPolicy(
        class_name="normal",
        alpha=1.0,
        min_weight_threshold=0.01,
        max_lifetime_hours=None,  # No expiry
        decay_enabled=True,
        allow_propagation=True,
        description="Standard beliefs with normal learning rate"
    ),

    "scratch": BeliefClassPolicy(
        class_name="scratch",
        alpha=0.1,  # Low impact
        min_weight_threshold=0.05,  # Higher threshold
        max_lifetime_hours=24.0,  # 24-hour expiry
        decay_enabled=True,
        allow_propagation=False,  # Don't propagate scratch beliefs
        description="Temporary investigation beliefs with minimal impact"
    ),

    "foundational": BeliefClassPolicy(
        class_name="foundational",
        alpha=0.5,
        min_weight_threshold=0.1,  # High threshold
        max_lifetime_hours=None,
        decay_enabled=False,  # No decay
        allow_propagation=True,
        description="Core foundational beliefs resistant to change"
    ),

    "experimental": BeliefClassPolicy(
        class_name="experimental",
        alpha=0.3,
        min_weight_threshold=0.02,
        max_lifetime_hours=168.0,  # 1 week
        decay_enabled=True,
        allow_propagation=False,
        description="Experimental beliefs for testing hypotheses"
    ),
}


# ============================================================================
# Policy Manager
# ============================================================================

class PolicyManager:
    """
    Manages belief update policies.

    Enforces:
    - Minimum weight thresholds (abstention)
    - Belief lifetime limits
    - Class-specific learning rates
    """

    def __init__(self, policies: Optional[Dict[str, BeliefClassPolicy]] = None):
        """
        Initialize policy manager.

        Args:
            policies: Optional custom policies
        """
        self.policies = policies or DEFAULT_POLICIES.copy()

    def get_policy(self, belief_class: str) -> BeliefClassPolicy:
        """
        Get policy for a belief class.

        Args:
            belief_class: Class identifier

        Returns:
            Policy for this class
        """
        return self.policies.get(
            belief_class,
            DEFAULT_POLICIES["normal"]
        )

    def set_policy(self, policy: BeliefClassPolicy):
        """
        Set or update a policy.

        Args:
            policy: Policy configuration
        """
        self.policies[policy.class_name] = policy

    def should_update(self,
                     belief: Belief,
                     weight: float) -> tuple[bool, Optional[str]]:
        """
        Check if belief should be updated (abstention policy).

        Args:
            belief: Belief to check
            weight: Computed update weight

        Returns:
            (should_update, reason_if_not)
        """
        policy = self.get_policy(belief.belief_class)

        # Check weight threshold
        if abs(weight) < policy.min_weight_threshold:
            return False, f"weight {abs(weight):.4f} below threshold {policy.min_weight_threshold}"

        # Check if expired
        if self.is_expired(belief):
            return False, f"belief expired (max lifetime {policy.max_lifetime_hours}h)"

        return True, None

    def is_expired(self, belief: Belief) -> bool:
        """
        Check if belief has expired.

        Args:
            belief: Belief to check

        Returns:
            True if expired
        """
        policy = self.get_policy(belief.belief_class)

        if policy.max_lifetime_hours is None:
            return False

        age = datetime.now() - belief.created_at
        max_age = timedelta(hours=policy.max_lifetime_hours)

        return age > max_age

    def should_propagate(self, belief: Belief) -> bool:
        """
        Check if updates to this belief should propagate.

        Args:
            belief: Belief to check

        Returns:
            True if propagation allowed
        """
        policy = self.get_policy(belief.belief_class)
        return policy.allow_propagation

    def get_alpha(self, belief_class: str) -> float:
        """
        Get learning rate for a belief class.

        Args:
            belief_class: Class identifier

        Returns:
            Alpha (learning rate)
        """
        policy = self.get_policy(belief_class)
        return policy.alpha

    def cleanup_expired(self, beliefs: list[Belief]) -> list[str]:
        """
        Find expired beliefs.

        Args:
            beliefs: Pool of beliefs

        Returns:
            List of expired belief IDs
        """
        expired = []
        for belief in beliefs:
            if self.is_expired(belief):
                expired.append(belief.id)
        return expired

    def get_statistics(self, beliefs: list[Belief]) -> Dict:
        """
        Get policy statistics.

        Args:
            beliefs: Pool of beliefs

        Returns:
            Dict with stats
        """
        # Count by class
        by_class = {}
        for belief in beliefs:
            cls = belief.belief_class
            by_class[cls] = by_class.get(cls, 0) + 1

        # Count expired
        expired_count = len(self.cleanup_expired(beliefs))

        return {
            'total_beliefs': len(beliefs),
            'by_class': by_class,
            'expired_count': expired_count,
            'policies': {
                name: {
                    'alpha': pol.alpha,
                    'min_weight_threshold': pol.min_weight_threshold,
                    'max_lifetime_hours': pol.max_lifetime_hours
                }
                for name, pol in self.policies.items()
            }
        }


# ============================================================================
# Scratch Belief Helpers
# ============================================================================

def create_scratch_belief(content: str,
                         context: str = "investigation",
                         **kwargs) -> Belief:
    """
    Create a scratch belief for temporary investigation.

    Scratch beliefs:
    - Have low learning rate (α = 0.1)
    - Expire after 24 hours
    - Don't propagate to other beliefs

    Args:
        content: Belief content
        context: Context
        **kwargs: Additional Belief parameters

    Returns:
        Scratch Belief
    """
    return Belief(
        content=content,
        context=context,
        belief_class="scratch",
        **kwargs
    )


def is_scratch_belief(belief: Belief) -> bool:
    """Check if belief is a scratch belief."""
    return belief.belief_class == "scratch"


def promote_scratch_to_normal(belief: Belief):
    """
    Promote a scratch belief to normal class.

    Use this when scratch investigation confirms the hypothesis.

    Args:
        belief: Scratch belief to promote
    """
    if belief.belief_class == "scratch":
        belief.belief_class = "normal"
        belief.metadata['promoted_from'] = 'scratch'
        belief.metadata['promoted_at'] = datetime.now().isoformat()


# ============================================================================
# Abstention Helpers
# ============================================================================

class AbstentionTracker:
    """
    Tracks when updates were abstained (not applied).

    Useful for debugging and understanding why beliefs aren't updating.
    """

    def __init__(self):
        """Initialize tracker."""
        self.abstentions: list[Dict] = []

    def record_abstention(self,
                         belief_id: str,
                         weight: float,
                         reason: str):
        """
        Record an abstained update.

        Args:
            belief_id: Belief that wasn't updated
            weight: Weight that was computed
            reason: Reason for abstention
        """
        self.abstentions.append({
            'belief_id': belief_id,
            'weight': weight,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

    def get_abstentions_for_belief(self, belief_id: str) -> list[Dict]:
        """Get all abstentions for a belief."""
        return [
            a for a in self.abstentions
            if a['belief_id'] == belief_id
        ]

    def get_statistics(self) -> Dict:
        """Get abstention statistics."""
        if not self.abstentions:
            return {
                'total': 0,
                'by_reason': {},
                'avg_weight': 0.0
            }

        # Count by reason
        by_reason = {}
        for a in self.abstentions:
            reason = a['reason'].split(':')[0]  # Get reason category
            by_reason[reason] = by_reason.get(reason, 0) + 1

        # Average weight
        avg_weight = sum(abs(a['weight']) for a in self.abstentions) / len(self.abstentions)

        return {
            'total': len(self.abstentions),
            'by_reason': by_reason,
            'avg_weight': avg_weight
        }

    def clear(self):
        """Clear abstention history."""
        self.abstentions.clear()
