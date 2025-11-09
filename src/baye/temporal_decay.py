"""
Temporal decay system for belief evidence (US-04).

Applies decay λ to Beta parameters (a, b) to give more weight to recent evidence.
Decay formula: a' = 1 + (a - 1) * (1 - λ)

Features:
- Configurable decay rates per belief class
- Time-based automatic decay
- Policy-based enabling/disabling
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from .belief_types import Belief, BeliefID


# ============================================================================
# Decay Policy
# ============================================================================

@dataclass
class DecayPolicy:
    """
    Policy for temporal decay of beliefs.

    Attributes:
        belief_class: Class of belief this applies to
        lambda_per_day: Decay rate per day (0 = no decay, 1 = full reset per day)
        half_life_days: Alternative parameterization (days until 50% decay)
        enabled: Whether decay is active
        min_evidence_count: Only decay if evidence count > this
    """
    belief_class: str
    lambda_per_day: float = 0.0
    half_life_days: Optional[float] = None
    enabled: bool = True
    min_evidence_count: float = 2.0  # Don't decay very weak beliefs

    def __post_init__(self):
        """Calculate lambda from half-life if provided."""
        if self.half_life_days is not None:
            # λ = 1 - exp(-ln(2) / half_life)
            # Approximation: λ ≈ ln(2) / half_life for small decay
            import math
            self.lambda_per_day = 1.0 - math.exp(-math.log(2) / self.half_life_days)

    def calculate_decay_factor(self, days_elapsed: float) -> float:
        """
        Calculate decay factor for elapsed time.

        Args:
            days_elapsed: Days since last decay

        Returns:
            Decay factor λ ∈ [0, 1]
        """
        if not self.enabled:
            return 0.0

        # Total decay = 1 - (1 - λ_per_day)^days
        import math
        total_decay = 1.0 - math.pow(1.0 - self.lambda_per_day, days_elapsed)
        return total_decay


# ============================================================================
# Decay Manager
# ============================================================================

class DecayManager:
    """
    Manages temporal decay for beliefs.

    Default policies:
    - normal: 30-day half-life (slow decay)
    - scratch: 7-day half-life (fast decay)
    - foundational: no decay
    """

    DEFAULT_POLICIES = {
        "normal": DecayPolicy(
            belief_class="normal",
            half_life_days=30.0,  # Slow decay
            enabled=True
        ),
        "scratch": DecayPolicy(
            belief_class="scratch",
            half_life_days=7.0,  # Fast decay
            enabled=True
        ),
        "foundational": DecayPolicy(
            belief_class="foundational",
            lambda_per_day=0.0,  # No decay
            enabled=False
        ),
    }

    def __init__(self, policies: Optional[Dict[str, DecayPolicy]] = None):
        """
        Initialize decay manager.

        Args:
            policies: Optional custom policies by belief class
        """
        self.policies = policies or self.DEFAULT_POLICIES.copy()

    def get_policy(self, belief_class: str) -> DecayPolicy:
        """
        Get decay policy for belief class.

        Args:
            belief_class: Belief class identifier

        Returns:
            DecayPolicy for this class
        """
        return self.policies.get(
            belief_class,
            self.DEFAULT_POLICIES.get("normal", DecayPolicy(belief_class="default"))
        )

    def set_policy(self, belief_class: str, policy: DecayPolicy):
        """
        Set policy for a belief class.

        Args:
            belief_class: Class identifier
            policy: Decay policy
        """
        self.policies[belief_class] = policy

    def should_decay(self, belief: Belief) -> bool:
        """
        Check if belief should be decayed.

        Args:
            belief: Belief to check

        Returns:
            True if decay should be applied
        """
        policy = self.get_policy(belief.belief_class)

        if not policy.enabled:
            return False

        # Don't decay very weak beliefs (let them die naturally)
        if belief.total_evidence < policy.min_evidence_count:
            return False

        # Check if enough time has passed (at least 1 day)
        time_since_decay = datetime.now() - belief.last_decay_at
        if time_since_decay < timedelta(days=1):
            return False

        return True

    def apply_decay(self, belief: Belief, force: bool = False) -> float:
        """
        Apply decay to a belief.

        Args:
            belief: Belief to decay (modified in place)
            force: Force decay even if policy says no

        Returns:
            Decay factor applied (0 = no decay, 1 = full reset)
        """
        if not force and not self.should_decay(belief):
            return 0.0

        policy = self.get_policy(belief.belief_class)

        # Calculate time elapsed
        time_since_decay = datetime.now() - belief.last_decay_at
        days_elapsed = time_since_decay.total_seconds() / 86400.0

        # Calculate decay factor
        decay_factor = policy.calculate_decay_factor(days_elapsed)

        # Apply decay
        belief.apply_decay(decay_factor)

        return decay_factor

    def batch_decay(self,
                   beliefs: List[Belief],
                   force: bool = False) -> Dict[BeliefID, float]:
        """
        Apply decay to multiple beliefs.

        Args:
            beliefs: List of beliefs to decay
            force: Force decay even if policies say no

        Returns:
            Dict mapping belief_id -> decay_factor applied
        """
        results = {}

        for belief in beliefs:
            decay_factor = self.apply_decay(belief, force=force)
            if decay_factor > 0:
                results[belief.id] = decay_factor

        return results

    def enable_decay(self, belief_class: str):
        """Enable decay for a belief class."""
        if belief_class in self.policies:
            self.policies[belief_class].enabled = True

    def disable_decay(self, belief_class: str):
        """Disable decay for a belief class."""
        if belief_class in self.policies:
            self.policies[belief_class].enabled = False

    def set_half_life(self, belief_class: str, half_life_days: float):
        """
        Set half-life for a belief class.

        Args:
            belief_class: Class identifier
            half_life_days: Days until 50% decay
        """
        if belief_class not in self.policies:
            self.policies[belief_class] = DecayPolicy(
                belief_class=belief_class,
                half_life_days=half_life_days
            )
        else:
            # Update existing policy
            policy = self.policies[belief_class]
            policy.half_life_days = half_life_days
            policy.__post_init__()  # Recalculate lambda

    def get_statistics(self, beliefs: List[Belief]) -> Dict:
        """
        Get decay statistics.

        Args:
            beliefs: Beliefs to analyze

        Returns:
            Dict with stats
        """
        total = len(beliefs)
        eligible_for_decay = sum(1 for b in beliefs if self.should_decay(b))

        avg_time_since_decay = sum(
            (datetime.now() - b.last_decay_at).total_seconds() / 86400.0
            for b in beliefs
        ) / total if total > 0 else 0.0

        return {
            'total_beliefs': total,
            'eligible_for_decay': eligible_for_decay,
            'avg_days_since_decay': avg_time_since_decay,
            'enabled_classes': [
                cls for cls, pol in self.policies.items() if pol.enabled
            ]
        }


# ============================================================================
# Auto-Decay Scheduler
# ============================================================================

class AutoDecayScheduler:
    """
    Automatically schedules decay for beliefs.

    Can be run periodically (e.g., daily cron job).
    """

    def __init__(self, decay_manager: DecayManager):
        """
        Initialize scheduler.

        Args:
            decay_manager: Decay manager to use
        """
        self.decay_manager = decay_manager
        self.last_run: Optional[datetime] = None

    def run(self, beliefs: List[Belief]) -> Dict:
        """
        Run scheduled decay.

        Args:
            beliefs: All beliefs in system

        Returns:
            Dict with run statistics
        """
        start_time = datetime.now()

        # Apply decay
        decay_results = self.decay_manager.batch_decay(beliefs)

        # Record run
        self.last_run = start_time

        return {
            'timestamp': start_time.isoformat(),
            'beliefs_processed': len(beliefs),
            'beliefs_decayed': len(decay_results),
            'avg_decay_factor': sum(decay_results.values()) / len(decay_results)
                               if decay_results else 0.0,
            'duration_ms': (datetime.now() - start_time).total_seconds() * 1000
        }

    def should_run(self, interval_hours: float = 24.0) -> bool:
        """
        Check if scheduler should run.

        Args:
            interval_hours: Hours between runs

        Returns:
            True if it's time to run
        """
        if self.last_run is None:
            return True

        elapsed = datetime.now() - self.last_run
        return elapsed >= timedelta(hours=interval_hours)
