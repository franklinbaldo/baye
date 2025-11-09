"""
Self-Modifying Beliefs: Beliefs that Learn Their Own Update Rules

Implements NL's "Self-Modifying Titans": models that learn their own update algorithms.

Key Insight (from NL Section 2):
"A novel sequence model that learns how to modify itself by learning its own
update algorithm."

We apply this to beliefs: each belief learns a domain-specific update strategy.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class UpdateOutcome:
    """Record of an update and its outcome."""
    signal: float
    r: float  # Reliability
    n: float  # Novelty
    q: float  # Quality
    predicted_change: float
    actual_change: float
    loss: float


class BeliefUpdateStrategy:
    """
    Learnable update rule for a belief.

    Instead of fixed Bayesian update:
        a += r * n * q * signal

    We learn a modification function:
        a += (r * n * q * signal) + learned_modification(context)
    """

    def __init__(self, belief_id: str, domain: str):
        """
        Args:
            belief_id: ID of belief this strategy belongs to
            domain: Domain context (security, performance, etc.)
        """
        self.belief_id = belief_id
        self.domain = domain

        # Learnable parameters (simple linear model to start)
        # More complex: could use neural network
        self.params = {
            'signal_amplification': 1.0,  # Multiply signal by this
            'conservatism_bias': 0.0,     # Add/subtract from update
            'reliability_sensitivity': 1.0,  # Weight for r
            'novelty_sensitivity': 1.0,      # Weight for n
            'quality_sensitivity': 1.0,      # Weight for q
        }

        # Learning rate for parameter updates
        self.learning_rate = 0.01

        # History of updates for analysis
        self.update_history: List[UpdateOutcome] = []

    def compute_modification(
        self,
        signal: float,
        r: float,
        n: float,
        q: float,
        current_confidence: float,
        recent_updates: List[float]
    ) -> float:
        """
        Compute learned modification to standard update.

        Args:
            signal: Observed signal [0, 1]
            r: Reliability
            n: Novelty
            q: Quality
            current_confidence: Current belief confidence
            recent_updates: Last few confidence changes

        Returns:
            Modification to add to standard update
        """
        # Standard update (baseline)
        standard_weight = r * n * q

        # Apply learned parameters
        amplified_signal = signal * self.params['signal_amplification']
        weighted_r = r * self.params['reliability_sensitivity']
        weighted_n = n * self.params['novelty_sensitivity']
        weighted_q = q * self.params['quality_sensitivity']

        modified_weight = weighted_r * weighted_n * weighted_q

        # Modification = difference from standard
        modification = (modified_weight - standard_weight) * amplified_signal

        # Add conservatism bias
        # Positive bias: more optimistic (increase confidence faster)
        # Negative bias: more conservative (slower updates)
        modification += self.params['conservatism_bias'] * amplified_signal

        # Domain-specific logic
        if self.domain == "security":
            # Security: conservative on positive signals, aggressive on negative
            if signal > 0.5:
                modification *= 0.8  # More evidence needed for positive
            else:
                modification *= 1.2  # Quick to reduce confidence on threats

        elif self.domain == "performance":
            # Performance: more optimistic, try optimizations eagerly
            if signal > 0.5:
                modification *= 1.1  # Slightly boost positive signals

        # Clip modification to prevent instability
        return np.clip(modification, -0.1, 0.1)

    def learn_from_outcome(
        self,
        signal: float,
        r: float,
        n: float,
        q: float,
        predicted_change: float,
        actual_change: float
    ):
        """
        Update strategy parameters based on outcome.

        Uses simple gradient descent on prediction error.

        Args:
            signal, r, n, q: Update inputs
            predicted_change: What we predicted the confidence change would be
            actual_change: Actual confidence change observed
        """
        # Compute loss (mean squared error)
        loss = (predicted_change - actual_change) ** 2

        # Record outcome
        outcome = UpdateOutcome(
            signal=signal,
            r=r,
            n=n,
            q=q,
            predicted_change=predicted_change,
            actual_change=actual_change,
            loss=loss
        )
        self.update_history.append(outcome)

        # Gradient descent on parameters
        # Simple approximation: adjust parameters to reduce error

        # If we over-predicted: reduce amplification/bias
        # If we under-predicted: increase them
        error = predicted_change - actual_change

        # Update signal amplification
        # If signal was strong and we over-predicted: reduce amplification
        if abs(signal - 0.5) > 0.3:  # Strong signal
            self.params['signal_amplification'] -= (
                self.learning_rate * error * signal
            )
            self.params['signal_amplification'] = np.clip(
                self.params['signal_amplification'], 0.5, 2.0
            )

        # Update conservatism bias
        self.params['conservatism_bias'] -= self.learning_rate * error * 0.1
        self.params['conservatism_bias'] = np.clip(
            self.params['conservatism_bias'], -0.2, 0.2
        )

        # Update sensitivity parameters based on which factors were present
        if r > 0.8:  # High reliability
            self.params['reliability_sensitivity'] -= (
                self.learning_rate * error * 0.05
            )
        if n > 0.8:  # High novelty
            self.params['novelty_sensitivity'] -= (
                self.learning_rate * error * 0.05
            )
        if q > 0.8:  # High quality
            self.params['quality_sensitivity'] -= (
                self.learning_rate * error * 0.05
            )

        # Clip sensitivities
        for key in ['reliability_sensitivity', 'novelty_sensitivity', 'quality_sensitivity']:
            self.params[key] = np.clip(self.params[key], 0.1, 2.0)

    def get_statistics(self) -> Dict:
        """Get statistics about learned strategy."""
        if not self.update_history:
            return {
                'num_updates': 0,
                'avg_loss': 0.0,
                'params': self.params.copy()
            }

        return {
            'num_updates': len(self.update_history),
            'avg_loss': np.mean([o.loss for o in self.update_history]),
            'recent_loss': np.mean([o.loss for o in self.update_history[-10:]]) if len(self.update_history) >= 10 else 0.0,
            'params': self.params.copy(),
            'domain': self.domain,
        }


class SelfModifyingBelief:
    """
    Belief that learns its own update rule.

    Extends base Belief with a learnable update strategy.
    """

    def __init__(self, base_belief, domain: str = "general"):
        """
        Args:
            base_belief: Base Belief object to enhance
            domain: Domain context for strategy learning
        """
        self.belief = base_belief
        self.update_strategy = BeliefUpdateStrategy(
            belief_id=base_belief.id,
            domain=domain
        )

        # Track update history for learning
        self.recent_confidences: List[float] = [base_belief.confidence]
        self.recent_updates: List[float] = []

    def update(self, signal: float, r: float = 1.0, n: float = 1.0, q: float = 1.0):
        """
        Update belief using learned strategy.

        Args:
            signal: Observed signal [0, 1]
            r: Reliability weight
            n: Novelty weight
            q: Quality weight
        """
        # Store old confidence
        old_confidence = self.belief.confidence

        # Standard Bayesian update (if belief has pseudo-counts)
        if hasattr(self.belief, 'a') and hasattr(self.belief, 'b'):
            # Standard Update-on-Use
            weight = r * n * q
            self.belief.a += weight * signal
            self.belief.b += weight * (1 - signal)

            standard_new_conf = self.belief.a / (self.belief.a + self.belief.b)

            # Compute learned modification
            modification = self.update_strategy.compute_modification(
                signal=signal,
                r=r,
                n=n,
                q=q,
                current_confidence=old_confidence,
                recent_updates=self.recent_updates
            )

            # Apply modification
            modified_new_conf = np.clip(standard_new_conf + modification, 0.0, 1.0)

            # Update confidence
            # Back-calculate pseudo-counts that would give this confidence
            total = self.belief.a + self.belief.b
            self.belief.a = modified_new_conf * total
            self.belief.b = (1 - modified_new_conf) * total

            self.belief.confidence = modified_new_conf

        else:
            # Simple confidence update (no pseudo-counts)
            weight = r * n * q
            standard_delta = weight * (signal - old_confidence)

            modification = self.update_strategy.compute_modification(
                signal=signal,
                r=r,
                n=n,
                q=q,
                current_confidence=old_confidence,
                recent_updates=self.recent_updates
            )

            new_confidence = np.clip(
                old_confidence + standard_delta + modification,
                0.0,
                1.0
            )
            self.belief.confidence = new_confidence

        # Learn from outcome (simple version: use confidence change as outcome)
        predicted_change = modification
        actual_change = self.belief.confidence - old_confidence

        self.update_strategy.learn_from_outcome(
            signal=signal,
            r=r,
            n=n,
            q=q,
            predicted_change=predicted_change,
            actual_change=actual_change
        )

        # Track history
        self.recent_confidences.append(self.belief.confidence)
        self.recent_updates.append(actual_change)

        # Keep only recent history
        if len(self.recent_confidences) > 20:
            self.recent_confidences = self.recent_confidences[-20:]
            self.recent_updates = self.recent_updates[-20:]

    def get_strategy_stats(self) -> Dict:
        """Get statistics about the learned strategy."""
        return self.update_strategy.get_statistics()

    @property
    def id(self):
        return self.belief.id

    @property
    def content(self):
        return self.belief.content

    @property
    def confidence(self):
        return self.belief.confidence

    @property
    def context(self):
        return self.belief.context

    def __getattr__(self, name):
        """Delegate attribute access to base belief."""
        return getattr(self.belief, name)
