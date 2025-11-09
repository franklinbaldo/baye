"""
Meta-Learner: Learn How to Learn

Implements the outermost level of nested optimization: learning the learning process itself.

From NL: Level 3 optimization that learns optimal hyperparameters and strategies
for Level 2 (propagation) and Level 1 (belief updates).
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class DomainStatistics:
    """Statistics for a domain's learning behavior."""
    domain: str
    num_updates: int
    avg_propagation_surprise: float
    avg_update_loss: float
    optimal_alpha: float
    optimal_beta: float
    optimal_learning_rate: float


class MetaOptimizer:
    """
    Optimizes the optimization process.

    Learns:
    - Initial propagation weights (α, β) per domain
    - Learning rates for different update types
    - Consolidation schedules
    """

    def __init__(self):
        # Domain-specific learned hyperparameters
        self.domain_hyperparameters: Dict[str, Dict] = {}

        # Global statistics
        self.episodes_processed = 0

    def learn_domain_hyperparameters(
        self,
        domain: str,
        propagation_history: List,
        update_history: List
    ) -> Dict:
        """
        Learn optimal hyperparameters for a domain.

        Args:
            domain: Domain name
            propagation_history: History of propagation outcomes
            update_history: History of belief updates

        Returns:
            Learned hyperparameters
        """
        # Extract statistics from history
        if propagation_history:
            avg_surprise = np.mean([
                getattr(h, 'outcome_surprise', 0) for h in propagation_history
            ])

            # Learn optimal α, β
            # Find values that minimize surprise
            alphas = [getattr(h, 'alpha', 0.7) for h in propagation_history]
            betas = [getattr(h, 'beta', 0.3) for h in propagation_history]
            surprises = [getattr(h, 'outcome_surprise', 0) for h in propagation_history]

            # Simple heuristic: average weights from low-surprise episodes
            low_surprise = [i for i, s in enumerate(surprises) if s < np.median(surprises)]

            if low_surprise:
                optimal_alpha = np.mean([alphas[i] for i in low_surprise])
                optimal_beta = np.mean([betas[i] for i in low_surprise])
            else:
                optimal_alpha = 0.7
                optimal_beta = 0.3
        else:
            avg_surprise = 0.0
            optimal_alpha = 0.7
            optimal_beta = 0.3

        if update_history:
            avg_loss = np.mean([
                getattr(h, 'loss', 0) for h in update_history
            ])
        else:
            avg_loss = 0.0

        # Learn optimal learning rate
        # Heuristic: higher loss → need higher learning rate
        if avg_loss > 0.1:
            optimal_lr = 0.02  # Increase learning rate
        elif avg_loss < 0.01:
            optimal_lr = 0.005  # Decrease learning rate (fine-tuning)
        else:
            optimal_lr = 0.01  # Default

        # Store learned hyperparameters
        self.domain_hyperparameters[domain] = {
            'alpha_init': optimal_alpha,
            'beta_init': optimal_beta,
            'learning_rate': optimal_lr,
            'avg_surprise': avg_surprise,
            'avg_loss': avg_loss,
        }

        return self.domain_hyperparameters[domain]

    def get_initial_weights(self, domain: str) -> tuple:
        """
        Get learned initial weights for a domain.

        Args:
            domain: Domain name

        Returns:
            (alpha, beta) tuple
        """
        if domain in self.domain_hyperparameters:
            params = self.domain_hyperparameters[domain]
            return params['alpha_init'], params['beta_init']
        else:
            # Default
            return 0.7, 0.3

    def get_learning_rate(self, domain: str) -> float:
        """Get learned learning rate for a domain."""
        if domain in self.domain_hyperparameters:
            return self.domain_hyperparameters[domain]['learning_rate']
        else:
            return 0.01

    def get_statistics(self) -> Dict:
        """Get meta-learning statistics."""
        return {
            'domains_learned': list(self.domain_hyperparameters.keys()),
            'hyperparameters': self.domain_hyperparameters.copy(),
            'episodes_processed': self.episodes_processed,
        }


class MetaLearner:
    """
    Meta-learner that consolidates learning across domains.

    Runs periodically to optimize the entire learning system.
    """

    def __init__(self, consolidation_interval: int = 100):
        """
        Args:
            consolidation_interval: Number of updates between meta-learning
        """
        self.consolidation_interval = consolidation_interval
        self.update_count = 0

        # Meta-optimizer
        self.optimizer = MetaOptimizer()

        # Domain-specific histories
        self.domain_histories: Dict[str, Dict] = {}

    async def consolidate(
        self,
        propagation_history: List,
        outcomes: Dict
    ):
        """
        Meta-learning consolidation step.

        Analyzes recent learning and optimizes hyperparameters.

        Args:
            propagation_history: Recent propagation memory
            outcomes: Outcomes by domain
        """
        # Group by domain
        domains = set()
        for entry in propagation_history:
            if hasattr(entry, 'context') and hasattr(entry.context, 'context'):
                domains.add(entry.context.context)

        # Learn hyperparameters for each domain
        for domain in domains:
            # Filter history for this domain
            domain_prop_history = [
                e for e in propagation_history
                if hasattr(e, 'context') and hasattr(e.context, 'context') and
                e.context.context == domain
            ]

            domain_update_history = outcomes.get(domain, [])

            # Learn optimal hyperparameters
            self.optimizer.learn_domain_hyperparameters(
                domain=domain,
                propagation_history=domain_prop_history,
                update_history=domain_update_history
            )

        self.optimizer.episodes_processed += 1

    def get_meta_statistics(self) -> Dict:
        """Get meta-learning statistics."""
        return self.optimizer.get_statistics()
