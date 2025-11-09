"""
Propagation Memory: Deep Optimizers for Belief Propagation

Implements the NL insight that optimizers are associative memory modules.
Instead of fixed α=0.7, β=0.3, we learn domain-specific weights with momentum.

Key Idea (from NL paper):
"Well-known gradient-based optimizers (e.g., Adam, SGD with Momentum) are
associative memory modules that aim to compress the gradients."

We apply this to propagation: treat (α, β) as learnable parameters that
compress propagation history into optimal weights for the current context.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque


@dataclass
class PropagationContext:
    """
    Context for a propagation event.
    Used as input to PropagationMemory for weight prediction.
    """
    belief_id: str
    delta: float
    context: str  # Domain (security, performance, etc.)
    graph_depth: int
    num_supporters: int
    num_dependents: int
    avg_neighbor_confidence: float

    def to_vector(self) -> np.ndarray:
        """Encode context as feature vector."""
        # Simple encoding (can be enhanced with learned embeddings)
        return np.array([
            self.delta,
            self.graph_depth / 10.0,  # Normalize
            self.num_supporters / 10.0,
            self.num_dependents / 10.0,
            self.avg_neighbor_confidence,
            1.0 if self.context == "security" else 0.0,
            1.0 if self.context == "performance" else 0.0,
            1.0 if self.context == "reliability" else 0.0,
        ])


@dataclass
class PropagationMemoryEntry:
    """
    A memory entry: (context, weights, outcome).
    """
    context: PropagationContext
    alpha: float
    beta: float
    outcome_surprise: float  # How well did this (α, β) work?
    timestamp: float


class PropagationMemory:
    """
    Associative memory for learning propagation weights.

    Maps: (belief_update_context) → (α, β)

    Inspired by NL's "Deep Optimizers": just as SGD with momentum
    compresses gradient history, we compress propagation history into
    optimal weights.

    Three mechanisms:
    1. K-NN lookup: Find similar past propagations
    2. Momentum: Smooth weight changes like SGD with momentum
    3. Adaptive learning: Update based on propagation surprise
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        memory_size: int = 1000,
        k_neighbors: int = 5
    ):
        """
        Initialize propagation memory.

        Args:
            learning_rate: How fast to adapt weights (like optimizer η)
            momentum: Momentum factor for smooth updates (like SGD momentum)
            memory_size: Max entries in memory bank
            k_neighbors: K for K-NN weight lookup
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.k_neighbors = k_neighbors

        # Memory bank: recent propagation contexts and their outcomes
        self.memory: deque = deque(maxlen=memory_size)

        # Momentum buffers (like m_t in SGD with momentum)
        self.m_alpha = 0.7  # Initialize to Baye V1.5 default
        self.m_beta = 0.3

        # Domain-specific learned weights (cached)
        self.domain_weights: Dict[str, Tuple[float, float]] = {}

        # Statistics for analysis
        self.weight_history: List[Tuple[float, float]] = []
        self.surprise_history: List[float] = []

    def compute_weights(
        self,
        context: PropagationContext
    ) -> Tuple[float, float]:
        """
        Compute adaptive propagation weights for current context.

        Like Adam optimizer: combines K-NN lookup with momentum.

        Returns:
            (alpha, beta): Causal and semantic propagation weights
        """
        # 1. Check if we have domain-specific cached weights
        if context.context in self.domain_weights:
            α_cached, β_cached = self.domain_weights[context.context]

            # Use cached weights with momentum
            self.m_alpha = self.momentum * self.m_alpha + (1 - self.momentum) * α_cached
            self.m_beta = self.momentum * self.m_beta + (1 - self.momentum) * β_cached

            return self.m_alpha, self.m_beta

        # 2. K-NN lookup: find similar past propagations
        if len(self.memory) >= self.k_neighbors:
            neighbors = self._find_k_nearest(context)

            if len(neighbors) > 0:
                # Weighted average based on similarity
                α_knn, β_knn = self._weighted_average(neighbors, context)
            else:
                # No similar neighbors: use current momentum values
                α_knn, β_knn = self.m_alpha, self.m_beta
        else:
            # Not enough memory: use defaults
            α_knn, β_knn = 0.7, 0.3

        # 3. Apply momentum (smooth like SGD with momentum)
        self.m_alpha = self.momentum * self.m_alpha + (1 - self.momentum) * α_knn
        self.m_beta = self.momentum * self.m_beta + (1 - self.momentum) * β_knn

        # 4. Clip to valid range
        self.m_alpha = np.clip(self.m_alpha, 0.1, 0.95)
        self.m_beta = np.clip(self.m_beta, 0.05, 0.9)

        # 5. Ensure α + β doesn't exceed reasonable bound
        total = self.m_alpha + self.m_beta
        if total > 1.2:
            # Normalize
            self.m_alpha = self.m_alpha / total * 1.2
            self.m_beta = self.m_beta / total * 1.2

        # Log for analysis
        self.weight_history.append((self.m_alpha, self.m_beta))

        return self.m_alpha, self.m_beta

    def update(
        self,
        context: PropagationContext,
        alpha: float,
        beta: float,
        outcome_surprise: float
    ):
        """
        Update memory based on propagation outcome.

        This is the "gradient descent" step on the propagation strategy.
        Like training an optimizer with its own optimizer.

        Args:
            context: The propagation context
            alpha, beta: Weights that were used
            outcome_surprise: How surprising was the cascade?
                            (lower = better weights)
        """
        import time

        # 1. Store in memory
        entry = PropagationMemoryEntry(
            context=context,
            alpha=alpha,
            beta=beta,
            outcome_surprise=outcome_surprise,
            timestamp=time.time()
        )
        self.memory.append(entry)

        # 2. Compute optimal weights (gradient descent)
        # If surprise was high: weights were suboptimal
        # Adjust them to reduce surprise

        if outcome_surprise > 0.1:  # Significant surprise
            # Reduce weights to dampen future propagation
            α_optimal = alpha * (1 - self.learning_rate * outcome_surprise)
            β_optimal = beta * (1 - self.learning_rate * outcome_surprise)
        else:
            # Low surprise: reinforce current weights
            α_optimal = alpha * (1 + self.learning_rate * (0.1 - outcome_surprise))
            β_optimal = beta * (1 + self.learning_rate * (0.1 - outcome_surprise))

        # 3. Update domain-specific cache
        self.domain_weights[context.context] = (
            np.clip(α_optimal, 0.1, 0.95),
            np.clip(β_optimal, 0.05, 0.9)
        )

        # 4. Log statistics
        self.surprise_history.append(outcome_surprise)

    def _find_k_nearest(
        self,
        query: PropagationContext
    ) -> List[PropagationMemoryEntry]:
        """
        Find K nearest neighbors using context similarity.
        """
        if len(self.memory) == 0:
            return []

        # Compute similarity to all memory entries
        query_vec = query.to_vector()

        similarities = []
        for entry in self.memory:
            entry_vec = entry.context.to_vector()

            # Cosine similarity
            sim = np.dot(query_vec, entry_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(entry_vec) + 1e-8
            )

            # Bonus for same domain
            if entry.context.context == query.context:
                sim += 0.2

            similarities.append((sim, entry))

        # Sort and take top K
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Filter by minimum similarity threshold
        neighbors = [entry for sim, entry in similarities if sim > 0.5]

        return neighbors[:self.k_neighbors]

    def _weighted_average(
        self,
        neighbors: List[PropagationMemoryEntry],
        query: PropagationContext
    ) -> Tuple[float, float]:
        """
        Compute weighted average of neighbor weights.

        Weights based on:
        1. Similarity to query
        2. Recency (recent memories more important)
        3. Outcome quality (low surprise = good weights)
        """
        query_vec = query.to_vector()

        weighted_alpha = 0.0
        weighted_beta = 0.0
        total_weight = 0.0

        import time
        current_time = time.time()

        for entry in neighbors:
            # Similarity component
            entry_vec = entry.context.to_vector()
            sim = np.dot(query_vec, entry_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(entry_vec) + 1e-8
            )

            # Recency component (exponential decay)
            age_hours = (current_time - entry.timestamp) / 3600
            recency = np.exp(-age_hours / 24)  # Decay over 24 hours

            # Outcome quality (inverse of surprise)
            quality = 1.0 / (1.0 + entry.outcome_surprise)

            # Combined weight
            weight = sim * recency * quality

            weighted_alpha += weight * entry.alpha
            weighted_beta += weight * entry.beta
            total_weight += weight

        if total_weight > 0:
            return weighted_alpha / total_weight, weighted_beta / total_weight
        else:
            return self.m_alpha, self.m_beta

    def get_statistics(self) -> Dict:
        """
        Get statistics about learned weights.
        """
        if len(self.weight_history) == 0:
            return {}

        α_history = [α for α, β in self.weight_history]
        β_history = [β for α, β in self.weight_history]

        return {
            'current_alpha': self.m_alpha,
            'current_beta': self.m_beta,
            'alpha_mean': np.mean(α_history),
            'alpha_std': np.std(α_history),
            'beta_mean': np.mean(β_history),
            'beta_std': np.std(β_history),
            'avg_surprise': np.mean(self.surprise_history) if self.surprise_history else 0.0,
            'memory_size': len(self.memory),
            'domains_learned': list(self.domain_weights.keys()),
        }

    def reset_domain(self, domain: str):
        """Reset learned weights for a specific domain."""
        if domain in self.domain_weights:
            del self.domain_weights[domain]

    def export_weights(self) -> Dict[str, Tuple[float, float]]:
        """Export learned domain-specific weights."""
        return self.domain_weights.copy()

    def import_weights(self, weights: Dict[str, Tuple[float, float]]):
        """Import pre-trained domain weights."""
        self.domain_weights.update(weights)
