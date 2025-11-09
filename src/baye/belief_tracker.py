"""
BeliefTracker - Integrates Update-on-Use (Cogito) with Justification Graphs (Baye)

This module implements the core belief tracking system that combines:
- Update-on-Use: Bayesian updates with pseudo-counts (Beta distribution)
- K-NN Gradient Estimation: Self-supervised learning from semantic neighbors
- Propagation: Causal and semantic belief propagation through justification graphs
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

from .belief_types import Belief, BeliefID, Confidence
from .justification_graph import JustificationGraph
from .belief_estimation import SemanticEstimator


@dataclass
class BeliefUpdate:
    """Result of a belief update operation"""
    belief_id: BeliefID
    old_confidence: Confidence
    new_confidence: Confidence
    pseudo_counts: Tuple[float, float]  # (a, b)
    loss: float  # Training loss
    affected_beliefs: List[BeliefID]
    k_nn_neighbors: List[BeliefID]
    provenance: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingSignal:
    """Training signal for meta-learning"""
    context: str
    p_hat: float  # Agent's estimate
    p_star: float  # Target (signal + K-NN)
    loss: float
    certainty: float  # a + b (pseudo-count sum)
    features: Dict  # Context features for learning


class BeliefTracker:
    """
    Integrates Update-on-Use with Justification Graphs

    Key features:
    - Bayesian updates with Beta distribution pseudo-counts
    - K-NN gradient estimation for self-supervised learning
    - Automatic propagation through justification graph
    - Training signal generation for meta-learning
    """

    def __init__(
        self,
        graph: Optional[JustificationGraph] = None,
        estimator: Optional[SemanticEstimator] = None,
        k_neighbors: int = 5,
        gradient_weight: float = 0.3,  # Weight for K-NN gradient vs signal
    ):
        self.graph = graph or JustificationGraph()
        self.estimator = estimator or SemanticEstimator()
        self.k_neighbors = k_neighbors
        self.gradient_weight = gradient_weight

        # Track pseudo-counts (a, b) for each belief (Beta distribution)
        self.pseudo_counts: Dict[BeliefID, Tuple[float, float]] = {}

        # Training signals for meta-learning
        self.training_signals: List[TrainingSignal] = []

    def add_belief(
        self,
        content: str,
        context: str = "general",
        initial_confidence: Optional[float] = None,
        auto_estimate: bool = True,
        auto_link: bool = True,
    ) -> Belief:
        """
        Add new belief with optional auto-estimation

        Args:
            content: Belief content
            context: Context/domain
            initial_confidence: Manual confidence (if not auto-estimating)
            auto_estimate: Use K-NN estimation if no confidence provided
            auto_link: Auto-link to similar beliefs

        Returns:
            Created belief
        """
        if initial_confidence is None and auto_estimate:
            belief = self.graph.add_belief_with_estimation(
                content=content,
                context=context,
                k=self.k_neighbors,
                auto_link=auto_link,
            )
        else:
            conf = initial_confidence if initial_confidence is not None else 0.5
            belief = self.graph.add_belief(content, conf, context)

        # Initialize pseudo-counts from confidence
        # Use confidence to set initial a, b with total count = 2
        # confidence = a / (a + b), so a = conf * 2, b = (1 - conf) * 2
        a = max(0.1, belief.confidence * 2)
        b = max(0.1, (1 - belief.confidence) * 2)
        self.pseudo_counts[belief.id] = (a, b)

        return belief

    async def update_belief(
        self,
        belief_id: BeliefID,
        p_hat: float,  # Agent's estimate
        signal: float,  # Observed outcome [0, 1]
        r: float = 1.0,  # Relevance weight
        n: float = 1.0,  # Narrative weight
        q: float = 1.0,  # Quality weight
        provenance: Optional[Dict] = None,
    ) -> BeliefUpdate:
        """
        Update belief using Update-on-Use with K-NN gradient estimation

        This implements the core Cogito algorithm:
        1. Update pseudo-counts: a += w * signal, b += w * (1 - signal)
        2. Calculate K-NN gradient: p_star = α * signal + (1-α) * p_knn
        3. Calculate loss: (p_hat - p_star)^2 * certainty
        4. Propagate through graph
        5. Generate training signal

        Args:
            belief_id: ID of belief to update
            p_hat: Agent's confidence estimate
            signal: Observed outcome (0 = failed, 1 = succeeded)
            r, n, q: Weights for relevance, narrative, quality
            provenance: Metadata about update source

        Returns:
            BeliefUpdate with details of the update
        """
        belief = self.graph.beliefs.get(belief_id)
        if not belief:
            raise ValueError(f"Belief {belief_id} not found")

        # Get current pseudo-counts
        old_a, old_b = self.pseudo_counts.get(belief_id, (1.0, 1.0))
        old_confidence = old_a / (old_a + old_b)

        # Calculate update weight
        weight = r * n * q

        # 1. Update pseudo-counts (Update-on-Use)
        new_a = old_a + weight * signal
        new_b = old_b + weight * (1 - signal)
        self.pseudo_counts[belief_id] = (new_a, new_b)

        # 2. K-NN gradient estimation
        neighbors = self._find_knn_neighbors(belief)
        p_knn = np.mean([self.graph.beliefs[nid].confidence
                        for nid in neighbors]) if neighbors else signal

        # Combine signal and K-NN gradient
        alpha = 1 - self.gradient_weight
        p_star = alpha * signal + self.gradient_weight * p_knn

        # 3. Calculate training loss
        certainty = old_a + old_b
        loss = ((p_hat - p_star) ** 2) * certainty

        # 4. Update belief confidence
        new_confidence = new_a / (new_a + new_b)
        belief.confidence = new_confidence

        # 5. Propagate to connected beliefs
        affected = []
        if abs(new_confidence - old_confidence) > 0.01:  # Threshold for propagation
            delta = new_confidence - old_confidence
            prop_result = self.graph.propagate_from(belief_id, delta)
            affected = list({event.belief_id for event in prop_result.events})

        # 6. Generate training signal
        self.training_signals.append(TrainingSignal(
            context=belief.context,
            p_hat=p_hat,
            p_star=p_star,
            loss=loss,
            certainty=certainty,
            features={
                "content": belief.content,
                "k_nn_mean": p_knn,
                "signal": signal,
                "weights": {"r": r, "n": n, "q": q},
            }
        ))

        return BeliefUpdate(
            belief_id=belief_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            pseudo_counts=(new_a, new_b),
            loss=loss,
            affected_beliefs=affected,
            k_nn_neighbors=neighbors,
            provenance=provenance or {},
        )

    def apply_manual_delta(
        self,
        belief_id: BeliefID,
        delta: float,
        provenance: Optional[Dict] = None,
    ) -> BeliefUpdate:
        """
        Apply a direct confidence delta to a belief.

        This is used by the chat tool when the LLM explicitly requests a
        confidence adjustment via `delta`.
        """
        belief = self.graph.beliefs.get(belief_id)
        if not belief:
            raise ValueError(f"Belief {belief_id} not found")

        old_a, old_b = self.pseudo_counts.get(belief_id, (1.0, 1.0))
        old_confidence = belief.confidence

        # Clamp new confidence
        new_confidence = max(0.0, min(1.0, old_confidence + delta))

        # Update belief + pseudo-counts while preserving certainty budget
        total = max(2.0, old_a + old_b)
        new_a = max(0.1, new_confidence * total)
        new_b = max(0.1, total - new_a)

        belief.confidence = new_confidence
        belief.updated_at = datetime.now()
        self.pseudo_counts[belief_id] = (new_a, new_b)

        # Propagate if change is meaningful
        affected = []
        applied_delta = new_confidence - old_confidence
        if abs(applied_delta) > 0.01:
            prop_result = self.graph.propagate_from(belief_id, applied_delta)
            affected = list({event.belief_id for event in prop_result.events})

        neighbors = self._find_knn_neighbors(belief)

        return BeliefUpdate(
            belief_id=belief_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            pseudo_counts=(new_a, new_b),
            loss=0.0,
            affected_beliefs=affected,
            k_nn_neighbors=neighbors,
            provenance=provenance or {},
        )

    def _find_knn_neighbors(self, belief: Belief) -> List[BeliefID]:
        """Find K nearest neighbors for gradient estimation"""
        # Get all other beliefs
        other_beliefs = [b for b in self.graph.beliefs.values()
                        if b.id != belief.id]

        if not other_beliefs:
            return []

        # Use estimator to find similar beliefs
        _, neighbor_ids, _ = self.estimator.estimate_confidence(
            belief.content,
            other_beliefs,
            k=self.k_neighbors,
        )

        return neighbor_ids

    def get_belief_stats(self, belief_id: BeliefID) -> Dict:
        """Get detailed statistics for a belief"""
        belief = self.graph.beliefs.get(belief_id)
        if not belief:
            return {}

        a, b = self.pseudo_counts.get(belief_id, (1.0, 1.0))
        confidence = a / (a + b)
        certainty = a + b

        # Variance of Beta distribution: ab / ((a+b)^2 * (a+b+1))
        variance = (a * b) / ((a + b) ** 2 * (a + b + 1))

        return {
            "id": belief_id,
            "content": belief.content,
            "confidence": confidence,
            "certainty": certainty,
            "variance": variance,
            "pseudo_counts": {"a": a, "b": b},
            "supporters": len(belief.supported_by),
            "contradictors": len(belief.contradicted_by),
        }

    def explain_confidence(self, belief_id: BeliefID) -> Dict:
        """Explain how belief's confidence was derived"""
        stats = self.get_belief_stats(belief_id)
        belief = self.graph.beliefs.get(belief_id)

        if not belief:
            return {}

        # Get K-NN neighbors
        neighbors = self._find_knn_neighbors(belief)
        neighbor_info = [
            self.get_belief_stats(nid) for nid in neighbors
        ]

        # Get graph relationships
        supporters = [self.graph.beliefs[sid] for sid in belief.supported_by]
        contradictors = [self.graph.beliefs[cid] for cid in belief.contradicted_by]

        return {
            "belief": stats,
            "neighbors": neighbor_info,
            "supporters": [{"id": s.id, "content": s.content, "confidence": s.confidence}
                          for s in supporters],
            "contradictors": [{"id": c.id, "content": c.content, "confidence": c.confidence}
                             for c in contradictors],
        }

    def get_training_summary(self) -> Dict:
        """Get summary of training signals"""
        if not self.training_signals:
            return {
                "total_signals": 0,
                "mean_loss": 0,
                "mean_certainty": 0,
            }

        losses = [s.loss for s in self.training_signals]
        certainties = [s.certainty for s in self.training_signals]

        return {
            "total_signals": len(self.training_signals),
            "mean_loss": np.mean(losses),
            "std_loss": np.std(losses),
            "mean_certainty": np.mean(certainties),
            "recent_signals": [
                {
                    "context": s.context,
                    "p_hat": s.p_hat,
                    "p_star": s.p_star,
                    "loss": s.loss,
                }
                for s in self.training_signals[-5:]  # Last 5
            ]
        }
