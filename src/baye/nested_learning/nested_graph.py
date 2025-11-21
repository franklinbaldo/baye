"""
Nested Belief Graph: Complete 3-Level Nested Optimization

Integrates all nested learning components:
- Level 1: Belief updates (immediate)
- Level 2: Learned propagation (medium-term)
- Level 3: Meta-learning (long-term)

Plus:
- Continuum memory (online + offline consolidation)
- Self-modifying beliefs (learned update rules)
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from .propagation_memory import PropagationMemory, PropagationContext
from .continuum_memory import ContinuumMemoryGraph
from .self_modifying import SelfModifyingBelief
from .meta_learner import MetaLearner


class NestedBeliefGraph:
    """
    Complete nested learning architecture for belief tracking.

    Combines:
    1. Base justification graph (Level 1)
    2. Propagation memory with learned weights (Level 2)
    3. Meta-learner for hyperparameter optimization (Level 3)
    4. Continuum memory for consolidation
    5. Self-modifying beliefs
    """

    def __init__(self, base_graph, enable_all_features: bool = True):
        """
        Args:
            base_graph: Base JustificationGraph instance
            enable_all_features: Enable all NL features
        """
        from ..justification_graph import JustificationGraph

        if not isinstance(base_graph, JustificationGraph):
            raise TypeError("base_graph must be a JustificationGraph")

        # Base graph
        self.graph = base_graph

        # Level 2: Propagation memory (deep optimizers)
        self.propagation_memory = PropagationMemory(
            learning_rate=0.01,
            momentum=0.9,
            k_neighbors=5
        )

        # Continuum memory (online + offline)
        self.continuum = ContinuumMemoryGraph(
            base_graph=base_graph,
            consolidation_interval=60.0  # 1 minute
        )

        # Level 3: Meta-learner
        self.meta_learner = MetaLearner(consolidation_interval=100)

        # Self-modifying beliefs registry
        self.self_modifying_beliefs: Dict[str, SelfModifyingBelief] = {}

        # Configuration
        self.enable_all_features = enable_all_features

        # Statistics
        self.update_count = 0
        self.propagation_outcomes: Dict[str, List] = {}

    async def update_belief_nested(
        self,
        belief_id: str,
        signal: float,
        context: str = "general",
        r: float = 1.0,
        n: float = 1.0,
        q: float = 1.0,
        enable_self_modification: bool = True
    ):
        """
        Three-level nested update.

        Args:
            belief_id: Belief to update
            signal: Observed signal [0, 1]
            context: Domain context
            r, n, q: Reliability, novelty, quality weights
            enable_self_modification: Use self-modifying update rule

        Returns:
            Update result with all nested information
        """
        # Get belief
        belief = self.graph.beliefs.get(belief_id)
        if not belief:
            raise ValueError(f"Belief {belief_id} not found")

        old_confidence = belief.confidence

        # LEVEL 1: Belief update
        if enable_self_modification and self.enable_all_features:
            # Use self-modifying belief
            if belief_id not in self.self_modifying_beliefs:
                self.self_modifying_beliefs[belief_id] = SelfModifyingBelief(
                    base_belief=belief,
                    domain=context
                )

            self_modifying = self.self_modifying_beliefs[belief_id]
            self_modifying.update(signal, r, n, q)

        else:
            # Standard update
            if hasattr(belief, 'update_confidence'):
                delta = signal - old_confidence
                belief.update_confidence(delta)
            else:
                # Simple confidence update
                belief.confidence = signal

        new_confidence = belief.confidence
        delta = new_confidence - old_confidence

        # LEVEL 2: Propagation with learned weights
        # Get graph context for propagation memory
        prop_context = PropagationContext(
            belief_id=belief_id,
            delta=delta,
            context=context,
            graph_depth=self._get_graph_depth(belief_id),
            num_supporters=len(getattr(belief, 'supporters', [])),
            num_dependents=len(getattr(belief, 'dependents', [])),
            avg_neighbor_confidence=self._get_avg_neighbor_confidence(belief_id)
        )

        # Compute learned weights
        alpha, beta = self.propagation_memory.compute_weights(prop_context)

        # Propagate with learned weights (via continuum memory for consolidation)
        if self.enable_all_features:
            result = await self.continuum.update_belief_continuum(
                belief_id=belief_id,
                delta=delta,
                context=context
            )
        else:
            # Direct propagation
            result = self.graph.propagate_from(
                origin_id=belief_id,
                initial_delta=delta
            )

        # Measure propagation surprise (for learning)
        surprise = self._compute_propagation_surprise(result, delta)

        # Update propagation memory (Level 2 learning)
        self.propagation_memory.update(
            context=prop_context,
            alpha=alpha,
            beta=beta,
            outcome_surprise=surprise
        )

        # Track outcomes for meta-learning
        if context not in self.propagation_outcomes:
            self.propagation_outcomes[context] = []
        self.propagation_outcomes[context].append({
            'signal': signal,
            'delta': delta,
            'surprise': surprise,
        })

        # LEVEL 3: Meta-learning (every N updates)
        self.update_count += 1
        if self.update_count % 100 == 0:
            await self.meta_learner.consolidate(
                propagation_history=list(self.propagation_memory.memory),
                outcomes=self.propagation_outcomes
            )

        return {
            'belief_id': belief_id,
            'old_confidence': old_confidence,
            'new_confidence': new_confidence,
            'delta': delta,
            'learned_weights': (alpha, beta),
            'propagation_result': result,
            'propagation_surprise': surprise,
            'update_count': self.update_count,
        }

    def _get_graph_depth(self, belief_id: str) -> int:
        """Get depth of belief in graph (max distance from root)."""
        # Simple heuristic: count supporters recursively
        visited = set()

        def depth(bid):
            if bid in visited:
                return 0
            visited.add(bid)

            belief = self.graph.beliefs.get(bid)
            if not belief:
                return 0

            supporters = getattr(belief, 'supporters', [])
            if not supporters:
                return 0

            return 1 + max((depth(s) for s in supporters), default=0)

        return depth(belief_id)

    def _get_avg_neighbor_confidence(self, belief_id: str) -> float:
        """Get average confidence of neighboring beliefs."""
        belief = self.graph.beliefs.get(belief_id)
        if not belief:
            return 0.5

        neighbors = []
        neighbors.extend(getattr(belief, 'supporters', []))
        neighbors.extend(getattr(belief, 'dependents', []))

        if not neighbors:
            return 0.5

        confidences = []
        for neighbor_id in neighbors:
            neighbor = self.graph.beliefs.get(neighbor_id)
            if neighbor:
                confidences.append(getattr(neighbor, 'confidence', 0.5))

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _compute_propagation_surprise(self, result, initial_delta: float) -> float:
        """
        Compute how surprising the propagation outcome was.

        Lower surprise = better propagation weights.
        """
        if not result:
            return 0.0

        # Heuristic: surprise based on cascade size vs initial delta
        total_updated = getattr(result, 'total_beliefs_updated', 0)
        max_depth = getattr(result, 'max_depth_reached', 0)

        # Expected cascade size (rough heuristic)
        expected_cascade = abs(initial_delta) * 10

        # Surprise = deviation from expected
        actual_cascade = total_updated
        surprise = abs(actual_cascade - expected_cascade) / (expected_cascade + 1)

        return min(surprise, 1.0)  # Clip to [0, 1]

    def get_nested_statistics(self) -> Dict:
        """
        Get statistics across all nested levels.
        """
        return {
            'level_1_updates': self.update_count,
            'level_2_propagation': self.propagation_memory.get_statistics(),
            'level_3_meta': self.meta_learner.get_meta_statistics(),
            'continuum_memory': self.continuum.get_consolidation_status(),
            'self_modifying_beliefs': len(self.self_modifying_beliefs),
            'self_modifying_stats': {
                bid: smb.get_strategy_stats()
                for bid, smb in self.self_modifying_beliefs.items()
            }
        }

    async def start_background_consolidation(self):
        """Start background consolidation loop."""
        async def consolidation_loop():
            while True:
                await asyncio.sleep(60)  # Every minute
                await self.continuum.consolidate_offline()

        return asyncio.create_task(consolidation_loop())

    def export_learned_parameters(self) -> Dict:
        """
        Export all learned parameters for persistence.
        """
        return {
            'propagation_weights': self.propagation_memory.export_weights(),
            'meta_hyperparameters': self.meta_learner.get_meta_statistics(),
            'self_modifying_strategies': {
                bid: smb.get_strategy_stats()
                for bid, smb in self.self_modifying_beliefs.items()
            },
        }

    def import_learned_parameters(self, params: Dict):
        """
        Import learned parameters from previous session.
        """
        if 'propagation_weights' in params:
            self.propagation_memory.import_weights(params['propagation_weights'])

        # TODO: Import meta-hyperparameters and strategies
