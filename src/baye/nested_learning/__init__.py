"""
Nested Learning Integration for Baye Belief Tracking System

Based on: "Nested Learning: The Illusion of Deep Learning Architectures"
         (Behrouz et al., NeurIPS 2025)

This module implements nested optimization levels for belief tracking:
- Level 1: Belief confidence updates (immediate)
- Level 2: Learned propagation strategies (medium-term)
- Level 3: Meta-learning of update rules (long-term)

Key Components:
- PropagationMemory: Learnable propagation weights (Deep Optimizers)
- ContinuumMemoryGraph: Online + Offline consolidation
- SelfModifyingBelief: Beliefs that learn their own update rules
- MetaLearner: Meta-optimization across domains
"""

from .propagation_memory import PropagationMemory, PropagationContext
from .continuum_memory import ContinuumMemoryGraph, ConsolidationScheduler
from .self_modifying import SelfModifyingBelief, BeliefUpdateStrategy
from .meta_learner import MetaLearner, MetaOptimizer
from .nested_graph import NestedBeliefGraph

__all__ = [
    'PropagationMemory',
    'PropagationContext',
    'ContinuumMemoryGraph',
    'ConsolidationScheduler',
    'SelfModifyingBelief',
    'BeliefUpdateStrategy',
    'MetaLearner',
    'MetaOptimizer',
    'NestedBeliefGraph',
]

__version__ = "2.0.0-alpha"
