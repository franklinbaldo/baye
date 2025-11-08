"""
Baye: Neural-symbolic belief maintenance system.

A justification-based belief tracking system combining causal deterministic
tracking with probabilistic semantic propagation, powered by LLMs.
"""

__version__ = "1.5.0"

# Core types
from .belief_types import (
    Belief,
    BeliefID,
    Confidence,
    Delta,
    PropagationEvent,
    PropagationResult,
    RelationType,
)

# Graph and estimation
from .justification_graph import JustificationGraph
from .belief_estimation import SemanticEstimator

# LLM agents
from .llm_agents import (
    RelationshipAnalysis,
    ConflictResolution,
    detect_relationship,
    resolve_conflict,
    find_related_beliefs,
    generate_embedding,
    check_gemini_api_key,
)

__all__ = [
    # Version
    "__version__",
    # Types
    "Belief",
    "BeliefID",
    "Confidence",
    "Delta",
    "PropagationEvent",
    "PropagationResult",
    "RelationType",
    # Core classes
    "JustificationGraph",
    "SemanticEstimator",
    # LLM models
    "RelationshipAnalysis",
    "ConflictResolution",
    # LLM functions
    "detect_relationship",
    "resolve_conflict",
    "find_related_beliefs",
    "generate_embedding",
    "check_gemini_api_key",
]
