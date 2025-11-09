"""
Baye: Neural-symbolic belief maintenance system with Update-on-Use.

A justification-based belief tracking system combining:
- Bayesian belief updating (Beta distribution)
- Update-on-Use from tool calls
- Semantic retrieval with MMR
- Temporal decay and watchers
- Full observability and audit trail

Version 2.0.0 - Update-on-Use + Retrieval Epic
"""

__version__ = "2.0.0"

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

# ============================================================================
# NEW: Update-on-Use + Retrieval System (Epic v2.0)
# ============================================================================

# Evidence system (US-01, US-03)
from .evidence import (
    Evidence,
    EvidenceStore,
    EvidenceUpdate,
    UpdateOnUseEngine,
)

# Reliability catalog (US-02)
from .reliability_catalog import (
    ReliabilityProfile,
    ReliabilityCatalog,
    create_default_catalog,
)

# Temporal decay (US-04)
from .temporal_decay import (
    DecayPolicy,
    DecayManager,
    AutoDecayScheduler,
)

# Watchers (US-05)
from .watchers import (
    WatcherConfig,
    WatcherEvent,
    WatcherSystem,
    ThresholdType,
    Action,
)

# Retrieval (US-06, US-07, US-08)
from .retrieval import (
    CandidateBelief,
    CandidateGenerator,
    BeliefRanker,
    TensionPair,
    TensionDetector,
)

# Context builder (US-09)
from .context_builder import (
    BeliefCard,
    BeliefCardFormatter,
    ContextPackBuilder,
    create_simple_context,
)

# Policies (US-10)
from .policies import (
    BeliefClassPolicy,
    PolicyManager,
    create_scratch_belief,
    AbstentionTracker,
)

# I18n (US-11)
from .i18n import (
    TranslationTemplate,
    CardTranslator,
    detect_language,
    auto_translate_context_pack,
)

# Observability (US-12)
from .observability import (
    AuditLogEntry,
    MetricsSnapshot,
    MetricsTracker,
    AuditLogger,
    BeliefObserver,
)

# Main API (US-14)
from .api import (
    FeatureFlags,
    BeliefSystem,
    UpdateOnUseTool,
    create_belief_system,
)

__all__ = [
    # Version
    "__version__",

    # Core types
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

    # LLM
    "RelationshipAnalysis",
    "ConflictResolution",
    "detect_relationship",
    "resolve_conflict",
    "find_related_beliefs",
    "generate_embedding",
    "check_gemini_api_key",

    # ========================================================================
    # UPDATE-ON-USE + RETRIEVAL SYSTEM (v2.0)
    # ========================================================================

    # Evidence (US-01, US-03)
    "Evidence",
    "EvidenceStore",
    "EvidenceUpdate",
    "UpdateOnUseEngine",

    # Reliability (US-02)
    "ReliabilityProfile",
    "ReliabilityCatalog",
    "create_default_catalog",

    # Decay (US-04)
    "DecayPolicy",
    "DecayManager",
    "AutoDecayScheduler",

    # Watchers (US-05)
    "WatcherConfig",
    "WatcherEvent",
    "WatcherSystem",
    "ThresholdType",
    "Action",

    # Retrieval (US-06, US-07, US-08)
    "CandidateBelief",
    "CandidateGenerator",
    "BeliefRanker",
    "TensionPair",
    "TensionDetector",

    # Context (US-09)
    "BeliefCard",
    "BeliefCardFormatter",
    "ContextPackBuilder",
    "create_simple_context",

    # Policies (US-10)
    "BeliefClassPolicy",
    "PolicyManager",
    "create_scratch_belief",
    "AbstentionTracker",

    # I18n (US-11)
    "TranslationTemplate",
    "CardTranslator",
    "detect_language",
    "auto_translate_context_pack",

    # Observability (US-12)
    "AuditLogEntry",
    "MetricsSnapshot",
    "MetricsTracker",
    "AuditLogger",
    "BeliefObserver",

    # MAIN API (US-14) - Primary integration points
    "FeatureFlags",
    "BeliefSystem",
    "UpdateOnUseTool",
    "create_belief_system",
]
