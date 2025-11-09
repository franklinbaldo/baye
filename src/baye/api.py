"""
Stable API for agent integration (US-14).

Provides high-level interfaces for:
- retrieve_context_for_prompt(): Get relevant beliefs for chat
- UpdateOnUseTool: Wrapper for tool calls with automatic update
- Feature flags for backends (embeddings vs Jaccard, etc.)

This is the main integration point for agentic systems.
"""

from typing import List, Optional, Dict, Callable, Tuple, Any
from dataclasses import dataclass
import time

from .belief_types import Belief, BeliefID
from .justification_graph import JustificationGraph
from .evidence import EvidenceStore, UpdateOnUseEngine, Evidence
from .reliability_catalog import ReliabilityCatalog
from .temporal_decay import DecayManager
from .watchers import WatcherSystem
from .retrieval import CandidateGenerator, BeliefRanker, TensionDetector
from .context_builder import ContextPackBuilder
from .policies import PolicyManager
from .i18n import detect_language, CardTranslator
from .observability import BeliefObserver, AuditLogEntry


# ============================================================================
# Feature Flags
# ============================================================================

@dataclass
class FeatureFlags:
    """
    Feature flags for runtime configuration.

    Attributes:
        use_embeddings: Use embeddings vs Jaccard for similarity
        enable_decay: Enable temporal decay
        enable_tensions: Include tension pairs in context
        enable_i18n: Auto-translate cards to prompt language
        enable_mmr: Use MMR for diversity vs pure relevance ranking
        enable_watchers: Enable threshold watchers
    """
    use_embeddings: bool = False  # Default to Jaccard (no deps)
    enable_decay: bool = True
    enable_tensions: bool = True
    enable_i18n: bool = True
    enable_mmr: bool = True
    enable_watchers: bool = True


# ============================================================================
# Integrated Belief System
# ============================================================================

class BeliefSystem:
    """
    Integrated belief tracking system.

    This is the main entry point for all belief operations.

    Features:
    - Update-on-Use from tool calls
    - Context retrieval for chat
    - Automatic decay, watchers, observability
    - Feature flags for customization
    """

    def __init__(self,
                 graph: Optional[JustificationGraph] = None,
                 feature_flags: Optional[FeatureFlags] = None):
        """
        Initialize belief system.

        Args:
            graph: Optional existing graph (creates new if None)
            feature_flags: Optional feature configuration
        """
        # Core components
        self.graph = graph or JustificationGraph()
        self.evidence_store = EvidenceStore()
        self.reliability_catalog = ReliabilityCatalog()
        self.decay_manager = DecayManager()
        self.watcher_system = WatcherSystem()
        self.policy_manager = PolicyManager()
        self.observer = BeliefObserver()

        # Feature flags
        self.flags = feature_flags or FeatureFlags()

        # Update engine
        self.uou_engine = UpdateOnUseEngine(
            evidence_store=self.evidence_store,
            alpha_by_class=self._get_alpha_by_class()
        )

        # Retrieval components
        self.candidate_generator = CandidateGenerator(
            use_embeddings=self.flags.use_embeddings
        )
        self.ranker = BeliefRanker()
        self.tension_detector = TensionDetector()

        # Context builder
        self.context_builder = ContextPackBuilder(
            include_tensions=self.flags.enable_tensions
        )

        # Translation
        self.translator = CardTranslator()

    def _get_alpha_by_class(self) -> Dict[str, float]:
        """Get learning rates from policy manager."""
        return {
            cls: policy.alpha
            for cls, policy in self.policy_manager.policies.items()
        }

    # ========================================================================
    # Main API: retrieve_context_for_prompt (US-14)
    # ========================================================================

    def retrieve_context_for_prompt(self,
                                    prompt: str,
                                    k: int = 8,
                                    token_budget: int = 1000,
                                    context_beliefs: Optional[List[BeliefID]] = None,
                                    language: Optional[str] = None) -> str:
        """
        Retrieve relevant beliefs for a prompt (US-06, US-07, US-08, US-09).

        This is the main entry point for chat context retrieval.

        Args:
            prompt: User prompt/query
            k: Number of beliefs to retrieve
            token_budget: Maximum tokens for context
            context_beliefs: Recent conversation context (belief IDs)
            language: Target language (auto-detected if None)

        Returns:
            Formatted context pack string

        Example:
            >>> system = BeliefSystem()
            >>> context = system.retrieve_context_for_prompt(
            ...     prompt="Como lidar com timeouts de API?",
            ...     k=5,
            ...     token_budget=500
            ... )
            >>> print(context)
            # Crenças Relevantes
            [CRENÇA] APIs podem ter timeouts inesperados
            Confiança: 75% | Atualizado: 2025-01-08
            ...
        """
        # Time the operation
        with self.observer.performance_monitor.time_operation('retrieval'):
            # Auto-detect language
            if language is None and self.flags.enable_i18n:
                language = detect_language(prompt)

            # Apply decay if enabled
            if self.flags.enable_decay:
                self.decay_manager.batch_decay(list(self.graph.beliefs.values()))

            # Generate candidates
            candidates = self.candidate_generator.generate_candidates(
                query=prompt,
                beliefs=list(self.graph.beliefs.values()),
                belief_graph=self.graph.beliefs,
                context_beliefs=context_beliefs,
                k_per_channel=20
            )

            # Rank candidates
            if self.flags.enable_mmr:
                ranked = self.ranker.rank_with_mmr(
                    candidates=candidates,
                    query=prompt,
                    belief_graph=self.graph.beliefs,
                    evidence_store=self.evidence_store,
                    k=k
                )
            else:
                ranked = self.ranker.rank_beliefs(
                    candidates=candidates,
                    query=prompt,
                    belief_graph=self.graph.beliefs,
                    evidence_store=self.evidence_store,
                    k=k
                )

            # Detect tensions if enabled
            tensions = None
            if self.flags.enable_tensions:
                tensions = self.tension_detector.detect_tensions(
                    ranked_beliefs=ranked,
                    belief_graph=self.graph.beliefs
                )

            # Build context pack
            context_pack = self.context_builder.build_pack(
                ranked_beliefs=ranked,
                tensions=tensions,
                token_budget=token_budget,
                evidence_store=self.evidence_store,
                language=language or 'en'
            )

            # Translate if needed
            if self.flags.enable_i18n and language and language != 'en':
                context_pack = self.translator.translate_context_pack(
                    context_pack, language
                )

            return context_pack

    # ========================================================================
    # Tool Integration: update_from_tool_call (US-01)
    # ========================================================================

    def update_from_tool_call(self,
                             belief_id: BeliefID,
                             tool_result: str,
                             tool_name: str,
                             sentiment: float = 1.0,
                             quality: float = 1.0,
                             metadata: Optional[Dict] = None) -> Tuple[Evidence, Dict]:
        """
        Update belief from tool call (US-01 main entry point).

        This is called after every tool execution to update beliefs.

        Args:
            belief_id: Belief to update
            tool_result: Text result from tool
            tool_name: Tool identifier (e.g., "web_search", "api_call")
            sentiment: +1 (supports), -1 (contradicts), 0 (neutral)
            quality: Quality score [0, 1]
            metadata: Additional context

        Returns:
            (evidence, update_info) tuple with evidence and update details

        Example:
            >>> evidence, update = system.update_from_tool_call(
            ...     belief_id="belief_123",
            ...     tool_result="API returned 500 error",
            ...     tool_name="api_monitor",
            ...     sentiment=-1.0  # Contradicts reliability assumption
            ... )
            >>> print(f"Confidence changed: {update['confidence_delta']}")
        """
        # Time the operation
        with self.observer.performance_monitor.time_operation('update'):
            belief = self.graph.beliefs[belief_id]

            # Get reliability for tool
            reliability = self.reliability_catalog.get_reliability(f"tool:{tool_name}")

            # Record old state
            old_confidence = belief.confidence
            old_a, old_b = belief.a, belief.b

            # Process tool call
            evidence, update_record = self.uou_engine.process_tool_call(
                belief_id=belief_id,
                belief=belief,
                tool_result=tool_result,
                tool_name=tool_name,
                reliability=reliability,
                sentiment=sentiment,
                quality=quality,
                metadata=metadata or {}
            )

            # Record in observer
            audit_entry = AuditLogEntry(
                timestamp=update_record.timestamp,
                belief_id=belief_id,
                evidence_id=evidence.id,
                evidence_hash=evidence.hash,
                weight_w=update_record.weight_w,
                components={
                    's': update_record.sentiment_s,
                    'r': update_record.reliability_r,
                    'n': update_record.novelty_n,
                    'q': update_record.quality_q,
                    'alpha': update_record.alpha
                },
                beta_before=(update_record.a_before, update_record.b_before),
                beta_after=(update_record.a_after, update_record.b_after),
                confidence_before=old_confidence,
                confidence_after=belief.confidence,
                was_duplicate=update_record.was_duplicate
            )
            self.observer.audit_logger.log_update(audit_entry)

            # Track metrics
            self.observer.metrics_tracker.record_update(
                confidence_delta=abs(belief.confidence - old_confidence),
                weight=update_record.weight_w,
                was_duplicate=update_record.was_duplicate
            )

            # Check watchers if enabled
            if self.flags.enable_watchers:
                watcher_events = self.watcher_system.on_belief_updated(
                    belief=belief,
                    old_confidence=old_confidence
                )

            # Return evidence and update info
            update_info = {
                'confidence_before': old_confidence,
                'confidence_after': belief.confidence,
                'confidence_delta': belief.confidence - old_confidence,
                'weight': update_record.weight_w,
                'was_duplicate': update_record.was_duplicate,
                'components': {
                    's': update_record.sentiment_s,
                    'r': update_record.reliability_r,
                    'n': update_record.novelty_n,
                    'q': update_record.quality_q,
                    'alpha': update_record.alpha
                }
            }

            return evidence, update_info

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    def add_belief(self,
                  content: str,
                  confidence: Optional[float] = None,
                  context: str = "general",
                  belief_class: str = "normal") -> Belief:
        """
        Add a new belief to the system.

        Args:
            content: Belief content
            confidence: Initial confidence (None = estimate from graph)
            context: Domain context
            belief_class: Belief class (normal, scratch, foundational, etc.)

        Returns:
            Created Belief
        """
        if confidence is None:
            # Use estimation if graph has beliefs
            if self.graph.beliefs:
                return self.graph.add_belief_with_estimation(
                    content=content,
                    context=context,
                    source_task="api"
                )
            else:
                confidence = 0.0  # Neutral for empty graph

        return self.graph.add_belief(
            content=content,
            confidence=confidence,
            context=context,
            source_task="api"
        )

    def get_dashboard_data(self) -> Dict:
        """Get observability dashboard data."""
        return self.observer.get_dashboard_data()

    def export_audit_trail(self, filepath: str, format: str = 'json'):
        """Export audit trail."""
        self.observer.export_full_audit(filepath, format)


# ============================================================================
# Update-on-Use Tool Wrapper (US-14)
# ============================================================================

class UpdateOnUseTool:
    """
    Decorator/wrapper for tools to automatically update beliefs.

    Usage:
        >>> system = BeliefSystem()
        >>> @UpdateOnUseTool(system, belief_id="api_reliability")
        ... def call_api(endpoint):
        ...     response = requests.get(endpoint)
        ...     return response.text
        >>>
        >>> result = call_api("https://api.example.com/data")
        # Belief "api_reliability" is automatically updated based on success/failure
    """

    def __init__(self,
                 system: BeliefSystem,
                 belief_id: BeliefID,
                 tool_name: Optional[str] = None,
                 sentiment_fn: Optional[Callable[[Any], float]] = None,
                 quality_fn: Optional[Callable[[Any], float]] = None):
        """
        Initialize tool wrapper.

        Args:
            system: BeliefSystem instance
            belief_id: Belief to update on tool use
            tool_name: Tool name (auto-detected if None)
            sentiment_fn: Function to compute sentiment from result
            quality_fn: Function to compute quality from result
        """
        self.system = system
        self.belief_id = belief_id
        self.tool_name = tool_name
        self.sentiment_fn = sentiment_fn or (lambda x: 1.0)  # Default: supports
        self.quality_fn = quality_fn or (lambda x: 1.0)  # Default: high quality

    def __call__(self, func):
        """Wrap function."""
        def wrapper(*args, **kwargs):
            # Execute tool
            try:
                result = func(*args, **kwargs)

                # Compute sentiment and quality
                sentiment = self.sentiment_fn(result)
                quality = self.quality_fn(result)

                # Update belief
                tool_name = self.tool_name or func.__name__
                evidence, update = self.system.update_from_tool_call(
                    belief_id=self.belief_id,
                    tool_result=str(result)[:500],  # Truncate long results
                    tool_name=tool_name,
                    sentiment=sentiment,
                    quality=quality,
                    metadata={'args': str(args), 'kwargs': str(kwargs)}
                )

                return result

            except Exception as e:
                # Tool failed: negative evidence
                self.system.update_from_tool_call(
                    belief_id=self.belief_id,
                    tool_result=f"Tool failed: {str(e)}",
                    tool_name=self.tool_name or func.__name__,
                    sentiment=-1.0,  # Contradicts reliability
                    quality=0.5,
                    metadata={'error': str(e)}
                )
                raise

        return wrapper


# ============================================================================
# Convenience Factory
# ============================================================================

def create_belief_system(use_embeddings: bool = False,
                        enable_all_features: bool = True) -> BeliefSystem:
    """
    Create a BeliefSystem with sensible defaults.

    Args:
        use_embeddings: Use embeddings vs Jaccard
        enable_all_features: Enable all features vs minimal

    Returns:
        Configured BeliefSystem
    """
    if enable_all_features:
        flags = FeatureFlags(
            use_embeddings=use_embeddings,
            enable_decay=True,
            enable_tensions=True,
            enable_i18n=True,
            enable_mmr=True,
            enable_watchers=True
        )
    else:
        flags = FeatureFlags(
            use_embeddings=use_embeddings,
            enable_decay=False,
            enable_tensions=False,
            enable_i18n=False,
            enable_mmr=False,
            enable_watchers=False
        )

    return BeliefSystem(feature_flags=flags)
