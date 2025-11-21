"""
Continuum Memory System: Online + Offline Consolidation

Implements the NL paper's insight on two-phase memory consolidation:
1. Online (immediate) - like hippocampal encoding
2. Offline (background) - like cortical consolidation during sleep

Key Insight (from NL Section 1.1):
"Human brain involves at least two distinct consolidation processes:
(1) Online consolidation - immediate, during wakefulness
(2) Offline consolidation - replay during sleep, strengthens and reorganizes"
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np


@dataclass
class ConsolidationEvent:
    """Record of a consolidation operation."""
    timestamp: datetime
    updates_processed: int
    beliefs_strengthened: int
    beliefs_pruned: int
    relationships_discovered: int
    duration_ms: float


class ConsolidationScheduler:
    """
    Schedules offline consolidation events.

    Like sleep cycles: periodic background processing.
    """

    def __init__(
        self,
        interval_seconds: float = 60.0,
        min_queue_size: int = 5
    ):
        """
        Args:
            interval_seconds: Time between consolidations
            min_queue_size: Minimum updates before consolidating
        """
        self.interval = timedelta(seconds=interval_seconds)
        self.min_queue_size = min_queue_size
        self.last_consolidation = datetime.now()
        self.consolidation_history: List[ConsolidationEvent] = []

    def should_consolidate(self, queue_size: int) -> bool:
        """
        Determine if it's time to consolidate.

        Triggers if:
        1. Enough time has passed AND
        2. Enough updates have accumulated
        """
        time_elapsed = datetime.now() - self.last_consolidation
        return (
            time_elapsed >= self.interval and
            queue_size >= self.min_queue_size
        )

    def record_consolidation(self, event: ConsolidationEvent):
        """Record a consolidation event."""
        self.last_consolidation = datetime.now()
        self.consolidation_history.append(event)

    def get_statistics(self) -> Dict:
        """Get consolidation statistics."""
        if not self.consolidation_history:
            return {
                'total_consolidations': 0,
                'avg_updates_per_consolidation': 0,
                'avg_duration_ms': 0,
            }

        return {
            'total_consolidations': len(self.consolidation_history),
            'avg_updates_per_consolidation': np.mean([
                e.updates_processed for e in self.consolidation_history
            ]),
            'avg_beliefs_strengthened': np.mean([
                e.beliefs_strengthened for e in self.consolidation_history
            ]),
            'avg_beliefs_pruned': np.mean([
                e.beliefs_pruned for e in self.consolidation_history
            ]),
            'avg_duration_ms': np.mean([
                e.duration_ms for e in self.consolidation_history
            ]),
            'last_consolidation': self.last_consolidation.isoformat(),
        }


class ContinuumMemoryGraph:
    """
    Extended JustificationGraph with dual-phase memory consolidation.

    Online Phase: Immediate updates with limited compute budget
    Offline Phase: Background consolidation with full budget
    """

    def __init__(self, base_graph, consolidation_interval: float = 60.0):
        """
        Args:
            base_graph: Base JustificationGraph instance
            consolidation_interval: Seconds between consolidations
        """
        from ..justification_graph import JustificationGraph

        if not isinstance(base_graph, JustificationGraph):
            raise TypeError("base_graph must be a JustificationGraph")

        self.graph = base_graph

        # Online buffer: recent updates awaiting consolidation
        self.consolidation_queue: List[Dict] = []

        # Scheduler
        self.scheduler = ConsolidationScheduler(
            interval_seconds=consolidation_interval
        )

        # Background task
        self.consolidation_task: Optional[asyncio.Task] = None
        self.is_consolidating = False

    async def update_belief_continuum(
        self,
        belief_id: str,
        delta: float,
        context: str = ""
    ):
        """
        Two-phase update: online (immediate) + offline (scheduled).

        Args:
            belief_id: Belief to update
            delta: Confidence change
            context: Domain context

        Returns:
            Immediate propagation result
        """
        # PHASE 1: ONLINE (immediate, limited budget)
        online_result = self.graph.propagate_from(
            origin_id=belief_id,
            initial_delta=delta,
            max_depth=3,  # Limited depth for speed
        )

        # Queue for offline consolidation
        self.consolidation_queue.append({
            'belief_id': belief_id,
            'delta': delta,
            'context': context,
            'timestamp': datetime.now(),
            'online_result': online_result,
        })

        # Check if consolidation needed
        if self.scheduler.should_consolidate(len(self.consolidation_queue)):
            # Trigger async consolidation (non-blocking)
            if not self.is_consolidating:
                asyncio.create_task(self.consolidate_offline())

        return online_result

    async def consolidate_offline(self):
        """
        PHASE 2: OFFLINE consolidation (background).

        Steps:
        1. Replay important updates with full compute budget
        2. Strengthen high-importance beliefs
        3. Discover new relationships
        4. Prune weak beliefs
        5. Merge redundant beliefs
        """
        if self.is_consolidating:
            return  # Already consolidating

        self.is_consolidating = True
        start_time = datetime.now()

        try:
            # Metrics
            updates_processed = len(self.consolidation_queue)
            beliefs_strengthened = 0
            beliefs_pruned = 0
            relationships_discovered = 0

            # 1. REPLAY: Re-process important updates with full budget
            important_updates = self._prioritize_for_replay(
                self.consolidation_queue
            )

            for update in important_updates:
                await self.graph.propagate_from(
                    origin_id=update['belief_id'],
                    initial_delta=update['delta'],
                    max_depth=10,  # Much deeper than online
                )

            # 2. STRENGTHEN: Reinforce high-importance beliefs
            high_importance = self._get_high_importance_beliefs()
            for belief_id in high_importance:
                belief = self.graph.beliefs.get(belief_id)
                if belief and hasattr(belief, 'a') and hasattr(belief, 'b'):
                    # Increase pseudo-counts (consolidate memory)
                    belief.a *= 1.05
                    belief.b *= 1.05
                    beliefs_strengthened += 1

            # 3. DISCOVER: Find missing relationships (optional, expensive)
            # This would use LLM to scan for connections
            # relationships_discovered = await self._discover_relationships(important_updates)

            # 4. PRUNE: Remove very weak beliefs
            beliefs_pruned = self._prune_weak_beliefs(threshold=0.05)

            # 5. MERGE: Consolidate redundant beliefs (future work)
            # await self._merge_redundant_beliefs()

            # Clear queue
            self.consolidation_queue = []

            # Record event
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            event = ConsolidationEvent(
                timestamp=datetime.now(),
                updates_processed=updates_processed,
                beliefs_strengthened=beliefs_strengthened,
                beliefs_pruned=beliefs_pruned,
                relationships_discovered=relationships_discovered,
                duration_ms=duration_ms
            )
            self.scheduler.record_consolidation(event)

        finally:
            self.is_consolidating = False

    def _prioritize_for_replay(self, updates: List[Dict]) -> List[Dict]:
        """
        Prioritize which updates to replay.

        Like hippocampal sharp-wave ripples: replay important events.

        Importance Score:
        - Cascade size (how many beliefs affected)
        - Surprise magnitude (how surprising was the update)
        - Usage frequency (how often belief is accessed)
        - Recency (recent events more important)
        """
        scored = []

        for update in updates:
            online_result = update.get('online_result')
            belief_id = update['belief_id']

            # Get belief
            belief = self.graph.beliefs.get(belief_id)

            # Cascade size
            cascade_size = (
                online_result.total_beliefs_updated
                if online_result else 1
            )

            # Surprise magnitude
            surprise = abs(update['delta'])

            # Usage frequency (if belief has usage tracking)
            usage = 0
            if belief and hasattr(belief, 'usage_count'):
                usage = belief.usage_count

            # Recency (seconds ago)
            age_seconds = (datetime.now() - update['timestamp']).total_seconds()
            recency = np.exp(-age_seconds / 3600)  # Decay over 1 hour

            # Combined importance
            importance = (
                0.4 * cascade_size / 10.0 +  # Normalize
                0.3 * surprise +
                0.2 * usage / 10.0 +
                0.1 * recency
            )

            scored.append((importance, update))

        # Sort by importance
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return top 20% (or minimum 3)
        n_replay = max(3, int(0.2 * len(scored)))
        return [update for _, update in scored[:n_replay]]

    def _get_high_importance_beliefs(self) -> List[str]:
        """
        Identify high-importance beliefs for strengthening.

        Criteria:
        - High centrality in graph (many dependents)
        - Frequently accessed
        - High confidence
        """
        important = []

        for belief_id, belief in self.graph.beliefs.items():
            # Centrality: many dependents
            centrality = len(getattr(belief, 'dependents', []))

            # Usage
            usage = getattr(belief, 'usage_count', 0)

            # Confidence
            confidence = getattr(belief, 'confidence', 0)

            # Importance score
            importance = (
                0.5 * centrality / 10.0 +
                0.3 * usage / 10.0 +
                0.2 * confidence
            )

            if importance > 0.5:  # Threshold
                important.append(belief_id)

        return important[:10]  # Top 10

    def _prune_weak_beliefs(self, threshold: float = 0.05) -> int:
        """
        Remove beliefs with very low confidence.

        Returns:
            Number of beliefs pruned
        """
        to_prune = []

        for belief_id, belief in self.graph.beliefs.items():
            confidence = getattr(belief, 'confidence', 0.5)

            # Prune if:
            # 1. Very low confidence
            # 2. No dependents (not supporting anything)
            # 3. Old enough (not recently created)
            has_dependents = len(getattr(belief, 'dependents', [])) > 0

            if confidence < threshold and not has_dependents:
                # Check age
                created = getattr(belief, 'created_at', datetime.now())
                age_hours = (datetime.now() - created).total_seconds() / 3600

                if age_hours > 1:  # At least 1 hour old
                    to_prune.append(belief_id)

        # Remove from graph
        for belief_id in to_prune:
            if belief_id in self.graph.beliefs:
                del self.graph.beliefs[belief_id]

                # Remove from NetworkX graph if present
                if hasattr(self.graph, 'nx_graph'):
                    if self.graph.nx_graph.has_node(belief_id):
                        self.graph.nx_graph.remove_node(belief_id)

        return len(to_prune)

    def get_consolidation_status(self) -> Dict:
        """
        Get current consolidation status.

        Useful for UI display.
        """
        return {
            'is_consolidating': self.is_consolidating,
            'queue_size': len(self.consolidation_queue),
            'next_consolidation_in_seconds': max(
                0,
                (self.scheduler.interval - (
                    datetime.now() - self.scheduler.last_consolidation
                )).total_seconds()
            ),
            'statistics': self.scheduler.get_statistics(),
        }
