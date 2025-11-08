"""
Watchers and threshold triggers for beliefs (US-05).

Monitors beliefs and triggers alerts/actions when confidence crosses thresholds.

Features:
- Configurable thresholds (c ≥ 0.8, c ≤ 0.2, etc.)
- Multiple watchers per belief
- Callback hooks for actions
- Event logging
"""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .belief_types import Belief, BeliefID


# ============================================================================
# Threshold Types
# ============================================================================

class ThresholdType(Enum):
    """Types of threshold crossings."""
    ABOVE = "above"  # Confidence goes above threshold
    BELOW = "below"  # Confidence goes below threshold
    CROSSES_UP = "crosses_up"  # Crosses threshold going up
    CROSSES_DOWN = "crosses_down"  # Crosses threshold going down
    CHANGES_BY = "changes_by"  # Absolute change exceeds threshold


class Action(Enum):
    """Actions to take when threshold is crossed."""
    ALERT = "alert"  # Just log an alert
    MARK_ADOPTED = "mark_adopted"  # Mark as premise to adopt
    MARK_REVIEW = "mark_review"  # Flag for human review
    MARK_ABANDONED = "mark_abandoned"  # Mark to abandon
    CALLBACK = "callback"  # Execute custom callback


# ============================================================================
# Watcher Configuration
# ============================================================================

@dataclass
class WatcherConfig:
    """
    Configuration for a belief watcher.

    Attributes:
        name: Watcher identifier
        threshold: Threshold value (meaning depends on type)
        threshold_type: Type of threshold check
        action: Action to take on trigger
        callback: Optional custom callback function(belief, event) -> None
        enabled: Whether watcher is active
        metadata: Additional context
    """
    name: str
    threshold: float
    threshold_type: ThresholdType
    action: Action
    callback: Optional[Callable[[Belief, 'WatcherEvent'], None]] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatcherEvent:
    """
    Event fired when watcher triggers.

    Attributes:
        watcher_name: Name of watcher that triggered
        belief_id: Belief that triggered
        old_confidence: Confidence before change
        new_confidence: Confidence after change
        threshold: Threshold that was crossed
        threshold_type: Type of crossing
        action: Action taken
        timestamp: When event occurred
        metadata: Additional context
    """
    watcher_name: str
    belief_id: BeliefID
    old_confidence: float
    new_confidence: float
    threshold: float
    threshold_type: ThresholdType
    action: Action
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            'watcher_name': self.watcher_name,
            'belief_id': self.belief_id,
            'old_confidence': self.old_confidence,
            'new_confidence': self.new_confidence,
            'threshold': self.threshold,
            'threshold_type': self.threshold_type.value,
            'action': self.action.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


# ============================================================================
# Watcher System
# ============================================================================

class WatcherSystem:
    """
    System for monitoring beliefs and triggering on thresholds.

    Default watchers:
    - High confidence (c ≥ 0.8): mark for adoption as premise
    - Low confidence (c ≤ 0.2): mark for human review
    - Medium confidence (0.4 ≤ c ≤ 0.6): mark as uncertain
    """

    DEFAULT_WATCHERS = [
        WatcherConfig(
            name="high_confidence",
            threshold=0.8,
            threshold_type=ThresholdType.CROSSES_UP,
            action=Action.MARK_ADOPTED,
            metadata={"description": "Mark high-confidence beliefs for adoption"}
        ),
        WatcherConfig(
            name="low_confidence",
            threshold=0.2,
            threshold_type=ThresholdType.CROSSES_DOWN,
            action=Action.MARK_REVIEW,
            metadata={"description": "Flag low-confidence beliefs for review"}
        ),
        WatcherConfig(
            name="abandoned",
            threshold=-0.6,
            threshold_type=ThresholdType.CROSSES_DOWN,
            action=Action.MARK_ABANDONED,
            metadata={"description": "Mark very negative beliefs for abandonment"}
        ),
    ]

    def __init__(self, watchers: Optional[List[WatcherConfig]] = None):
        """
        Initialize watcher system.

        Args:
            watchers: Optional list of watcher configs
        """
        self.watchers: Dict[str, WatcherConfig] = {}
        self.belief_watchers: Dict[BeliefID, List[str]] = {}  # belief -> watcher names
        self.event_log: List[WatcherEvent] = []

        # Add default watchers
        for watcher in (watchers or self.DEFAULT_WATCHERS):
            self.add_watcher(watcher)

    def add_watcher(self, watcher: WatcherConfig):
        """
        Add a watcher.

        Args:
            watcher: Watcher configuration
        """
        self.watchers[watcher.name] = watcher

    def remove_watcher(self, watcher_name: str):
        """Remove a watcher by name."""
        if watcher_name in self.watchers:
            del self.watchers[watcher_name]

    def watch_belief(self, belief_id: BeliefID, watcher_name: str):
        """
        Attach a watcher to a specific belief.

        Args:
            belief_id: Belief to watch
            watcher_name: Watcher to attach
        """
        if watcher_name not in self.watchers:
            raise ValueError(f"Watcher {watcher_name} not found")

        if belief_id not in self.belief_watchers:
            self.belief_watchers[belief_id] = []

        if watcher_name not in self.belief_watchers[belief_id]:
            self.belief_watchers[belief_id].append(watcher_name)

    def unwatch_belief(self, belief_id: BeliefID, watcher_name: str):
        """Remove watcher from belief."""
        if belief_id in self.belief_watchers:
            if watcher_name in self.belief_watchers[belief_id]:
                self.belief_watchers[belief_id].remove(watcher_name)

    def check_threshold(self,
                       belief: Belief,
                       old_confidence: float,
                       watcher: WatcherConfig) -> bool:
        """
        Check if threshold was crossed.

        Args:
            belief: Current belief state
            old_confidence: Previous confidence
            watcher: Watcher config

        Returns:
            True if threshold crossed
        """
        new_confidence = belief.confidence
        threshold = watcher.threshold

        if watcher.threshold_type == ThresholdType.ABOVE:
            return new_confidence >= threshold

        elif watcher.threshold_type == ThresholdType.BELOW:
            return new_confidence <= threshold

        elif watcher.threshold_type == ThresholdType.CROSSES_UP:
            return old_confidence < threshold <= new_confidence

        elif watcher.threshold_type == ThresholdType.CROSSES_DOWN:
            return old_confidence >= threshold > new_confidence

        elif watcher.threshold_type == ThresholdType.CHANGES_BY:
            return abs(new_confidence - old_confidence) >= threshold

        return False

    def on_belief_updated(self,
                         belief: Belief,
                         old_confidence: float) -> List[WatcherEvent]:
        """
        Called when a belief is updated.

        Checks all watchers and triggers actions.

        Args:
            belief: Updated belief
            old_confidence: Confidence before update

        Returns:
            List of events that fired
        """
        events = []

        # Get watchers for this belief (specific + global)
        watcher_names = self.belief_watchers.get(belief.id, [])

        # Also check global watchers (enabled by default)
        for name, watcher in self.watchers.items():
            if watcher.enabled and name not in watcher_names:
                watcher_names.append(name)

        # Check each watcher
        for watcher_name in watcher_names:
            if watcher_name not in self.watchers:
                continue

            watcher = self.watchers[watcher_name]

            if not watcher.enabled:
                continue

            # Check threshold
            if self.check_threshold(belief, old_confidence, watcher):
                # Create event
                event = WatcherEvent(
                    watcher_name=watcher_name,
                    belief_id=belief.id,
                    old_confidence=old_confidence,
                    new_confidence=belief.confidence,
                    threshold=watcher.threshold,
                    threshold_type=watcher.threshold_type,
                    action=watcher.action
                )

                # Execute action
                self._execute_action(belief, watcher, event)

                # Log event
                self.event_log.append(event)
                events.append(event)

        return events

    def _execute_action(self,
                       belief: Belief,
                       watcher: WatcherConfig,
                       event: WatcherEvent):
        """
        Execute action for triggered watcher.

        Args:
            belief: Belief that triggered
            watcher: Watcher configuration
            event: Event details
        """
        if watcher.action == Action.MARK_ADOPTED:
            belief.metadata['status'] = 'adopted'
            belief.metadata['adopted_at'] = datetime.now().isoformat()

        elif watcher.action == Action.MARK_REVIEW:
            belief.metadata['status'] = 'needs_review'
            belief.metadata['flagged_at'] = datetime.now().isoformat()

        elif watcher.action == Action.MARK_ABANDONED:
            belief.metadata['status'] = 'abandoned'
            belief.metadata['abandoned_at'] = datetime.now().isoformat()

        elif watcher.action == Action.CALLBACK and watcher.callback:
            watcher.callback(belief, event)

        # Always log alert
        event.metadata['action_executed'] = watcher.action.value

    def get_events_for_belief(self, belief_id: BeliefID) -> List[WatcherEvent]:
        """Get all events for a belief."""
        return [e for e in self.event_log if e.belief_id == belief_id]

    def get_recent_events(self, limit: int = 100) -> List[WatcherEvent]:
        """Get recent events."""
        return self.event_log[-limit:]

    def get_beliefs_by_status(self,
                             beliefs: List[Belief],
                             status: str) -> List[Belief]:
        """
        Get beliefs with a specific status.

        Args:
            beliefs: Pool of beliefs
            status: Status to filter ('adopted', 'needs_review', etc.)

        Returns:
            Filtered beliefs
        """
        return [
            b for b in beliefs
            if b.metadata.get('status') == status
        ]

    def export_events(self,
                     belief_id: Optional[BeliefID] = None) -> List[Dict]:
        """
        Export events as dicts.

        Args:
            belief_id: Optional filter

        Returns:
            List of event dicts
        """
        events = self.event_log
        if belief_id:
            events = [e for e in events if e.belief_id == belief_id]

        return [e.to_dict() for e in events]

    def clear_events(self):
        """Clear event log."""
        self.event_log.clear()

    def get_statistics(self) -> Dict:
        """Get watcher statistics."""
        if not self.event_log:
            return {
                'total_events': 0,
                'by_action': {},
                'by_watcher': {},
                'unique_beliefs_triggered': 0
            }

        # Count by action
        by_action = {}
        for event in self.event_log:
            action = event.action.value
            by_action[action] = by_action.get(action, 0) + 1

        # Count by watcher
        by_watcher = {}
        for event in self.event_log:
            name = event.watcher_name
            by_watcher[name] = by_watcher.get(name, 0) + 1

        # Unique beliefs
        unique_beliefs = len(set(e.belief_id for e in self.event_log))

        return {
            'total_events': len(self.event_log),
            'by_action': by_action,
            'by_watcher': by_watcher,
            'unique_beliefs_triggered': unique_beliefs
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_custom_watcher(name: str,
                         threshold: float,
                         on_cross_up: bool = True,
                         callback: Optional[Callable] = None) -> WatcherConfig:
    """
    Create a custom watcher configuration.

    Args:
        name: Watcher name
        threshold: Confidence threshold
        on_cross_up: True for crossing up, False for crossing down
        callback: Optional callback function

    Returns:
        WatcherConfig
    """
    return WatcherConfig(
        name=name,
        threshold=threshold,
        threshold_type=ThresholdType.CROSSES_UP if on_cross_up else ThresholdType.CROSSES_DOWN,
        action=Action.CALLBACK if callback else Action.ALERT,
        callback=callback
    )
