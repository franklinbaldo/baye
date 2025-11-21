"""
Observability and audit trail system (US-12).

Features:
- Comprehensive logging of all updates
- Metrics tracking (duplicate rate, confidence variance, latency)
- Audit trail export
- Performance monitoring
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
from collections import defaultdict


# ============================================================================
# Audit Log Entry
# ============================================================================

@dataclass
class AuditLogEntry:
    """
    Single audit log entry for belief update.

    Attributes:
        timestamp: When update occurred
        belief_id: Belief that was updated
        evidence_id: Evidence that triggered update (if applicable)
        evidence_hash: Hash of evidence
        weight_w: Computed weight
        components: Weight components (s, r, n, q, α)
        beta_before: (a, b) before update
        beta_after: (a, b) after update
        confidence_before: Confidence before
        confidence_after: Confidence after
        was_duplicate: Whether evidence was duplicate
        was_abstained: Whether update was abstained
        abstention_reason: Reason if abstained
        metadata: Additional context
    """
    timestamp: datetime
    belief_id: str
    evidence_id: Optional[str] = None
    evidence_hash: Optional[str] = None
    weight_w: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    beta_before: tuple[float, float] = (1.0, 1.0)
    beta_after: tuple[float, float] = (1.0, 1.0)
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    was_duplicate: bool = False
    was_abstained: bool = False
    abstention_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dict for export."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'belief_id': self.belief_id,
            'evidence_id': self.evidence_id,
            'evidence_hash': self.evidence_hash,
            'weight': self.weight_w,
            'components': self.components,
            'beta': {
                'before': {'a': self.beta_before[0], 'b': self.beta_before[1]},
                'after': {'a': self.beta_after[0], 'b': self.beta_after[1]},
            },
            'confidence': {
                'before': self.confidence_before,
                'after': self.confidence_after,
                'delta': self.confidence_after - self.confidence_before
            },
            'was_duplicate': self.was_duplicate,
            'was_abstained': self.was_abstained,
            'abstention_reason': self.abstention_reason,
            'metadata': self.metadata
        }


# ============================================================================
# Metrics Tracker
# ============================================================================

@dataclass
class MetricsSnapshot:
    """
    Snapshot of system metrics.

    Attributes:
        timestamp: When snapshot was taken
        total_updates: Total updates processed
        duplicate_rate: Fraction of duplicates
        avg_confidence_delta: Average absolute confidence change
        avg_weight: Average weight computed
        retrieval_latency_p50: Median retrieval latency (ms)
        retrieval_latency_p95: 95th percentile latency
        update_latency_p50: Median update latency (ms)
        update_latency_p95: 95th percentile latency
    """
    timestamp: datetime = field(default_factory=datetime.now)
    total_updates: int = 0
    duplicate_rate: float = 0.0
    avg_confidence_delta: float = 0.0
    avg_weight: float = 0.0
    retrieval_latency_p50: float = 0.0
    retrieval_latency_p95: float = 0.0
    update_latency_p50: float = 0.0
    update_latency_p95: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_updates': self.total_updates,
            'duplicate_rate': self.duplicate_rate,
            'avg_confidence_delta': self.avg_confidence_delta,
            'avg_weight': self.avg_weight,
            'retrieval_latency': {
                'p50': self.retrieval_latency_p50,
                'p95': self.retrieval_latency_p95
            },
            'update_latency': {
                'p50': self.update_latency_p50,
                'p95': self.update_latency_p95
            }
        }


class MetricsTracker:
    """
    Tracks system metrics.

    Metrics:
    - Duplicate rate
    - Average confidence variance
    - Latency percentiles (P50, P95)
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracker.

        Args:
            window_size: Number of recent operations to track
        """
        self.window_size = window_size

        # Metrics buffers (ring buffers)
        self.confidence_deltas: List[float] = []
        self.weights: List[float] = []
        self.retrieval_latencies: List[float] = []
        self.update_latencies: List[float] = []
        self.duplicate_flags: List[bool] = []

        # Counters
        self.total_updates = 0
        self.total_duplicates = 0

    def record_update(self,
                     confidence_delta: float,
                     weight: float,
                     was_duplicate: bool):
        """
        Record an update.

        Args:
            confidence_delta: Absolute change in confidence
            weight: Weight computed
            was_duplicate: Whether evidence was duplicate
        """
        self.total_updates += 1

        # Add to buffers
        self._add_to_buffer(self.confidence_deltas, abs(confidence_delta))
        self._add_to_buffer(self.weights, abs(weight))
        self._add_to_buffer(self.duplicate_flags, was_duplicate)

        if was_duplicate:
            self.total_duplicates += 1

    def record_retrieval_latency(self, latency_ms: float):
        """Record retrieval latency."""
        self._add_to_buffer(self.retrieval_latencies, latency_ms)

    def record_update_latency(self, latency_ms: float):
        """Record update latency."""
        self._add_to_buffer(self.update_latencies, latency_ms)

    def _add_to_buffer(self, buffer: List, value):
        """Add to ring buffer."""
        buffer.append(value)
        if len(buffer) > self.window_size:
            buffer.pop(0)

    def get_metrics(self) -> MetricsSnapshot:
        """
        Get current metrics snapshot.

        Returns:
            MetricsSnapshot
        """
        return MetricsSnapshot(
            total_updates=self.total_updates,
            duplicate_rate=self._duplicate_rate(),
            avg_confidence_delta=self._avg(self.confidence_deltas),
            avg_weight=self._avg(self.weights),
            retrieval_latency_p50=self._percentile(self.retrieval_latencies, 50),
            retrieval_latency_p95=self._percentile(self.retrieval_latencies, 95),
            update_latency_p50=self._percentile(self.update_latencies, 50),
            update_latency_p95=self._percentile(self.update_latencies, 95)
        )

    def _duplicate_rate(self) -> float:
        """Calculate duplicate rate."""
        if not self.duplicate_flags:
            return 0.0
        return sum(self.duplicate_flags) / len(self.duplicate_flags)

    def _avg(self, values: List[float]) -> float:
        """Calculate average."""
        return sum(values) / len(values) if values else 0.0

    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]


# ============================================================================
# Audit Logger
# ============================================================================

class AuditLogger:
    """
    Logs all belief updates for audit trail (US-12).

    Provides:
    - Complete update history
    - Export to JSON/CSV
    - Filtering by belief, time range, etc.
    """

    def __init__(self, max_entries: int = 10000):
        """
        Initialize audit logger.

        Args:
            max_entries: Maximum log entries to keep in memory
        """
        self.max_entries = max_entries
        self.entries: List[AuditLogEntry] = []
        self.entries_by_belief: Dict[str, List[int]] = defaultdict(list)

    def log_update(self, entry: AuditLogEntry):
        """
        Log an update.

        Args:
            entry: Audit log entry
        """
        # Add to main log
        self.entries.append(entry)

        # Index by belief
        entry_idx = len(self.entries) - 1
        self.entries_by_belief[entry.belief_id].append(entry_idx)

        # Trim if necessary
        if len(self.entries) > self.max_entries:
            self._trim_old_entries()

    def _trim_old_entries(self):
        """Trim old entries to stay under max."""
        # Remove oldest 10%
        trim_count = self.max_entries // 10
        self.entries = self.entries[trim_count:]

        # Rebuild index
        self.entries_by_belief.clear()
        for idx, entry in enumerate(self.entries):
            self.entries_by_belief[entry.belief_id].append(idx)

    def get_entries_for_belief(self, belief_id: str) -> List[AuditLogEntry]:
        """Get all entries for a belief."""
        indices = self.entries_by_belief.get(belief_id, [])
        return [self.entries[idx] for idx in indices if idx < len(self.entries)]

    def get_entries_in_range(self,
                            start: datetime,
                            end: datetime) -> List[AuditLogEntry]:
        """Get entries in time range."""
        return [
            entry for entry in self.entries
            if start <= entry.timestamp <= end
        ]

    def get_recent_entries(self, limit: int = 100) -> List[AuditLogEntry]:
        """Get recent entries."""
        return self.entries[-limit:]

    def export_to_json(self,
                      filepath: str,
                      belief_id: Optional[str] = None):
        """
        Export log to JSON.

        Args:
            filepath: Output file path
            belief_id: Optional filter by belief
        """
        entries = self.entries
        if belief_id:
            entries = self.get_entries_for_belief(belief_id)

        data = [entry.to_dict() for entry in entries]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def export_to_csv(self,
                     filepath: str,
                     belief_id: Optional[str] = None):
        """
        Export log to CSV.

        Args:
            filepath: Output file path
            belief_id: Optional filter by belief
        """
        import csv

        entries = self.entries
        if belief_id:
            entries = self.get_entries_for_belief(belief_id)

        if not entries:
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'timestamp', 'belief_id', 'evidence_id', 'weight',
                's', 'r', 'n', 'q', 'alpha',
                'a_before', 'b_before', 'a_after', 'b_after',
                'conf_before', 'conf_after', 'conf_delta',
                'was_duplicate', 'was_abstained'
            ])

            # Rows
            for entry in entries:
                writer.writerow([
                    entry.timestamp.isoformat(),
                    entry.belief_id,
                    entry.evidence_id or '',
                    entry.weight_w,
                    entry.components.get('s', 0),
                    entry.components.get('r', 0),
                    entry.components.get('n', 0),
                    entry.components.get('q', 0),
                    entry.components.get('alpha', 0),
                    entry.beta_before[0],
                    entry.beta_before[1],
                    entry.beta_after[0],
                    entry.beta_after[1],
                    entry.confidence_before,
                    entry.confidence_after,
                    entry.confidence_after - entry.confidence_before,
                    entry.was_duplicate,
                    entry.was_abstained
                ])

    def get_statistics(self) -> Dict:
        """Get log statistics."""
        if not self.entries:
            return {
                'total_entries': 0,
                'unique_beliefs': 0,
                'date_range': None
            }

        return {
            'total_entries': len(self.entries),
            'unique_beliefs': len(self.entries_by_belief),
            'date_range': {
                'start': self.entries[0].timestamp.isoformat(),
                'end': self.entries[-1].timestamp.isoformat()
            },
            'duplicates': sum(1 for e in self.entries if e.was_duplicate),
            'abstentions': sum(1 for e in self.entries if e.was_abstained)
        }


# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """
    Monitors performance and latency (US-13).

    Context manager for timing operations.

    Usage:
        with monitor.time_operation('retrieval') as timer:
            results = retrieve_beliefs(...)
        # Latency is automatically recorded
    """

    def __init__(self, metrics_tracker: MetricsTracker):
        """
        Initialize monitor.

        Args:
            metrics_tracker: Metrics tracker
        """
        self.metrics = metrics_tracker

    def time_operation(self, operation_type: str):
        """
        Time an operation.

        Args:
            operation_type: 'retrieval' or 'update'

        Returns:
            Context manager
        """
        return OperationTimer(self.metrics, operation_type)


class OperationTimer:
    """Context manager for timing operations."""

    def __init__(self, metrics: MetricsTracker, operation_type: str):
        self.metrics = metrics
        self.operation_type = operation_type
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000.0

        if self.operation_type == 'retrieval':
            self.metrics.record_retrieval_latency(elapsed_ms)
        elif self.operation_type == 'update':
            self.metrics.record_update_latency(elapsed_ms)


# ============================================================================
# Integrated Observer
# ============================================================================

class BeliefObserver:
    """
    Integrated observability system.

    Combines:
    - Audit logging
    - Metrics tracking
    - Performance monitoring
    """

    def __init__(self):
        """Initialize observer."""
        self.audit_logger = AuditLogger()
        self.metrics_tracker = MetricsTracker()
        self.performance_monitor = PerformanceMonitor(self.metrics_tracker)

    def get_dashboard_data(self) -> Dict:
        """
        Get data for observability dashboard.

        Returns:
            Dict with all metrics and stats
        """
        return {
            'metrics': self.metrics_tracker.get_metrics().to_dict(),
            'audit_stats': self.audit_logger.get_statistics(),
            'timestamp': datetime.now().isoformat()
        }

    def export_full_audit(self, filepath: str, format: str = 'json'):
        """
        Export full audit trail.

        Args:
            filepath: Output path
            format: 'json' or 'csv'
        """
        if format == 'json':
            self.audit_logger.export_to_json(filepath)
        elif format == 'csv':
            self.audit_logger.export_to_csv(filepath)
