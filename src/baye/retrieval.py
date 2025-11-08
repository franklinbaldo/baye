"""
Belief retrieval and ranking system (US-06, US-07, US-08).

Features:
- Multi-channel candidate generation (text, structure, recency)
- Unified ranking with MMR for diversity
- Tension/contradiction detection
- Configurable weights and parameters

Ranking formula:
score = w_sim·sim + w_conf·g(c) + w_rec·f(Δt) + w_rel·r̄ - w_x·contradiction_pressure
"""

from typing import List, Tuple, Set, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import math

from .belief_types import Belief, BeliefID


# ============================================================================
# Candidate Generation
# ============================================================================

@dataclass
class CandidateBelie

:
    """
    A belief candidate for retrieval.

    Attributes:
        belief: The belief object
        score: Relevance score
        similarity: Text similarity to query
        recency_score: Recency component
        confidence_score: Confidence component
        reliability_score: Average reliability
        contradiction_pressure: Penalty from contradictions
        metadata: Additional scoring details
    """
    belief: Belief
    score: float = 0.0
    similarity: float = 0.0
    recency_score: float = 0.0
    confidence_score: float = 0.0
    reliability_score: float = 0.0
    contradiction_pressure: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CandidateGenerator:
    """
    Generates candidate beliefs through multiple channels (US-06).

    Channels:
    1. Text similarity (BM25/TF-IDF + semantic)
    2. Graph structure (neighbors of recent beliefs)
    3. Recency (recently updated)
    """

    def __init__(self,
                 similarity_fn: Optional[Callable[[str, str], float]] = None,
                 use_embeddings: bool = False):
        """
        Initialize candidate generator.

        Args:
            similarity_fn: Custom similarity function(text1, text2) -> [0, 1]
            use_embeddings: Use embeddings if available
        """
        self.similarity_fn = similarity_fn or self._jaccard_similarity
        self.use_embeddings = use_embeddings

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity fallback."""
        from .belief_estimation import SemanticEstimator
        estimator = SemanticEstimator()
        return estimator._calculate_similarity(text1, text2)

    def generate_by_text(self,
                        query: str,
                        beliefs: List[Belief],
                        k: int = 20) -> List[Tuple[Belief, float]]:
        """
        Generate candidates by text similarity.

        Args:
            query: User query text
            beliefs: Pool of beliefs
            k: Number of candidates

        Returns:
            List of (belief, similarity) tuples
        """
        candidates = []

        for belief in beliefs:
            sim = self.similarity_fn(query, belief.content)
            candidates.append((belief, sim))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:k]

    def generate_by_structure(self,
                             seed_beliefs: List[BeliefID],
                             belief_graph: Dict[BeliefID, Belief],
                             max_hops: int = 2,
                             k: int = 20) -> List[Tuple[Belief, float]]:
        """
        Generate candidates by graph structure (US-06).

        Expands 1-2 hops from seed beliefs (e.g., recent conversation context).

        Args:
            seed_beliefs: Starting belief IDs
            belief_graph: Dict of all beliefs
            max_hops: Maximum hops to expand
            k: Number of candidates

        Returns:
            List of (belief, relevance) tuples
        """
        visited: Set[BeliefID] = set()
        candidates: Dict[BeliefID, float] = {}

        def expand(belief_id: BeliefID, hop: int, score: float):
            if hop > max_hops or belief_id in visited:
                return

            visited.add(belief_id)

            if belief_id in belief_graph:
                belief = belief_graph[belief_id]

                # Add to candidates with distance-discounted score
                discount = 0.7 ** hop  # Decay by hop distance
                candidates[belief_id] = max(
                    candidates.get(belief_id, 0.0),
                    score * discount
                )

                # Expand to neighbors
                for neighbor_id in belief.supported_by + belief.supports:
                    expand(neighbor_id, hop + 1, score * 0.8)

        # Start expansion from seeds
        for seed_id in seed_beliefs:
            expand(seed_id, hop=0, score=1.0)

        # Convert to list
        result = [
            (belief_graph[bid], score)
            for bid, score in candidates.items()
            if bid in belief_graph
        ]

        # Sort by score
        result.sort(key=lambda x: x[1], reverse=True)

        return result[:k]

    def generate_by_recency(self,
                           beliefs: List[Belief],
                           k: int = 20) -> List[Tuple[Belief, float]]:
        """
        Generate candidates by recency.

        Args:
            beliefs: Pool of beliefs
            k: Number of candidates

        Returns:
            List of (belief, recency_score) tuples
        """
        now = datetime.now()

        candidates = []
        for belief in beliefs:
            # Time since last update (in hours)
            delta = (now - belief.updated_at).total_seconds() / 3600.0

            # Recency score: exponential decay
            # Half-life = 7 days (168 hours)
            half_life = 168.0
            recency = math.exp(-math.log(2) * delta / half_life)

            candidates.append((belief, recency))

        # Sort by recency
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:k]

    def generate_candidates(self,
                           query: str,
                           beliefs: List[Belief],
                           belief_graph: Dict[BeliefID, Belief],
                           context_beliefs: Optional[List[BeliefID]] = None,
                           k_per_channel: int = 20) -> List[Belief]:
        """
        Generate candidates through all channels (US-06).

        Args:
            query: User query
            beliefs: Pool of beliefs
            belief_graph: Belief graph for structure
            context_beliefs: Recent conversation context
            k_per_channel: Candidates per channel

        Returns:
            Deduplicated list of candidate beliefs
        """
        candidates_set: Set[BeliefID] = set()
        candidates: List[Belief] = []

        # Channel 1: Text similarity
        text_candidates = self.generate_by_text(query, beliefs, k_per_channel)
        for belief, _ in text_candidates:
            if belief.id not in candidates_set:
                candidates_set.add(belief.id)
                candidates.append(belief)

        # Channel 2: Graph structure
        if context_beliefs:
            struct_candidates = self.generate_by_structure(
                context_beliefs, belief_graph, max_hops=2, k=k_per_channel
            )
            for belief, _ in struct_candidates:
                if belief.id not in candidates_set:
                    candidates_set.add(belief.id)
                    candidates.append(belief)

        # Channel 3: Recency
        recent_candidates = self.generate_by_recency(beliefs, k_per_channel)
        for belief, _ in recent_candidates:
            if belief.id not in candidates_set:
                candidates_set.add(belief.id)
                candidates.append(belief)

        return candidates


# ============================================================================
# Unified Ranking (US-07)
# ============================================================================

class BeliefRanker:
    """
    Unified ranking with MMR for diversity (US-07).

    Score formula:
    score = w_sim·sim + w_conf·g(c) + w_rec·f(Δt) + w_rel·r̄ - w_x·contradiction_pressure

    Where:
    - g(c) = confidence transformation (linear or sigmoid)
    - f(Δt) = recency decay function
    - r̄ = average reliability of evidence
    """

    def __init__(self,
                 w_sim: float = 0.4,
                 w_conf: float = 0.25,
                 w_rec: float = 0.15,
                 w_rel: float = 0.15,
                 w_x: float = 0.05,
                 mmr_lambda: float = 0.7,
                 similarity_fn: Optional[Callable[[str, str], float]] = None):
        """
        Initialize ranker.

        Args:
            w_sim: Weight for similarity
            w_conf: Weight for confidence
            w_rec: Weight for recency
            w_rel: Weight for reliability
            w_x: Weight for contradiction penalty
            mmr_lambda: MMR diversity parameter (0 = max diversity, 1 = max relevance)
            similarity_fn: Similarity function
        """
        self.w_sim = w_sim
        self.w_conf = w_conf
        self.w_rec = w_rec
        self.w_rel = w_rel
        self.w_x = w_x
        self.mmr_lambda = mmr_lambda
        self.similarity_fn = similarity_fn or self._jaccard_similarity

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity fallback."""
        from .belief_estimation import SemanticEstimator
        estimator = SemanticEstimator()
        return estimator._calculate_similarity(text1, text2)

    def _confidence_transform(self, confidence: float) -> float:
        """Transform confidence to [0, 1] scale."""
        # Map [-1, 1] → [0, 1]
        return (confidence + 1.0) / 2.0

    def _recency_score(self, belief: Belief) -> float:
        """Calculate recency score."""
        now = datetime.now()
        delta_hours = (now - belief.updated_at).total_seconds() / 3600.0

        # Exponential decay with 7-day half-life
        half_life = 168.0  # hours
        return math.exp(-math.log(2) * delta_hours / half_life)

    def _reliability_score(self, belief: Belief, evidence_store=None) -> float:
        """
        Calculate average reliability of belief's evidence.

        Args:
            belief: Belief to score
            evidence_store: Optional evidence store to look up evidence

        Returns:
            Average reliability in [0, 1]
        """
        if evidence_store is None:
            # Fallback: use metadata if available
            return belief.metadata.get('avg_reliability', 0.5)

        evidences = evidence_store.get_evidence_for_belief(belief.id)
        if not evidences:
            return 0.5  # Default

        # Average reliability from evidence metadata
        reliabilities = [
            ev.metadata.get('reliability', 0.5)
            for ev in evidences
        ]

        return sum(reliabilities) / len(reliabilities) if reliabilities else 0.5

    def _contradiction_pressure(self,
                                belief: Belief,
                                belief_graph: Dict[BeliefID, Belief]) -> float:
        """
        Calculate contradiction pressure (US-08).

        Higher pressure if contradicting beliefs have high confidence.

        Args:
            belief: Belief to score
            belief_graph: Graph of all beliefs

        Returns:
            Pressure score in [0, 1]
        """
        if not belief.contradicts:
            return 0.0

        # Sum of transformed confidences of contradicting beliefs
        pressures = []
        for contra_id in belief.contradicts:
            if contra_id in belief_graph:
                contra_belief = belief_graph[contra_id]
                # Use absolute confidence (negatives also contribute)
                pressures.append(abs(contra_belief.confidence))

        if not pressures:
            return 0.0

        # Average pressure
        return sum(pressures) / len(pressures)

    def score_belief(self,
                    belief: Belief,
                    query: str,
                    belief_graph: Dict[BeliefID, Belief],
                    evidence_store=None) -> CandidateBelief:
        """
        Score a single belief.

        Args:
            belief: Belief to score
            query: User query
            belief_graph: Graph for contradiction lookup
            evidence_store: Optional evidence store

        Returns:
            CandidateBelief with scores
        """
        # Component scores
        sim = self.similarity_fn(query, belief.content)
        conf = self._confidence_transform(belief.confidence)
        rec = self._recency_score(belief)
        rel = self._reliability_score(belief, evidence_store)
        contra = self._contradiction_pressure(belief, belief_graph)

        # Combined score
        score = (
            self.w_sim * sim +
            self.w_conf * conf +
            self.w_rec * rec +
            self.w_rel * rel -
            self.w_x * contra
        )

        return CandidateBelief(
            belief=belief,
            score=score,
            similarity=sim,
            recency_score=rec,
            confidence_score=conf,
            reliability_score=rel,
            contradiction_pressure=contra
        )

    def rank_beliefs(self,
                    candidates: List[Belief],
                    query: str,
                    belief_graph: Dict[BeliefID, Belief],
                    evidence_store=None,
                    k: int = 8) -> List[CandidateBelief]:
        """
        Rank candidates without MMR.

        Args:
            candidates: Candidate beliefs
            query: User query
            belief_graph: Belief graph
            evidence_store: Optional evidence store
            k: Number to return

        Returns:
            Top-k ranked candidates
        """
        scored = [
            self.score_belief(belief, query, belief_graph, evidence_store)
            for belief in candidates
        ]

        # Sort by score
        scored.sort(key=lambda x: x.score, reverse=True)

        return scored[:k]

    def rank_with_mmr(self,
                     candidates: List[Belief],
                     query: str,
                     belief_graph: Dict[BeliefID, Belief],
                     evidence_store=None,
                     k: int = 8) -> List[CandidateBelief]:
        """
        Rank candidates with MMR for diversity (US-07).

        MMR balances relevance and diversity:
        MMR = λ·relevance - (1-λ)·max_similarity_to_selected

        Args:
            candidates: Candidate beliefs
            query: User query
            belief_graph: Belief graph
            evidence_store: Optional evidence store
            k: Number to return

        Returns:
            Top-k diverse candidates
        """
        # Score all candidates
        scored = [
            self.score_belief(belief, query, belief_graph, evidence_store)
            for belief in candidates
        ]

        if not scored:
            return []

        # MMR selection
        selected: List[CandidateBelief] = []
        remaining = scored.copy()

        while len(selected) < k and remaining:
            mmr_scores = []

            for candidate in remaining:
                # Relevance component
                relevance = candidate.score

                # Diversity component (max similarity to selected)
                if selected:
                    max_sim = max(
                        self.similarity_fn(candidate.belief.content, s.belief.content)
                        for s in selected
                    )
                else:
                    max_sim = 0.0

                # MMR score
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                mmr_scores.append((candidate, mmr))

            # Select best MMR score
            best = max(mmr_scores, key=lambda x: x[1])
            selected.append(best[0])
            remaining.remove(best[0])

        return selected


# ============================================================================
# Tension Detection (US-08)
# ============================================================================

@dataclass
class TensionPair:
    """
    A pair of contradicting beliefs in tension.

    Attributes:
        belief_a: First belief
        belief_b: Second belief (contradicts A)
        score_a: Relevance score of A
        score_b: Relevance score of B
        severity: How severe the tension is
    """
    belief_a: Belief
    belief_b: Belief
    score_a: float
    score_b: float
    severity: float = 0.0

    def __post_init__(self):
        """Calculate tension severity."""
        # Severity = min(score_a, score_b) * conflict_strength
        # High severity when both are highly relevant
        conflict_strength = min(abs(self.belief_a.confidence), abs(self.belief_b.confidence))
        self.severity = min(self.score_a, self.score_b) * conflict_strength


class TensionDetector:
    """
    Detects relevant contradictions (US-08).

    When two contradicting beliefs are both highly relevant,
    flags them as "in tension" for explicit deliberation.
    """

    def __init__(self, min_score_threshold: float = 0.5):
        """
        Initialize tension detector.

        Args:
            min_score_threshold: Minimum score for both beliefs to be "in tension"
        """
        self.min_score_threshold = min_score_threshold

    def detect_tensions(self,
                       ranked_beliefs: List[CandidateBelief],
                       belief_graph: Dict[BeliefID, Belief]) -> List[TensionPair]:
        """
        Detect tension pairs in ranked results.

        Args:
            ranked_beliefs: Ranked candidate beliefs
            belief_graph: Belief graph for contradiction lookup

        Returns:
            List of tension pairs
        """
        tensions: List[TensionPair] = []
        seen_pairs: Set[Tuple[BeliefID, BeliefID]] = set()

        for i, cand_a in enumerate(ranked_beliefs):
            belief_a = cand_a.belief

            # Check if any later belief contradicts this one
            for j in range(i + 1, len(ranked_beliefs)):
                cand_b = ranked_beliefs[j]
                belief_b = cand_b.belief

                # Check if they contradict
                if belief_b.id in belief_a.contradicts or belief_a.id in belief_b.contradicts:
                    # Both must have sufficient score
                    if cand_a.score >= self.min_score_threshold and \
                       cand_b.score >= self.min_score_threshold:

                        # Avoid duplicates
                        pair_key = tuple(sorted([belief_a.id, belief_b.id]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)

                            tension = TensionPair(
                                belief_a=belief_a,
                                belief_b=belief_b,
                                score_a=cand_a.score,
                                score_b=cand_b.score
                            )
                            tensions.append(tension)

        # Sort by severity
        tensions.sort(key=lambda t: t.severity, reverse=True)

        return tensions
