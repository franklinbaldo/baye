"""
Fact Store - Ground truth facts with semantic retrieval

Facts are observed truths that entered the model's context, stored with
provenance tracking and used to validate claims.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import uuid


@dataclass
class Fact:
    """Ground truth fact with provenance"""
    id: str
    content: str
    source_type: str  # "user_message", "document", "api", "manual"
    source_id: str  # UUID of the source
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0  # Usually 1.0, can be lower for uncertain sources
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def __repr__(self):
        return (
            f"Fact(id={self.id[:8]}..., "
            f"content='{self.content[:50]}...', "
            f"source={self.source_type})"
        )


class FactStore:
    """
    Vector store for facts with semantic retrieval

    Features:
    - Semantic search via embeddings
    - Find contradicting facts/beliefs
    - Provenance tracking
    - Integration with SemanticEstimator
    """

    def __init__(self, estimator=None):
        """
        Args:
            estimator: SemanticEstimator for generating embeddings
        """
        self.facts: Dict[str, Fact] = {}  # id → Fact
        self.estimator = estimator  # Will be set by BeliefTracker

    def add_fact(
        self,
        content: str,
        source_type: str,
        source_id: str,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> Fact:
        """
        Add a fact to the store

        Args:
            content: The factual statement
            source_type: Type of source (user_message, document, etc.)
            source_id: UUID of the source
            confidence: Confidence in this fact (default: 1.0)
            metadata: Additional metadata

        Returns:
            Created Fact
        """
        fact_id = str(uuid.uuid4())

        # Generate embedding if estimator available
        embedding = None
        if self.estimator:
            # Use estimator to generate embedding
            # We'll create a temporary belief-like object for embedding
            from .belief_types import Belief
            temp_belief = Belief(
                id=fact_id,
                content=content,
                confidence=confidence,
                context="fact"
            )
            # Get embedding via estimator's internal method
            # (We'll need to expose this or use the estimator directly)
            # For now, we'll defer embedding generation
            pass

        fact = Fact(
            id=fact_id,
            content=content,
            source_type=source_type,
            source_id=source_id,
            confidence=confidence,
            metadata=metadata or {},
            embedding=embedding
        )

        self.facts[fact_id] = fact
        return fact

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get fact by ID"""
        return self.facts.get(fact_id)

    def find_contradicting(
        self,
        content: str,
        k: int = 3,
        include_beliefs: bool = True,
        belief_graph=None
    ) -> List[Tuple[str, str, float, str]]:
        """
        Find facts (and optionally beliefs) that contradict the given content

        This is called when a claim is outside margin to show the LLM
        what contradicts its claim.

        Args:
            content: The claim to check
            k: Number of contradictions to return
            include_beliefs: Also search beliefs for contradictions
            belief_graph: JustificationGraph to search beliefs

        Returns:
            List of (type, id, confidence, content) tuples
            type: "fact" or "belief"
            id: UUID
            confidence: Confidence value
            content: The contradicting content
        """
        contradictions = []

        # Search facts
        for fact in self.facts.values():
            # Simple semantic similarity (inverse for contradiction)
            # In production, use proper semantic similarity
            similarity = self._simple_similarity(content, fact.content)

            # Low similarity might indicate contradiction
            # (This is simplified - proper implementation would use
            #  semantic analysis to detect actual contradictions)
            if similarity < 0.3:  # Threshold for potential contradiction
                contradictions.append((
                    "fact",
                    fact.id,
                    fact.confidence,
                    fact.content
                ))

        # Search beliefs if requested
        if include_beliefs and belief_graph:
            for belief in belief_graph.beliefs.values():
                similarity = self._simple_similarity(content, belief.content)
                if similarity < 0.3:
                    contradictions.append((
                        "belief",
                        belief.id,
                        belief.confidence,
                        belief.content
                    ))

        # Sort by confidence (descending) and take top k
        contradictions.sort(key=lambda x: x[2], reverse=True)
        return contradictions[:k]

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        Simple similarity metric (word overlap)

        TODO: Replace with proper semantic similarity using embeddings
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def find_similar(
        self,
        content: str,
        k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[Fact, float]]:
        """
        Find k most similar facts

        Args:
            content: Query content
            k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (Fact, similarity) tuples
        """
        similarities = []

        for fact in self.facts.values():
            similarity = self._simple_similarity(content, fact.content)
            if similarity >= min_similarity:
                similarities.append((fact, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def verify_claim(
        self,
        content: str,
        threshold: float = 0.8
    ) -> Optional[Fact]:
        """
        Check if claim matches a known fact

        Args:
            content: Claim to verify
            threshold: Similarity threshold for match

        Returns:
            Matching fact if found, None otherwise
        """
        similar = self.find_similar(content, k=1, min_similarity=threshold)
        if similar:
            return similar[0][0]
        return None

    def format_for_context(self, max_facts: int = 10) -> str:
        """
        Format facts for inclusion in LLM context

        Returns a highlighted section showing recent/relevant facts
        """
        if not self.facts:
            return ""

        # Get most recent facts
        recent_facts = sorted(
            self.facts.values(),
            key=lambda f: f.timestamp,
            reverse=True
        )[:max_facts]

        lines = ["📌 **KNOWN FACTS** (Ground Truth):\n"]

        for fact in recent_facts:
            source_info = f"{fact.source_type}"
            if fact.source_id:
                source_info += f" ({fact.source_id[:8]})"

            lines.append(
                f"  • [{fact.id[:8]}] {fact.content}\n"
                f"    Source: {source_info} | "
                f"Confidence: {fact.confidence:.2f} | "
                f"Time: {fact.timestamp.strftime('%Y-%m-%d %H:%M')}"
            )

        return "\n".join(lines)

    def list_facts(self, limit: int = 20) -> List[Dict]:
        """
        List facts for CLI display

        Returns:
            List of fact dictionaries
        """
        facts_list = sorted(
            self.facts.values(),
            key=lambda f: f.timestamp,
            reverse=True
        )[:limit]

        return [
            {
                "id": f.id,
                "content": f.content,
                "source_type": f.source_type,
                "source_id": f.source_id,
                "confidence": f.confidence,
                "timestamp": f.timestamp,
            }
            for f in facts_list
        ]
