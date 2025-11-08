"""
Semantic confidence estimation using K-Nearest Neighbors.

This module provides cold-start confidence initialization for new beliefs
by leveraging semantic similarity to existing beliefs in the graph.

Core idea: A new belief's confidence is estimated as a weighted average
of similar existing beliefs, where weights are semantic similarity scores.
"""

from typing import List, Tuple, Optional
import numpy as np

from belief_types import Belief, BeliefID


class SemanticEstimator:
    """
    Estimates confidence for new beliefs using semantic K-NN.
    
    When adding a new belief without explicit confidence, the system
    can infer an appropriate initial value based on:
    1. Semantic similarity to existing beliefs
    2. Confidences of those similar beliefs
    3. Variance in neighbor confidences (uncertainty measure)
    """
    
    def __init__(self, similarity_threshold: float = 0.2, 
                 dampening_factor: float = 0.9):
        """
        Initialize semantic estimator.
        
        Args:
            similarity_threshold: Minimum similarity to consider (filters noise)
            dampening_factor: Attenuate extreme similarities to prevent overfitting
        """
        self.similarity_threshold = similarity_threshold
        self.dampening_factor = dampening_factor
    
    def estimate_confidence(
        self,
        new_content: str,
        existing_beliefs: List[Belief],
        k: int = 5,
        verbose: bool = False
    ) -> Tuple[float, List[BeliefID], List[float]]:
        """
        Estimate confidence for a new belief using K-NN.
        
        Algorithm:
        1. Calculate semantic similarity to all existing beliefs
        2. Filter by threshold (remove noise)
        3. Apply dampening to extreme similarities
        4. Take top-K most similar
        5. Weighted average: conf = Σ(sim_i * conf_i) / Σ(sim_i)
        
        Args:
            new_content: Content of new belief
            existing_beliefs: List of beliefs in graph
            k: Number of neighbors to consider
            verbose: Print detailed breakdown
            
        Returns:
            (estimated_confidence, used_belief_ids, similarity_weights)
        """
        if not existing_beliefs:
            return 0.0, [], []  # Neutral confidence if no reference
        
        # Calculate similarities to all beliefs
        similarities = []
        for belief in existing_beliefs:
            sim = self._calculate_similarity(new_content, belief.content)
            
            # Apply dampening to very high similarities
            # Rationale: perfect matches (1.0) shouldn't completely dominate
            if sim > 0.9:
                original_sim = sim
                sim = 0.9 + (sim - 0.9) * self.dampening_factor
                if verbose:
                    print(f"  [DAMPEN] {original_sim:.3f} -> {sim:.3f} for '{belief.content[:30]}'")
            
            # Filter by threshold
            if sim >= self.similarity_threshold:
                similarities.append((belief, sim))
        
        if not similarities:
            if verbose:
                print(f"  [WARN] No similar beliefs found (threshold={self.similarity_threshold})")
            return 0.0, [], []
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-K
        top_k = similarities[:k]
        
        # Weighted average
        weighted_sum = sum(belief.confidence * sim for belief, sim in top_k)
        total_weight = sum(sim for _, sim in top_k)
        
        estimated_conf = weighted_sum / total_weight
        
        # Clamp to valid range
        estimated_conf = max(-1.0, min(1.0, estimated_conf))
        
        # Extract provenance
        used_ids = [belief.id for belief, _ in top_k]
        weights = [sim for _, sim in top_k]
        
        if verbose:
            self._print_estimation_details(new_content, top_k, estimated_conf)
        
        return estimated_conf, used_ids, weights
    
    def estimate_with_uncertainty(
        self,
        new_content: str,
        existing_beliefs: List[Belief],
        k: int = 5,
        verbose: bool = False
    ) -> Tuple[float, float, List[BeliefID]]:
        """
        Estimate confidence with uncertainty measure.
        
        Uncertainty is high when:
        - Neighbors have divergent confidences (high variance)
        - Similarities are spread out (no clear consensus)
        - Few neighbors found
        
        Args:
            new_content: Content of new belief
            existing_beliefs: List of beliefs
            k: Number of neighbors
            verbose: Print details
            
        Returns:
            (estimated_confidence, uncertainty, used_belief_ids)
            uncertainty ∈ [0, 1] where 0=certain, 1=very uncertain
        """
        conf, ids, weights = self.estimate_confidence(
            new_content, existing_beliefs, k, verbose
        )
        
        if not ids:
            return conf, 1.0, []  # Maximum uncertainty when no neighbors
        
        # Get beliefs that were used
        used_beliefs = [b for b in existing_beliefs if b.id in ids]
        
        # Confidence variance among neighbors
        confidences = [b.confidence for b in used_beliefs]
        conf_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        
        # Similarity spread (low spread = strong consensus)
        sim_variance = np.var(weights) if len(weights) > 1 else 0.0
        
        # Penalize small sample size
        sample_penalty = max(0, (k - len(ids)) / k)  # Penalty if fewer than k neighbors
        
        # Combined uncertainty
        # High when: divergent confidences OR spread out similarities OR few samples
        uncertainty = (conf_variance * 0.5 + sim_variance * 0.3 + sample_penalty * 0.2)
        uncertainty = min(1.0, uncertainty)  # Clamp to [0, 1]
        
        if verbose:
            print(f"\n[UNCERTAINTY BREAKDOWN]")
            print(f"  Confidence variance: {conf_variance:.3f}")
            print(f"  Similarity variance: {sim_variance:.3f}")
            print(f"  Sample penalty: {sample_penalty:.3f}")
            print(f"  Combined uncertainty: {uncertainty:.3f}")
        
        return conf, uncertainty, ids
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        V1.0: Uses enhanced Jaccard with normalization
        V1.5+: Replace with sentence-transformers cosine similarity
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score in [0, 1]
        """
        # Normalize and tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "can", "be", "to", "for", "of", "in", "on"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # Boost score if there's any overlap (helps with short phrases)
        if intersection > 0:
            # Boost proportional to how much of the smaller set overlaps
            smaller_set_size = min(len(words1), len(words2))
            overlap_ratio = intersection / smaller_set_size
            # Weighted average: 70% Jaccard, 30% overlap ratio
            boosted = 0.7 * jaccard + 0.3 * overlap_ratio
            return min(1.0, boosted)
        
        return jaccard
    
    def _print_estimation_details(self, new_content: str, 
                                  neighbors: List[Tuple[Belief, float]], 
                                  estimated_conf: float):
        """Pretty print estimation breakdown."""
        print(f"\n[ESTIMATE] '{new_content[:50]}...'")
        print(f"  Using {len(neighbors)} neighbors → confidence: {estimated_conf:.2f}")
        print(f"  Neighbors:")
        
        for belief, sim in neighbors:
            arrow = "↑" if belief.confidence > 0 else "↓"
            conf_bar = "█" * int(abs(belief.confidence) * 10)
            print(f"    {arrow} [{belief.confidence:+.2f}] {conf_bar:10} "
                  f"(sim: {sim:.2f}) {belief.content[:40]}")


class BeliefInitializer:
    """
    High-level interface for adding beliefs with automatic confidence estimation.
    """
    
    def __init__(self, estimator: SemanticEstimator):
        self.estimator = estimator
    
    def should_use_estimation(self, new_content: str, 
                            existing_beliefs: List[Belief],
                            min_neighbors: int = 2) -> bool:
        """
        Decide whether to use estimation or default confidence.
        
        Args:
            new_content: New belief content
            existing_beliefs: Existing beliefs
            min_neighbors: Minimum similar beliefs required
            
        Returns:
            True if estimation is reliable, False to use default
        """
        # Quick check: are there enough similar beliefs?
        similar_count = sum(
            1 for b in existing_beliefs 
            if self.estimator._calculate_similarity(new_content, b.content) 
               >= self.estimator.similarity_threshold
        )
        
        return similar_count >= min_neighbors
    
    def initialize_with_strategy(
        self,
        new_content: str,
        existing_beliefs: List[Belief],
        default_confidence: float = 0.5,
        k: int = 5,
        uncertainty_threshold: float = 0.7,
        verbose: bool = False
    ) -> Tuple[float, str]:
        """
        Initialize belief confidence with fallback strategy.
        
        Strategy:
        1. Try K-NN estimation
        2. If uncertainty too high, use conservative default
        3. If no neighbors, use provided default
        
        Args:
            new_content: New belief content
            existing_beliefs: Existing beliefs
            default_confidence: Fallback value
            k: Number of neighbors for estimation
            uncertainty_threshold: Max acceptable uncertainty
            verbose: Print decisions
            
        Returns:
            (confidence, strategy_used)
            strategy_used ∈ {"knn", "conservative", "default"}
        """
        # Try estimation
        conf, uncertainty, ids = self.estimator.estimate_with_uncertainty(
            new_content, existing_beliefs, k, verbose
        )
        
        if not ids:
            # No neighbors found
            if verbose:
                print(f"[INIT] No neighbors → default confidence: {default_confidence:.2f}")
            return default_confidence, "default"
        
        if uncertainty > uncertainty_threshold:
            # High uncertainty: use conservative estimate
            conservative_conf = conf * (1 - uncertainty)  # Shrink toward zero
            if verbose:
                print(f"[INIT] High uncertainty ({uncertainty:.2f}) → "
                      f"conservative: {conservative_conf:.2f}")
            return conservative_conf, "conservative"
        
        # Good estimation
        if verbose:
            print(f"[INIT] K-NN estimate with low uncertainty → {conf:.2f}")
        return conf, "knn"


# Utility functions for common patterns
def estimate_belief_confidence(
    new_content: str,
    existing_beliefs: List[Belief],
    k: int = 5,
    verbose: bool = False
) -> float:
    """
    Convenience function: estimate confidence for new belief.
    
    Args:
        new_content: New belief text
        existing_beliefs: Existing beliefs in graph
        k: Number of neighbors
        verbose: Print details
        
    Returns:
        Estimated confidence in [-1, 1]
    """
    estimator = SemanticEstimator()
    conf, _, _ = estimator.estimate_confidence(new_content, existing_beliefs, k, verbose)
    return conf


def get_supporting_neighbors(
    new_content: str,
    existing_beliefs: List[Belief],
    k: int = 5
) -> List[BeliefID]:
    """
    Get IDs of beliefs that support estimating new belief's confidence.
    
    Useful for auto-linking new beliefs to their semantic neighbors.
    
    Args:
        new_content: New belief text
        existing_beliefs: Existing beliefs
        k: Number of neighbors
        
    Returns:
        List of belief IDs sorted by similarity
    """
    estimator = SemanticEstimator()
    _, ids, _ = estimator.estimate_confidence(new_content, existing_beliefs, k)
    return ids
