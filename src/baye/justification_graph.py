"""
Justification Graph: Core belief tracking system with causal propagation.

This is the main engine that maintains a directed graph of beliefs where edges
represent justification relationships. When a belief changes, the system
propagates updates through the graph using both deterministic (causal) and
probabilistic (semantic) mechanisms.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import networkx as nx
from datetime import datetime

from .belief_types import (
    Belief, PropagationEvent, PropagationResult, RelationType,
    BeliefID, Confidence, Delta
)


class JustificationGraph:
    """
    A graph-based belief tracking system that maintains causal relationships
    between beliefs and propagates confidence updates through justification chains.
    
    Key features:
    - Automatic justification link discovery
    - Dual propagation (causal + semantic)
    - Cycle detection and handling
    - Conflict resolution
    - Budget-limited propagation to prevent explosion
    """
    
    def __init__(self, max_depth: int = 4):
        """
        Initialize the justification graph.
        
        Args:
            max_depth: Maximum depth for propagation cascades
        """
        self.beliefs: Dict[BeliefID, Belief] = {}
        self.nx_graph = nx.DiGraph()  # NetworkX graph for analysis
        
        # Propagation parameters
        self.max_depth = max_depth
        self.propagation_budget = {0: 8, 1: 5, 2: 3, 3: 2}  # Updates per depth level
        self.min_delta_threshold = 0.05  # Minimum delta to propagate
        
        # History tracking
        self.propagation_history: List[PropagationResult] = []
        
        # Statistics
        self.total_propagations = 0
        self.total_beliefs_created = 0
    
    def add_belief(self, content: str, confidence: float, 
                   context: str = "general", source_task: str = "unknown",
                   supported_by: Optional[List[BeliefID]] = None) -> Belief:
        """
        Add a new belief to the graph.
        
        Args:
            content: The belief statement
            confidence: Initial confidence in [-1, 1]
            context: Domain category
            source_task: Task that generated this belief
            supported_by: Optional list of existing belief IDs that justify this one
            
        Returns:
            The created Belief object
        """
        belief = Belief(content, confidence, context, source_task)
        
        # Add justification links if provided
        if supported_by:
            for parent_id in supported_by:
                if parent_id in self.beliefs:
                    belief.add_supporter(parent_id)
                    self.beliefs[parent_id].add_dependent(belief.id)
        
        # Store belief
        self.beliefs[belief.id] = belief
        
        # Update NetworkX graph
        self.nx_graph.add_node(belief.id, 
                               confidence=confidence,
                               content=content,
                               context=context)
        
        # Add edges for justifications
        for parent_id in belief.supported_by:
            self.nx_graph.add_edge(parent_id, belief.id, weight=1.0)
        
        self.total_beliefs_created += 1
        
        return belief
    
    def link_beliefs(self, parent_id: BeliefID, child_id: BeliefID, 
                    relation: RelationType = RelationType.SUPPORTS):
        """
        Create an explicit relationship between two beliefs.
        
        Args:
            parent_id: Belief that justifies/affects
            child_id: Belief that is justified/affected
            relation: Type of relationship
        """
        if parent_id not in self.beliefs or child_id not in self.beliefs:
            raise ValueError("Both beliefs must exist in graph")
        
        parent = self.beliefs[parent_id]
        child = self.beliefs[child_id]
        
        if relation == RelationType.SUPPORTS:
            parent.add_dependent(child_id)
            child.add_supporter(parent_id)
            self.nx_graph.add_edge(parent_id, child_id, weight=1.0)
            
        elif relation == RelationType.CONTRADICTS:
            parent.contradicts.append(child_id)
            child.contradicts.append(parent_id)
    
    def propagate_from(self, origin_id: BeliefID, initial_delta: float = 0.0) -> PropagationResult:
        """
        Propagate confidence updates starting from a belief.
        
        This is the main entry point for belief updates. When a belief changes,
        it triggers a cascade through both causal (graph) and semantic (similarity)
        channels.
        
        Args:
            origin_id: Belief to start propagation from
            initial_delta: Initial change (0.0 if belief is new)
            
        Returns:
            PropagationResult with all events and statistics
        """
        if origin_id not in self.beliefs:
            raise ValueError(f"Belief {origin_id} not found")
        
        result = PropagationResult(
            origin_belief_id=origin_id,
            total_beliefs_updated=0,
            max_depth_reached=0
        )
        
        visited: Set[BeliefID] = set()
        
        # Start propagation
        self._propagate_recursive(
            current_id=origin_id,
            delta=initial_delta,
            depth=0,
            visited=visited,
            result=result
        )
        
        result.total_beliefs_updated = len(visited)
        self.propagation_history.append(result)
        self.total_propagations += 1
        
        return result
    
    def _propagate_recursive(self, current_id: BeliefID, delta: Delta, 
                            depth: int, visited: Set[BeliefID],
                            result: PropagationResult):
        """
        Recursive propagation through the graph.
        
        Args:
            current_id: Current belief being processed
            delta: Change in confidence to propagate
            depth: Current recursion depth
            visited: Set of already-processed beliefs (prevents cycles)
            result: Accumulator for propagation events
        """
        # Termination conditions
        if depth >= self.max_depth:
            return
        
        if current_id in visited:
            return
        
        visited.add(current_id)
        result.max_depth_reached = max(result.max_depth_reached, depth)
        
        # Get propagation budget for this depth
        budget = self.propagation_budget.get(depth, 1)
        
        # Calculate updates for children via causal propagation
        causal_updates = self._causal_propagation(current_id, delta, depth)
        
        # Sort by impact and apply budget
        sorted_updates = sorted(
            causal_updates.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:budget]
        
        # Apply updates and recurse
        for child_id, child_delta in sorted_updates:
            # Check threshold
            depth_threshold = self.min_delta_threshold * (1.2 ** depth)
            if abs(child_delta) < depth_threshold:
                continue
            
            # Apply update
            old_conf = self.beliefs[child_id].confidence
            actual_delta = self.beliefs[child_id].update_confidence(child_delta)
            new_conf = self.beliefs[child_id].confidence
            
            # Record event
            event = PropagationEvent(
                belief_id=child_id,
                old_confidence=old_conf,
                new_confidence=new_conf,
                delta=actual_delta,
                depth=depth,
                propagation_type="causal",
                source_belief_id=current_id
            )
            result.events.append(event)
            
            # Recurse if significant change
            if abs(actual_delta) > depth_threshold:
                self._propagate_recursive(child_id, actual_delta, depth + 1, visited, result)
    
    def _causal_propagation(self, origin_id: BeliefID, delta: Delta, depth: int) -> Dict[BeliefID, Delta]:
        """
        Calculate confidence updates via causal (graph) relationships.
        
        This is deterministic propagation through explicit justification links.
        Uses logistic saturation to prevent linear explosion.
        
        Args:
            origin_id: Belief that changed
            delta: How much it changed
            depth: Current depth (for dampening)
            
        Returns:
            Dict mapping child belief IDs to their deltas
        """
        updates = {}
        origin_belief = self.beliefs[origin_id]
        origin_conf = origin_belief.confidence
        
        # For each belief that this one supports
        for child_id in origin_belief.supports:
            # Calculate dependency strength
            dependency = self._calculate_dependency(child_id, origin_id)
            
            # Calculate relative change in supporter
            # If origin went from 0.8 to 0.5, relative_change = 0.5/0.8 = 0.625
            if origin_conf != 0:
                new_origin_conf = origin_conf + delta
                relative_change = new_origin_conf / origin_conf
            else:
                relative_change = 1.0
            
            # Child confidence changes proportionally
            child_belief = self.beliefs[child_id]
            child_delta = child_belief.confidence * (relative_change - 1.0) * dependency
            
            # Apply centrality dampening
            centrality_factor = self._centrality_dampening(origin_id)
            child_delta *= centrality_factor
            
            updates[child_id] = child_delta
        
        return updates
    
    def _calculate_dependency(self, child_id: BeliefID, parent_id: BeliefID) -> float:
        """
        Calculate how much child depends on parent using logistic saturation.
        
        Prevents linear explosion when supporters have very high confidence.
        
        Args:
            child_id: Dependent belief
            parent_id: Supporting belief
            
        Returns:
            Dependency strength in [0, 1]
        """
        child = self.beliefs[child_id]
        
        if parent_id not in child.supported_by:
            return 0.0
        
        if not child.supported_by:
            return 0.0
        
        # Base weight: equal split among all supporters
        num_supporters = len(child.supported_by)
        base_weight = 1.0 / num_supporters
        
        # Logistic saturation of parent confidence
        parent_conf = self.beliefs[parent_id].confidence
        parent_influence = self._logistic(parent_conf, k=10, midpoint=0.5)
        
        # Total influence from all supporters
        total_influence = sum(
            self._logistic(self.beliefs[sid].confidence, k=10, midpoint=0.5)
            for sid in child.supported_by
        )
        
        if total_influence == 0:
            return 0.0
        
        # Weighted by relative influence
        return base_weight * (parent_influence / total_influence)
    
    @staticmethod
    def _logistic(x: float, k: float = 10, midpoint: float = 0.5) -> float:
        """
        Logistic function for saturation.
        
        Maps confidence to influence with saturation at extremes.
        At confidence=0.9, influence doesn't grow much more.
        
        Args:
            x: Input value (confidence)
            k: Steepness parameter
            midpoint: Inflection point
            
        Returns:
            Saturated value in [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-k * (x - midpoint)))
    
    def _centrality_dampening(self, belief_id: BeliefID) -> float:
        """
        Hub beliefs (with many dependents) propagate with less force.
        
        Prevents central beliefs from causing cascading updates on every
        micro-adjustment.
        
        Args:
            belief_id: Belief to check
            
        Returns:
            Dampening factor in (0, 1]
        """
        num_dependents = len(self.beliefs[belief_id].supports)
        return 1.0 / np.log2(2 + num_dependents)
    
    def detect_cycles(self) -> List[List[BeliefID]]:
        """
        Detect circular justification chains.
        
        Returns:
            List of cycles, each cycle is a list of belief IDs
        """
        try:
            cycles = list(nx.simple_cycles(self.nx_graph))
            return cycles
        except:
            return []
    
    def get_belief_path(self, from_id: BeliefID, to_id: BeliefID) -> Optional[List[BeliefID]]:
        """
        Find justification path between two beliefs.
        
        Args:
            from_id: Starting belief
            to_id: Target belief
            
        Returns:
            List of belief IDs forming the path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.nx_graph, from_id, to_id)
        except nx.NetworkXNoPath:
            return None
    
    def get_centrality_scores(self) -> Dict[BeliefID, float]:
        """
        Calculate PageRank-style centrality for all beliefs.
        
        Identifies "hub" beliefs that many others depend on.
        
        Returns:
            Dict mapping belief IDs to centrality scores
        """
        if len(self.beliefs) == 0:
            return {}
        
        try:
            return nx.pagerank(self.nx_graph)
        except:
            return {bid: 1.0 / len(self.beliefs) for bid in self.beliefs}
    
    def explain_confidence(self, belief_id: BeliefID, max_depth: int = 3) -> str:
        """
        Generate human-readable explanation of why a belief has its confidence.
        
        Traces back through justification chain.
        
        Args:
            belief_id: Belief to explain
            max_depth: How far back to trace
            
        Returns:
            Formatted explanation string
        """
        if belief_id not in self.beliefs:
            return f"Belief {belief_id} not found"
        
        belief = self.beliefs[belief_id]
        lines = [f"Belief: '{belief.content}'"]
        lines.append(f"Confidence: {belief.confidence:.2f}")
        lines.append(f"Context: {belief.context}")
        lines.append(f"Last updated: {belief.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        if belief.supported_by:
            lines.append("Justified by:")
            for supporter_id in belief.supported_by:
                supporter = self.beliefs[supporter_id]
                lines.append(f"  • [{supporter.confidence:.2f}] {supporter.content[:60]}")
        else:
            lines.append("(No explicit justifications - foundational belief)")
        
        if belief.supports:
            lines.append("")
            lines.append("Justifies:")
            for dependent_id in belief.supports[:5]:  # Show top 5
                dependent = self.beliefs[dependent_id]
                lines.append(f"  • [{dependent.confidence:.2f}] {dependent.content[:60]}")
        
        return "\n".join(lines)
    
    def add_belief_with_estimation(
        self,
        content: str,
        context: str = "general",
        source_task: str = "unknown",
        k: int = 5,
        auto_link: bool = True,
        link_threshold: float = 0.7,
        verbose: bool = False
    ) -> Belief:
        """
        Add a new belief with confidence estimated from existing beliefs.
        
        Uses K-NN semantic estimation to infer appropriate confidence based
        on similar existing beliefs. Optionally auto-links to semantic neighbors.
        
        Args:
            content: The belief statement
            context: Domain category
            source_task: Task that generated this belief
            k: Number of neighbors for estimation
            auto_link: Whether to auto-link to similar beliefs
            link_threshold: Minimum similarity for auto-linking
            verbose: Print estimation details
            
        Returns:
            The created Belief object with estimated confidence
        """
        from .belief_estimation import SemanticEstimator
        
        if not self.beliefs:
            # No existing beliefs to estimate from
            return self.add_belief(content, 0.5, context, source_task)
        
        # Estimate confidence
        estimator = SemanticEstimator()
        estimated_conf, neighbor_ids, similarities = estimator.estimate_confidence(
            content,
            list(self.beliefs.values()),
            k=k,
            verbose=verbose
        )
        
        # Create belief
        belief = self.add_belief(content, estimated_conf, context, source_task)
        
        # Auto-link to similar beliefs if requested
        if auto_link and neighbor_ids:
            for neighbor_id, similarity in zip(neighbor_ids, similarities):
                if similarity >= link_threshold:
                    # Link semantically similar beliefs
                    # Direction: neighbor supports new belief (if neighbor has higher conf)
                    neighbor = self.beliefs[neighbor_id]
                    if neighbor.confidence > estimated_conf:
                        self.link_beliefs(neighbor_id, belief.id)
                        if verbose:
                            print(f"  [AUTO-LINK] {neighbor.content[:30]} → {content[:30]}")
        
        return belief
    
    def batch_add_beliefs_with_estimation(
        self,
        beliefs_data: List[Tuple[str, str]],
        k: int = 5,
        verbose: bool = False
    ) -> List[BeliefID]:
        """
        Add multiple beliefs with estimation in batch.
        
        Useful for initializing a graph from a list of belief statements.
        
        Args:
            beliefs_data: List of (content, context) tuples
            k: Number of neighbors for each estimation
            verbose: Print details
            
        Returns:
            List of created belief IDs
        """
        created_ids = []
        
        for i, (content, context) in enumerate(beliefs_data):
            if verbose:
                print(f"\n[{i+1}/{len(beliefs_data)}] Adding: '{content[:40]}...'")
            
            belief = self.add_belief_with_estimation(
                content, context, k=k, verbose=verbose
            )
            created_ids.append(belief.id)
        
        return created_ids
    
    def __repr__(self) -> str:
        num_beliefs = len(self.beliefs)
        num_edges = self.nx_graph.number_of_edges()
        return f"JustificationGraph(beliefs={num_beliefs}, links={num_edges})"
