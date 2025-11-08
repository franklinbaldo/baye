"""
Integrated example: Adding beliefs with automatic confidence estimation.

This demonstrates the complete workflow of using semantic K-NN to initialize
new beliefs without manual confidence specification.
"""

from baye import JustificationGraph, SemanticEstimator
from baye.belief_estimation import BeliefInitializer


def main():
    print("="*70)
    print("INTEGRATED EXAMPLE: SEMANTIC CONFIDENCE ESTIMATION")
    print("="*70 + "\n")
    
    # Step 1: Create graph with initial beliefs (manual confidence)
    print("Step 1: Initialize graph with foundational beliefs\n")
    print("-" * 70)
    
    graph = JustificationGraph()
    
    # Add foundational beliefs manually
    b1 = graph.add_belief(
        "External services and APIs are unreliable",
        confidence=0.7,
        context="infrastructure"
    )
    print(f"Added: {b1.content} [{b1.confidence:.2f}]")
    
    b2 = graph.add_belief(
        "Always validate and sanitize user input data",
        confidence=0.8,
        context="security",
        supported_by=[b1.id]
    )
    print(f"Added: {b2.content} [{b2.confidence:.2f}]")
    print(f"  ↳ supported by: {b1.content[:30]}...")
    
    b3 = graph.add_belief(
        "Use defensive programming and error handling",
        confidence=0.6,
        context="development",
        supported_by=[b1.id]
    )
    print(f"Added: {b3.content} [{b3.confidence:.2f}]")
    print(f"  ↳ supported by: {b1.content[:30]}...")
    
    # Step 2: Add new beliefs with automatic estimation
    print("\n" + "="*70)
    print("Step 2: Add new beliefs with AUTOMATIC confidence estimation")
    print("="*70 + "\n")
    
    # New belief 1: Similar to existing API/infrastructure beliefs
    print("--- New Belief 1 ---\n")
    new1 = graph.add_belief_with_estimation(
        content="APIs and external services can timeout",
        context="infrastructure",
        k=3,
        auto_link=True,
        verbose=True
    )
    print(f"\n✓ Added with estimated confidence: {new1.confidence:.2f}\n")
    
    # New belief 2: Similar to security beliefs
    print("--- New Belief 2 ---\n")
    new2 = graph.add_belief_with_estimation(
        content="Sanitize and validate all user data input",
        context="security",
        k=3,
        auto_link=True,
        verbose=True
    )
    print(f"\n✓ Added with estimated confidence: {new2.confidence:.2f}\n")
    
    # New belief 3: Similar to defensive programming
    print("--- New Belief 3 ---\n")
    new3 = graph.add_belief_with_estimation(
        content="Defensive programming with error handling is important",
        context="development",
        k=3,
        auto_link=True,
        verbose=True
    )
    print(f"\n✓ Added with estimated confidence: {new3.confidence:.2f}\n")
    
    # Step 3: Batch addition
    print("="*70)
    print("Step 3: Batch add multiple beliefs")
    print("="*70 + "\n")
    
    batch_beliefs = [
        ("APIs and services should cache responses", "performance"),
        ("Log and debug all errors", "logging"),
        ("Handle all programming exceptions", "development"),
    ]
    
    batch_ids = graph.batch_add_beliefs_with_estimation(
        batch_beliefs, k=5, verbose=False
    )
    
    print(f"Added {len(batch_ids)} beliefs via batch estimation:\n")
    for bid in batch_ids:
        belief = graph.beliefs[bid]
        print(f"  [{belief.confidence:.2f}] {belief.content}")
    
    # Step 4: Analyze final graph state
    print("\n" + "="*70)
    print("Step 4: Final graph state")
    print("="*70 + "\n")
    
    print(f"Graph: {graph}\n")
    print("All beliefs sorted by confidence:\n")
    
    all_beliefs = sorted(
        graph.beliefs.values(), 
        key=lambda b: b.confidence, 
        reverse=True
    )
    
    for belief in all_beliefs:
        conf_bar = "█" * int(belief.confidence * 10)
        print(f"[{belief.confidence:.2f}] {conf_bar:10} {belief.content}")
    
    # Step 5: Show justification trace
    print("\n" + "="*70)
    print("Step 5: Justification traces for estimated beliefs")
    print("="*70 + "\n")
    
    print(graph.explain_confidence(new1.id))
    print("\n" + "-"*70 + "\n")
    print(graph.explain_confidence(new2.id))
    
    # Step 6: Demonstrate uncertainty-based estimation
    print("\n" + "="*70)
    print("Step 6: Estimation with uncertainty awareness")
    print("="*70 + "\n")
    
    # Add conflicting beliefs to demonstrate uncertainty
    graph.add_belief(
        "External services and APIs are always reliable",
        confidence=-0.5,  # Contradicts b1
        context="infrastructure"
    )
    
    estimator = SemanticEstimator()
    initializer = BeliefInitializer(estimator)
    
    # Try to estimate with conflicting evidence
    new_content = "External APIs and services have variable reliability"
    conf, uncertainty, ids = estimator.estimate_with_uncertainty(
        new_content,
        list(graph.beliefs.values()),
        k=5,
        verbose=True
    )
    
    print(f"\nEstimated confidence: {conf:.2f}")
    print(f"Uncertainty: {uncertainty:.2f}")
    print(f"Interpretation: ", end="")
    
    if uncertainty < 0.3:
        print("HIGH CONFIDENCE - neighbors agree strongly")
    elif uncertainty < 0.6:
        print("MODERATE CONFIDENCE - some disagreement")
    else:
        print("LOW CONFIDENCE - high variance, use cautiously")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTotal beliefs: {len(graph.beliefs)}")
    print(f"  - Manually initialized: 4")
    print(f"  - Auto-estimated: {len(graph.beliefs) - 4}")
    print(f"\nAverage confidence: {sum(b.confidence for b in graph.beliefs.values()) / len(graph.beliefs):.2f}")
    print(f"Justification links: {graph.nx_graph.number_of_edges()}")
    print("\n✓ Demonstration complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
