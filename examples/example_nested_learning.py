"""
Nested Learning Integration Example

Demonstrates the complete 3-level nested optimization architecture:
- Level 1: Immediate belief updates
- Level 2: Learned propagation weights (Deep Optimizers)
- Level 3: Meta-learning of hyperparameters

Plus:
- Continuum memory (online + offline consolidation)
- Self-modifying beliefs

Based on: "Nested Learning: The Illusion of Deep Learning Architectures"
         (Behrouz et al., NeurIPS 2025)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baye import JustificationGraph, Belief
from baye.nested_learning import NestedBeliefGraph


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


async def main():
    print("\n🧠 NESTED LEARNING FOR BELIEF TRACKING")
    print("=" * 70)
    print("Demonstrating 3-level nested optimization + continuum memory")
    print("=" * 70)

    # =========================================================================
    # SETUP: Create base graph
    # =========================================================================
    print_section("SETUP: Initializing Nested Belief Graph")

    base_graph = JustificationGraph(max_depth=10)

    # Add initial beliefs across different domains
    security_beliefs = []
    performance_beliefs = []

    print("Adding initial beliefs...")

    # Security domain
    sec1 = base_graph.add_belief(
        content="Always validate user input to prevent injection attacks",
        confidence=0.8,
        context="security"
    )
    security_beliefs.append(sec1)

    sec2 = base_graph.add_belief(
        content="Use parameterized queries for database access",
        confidence=0.75,
        context="security"
    )
    security_beliefs.append(sec2)

    # Link them
    base_graph.link_beliefs(sec1.id, sec2.id, relation="supports")

    # Performance domain
    perf1 = base_graph.add_belief(
        content="Caching can reduce API response times",
        confidence=0.7,
        context="performance"
    )
    performance_beliefs.append(perf1)

    perf2 = base_graph.add_belief(
        content="Database query optimization improves throughput",
        confidence=0.65,
        context="performance"
    )
    performance_beliefs.append(perf2)

    print(f"✓ Added {len(security_beliefs)} security beliefs")
    print(f"✓ Added {len(performance_beliefs)} performance beliefs")

    # Create nested belief graph
    nested_graph = NestedBeliefGraph(
        base_graph=base_graph,
        enable_all_features=True
    )

    print("\n✓ Nested belief graph initialized with:")
    print("  - Level 1: Belief updates")
    print("  - Level 2: Propagation memory (Deep Optimizers)")
    print("  - Level 3: Meta-learner")
    print("  - Continuum memory (online + offline)")
    print("  - Self-modifying beliefs")

    # Start background consolidation
    consolidation_task = await nested_graph.start_background_consolidation()
    print("\n✓ Background consolidation started (every 60s)")

    # =========================================================================
    # EXAMPLE 1: Security Belief Update with Learned Weights
    # =========================================================================
    print_section("EXAMPLE 1: Security Belief Update (Domain-Specific Learning)")

    print("Scenario: XSS vulnerability discovered in production")
    print("Signal: High confidence (0.95) that input validation is critical\n")

    result1 = await nested_graph.update_belief_nested(
        belief_id=sec1.id,
        signal=0.95,
        context="security",
        r=1.0,  # Reliable source
        n=0.8,  # Somewhat novel
        q=1.0,  # High quality evidence
        enable_self_modification=True
    )

    print(f"Belief Updated: \"{sec1.content[:50]}...\"")
    print(f"  Old confidence: {result1['old_confidence']:.3f}")
    print(f"  New confidence: {result1['new_confidence']:.3f}")
    print(f"  Delta: {result1['delta']:+.3f}")
    print(f"\nLearned Propagation Weights (Level 2):")
    alpha, beta = result1['learned_weights']
    print(f"  α (causal): {alpha:.3f}")
    print(f"  β (semantic): {beta:.3f}")
    print(f"\nPropagation Outcome:")
    print(f"  Beliefs updated: {result1['propagation_result'].total_beliefs_updated}")
    print(f"  Surprise score: {result1['propagation_surprise']:.3f}")

    # =========================================================================
    # EXAMPLE 2: Performance Belief with Different Learned Weights
    # =========================================================================
    print_section("EXAMPLE 2: Performance Belief Update (Different Domain)")

    print("Scenario: Caching experiment showed 2x speedup")
    print("Signal: Moderate confidence (0.75)\n")

    result2 = await nested_graph.update_belief_nested(
        belief_id=perf1.id,
        signal=0.75,
        context="performance",
        r=0.9,
        n=0.6,
        q=0.8,
        enable_self_modification=True
    )

    print(f"Belief Updated: \"{perf1.content[:50]}...\"")
    print(f"  Old confidence: {result2['old_confidence']:.3f}")
    print(f"  New confidence: {result2['new_confidence']:.3f}")
    print(f"  Delta: {result2['delta']:+.3f}")
    print(f"\nLearned Propagation Weights (Level 2):")
    alpha2, beta2 = result2['learned_weights']
    print(f"  α (causal): {alpha2:.3f}")
    print(f"  β (semantic): {beta2:.3f}")

    print(f"\n💡 Notice: Different domains learn different weights!")
    print(f"   Security: α={alpha:.3f}, β={beta:.3f}")
    print(f"   Performance: α={alpha2:.3f}, β={beta2:.3f}")

    # =========================================================================
    # EXAMPLE 3: Multiple Updates to Demonstrate Learning
    # =========================================================================
    print_section("EXAMPLE 3: Repeated Updates (Weights Improve Over Time)")

    print("Running 10 updates on security beliefs...")
    print("Watching how propagation weights adapt...\n")

    import random

    weight_evolution = []

    for i in range(10):
        # Random security belief
        belief_id = random.choice(security_beliefs).id

        # Random signal (mostly positive for security)
        signal = random.uniform(0.7, 0.95)

        result = await nested_graph.update_belief_nested(
            belief_id=belief_id,
            signal=signal,
            context="security",
            r=random.uniform(0.8, 1.0),
            n=random.uniform(0.5, 0.9),
            q=random.uniform(0.8, 1.0),
            enable_self_modification=True
        )

        alpha, beta = result['learned_weights']
        weight_evolution.append((alpha, beta, result['propagation_surprise']))

        if i % 3 == 0:
            print(f"  Update {i+1}: α={alpha:.3f}, β={beta:.3f}, surprise={result['propagation_surprise']:.3f}")

    print("\n✓ Updates complete. Weights adapted based on propagation outcomes.")

    # Show weight convergence
    print("\nWeight Evolution:")
    print("  Initial: α={:.3f}, β={:.3f}".format(weight_evolution[0][0], weight_evolution[0][1]))
    print("  Final:   α={:.3f}, β={:.3f}".format(weight_evolution[-1][0], weight_evolution[-1][1]))
    print(f"  Avg surprise: {sum(s for _, _, s in weight_evolution) / len(weight_evolution):.3f}")

    # =========================================================================
    # EXAMPLE 4: Self-Modifying Beliefs
    # =========================================================================
    print_section("EXAMPLE 4: Self-Modifying Beliefs (Learn Own Update Rules)")

    print("Adding a new self-modifying belief...")

    sec3 = base_graph.add_belief(
        content="Encryption should use AES-256 or stronger",
        confidence=0.6,
        context="security"
    )

    print(f"Created: \"{sec3.content}\"")
    print(f"Initial confidence: {sec3.confidence:.3f}\n")

    print("Updating 5 times with varying signals...")
    print("The belief will learn its own update strategy!\n")

    signals = [0.85, 0.9, 0.75, 0.95, 0.88]
    for i, sig in enumerate(signals):
        result = await nested_graph.update_belief_nested(
            belief_id=sec3.id,
            signal=sig,
            context="security",
            r=0.95,
            n=0.7,
            q=0.9,
            enable_self_modification=True
        )

        belief = nested_graph.graph.beliefs[sec3.id]
        print(f"  Update {i+1}: signal={sig:.2f} → confidence={belief.confidence:.3f}")

    # Get self-modifying stats
    if sec3.id in nested_graph.self_modifying_beliefs:
        smb = nested_graph.self_modifying_beliefs[sec3.id]
        stats = smb.get_strategy_stats()

        print(f"\nLearned Update Strategy:")
        print(f"  Signal amplification: {stats['params']['signal_amplification']:.3f}")
        print(f"  Conservatism bias: {stats['params']['conservatism_bias']:+.3f}")
        print(f"  Avg loss: {stats['avg_loss']:.4f}")

    # =========================================================================
    # EXAMPLE 5: Meta-Learning (Level 3)
    # =========================================================================
    print_section("EXAMPLE 5: Meta-Learning (Learning to Learn)")

    print("After 100+ updates, meta-learner optimizes hyperparameters...")
    print("Running additional updates to trigger meta-learning...\n")

    # Run enough updates to trigger meta-learning
    for _ in range(105 - nested_graph.update_count):
        belief_id = random.choice(security_beliefs + performance_beliefs).id
        belief = nested_graph.graph.beliefs[belief_id]
        context = belief.context

        await nested_graph.update_belief_nested(
            belief_id=belief_id,
            signal=random.uniform(0.6, 0.9),
            context=context,
            enable_self_modification=False  # Skip self-mod for speed
        )

    meta_stats = nested_graph.meta_learner.get_meta_statistics()

    print(f"✓ Meta-learning triggered at {nested_graph.update_count} updates")
    print(f"\nLearned Domain-Specific Hyperparameters:")

    for domain, params in meta_stats.get('hyperparameters', {}).items():
        print(f"\n  {domain.upper()}:")
        print(f"    Optimal α: {params.get('alpha_init', 0):.3f}")
        print(f"    Optimal β: {params.get('beta_init', 0):.3f}")
        print(f"    Learning rate: {params.get('learning_rate', 0):.4f}")
        print(f"    Avg surprise: {params.get('avg_surprise', 0):.3f}")

    # =========================================================================
    # EXAMPLE 6: Continuum Memory Status
    # =========================================================================
    print_section("EXAMPLE 6: Continuum Memory (Online + Offline)")

    consolidation_status = nested_graph.continuum.get_consolidation_status()

    print("Memory System Status:")
    print(f"  Currently consolidating: {consolidation_status['is_consolidating']}")
    print(f"  Updates in queue: {consolidation_status['queue_size']}")
    print(f"  Next consolidation in: {consolidation_status['next_consolidation_in_seconds']:.1f}s")

    stats = consolidation_status['statistics']
    print(f"\nConsolidation Statistics:")
    print(f"  Total consolidations: {stats['total_consolidations']}")
    if stats['total_consolidations'] > 0:
        print(f"  Avg updates/consolidation: {stats['avg_updates_per_consolidation']:.1f}")
        print(f"  Avg beliefs strengthened: {stats['avg_beliefs_strengthened']:.1f}")
        print(f"  Avg beliefs pruned: {stats['avg_beliefs_pruned']:.1f}")

    # =========================================================================
    # FINAL STATISTICS
    # =========================================================================
    print_section("FINAL STATISTICS: Complete Nested Architecture")

    nested_stats = nested_graph.get_nested_statistics()

    print("Level 1 (Belief Updates):")
    print(f"  Total updates: {nested_stats['level_1_updates']}")

    print("\nLevel 2 (Propagation Memory - Deep Optimizers):")
    level2 = nested_stats['level_2_propagation']
    print(f"  Current α: {level2.get('current_alpha', 0):.3f}")
    print(f"  Current β: {level2.get('current_beta', 0):.3f}")
    print(f"  Memory size: {level2.get('memory_size', 0)}")
    print(f"  Domains learned: {level2.get('domains_learned', [])}")

    print("\nLevel 3 (Meta-Learning):")
    level3 = nested_stats['level_3_meta']
    print(f"  Domains optimized: {len(level3.get('domains_learned', []))}")
    print(f"  Episodes processed: {level3.get('episodes_processed', 0)}")

    print("\nSelf-Modifying Beliefs:")
    print(f"  Total: {nested_stats['self_modifying_beliefs']}")

    print("\nContinuum Memory:")
    cm = nested_stats['continuum_memory']
    print(f"  Queue size: {cm['queue_size']}")
    print(f"  Total consolidations: {cm['statistics']['total_consolidations']}")

    # =========================================================================
    # EXPORT LEARNED PARAMETERS
    # =========================================================================
    print_section("EXPORT: Saving Learned Parameters")

    learned_params = nested_graph.export_learned_parameters()

    print("Exporting learned parameters for future sessions:")
    print(f"  Propagation weights: {len(learned_params['propagation_weights'])} domains")
    print(f"  Meta-hyperparameters: {len(learned_params['meta_hyperparameters'].get('domains_learned', []))} domains")
    print(f"  Self-modifying strategies: {len(learned_params['self_modifying_strategies'])} beliefs")

    print("\n✓ Parameters can be imported in next session with:")
    print("  nested_graph.import_learned_parameters(params)")

    # =========================================================================
    # CONCLUSION
    # =========================================================================
    print_section("CONCLUSION")

    print("🎉 Nested Learning Integration Complete!")
    print("\nKey Achievements:")
    print("  ✓ 3-level nested optimization working")
    print("  ✓ Deep optimizers learning domain-specific propagation weights")
    print("  ✓ Continuum memory with online + offline consolidation")
    print("  ✓ Self-modifying beliefs learning their own update rules")
    print("  ✓ Meta-learner optimizing hyperparameters across domains")
    print("\nTheoretical Foundation:")
    print("  Based on 'Nested Learning' (Behrouz et al., NeurIPS 2025)")
    print("  Beliefs are associative memories compressing context flow")
    print("  Each level learns how to learn at the level below")
    print("\nPractical Benefits:")
    print("  • No manual hyperparameter tuning (α, β learned)")
    print("  • Domain-specific adaptation (security ≠ performance)")
    print("  • Better long-term memory (offline consolidation)")
    print("  • Interpretable learning (full audit trail)")

    print("\n" + "=" * 70)
    print("  Ready for deployment in autonomous agents!")
    print("=" * 70 + "\n")

    # Cancel background task
    consolidation_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
