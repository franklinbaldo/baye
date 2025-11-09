"""
Example: Update-on-Use + Chat Context Retrieval

Demonstrates the full v2.0 system:
- Creating beliefs
- Updating from tool calls (Update-on-Use)
- Retrieving relevant context for chat
- Observability and metrics

Usage:
    python examples/example_uou_chat.py
"""

from baye import (
    create_belief_system,
    Belief,
    FeatureFlags,
)


def main():
    print("=" * 70)
    print("🧠 Baye v2.0: Update-on-Use + Chat Context Retrieval")
    print("=" * 70)
    print()

    # ========================================================================
    # Setup: Create belief system with all features enabled
    # ========================================================================

    print("📦 Initializing BeliefSystem...")
    system = create_belief_system(
        use_embeddings=False,  # Use Jaccard (no extra dependencies)
        enable_all_features=True
    )
    print("✓ System ready\n")

    # ========================================================================
    # Scenario: Agent learning about API reliability
    # ========================================================================

    print("📖 Scenario: Agent Learning About API Reliability")
    print("-" * 70)
    print()

    # Add initial beliefs
    print("Step 1: Adding initial beliefs...")
    b1 = Belief.from_confidence(
        content="Third-party APIs are generally reliable",
        confidence=0.6,
        context="api_reliability"
    )

    b2 = Belief.from_confidence(
        content="Always implement retry logic for API calls",
        confidence=0.7,
        context="best_practices"
    )

    b3 = Belief.from_confidence(
        content="Timeouts should be set to 30 seconds",
        confidence=0.5,
        context="configuration"
    )

    system.graph.beliefs[b1.id] = b1
    system.graph.beliefs[b2.id] = b2
    system.graph.beliefs[b3.id] = b3

    print(f"  ✓ Added belief: '{b1.content}' (conf: {b1.confidence:.2f})")
    print(f"  ✓ Added belief: '{b2.content}' (conf: {b2.confidence:.2f})")
    print(f"  ✓ Added belief: '{b3.content}' (conf: {b3.confidence:.2f})")
    print()

    # ========================================================================
    # Update-on-Use: Tool calls update beliefs
    # ========================================================================

    print("Step 2: Simulating tool calls (Update-on-Use)...")
    print()

    # Tool call 1: API succeeds
    print("  🔧 Tool: api_monitor")
    print("     Result: Stripe API responded in 120ms (success)")
    evidence1, update1 = system.update_from_tool_call(
        belief_id=b1.id,
        tool_result="Stripe API responded in 120ms with status 200",
        tool_name="api_monitor",
        sentiment=1.0,  # Supports reliability
        quality=0.9
    )

    print(f"     ✓ Updated belief '{b1.id}'")
    print(f"       Confidence: {update1['confidence_before']:.2f} → {update1['confidence_after']:.2f}")
    print(f"       Weight: {update1['weight']:.3f}")
    print(f"       Components: r={update1['components']['r']:.2f}, "
          f"n={update1['components']['n']:.2f}")
    print()

    # Tool call 2: API timeout
    print("  🔧 Tool: api_monitor")
    print("     Result: Payment API timed out after 30s")
    evidence2, update2 = system.update_from_tool_call(
        belief_id=b1.id,
        tool_result="Payment API timed out after 30 seconds",
        tool_name="api_monitor",
        sentiment=-1.0,  # Contradicts reliability
        quality=1.0
    )

    print(f"     ✓ Updated belief '{b1.id}'")
    print(f"       Confidence: {update2['confidence_before']:.2f} → {update2['confidence_after']:.2f}")
    print(f"       Weight: {update2['weight']:.3f}")
    print()

    # Tool call 3: Configuration check
    print("  🔧 Tool: config_validator")
    print("     Result: Timeout of 30s caused failure")
    evidence3, update3 = system.update_from_tool_call(
        belief_id=b3.id,
        tool_result="Timeout of 30 seconds caused request failure",
        tool_name="config_validator",
        sentiment=-1.0,  # Contradicts 30s timeout belief
        quality=0.8
    )

    print(f"     ✓ Updated belief '{b3.id}'")
    print(f"       Confidence: {update3['confidence_before']:.2f} → {update3['confidence_after']:.2f}")
    print()

    # Tool call 4: Duplicate evidence (idempotency test)
    print("  🔧 Tool: api_monitor")
    print("     Result: Payment API timed out after 30s (DUPLICATE)")
    evidence4, update4 = system.update_from_tool_call(
        belief_id=b1.id,
        tool_result="Payment API timed out after 30 seconds",  # SAME as evidence2
        tool_name="api_monitor",
        sentiment=-1.0
    )

    if update4['was_duplicate']:
        print(f"     ⚠  Duplicate detected! No update applied.")
        print(f"       Confidence unchanged: {update4['confidence_after']:.2f}")
    print()

    # ========================================================================
    # Retrieval: Get context for chat
    # ========================================================================

    print("Step 3: Retrieving context for chat prompt...")
    print()

    prompt_en = "How should I configure API timeouts to avoid failures?"
    print(f"  💬 User prompt: '{prompt_en}'")
    print()

    context_en = system.retrieve_context_for_prompt(
        prompt=prompt_en,
        k=3,
        token_budget=800,
        language='en'
    )

    print("  📄 Retrieved Context (English):")
    print()
    print(context_en)
    print()

    # Test Portuguese (i18n)
    prompt_pt = "Como devo configurar timeouts de API para evitar falhas?"
    print(f"  💬 User prompt (PT): '{prompt_pt}'")
    print()

    context_pt = system.retrieve_context_for_prompt(
        prompt=prompt_pt,
        k=3,
        token_budget=800
    )

    print("  📄 Retrieved Context (Portuguese):")
    print()
    print(context_pt)
    print()

    # ========================================================================
    # Observability: Metrics and audit trail
    # ========================================================================

    print("Step 4: Observability & Metrics...")
    print()

    dashboard = system.get_dashboard_data()

    print("  📊 Metrics:")
    metrics = dashboard['metrics']
    print(f"     Total updates: {metrics['total_updates']}")
    print(f"     Duplicate rate: {metrics['duplicate_rate']:.1%}")
    print(f"     Avg confidence delta: {metrics['avg_confidence_delta']:.3f}")
    print(f"     Update latency (P95): {metrics['update_latency']['p95']:.1f}ms")
    print()

    print("  📝 Audit Stats:")
    audit = dashboard['audit_stats']
    print(f"     Total log entries: {audit['total_entries']}")
    print(f"     Unique beliefs: {audit['unique_beliefs']}")
    print(f"     Duplicates: {audit['duplicates']}")
    print()

    # Export audit trail
    print("  💾 Exporting audit trail...")
    system.export_audit_trail('audit_trail.json', format='json')
    print("     ✓ Saved to audit_trail.json")
    print()

    # ========================================================================
    # Watchers: Threshold alerts
    # ========================================================================

    print("Step 5: Watcher System (Threshold Alerts)...")
    print()

    # Check for triggered watchers
    recent_events = system.watcher_system.get_recent_events(limit=10)

    if recent_events:
        print(f"  🚨 {len(recent_events)} watcher event(s) triggered:")
        for event in recent_events:
            print(f"     • {event.watcher_name}: Belief {event.belief_id}")
            print(f"       Confidence: {event.old_confidence:.2f} → {event.new_confidence:.2f}")
            print(f"       Action: {event.action.value}")
        print()
    else:
        print("  ℹ️  No watcher events triggered")
        print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  • Update-on-Use automatically updates beliefs from tool results")
    print("  • Duplicate detection prevents redundant updates (idempotency)")
    print("  • Context retrieval provides relevant beliefs for chat")
    print("  • Multi-language support (auto-detected from prompt)")
    print("  • Full observability with metrics and audit trail")
    print("  • Watchers trigger actions at confidence thresholds")
    print()
    print("Next steps:")
    print("  • Integrate with your agent's tool execution loop")
    print("  • Use `retrieve_context_for_prompt()` before LLM calls")
    print("  • Monitor dashboard data for system health")
    print("  • Configure reliability catalog for your tools")
    print()


if __name__ == "__main__":
    main()
