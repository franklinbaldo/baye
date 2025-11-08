"""
Example: Using PydanticAI + Gemini for intelligent belief tracking.

This demonstrates the V1.5 features:
- Automatic relationship detection via LLM
- Conflict resolution with nuanced beliefs
- Semantic embeddings for similarity

Run with:
    export GOOGLE_API_KEY="your-gemini-key"
    uv run python example_llm_integration.py
"""

import asyncio
import os
from belief_types import Belief
from llm_agents import (
    detect_relationship,
    resolve_conflict,
    find_related_beliefs,
    check_gemini_api_key,
)


async def main():
    """Demonstrate LLM-powered belief tracking."""

    # Check API key
    try:
        check_gemini_api_key()
    except ValueError as e:
        print(f"❌ {e}")
        print("\nSet your API key:")
        print("  export GOOGLE_API_KEY='your-key'")
        print("  # or")
        print("  export GEMINI_API_KEY='your-key'")
        return

    print("🧠 Belief Tracking with PydanticAI + Gemini\n")
    print("=" * 70)

    # ========================================================================
    # Scenario: Learning from a Stripe API failure
    # ========================================================================

    print("\n📖 Scenario: Stripe API Failure\n")

    # Initial beliefs
    b1 = Belief(
        content="Third-party payment services are generally reliable",
        confidence=0.7,
        context="infrastructure",
        source_task="initial_knowledge"
    )

    b2 = Belief(
        content="Always validate and handle API responses gracefully",
        confidence=0.6,
        context="best_practices",
        source_task="coding_standards"
    )

    b3 = Belief(
        content="Established services like Stripe don't need defensive programming",
        confidence=0.4,
        context="development_practices",
        source_task="initial_knowledge"
    )

    print(f"Initial beliefs:")
    print(f"  B1: {b1.content} (conf: {b1.confidence})")
    print(f"  B2: {b2.content} (conf: {b2.confidence})")
    print(f"  B3: {b3.content} (conf: {b3.confidence})")

    # New lesson from failure
    lesson = Belief(
        content="Stripe API returned 500 errors during checkout flow",
        confidence=0.9,
        context="incident_response",
        source_task="production_failure"
    )

    print(f"\n💥 Incident: {lesson.content}\n")

    # ========================================================================
    # Step 1: Automatic Relationship Detection
    # ========================================================================

    print("🔍 Step 1: Detecting relationships with existing beliefs...\n")

    relationships = await find_related_beliefs(
        new_belief=lesson,
        existing_beliefs=[b1, b2, b3],
        min_confidence=0.6
    )

    for belief_id, rel_type, conf in relationships:
        belief = next((b for b in [b1, b2, b3] if b.id == belief_id), None)
        if belief:
            print(f"  • {rel_type.value.upper()} B{[b1, b2, b3].index(belief) + 1}")
            print(f"    Confidence: {conf:.2f}")
            print(f"    → {belief.content[:60]}...")
            print()

    # ========================================================================
    # Step 2: Detailed Relationship Analysis
    # ========================================================================

    print("🔬 Step 2: Analyzing relationship with B1...\n")

    analysis = await detect_relationship(lesson, b1)

    print(f"  Relationship: {analysis.relationship.upper()}")
    print(f"  Confidence: {analysis.confidence:.2f}")
    print(f"  Explanation: {analysis.explanation}")

    # ========================================================================
    # Step 3: Conflict Resolution
    # ========================================================================

    print("\n🤝 Step 3: Resolving contradiction between lesson and B1...\n")

    resolution = await resolve_conflict(
        belief1=lesson,
        belief2=b1,
        context="Production incident affecting revenue"
    )

    print(f"  Resolved Belief:")
    print(f"    \"{resolution.resolved_belief}\"")
    print(f"\n  Confidence: {resolution.confidence:.2f}")
    print(f"  Reasoning: {resolution.reasoning}")
    print(f"\n  Supports lesson: {resolution.supports_first}")
    print(f"  Supports original: {resolution.supports_second}")

    # ========================================================================
    # Step 4: Create Nuanced Belief
    # ========================================================================

    print("\n✨ Step 4: Creating nuanced belief from resolution...\n")

    nuanced_belief = Belief(
        content=resolution.resolved_belief,
        confidence=resolution.confidence,
        context="learned_wisdom",
        source_task="conflict_resolution"
    )

    print(f"  New Belief ID: {nuanced_belief.id}")
    print(f"  Content: {nuanced_belief.content}")
    print(f"  Confidence: {nuanced_belief.confidence:.2f}")
    print(f"  Context: {nuanced_belief.context}")

    # ========================================================================
    # Step 5: Another Example - Support Relationship
    # ========================================================================

    print("\n🔗 Step 5: Analyzing support relationship...\n")

    b4 = Belief(
        content="Network calls can fail at any time",
        confidence=0.8,
        context="distributed_systems",
        source_task="systems_knowledge"
    )

    analysis2 = await detect_relationship(lesson, b4)

    print(f"  Lesson: {lesson.content}")
    print(f"  Related: {b4.content}")
    print(f"\n  Relationship: {analysis2.relationship.upper()}")
    print(f"  Confidence: {analysis2.confidence:.2f}")
    print(f"  Explanation: {analysis2.explanation}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("\n✅ Demo Complete!")
    print("\nKey Takeaways:")
    print("  • LLM automatically detected contradictions and supports")
    print("  • Generated nuanced resolution instead of binary choice")
    print("  • Confidence scores guide propagation strength")
    print("  • Context-aware analysis considers incident severity")
    print("\n📚 This enables V1.5 roadmap features:")
    print("  ✓ Automatic relationship discovery")
    print("  ✓ Conflict resolution via LLM")
    print("  ✓ Context-sensitive reasoning")
    print("  • Semantic embeddings (next step)")
    print("  • Bidirectional propagation (next step)")


if __name__ == "__main__":
    asyncio.run(main())
