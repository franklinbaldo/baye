#!/usr/bin/env python3
"""
Test ChromaDB persistence
"""

import os
import shutil

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.fact_store import FactStore, InputMode

def test_persistence():
    print("=" * 80)
    print("PERSISTENCE TEST")
    print("=" * 80)

    # Clean up any existing test data
    test_dir = ".baye_test_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Session 1: Create facts
    print("\n[Session 1] Creating facts...")
    print("-" * 80)

    store1 = FactStore(persist_directory=test_dir)
    print(f"Initial facts count: {len(store1.facts)}")

    # Add some facts
    facts1 = store1.add_context(
        content="Trump é presidente dos EUA em 2025",
        input_mode=InputMode.USER_INPUT,
        author_uuid=store1.user_uuid,
        source_context_id="test_session_1",
        confidence=0.8
    )

    facts2 = store1.add_context(
        content="Estamos em 2025",
        input_mode=InputMode.USER_INPUT,
        author_uuid=store1.user_uuid,
        source_context_id="test_session_1",
        confidence=0.9
    )

    print(f"Created {len(facts1) + len(facts2)} facts")
    for fact in facts1 + facts2:
        print(f"  - Fact #{fact.seq_id}: {fact.content}")

    # Session 2: Load facts from persistence
    print("\n[Session 2] Loading facts from persistence...")
    print("-" * 80)

    store2 = FactStore(persist_directory=test_dir)
    print(f"Loaded facts count: {len(store2.facts)}")

    for fact in store2.facts.values():
        print(f"  - Fact #{fact.seq_id}: {fact.content}")

    # Test semantic search
    print("\n[Test] Searching for contradictions with 'Joe Biden é presidente'...")
    print("-" * 80)

    contradictions = store2.find_contradicting(
        content="Joe Biden é o presidente dos EUA",
        k=3,
        include_beliefs=False
    )

    if contradictions:
        print(f"Found {len(contradictions)} potential contradictions:")
        for fact_type, fact_id, confidence, content in contradictions:
            print(f"  - {fact_type} ({confidence:.2f}): {content}")
    else:
        print("  ✗ No contradictions found")

    # Test that sequential IDs continue correctly
    print("\n[Test] Adding new fact to verify seq_id continues...")
    print("-" * 80)

    facts3 = store2.add_context(
        content="Lula é presidente do Brasil",
        input_mode=InputMode.USER_INPUT,
        author_uuid=store2.user_uuid,
        source_context_id="test_session_2",
        confidence=0.85
    )

    print(f"New fact seq_id: {facts3[0].seq_id} (expected: {len(facts1) + len(facts2) + 1})")

    # Session 3: Verify new fact was persisted
    print("\n[Session 3] Verify new fact persisted...")
    print("-" * 80)

    store3 = FactStore(persist_directory=test_dir)
    print(f"Total facts: {len(store3.facts)}")

    for fact in sorted(store3.facts.values(), key=lambda f: f.seq_id):
        print(f"  - Fact #{fact.seq_id}: {fact.content}")

    print("\n" + "=" * 80)
    print("✓ PERSISTENCE TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_persistence()
