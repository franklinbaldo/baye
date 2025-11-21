#!/usr/bin/env python3
"""
Test fact search to debug why Trump fact isn't being found
"""

import asyncio
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession

async def test_flow():
    print("=" * 80)
    print("FACT SEARCH DEBUG")
    print("=" * 80)

    # Create session
    session = ChatSession(mode="claim-based")

    # Add user fact: Trump is president
    print("\n[Step 1] Adding user fact: 'Trump é presidente dos EUA'")
    print("-" * 80)

    await session._extract_user_facts("Trump é presidente dos EUA")

    print(f"Facts in store: {len(session.fact_store.facts)}")
    for fact_id, fact in session.fact_store.facts.items():
        print(f"  Fact #{fact.seq_id}: {fact.content}")

    # Now search for contradictions with "Joe Biden é presidente"
    print("\n[Step 2] Searching for contradictions with 'Joe Biden é o presidente dos EUA'")
    print("-" * 80)

    contradictions = session.fact_store.find_contradicting(
        content="Joe Biden é o presidente dos EUA",
        k=3,
        include_beliefs=True,
        belief_graph=session.tracker.graph
    )

    if contradictions:
        print(f"Found {len(contradictions)} contradictions:")
        for fact_type, fact_id, confidence, content in contradictions:
            print(f"  - {fact_type} ({confidence:.2f}): {content[:60]}...")
    else:
        print("  ✗ No contradictions found!")

    # Try with more generic search
    print("\n[Step 3] Searching for contradictions with 'O presidente dos EUA é Biden'")
    print("-" * 80)

    contradictions = session.fact_store.find_contradicting(
        content="O presidente dos EUA é Biden",
        k=3,
        include_beliefs=True,
        belief_graph=session.tracker.graph
    )

    if contradictions:
        print(f"Found {len(contradictions)} contradictions:")
        for fact_type, fact_id, confidence, content in contradictions:
            print(f"  - {fact_type} ({confidence:.2f}): {content[:60]}...")
    else:
        print("  ✗ No contradictions found!")

if __name__ == "__main__":
    asyncio.run(test_flow())
