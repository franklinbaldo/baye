#!/usr/bin/env python3
"""
Test acknowledgment with verbose output
"""

import asyncio
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession

async def test_flow():
    print("=" * 80)
    print("VERBOSE TEST: Acknowledgment Format")
    print("=" * 80)

    # Create session
    session = ChatSession(mode="claim-based")

    # Test 1: User asserts Trump is president
    print("\n[Test 1] User: 'Trump é presidente dos EUA'")
    print("-" * 80)

    response = await session.process_message("Trump é presidente dos EUA")
    print(f"Response: {response.response_text}")
    print(f"Score: {session.score:.2f}pts")

    # Show what facts were saved
    print(f"\nFacts in store: {len(session.fact_store.facts)}")
    for fact_id, fact in list(session.fact_store.facts.items())[:5]:
        print(f"  - Fact #{fact.seq_id}: {fact.content[:60]}...")

    # Test 2: Ask about president
    print("\n[Test 2] User: 'Quem é presidente dos EUA?'")
    print("-" * 80)

    response = await session.process_message("Quem é presidente dos EUA?")
    print(f"Response: {response.response_text}")
    print(f"Score: {session.score:.2f}pts")

    # Check if response mentions Trump
    if "Trump" in response.response_text:
        print("  ✓ LLM mentioned Trump")
    else:
        print("  ✗ LLM did NOT mention Trump")

    # Check if response uses acknowledgment format
    if "Apesar do usuário" in response.response_text:
        print("  ✓ LLM used acknowledgment format")
    else:
        print("  ✗ LLM did NOT use acknowledgment format")

    print("\n" + "=" * 80)
    print(f"FINAL SCORE: {session.score:.2f}pts")
    print(f"Successes: {session.successful_claims} | Failures: {session.failed_claims}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_flow())
