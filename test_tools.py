#!/usr/bin/env python3
"""
Test tool system
"""

import asyncio
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession

async def test_tools():
    print("=" * 80)
    print("TOOL SYSTEM TEST")
    print("=" * 80)

    session = ChatSession(mode="claim-based")

    tests = [
        "What is 17 * 23? Use Python to calculate it.",
        "I like coffee. Can you remember that?",
        "What did I just say I like? Use query_facts to check.",
    ]

    for question in tests:
        print(f"\n[Test] User: '{question}'")
        print("-" * 80)

        try:
            # Count facts before
            facts_before = len(session.fact_store.facts)

            response = await session.process_message(question)

            # Count facts after (tool results add facts)
            facts_after = len(session.fact_store.facts)
            facts_added = facts_after - facts_before

            print(f"Response: {response.response_text}")
            print(f"Facts added by tools: {facts_added}")
            print(f"Score: {session.score:.2f}pts")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"FINAL SCORE: {session.score:.2f}pts")
    print(f"Facts in store: {len(session.fact_store.facts)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(test_tools())
