#!/usr/bin/env python3
"""
Test date question that triggers Python tool
"""

import asyncio
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession

async def test():
    session = ChatSession(mode="claim-based")

    print("Testing: 'que dia é hoje?'")
    print("=" * 80)

    response = await session.process_message("que dia é hoje?")

    print(f"\nResponse: {response.response_text}")
    print(f"Score: {session.score:.2f}pts")
    print(f"\nFacts in store: {len(session.fact_store.facts)}")

    # Check for tool results in facts
    tool_facts = [
        f for f in session.fact_store.facts.values()
        if f.input_mode.value == "tool_return"
    ]

    if tool_facts:
        print(f"\nTool facts found: {len(tool_facts)}")
        for fact in tool_facts[-3:]:  # Show last 3
            print(f"  - {fact.content[:100]}")

asyncio.run(test())
