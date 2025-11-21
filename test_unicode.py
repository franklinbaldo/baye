#!/usr/bin/env python3
"""
Test Unicode support in claims
"""

import asyncio
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession

async def test_unicode():
    print("=" * 80)
    print("UNICODE TEST: Emoji and Special Characters")
    print("=" * 80)

    session = ChatSession(mode="claim-based")

    tests = [
        ("Is there a seahorse emoji?", "🦭"),
        ("What is the heart emoji?", "♥"),
        ("What is the value of pi?", "π"),
        ("Show me a checkmark", "✓"),
    ]

    for question, expected_unicode in tests:
        print(f"\n[Test] User: '{question}'")
        print(f"Expected to see: {expected_unicode}")
        print("-" * 80)

        try:
            response = await session.process_message(question)
            print(f"Response: {response.response_text}")
            print(f"Score: {session.score:.2f}pts")

            if expected_unicode in response.response_text:
                print(f"✓ Found Unicode character: {expected_unicode}")
            else:
                print(f"✗ Missing Unicode character: {expected_unicode}")

        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n{'='*80}")
    print(f"FINAL SCORE: {session.score:.2f}pts")
    print(f"Successes: {session.successful_claims} | Failures: {session.failed_claims}")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(test_unicode())
