#!/usr/bin/env python3
"""
Automated test for claim-based mode with user facts
"""

import asyncio
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession

async def test_flow():
    print("=" * 80)
    print("AUTOMATED TEST: User Facts → LLM Validation")
    print("=" * 80)

    # Create session
    session = ChatSession(mode="claim-based")

    # Test 1: User asserts a fact
    print("\n[Test 1] User: 'Estamos em 2025'")
    print("-" * 80)

    try:
        response = await session.process_message("Estamos em 2025")
        print(f"✓ Response: {response.response_text[:100]}...")
        print(f"  Score: {session.score:.2f}pts")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Ask about year (should use user's fact)
    print("\n[Test 2] User: 'Em que ano estamos?'")
    print("-" * 80)

    try:
        response = await session.process_message("Em que ano estamos?")
        print(f"✓ Response: {response.response_text[:100]}...")
        print(f"  Score: {session.score:.2f}pts")

        # Check if LLM mentioned 2025
        if "2025" in response.response_text:
            print("  ✓ LLM correctly used user's fact (2025)")
        else:
            print("  ✗ LLM did not use user's fact")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: User asserts president
    print("\n[Test 3] User: 'Trump é presidente dos EUA'")
    print("-" * 80)

    try:
        response = await session.process_message("Trump é presidente dos EUA")
        print(f"✓ Response: {response.response_text[:100]}...")
        print(f"  Score: {session.score:.2f}pts")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 4: Ask about president (should use user's fact)
    print("\n[Test 4] User: 'Quem é presidente dos EUA?'")
    print("-" * 80)

    try:
        response = await session.process_message("Quem é presidente dos EUA?")
        print(f"✓ Response: {response.response_text[:100]}...")
        print(f"  Score: {session.score:.2f}pts")

        # Check if LLM mentioned Trump
        if "Trump" in response.response_text:
            print("  ✓ LLM correctly used user's fact (Trump)")
        else:
            print("  ✗ LLM did not use user's fact")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print(f"FINAL SCORE: {session.score:.2f}pts")
    print(f"Successes: {session.successful_claims} | Failures: {session.failed_claims}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_flow())
