#!/usr/bin/env python3
"""
Test epistemic humility - LLM should not be overconfident without memories
"""

import asyncio
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession

async def test():
    session = ChatSession(mode="claim-based")

    # Question about something LLM "knows" but has no memories about
    response = await session.process_message("quem é presidente dos eua?")

    print(f"Response: {response.response_text}")
    print(f"Score: {session.score:.2f}pts")

asyncio.run(test())
