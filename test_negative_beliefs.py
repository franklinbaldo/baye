#!/usr/bin/env python3
"""
Test script to verify negative belief handling

This tests that the model correctly uses negative confidence values
to express disbelief while keeping the statement text constant.
"""

import asyncio
import os
from baye.chat_session import ChatSession
from baye.llm_agents import check_gemini_api_key

async def test_negative_belief():
    """Test that model uses negative values for disbelief"""

    print("=" * 80)
    print("TEST: Negative Belief Handling")
    print("=" * 80)
    print()

    # Ensure API key is set
    try:
        check_gemini_api_key()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nPlease set GOOGLE_API_KEY or GEMINI_API_KEY environment variable:")
        print("  export GOOGLE_API_KEY=\"your-key-here\"")
        return

    # Create session
    print("Creating chat session in legacy mode...")
    session = ChatSession(mode="legacy")
    print(f"✓ Session created")
    print()

    # Test 1: Ask about Obama being president
    print("-" * 80)
    print("Test 1: Ask if Barack Obama is president")
    print("-" * 80)

    response = await session.process_message(
        "Barack Obama é o presidente dos EUA?"
    )

    print(f"\n📝 Response text:")
    print(f"   {response.text}")
    print()

    print(f"📊 Belief metadata:")
    print(f"   Belief ID: {response.belief_id[:8]}...")
    print(f"   Palpite: {response.belief_value_guessed:.4f}")
    print(f"   Confidence real: {response.actual_confidence:.4f}")
    print(f"   Margem: ±{response.margin:.4f}")
    print()

    # Check if negative value was used
    if response.belief_value_guessed < 0:
        print("✅ CORRETO: Modelo usou valor NEGATIVO para expressar descrença")
        print(f"   (valor: {response.belief_value_guessed:.4f})")
    else:
        print("❌ INCORRETO: Modelo deveria usar valor NEGATIVO")
        print(f"   (valor usado: {response.belief_value_guessed:.4f})")
    print()

    # Test 2: Ask to use Obama statement with negative guess
    print("-" * 80)
    print("Test 2: Explicitly request negative confidence")
    print("-" * 80)

    response = await session.process_message(
        "Use a afirmação 'Barack Obama é presidente dos EUA' com um palpite negativo, "
        "indicando que você não acredita que seja verdade."
    )

    print(f"\n📝 Response text:")
    print(f"   {response.text}")
    print()

    print(f"📊 Belief metadata:")
    print(f"   Belief ID: {response.belief_id[:8]}...")
    print(f"   Palpite: {response.belief_value_guessed:.4f}")
    print(f"   Confidence real: {response.actual_confidence:.4f}")
    print()

    # Check statement text
    if "não" in response.text.lower() or "nao" in response.text.lower():
        print("⚠️  ATENÇÃO: Texto contém negação ('não')")
        print("   Idealmente deveria manter afirmação positiva com confiança negativa")
    else:
        print("✅ CORRETO: Texto mantido positivo")

    # Check if negative value was used
    if response.belief_value_guessed < 0:
        print("✅ CORRETO: Modelo usou valor NEGATIVO")
        print(f"   (valor: {response.belief_value_guessed:.4f})")
    else:
        print("❌ INCORRETO: Modelo deveria usar valor NEGATIVO")
        print(f"   (valor usado: {response.belief_value_guessed:.4f})")
    print()

    # Test 3: True statement with strong belief
    print("-" * 80)
    print("Test 3: Ask about current president (should use positive)")
    print("-" * 80)

    response = await session.process_message(
        "Quem é o presidente dos EUA em janeiro de 2025?"
    )

    print(f"\n📝 Response text:")
    print(f"   {response.text}")
    print()

    print(f"📊 Belief metadata:")
    print(f"   Belief ID: {response.belief_id[:8]}...")
    print(f"   Palpite: {response.belief_value_guessed:.4f}")
    print(f"   Confidence real: {response.actual_confidence:.4f}")
    print()

    if response.belief_value_guessed > 0:
        print("✅ CORRETO: Modelo usou valor POSITIVO para crença")
        print(f"   (valor: {response.belief_value_guessed:.4f})")
    else:
        print("⚠️  ATENÇÃO: Valor não é positivo")
        print(f"   (valor usado: {response.belief_value_guessed:.4f})")
    print()

    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_negative_belief())
