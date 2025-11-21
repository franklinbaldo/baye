#!/usr/bin/env python3
"""
Test granular claims and multiple tool calls
"""

import asyncio
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession
from rich.console import Console

console = Console()

async def test_granular_claims():
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]GRANULAR CLAIMS & MULTIPLE TOOLS TEST[/bold cyan]")
    console.print("=" * 80 + "\n")

    session = ChatSession(mode="claim-based")

    tests = [
        # Test 1: Simple question that should generate multiple claims
        ("Multiple claims", "Tell me about Python programming language"),

        # Test 2: Question requiring multiple tool calls
        ("Multiple tools", "Use Python to calculate 5!, 6!, and 7!"),

        # Test 3: Complex question with tools and multiple claims
        ("Tools + claims", "Calculate the square of 12 using Python, then tell me three facts about the number 144"),
    ]

    for test_name, question in tests:
        console.print(f"\n[bold yellow]Test:[/bold yellow] {test_name}")
        console.print(f"[dim]User: {question}[/dim]")
        console.print("-" * 80)

        try:
            response = await session.process_message(question)

            # Count claims in response
            # We need to inspect the actual response to see claims
            # For now, just show the response
            console.print(f"[green]✓[/green] {response.response_text}")
            console.print(f"[dim]Score: {session.score:.2f}pts[/dim]")

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            import traceback
            traceback.print_exc()

    console.print(f"\n{'='*80}")
    console.print(f"[bold]FINAL SCORE:[/bold] {session.score:.2f}pts")
    console.print(f"[bold]Total facts:[/bold] {len(session.fact_store.facts)}")
    console.print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_granular_claims())
