#!/usr/bin/env python3
"""
Comprehensive test for all tool types
"""

import asyncio
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession
from rich.console import Console

console = Console()

async def test_all_tools():
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]COMPREHENSIVE TOOL SYSTEM TEST[/bold cyan]")
    console.print("=" * 80 + "\n")

    session = ChatSession(mode="claim-based")

    tests = [
        # Python tool
        ("Python calculation", "Calculate the square root of 144 using Python"),
        ("Python with imports", "Use Python to get today's date"),

        # Facts storage and retrieval
        ("Store fact", "My favorite color is blue"),
        ("Query fact", "What is my favorite color? Use query_facts to find out"),

        # Beliefs tracking
        ("Make claim", "Python is a programming language"),
        ("Query belief", "What do I believe about Python? Use query_beliefs to check"),

        # Error handling
        ("Python error", "Use Python to divide 10 by zero"),
    ]

    for test_name, question in tests:
        console.print(f"\n[bold yellow]Test:[/bold yellow] {test_name}")
        console.print(f"[dim]User: {question}[/dim]")
        console.print("-" * 80)

        try:
            facts_before = len(session.fact_store.facts)
            response = await session.process_message(question)
            facts_after = len(session.fact_store.facts)

            console.print(f"[green]✓[/green] {response.response_text}")
            console.print(f"[dim]Facts added: {facts_after - facts_before} | Score: {session.score:.2f}pts[/dim]")

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")

    console.print(f"\n{'='*80}")
    console.print(f"[bold]FINAL SCORE:[/bold] {session.score:.2f}pts")
    console.print(f"[bold]Total facts:[/bold] {len(session.fact_store.facts)}")
    console.print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_all_tools())
