#!/usr/bin/env python3
"""
Test chunking for large tool outputs
"""

import asyncio
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession
from rich.console import Console

console = Console()

async def test_chunking():
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]CHUNKING TEST - Large Tool Outputs[/bold cyan]")
    console.print("=" * 80 + "\n")

    session = ChatSession(mode="claim-based")

    tests = [
        # Small output (no chunking needed)
        ("Small output", "Use Python to print 'Hello World'"),

        # Medium output (might need chunking)
        ("Medium output", "Use Python to print all numbers from 1 to 100"),

        # Large output (definitely needs chunking)
        ("Large output", """Use Python to generate a long text:
print('A' * 10000)  # 10k characters
for i in range(100):
    print(f'Line {i}: This is a test of chunking with a reasonably long sentence to ensure we exceed token limits.')
"""),
    ]

    for test_name, question in tests:
        console.print(f"\n[bold yellow]Test:[/bold yellow] {test_name}")
        console.print(f"[dim]{question[:80]}...[/dim]" if len(question) > 80 else f"[dim]{question}[/dim]")
        console.print("-" * 80)

        try:
            facts_before = len(session.fact_store.facts)
            response = await session.process_message(question)
            facts_after = len(session.fact_store.facts)
            facts_added = facts_after - facts_before

            # Check for chunked facts
            chunked_facts = [
                f for f in session.fact_store.facts.values()
                if f.total_chunks > 1
            ]

            console.print(f"[green]✓[/green] Response received")
            console.print(f"[dim]Facts added: {facts_added}[/dim]")

            if chunked_facts:
                # Group by source_context_id
                chunks_by_source = {}
                for fact in chunked_facts:
                    if fact.source_context_id not in chunks_by_source:
                        chunks_by_source[fact.source_context_id] = []
                    chunks_by_source[fact.source_context_id].append(fact)

                console.print(f"[yellow]⚠️  Chunked facts detected:[/yellow]")
                for source_id, chunks in chunks_by_source.items():
                    console.print(f"  Source: {source_id}")
                    console.print(f"  Total chunks: {chunks[0].total_chunks}")
                    console.print(f"  Chunk sizes: {[len(c.content) for c in sorted(chunks, key=lambda x: x.chunk_index)]}")

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            import traceback
            traceback.print_exc()

    console.print(f"\n{'='*80}")
    console.print(f"[bold]FINAL SCORE:[/bold] {session.score:.2f}pts")
    console.print(f"[bold]Total facts:[/bold] {len(session.fact_store.facts)}")

    # Count chunked vs single facts
    chunked = sum(1 for f in session.fact_store.facts.values() if f.total_chunks > 1)
    single = sum(1 for f in session.fact_store.facts.values() if f.total_chunks == 1)
    console.print(f"[bold]Chunked facts:[/bold] {chunked}")
    console.print(f"[bold]Single facts:[/bold] {single}")
    console.print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_chunking())
