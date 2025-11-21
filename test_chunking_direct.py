#!/usr/bin/env python3
"""
Direct test of chunking logic (without LLM)
"""

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.fact_store import FactStore, InputMode
from rich.console import Console

console = Console()

def test_chunking_logic():
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]DIRECT CHUNKING TEST[/bold cyan]")
    console.print("=" * 80 + "\n")

    fact_store = FactStore()

    # Test 1: Small text (no chunking)
    console.print("[bold yellow]Test 1:[/bold yellow] Small text (< 1500 tokens)")
    small_text = "This is a small text that should not be chunked."
    facts = fact_store.add_context(
        content=small_text,
        input_mode=InputMode.TOOL_RETURN,
        author_uuid="test-tool",
        source_context_id="test-1"
    )
    console.print(f"  Input length: {len(small_text)} chars (~{len(small_text)//4} tokens)")
    console.print(f"  Facts created: {len(facts)}")
    console.print(f"  Total chunks: {facts[0].total_chunks}")
    console.print()

    # Test 2: Medium text (might chunk)
    console.print("[bold yellow]Test 2:[/bold yellow] Medium text (~3000 tokens)")
    medium_text = "A" * 12000  # ~3000 tokens
    facts = fact_store.add_context(
        content=medium_text,
        input_mode=InputMode.TOOL_RETURN,
        author_uuid="test-tool",
        source_context_id="test-2"
    )
    console.print(f"  Input length: {len(medium_text)} chars (~{len(medium_text)//4} tokens)")
    console.print(f"  Facts created: {len(facts)}")
    console.print(f"  Total chunks: {facts[0].total_chunks}")
    if len(facts) > 1:
        console.print(f"  [green]✓ Chunking occurred![/green]")
        for i, fact in enumerate(facts):
            console.print(f"    Chunk {i}: {len(fact.content)} chars, index={fact.chunk_index}/{fact.total_chunks-1}")
    console.print()

    # Test 3: Large text with sentences (realistic)
    console.print("[bold yellow]Test 3:[/bold yellow] Large text with sentences (~8000 tokens)")
    large_text = ". ".join([
        f"This is sentence number {i} and it contains some meaningful content to test the chunking logic"
        for i in range(500)
    ]) + "."
    facts = fact_store.add_context(
        content=large_text,
        input_mode=InputMode.TOOL_RETURN,
        author_uuid="test-tool",
        source_context_id="test-3"
    )
    console.print(f"  Input length: {len(large_text)} chars (~{len(large_text)//4} tokens)")
    console.print(f"  Facts created: {len(facts)}")
    console.print(f"  Total chunks: {facts[0].total_chunks}")
    if len(facts) > 1:
        console.print(f"  [green]✓ Chunking occurred![/green]")
        for i, fact in enumerate(facts):
            console.print(f"    Chunk {i}: {len(fact.content)} chars (~{len(fact.content)//4} tokens), index={fact.chunk_index}/{fact.total_chunks-1}")

            # Show first 100 chars of each chunk
            preview = fact.content[:100].replace("\n", " ")
            console.print(f"      Preview: {preview}...")
    console.print()

    # Test 4: Very long sentence (edge case)
    console.print("[bold yellow]Test 4:[/bold yellow] Very long sentence without periods")
    very_long = "word " * 2000  # ~2000 tokens, no sentence boundaries
    facts = fact_store.add_context(
        content=very_long,
        input_mode=InputMode.TOOL_RETURN,
        author_uuid="test-tool",
        source_context_id="test-4"
    )
    console.print(f"  Input length: {len(very_long)} chars (~{len(very_long)//4} tokens)")
    console.print(f"  Facts created: {len(facts)}")
    console.print(f"  Total chunks: {facts[0].total_chunks}")
    if len(facts) > 1:
        console.print(f"  [green]✓ Chunking occurred (word-level)![/green]")
        for i, fact in enumerate(facts):
            console.print(f"    Chunk {i}: {len(fact.content)} chars (~{len(fact.content)//4} tokens)")
    console.print()

    console.print("=" * 80)
    console.print(f"[bold]Total facts in store:[/bold] {len(fact_store.facts)}")

    # Count chunked vs non-chunked
    chunked = sum(1 for f in fact_store.facts.values() if f.total_chunks > 1)
    single = sum(1 for f in fact_store.facts.values() if f.total_chunks == 1)
    console.print(f"[bold]Chunked facts:[/bold] {chunked}")
    console.print(f"[bold]Single facts:[/bold] {single}")
    console.print("=" * 80 + "\n")

if __name__ == "__main__":
    test_chunking_logic()
