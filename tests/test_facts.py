#!/usr/bin/env python3
"""
Test script for Facts system

Run with: uv run test-facts
"""

from baye.fact_store import FactStore, InputMode
from datetime import datetime
import time

def test_basic_facts():
    """Test basic fact creation and retrieval"""
    print("=" * 80)
    print("TEST 1: Basic Fact Creation")
    print("=" * 80)

    store = FactStore(chunk_size=100)

    # Simulate user message
    print("\n1. Adding user message as fact...")
    user_facts = store.add_context(
        content="Trump assumiu a presidência dos EUA em 20 de janeiro de 2025",
        input_mode=InputMode.USER_INPUT,
        author_uuid=store.user_uuid,
        source_context_id="msg_abc123"
    )

    print(f"   Created {len(user_facts)} fact(s)")
    for fact in user_facts:
        print(f"\n{fact.format_structured()}")

    # Simulate tool return
    print("\n2. Adding tool return as fact...")
    tool_uuid = store.register_tool("search_tool")
    print(f"   Tool UUID: {tool_uuid}")

    tool_facts = store.add_context(
        content="Pesquisa retornou: Donald Trump venceu eleição presidencial em novembro de 2024",
        input_mode=InputMode.TOOL_RETURN,
        author_uuid=tool_uuid,
        source_context_id="tool_call_xyz789"
    )

    for fact in tool_facts:
        print(f"\n{fact.format_structured()}")

    # Simulate system prompt
    print("\n3. Adding system prompt as fact...")
    sys_facts = store.add_context(
        content="Você é um assistente prestativo com rastreamento de crenças.",
        input_mode=InputMode.SYSTEM_PROMPT,
        author_uuid=store.system_uuid,
        source_context_id="sys_init"
    )

    for fact in sys_facts:
        print(f"\n{fact.format_structured()}")

    print("\n" + "=" * 80)
    print(f"Total facts in store: {len(store.facts)}")
    print("=" * 80)


def test_chunking():
    """Test automatic chunking of large content"""
    print("\n\n" + "=" * 80)
    print("TEST 2: Automatic Chunking")
    print("=" * 80)

    store = FactStore(chunk_size=50)  # Small chunks for demo

    large_content = (
        "Este é um texto muito longo que será automaticamente dividido em chunks. "
        "Cada chunk é armazenado como um fact separado com o mesmo source_context_id. "
        "Isso permite rastrear grandes documentos mantendo chunks gerenciáveis. "
        "O sistema sabe quantos chunks existem no total e qual é o índice de cada um."
    )

    print(f"\nContent length: {len(large_content)} chars")
    print(f"Chunk size: 50 chars")
    print(f"\nAdding content...")

    facts = store.add_context(
        content=large_content,
        input_mode=InputMode.DOCUMENT,
        author_uuid=store.user_uuid,
        source_context_id="doc_xyz"
    )

    print(f"\nCreated {len(facts)} chunks:\n")

    for fact in facts:
        print(f"[Fact #{fact.seq_id}] Chunk {fact.chunk_index + 1}/{fact.total_chunks}")
        print(f"  Content: \"{fact.content}\"")
        print()


def test_lookup():
    """Test fact lookup by UUID and seq_id"""
    print("\n" + "=" * 80)
    print("TEST 3: Fact Lookup")
    print("=" * 80)

    store = FactStore()

    # Add several facts
    for i in range(5):
        store.add_context(
            content=f"Fact number {i+1}",
            input_mode=InputMode.USER_INPUT,
            author_uuid=store.user_uuid,
            source_context_id=f"msg_{i}"
        )
        time.sleep(0.01)  # Small delay for distinct timestamps

    print("\n1. Lookup by sequential ID:")
    fact = store.get_fact_by_seq(3)
    if fact:
        print(f"\n{fact.format_structured()}")

    print("\n2. Lookup by UUID:")
    # Get first fact's UUID
    first_fact = store.facts_by_seq[1]
    fact = store.get_fact(first_fact.id)
    if fact:
        print(f"\n{fact.format_structured()}")


def test_similarity():
    """Test similarity search"""
    print("\n" + "=" * 80)
    print("TEST 4: Similarity Search")
    print("=" * 80)

    store = FactStore()

    # Add facts about different topics
    facts_data = [
        "Trump assumiu presidência dos EUA",
        "Biden foi presidente antes de Trump",
        "Python é uma linguagem de programação",
        "JavaScript é usado para web development",
        "Eleições presidenciais acontecem a cada 4 anos",
    ]

    for i, content in enumerate(facts_data):
        store.add_context(
            content=content,
            input_mode=InputMode.USER_INPUT,
            author_uuid=store.user_uuid,
            source_context_id=f"msg_{i}"
        )

    print("\nFacts in store:")
    for fact in store.facts.values():
        print(f"  [#{fact.seq_id}] {fact.content}")

    # Search for similar facts
    query = "quem é presidente dos EUA?"
    print(f"\n\nSearching for facts similar to: \"{query}\"")

    similar = store.find_similar(query, k=3, min_similarity=0.1)

    print(f"\nFound {len(similar)} similar facts:\n")
    for fact, similarity in similar:
        print(f"  Similarity: {similarity:.2f}")
        print(f"  [#{fact.seq_id}] {fact.content}")
        print()


def test_contradictions():
    """Test finding contradictions"""
    print("\n" + "=" * 80)
    print("TEST 5: Finding Contradictions")
    print("=" * 80)

    store = FactStore()

    # Add facts
    facts_data = [
        ("Trump é presidente dos EUA desde janeiro 2025", 1.0),
        ("Biden foi presidente dos EUA até janeiro 2025", 1.0),
        ("Eleições presidenciais dos EUA ocorreram em novembro 2024", 1.0),
    ]

    for content, conf in facts_data:
        store.add_context(
            content=content,
            input_mode=InputMode.USER_INPUT,
            author_uuid=store.user_uuid,
            source_context_id="msg_xyz",
            confidence=conf
        )

    print("\nFacts in store:")
    for fact in store.facts.values():
        print(f"  [#{fact.seq_id}] {fact.content}")

    # Find contradictions
    claim = "Joe Biden é o atual presidente dos EUA"
    print(f"\n\nFinding contradictions for: \"{claim}\"")

    contradictions = store.find_contradicting(claim, k=3)

    print(f"\nFound {len(contradictions)} potential contradictions:\n")
    for type_, id_, conf, content in contradictions:
        print(f"  Type: {type_} | Confidence: {conf:.2f}")
        print(f"  ID: {id_[:16]}...")
        print(f"  Content: {content}")
        print()


def test_export():
    """Test fact export"""
    print("\n" + "=" * 80)
    print("TEST 6: Fact Export")
    print("=" * 80)

    store = FactStore()

    # Add a few facts
    for i in range(3):
        store.add_context(
            content=f"Test fact {i+1}",
            input_mode=InputMode.USER_INPUT,
            author_uuid=store.user_uuid,
            source_context_id=f"msg_{i}"
        )

    print("\nExporting all facts as JSON:\n")

    import json
    facts_json = store.export_all()
    print(json.dumps(facts_json, indent=2))


def test_context_format():
    """Test formatting facts for LLM context"""
    print("\n" + "=" * 80)
    print("TEST 7: Format for LLM Context")
    print("=" * 80)

    store = FactStore()

    # Add facts
    facts_data = [
        "Trump assumiu presidência em janeiro 2025",
        "Eleições ocorreram em novembro 2024",
        "Biden foi presidente até janeiro 2025",
    ]

    for content in facts_data:
        store.add_context(
            content=content,
            input_mode=InputMode.USER_INPUT,
            author_uuid=store.user_uuid,
            source_context_id="msg_xyz"
        )

    print("\nFormatted for LLM context:\n")
    print(store.format_for_context(max_facts=5))


def main():
    """Run all tests"""
    tests = [
        ("Basic Facts", test_basic_facts),
        ("Chunking", test_chunking),
        ("Lookup", test_lookup),
        ("Similarity", test_similarity),
        ("Contradictions", test_contradictions),
        ("Export", test_export),
        ("Context Format", test_context_format),
    ]

    print("\n" + "=" * 80)
    print("FACTS SYSTEM TEST SUITE")
    print("=" * 80)

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
