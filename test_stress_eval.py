#!/usr/bin/env python3
"""
Stress test / Evaluation suite for belief tracking

Tests various types of claims with known ground truth to evaluate
the model's ability to use negative/positive confidence correctly.
"""

import asyncio
import os
from baye.chat_session import ChatSession
from baye.llm_agents import check_gemini_api_key
from typing import List, Dict, Optional

# Test cases with ground truth
TEST_CASES = [
    {
        "question": "Existe um emoji para o cavalo marinho?",
        "expected_answer": "Não existe",  # 🐴 (horse) + 🌊 (water) but no seahorse emoji
        "expected_confidence_sign": "negative",  # Should disbelieve there's a seahorse emoji
        "category": "knowledge_specific",
        "notes": "Tricky - there's horse and water emojis but no seahorse"
    },
    {
        "question": "Barack Obama é o presidente dos EUA atualmente?",
        "expected_answer": "Não",
        "expected_confidence_sign": "negative",
        "category": "current_events",
        "notes": "Obama was president 2009-2017"
    },
    {
        "question": "Donald Trump foi presidente dos EUA?",
        "expected_answer": "Sim",
        "expected_confidence_sign": "positive",
        "category": "historical_fact",
        "notes": "Trump was president 2017-2021"
    },
    {
        "question": "Python é uma linguagem de programação?",
        "expected_answer": "Sim",
        "expected_confidence_sign": "positive",
        "category": "general_knowledge",
        "notes": "Well-known programming language"
    },
    {
        "question": "A Terra é plana?",
        "expected_answer": "Não",
        "expected_confidence_sign": "negative",
        "category": "scientific_fact",
        "notes": "Should strongly disbelieve flat earth"
    },
    {
        "question": "2 + 2 = 4?",
        "expected_answer": "Sim",
        "expected_confidence_sign": "positive",
        "category": "mathematics",
        "notes": "Basic arithmetic"
    },
    {
        "question": "O Sol gira em torno da Terra?",
        "expected_answer": "Não",
        "expected_confidence_sign": "negative",
        "category": "scientific_fact",
        "notes": "Heliocentric model"
    },
    {
        "question": "Existe um emoji de unicórnio?",
        "expected_answer": "Sim",  # 🦄 exists
        "expected_confidence_sign": "positive",
        "category": "knowledge_specific",
        "notes": "🦄 is a real emoji"
    },
    {
        "question": "JavaScript é usado para desenvolvimento web?",
        "expected_answer": "Sim",
        "expected_confidence_sign": "positive",
        "category": "technical_knowledge",
        "notes": "Primary web programming language"
    },
    {
        "question": "Gatos podem voar naturalmente?",
        "expected_answer": "Não",
        "expected_confidence_sign": "negative",
        "category": "common_sense",
        "notes": "Basic biological knowledge"
    },
    {
        "question": "A água ferve a 100°C ao nível do mar?",
        "expected_answer": "Sim",
        "expected_confidence_sign": "positive",
        "category": "scientific_fact",
        "notes": "Standard physics knowledge"
    },
    {
        "question": "Vampiros existem na vida real?",
        "expected_answer": "Não",
        "expected_confidence_sign": "negative",
        "category": "mythology_vs_reality",
        "notes": "Should disbelieve supernatural entities"
    },
]


class EvalResult:
    """Result of evaluating one test case"""
    def __init__(
        self,
        question: str,
        response_text: str,
        confidence: float,
        expected_sign: str,
        category: str,
        notes: str
    ):
        self.question = question
        self.response_text = response_text
        self.confidence = confidence
        self.expected_sign = expected_sign
        self.category = category
        self.notes = notes

        # Evaluate correctness
        if expected_sign == "positive":
            self.correct_sign = confidence > 0
        else:
            self.correct_sign = confidence < 0

        # Check if text was negated (which we don't want)
        self.text_negated = "não" in response_text.lower() or "nao" in response_text.lower()


async def run_eval():
    """Run evaluation on all test cases"""

    print("=" * 80)
    print("BELIEF TRACKING STRESS TEST / EVALUATION")
    print("=" * 80)
    print()

    # Ensure API key is set
    try:
        check_gemini_api_key()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nPlease set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return

    # Create session
    print("Creating chat session in legacy mode...")
    session = ChatSession(mode="legacy")
    print(f"✓ Session created with {len(TEST_CASES)} test cases")
    print()

    results: List[EvalResult] = []

    # Run each test case
    for i, test_case in enumerate(TEST_CASES, 1):
        print("-" * 80)
        print(f"Test {i}/{len(TEST_CASES)}: {test_case['category'].upper()}")
        print("-" * 80)
        print(f"Question: {test_case['question']}")
        print(f"Expected: {test_case['expected_answer']} (confidence {test_case['expected_confidence_sign']})")
        print()

        try:
            # Get response
            response = await session.process_message(test_case['question'])

            # Create result
            result = EvalResult(
                question=test_case['question'],
                response_text=response.text,
                confidence=response.belief_value_guessed,
                expected_sign=test_case['expected_confidence_sign'],
                category=test_case['category'],
                notes=test_case['notes']
            )
            results.append(result)

            # Display result
            print(f"Response: {response.text[:100]}...")
            print(f"Confidence: {response.belief_value_guessed:.4f} (expected: {test_case['expected_confidence_sign']})")

            # Check correctness
            if result.correct_sign:
                print("✅ Confidence sign CORRECT")
            else:
                print(f"❌ Confidence sign WRONG (got {response.belief_value_guessed:.4f}, expected {test_case['expected_confidence_sign']})")

            if result.text_negated and test_case['expected_confidence_sign'] == 'negative':
                print("⚠️  Text contains negation (ideally should keep positive statement)")

            print()

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Print summary
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()

    # Overall stats
    total = len(results)
    correct_sign = sum(1 for r in results if r.correct_sign)
    text_negated_count = sum(1 for r in results if r.text_negated)

    print(f"Total test cases: {total}")
    print(f"Correct confidence sign: {correct_sign}/{total} ({100*correct_sign/total:.1f}%)")
    print(f"Text negations: {text_negated_count}/{total} ({100*text_negated_count/total:.1f}%)")
    print()

    # By category
    print("Results by category:")
    categories = {}
    for result in results:
        if result.category not in categories:
            categories[result.category] = {"total": 0, "correct": 0}
        categories[result.category]["total"] += 1
        if result.correct_sign:
            categories[result.category]["correct"] += 1

    for category, stats in sorted(categories.items()):
        accuracy = 100 * stats["correct"] / stats["total"]
        print(f"  {category:30s} {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    print()

    # Failed cases
    failed = [r for r in results if not r.correct_sign]
    if failed:
        print("❌ Failed cases:")
        for result in failed:
            print(f"  - {result.question}")
            print(f"    Got: {result.confidence:.4f} | Expected: {result.expected_sign}")
            print(f"    Response: {result.response_text[:80]}...")
            print()
    else:
        print("✅ All cases passed!")

    # Export results
    print("=" * 80)
    print("Detailed results saved to: eval_results.txt")
    print("=" * 80)

    with open("eval_results.txt", "w") as f:
        f.write("BELIEF TRACKING EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"Test {i}: {result.category}\n")
            f.write(f"Question: {result.question}\n")
            f.write(f"Response: {result.response_text}\n")
            f.write(f"Confidence: {result.confidence:.4f} (expected: {result.expected_sign})\n")
            f.write(f"Correct sign: {'YES' if result.correct_sign else 'NO'}\n")
            f.write(f"Text negated: {'YES' if result.text_negated else 'NO'}\n")
            f.write(f"Notes: {result.notes}\n")
            f.write("-" * 80 + "\n\n")

        f.write(f"\nSUMMARY\n")
        f.write(f"Total: {total}\n")
        f.write(f"Correct sign: {correct_sign}/{total} ({100*correct_sign/total:.1f}%)\n")
        f.write(f"Text negations: {text_negated_count}/{total} ({100*text_negated_count/total:.1f}%)\n")


if __name__ == "__main__":
    asyncio.run(run_eval())
