#!/usr/bin/env python3
"""
Stress test for claim-based mode with user facts
Tests multiple scenarios to verify robustness
"""

import asyncio
import os
import shutil
from datetime import datetime

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

from baye.chat_session import ChatSession


class TestScenario:
    def __init__(self, name: str, user_inputs: list[str], expected_checks: list[tuple[str, str]]):
        """
        Args:
            name: Scenario name
            user_inputs: List of user messages to send
            expected_checks: List of (response_text, keyword_to_check) tuples
        """
        self.name = name
        self.user_inputs = user_inputs
        self.expected_checks = expected_checks
        self.passed = False
        self.errors = []


async def run_scenario(scenario: TestScenario, session: ChatSession) -> bool:
    """Run a test scenario and return success/failure"""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario.name}")
    print(f"{'='*80}")

    for i, user_input in enumerate(scenario.user_inputs):
        print(f"\n[Step {i+1}/{len(scenario.user_inputs)}] User: '{user_input}'")
        print("-" * 80)

        try:
            response = await session.process_message(user_input)
            print(f"✓ Response: {response.response_text[:100]}...")
            print(f"  Score: {session.score:.2f}pts")

            # Check expectations if provided
            if i < len(scenario.expected_checks):
                keyword = scenario.expected_checks[i]
                if keyword and keyword.lower() in response.response_text.lower():
                    print(f"  ✓ Found expected keyword: '{keyword}'")
                else:
                    if keyword:
                        error = f"  ✗ Missing expected keyword: '{keyword}'"
                        print(error)
                        scenario.errors.append(error)

        except Exception as e:
            error = f"✗ Error: {e}"
            print(error)
            scenario.errors.append(error)
            return False

    # Final summary
    print(f"\n{'='*80}")
    print(f"SCENARIO RESULT: {scenario.name}")
    print(f"Final Score: {session.score:.2f}pts")
    print(f"Successes: {session.successful_claims} | Failures: {session.failed_claims}")

    if not scenario.errors:
        print("✓ PASSED")
        scenario.passed = True
        return True
    else:
        print("✗ FAILED")
        for error in scenario.errors:
            print(f"  - {error}")
        return False


async def stress_test():
    print("=" * 80)
    print("STRESS TEST: Claim-Based Mode with User Facts")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Clean test data
    test_dir = ".baye_stress_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    scenarios = [
        # Scenario 1: Basic fact assertion and retrieval
        TestScenario(
            name="Basic Fact Storage",
            user_inputs=[
                "Estamos em 2025",
                "Em que ano estamos?"
            ],
            expected_checks=[
                None,  # First message is assertion
                "2025"  # Should return the year
            ]
        ),

        # Scenario 2: Contradicting known facts
        TestScenario(
            name="Contradiction Handling",
            user_inputs=[
                "Trump é presidente dos EUA",
                "Quem é presidente dos EUA?"
            ],
            expected_checks=[
                None,
                "Trump"  # Should use user's fact
            ]
        ),

        # Scenario 3: Multiple facts in same domain
        TestScenario(
            name="Multiple Related Facts",
            user_inputs=[
                "Python é uma linguagem de programação",
                "Python foi criado por Guido van Rossum",
                "Quem criou Python?"
            ],
            expected_checks=[
                None,
                None,
                "Guido"  # Should recall the creator
            ]
        ),

        # Scenario 4: Temporal reasoning
        TestScenario(
            name="Temporal Facts",
            user_inputs=[
                "Hoje é segunda-feira",
                "Amanhã tenho reunião às 10h",
                "Que dia é hoje?"
            ],
            expected_checks=[
                None,
                None,
                "segunda"  # Should remember it's Monday
            ]
        ),

        # Scenario 5: Numerical facts
        TestScenario(
            name="Numerical Facts",
            user_inputs=[
                "A população do Brasil é 215 milhões",
                "Quantos habitantes tem o Brasil?"
            ],
            expected_checks=[
                None,
                "215"  # Should recall the number
            ]
        ),

        # Scenario 6: Personal preferences
        TestScenario(
            name="Personal Preferences",
            user_inputs=[
                "Minha cor favorita é azul",
                "Gosto de café sem açúcar",
                "Qual minha cor favorita?"
            ],
            expected_checks=[
                None,
                None,
                "azul"  # Should remember preference
            ]
        ),

        # Scenario 7: Chain of reasoning
        TestScenario(
            name="Chain of Facts",
            user_inputs=[
                "São Paulo é a maior cidade do Brasil",
                "O Brasil fica na América do Sul",
                "Onde fica São Paulo?"
            ],
            expected_checks=[
                None,
                None,
                "Brasil"  # Should connect the facts
            ]
        ),

        # Scenario 8: Correction of previous fact
        TestScenario(
            name="Fact Correction",
            user_inputs=[
                "Meu nome é João",
                "Na verdade, meu nome é Maria",
                "Qual é meu nome?"
            ],
            expected_checks=[
                None,
                None,
                "Maria"  # Should use most recent fact
            ]
        ),

        # Scenario 9: High confidence assertions
        TestScenario(
            name="Confident Assertions",
            user_inputs=[
                "O céu é azul",
                "De que cor é o céu?"
            ],
            expected_checks=[
                None,
                "azul"
            ]
        ),

        # Scenario 10: Persistence across sessions
        TestScenario(
            name="Cross-Session Persistence",
            user_inputs=[
                "Meu aniversário é em dezembro",
            ],
            expected_checks=[
                None
            ]
        ),
    ]

    # Run all scenarios
    results = []
    total_score = 0

    for idx, scenario in enumerate(scenarios):
        # Create new session for each scenario (but same persist_directory)
        session = ChatSession(
            mode="claim-based",
            persist_directory=test_dir
        )

        success = await run_scenario(scenario, session)
        results.append((scenario.name, success, session.score))
        total_score += session.score

        # Small delay between scenarios
        await asyncio.sleep(0.5)

    # Test persistence: create new session and check if facts are still there
    print(f"\n{'='*80}")
    print("PERSISTENCE CHECK: Testing cross-session fact retrieval")
    print(f"{'='*80}")

    new_session = ChatSession(mode="claim-based", persist_directory=test_dir)
    print(f"Facts loaded: {len(new_session.fact_store.facts)}")

    # Try to retrieve a fact from earlier session
    try:
        response = await new_session.process_message("Em que mês é meu aniversário?")
        if "dezembro" in response.response_text.lower():
            print("✓ Successfully retrieved fact from previous session!")
            results.append(("Cross-Session Retrieval", True, new_session.score))
        else:
            print("✗ Failed to retrieve fact from previous session")
            results.append(("Cross-Session Retrieval", False, new_session.score))
    except Exception as e:
        print(f"✗ Error testing persistence: {e}")
        results.append(("Cross-Session Retrieval", False, 0))

    # Final report
    print(f"\n{'='*80}")
    print("FINAL REPORT")
    print(f"{'='*80}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"Results: {passed}/{total} scenarios passed ({passed/total*100:.1f}%)")
    print(f"Total accumulated score: {total_score:.2f}pts")
    print()

    # Detailed results
    print("Scenario Details:")
    for name, success, score in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} | {score:6.2f}pts | {name}")

    print(f"\n{'='*80}")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")

    print(f"{'='*80}")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(stress_test())
    exit(0 if success else 1)
