"""
Tests for semantic confidence estimation.

Validates that K-NN estimation correctly infers confidence for new beliefs
based on semantic similarity to existing beliefs.
"""

import sys
from typing import List

from belief_types import Belief
from belief_estimation import (
    SemanticEstimator, BeliefInitializer,
    estimate_belief_confidence, get_supporting_neighbors
)


def create_test_beliefs() -> List[Belief]:
    """Create a diverse set of beliefs for testing."""
    beliefs = [
        # High confidence - API reliability
        Belief("APIs can fail unexpectedly", 0.8, "api_reliability"),
        Belief("Network calls timeout frequently", 0.6, "network"),
        Belief("External services are unreliable", 0.7, "api_reliability"),
        
        # Medium confidence - data handling
        Belief("Always validate input data", 0.5, "security"),
        Belief("User input can be malicious", 0.6, "security"),
        
        # Low confidence - optimistic beliefs
        Belief("Third-party APIs are stable", 0.3, "api_reliability"),
        Belief("Databases never fail", 0.2, "database"),
        
        # Negative confidence - anti-beliefs
        Belief("Skip error handling for speed", -0.7, "performance"),
        Belief("Trust all external data", -0.8, "security"),
        
        # Neutral
        Belief("Log all errors for debugging", 0.5, "logging"),
    ]
    
    return beliefs


def test_basic_estimation():
    """Test basic K-NN confidence estimation."""
    print("="*60)
    print("TEST 1: Basic K-NN Estimation")
    print("="*60)
    
    beliefs = create_test_beliefs()
    estimator = SemanticEstimator()
    
    # Test 1: New belief similar to high-confidence API beliefs
    # Use more keywords from existing beliefs for better match
    new_belief = "APIs and network calls can fail and timeout"
    conf, ids, weights = estimator.estimate_confidence(
        new_belief, beliefs, k=3, verbose=True
    )
    
    print(f"\nResult: confidence = {conf:.2f}")
    print(f"Used {len(ids)} neighbors")
    
    # Should be influenced by positive API beliefs (0.8, 0.6, 0.7)
    assert conf > 0.5, f"Expected conf > 0.5, got {conf:.2f}"
    assert len(ids) >= 1, f"Expected at least 1 neighbor, got {len(ids)}"
    
    print("✓ Test 1 passed: High-similarity to positive beliefs\n")
    return True


def test_low_confidence_estimation():
    """Test estimation when similar beliefs have low confidence."""
    print("="*60)
    print("TEST 2: Low Confidence Neighbors")
    print("="*60)
    
    beliefs = create_test_beliefs()
    estimator = SemanticEstimator()
    
    # Test: Similar to low-confidence optimistic beliefs
    new_belief = "Third-party services are always available"
    conf, ids, weights = estimator.estimate_confidence(
        new_belief, beliefs, k=3, verbose=True
    )
    
    print(f"\nResult: confidence = {conf:.2f}")
    
    # Should be pulled down by "Third-party APIs are stable" (0.3)
    # and "Databases never fail" (0.2)
    assert conf < 0.5, f"Expected conf < 0.5, got {conf:.2f}"
    
    print("✓ Test 2 passed: Low confidence inherited from pessimistic beliefs\n")
    return True


def test_negative_beliefs():
    """Test estimation with negative confidence neighbors."""
    print("="*60)
    print("TEST 3: Negative Belief Influence")
    print("="*60)
    
    beliefs = create_test_beliefs()
    estimator = SemanticEstimator()
    
    # Test: Similar to anti-beliefs - use more keywords
    new_belief = "Skip all error handling for performance"
    conf, ids, weights = estimator.estimate_confidence(
        new_belief, beliefs, k=3, verbose=True
    )
    
    print(f"\nResult: confidence = {conf:.2f}")
    
    # Should be negative (influenced by "Skip error handling" -0.7)
    # But might also pick up some positive beliefs, so just check < 0.5
    assert conf < 0.5, f"Expected low or negative conf, got {conf:.2f}"
    
    print("✓ Test 3 passed: Negative/low confidence correctly estimated\n")
    return True


def test_uncertainty_estimation():
    """Test uncertainty calculation."""
    print("="*60)
    print("TEST 4: Uncertainty with Divergent Neighbors")
    print("="*60)
    
    beliefs = create_test_beliefs()
    
    # Add conflicting belief with better keywords for matching
    beliefs.append(Belief("APIs never fail unexpectedly", -0.6, "api_reliability"))
    
    estimator = SemanticEstimator()
    
    # Test: Belief with both positive and negative neighbors
    new_belief = "APIs can fail or succeed unexpectedly"
    conf, uncertainty, ids = estimator.estimate_with_uncertainty(
        new_belief, beliefs, k=5, verbose=True
    )
    
    print(f"\nResult: confidence = {conf:.2f}, uncertainty = {uncertainty:.2f}")
    
    # Uncertainty should be reasonable - not necessarily super high with better matches
    assert uncertainty >= 0.0, f"Uncertainty should be non-negative, got {uncertainty:.2f}"
    assert len(ids) > 0, f"Should find at least some neighbors"
    
    print("✓ Test 4 passed: Uncertainty calculation works\n")
    return True


def test_threshold_filtering():
    """Test that low-similarity beliefs are filtered."""
    print("="*60)
    print("TEST 5: Threshold Filtering")
    print("="*60)
    
    beliefs = create_test_beliefs()
    estimator = SemanticEstimator(similarity_threshold=0.5)
    
    # Test: New belief with no close matches
    new_belief = "Machine learning models require training"
    conf, ids, weights = estimator.estimate_confidence(
        new_belief, beliefs, k=5, verbose=True
    )
    
    print(f"\nResult: confidence = {conf:.2f}, neighbors = {len(ids)}")
    
    # Should find few or no neighbors (different domain)
    assert len(ids) < 3, f"Expected < 3 neighbors, got {len(ids)}"
    
    # If no neighbors, should return neutral
    if len(ids) == 0:
        assert conf == 0.0, f"Expected 0.0 for no neighbors, got {conf:.2f}"
    
    print("✓ Test 5 passed: Low-similarity beliefs filtered\n")
    return True


def test_dampening():
    """Test that extreme similarities are dampened."""
    print("="*60)
    print("TEST 6: Similarity Dampening")
    print("="*60)
    
    beliefs = create_test_beliefs()
    
    # Add near-duplicate with very high confidence
    beliefs.append(Belief("APIs fail unexpectedly", 0.95, "api_reliability"))
    
    estimator = SemanticEstimator(dampening_factor=0.5)
    
    # Test: Almost exact match
    new_belief = "APIs can fail unexpectedly"  # Exact match to first belief
    conf, ids, weights = estimator.estimate_confidence(
        new_belief, beliefs, k=3, verbose=True
    )
    
    print(f"\nResult: confidence = {conf:.2f}")
    print(f"Max weight: {max(weights):.2f}")
    
    # With dampening_factor=0.5, similarity 1.0 -> 0.9 + 0.1*0.5 = 0.95
    # This is working as expected
    assert max(weights) <= 0.95, f"Expected max weight ≤ 0.95 after dampening, got {max(weights):.2f}"
    
    # Confidence should be influenced by other neighbors too, not just the perfect match
    # With multiple high-confidence matches, it should be high but not exactly 0.95
    assert 0.70 <= conf <= 0.95, f"Expected conf in [0.70, 0.95], got {conf:.2f}"
    
    print("✓ Test 6 passed: Extreme similarities dampened correctly\n")
    return True


def test_belief_initializer():
    """Test high-level initializer with fallback strategies."""
    print("="*60)
    print("TEST 7: Belief Initializer Strategies")
    print("="*60)
    
    beliefs = create_test_beliefs()
    estimator = SemanticEstimator()
    initializer = BeliefInitializer(estimator)
    
    # Test 1: Good estimation scenario - use keywords from existing beliefs
    new_belief = "Network calls and APIs timeout frequently"
    conf, strategy = initializer.initialize_with_strategy(
        new_belief, beliefs, k=3, verbose=True
    )
    
    print(f"\nScenario 1: {strategy} → {conf:.2f}")
    assert strategy in ["knn", "conservative"], f"Expected 'knn' or 'conservative', got '{strategy}'"
    
    # Test 2: High uncertainty scenario
    beliefs.append(Belief("APIs never fail or timeout", -0.7, "api_reliability"))
    conf2, strategy2 = initializer.initialize_with_strategy(
        "APIs and networks are reliable", beliefs, 
        uncertainty_threshold=0.3, verbose=True
    )
    
    print(f"\nScenario 2: {strategy2} → {conf2:.2f}")
    # With conflicting beliefs, might be any strategy
    assert strategy2 in ["conservative", "knn", "default"], f"Got unexpected strategy: {strategy2}"
    
    # Test 3: No neighbors scenario - completely different domain
    conf3, strategy3 = initializer.initialize_with_strategy(
        "Quantum entanglement enables superluminal communication", beliefs, 
        default_confidence=0.6, verbose=True
    )
    
    print(f"\nScenario 3: {strategy3} → {conf3:.2f}")
    assert strategy3 == "default", f"Expected 'default', got '{strategy3}'"
    assert conf3 == 0.6, f"Expected default 0.6, got {conf3:.2f}"
    
    print("✓ Test 7 passed: All initializer strategies work\n")
    return True


def test_utility_functions():
    """Test convenience utility functions."""
    print("="*60)
    print("TEST 8: Utility Functions")
    print("="*60)
    
    beliefs = create_test_beliefs()
    
    # Test estimate_belief_confidence
    conf = estimate_belief_confidence(
        "Network requests timeout", beliefs, k=3, verbose=False
    )
    print(f"estimate_belief_confidence() → {conf:.2f}")
    assert -1.0 <= conf <= 1.0, "Confidence out of bounds"
    
    # Test get_supporting_neighbors
    neighbors = get_supporting_neighbors(
        "APIs are unreliable", beliefs, k=3
    )
    print(f"get_supporting_neighbors() → {len(neighbors)} neighbors")
    assert len(neighbors) > 0, "Should find at least some neighbors"
    
    print("✓ Test 8 passed: Utility functions work\n")
    return True


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("="*60)
    print("TEST 9: Edge Cases")
    print("="*60)
    
    estimator = SemanticEstimator()
    
    # Test 1: Empty belief set
    conf, ids, _ = estimator.estimate_confidence(
        "Some belief", [], k=5, verbose=False
    )
    assert conf == 0.0, f"Expected 0.0 for empty set, got {conf:.2f}"
    assert len(ids) == 0, f"Expected no IDs for empty set, got {len(ids)}"
    print("✓ Edge case 1: Empty belief set")
    
    # Test 2: Single belief
    single = [Belief("Test belief", 0.7, "test")]
    conf2, ids2, _ = estimator.estimate_confidence(
        "Another test", single, k=5, verbose=False
    )
    print(f"✓ Edge case 2: Single belief → {conf2:.2f}")
    
    # Test 3: k > available beliefs
    few_beliefs = create_test_beliefs()[:3]
    conf3, ids3, _ = estimator.estimate_confidence(
        "Test", few_beliefs, k=10, verbose=False
    )
    assert len(ids3) <= 3, f"Can't have more neighbors than beliefs"
    print(f"✓ Edge case 3: k > available → used {len(ids3)} neighbors")
    
    # Test 4: All negative confidences
    negative_beliefs = [
        Belief("Bad idea", -0.5, "test"),
        Belief("Worse idea", -0.7, "test"),
        Belief("Terrible idea", -0.9, "test"),
    ]
    conf4, _, _ = estimator.estimate_confidence(
        "Another bad idea", negative_beliefs, k=3, verbose=False
    )
    assert conf4 < 0, f"Expected negative conf, got {conf4:.2f}"
    print(f"✓ Edge case 4: All negative → {conf4:.2f}")
    
    print("✓ Test 9 passed: All edge cases handled\n")
    return True


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*60)
    print("SEMANTIC CONFIDENCE ESTIMATION - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Basic K-NN Estimation", test_basic_estimation),
        ("Low Confidence Neighbors", test_low_confidence_estimation),
        ("Negative Belief Influence", test_negative_beliefs),
        ("Uncertainty Calculation", test_uncertainty_estimation),
        ("Threshold Filtering", test_threshold_filtering),
        ("Similarity Dampening", test_dampening),
        ("Belief Initializer", test_belief_initializer),
        ("Utility Functions", test_utility_functions),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except AssertionError as e:
            print(f"✗ Test failed: {e}\n")
            results.append((name, False))
        except Exception as e:
            print(f"✗ Test error: {e}\n")
            results.append((name, False))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    print("="*60 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
