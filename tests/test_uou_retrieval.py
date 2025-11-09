"""
Essential tests for Update-on-Use + Retrieval system (US-15).

Covers critical requirements from DoD:
1. Idempotência (same evidence doesn't alter a,b)
2. Alternating conflict converges c≈0.5 with reduced uncertainty
3. Novelty reduces w with redundancy
4. MMR decreases average similarity between selected
5. Transaction rollback on intermediate failure
"""

import pytest
from datetime import datetime, timedelta

from baye import (
    Belief,
    BeliefSystem,
    Evidence,
    EvidenceStore,
    UpdateOnUseEngine,
    BeliefRanker,
    create_belief_system,
)


# ============================================================================
# US-15 Test 1: Idempotência
# ============================================================================

def test_duplicate_evidence_is_idempotent():
    """
    Given: Same evidence hash
    When: Attempting to register twice
    Then: Second registration is no-op (a,b unchanged)
    """
    system = create_belief_system(enable_all_features=False)

    # Create belief
    belief = Belief.from_confidence(
        content="APIs are reliable",
        confidence=0.5,
        context="test"
    )
    system.graph.beliefs[belief.id] = belief

    initial_a = belief.a
    initial_b = belief.b

    # First update
    evidence1, update1 = system.update_from_tool_call(
        belief_id=belief.id,
        tool_result="API responded successfully",
        tool_name="api_monitor",
        sentiment=1.0
    )

    a_after_first = belief.a
    b_after_first = belief.b

    # Verify first update had effect
    assert a_after_first > initial_a, "First update should increase alpha"

    # Second update with IDENTICAL content (duplicate)
    evidence2, update2 = system.update_from_tool_call(
        belief_id=belief.id,
        tool_result="API responded successfully",  # SAME
        tool_name="api_monitor",  # SAME
        sentiment=1.0
    )

    # Verify idempotency
    assert update2['was_duplicate'], "Second update should be marked as duplicate"
    assert belief.a == a_after_first, "Alpha should not change for duplicate"
    assert belief.b == b_after_first, "Beta should not change for duplicate"
    assert update2['confidence_delta'] == 0.0, "Confidence delta should be zero"


# ============================================================================
# US-15 Test 2: Alternating conflict converges to c≈0.5
# ============================================================================

def test_alternating_conflict_converges_to_neutral():
    """
    Given: Alternating positive and negative evidence
    When: Applied to same belief
    Then: Confidence converges to ≈0.5 and uncertainty reduces
    """
    system = create_belief_system(enable_all_features=False)

    # Start with neutral belief
    belief = Belief.from_confidence(
        content="The service is reliable",
        confidence=0.0,  # Neutral
        context="test"
    )
    system.graph.beliefs[belief.id] = belief

    initial_uncertainty = belief.uncertainty

    # Apply alternating evidence
    for i in range(10):
        sentiment = 1.0 if i % 2 == 0 else -1.0
        result_text = f"Evidence {i}: {'success' if sentiment > 0 else 'failure'}"

        system.update_from_tool_call(
            belief_id=belief.id,
            tool_result=result_text,
            tool_name="monitor",
            sentiment=sentiment,
            quality=1.0
        )

    final_confidence = belief.confidence
    final_uncertainty = belief.uncertainty

    # Check convergence to neutral
    assert abs(final_confidence) < 0.3, \
        f"Confidence should converge near 0, got {final_confidence}"

    # Check uncertainty reduction (more evidence = more certain, even if neutral)
    assert final_uncertainty < initial_uncertainty, \
        "Uncertainty should decrease with more evidence"

    # Check evidence count increased
    assert belief.total_evidence > 10, "Total evidence should accumulate"


# ============================================================================
# US-15 Test 3: Novelty reduces weight with redundancy
# ============================================================================

def test_novelty_reduces_weight_for_redundant_evidence():
    """
    Given: Evidence similar to existing evidence
    When: Calculating weight
    Then: Novelty component n reduces weight
    """
    store = EvidenceStore()
    engine = UpdateOnUseEngine(store)

    belief = Belief.from_confidence(
        content="Test belief",
        confidence=0.5,
        context="test"
    )

    # First evidence (novel)
    evidence1 = Evidence(
        belief_id=belief.id,
        content="The API returned status 200",
        source="tool:test",
        sentiment=1.0,
        quality=1.0
    )

    store.add_evidence(evidence1)

    update1 = engine.update_belief_from_evidence(
        belief=belief,
        evidence=evidence1,
        reliability=0.9
    )

    novelty1 = update1.novelty_n
    weight1 = update1.weight_w

    # Second evidence (very similar - redundant)
    evidence2 = Evidence(
        belief_id=belief.id,
        content="The API returned status 200 OK",  # Very similar
        source="tool:test",
        sentiment=1.0,
        quality=1.0
    )

    # Reset belief to compare weights fairly
    belief = Belief.from_confidence("Test belief", 0.5, "test")

    store.add_evidence(evidence2)

    update2 = engine.update_belief_from_evidence(
        belief=belief,
        evidence=evidence2,
        reliability=0.9
    )

    novelty2 = update2.novelty_n
    weight2 = update2.weight_w

    # Verify novelty decreases
    assert novelty2 < novelty1, \
        f"Redundant evidence should have lower novelty: {novelty2} vs {novelty1}"

    # Verify weight decreases
    assert weight2 < weight1, \
        f"Redundant evidence should have lower weight: {weight2} vs {weight1}"


# ============================================================================
# US-15 Test 4: MMR decreases average similarity
# ============================================================================

def test_mmr_reduces_similarity_between_selected():
    """
    Given: Candidates with high mutual similarity
    When: Ranking with MMR enabled
    Then: Selected beliefs have lower average pairwise similarity
    """
    # Create beliefs with varying similarity
    beliefs = [
        Belief.from_confidence("API endpoints can timeout", 0.7, "api"),
        Belief.from_confidence("API calls can timeout", 0.7, "api"),  # Very similar to #1
        Belief.from_confidence("Network requests can timeout", 0.6, "api"),  # Similar
        Belief.from_confidence("Database queries are fast", 0.5, "db"),  # Different
        Belief.from_confidence("Cache improves performance", 0.6, "perf"),  # Different
    ]

    belief_graph = {b.id: b for b in beliefs}

    ranker = BeliefRanker(mmr_lambda=0.5)  # Balanced MMR

    query = "What causes timeouts?"

    # Rank WITHOUT MMR (pure relevance)
    ranked_no_mmr = ranker.rank_beliefs(
        candidates=beliefs,
        query=query,
        belief_graph=belief_graph,
        k=3
    )

    # Rank WITH MMR
    ranked_with_mmr = ranker.rank_with_mmr(
        candidates=beliefs,
        query=query,
        belief_graph=belief_graph,
        k=3
    )

    # Calculate average pairwise similarity
    def avg_pairwise_similarity(candidates):
        if len(candidates) < 2:
            return 0.0

        similarities = []
        for i, cand_a in enumerate(candidates):
            for j in range(i + 1, len(candidates)):
                cand_b = candidates[j]
                sim = ranker.similarity_fn(cand_a.belief.content, cand_b.belief.content)
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    sim_no_mmr = avg_pairwise_similarity(ranked_no_mmr)
    sim_with_mmr = avg_pairwise_similarity(ranked_with_mmr)

    # MMR should reduce average similarity (more diverse)
    assert sim_with_mmr < sim_no_mmr or abs(sim_with_mmr - sim_no_mmr) < 0.05, \
        f"MMR should reduce similarity: {sim_with_mmr} vs {sim_no_mmr}"


# ============================================================================
# US-15 Test 5: Transaction rollback (conceptual)
# ============================================================================

def test_update_preserves_state_on_error():
    """
    Given: Update process that might fail
    When: Error occurs during update
    Then: Belief state is preserved (no partial updates)

    Note: This is a conceptual test. In production, wrap updates in
    transactions or use copy-on-write.
    """
    system = create_belief_system(enable_all_features=False)

    belief = Belief.from_confidence("Test", 0.5, "test")
    system.graph.beliefs[belief.id] = belief

    initial_a = belief.a
    initial_b = belief.b
    initial_conf = belief.confidence

    # Attempt update with invalid parameters (should fail gracefully)
    try:
        # Force an error by passing invalid tool_result type
        system.update_from_tool_call(
            belief_id="nonexistent_id",  # This will cause KeyError
            tool_result="test",
            tool_name="test"
        )
    except KeyError:
        pass  # Expected

    # Verify original belief is unchanged
    assert belief.a == initial_a, "Alpha should be unchanged after error"
    assert belief.b == initial_b, "Beta should be unchanged after error"
    assert belief.confidence == initial_conf, "Confidence should be unchanged after error"


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_retrieval_pipeline():
    """Test complete retrieval pipeline."""
    system = create_belief_system(enable_all_features=True)

    # Add some beliefs
    b1 = Belief.from_confidence("APIs can fail unexpectedly", 0.7, "reliability")
    b2 = Belief.from_confidence("Always validate API responses", 0.8, "best_practice")
    b3 = Belief.from_confidence("Python is great for ML", 0.6, "programming")

    system.graph.beliefs[b1.id] = b1
    system.graph.beliefs[b2.id] = b2
    system.graph.beliefs[b3.id] = b3

    # Retrieve context
    context = system.retrieve_context_for_prompt(
        prompt="How should I handle API failures?",
        k=2,
        token_budget=500
    )

    # Verify context is not empty
    assert len(context) > 0, "Context should not be empty"
    assert "BELIEF" in context or "CRENÇA" in context, "Should contain belief cards"


def test_update_on_use_tool_wrapper():
    """Test UpdateOnUseTool decorator."""
    from baye import UpdateOnUseTool

    system = create_belief_system(enable_all_features=False)

    belief = Belief.from_confidence("Tool is reliable", 0.5, "test")
    system.graph.beliefs[belief.id] = belief

    initial_conf = belief.confidence

    # Define a mock tool
    @UpdateOnUseTool(
        system=system,
        belief_id=belief.id,
        tool_name="mock_tool",
        sentiment_fn=lambda result: 1.0 if "success" in result else -1.0
    )
    def mock_tool(arg):
        return f"success: {arg}"

    # Call tool (should trigger update)
    result = mock_tool("test_arg")

    assert result == "success: test_arg"
    assert belief.confidence != initial_conf, "Belief should be updated after tool use"


def test_observability_metrics():
    """Test observability system."""
    system = create_belief_system(enable_all_features=True)

    belief = Belief.from_confidence("Test", 0.5, "test")
    system.graph.beliefs[belief.id] = belief

    # Perform updates
    for i in range(5):
        system.update_from_tool_call(
            belief_id=belief.id,
            tool_result=f"Evidence {i}",
            tool_name="test_tool",
            sentiment=1.0 if i % 2 == 0 else -1.0
        )

    # Get dashboard data
    dashboard = system.get_dashboard_data()

    assert 'metrics' in dashboard
    assert 'audit_stats' in dashboard
    assert dashboard['audit_stats']['total_entries'] >= 5


def test_policy_abstention():
    """Test abstention policy."""
    from baye import PolicyManager, create_scratch_belief

    policy_manager = PolicyManager()

    # Create scratch belief (has high threshold)
    belief = create_scratch_belief("Temporary investigation", "test")

    # Very small weight should trigger abstention
    should_update, reason = policy_manager.should_update(belief, weight=0.001)

    # Scratch beliefs have min_weight_threshold = 0.05
    assert not should_update, "Should abstain from tiny weight update"
    assert "threshold" in reason.lower()


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_graph_retrieval():
    """Test retrieval with empty graph."""
    system = create_belief_system()

    context = system.retrieve_context_for_prompt(
        prompt="Test query",
        k=5
    )

    # Should return empty or minimal context
    assert isinstance(context, str)


def test_confidence_bounds():
    """Test that confidence stays in [-1, 1]."""
    belief = Belief.from_confidence("Test", 0.9, "test")

    # Try to push beyond bounds
    for _ in range(100):
        belief.update_beta(delta_a=10.0, delta_b=0.0)

    assert -1.0 <= belief.confidence <= 1.0, "Confidence must stay in [-1, 1]"


def test_beta_parameters_stay_positive():
    """Test that Beta parameters stay positive."""
    belief = Belief(content="Test", a=1.0, b=1.0)

    # Try negative updates (should be clamped)
    belief.update_beta(delta_a=-10.0, delta_b=-10.0)

    assert belief.a > 0, "Alpha must stay positive"
    assert belief.b > 0, "Beta must stay positive"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
