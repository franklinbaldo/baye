# Testing Refactoring Plan

## Executive Summary

This document outlines a comprehensive plan to refactor the Baye project's testing infrastructure to align with proper E2E testing principles. It addresses the current state, identifies gaps, and provides a phased migration strategy.

## Current State Analysis

### Existing Test Structure

**Location**: `tests/test_estimation.py` (single file, 381 lines)

**Type**: Unit/Integration tests

**Coverage**:
- ✅ SemanticEstimator K-NN confidence estimation
- ✅ BeliefInitializer with fallback strategies
- ✅ Utility functions (estimate_belief_confidence, get_supporting_neighbors)
- ✅ Edge cases (empty sets, single beliefs, negative confidences)
- ✅ Uncertainty calculation with divergent neighbors
- ✅ Threshold filtering and similarity dampening

**Strengths**:
- Comprehensive coverage of semantic estimation features
- Good edge case handling
- Self-contained test fixtures
- Clear test organization with descriptive names

**Gaps**:
- ❌ No tests for `JustificationGraph` core operations
- ❌ No tests for propagation strategies (causal, semantic)
- ❌ No tests for belief linking and graph operations
- ❌ No E2E tests validating complete workflows
- ❌ No tests for conflict resolution
- ❌ No tests for LLM agent integration (`llm_agents.py`)
- ❌ Missing pytest fixtures and conftest.py
- ❌ No test data fixtures or factories

### Configuration

**PyTest Setup** (`pyproject.toml`):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Dependencies**:
- pytest ≥ 7.0
- pytest-asyncio ≥ 0.21.0

---

## Critical Issue: E2E Strategy Mismatch

### Problem

The current `docs/testing/e2e_strategy.md` references an **"Egregora pipeline"** with components that don't exist in the Baye codebase:

**Mentioned but Not Present**:
- WhatsApp ZIP adapters
- DuckDB storage managers
- Parquet file operations
- MkDocs/Eleventy output adapters
- IR (Interchange Representation) schema
- Window-based processing
- Media extraction

**Actual Baye Components** (from `ARCHITECTURE.md`):
- JustificationGraph (belief storage + NetworkX)
- SemanticEstimator (K-NN for confidence)
- Propagation strategies (causal, semantic)
- Belief types and linking
- LLM agents (PydanticAI + Gemini)

### Resolution

**Option A**: Update `e2e_strategy.md` to reflect Baye's actual architecture
**Option B**: Keep `e2e_strategy.md` as a template and create `baye_e2e_strategy.md`

**Recommendation**: Option B - preserve the generic template and create a Baye-specific strategy.

---

## Proposed E2E Testing Strategy for Baye

### Philosophy

1. **State Verification**: Test graph state changes, not just return values
2. **Realistic Workflows**: Test complete user journeys (add beliefs → link → propagate → query)
3. **LLM Mocking**: Mock expensive LLM calls but test orchestration logic
4. **Isolation**: Each E2E test uses fresh graph instances

### Test Categories

#### 1. Core Graph Operations E2E
**Goal**: Verify fundamental belief management workflows

**Test Scenarios**:
- **Add & Link**: Create belief → auto-link to neighbors → verify graph structure
- **Propagation Cascade**: Update belief → trigger propagation → verify confidence updates across 3+ levels
- **Conflict Detection**: Add contradictory beliefs → verify conflict markers
- **Bulk Import**: Add 100+ beliefs → verify graph consistency metrics

**Example Test Flow**:
```python
# tests/e2e/test_graph_operations.py
def test_belief_lifecycle_with_propagation(graph_fixture):
    # 1. Setup: Add foundation beliefs
    api_belief = graph.add_belief("APIs fail", 0.8, "infrastructure")
    net_belief = graph.add_belief("Networks timeout", 0.7, "infrastructure")

    # 2. Link them
    graph.link_beliefs(net_belief.id, api_belief.id, link_type="supports")

    # 3. Add new belief with estimation
    new_id = graph.add_belief_with_estimation(
        "External services are unreliable",
        context="infrastructure",
        k=3
    )

    # 4. Verify estimation used neighbors
    new_belief = graph.get_belief(new_id)
    assert 0.6 <= new_belief.confidence <= 0.85

    # 5. Update foundation belief
    graph.update_belief_confidence(api_belief.id, 0.3)

    # 6. Propagate changes
    result = graph.propagate_from(api_belief.id)

    # 7. Verify cascade
    assert new_belief.confidence < 0.7  # Should decrease
    assert len(result.updated_beliefs) >= 2
```

#### 2. LLM Agent Integration E2E
**Goal**: Test PydanticAI agent orchestration (mocked Gemini API)

**Test Scenarios**:
- **Relationship Detection**: Agent analyzes two beliefs → returns relationship type
- **Evidence Scoring**: Agent rates evidence strength → updates belief confidence
- **Nuance Generation**: Agent resolves conflicting beliefs → generates refined belief

**Mocking Strategy**:
```python
# tests/conftest.py
@pytest.fixture
def mock_gemini_agent(mocker):
    """Mock PydanticAI agent with deterministic responses"""
    mock_run = mocker.patch("pydantic_ai.Agent.run")

    # Define response mapping
    mock_run.side_effect = lambda prompt: {
        "relationship": "supports" if "align" in prompt else "contradicts",
        "confidence": 0.75
    }

    return mock_run
```

**Example Test Flow**:
```python
# tests/e2e/test_llm_integration.py
def test_agent_detects_relationships(graph_fixture, mock_gemini_agent):
    # 1. Add two unlinked beliefs
    b1 = graph.add_belief("Code reviews catch bugs", 0.7, "quality")
    b2 = graph.add_belief("Peer review improves code", 0.8, "quality")

    # 2. Use agent to detect relationship
    from baye.llm_agents import detect_relationship
    rel_type, confidence = detect_relationship(b1.content, b2.content)

    # 3. Verify agent was called
    mock_gemini_agent.assert_called_once()

    # 4. Verify result
    assert rel_type in ["supports", "contradicts", "refines"]
    assert 0 <= confidence <= 1
```

#### 3. Propagation Strategies E2E
**Goal**: Validate multi-level belief updates with mixed strategies

**Test Scenarios**:
- **Causal-Only**: Linear chain A→B→C, update A, verify C changes
- **Semantic-Only**: Update high-centrality belief, verify similar beliefs update
- **Hybrid (70/30)**: Verify causal updates dominate but semantic provides fine-tuning
- **Budget Constraints**: Verify propagation respects depth budgets [8, 5, 3, 2, 1]

**Example Test Flow**:
```python
# tests/e2e/test_propagation.py
def test_hybrid_propagation_respects_weights(large_graph_fixture):
    # Setup: Graph with 50 beliefs, some linked causally, some semantically similar
    origin = large_graph_fixture.get_belief("origin_belief_id")

    # Execute propagation with tracking
    result = large_graph_fixture.propagate_from(
        origin.id,
        max_depth=3,
        decay_factor=0.8
    )

    # Verify budget adherence
    assert len(result.depth_updates[0]) <= 8
    assert len(result.depth_updates[1]) <= 5

    # Verify causal updates dominate
    causal_updates = [u for u in result.updated_beliefs if u.update_type == "causal"]
    semantic_updates = [u for u in result.updated_beliefs if u.update_type == "semantic"]

    assert len(causal_updates) > len(semantic_updates)
```

---

## Refactoring Roadmap

### Phase 1: Foundation (Week 1)
**Goal**: Establish proper test infrastructure

**Tasks**:
1. Create `tests/conftest.py` with shared fixtures:
   ```python
   @pytest.fixture
   def empty_graph():
       return JustificationGraph()

   @pytest.fixture
   def small_graph():
       graph = JustificationGraph()
       # Add 5-10 interconnected beliefs
       return graph

   @pytest.fixture
   def large_graph():
       graph = JustificationGraph()
       # Add 50+ beliefs with varied connections
       return graph
   ```

2. Create test data factories:
   ```python
   # tests/factories.py
   class BeliefFactory:
       @staticmethod
       def create(content=None, confidence=0.5, context="test"):
           return Belief(
               content=content or f"Test belief {uuid4()}",
               confidence=confidence,
               context=context
           )
   ```

3. Reorganize test structure:
   ```
   tests/
   ├── conftest.py          # Shared fixtures
   ├── factories.py         # Test data builders
   ├── unit/
   │   ├── test_belief_types.py
   │   ├── test_estimation.py (renamed from root)
   │   └── test_propagation_math.py
   ├── integration/
   │   ├── test_graph_operations.py
   │   └── test_linking.py
   └── e2e/
       ├── test_lifecycle.py
       ├── test_llm_agents.py
       └── test_propagation.py
   ```

### Phase 2: Fill Coverage Gaps (Week 2)
**Goal**: Test untested modules

**New Test Files**:
1. `tests/unit/test_justification_graph.py`:
   - add_belief, remove_belief, get_belief
   - Graph traversal methods
   - Consistency checks

2. `tests/unit/test_propagation_math.py`:
   - calculate_dependency edge cases
   - Logistic saturation function
   - Decay factor application

3. `tests/integration/test_graph_operations.py`:
   - Multi-step workflows
   - State mutations across operations

### Phase 3: E2E Implementation (Week 3)
**Goal**: Implement true end-to-end tests

**Test Files** (as described in "Proposed E2E Testing Strategy" above):
- `tests/e2e/test_lifecycle.py`
- `tests/e2e/test_llm_agents.py`
- `tests/e2e/test_propagation.py`

### Phase 4: Quality & Performance (Week 4)
**Goal**: Add non-functional tests

**New Capabilities**:
1. **Performance tests**:
   ```python
   # tests/performance/test_scaling.py
   @pytest.mark.benchmark
   def test_propagation_scales_linearly(benchmark):
       graph = create_graph_with_n_beliefs(1000)
       result = benchmark(graph.propagate_from, "root_id")
       assert result.elapsed_time < 1.0  # seconds
   ```

2. **Property-based tests** (using Hypothesis):
   ```python
   from hypothesis import given, strategies as st

   @given(confidence=st.floats(min_value=-1, max_value=1))
   def test_confidence_always_in_bounds(confidence):
       belief = Belief("test", confidence, "test")
       assert -1 <= belief.confidence <= 1
   ```

3. **Snapshot tests** (for graph state):
   ```python
   def test_propagation_produces_expected_graph(snapshot):
       graph = setup_test_scenario()
       graph.propagate_from("origin")
       snapshot.assert_match(graph.to_dict())
   ```

---

## Migration Strategy

### Backward Compatibility

**Existing tests must continue to pass throughout migration.**

**Approach**:
1. Move `tests/test_estimation.py` → `tests/unit/test_estimation.py` (no changes to content)
2. Update imports in test files as needed
3. Add new tests alongside existing ones
4. Run full suite after each phase

### CI/CD Integration

**GitHub Actions workflow** (`.github/workflows/test.yml`):
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run unit tests
        run: pytest tests/unit -v

      - name: Run integration tests
        run: pytest tests/integration -v

      - name: Run E2E tests
        run: pytest tests/e2e -v --tb=short

      - name: Generate coverage report
        run: |
          pip install pytest-cov
          pytest --cov=src/baye --cov-report=html --cov-report=term
```

---

## Success Metrics

### Coverage Targets
- **Unit tests**: ≥ 90% line coverage for core modules
- **Integration tests**: ≥ 80% coverage for graph operations
- **E2E tests**: 100% coverage of user-facing workflows

### Quality Metrics
- **Test execution time**: < 10s for unit, < 30s for integration, < 60s for E2E
- **Flakiness**: 0 flaky tests (all deterministic)
- **Maintainability**: Average test < 50 lines, clear naming

### Timeline
- **Phase 1**: 1 week (foundation)
- **Phase 2**: 1 week (coverage gaps)
- **Phase 3**: 1 week (E2E implementation)
- **Phase 4**: 1 week (quality & performance)

**Total**: 4 weeks to complete refactoring

---

## Appendix: File Structure Reference

```
baye/
├── src/baye/
│   ├── __init__.py
│   ├── belief_types.py         # ✅ Tested
│   ├── belief_estimation.py    # ✅ Tested
│   ├── justification_graph.py  # ❌ Needs tests
│   └── llm_agents.py           # ❌ Needs tests
├── tests/
│   ├── conftest.py             # 🆕 Create
│   ├── factories.py            # 🆕 Create
│   ├── unit/
│   │   ├── test_belief_types.py          # 🆕 Create
│   │   ├── test_estimation.py            # ♻️ Move from root
│   │   ├── test_justification_graph.py   # 🆕 Create
│   │   └── test_propagation_math.py      # 🆕 Create
│   ├── integration/
│   │   ├── test_graph_operations.py      # 🆕 Create
│   │   └── test_linking.py               # 🆕 Create
│   ├── e2e/
│   │   ├── test_lifecycle.py             # 🆕 Create
│   │   ├── test_llm_agents.py            # 🆕 Create
│   │   └── test_propagation.py           # 🆕 Create
│   └── performance/
│       └── test_scaling.py               # 🆕 Create (Phase 4)
├── docs/testing/
│   ├── e2e_strategy.md           # Generic template (keep)
│   ├── baye_e2e_strategy.md      # 🆕 Create (Baye-specific)
│   └── refactoring_plan.md       # This document
└── .github/workflows/
    └── test.yml                  # 🆕 Create CI pipeline
```

---

**Version**: 1.0
**Last Updated**: 2025-11-19
**Status**: Draft - Ready for Review
