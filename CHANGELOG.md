# Changelog - Justification-Based Belief Tracking System

## V1.5 - Semantic Confidence Estimation (2025-11-08)

### ✨ New Features

#### K-NN Confidence Estimation
**Solves the cold-start problem**: New beliefs can now automatically infer appropriate confidence based on semantic similarity to existing beliefs.

**What it does:**
- Finds K nearest semantic neighbors using content similarity
- Calculates weighted average confidence based on similarity scores
- Applies dampening to prevent overfitting to near-duplicates
- Provides uncertainty estimates based on neighbor variance

**Key Components:**
- `belief_estimation.py`: Core estimation engine
  - `SemanticEstimator`: K-NN confidence calculator
  - `BeliefInitializer`: High-level interface with fallback strategies
  - Utility functions for common patterns

- `justification_graph.py` (updated):
  - `add_belief_with_estimation()`: Add belief with auto-estimated confidence
  - `batch_add_beliefs_with_estimation()`: Batch processing with estimation
  - Auto-linking to semantic neighbors

**Example Usage:**
```python
# Before V1.5: Manual confidence required
belief = graph.add_belief("APIs can timeout", confidence=0.7)

# V1.5+: Automatic estimation
belief = graph.add_belief_with_estimation(
    "APIs can timeout",
    context="infrastructure",
    k=5,  # Use 5 nearest neighbors
    auto_link=True  # Auto-link to similar beliefs
)
# Confidence automatically estimated as 0.68 based on similar beliefs
```

**Advanced Features:**
```python
from belief_estimation import SemanticEstimator

estimator = SemanticEstimator(
    similarity_threshold=0.2,  # Minimum similarity to consider
    dampening_factor=0.9       # Attenuate extreme similarities
)

# Get confidence with uncertainty
conf, uncertainty, ids = estimator.estimate_with_uncertainty(
    "New belief content",
    existing_beliefs,
    k=5
)

print(f"Confidence: {conf:.2f} ± {uncertainty:.2f}")
# Output: Confidence: 0.65 ± 0.12
```

### 🔧 Improvements

#### Enhanced Similarity Function
- Upgraded from basic Jaccard to enhanced version with:
  - Stop word filtering ("the", "a", "is", etc.)
  - Overlap ratio boosting for short phrases
  - Weighted combination (70% Jaccard, 30% overlap)
  
- Adjusted default threshold from 0.4 → 0.2 for better recall

#### Fallback Strategies
`BeliefInitializer` provides three automatic strategies:
1. **K-NN**: Use estimation when uncertainty is low
2. **Conservative**: Shrink estimate toward zero when uncertainty is high
3. **Default**: Use provided default when no neighbors found

```python
initializer = BeliefInitializer(estimator)

conf, strategy = initializer.initialize_with_strategy(
    "New belief",
    existing_beliefs,
    default_confidence=0.5,
    uncertainty_threshold=0.7
)

# Returns: (0.65, "knn") or (0.45, "conservative") or (0.5, "default")
```

### 📊 Validation

**Test Coverage:**
- 9 comprehensive tests covering:
  - Basic K-NN estimation
  - Low/negative confidence propagation
  - Uncertainty calculation
  - Threshold filtering
  - Dampening effects
  - Fallback strategies
  - Edge cases (empty sets, single beliefs, etc.)

**All tests passing:** ✅ 9/9

### 📈 Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Estimation (mock similarity) | O(N) | Linear scan of existing beliefs |
| Estimation (real embeddings) | O(N log N) | With vector index |
| Batch estimation (M beliefs) | O(M × N) | Can be parallelized |

**Memory:**
- No additional storage required
- Estimation is stateless (computed on-demand)

### 🎯 Use Cases Enabled

1. **Agent Learning Loop:**
```python
# After task failure
lesson = extract_lesson_from_failure(task_result)

# No need to guess confidence
belief = graph.add_belief_with_estimation(
    lesson,
    context=task_result["domain"]
)

graph.propagate_from(belief.id)
```

2. **Bulk Initialization:**
```python
# Initialize belief system from text corpus
belief_statements = [
    ("APIs can fail", "infrastructure"),
    ("Validate input", "security"),
    # ... 100 more statements
]

ids = graph.batch_add_beliefs_with_estimation(belief_statements)
# All confidences automatically estimated
```

3. **Uncertainty-Aware Decision Making:**
```python
conf, uncertainty, _ = estimator.estimate_with_uncertainty(
    "Should I trust this API?",
    graph.beliefs.values()
)

if uncertainty > 0.7:
    # High uncertainty - request human feedback
    human_confidence = ask_human(belief_content)
    belief = graph.add_belief(content, human_confidence)
else:
    # Low uncertainty - trust estimation
    belief = graph.add_belief(content, conf)
```

### 📝 Documentation

**New Files:**
- `belief_estimation.py` - Core estimation logic (380 lines)
- `test_estimation.py` - Comprehensive test suite (450 lines)
- `example_estimation_integrated.py` - Full demonstration (220 lines)

**Updated Files:**
- `justification_graph.py` - Added estimation methods (60 lines added)
- `README.md` - Updated with V1.5 features

### 🚀 Migration Guide

**V1.0 → V1.5:**

No breaking changes! All V1.0 code continues to work.

To use new features:
```python
# Old way (still works)
belief = graph.add_belief("Content", 0.7, "context")

# New way (optional)
belief = graph.add_belief_with_estimation("Content", "context")
```

### 🔮 Next Steps (V2.0)

With estimation now working, the next priorities are:

1. **Real Embeddings**: Replace Jaccard with sentence-transformers
2. **LLM Integration**: Automatic relationship detection
3. **Bidirectional Propagation**: Support evidence → hypothesis updates
4. **Persistence**: Neo4j + Chroma for large-scale graphs

---

## V1.0-minimal (2025-11-08)

Initial release with:
- Justification graph with causal propagation
- Belief tracking with confidence
- Cycle detection
- Graph analysis tools
- Stripe API failure test scenario

See README.md for full V1.0 documentation.
