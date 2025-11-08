# 🏗️ Arquitetura do Sistema V1.5

## Visão Geral

```
┌─────────────────────────────────────────────────────────────────┐
│                      Justification Graph                         │
│                     (justification_graph.py)                     │
│                                                                   │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────┐           │
│  │  add_belief│  │   link_    │  │   propagate_    │           │
│  │            │  │  beliefs   │  │      from       │           │
│  └─────┬──────┘  └──────┬─────┘  └────────┬────────┘           │
│        │                 │                  │                    │
│        ▼                 ▼                  ▼                    │
│  ┌──────────────────────────────────────────────────┐          │
│  │           Belief Storage (Dict + NetworkX)        │          │
│  │    {id → Belief(content, conf, supporters, ...)} │          │
│  └──────────────────────────────────────────────────┘          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
           ┌────────▼────────┐      ┌───────▼────────┐
           │  Propagation    │      │   Estimation   │
           │   Strategies    │      │     Engine     │
           │ (V1.0)          │      │   (V1.5 NEW)   │
           └────────┬────────┘      └───────┬────────┘
                    │                       │
    ┌───────────────┼────────────┐         │
    │               │            │         │
┌───▼────┐  ┌──────▼─────┐  ┌──▼─▼────┐  │
│ Causal │  │ Semantic   │  │ Conflict│  │
│Propag. │  │ Propagator │  │Resolver │  │
└────────┘  └────────────┘  └─────────┘  │
                                          │
                         ┌────────────────▼───────────────┐
                         │   SemanticEstimator (K-NN)     │
                         │  ┌──────────────────────────┐  │
                         │  │ 1. Find K neighbors      │  │
                         │  │ 2. Weight by similarity  │  │
                         │  │ 3. Calculate conf        │  │
                         │  │ 4. Measure uncertainty   │  │
                         │  └──────────────────────────┘  │
                         └────────────────────────────────┘
```

## Data Flow: Adding a Belief with Estimation

```
User Request
     │
     ▼
┌─────────────────────────────────────────┐
│ add_belief_with_estimation()            │
│  "APIs and services can timeout"        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ SemanticEstimator.estimate_confidence() │
│                                         │
│  ┌────────────────────────────────┐    │
│  │ For each existing belief:      │    │
│  │   similarity = jaccard(new,    │    │
│  │                        existing) │    │
│  │   if sim > threshold:          │    │
│  │     neighbors.append()         │    │
│  └────────────────────────────────┘    │
│                                         │
│  Top K neighbors by similarity          │
│    ↓                                    │
│  ["External APIs..." (0.7, sim=0.71)]  │
│  ["Network calls..." (0.6, sim=0.59)]  │
│                                         │
│  Weighted average:                      │
│    conf = Σ(sim_i × conf_i) / Σ(sim_i)│
│         = (0.71×0.7 + 0.59×0.6) / 1.3  │
│         = 0.68                         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ create_belief()                         │
│   Belief(                               │
│     content="APIs and services...",     │
│     confidence=0.68,  ← ESTIMATED       │
│     context="infrastructure"            │
│   )                                     │
└─────────────────┬───────────────────────┘
                  │
                  ▼ (if auto_link=True)
┌─────────────────────────────────────────┐
│ link_to_neighbors()                     │
│   For each neighbor with sim > 0.7:    │
│     neighbor → new_belief               │
└─────────────────────────────────────────┘
```

## Propagation Flow (V1.0 + V1.5)

```
Belief Update Event
        │
        ▼
┌────────────────────┐
│ propagate_from()   │
│   origin_id        │
│   initial_delta    │
└──────────┬─────────┘
           │
    ┌──────┴───────┐
    │ Visited Set  │ ← Prevents cycles
    └──────┬───────┘
           │
           ▼
┌──────────────────────────────────┐
│ _propagate_recursive(depth=0)   │
│                                  │
│  1. Check termination:           │
│     - depth >= max_depth?        │
│     - already visited?           │
│     - delta < threshold?         │
│                                  │
│  2. Get propagation budget       │
│     budget[0]=8, [1]=5, ...      │
│                                  │
│  3. Calculate updates:           │
│     ┌──────────────┐             │
│     │ Causal       │ (70% weight)│
│     │ Propagation  │             │
│     └──────┬───────┘             │
│            │                     │
│     ┌──────▼───────┐             │
│     │ Semantic     │ (30% weight)│
│     │ Propagation  │             │
│     └──────┬───────┘             │
│            │                     │
│            ▼                     │
│     Merge & Sort by |delta|      │
│            │                     │
│     Take top [budget] updates    │
│                                  │
│  4. Apply updates:               │
│     For each child:              │
│       old_conf → new_conf        │
│       Record event               │
│                                  │
│  5. Recurse if |delta| significant│
│     _propagate_recursive(depth+1)│
└──────────────────────────────────┘
```

## Key Algorithms

### 1. Dependency Calculation (Causal)

```python
def calculate_dependency(child, parent):
    """
    How much does child depend on parent?
    
    Returns float in [0, 1]
    """
    # Base: equal split among supporters
    base = 1.0 / len(child.supporters)
    
    # Logistic saturation prevents explosion
    parent_influence = logistic(parent.conf)
    total_influence = sum(logistic(s.conf) 
                          for s in child.supporters)
    
    return base * (parent_influence / total_influence)

# Logistic: 1 / (1 + e^(-k(x - mid)))
# At conf=0.9: influence plateaus (saturation)
```

### 2. K-NN Estimation (V1.5)

```python
def estimate_confidence(new_content, beliefs, k=5):
    """
    Estimate confidence for new belief.
    
    Returns (confidence, neighbor_ids, similarities)
    """
    # 1. Calculate similarities
    similarities = []
    for b in beliefs:
        sim = jaccard_enhanced(new_content, b.content)
        if sim > 0.9:
            sim = 0.9 + (sim - 0.9) * dampening  # Attenuate
        if sim >= threshold:
            similarities.append((b, sim))
    
    # 2. Sort and take top-K
    top_k = sorted(similarities, 
                   key=lambda x: x[1], 
                   reverse=True)[:k]
    
    # 3. Weighted average
    conf = sum(b.conf * sim for b, sim in top_k) / \
           sum(sim for _, sim in top_k)
    
    return conf, [b.id for b, _ in top_k], [sim for _, sim in top_k]
```

### 3. Uncertainty Calculation

```python
def estimate_with_uncertainty(new_content, beliefs, k=5):
    """
    Returns (confidence, uncertainty, ids)
    """
    conf, ids, sims = estimate_confidence(...)
    
    # Variance in neighbor confidences
    conf_var = variance([b.conf for b in neighbors])
    
    # Variance in similarities (spread)
    sim_var = variance(sims)
    
    # Penalty for small sample
    sample_penalty = (k - len(ids)) / k
    
    # Combined
    uncertainty = 0.5 * conf_var + \
                  0.3 * sim_var + \
                  0.2 * sample_penalty
    
    return conf, min(uncertainty, 1.0), ids
```

## Module Dependencies

```
belief_types.py (no deps)
     ↓
justification_graph.py
     ↓ uses
propagation_strategies.py
     ↓ uses
belief_estimation.py (new V1.5)
     ↓
[all use belief_types]
```

## Storage Model

### In-Memory (V1.0-1.5)

```python
class JustificationGraph:
    beliefs: Dict[str, Belief]  # O(1) lookup
    nx_graph: nx.DiGraph        # For graph algorithms
    propagation_history: List[PropagationResult]
```

### Future: Persistent (V2.0)

```
┌────────────────────────────────┐
│   Neo4j (Graph Structure)      │
│                                │
│  (Belief)-[SUPPORTS]->(Belief) │
│  (Belief)-[CONTRADICTS]-(...)  │
└────────────────────────────────┘
          ↕
┌────────────────────────────────┐
│  Chroma/FAISS (Vector Search)  │
│                                │
│  embeddings[belief_id] = vec   │
│  K-NN query in O(log N)        │
└────────────────────────────────┘
          ↕
┌────────────────────────────────┐
│    SQLite (Metadata)           │
│                                │
│  events, statistics, history   │
└────────────────────────────────┘
```

## Performance Characteristics

| Operation | V1.0 (mock) | V1.5 (mock) | V2.0 (target) |
|-----------|-------------|-------------|---------------|
| Add belief (manual) | O(1) | O(1) | O(1) |
| Add belief (estimated) | - | O(N) | O(log N) |
| Propagate (depth D) | O(E × D) | O(E × D) | O(E × D) |
| Find similar | - | O(N) | O(log N) |
| Batch add (M beliefs) | O(M) | O(M × N) | O(M × log N) |

Where:
- N = number of beliefs
- E = number of edges
- D = propagation depth
- M = batch size

## Extension Points (V2.0)

```
┌─────────────────────────────────────────┐
│   LLM Integration Layer (future)        │
│                                         │
│  ┌────────────────────────────────┐    │
│  │ Relationship Detector          │    │
│  │   "Is A a justification for B?"│    │
│  │   → supports / contradicts /   │    │
│  │     refines / independent      │    │
│  └────────────────────────────────┘    │
│                                         │
│  ┌────────────────────────────────┐    │
│  │ Nuance Generator               │    │
│  │   Conflicting beliefs →        │    │
│  │   Generate contextual refinement│    │
│  └────────────────────────────────┘    │
│                                         │
│  ┌────────────────────────────────┐    │
│  │ Evidence Scorer                │    │
│  │   Rate evidence strength for   │    │
│  │   Bayesian updates             │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Metrics & Observability

```python
# Graph health
consistency_score = calculate_belief_consistency()
# Beliefs should have conf ≤ avg(supporters)

unstable = identify_unstable_beliefs()
# High conf but weak/no support

centrality = get_centrality_scores()
# PageRank-style importance

cycles = detect_cycles()
# Circular justifications

# Estimation quality
estimation_error = actual_conf - estimated_conf
calibration = correlation(uncertainties, errors)
```

---

**Version**: 1.5  
**Last Updated**: 2025-11-08  
**Status**: Production-ready ✅
