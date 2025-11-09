# Justification-Based Belief Tracking System

A neural-symbolic belief maintenance system that combines deterministic causal tracking with probabilistic semantic propagation, powered by LLMs.

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/franklinbaldo/baye.git
cd baye

# 2. Install
uv sync

# 3. Configure API key
export GOOGLE_API_KEY="your-gemini-api-key"

# 4. Run!
./run.sh
# or
uv run python examples/example_llm_integration.py
```

**📖 Complete guide**: [QUICKSTART_EN.md](QUICKSTART_EN.md)

---

## 🎯 Core Concept

Instead of just storing isolated beliefs, the system maintains a **justification graph** where:
- **Nodes**: Beliefs (natural language statements) with probabilistic confidence
- **Edges**: Justification relationships (A supports B, A contradicts C)
- **Propagation**: Changes propagate through the graph via two mechanisms:
  1. **Causal** (deterministic): through explicit justification links
  2. **Semantic** (probabilistic): through content similarity
- **LLM Integration**: Automatic relationship detection and conflict resolution via Gemini

## 🏗️ Architecture

```
baye/
├── src/baye/              # Main package
│   ├── __init__.py        # Public exports
│   ├── belief_types.py    # Core data structures
│   ├── justification_graph.py  # Main engine
│   ├── belief_estimation.py    # Semantic K-NN
│   └── llm_agents.py      # PydanticAI + Gemini agents
├── examples/              # Usage examples
│   ├── example_llm_integration.py
│   └── example_estimation_integrated.py
├── tests/                 # Tests
│   └── test_estimation.py
├── pyproject.toml         # uv config
├── run.sh                 # Quick script
├── QUICKSTART.md          # Step-by-step guide
└── README.md
```

## 💡 Quick Usage

### Mode V1.5: With LLM (Recommended)

```python
from baye import Belief, detect_relationship, resolve_conflict
import asyncio

async def main():
    # Create beliefs
    b1 = Belief(
        content="Third-party services are reliable",
        confidence=0.7,
        context="infrastructure"
    )

    lesson = Belief(
        content="Stripe API returned 500 errors",
        confidence=0.9,
        context="incident"
    )

    # Detect relationship automatically via LLM
    analysis = await detect_relationship(b1, lesson)
    print(f"Relationship: {analysis.relationship}")  # "contradicts"
    print(f"Confidence: {analysis.confidence}")      # 0.70

    # Resolve conflict via LLM
    if analysis.relationship == "contradicts":
        resolution = await resolve_conflict(b1, lesson)
        print(f"Resolved: {resolution.resolved_belief}")
        # "While third-party services are generally reliable,
        #  critical paths like payments need defensive programming"

asyncio.run(main())
```

### Mode V1.0: Manual (without LLM)

```python
from baye import JustificationGraph, Belief

# Create graph
graph = JustificationGraph(max_depth=4)

# Add beliefs manually
b1 = graph.add_belief(
    content="APIs can fail unexpectedly",
    confidence=0.6,
    context="api_reliability"
)

b2 = graph.add_belief(
    content="Always validate API responses",
    confidence=0.7,
    context="best_practices",
    supported_by=[b1.id]
)

# Propagate changes
result = graph.propagate_from(origin_id=b1.id)
print(f"Updated {result.total_beliefs_updated} beliefs")
```

## 📊 Complete Example

Run the LLM example (requires API key):

```bash
export GOOGLE_API_KEY="your-key"
uv run python examples/example_llm_integration.py
```

**Expected output:**
```
🧠 Belief Tracking with PydanticAI + Gemini
======================================================================

📖 Scenario: Stripe API Failure

Initial beliefs:
  B1: Third-party payment services are generally reliable (conf: 0.7)
  B2: Always validate and handle API responses gracefully (conf: 0.6)
  B3: Established services like Stripe don't need defensive programming (conf: 0.4)

💥 Incident: Stripe API returned 500 errors during checkout flow

🔍 Step 1: Detecting relationships with existing beliefs...

  • CONTRADICTS B1
    Confidence: 0.70
    → Third-party payment services are generally reliable...

  • SUPPORTS B2
    Confidence: 0.70
    → Always validate and handle API responses gracefully...

🤝 Step 3: Resolving contradiction between lesson and B1...

  Resolved Belief:
    "While third-party payment services are generally reliable, specific
     incidents like Stripe API returning 500 errors can occur and severely
     impact revenue. Robust error handling and monitoring are essential."

  Confidence: 0.80
```

## 🔑 Key Concepts

### 1. LLM-Powered Relationship Detection

Uses Gemini via PydanticAI to automatically detect if beliefs:
- **SUPPORT**: One provides evidence for the other
- **CONTRADICT**: Cannot both be true simultaneously
- **REFINE**: One is a more specific version of the other
- **UNRELATED**: No significant logical connection

### 2. Conflict Resolution

When beliefs contradict, the LLM generates a nuanced belief that:
- Acknowledges valid aspects of both
- Identifies conditions where each applies
- Provides balanced and actionable synthesis

### 3. Structured Outputs

All agents return validated Pydantic models:
```python
class RelationshipAnalysis(BaseModel):
    relationship: Literal["supports", "contradicts", "refines", "unrelated"]
    confidence: float
    explanation: str

class ConflictResolution(BaseModel):
    resolved_belief: str
    confidence: float
    reasoning: str
    supports_first: bool
    supports_second: bool
```

## 🛣️ Roadmap

### V1.0-minimal ✅
- [x] Basic causal graph
- [x] Deterministic propagation
- [x] Cycle detection
- [x] Working Stripe test

### V1.5 (LLM Integration) ✅ **COMPLETED**
- [x] Relationship discovery via LLM (PydanticAI + Gemini)
- [x] Automatic conflict resolution via LLM
- [x] Structured outputs with Pydantic models
- [x] Batch relationship detection
- [x] src/baye/ organization
- [x] QUICKSTART.md and run.sh
- [ ] Bidirectional propagation (next)
- [ ] Real embeddings via Gemini (next)

### V2.0 (Scalability) 🎯
- [ ] Persistence (Neo4j + vector DB)
- [ ] Batch propagation (multiple lessons)
- [ ] Visualization dashboard (NetworkX + Plotly)
- [ ] REST API for integration

### V2.5 (Intelligence) 🧠
- [ ] Edge weight learning
- [ ] Meta-beliefs ("trust security beliefs more")
- [ ] Temporal decay (old beliefs lose relevance)
- [ ] Active learning (system requests clarification when uncertain)

## 📚 API Reference

### Core Types

```python
from baye import Belief, BeliefID, Confidence, RelationType

# Create belief
belief = Belief(
    content="APIs can fail",
    confidence=0.8,
    context="reliability"
)

# Update confidence
belief.update_confidence(delta=0.1)  # Increases to 0.9
```

### LLM Agents

```python
from baye import (
    detect_relationship,
    resolve_conflict,
    find_related_beliefs,
    check_gemini_api_key
)

# Check API key
check_gemini_api_key()  # Raises ValueError if not configured

# Detect relationship
analysis = await detect_relationship(belief1, belief2)

# Resolve conflict
resolution = await resolve_conflict(belief1, belief2, context="optional")

# Find related beliefs in batch
relationships = await find_related_beliefs(
    new_belief,
    existing_beliefs,
    min_confidence=0.7
)
```

### Graph Operations

```python
from baye import JustificationGraph

graph = JustificationGraph(max_depth=4)

# Add belief
b = graph.add_belief(content="...", confidence=0.7)

# Link beliefs
graph.link_beliefs(parent_id, child_id, relation=RelationType.SUPPORTS)

# Propagate changes
result = graph.propagate_from(origin_id=b.id)
print(f"Updated: {result.total_beliefs_updated}")
print(f"Max depth: {result.max_depth_reached}")
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/

# Specific test
uv run pytest tests/test_estimation.py -v

# With coverage
uv run pytest --cov=src/baye tests/
```

## 🤝 Contributing

Priority areas:
1. **Real embeddings**: Integrate Gemini Embeddings API
2. **Bidirectional propagation**: Supporters should also be updated
3. **Visualization**: Interactive dashboard
4. **Benchmarks**: Agent failure datasets

## 📄 License

MIT License - use freely in commercial or academic projects.

## 🙏 Acknowledgments

Inspired by discussions on Truth Maintenance Systems (TMS), Bayesian program learning, and autonomous agent architectures.

---

**Status**: V1.5 (LLM Integration) ✅ COMPLETED
**Next**: V2.0 (real embeddings + bidirectional propagation)
**Author**: Franklin Baldo ([@franklinbaldo](https://github.com/franklinbaldo))
