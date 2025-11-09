# Facts System - Ground Truth for Belief Validation

## Conceito

**Fato**: Todo conteúdo que o modelo teve acesso e foi salvo com proveniência rastreável.

### Diferença: Facts vs Beliefs

| **Facts** | **Beliefs** |
|-----------|-------------|
| Ground truth observado | Inferências/hipóteses |
| Confidence = 1.0 (ou derivado de fonte) | Confidence variável (0-1) |
| Imutável (após criação) | Mutável (update-on-use) |
| Proveniência rastreável (UUID) | Derivado de raciocínio |
| Recuperado por similaridade (vector store) | Propagado por grafo |

**Exemplo**:
```
Fact: "Donald Trump assumiu presidência dos EUA em 20/01/2025"
  → source: "user_message_abc123"
  → confidence: 1.0
  → embedding: [0.1, 0.2, ...]

Belief: "Transições presidenciais causam volatilidade política"
  → confidence: 0.75 (inferido)
  → supported_by: [fact_abc123]
```

## Arquitetura

### 1. Fact Model

```python
@dataclass
class Fact:
    """Ground truth fact with provenance"""
    id: str  # UUID
    content: str  # The factual statement
    source_type: str  # "user_message", "document", "api", "web"
    source_id: str  # UUID of the source
    timestamp: datetime
    confidence: float = 1.0  # Can be < 1.0 for uncertain sources
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # Semantic embedding
```

### 2. Vector Store

**Storage**: In-memory vector store (numpy-based) com opção de persistência

```python
class FactStore:
    """Vector store for facts with semantic retrieval"""

    def __init__(self):
        self.facts: Dict[str, Fact] = {}  # id → Fact
        self.embeddings: np.ndarray = None  # (n_facts, embedding_dim)
        self.fact_ids: List[str] = []  # Index → fact_id mapping

    def add_fact(self, content: str, source_type: str, source_id: str) -> Fact:
        """Add fact and generate embedding"""
        ...

    def find_similar(self, query: str, k: int = 5) -> List[Tuple[Fact, float]]:
        """Find k most similar facts (cosine similarity)"""
        ...

    def verify_claim(self, claim: str, threshold: float = 0.8) -> Optional[Fact]:
        """Check if claim matches a known fact"""
        ...
```

### 3. Provenance Chain

```
User Message (UUID: msg_abc123)
  ↓
Fact Extraction (automatic or explicit)
  ↓
Fact (UUID: fact_def456, source: msg_abc123)
  ↓
Vector Store (embedding + metadata)
  ↓
Claim Validation (similarity search)
```

### 4. Integration with Claim Validation

**New Validation Flow**:

```python
async def _validate_claim(self, claim: ValidatedClaim) -> ClaimValidationStep:
    """
    Validate claim against facts first, then beliefs

    1. Check facts: Does this claim match a known fact?
       → If yes: Use fact's confidence (usually 1.0)
       → If similar but different: Flag as potential conflict

    2. If no fact match: Check beliefs (K-NN as before)

    3. Calculate error: claim.estimate vs (fact OR belief)
    """

    # Step 1: Check facts
    matching_fact = self.fact_store.verify_claim(claim.content)
    if matching_fact:
        actual = matching_fact.confidence
        source = f"fact:{matching_fact.id[:8]}"
        margin = 0.05  # Tighter margin for facts
    else:
        # Step 2: Fallback to beliefs (existing K-NN logic)
        belief = await self._get_or_create_belief_for_claim(claim)
        actual = belief.confidence
        source = f"belief:{belief.id[:8]}"
        margin = self._get_margin(belief.id)

    # Step 3: Calculate error
    error = actual - claim.confidence_estimate
    ...
```

## Fact Extraction

### Automatic Extraction from Context

```python
class FactExtractor:
    """Extract facts from user messages and documents"""

    async def extract_from_message(self, message: str, message_id: str) -> List[Fact]:
        """
        Use LLM to extract factual statements from message

        Example:
        User: "Li no jornal que Trump venceu a eleição em novembro de 2024"

        Facts extracted:
        1. "Donald Trump venceu eleição presidencial dos EUA"
           - source: user_message_xyz
           - confidence: 0.8 (secondary source)

        2. "Eleição presidencial dos EUA ocorreu em novembro de 2024"
           - source: user_message_xyz
           - confidence: 1.0 (temporal fact)
        """
        ...
```

### Manual Fact Addition

```python
# CLI command
/addfact "Donald Trump é presidente dos EUA desde janeiro 2025"

# API
session.add_fact(
    content="Donald Trump é presidente dos EUA desde janeiro 2025",
    source_type="manual",
    source_id="user_input"
)
```

## Validation Examples

### Example 1: Claim Matches Fact

```
User added fact: "Trump assumiu presidência em 20/01/2025"

User: "quem é presidente dos EUA?"
LLM Claim: "Donald Trump é presidente dos EUA" (confidence: 0.9)

Validation:
  → Fact found: "Trump assumiu presidência em 20/01/2025" (similarity: 0.92)
  → Actual confidence: 1.0 (from fact)
  → Error: 1.0 - 0.9 = +0.1
  → Within margin: ✓ (error < 0.05 for facts? Maybe relax to 0.15)

Result: ✅ Claim validated against fact
```

### Example 2: Claim Conflicts with Fact

```
Fact: "Trump assumiu presidência em 20/01/2025"

User: "Biden ainda é presidente?"
LLM Claim: "Joe Biden é presidente dos EUA" (confidence: 0.85)

Validation:
  → Fact found: "Trump assumiu presidência..." (semantic conflict detected)
  → Actual: 0.0 (conflict with fact)
  → Error: 0.0 - 0.85 = -0.85
  → OUTSIDE MARGIN

Result: ❌ Claim conflicts with known fact!

Error Message:
```
❌ CLAIM CONFLICTS WITH FACT

Your claim: "Joe Biden é presidente dos EUA" (confidence: 0.85)
Known fact: "Trump assumiu presidência em 20/01/2025"
  Source: user_message (2025-01-20)
  Conflict: These statements contradict

Your claim appears outdated or incorrect based on known facts.
```
```

### Example 3: No Fact, Use Belief

```
No facts about "PostgreSQL performance"

User: "PostgreSQL é rápido para consultas complexas?"
LLM Claim: "PostgreSQL tem boa performance para queries complexas" (conf: 0.75)

Validation:
  → No matching fact
  → Fallback to belief K-NN
  → Actual: 0.70 (from similar beliefs)
  → Error: 0.70 - 0.75 = -0.05
  → Within margin: ✓

Result: ✅ Claim validated against beliefs (no facts available)
```

## CLI Commands

### New Commands

```bash
# List facts
/facts [N]

# Show fact details
/fact <id>

# Add fact manually
/addfact "<content>"

# Search facts
/searchfacts "<query>"

# Import facts from file
/importfacts <path>
```

### Example Session

```
You: /addfact "Trump assumiu presidência dos EUA em 20/01/2025"
✓ Fact added: fact_abc123

You: quem é presidente dos EUA?

🤖 Assistant (claim-based):
Donald Trump é o presidente dos EUA desde janeiro de 2025.

Claims validated:
  ✓ "Donald Trump é presidente dos EUA" [0.90 → 1.00 (fact), err: +0.10]
    Source: fact_abc123
```

## Implementation Plan

### Phase 1: Core Fact System
1. Create `Fact` dataclass
2. Implement `FactStore` with in-memory vectors
3. Simple embedding generation (use existing SemanticEstimator)
4. Basic CRUD operations

### Phase 2: Integration with Validation
1. Modify `_validate_claim()` to check facts first
2. Add fact-vs-belief priority logic
3. Update error messages to show fact provenance
4. Tighter margins for fact-based validation

### Phase 3: Fact Extraction
1. LLM-powered fact extraction from messages
2. Automatic fact creation from user statements
3. Confidence scoring for secondary sources
4. Fact deduplication (semantic similarity check)

### Phase 4: CLI & UX
1. `/facts`, `/fact`, `/addfact` commands
2. Fact rendering in CLI (table view)
3. Show fact provenance in claim validation
4. Fact conflict warnings

### Phase 5: Persistence (Optional)
1. Save facts to JSON/SQLite
2. Load facts on session start
3. Export/import fact database

## Benefits

1. **Ground Truth**: Claims validated against observed facts, not just inferences
2. **Provenance**: Every fact traceable to source
3. **Conflict Detection**: Automatically flag claims that contradict facts
4. **Learning**: Facts seed belief graph with high-confidence anchors
5. **Debugging**: Clear separation between "what we know" vs "what we believe"

## Edge Cases

### 1. Temporal Facts
```
Fact (2023): "Biden é presidente dos EUA"
Fact (2025): "Trump é presidente dos EUA"

Solution: Facts have timestamps, most recent wins for temporal queries
```

### 2. Contradictory Facts from Different Sources
```
Source A: "Russia invaded Ukraine in 2022"
Source B: "Russia special operation in Ukraine 2022"

Solution: Store both with source confidence, flag conflict for user review
```

### 3. Fact vs Belief Overlap
```
User provides fact: "APIs can timeout"
System already has belief: "APIs can timeout" (confidence: 0.75)

Solution: Upgrade belief to fact, preserve graph connections
```

## Future Extensions

1. **Fact Sources**: Wikipedia, news APIs, documents
2. **Fact Expiration**: Auto-expire temporal facts
3. **Fact Voting**: Multiple sources increase confidence
4. **Fact Chains**: Facts that depend on other facts
5. **Fact Challenges**: User can challenge/dispute facts

---

This system creates a **two-tier epistemology**:
- **Tier 1 (Facts)**: Observed ground truth
- **Tier 2 (Beliefs)**: Inferences and hypotheses

Claims are validated against Tier 1 first (tight margin), falling back to Tier 2 (K-NN with looser margin).
