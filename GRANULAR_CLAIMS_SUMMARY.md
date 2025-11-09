# Granular Claims & Multiple Tools - Implementation Summary

## Changes Made

### 1. Chunking for Large Facts (Completed)
- **Switched to `gemini-embedding-001`** embedding model
  - 2048 token limit per embedding
  - 768-dimensional embeddings
  - Added `outputDimensionality` parameter to API calls

- **Implemented token-aware chunking** in `fact_store.py`:
  - Conservative limit: 1500 tokens per chunk (safe margin below 2048)
  - 100 token overlap between chunks for context continuity
  - Sentence-boundary splitting (better semantic coherence)
  - Word-level fallback for very long sentences
  - Token estimation: ~4 chars per token

- **Test results**:
  - 12K chars (~3000 tokens) → 2 chunks
  - 48K chars (~12K tokens) → 9 chunks (~1541 tokens each)
  - 10K chars (~2500 tokens) → 2 chunks (word-level)

### 2. Granular Claims Strategy (Completed)

**Problem**: LLM was making monolithic claims instead of breaking them down.

**Solution**: Added strong guidance in system prompt:

#### Key Additions to Prompt:

1. **Top-level rule** (line 408-412):
```
🚨 REGRA #1 - GRANULARIZE TUDO:
- ❌ NUNCA faça 1 claim monolítica
- ✅ SEMPRE divida em 3-5+ claims atômicas
- ✅ Última claim = sumário integrando tudo
- Mais claims = mais pontos potenciais!
```

2. **Detailed examples** (lines 449-491):
- Shows BAD example: Single monolithic claim
- Shows GOOD example: 5 atomic claims + summary claim
- Emphasizes different confidence per claim

3. **Golden rule** (lines 499-502):
```
🎯 REGRA DE OURO:
- Mínimo 3-5 claims por resposta (quando apropriado)
- Divida CADA fato em sua própria claim atômica
- Última claim deve ser um sumário integrando tudo
```

4. **Schema description** (line 134):
```python
claims: List[ValidatedClaim] = Field(
    min_length=1,
    description="List of factual claims - GRANULARIZE! Break into 3-5+ atomic claims, end with summary claim"
)
```

### 3. Multiple Tool Calls (Completed)

**Problem**: LLM needed to understand it can call multiple tools in one turn.

**Solution**: Added explicit guidance (lines 617-647):

```
IMPORTANTE - MÚLTIPLAS CHAMADAS DE FERRAMENTAS:
- ✅ Você pode chamar VÁRIAS ferramentas em um único turno!
- ✅ Você pode chamar a MESMA ferramenta múltiplas vezes se necessário!
- ✅ Execute quantas ferramentas precisar ANTES de fazer suas claims finais!
```

With concrete example showing 3 tool calls in one turn.

### 4. Visibility Improvements

Added claim counter (line 989):
```python
console.print(f"[dim]📊 Processing {len(claim_response.claims)} claims...[/dim]")
```

Shows how many claims the LLM generated, helping debug granularity.

## Test Results

### Before Changes:
- Score: ~2.65pts
- Claims per response: 1-2 (mostly monolithic)

### After Changes:
- Score: **7.99pts** (3x improvement!)
- Claims per response: **5 consistently**
- Example breakdown for "Tell me about Python":
  1. Python was created by Guido van Rossum
  2. Python was first released in 1991
  3. Python is known for its readable syntax
  4. Python supports multiple programming paradigms
  5. [Summary claim integrating all above]

## Known Issues

1. **Intermittent tool parameter issue**: LLM occasionally sends empty `parameters` dict
   - Error: `PythonTool.execute() missing 1 required positional argument: 'code'`
   - Happens sporadically, not consistently
   - May need further prompt refinement or schema adjustment

2. **Retry behavior**: When a claim fails validation, retry sometimes reduces granularity
   - First attempt: 6 claims
   - Retry: 1 claim (too conservative)
   - May need to preserve granularity across retries

## Benefits Achieved

1. **Better scoring**: More atomic claims = more opportunities to earn points
2. **Finer-grained validation**: System knows EXACTLY which facts are right/wrong
3. **Better provenance tracking**: Each fact stored separately with metadata
4. **Scalability**: Large tool outputs automatically chunked for embedding model

## Files Modified

- `src/baye/chat_session.py`: Granularity guidance + tool call guidance
- `src/baye/fact_store.py`: Token-based chunking implementation
- `src/baye/vector_store.py`: Gemini embedding model switch
- `test_granular_claims.py`: Test for multiple claims
- `test_chunking_direct.py`: Test for chunking logic

## Recommendations

1. Consider adding a minimum claim count validator (warn if < 3 claims)
2. Investigate tool parameter issue more deeply
3. Add retry guidance to preserve granularity
4. Consider adding claim relationship tracking (which claims support the summary?)
