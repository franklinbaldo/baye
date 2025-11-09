# Proposta: Validação de Claims Individuais

## Problema Atual

O sistema atual tem uma falha fundamental no design:

1. **Extraction separada da Response**: Beliefs são extraídas em uma fase, response em outra
2. **Single active belief**: Apenas 1 belief é validada por resposta, mesmo que a resposta faça múltiplos claims
3. **Sem validação granular**: Não há como o modelo saber QUAL claim específico está fora da margem

### Exemplo do Problema

```
User: "quem é presidente dos EUA?"
Assistant: "Joe Biden é presidente dos EUA"
  → Deveria validar: claim("Biden é presidente", conf=0.95) vs real(0.30) → ERRO
  → Mas valida: belief genérica("modelos não confiáveis", conf=0.70) → OK ❌
```

## Solução: Response como Lista de Claims Validados

### Novo Design

```python
class ValidatedClaim(BaseModel):
    """Single factual claim that needs validation"""
    content: str = Field(description="The claim/assertion being made")
    confidence_estimate: float = Field(
        ge=0.0, le=1.0,
        description="Your precise confidence in this claim"
    )
    context: str = Field(
        default="general",
        description="Domain/context of this claim"
    )

class ClaimBasedResponse(BaseModel):
    """Response structured as list of validated claims"""
    claims: List[ValidatedClaim] = Field(
        min_length=1,
        description="List of factual claims in your response"
    )
    response_text: str = Field(
        description="Natural language response to user (can reference claims)"
    )
```

### Fluxo de Validação

```
1. LLM generates ClaimBasedResponse
   ↓
2. For each claim:
   - Check if belief exists for this claim
   - If not: create belief with K-NN estimation
   - Get margin from K-NN
   - Compare estimate vs real
   ↓
3. If ANY claim is outside margin:
   → ERRO specific to that claim
   → Show error to user AND LLM
   → LLM can adjust in next response
   ↓
4. If all claims within margin:
   → Render response to user
```

### Exemplo Completo

**User**: "quem é presidente dos EUA?"

**LLM Response** (attempt 1):
```json
{
  "claims": [
    {
      "content": "Joe Biden é o presidente dos EUA em 2025",
      "confidence_estimate": 0.95,
      "context": "politics"
    }
  ],
  "response_text": "O presidente dos EUA é Joe Biden."
}
```

**System Validation**:
```
Claim 1: "Joe Biden é o presidente dos EUA em 2025"
  - K-NN estimate: 0.30 (neighbors: Trump win beliefs)
  - Margin: ±0.15
  - LLM estimate: 0.95
  - Error: -0.65 (OUTSIDE MARGIN!)

❌ ERRO DE CALIBRAÇÃO:
Claim: "Joe Biden é o presidente dos EUA em 2025"
Seu palpite: 0.95
Valor estimado (K-NN): 0.30
Margem: ±0.15
Erro: -0.65

Use delta=0 AGORA. Na próxima resposta você pode ajustar.
```

**LLM Response** (attempt 2 - after seeing error):
```json
{
  "claims": [
    {
      "content": "Joe Biden foi o presidente dos EUA até janeiro de 2025",
      "confidence_estimate": 0.85,
      "context": "politics"
    },
    {
      "content": "Donald Trump assumiu a presidência em janeiro de 2025",
      "confidence_estimate": 0.30,
      "context": "politics"
    }
  ],
  "response_text": "Na verdade, houve uma mudança recente. Donald Trump assumiu a presidência em janeiro de 2025. Joe Biden foi o presidente até então."
}
```

**System Validation**:
```
Claim 1: Biden até jan/2025
  - K-NN: 0.90, LLM: 0.85, Error: -0.05 ✓ OK

Claim 2: Trump desde jan/2025
  - K-NN: 0.35, LLM: 0.30, Error: -0.05 ✓ OK

✅ All claims validated, response rendered.
```

## Benefícios

1. **Calibração Granular**: Erro mostrado para claim específico
2. **Multi-Claim Responses**: Pode fazer várias afirmações, cada uma validada
3. **Self-Correction**: LLM vê exatamente qual claim falhou
4. **User Transparency**: Usuário vê qual parte da resposta foi incerta
5. **Better Meta-Learning**: Training signals por claim, não por resposta inteira

## Mudanças Necessárias

### 1. Remover Dual-Agent Architecture

Atualmente:
- `extraction_agent`: Extrai beliefs
- `response_agent`: Gera resposta

Novo:
- `response_agent`: Gera resposta **como lista de claims**
- Claims são automaticamente processados como beliefs

### 2. Novo System Prompt

```python
CLAIM_BASED_PROMPT = """
Você é um assistente prestativo com validação epistêmica de claims.

**Sua resposta DEVE ser estruturada como claims validados**:

{
  "claims": [
    {
      "content": "afirmação factual específica",
      "confidence_estimate": 0.73,  // PRECISE estimate
      "context": "domain"
    },
    ...
  ],
  "response_text": "resposta natural ao usuário"
}

**REGRAS**:
1. Divida sua resposta em claims factuais específicos
2. Cada claim é uma afirmação que pode ser verdadeira/falsa
3. Estime confidence PRECISA para cada claim (0.73, não 0.7)
4. Se você não tem certeza, use confidence baixa (0.3-0.5)
5. Claims são validados INDIVIDUALMENTE

**Exemplos de bons claims**:
✓ "Python usa indentação para blocos de código"
✓ "O comando git status mostra mudanças não commitadas"
✓ "PostgreSQL é um banco de dados relacional"

**Exemplos de claims ruins (muito vagos)**:
✗ "Programação é importante"
✗ "Bancos de dados armazenam dados"
✗ "Git é útil"

Se algum claim estiver fora da margem de confiança, você verá um erro.
Use esse erro para calibrar suas próximas respostas!
"""
```

### 3. Novo `_process_claims()` Method

```python
async def _process_claims(
    self,
    response: ClaimBasedResponse
) -> ClaimValidationResult:
    """
    Validate each claim individually

    Returns error if ANY claim is outside margin
    """
    validated_claims = []

    for claim in response.claims:
        # Get or create belief for this claim
        belief = await self._get_or_create_belief_for_claim(claim)

        # Get K-NN margin
        margin = self._get_margin(belief.id)

        # Calculate error
        actual = belief.confidence
        error = actual - claim.confidence_estimate

        # Check if outside margin
        if abs(error) > margin:
            raise ClaimCalibrationError(
                claim=claim.content,
                estimate=claim.confidence_estimate,
                actual=actual,
                margin=margin,
                error=error
            )

        validated_claims.append(...)

    return ClaimValidationResult(claims=validated_claims)
```

### 4. UI Changes

**Before** (single belief footer):
```
Belief abc123 | Palpite: 0.75 | Real: 0.80 | Erro: +0.05
```

**After** (per-claim validation):
```
Claims validated:
  ✓ "Biden foi presidente até jan/2025" [0.85 → 0.90, err: +0.05]
  ✓ "Trump assumiu em jan/2025" [0.30 → 0.35, err: +0.05]
```

**Error display**:
```
❌ CLAIM VALIDATION ERROR

Claim: "Joe Biden é presidente dos EUA em 2025"
Your estimate: 0.95
K-NN estimate: 0.30 ±0.15
Error: -0.65 (OUTSIDE MARGIN)

This claim failed validation. Revise and try again!
```

## Implementation Plan

1. ✅ Create this proposal document
2. ⬜ Define new Pydantic models (ValidatedClaim, ClaimBasedResponse)
3. ⬜ Remove extraction_agent, keep only response_agent
4. ⬜ Update system prompt to claim-based structure
5. ⬜ Implement `_process_claims()` with per-claim validation
6. ⬜ Update CLI to show per-claim validation
7. ⬜ Add ClaimCalibrationError exception
8. ⬜ Test with the failing conversation (Biden/Trump example)
9. ⬜ Update CHAT_CLI_README.md with new architecture

## Compatibility

- Keep `/beliefs`, `/explain`, `/feedback` commands unchanged
- Belief graph operations remain the same
- Only change: how responses are structured and validated
