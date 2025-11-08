# Justification-Based Belief Tracking System

Um sistema de manutenção de crenças neural-simbólico que combina rastreamento causal determinístico com propagação semântica probabilística, powered by LLMs.

## 🎯 Conceito Central

Ao invés de apenas armazenar crenças isoladas, o sistema mantém um **grafo de justificação** onde:
- **Nodes**: Beliefs (statements em linguagem natural) com confiança probabilística
- **Edges**: Relações de justificação (A suporta B, A contradiz C)
- **Propagação**: Mudanças se propagam através do grafo via dois mecanismos:
  1. **Causal** (determinístico): através de links explícitos de justificação
  2. **Semântica** (probabilístico): através de similaridade de conteúdo
- **LLM Integration**: Detecção automática de relacionamentos e resolução de conflitos via Gemini

## 🏗️ Arquitetura

```
baye/
├── src/baye/              # Package principal
│   ├── __init__.py        # Exports públicos
│   ├── belief_types.py    # Estruturas de dados core
│   ├── justification_graph.py  # Motor principal
│   ├── belief_estimation.py    # K-NN semântico
│   └── llm_agents.py      # Agentes PydanticAI + Gemini
├── examples/              # Exemplos de uso
│   ├── example_llm_integration.py
│   └── example_estimation_integrated.py
├── tests/                 # Testes
│   └── test_estimation.py
├── pyproject.toml         # Config uv
└── README.md
```

## 🚀 Instalação

```bash
# Clone o repositório
git clone https://github.com/franklinbaldo/baye.git
cd baye

# Instale com uv
uv sync

# Configure API key do Gemini
export GOOGLE_API_KEY="your-gemini-api-key"
```

## 💡 Uso Rápido

### Modo V1.5: Com LLM (Recomendado)

```python
from baye import Belief, detect_relationship, resolve_conflict
import asyncio

async def main():
    # Criar beliefs
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

    # Detectar relacionamento automaticamente via LLM
    analysis = await detect_relationship(b1, lesson)
    print(f"Relationship: {analysis.relationship}")  # "contradicts"
    print(f"Confidence: {analysis.confidence}")      # 0.70

    # Resolver conflito via LLM
    if analysis.relationship == "contradicts":
        resolution = await resolve_conflict(b1, lesson)
        print(f"Resolved: {resolution.resolved_belief}")
        # "While third-party services are generally reliable,
        #  critical paths like payments need defensive programming"

asyncio.run(main())
```

### Modo V1.0: Manual (sem LLM)

```python
from baye import JustificationGraph, Belief

# Criar grafo
graph = JustificationGraph(max_depth=4)

# Adicionar beliefs manualmente
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

# Propagar mudanças
result = graph.propagate_from(origin_id=b1.id)
print(f"Updated {result.total_beliefs_updated} beliefs")
```

## 📊 Exemplo Completo

Execute o exemplo com LLM (requer API key):

```bash
export GOOGLE_API_KEY="your-key"
uv run python examples/example_llm_integration.py
```

**Output esperado:**
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

## 🔑 Conceitos-Chave

### 1. LLM-Powered Relationship Detection

Usa Gemini via PydanticAI para detectar automaticamente se beliefs:
- **SUPPORT**: Um fornece evidência para o outro
- **CONTRADICT**: Não podem ser verdadeiros simultaneamente
- **REFINE**: Um é uma versão mais específica do outro
- **UNRELATED**: Sem conexão lógica significativa

### 2. Conflict Resolution

Quando beliefs contradizem, o LLM gera uma belief nuanceada que:
- Reconhece aspectos válidos de ambos
- Identifica condições onde cada um se aplica
- Fornece síntese balanceada e acionável

### 3. Structured Outputs

Todos os agentes retornam Pydantic models validados:
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
- [x] Grafo causal básico
- [x] Propagação determinística
- [x] Detecção de ciclos
- [x] Teste Stripe funcionando

### V1.5 (LLM Integration) ✅ **CONCLUÍDO**
- [x] Relationship discovery via LLM (PydanticAI + Gemini)
- [x] Conflict resolution automático via LLM
- [x] Structured outputs com Pydantic models
- [x] Batch relationship detection
- [x] Organização src/baye/
- [ ] Propagação bidirecional (próximo)
- [ ] Embeddings reais via Gemini (próximo)

### V2.0 (Escalabilidade) 🎯
- [ ] Persistência (Neo4j + vector DB)
- [ ] Batch propagation (múltiplas lessons)
- [ ] Dashboard de visualização (NetworkX + Plotly)
- [ ] API REST para integração

### V2.5 (Inteligência) 🧠
- [ ] Aprendizado de pesos de edges
- [ ] Meta-beliefs ("confio mais em security beliefs")
- [ ] Temporal decay (beliefs antigas perdem relevância)
- [ ] Active learning (sistema pede clarificação quando incerto)

## 📚 API Reference

### Core Types

```python
from baye import Belief, BeliefID, Confidence, RelationType

# Criar belief
belief = Belief(
    content="APIs can fail",
    confidence=0.8,
    context="reliability"
)

# Atualizar confiança
belief.update_confidence(delta=0.1)  # Aumenta para 0.9
```

### LLM Agents

```python
from baye import (
    detect_relationship,
    resolve_conflict,
    find_related_beliefs,
    check_gemini_api_key
)

# Verificar API key
check_gemini_api_key()  # Raises ValueError se não configurada

# Detectar relacionamento
analysis = await detect_relationship(belief1, belief2)

# Resolver conflito
resolution = await resolve_conflict(belief1, belief2, context="optional")

# Encontrar beliefs relacionadas em batch
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

# Adicionar belief
b = graph.add_belief(content="...", confidence=0.7)

# Linkar beliefs
graph.link_beliefs(parent_id, child_id, relation=RelationType.SUPPORTS)

# Propagar mudanças
result = graph.propagate_from(origin_id=b.id)
print(f"Updated: {result.total_beliefs_updated}")
print(f"Max depth: {result.max_depth_reached}")
```

## 🧪 Testing

```bash
# Rodar todos os testes
uv run pytest tests/

# Teste específico
uv run pytest tests/test_estimation.py -v

# Com coverage
uv run pytest --cov=src/baye tests/
```

## 🤝 Contribuindo

Áreas prioritárias:
1. **Embeddings reais**: Integrar Gemini Embeddings API
2. **Propagação bidirecional**: Supporters também devem ser atualizados
3. **Visualização**: Dashboard interativo
4. **Benchmarks**: Datasets de agent failures

## 📄 Licença

MIT License - use livremente em projetos comerciais ou acadêmicos.

## 🙏 Agradecimentos

Inspirado por discussões sobre Truth Maintenance Systems (TMS), Bayesian program learning, e arquiteturas de agentes autônomos.

---

**Status**: V1.5 (LLM Integration) ✅ CONCLUÍDO
**Próximo**: V2.0 (embeddings reais + propagação bidirecional)
**Autor**: Franklin Baldo ([@franklinbaldo](https://github.com/franklinbaldo))
