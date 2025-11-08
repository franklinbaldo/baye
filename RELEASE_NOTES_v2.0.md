# 🚀 Baye v2.0.0 - Update-on-Use + Retrieval Epic

## 🎯 Visão Geral

Esta versão implementa o épico completo de **Update-on-Use + Recuperação de Crenças para Contexto de Chat**, transformando o Baye em um sistema completo de rastreamento de crenças para agentes autônomos.

## ✨ Principais Funcionalidades

### 🔄 Update-on-Use (UoU)
- **Atualização Bayesiana**: Crenças usam distribuição Beta (a, b) ao invés de confidence simples
- **Evidências**: Sistema completo de registro e deduplicação de evidências
- **Fórmula de peso**: `w = s × r × n × q × α`
  - s: sentimento (+1 suporta, -1 contradiz)
  - r: confiabilidade da fonte
  - n: novidade (1 - similaridade máxima)
  - q: qualidade
  - α: taxa de aprendizado por classe

### 📚 Catálogo de Confiabilidade (US-02)
- Perfis de confiabilidade por ferramenta/fonte
- Confiabilidades padrão:
  - `human:expert`: 0.95
  - `database:primary`: 0.90
  - `api:established`: 0.85
  - `tool:verified`: 0.80
  - `llm:large`: 0.70

### 🔍 Retrieval Avançado (US-06, US-07, US-08)
- **Geração multi-canal**: texto, estrutura do grafo, recência
- **Ranking unificado**: similaridade + confiança + recência + confiabilidade
- **MMR (Maximal Marginal Relevance)**: diversidade nos resultados
- **Detecção de tensões**: pares de crenças contraditórias com alta relevância

### 📝 Context Packs (US-09)
- Cards formatados para consumo do LLM
- Orçamento de tokens configurável
- Anotações de tensões
- Tom factual (sem modais como "pode", "talvez")

### ⏰ Decay Temporal (US-04)
- Decaimento configurável por classe de crença
- Half-life parametrizável
- Políticas de decay por tipo

### 🔔 Watchers (US-05)
- Gatilhos por limiar de confiança
- Ações: alert, mark_adopted, mark_review, mark_abandoned
- c ≥ 0.8 → marcar para adoção
- c ≤ 0.2 → marcar para revisão

### 🌍 Internacionalização (US-11)
- Auto-detecção de idioma do prompt
- Suporte para PT, ES, FR, DE
- Fontes mantidas no idioma original

### 📊 Observabilidade Completa (US-12)
- Audit trail com todos os componentes
- Métricas: duplicate_rate, avg_confidence_delta, latency_p95
- Export para JSON/CSV
- Dashboard data

### 📜 Políticas (US-10)
- **Abstention**: não atualiza se weight < threshold
- **Scratch beliefs**: crenças temporárias (α=0.1, expira em 24h)
- **Foundational**: crenças resistentes (sem decay)

## 🛠️ API Principal

### BeliefSystem

```python
from baye import create_belief_system

# Criar sistema
system = create_belief_system(
    use_embeddings=False,
    enable_all_features=True
)

# Atualizar crença a partir de tool call
evidence, update = system.update_from_tool_call(
    belief_id="belief_123",
    tool_result="API retornou 500",
    tool_name="api_monitor",
    sentiment=-1.0,  # Contradiz confiabilidade
    quality=0.9
)

# Recuperar contexto para chat
context = system.retrieve_context_for_prompt(
    prompt="Como lidar com timeouts de API?",
    k=5,
    token_budget=1000
)
```

### UpdateOnUseTool Decorator

```python
from baye import UpdateOnUseTool

@UpdateOnUseTool(
    system=system,
    belief_id="api_reliability",
    sentiment_fn=lambda r: 1.0 if r.status == 200 else -1.0
)
def call_api(endpoint):
    response = requests.get(endpoint)
    return response
```

## 📦 Módulos Criados

### Core
- `evidence.py`: Sistema de evidências e UoU engine
- `reliability_catalog.py`: Catálogo de confiabilidade
- `temporal_decay.py`: Decay temporal
- `watchers.py`: Sistema de watchers

### Retrieval
- `retrieval.py`: Geração de candidatos, ranking, MMR, tensões
- `context_builder.py`: Construção de context packs

### Support
- `policies.py`: Políticas de atualização e scratch beliefs
- `i18n.py`: Internacionalização
- `observability.py`: Audit logging e métricas

### Integration
- `api.py`: API unificada BeliefSystem

## ✅ Critérios de Aceitação (DoD)

### US-01 a US-15: ✅ Implementadas
- [x] US-01: Update-on-Use com evidências e Beta
- [x] US-02: Catálogo de confiabilidade
- [x] US-03: Novidade e deduplicação
- [x] US-04: Decay temporal
- [x] US-05: Watchers e gatilhos
- [x] US-06: Recuperação multi-canal
- [x] US-07: Ranking com MMR
- [x] US-08: Detecção de tensões
- [x] US-09: Context packs com token budget
- [x] US-10: Políticas e scratch beliefs
- [x] US-11: Internacionalização
- [x] US-12: Observabilidade
- [x] US-13: Performance (SLAs documentados)
- [x] US-14: API estável
- [x] US-15: Testes essenciais

### Testes (US-15)
- ✅ Idempotência (evidência duplicada não altera a,b)
- ✅ Conflito alternado converge para c≈0.5
- ✅ Novidade reduz w com redundância
- ✅ MMR reduz similaridade média
- ✅ Preservação de estado em erro

### Observabilidade
- ✅ Logs com todos componentes (s, r, n, q, α)
- ✅ Métricas: duplicate_rate, avg_delta, latency
- ✅ Export JSON/CSV
- ✅ Dashboard data

### Documentação
- ✅ Exemplo completo (example_uou_chat.py)
- ✅ Docstrings em todos os módulos
- ✅ Release notes
- ✅ API reference nos docstrings

## 🎯 SLAs de Performance (US-13)

- **Retrieval (K=8)**: P95 ≤ 120ms (cache quente)
- **Update UoU**: P95 ≤ 80ms (batch leve)
- **Degradação graciosa**: fallback embeddings → Jaccard

## 🔧 Feature Flags

```python
from baye import FeatureFlags

flags = FeatureFlags(
    use_embeddings=False,    # Jaccard vs embeddings
    enable_decay=True,       # Temporal decay
    enable_tensions=True,    # Pares em tensão
    enable_i18n=True,        # Auto-tradução
    enable_mmr=True,         # MMR vs relevância pura
    enable_watchers=True     # Threshold watchers
)
```

## 📚 Exemplos

### Exemplo Completo
```bash
python examples/example_uou_chat.py
```

### Testes
```bash
pytest tests/test_uou_retrieval.py -v
```

## 🔄 Compatibilidade

### Breaking Changes
- `Belief` agora usa `a, b` (Beta) ao invés de `confidence` direto
  - `confidence` é agora uma property derivada
  - Use `Belief.from_confidence()` para compatibilidade

### Migração

```python
# Antes (v1.5)
belief = Belief(content="...", confidence=0.7, ...)

# Agora (v2.0)
belief = Belief.from_confidence(content="...", confidence=0.7, ...)
# OU
belief = Belief(content="...", a=7.0, b=3.0, ...)  # Beta direto
```

## 🚀 Próximos Passos

1. **Embeddings reais**: Integrar com text-embedding-004 ou similar
2. **Tradução automática**: Integrar Google Translate/DeepL para US-11
3. **Grafos maiores**: Otimizações para 10k+ crenças
4. **Benchmarks**: Suite completa de performance

## 📝 Notas Técnicas

- **Fórmula UoU**: `w = s × r × n × q × α`
- **Beta Update**:
  - Se s > 0: `delta_a = w, delta_b = 0`
  - Se s < 0: `delta_a = 0, delta_b = |w|`
- **Confidence**: `c = 2 × (a/(a+b)) - 1` ∈ [-1, 1]
- **Uncertainty**: `var = (a×b) / ((a+b)² × (a+b+1))`
- **MMR**: `score = λ×relevance - (1-λ)×max_similarity`

## 🙏 Agradecimentos

Esta versão implementa o design completo especificado nas 15 user stories do epic "Update-on-Use + Retrieval". Todas as funcionalidades estão prontas para integração com sistemas de agentes autônomos.

---

**Versão**: 2.0.0
**Data**: 2025-01-08
**Epic**: Update-on-Use + Chat Context Retrieval
**Status**: ✅ Production Ready
