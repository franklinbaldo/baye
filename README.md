# Justification-Based Belief Tracking System

Um sistema de manutenção de crenças neural-simbólico que combina rastreamento causal determinístico com propagação semântica probabilística.

## 🎯 Conceito Central

Ao invés de apenas armazenar crenças isoladas, o sistema mantém um **grafo de justificação** onde:
- **Nodes**: Beliefs (statements em linguagem natural) com confiança probabilística
- **Edges**: Relações de justificação (A suporta B, A contradiz C)
- **Propagação**: Mudanças se propagam através do grafo via dois mecanismos:
  1. **Causal** (determinístico): através de links explícitos de justificação
  2. **Semântica** (probabilístico): através de similaridade de conteúdo

## 🏗️ Arquitetura

```
belief_types.py              # Estruturas de dados core
├── Belief                   # Node do grafo com confiança + links
├── PropagationEvent         # Registro de um update individual
└── PropagationResult        # Resultado de uma cascata completa

justification_graph.py       # Motor principal
├── add_belief()             # Adiciona belief e descobre justificações
├── link_beliefs()           # Cria relacionamentos explícitos
├── propagate_from()         # Inicia cascata de propagação
└── explain_confidence()     # Gera justificativa em linguagem natural

propagation_strategies.py    # Algoritmos isolados
├── CausalPropagator         # Propagação determinística via grafo
├── SemanticPropagator       # Propagação probabilística via similaridade
├── ConflictResolver         # Detecção e resolução de contradições
└── PropagationAnalyzer      # Métricas de consistência

test_stripe_scenario.py      # Validação completa
└── Cenário realista: Stripe API failure
```

## 🔑 Conceitos-Chave

### 1. Dependency Strength (Força de Dependência)

Quando uma belief B é justificada por múltiplas beliefs {A1, A2, A3}, a força da dependência é calculada com:
- **Peso base**: `1/n` onde n = número de supporters
- **Saturação logística**: previne explosão quando supporters já são muito confiantes
- **Ponderação relativa**: confianças são normalizadas entre todos os supporters

```python
dependency = base_weight * (logistic(conf_parent) / sum(logistic(conf_all_parents)))
```

**Por quê saturation?** Se uma belief já tem confiança 0.99, aumentá-la para 0.995 não deveria causar cascata massiva.

### 2. Centrality Dampening (Amortecimento por Centralidade)

Beliefs "hub" (com muitos dependentes) propagam com menos força:

```python
dampening = 1 / log2(2 + num_dependents)
```

**Razão**: Uma belief fundamental que suporta 20 outras não deve causar micro-ajustes em todas elas a cada pequena mudança.

### 3. Propagação Dual

**Causal (70% do peso)**:
- Determinística através de edges explícitos
- Usa cálculo matemático de dependency
- Altamente interpretável (pode traçar caminho)

**Semântica (30% do peso)**:
- Probabilística via similaridade de conteúdo
- Captura relacionamentos implícitos
- Menos interpretável (black-box similarity)

**Merge strategy**:
```python
if belief in causal_updates:
    final = causal[belief] * 0.7
    if belief in semantic_updates:
        final += semantic[belief] * 0.3
else:
    final = semantic[belief] * 0.5  # Semantic sozinho é mais fraco
```

## 🚀 Uso Básico

### Modo V1.5: Com LLM (Recomendado)

```python
from belief_types import Belief
from llm_agents import detect_relationship, resolve_conflict
import asyncio
import os

# Configure API key
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"

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

    # Detectar relacionamento automaticamente
    analysis = await detect_relationship(b1, lesson)
    print(f"Relationship: {analysis.relationship}")  # "contradicts"
    print(f"Confidence: {analysis.confidence}")      # 0.75

    # Resolver conflito via LLM
    if analysis.relationship == "contradicts":
        resolution = await resolve_conflict(b1, lesson)
        print(f"Resolution: {resolution.resolved_belief}")
        # "Third-party services are generally reliable, but critical
        #  paths like payments need defensive programming"

asyncio.run(main())
```

### Modo V1.0: Manual

```python
from justification_graph import JustificationGraph

# Criar grafo
graph = JustificationGraph(max_depth=4)

# Adicionar beliefs manualmente
b1 = graph.add_belief(
    content="APIs can fail unexpectedly",
    confidence=0.6,
    context="api_reliability"
)

# Link manual
graph.link_beliefs(lesson.id, b1.id)

# Propagar mudanças
result = graph.propagate_from(origin_id=lesson.id)
```

## 📊 Exemplo: Stripe API Failure

Execute o exemplo com LLM (requer API key do Gemini):

```bash
export GOOGLE_API_KEY="your-key"
uv run python example_llm_integration.py
```

Ou use o teste V1.0 (sem LLM):

```bash
uv run python test_estimation.py
```

**Estado inicial**:
```
[0.70] Third-party payment services are reliable
[0.60] APIs can fail unexpectedly
[0.40] Skip defensive programming for established services
```

**Evento**: Stripe retorna erro 500

**Lição aprendida**: "Payment APIs can have unexpected downtime" (conf: 0.8)

**Propagação**:
1. Lição contradiz "Third-party reliable" → cai para 0.45
2. "Skip defensive" perde suporte → cai para 0.26
3. "APIs can fail" é reforçada (V1.5 feature)

**Estado final**:
```
[0.80] Payment APIs can have unexpected downtime
[0.45] Third-party payment services are reliable  (↓)
[0.26] Skip defensive programming                 (↓↓)
```

## 🎛️ Parâmetros de Propagação

```python
class JustificationGraph:
    max_depth = 4                                    # Profundidade máxima
    propagation_budget = {0: 8, 1: 5, 2: 3, 3: 2}   # Updates por nível
    min_delta_threshold = 0.05                       # Mínimo para propagar
```

**Budget**: Previne explosão combinatória ao limitar updates por nível.

**Threshold adaptativo**: `threshold * (1.2 ** depth)` - mais profundo = mais exigente.

## 🔬 Análise e Debugging

```python
from propagation_strategies import PropagationAnalyzer

# Verificar consistência interna
score = PropagationAnalyzer.calculate_belief_consistency(graph.beliefs)
# Retorna [0, 1]: beliefs devem ter confiança ≤ média dos supporters

# Identificar beliefs instáveis
unstable = PropagationAnalyzer.identify_unstable_beliefs(graph.beliefs)
# Retorna IDs de beliefs com alta confiança mas suporte fraco
```

## 🎯 Casos de Uso

### 1. Agentes Autônomos
```python
# Após falha em task
task_result = {"error": "JSON malformed", "api": "external"}
lesson = extract_lesson(task_result)
belief = graph.add_belief(lesson, confidence=0.7)
graph.propagate_from(belief.id)
```

### 2. Sistemas de Recomendação
```python
# Aprendizado de preferências
user_feedback = "I don't like spicy food"
preference = graph.add_belief(user_feedback, confidence=0.8)
# Propaga para beliefs relacionadas sobre restaurantes
```

### 3. Diagnóstico Médico
```python
# Atualizar hipóteses com novos sintomas
symptom = graph.add_belief("Patient has fever", confidence=0.9)
# Propaga para diagnósticos possíveis
```

## 🚧 Limitações do V1.0-minimal

### 1. Propagação Unidirecional
**Limitação**: Propagação vai apenas de supporters → dependents, não o inverso.

**Exemplo problemático**:
```
B1: "APIs fail" (0.6)
  ↓ supports
B2: "Validate responses" (0.7)

# Nova evidência
B3: "Stripe failed" (0.8) → supports B1

# B1 deveria aumentar, mas não aumenta no V1.0
```

**Solução (V1.5)**: Propagação bidirecional com pesos diferentes.

### 2. Embeddings Mock
**Limitação**: Similaridade semântica usa Jaccard (overlap de palavras).

**Problema**: "Validate input" e "Check data" são sinônimos mas têm baixo overlap.

**Solução (V1.5)**: Integrar sentence-transformers ou OpenAI embeddings.

### 3. Conflict Resolution Manual
**Limitação**: Contradições precisam ser marcadas manualmente.

**Exemplo**:
```python
# Manual no V1.0
lesson.contradicts.append(b4.id)
b4.update_confidence(-0.25)
```

**Solução (V1.5)**: LLM detecta contradições automaticamente e gera nuances.

### 4. Sem Aprendizado de Estrutura
**Limitação**: Links de justificação são criados manualmente ou por heurísticas.

**Solução (V1.5)**: LLM julga relacionamentos causais:
```python
async def find_justifications(new_belief):
    candidates = rag_search(new_belief.content)
    for c in candidates:
        rel = await llm_judge("Is A a justification for B?")
        if rel == "supports":
            link(c.id, new_belief.id)
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

## 📚 Conexões com Literatura

| Sistema Clássico | Analogia | Inovação |
|------------------|----------|----------|
| **TMS (Doyle, 1979)** | Justification graph | Similaridade semântica vs lógica propositional |
| **SOAR (Laird, 1987)** | Chunking (lesson → belief) | Propagação probabilística |
| **ACT-R (Anderson)** | Activation spreading | Confiança como proxy para activation |
| **Bayesian Networks** | Prior/posterior updates | LLM como likelihood function não-paramétrica |

**Contribuição principal**: Semantizar a propagação - usar proximidade em embedding space como função de influência ao invés de regras lógicas explícitas.

## 🤝 Contribuindo

Áreas prioritárias para contribuição:
1. **Embeddings reais**: Integrar sentence-transformers
2. **LLM integration**: Relationship detection + conflict resolution
3. **Visualização**: Dashboard interativo
4. **Benchmarks**: Datasets de agent failures

## 📄 Licença

MIT License - use livremente em projetos comerciais ou acadêmicos.

## 🙏 Agradecimentos

Inspirado por discussões sobre sistemas de manutenção de crenças, Bayesian program learning, e arquiteturas de agentes autônomos.

---

**Status**: V1.0-minimal completo ✅
**Próximo**: V1.5 (embeddings reais + LLM integration)
**Autor**: Franklin Baldo ([@franklinbaldo](https://github.com/franklinbaldo))
