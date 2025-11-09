# Fluxo Visual do Sistema de Treino V2.0

## Ciclo Completo: Da Observação ao Gradiente

```
┌──────────────────────────────────────────────────────────────────┐
│                    PASSO 1: OBSERVAÇÃO EPISTÊMICA                │
│                                                                  │
│  Agent executa tarefa → observa resultado → reflete             │
│  Exemplo: API call falhou com timeout                           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                PASSO 2: DECODIFICAÇÃO FORÇADA (Tool)             │
│                                                                  │
│  LLM DEVE gerar:                                                │
│  {                                                               │
│    "tool": "update_belief",                                      │
│    "parameters": {                                               │
│      "belief_id": "φ1",                                          │
│      "p_hat": 0.75,          ← Subjetividade do agent          │
│      "signal": 0.9,           ← Observação externa mapeada      │
│      "r": 0.8, "n": 1.0, "q": 0.9  ← Pesos UoU                 │
│      "provenance": {...}                                         │
│    }                                                             │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│              PASSO 3: UPDATE-ON-USE (Memória)                    │
│                                                                  │
│  Crença φ antes: a=1.2, b=0.8  → P(φ)=0.60                     │
│                                                                  │
│  Evidência nova:                                                 │
│    w = r × n × q = 0.8 × 1.0 × 0.9 = 0.72                      │
│    a' = a + w·signal = 1.2 + 0.72×0.9 = 1.848                  │
│    b' = b + w·(1-signal) = 0.8 + 0.72×0.1 = 0.872              │
│                                                                  │
│  Crença φ depois: a=1.848, b=0.872  → P(φ)=0.679               │
│                                                                  │
│  ✅ Evidência registrada com proveniência completa               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│           PASSO 4: ESTIMAÇÃO K-NN (Alvo de Treino)               │
│                                                                  │
│  Buscar 5 vizinhos semânticos de φ:                             │
│                                                                  │
│    φ₂: "Validar inputs" (sim=0.85, P=0.80, u=0.2)              │
│    φ₃: "Try-catch I/O" (sim=0.72, P=0.70, u=0.3)               │
│    φ₇: "Timeouts comuns" (sim=0.68, P=0.65, u=0.4)             │
│    φ₉: "Cache resultados" (sim=0.55, P=0.50, u=0.5)            │
│    φ₁₂: "Retry logic" (sim=0.48, P=0.75, u=0.3)                │
│                                                                  │
│  Pesos por incerteza:                                            │
│    w₂ = 1/(1+0.2) = 0.833  → normalizado: 0.31                 │
│    w₃ = 1/(1+0.3) = 0.769  → normalizado: 0.28                 │
│    w₇ = 1/(1+0.4) = 0.714  → normalizado: 0.26                 │
│    w₉ = 1/(1+0.5) = 0.667  → normalizado: 0.24                 │
│    w₁₂ = 1/(1+0.3) = 0.769 → normalizado: 0.28                 │
│                                                                  │
│  p_knn = Σ(wᵢ × Pᵢ)                                             │
│        = 0.31×0.80 + 0.28×0.70 + ... = 0.68                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│            PASSO 5: MIXAGEM DE ALVOS (p_star)                    │
│                                                                  │
│  p* = λ·signal + (1-λ)·p_knn                                    │
│     = 0.7×0.90 + 0.3×0.68                                       │
│     = 0.63 + 0.204 = 0.834                                      │
│                                                                  │
│  Se consenso fraco (mean_u > 0.5):                              │
│    p* ← 0.3×0.5 + 0.7×0.834 = 0.734  (pull toward prior)       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│              PASSO 6: CÁLCULO DA LOSS                            │
│                                                                  │
│  Brier Score:                                                    │
│    L_brier = (p_hat - p*)² = (0.75 - 0.834)² = 0.007           │
│                                                                  │
│  Peso por certeza da vizinhança:                                 │
│    conf_weight = 1 - mean_uncertainty                           │
│               = 1 - 0.32 = 0.68                                 │
│                                                                  │
│  Loss ponderada:                                                 │
│    L_weighted = 0.007 × 0.68 = 0.0048                           │
│                                                                  │
│  (Opcional) Tensão dialética:                                    │
│    Se φ CONTRADICTS ψ:                                          │
│      L_tension = relu(0.1 - |p_φ + p_ψ - 1|)                    │
│                                                                  │
│  (Opcional) ECE proxy (calibração por bins)                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│         PASSO 7: PROPAGAÇÃO LOCAL (Grafo)                        │
│                                                                  │
│  Δconf = 0.679 - 0.60 = +0.079                                  │
│                                                                  │
│  Para cada vizinho via arestas:                                  │
│                                                                  │
│    φ₃ (SUPPORTS φ):                                             │
│      similarity = 0.72                                           │
│      dampening = 0.5 × 0.72 = 0.36                              │
│      Δφ₃ = +0.079 × 0.36 = +0.028                               │
│      φ₃: 0.700 → 0.703 ✓                                        │
│                                                                  │
│    φ₄ (CONTRADICTS φ):                                          │
│      similarity = 0.65                                           │
│      dampening = 0.3 × 0.65 = 0.195                             │
│      Δφ₄ = -0.079 × 0.195 = -0.015                              │
│      φ₄: 0.500 → 0.496 ✓                                        │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│          PASSO 8: BUFFER DE TREINO                               │
│                                                                  │
│  Adicionar sample:                                               │
│  {                                                               │
│    "belief_id": "φ1",                                            │
│    "context": "Belief: APIs externas podem falhar...",          │
│    "p_hat": 0.75,          ← Input da LLM                       │
│    "p_star": 0.834,        ← Alvo calculado (K-NN + signal)     │
│    "uncertainties": 0.32,                                        │
│    "signal": 0.9,                                                │
│    "brier": 0.007,                                               │
│    "timestamp": "2025-11-08T10:30:00Z"                          │
│  }                                                               │
│                                                                  │
│  Quando buffer >= 100 samples → Batch training                  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│         PASSO 9: FINE-TUNING (Batch Periódico)                   │
│                                                                  │
│  Cada N tarefas ou M samples:                                    │
│                                                                  │
│  1. Carregar batch do buffer                                     │
│  2. Forward pass da LLM → hidden_states                         │
│  3. Calibration head → p_hat_predicted                          │
│  4. Loss = MSE(p_hat_predicted, p_star_batch)                   │
│  5. Backward → atualizar LoRA weights + calibration head        │
│  6. Limpar buffer                                                │
│                                                                  │
│  Resultado: LLM aprende a estimar p_hat calibrado               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│              PASSO 10: PRÓXIMA ITERAÇÃO                          │
│                                                                  │
│  Agent agora tem:                                                │
│   ✅ Crença φ atualizada com proveniência                        │
│   ✅ Memória justificatória completa                             │
│   ✅ Modelo calibrado para futuras estimações                    │
│   ✅ Grafo de justificação consistente                           │
│                                                                  │
│  → Loop continua para próxima tarefa                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Fluxo de Dados: Anatomia de um Update

```
Input (Agent/LLM)          Tool Processing           Output (Treino)
─────────────────          ───────────────           ───────────────

belief_id: "φ"             1. Recuperar φ            p_star: 0.834
p_hat: 0.75            ┌─→ 2. UoU update        ┌─→ uncertainties: 0.32
signal: 0.9            │   3. K-NN search       │   brier: 0.007
r, n, q: 0.8,1.0,0.9   │   4. Mixagem           │   
provenance: {...}      │   5. Loss calc         │   [Training Buffer]
                       │                        │   → Batch training
                       └────────────────────────┘   → Model update
```

---

## Grafo de Justificação: Exemplo Visual

```
                    φ₁: APIs podem falhar
                   (P=0.679, u=0.368)
                          │
                 ┌────────┴────────┐
                 │                 │
            SUPPORTS          CONTRADICTS
                 │                 │
                 ↓                 ↓
        φ₃: Try-catch I/O    φ₄: Confiar em
         (P=0.703, u=0.5)     serviços auth
                 │             (P=0.496, u=0.5)
            SUPPORTS
                 │
                 ↓
         φ₅: Logs ajudam
          (P=0.900, u=0.5)

Legenda:
  P = Confidence (probabilidade)
  u = Uncertainty (incerteza epistêmica)
  → Propagação flui pelas arestas com dampening
```

---

## K-NN Semântico: Visualização

```
              Espaço de Embeddings (2D projetado)

                     φ₂ (P=0.80)
                        ●
                         ╲
                          ╲ sim=0.85
                           ╲
                            ╲
                             ● φ (query)
                            ╱  P_knn=0.68
                  sim=0.72 ╱
                          ╱
                         ●
                     φ₃ (P=0.70)

  Quanto mais próximo (maior similaridade),
  maior o peso no cálculo de p_knn.
  
  Crenças com baixa incerteza contribuem mais.
```

---

## Comparação: Antes vs Depois do Update

```
┌─────────────────────────────────────────────────┐
│              ESTADO INICIAL (t=0)               │
├─────────────────────────────────────────────────┤
│ Crença φ₁: "APIs externas podem falhar"        │
│   a=1.2, b=0.8                                  │
│   P(φ₁) = 0.600                                 │
│   u(φ₁) = 0.500                                 │
│   Evidências: 0                                 │
│   Proveniência: []                              │
└─────────────────────────────────────────────────┘
                     ↓
         [Observação: Timeout em API]
         [Agent estima p_hat=0.75]
         [Signal externo: 0.9]
                     ↓
┌─────────────────────────────────────────────────┐
│             ESTADO ATUALIZADO (t=1)             │
├─────────────────────────────────────────────────┤
│ Crença φ₁: "APIs externas podem falhar"        │
│   a=1.848, b=0.872                              │
│   P(φ₁) = 0.679 (+0.079) ✓                     │
│   u(φ₁) = 0.368 (↓ mais certeza)               │
│   Evidências: 1                                 │
│   Proveniência:                                 │
│     - [2025-11-08] TimeoutError                │
│       source: task_execution                    │
│       weight: 0.72                              │
│       signal: 0.9                               │
└─────────────────────────────────────────────────┘
```

---

## Métricas de Calibração: Interpretação

```
Brier Score = 0.007     →  Excelente (próximo de 0)
ECE = 0.060             →  Calibrado (< 0.1 é bom)

Interpretação:
  - Agent está aprendendo a estimar probabilidades
    muito próximas do consenso local (K-NN)
  - Ainda há espaço para melhoria (p_hat vs p_star)
  - Com mais treino, p_hat → p_star

Objetivo: Minimizar Brier e ECE através do fine-tuning
```

---

## Auditabilidade: Rastreando uma Decisão

```
Query: "Por que o agent agora confia menos em serviços autenticados?"

Resposta auditável:
  
  1. Crença φ₄: "Confiar em serviços autenticados"
     Estado atual: P=0.496 (queda de 0.004)
  
  2. Causa raiz (backtracking):
     ← Propagação negativa de φ₁ (CONTRADICTS)
     ← φ₁ sofreu update positivo (+0.079)
     ← φ₁ recebeu evidência forte:
        - Timestamp: 2025-11-08T10:30:00Z
        - Source: task_execution
        - Signal: 0.9 (timeout em API)
        - Weight: 0.72 (alta confiabilidade)
  
  3. Justificativa lógica:
     "Se APIs externas falham frequentemente,
      então confiar cegamente em serviços autenticados
      é menos seguro."
  
  4. Evidências adicionais que suportariam revisão:
     - Observar sucesso consistente de APIs auth
     - Distinguir entre falhas de rede vs auth
```

---

## Próximos Passos de Implementação

```
Fase 1: Core [2 semanas]
  ├─ ✅ Esquema de dados (SQL + ChromaDB)
  ├─ ✅ Tool update_belief
  ├─ ✅ K-NN estimation
  └─ ✅ UoU logic

Fase 2: Training [2 semanas]
  ├─ [ ] Calibration head (PyTorch)
  ├─ [ ] Loss functions implementadas
  ├─ [ ] Training loop com LoRA
  └─ [ ] Inference pipeline

Fase 3: Robustez [1 semana]
  ├─ [ ] Diversified K-NN
  ├─ [ ] Uncertainty regularization
  ├─ [ ] Propagação avançada
  └─ [ ] Detecção de equilíbrio

Fase 4: Produção [1 semana]
  ├─ [ ] API REST
  ├─ [ ] Dashboard de auditoria
  ├─ [ ] Testes E2E
  └─ [ ] Documentação final
```

---

**Conclusão**: Este sistema une o melhor de três mundos:

1. **Simbólico**: Grafo de justificação explícito
2. **Probabilístico**: Update-on-Use com pseudo-contagens
3. **Neural**: Fine-tuning da LLM via gradientes locais

Resultado: Um agente que **aprende com seus próprios atos** de forma auditável, calibrada e disciplinada. 🎯
