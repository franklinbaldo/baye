# 📚 Belief Training System V2.0 - Índice Completo

**Versão**: 2.0-RC1  
**Data**: 2025-11-08  
**Status**: Design Completo + Prototype Funcional

---

## 🎯 Início Rápido

### Para Executar Agora

```bash
# 1. Rodar o prototype
python belief_training_prototype.py

# 2. Ver demo completa com:
#    - Criação de crenças
#    - Updates epistêmicos (UoU)
#    - Propagação via grafo
#    - Cálculo de K-NN
#    - Métricas de calibração
```

### Para Entender Rapidamente

1. **Leia primeiro**: [`README.md`](README.md) (10 min) ⭐
2. **Veja fluxos**: [`VISUAL_FLOW.md`](VISUAL_FLOW.md) (15 min)
3. **Execute**: `belief_training_prototype.py` (5 min)

---

## 📖 Documentação Completa

### 1. [`README.md`](README.md) - Executive Summary
**12 KB | Tempo: 10 min**

- 🎯 O que é e por que importa
- 🏗️ Arquitetura em 3 camadas
- 📦 Deliverables completos
- 🔑 Conceitos-chave (UoU, K-NN, Tensão)
- 📊 Métricas de sucesso
- 🚀 Roadmap (8 semanas)
- 💡 Casos de uso prioritários

**Comece aqui se**: Quer visão geral executiva

---

### 2. [`BELIEF_TRAINING_SPEC_V2.md`](BELIEF_TRAINING_SPEC_V2.md) - Especificação Técnica
**33 KB | Tempo: 45 min**

**Seções principais**:

#### Dados e Estruturas
- `BeliefState`: Pseudo-contagens + proveniência
- `Evidence`: Registro com r, n, q, provenance
- Schemas SQL (beliefs, evidence, edges)
- ChromaDB para busca vetorial

#### Tool Única
- Interface JSON completa
- Lógica Update-on-Use (UoU)
- Estimação K-NN detalhada
- Mixagem de alvos (signal + K-NN)

#### Pipeline de Treino
- Calibration head (PyTorch)
- Loss functions (Brier + tensão + ECE)
- Training loop com LoRA
- Inferência calibrada

#### Propagação
- Local update via grafo
- Dampening por similaridade
- Detecção de equilíbrio

#### Robustez
- Diversified K-NN sampling
- Uncertainty regularization
- Cold-start strategies
- Temporal decay

#### Implementação
- Roadmap 8 semanas (6 fases)
- Métricas de avaliação
- Comparação vs sistemas existentes

**Comece aqui se**: Vai implementar o sistema

---

### 3. [`VISUAL_FLOW.md`](VISUAL_FLOW.md) - Fluxos e Diagramas
**21 KB | Tempo: 20 min**

**Diagramas ASCII incluídos**:

- ✅ Ciclo completo (10 passos): Observação → Gradiente
- ✅ Fluxo de dados: Anatomia de um update
- ✅ Grafo de justificação: Exemplo visual
- ✅ K-NN semântico: Espaço de embeddings
- ✅ Antes vs Depois: Comparação de estados
- ✅ Métricas: Interpretação de Brier e ECE
- ✅ Auditabilidade: Rastreamento de decisões
- ✅ Roadmap: Fases de implementação

**Comece aqui se**: É visual e quer entender fluxos

---

### 4. [`COMPARATIVE_ANALYSIS.md`](COMPARATIVE_ANALYSIS.md) - Análise Comparativa
**16 KB | Tempo: 25 min**

**Comparações vs**:

#### Sistemas Clássicos
- Truth Maintenance Systems (TMS)
- SOAR (State, Operator, And Result)
- ACT-R (Adaptive Control of Thought)

#### Métodos Modernos
- RAG (Retrieval-Augmented Generation)
- RLHF (Reinforcement Learning from Human Feedback)
- Neural Episodic Control (NEC)

**Tabela resumo**: V2.0 = 7/7 critérios ✅

**Vantagens únicas**:
1. Proveniência auditável
2. Calibração on-policy
3. Resolução dialética
4. Cold-start inteligente

**Trade-offs honestos**:
- Benefício: Memória + Consistência + Calibração
- Custo: ~2x computação vs RAG puro

**Casos de uso diferenciados**:
- Compliance (financeiro/médico)
- Pesquisa científica
- Debugging de agentes

**Comece aqui se**: Quer justificar escolha arquitetural

---

### 5. [`PRACTICAL_EXAMPLES.md`](PRACTICAL_EXAMPLES.md) - Exemplos Práticos
**22 KB | Tempo: 30 min**

**6 exemplos completos com código**:

#### Exemplo 1: Agente de Suporte Técnico
- Aprendizado incremental com tickets
- Propagação de crenças relacionadas
- Treino batch após N interações

#### Exemplo 2: Pesquisa Médica
- Síntese de literatura científica
- Ponderação por qualidade de evidência
- Detecção de gaps na pesquisa

#### Exemplo 3: Debugging de Agent
- Rastreamento de falhas
- Backtrace de causas raízes
- Recomendações automáticas

#### Exemplo 4: Compliance Financeiro
- Decisão de crédito explicável
- Relatório GDPR Article 22
- Audit log completo

#### Exemplo 5: A/B Testing
- Mapear métricas → crenças
- Priorizar experimentos
- Recomendações data-driven

#### Exemplo 6: Loop de Treino Completo
- Execução de tarefas
- Reflexão do agent
- Fine-tuning periódico
- Checkpoints

**Comece aqui se**: Quer ver código real executável

---

### 6. [`belief_training_prototype.py`](belief_training_prototype.py) - Prototype
**17 KB | 600 linhas | Tempo: 5 min para executar**

**Classes implementadas**:
- `BeliefState`: Crença com pseudo-contagens
- `Evidence`: Evidência com proveniência
- `BeliefSystem`: Sistema completo

**Features**:
- ✅ Add/get beliefs
- ✅ K-NN semântico (cosine similarity)
- ✅ Tool `update_belief_tool` (UoU)
- ✅ Propagação local (grafo)
- ✅ Training buffer
- ✅ Métricas (Brier, ECE)

**Demo scenario**:
1. Cria 5 crenças
2. Constrói grafo (SUPPORTS/CONTRADICTS)
3. Simula 2 atualizações epistêmicas
4. Propaga mudanças
5. Calcula métricas
6. Mostra estado final

**Output esperado**:
```
Belief Training System V2.0 - Demo
============================================================
...
✅ Demo completo!
============================================================
```

**Comece aqui se**: Quer testar imediatamente

---

## 🗺️ Guia de Navegação

### Por Objetivo

| Seu Objetivo | Comece Aqui | Depois Vá Para |
|--------------|-------------|----------------|
| **Entender rapidamente** | README.md | VISUAL_FLOW.md |
| **Implementar sistema** | BELIEF_TRAINING_SPEC_V2.md | prototype |
| **Justificar decisão** | COMPARATIVE_ANALYSIS.md | README.md |
| **Ver código funcionando** | prototype | PRACTICAL_EXAMPLES.md |
| **Aprender conceitos** | VISUAL_FLOW.md | SPEC |

### Por Perfil

| Perfil | Arquivos Essenciais | Tempo Total |
|--------|---------------------|-------------|
| **Executive** | README.md | 10 min |
| **Product Manager** | README + COMPARATIVE | 35 min |
| **Developer** | SPEC + prototype + EXAMPLES | 90 min |
| **Researcher** | SPEC + COMPARATIVE + VISUAL | 120 min |
| **Auditor** | EXAMPLES (Ex. 4) + VISUAL | 45 min |

### Por Fase de Projeto

| Fase | Documentos | Ação |
|------|------------|------|
| **Discovery** | README + COMPARATIVE | Decidir se vale a pena |
| **Design** | SPEC + VISUAL | Planejar implementação |
| **Development** | SPEC + prototype | Codificar |
| **Testing** | EXAMPLES | Criar casos de teste |
| **Production** | SPEC (seções 8-10) | Deploy e monitoring |

---

## 📊 Métricas do Projeto

| Métrica | Valor |
|---------|-------|
| **Total de documentação** | 121 KB |
| **Linhas de código** | 600 (prototype) |
| **Tempo de desenvolvimento** | ~8 horas |
| **Exemplos práticos** | 6 completos |
| **Diagramas visuais** | 8 |
| **Comparações técnicas** | 6 sistemas |
| **Roadmap completo** | 8 semanas, 6 fases |

---

## 🎓 Conceitos por Arquivo

| Conceito | Explicado em | Código em |
|----------|--------------|-----------|
| **Update-on-Use** | SPEC § 3, VISUAL § 3 | prototype L140-155 |
| **K-NN Estimation** | SPEC § 4, VISUAL § 4 | prototype L85-115 |
| **Calibration Loss** | SPEC § 5.2, VISUAL § 6 | SPEC L450-490 |
| **Propagação** | SPEC § 6, VISUAL § 7 | prototype L240-280 |
| **Tensão Dialética** | SPEC § 5.2, COMPARATIVE § 3.3 | SPEC L470-480 |
| **Proveniência** | SPEC § 2.2, COMPARATIVE § 3.1 | prototype L20-35 |

---

## 🔗 Links Rápidos

### Arquivos Core
- [README.md](README.md) - Começar aqui ⭐
- [BELIEF_TRAINING_SPEC_V2.md](BELIEF_TRAINING_SPEC_V2.md) - Referência técnica
- [belief_training_prototype.py](belief_training_prototype.py) - Código executável

### Arquivos Suplementares
- [VISUAL_FLOW.md](VISUAL_FLOW.md) - Diagramas
- [COMPARATIVE_ANALYSIS.md](COMPARATIVE_ANALYSIS.md) - Justificativas
- [PRACTICAL_EXAMPLES.md](PRACTICAL_EXAMPLES.md) - Casos de uso

---

## ✅ Checklist de Compreensão

### Nível Básico (30 min)
- [ ] Executei o prototype e vi funcionando
- [ ] Entendo Update-on-Use (a, b, signal)
- [ ] Entendo K-NN (vizinhos semânticos)
- [ ] Sei o que é p_hat vs p_star

### Nível Intermediário (90 min)
- [ ] Li a spec completa
- [ ] Entendo o pipeline de treino
- [ ] Entendo propagação via grafo
- [ ] Entendo tensão dialética
- [ ] Consigo explicar vantagens vs RAG

### Nível Avançado (3h)
- [ ] Li tudo
- [ ] Entendo loss functions detalhadas
- [ ] Posso implementar um componente
- [ ] Posso debugar o sistema
- [ ] Posso justificar decisões arquiteturais

---

## 🚀 Próximos Passos

### Para Começar Desenvolvimento
```bash
# 1. Setup ambiente
python -m venv venv
source venv/bin/activate
pip install torch transformers peft chromadb

# 2. Testar prototype
python belief_training_prototype.py

# 3. Implementar Fase 1 (2 semanas)
# - Persistência (SQLite)
# - ChromaDB integration
# - Tool completa
```

### Para Discussão Técnica
1. Abrir issue no repo (quando criado)
2. Enviar email (a definir)
3. Marcar reunião técnica

### Para Contribuir
1. Fork do repo
2. Implementar componente
3. Testes + documentação
4. Pull request

---

## 📞 Suporte

**Status**: Open-source (Apache 2.0)  
**Repo**: (a ser criado)  
**Issues**: GitHub Issues  
**Email**: (a definir)

**Para perguntas sobre**:
- Arquitetura: Ver SPEC + COMPARATIVE
- Implementação: Ver prototype + EXAMPLES
- Casos de uso: Ver EXAMPLES
- Justificativas: Ver COMPARATIVE
- Visão geral: Ver README

---

## 🏆 Reconhecimentos

**Baseado em**:
- Update-on-Use (Justificatory Memory)
- Truth Maintenance Systems (Doyle, 1979)
- Neural Episodic Control (Pritzel et al., 2017)
- Modern calibration research (Guo et al., 2017)

**Inovações**:
- Híbrido simbólico-neural-estatístico único
- Self-supervised K-NN targets
- Auditabilidade by design
- On-policy training sem labels humanos

---

## 📝 Changelog

### V2.0-RC1 (2025-11-08)
- ✅ Especificação técnica completa
- ✅ Prototype funcional
- ✅ Documentação visual
- ✅ Análise comparativa
- ✅ 6 exemplos práticos
- ✅ Este índice

### Próximo (V2.1)
- [ ] Implementação Fase 1
- [ ] Benchmarks empíricos
- [ ] API REST
- [ ] Dashboard web

---

**Total**: 6 arquivos, 121 KB de documentação, prototype funcional.

**Pronto para**: Implementação imediata. 🚀
