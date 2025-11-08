# 💬 Chat CLI com Belief Training System - Proposta de Implementação

## 🎯 Conceito

Um chat interativo via CLI que demonstra o **Belief Training System V2.0** em ação, permitindo ao usuário conversar com um agente que:

1. **Mantém crenças** sobre o mundo e o próprio usuário
2. **Atualiza crenças** em tempo real conforme a conversa
3. **Explica seu raciocínio** mostrando quais beliefs foram usadas
4. **Aprende com feedback** do usuário (implícito e explícito)
5. **Mostra evolução** das crenças ao longo do tempo

## 🏗️ Arquitetura Proposta

```
┌─────────────────────────────────────────────────────────────┐
│                     CHAT CLI INTERFACE                       │
│  • Prompt interativo                                         │
│  • Comandos especiais (/beliefs, /explain, /graph)          │
│  • Histórico de conversação                                  │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┴────────────┐
       │                        │
┌──────▼─────────┐   ┌─────────▼──────────┐
│  Baye System   │   │  Cogito Training   │
│  (V1.5)        │   │  (V2.0)            │
│                │   │                    │
│  • Beliefs     │◄──┤  • Update-on-Use   │
│  • Propagation │   │  • K-NN Gradient   │
│  • LLM Agents  │   │  • Calibration     │
└────────────────┘   └────────────────────┘
       │                        │
       └───────────┬────────────┘
                   │
         ┌─────────▼──────────┐
         │  Gemini API        │
         │  (PydanticAI)      │
         └────────────────────┘
```

## 🔑 Features Principais

### 1. **Conversação Natural com Belief Tracking**

```
User: Oi! Me ajuda a aprender Python?
Agent: Claro! Vejo que você quer aprender Python. 🐍

[🧠 Crenças ativadas]
  φ1: "User is interested in programming" (P=0.50 → 0.82) ↑
  φ2: "User prefers hands-on learning" (P=0.60 → 0.65) ↑
  
[💭 Decisão]
  Vou sugerir um projeto prático baseado em φ2...

Por onde você quer começar? Tem algum projeto em mente?
```

### 2. **Comandos Especiais**

```bash
/beliefs          # Mostrar todas as crenças atuais
/explain          # Explicar última resposta
/graph            # Visualizar grafo de justificação
/history          # Ver evolução temporal
/confidence φ1    # Detalhar crença específica
/feedback         # Dar feedback sobre resposta
/reset            # Reiniciar sessão
/export           # Exportar sessão (JSON/Markdown)
```

### 3. **Update-on-Use em Ação**

Cada interação gera:

```python
# Exemplo: User diz "Na verdade, prefiro teoria antes de praticar"
{
  "belief_id": "φ2",
  "text": "User prefers hands-on learning",
  "p_hat": 0.65,  # Agent achava que era verdade
  "signal": 0.2,  # Mas evidência contradiz
  "update": {
    "a": 1.3 → 1.34,  # Ajuste nas pseudo-contagens
    "b": 0.7 → 1.16,
    "P": 0.65 → 0.54  # Nova confiança
  },
  "provenance": {
    "source": "user_correction",
    "timestamp": "2025-11-08T17:30:00",
    "conversation_id": "conv_123",
    "turn_number": 5
  }
}
```

### 4. **Propagação Visível**

```
User: Gosto mais de livros que vídeos

[🔄 Propagação]
  φ5: "User prefers visual content" (0.70 → 0.45) ↓ CONTRADICTED
    ├─ φ6: "Recommend YouTube tutorials" (0.80 → 0.55) ↓
    └─ φ7: "User learns faster with videos" (0.60 → 0.40) ↓
  
  φ8: "User prefers text-based resources" (0.40 → 0.72) ↑ SUPPORTED
    └─ φ9: "Recommend documentation reading" (0.50 → 0.75) ↑
```

### 5. **Explicabilidade**

```
User: /explain

[📊 Análise da Última Resposta]

Pergunta: "Por onde você quer começar?"

Crenças consultadas:
  ✓ φ1: User is interested in programming (P=0.82, used=3x)
  ✓ φ2: User prefers hands-on learning (P=0.65, used=1x)
  ✓ φ10: Beginners need simple projects (P=0.88, used=5x)

Raciocínio:
  1. Alta confiança em φ10 (88%) → Sugerir projeto simples
  2. Moderada em φ2 (65%) → Oferecer prática, mas não assumir
  3. Perguntar preferências → Gathering more evidence

Alternativas consideradas:
  ❌ "Vamos começar com teoria" (rejeitada por φ2)
  ❌ "Aqui está um livro" (rejeitada por φ2)
  ✓ "Por onde quer começar?" (escolhida: gathering info)
```

## 📊 Interface CLI Mockup

```
╔═══════════════════════════════════════════════════════════════╗
║  Baye Chat - Belief-Aware Conversational Agent                ║
║  Session: conv_abc123  |  Beliefs: 47  |  Turns: 12           ║
╚═══════════════════════════════════════════════════════════════╝

[User] Oi! Quero aprender machine learning

[Agent] 🤖 Que legal! Machine learning é uma área fascinante.

        📊 Belief Updates
        ═════════════════
        ↑ φ_tech_01: User interested in ML (0.50 → 0.85)
        ↑ φ_tech_02: User has programming bg (0.40 → 0.55)
        
        Você já programa? Isso ajuda a escolher o caminho certo.

[User] Sim, já sei Python

[Agent] 🐍 Ótimo! Python é perfeito para ML.

        📊 Belief Updates
        ═════════════════
        ✓ φ_tech_02: User has programming bg (0.55 → 0.92)
        ✓ φ_lang_01: User knows Python (0.30 → 0.90)
        ↑ φ_path_01: Can start with sklearn (0.45 → 0.78)
        
        🎯 Recomendação
        ═══════════════
        Baseado em: φ_lang_01 (0.90) + φ_path_01 (0.78)
        
        Sugiro começar com scikit-learn. Quer ver um exemplo
        de classificação?

[User] /beliefs

╔═══════════════════════════════════════════════════════════════╗
║  Current Beliefs (Top 10 by Confidence)                       ║
╚═══════════════════════════════════════════════════════════════╝

ID       Belief                                    P     Updated  Used
───────────────────────────────────────────────────────────────────
φ_lang_01  User knows Python                     0.90   Just now  2x
φ_tech_02  User has programming background       0.92   30s ago   3x
φ_tech_01  User interested in ML                 0.85   1m ago    2x
φ_path_01  Can start with sklearn                0.78   Just now  1x
φ_learn_01 User prefers hands-on learning        0.65   Session   0x
...

[User] /graph φ_path_01

   φ_lang_01 (0.90)
   "User knows Python"
         │
         │ SUPPORTS (0.85)
         ▼
   φ_path_01 (0.78)
   "Can start with sklearn"
         │
         │ SUPPORTS (0.72)
         ▼
   φ_rec_01 (0.65)
   "Recommend classification tutorial"

[User] Na verdade, prefiro começar com teoria

[Agent] 📚 Entendido! Vou ajustar minha abordagem.

        📊 Belief Updates
        ═════════════════
        ↓ φ_learn_01: Hands-on learning (0.65 → 0.38) [CORRECTED]
        ↑ φ_learn_02: Theory-first approach (0.35 → 0.72)
        
        🔄 Propagation (2 beliefs affected)
        ═══════════════════════════════════
        ↓ φ_rec_01: Recommend tutorials (0.65 → 0.45)
        ↑ φ_rec_02: Recommend books (0.40 → 0.68)
        
        Nesse caso, recomendo começar com "Pattern Recognition
        and Machine Learning" do Bishop. Muito completo!

───────────────────────────────────────────────────────────────────
Commands: /beliefs /explain /graph /history /feedback /help /exit
```

## 🔧 Implementação Técnica

### Estrutura de Arquivos

```
src/baye/
├── chat/                        # Novo módulo
│   ├── __init__.py
│   ├── cli.py                   # Interface CLI principal
│   ├── session.py               # Gerenciamento de sessão
│   ├── belief_tracker.py        # Integração Baye + Cogito
│   ├── commands.py              # Comandos especiais
│   ├── renderer.py              # Formatação de output
│   └── prompts.py               # System prompts
│
├── belief_types.py              # Já existe
├── llm_agents.py                # Já existe
├── justification_graph.py       # Já existe
└── cogito/                      # Novo: Cogito integration
    ├── __init__.py
    ├── update_on_use.py         # Update-on-Use logic
    ├── knn_gradient.py          # K-NN estimation
    └── training.py              # Fine-tuning loop
```

### Classes Principais

#### 1. **ChatSession**

```python
class ChatSession:
    """Gerencia uma sessão de chat com belief tracking."""
    
    def __init__(self, session_id: str, user_profile: Optional[Dict] = None):
        self.session_id = session_id
        self.conversation_history: List[Turn] = []
        self.belief_system = BeliefSystem()  # Baye
        self.training_buffer = TrainingBuffer()  # Cogito
        self.start_time = datetime.now()
        
    async def process_message(self, user_input: str) -> AgentResponse:
        """Processa mensagem do usuário."""
        # 1. Retrieve relevant beliefs (RAG-like)
        relevant_beliefs = self.belief_system.retrieve(user_input, k=5)
        
        # 2. Generate response with LLM (with beliefs in context)
        response = await self.generate_response(user_input, relevant_beliefs)
        
        # 3. Extract beliefs mentioned/updated
        mentioned_beliefs = self.extract_beliefs(response)
        
        # 4. Update-on-Use for each belief
        updates = []
        for belief_ref in mentioned_beliefs:
            update = await self.update_belief_tool(
                belief_id=belief_ref.id,
                p_hat=belief_ref.confidence_used,
                signal=self.infer_signal(user_input, response),
                provenance=self.build_provenance()
            )
            updates.append(update)
        
        # 5. Propagate changes
        self.belief_system.propagate_updates(updates)
        
        # 6. Add to training buffer
        self.training_buffer.add_sample(...)
        
        # 7. Store turn
        turn = Turn(user_input, response, updates)
        self.conversation_history.append(turn)
        
        return response
```

#### 2. **BeliefTracker**

```python
class BeliefTracker:
    """Integra Baye (beliefs) + Cogito (training)."""
    
    def __init__(self):
        self.graph = JustificationGraph()  # Baye V1.5
        self.uou_system = UpdateOnUseSystem()  # Cogito V2.0
        
    async def update_belief_tool(
        self,
        belief_id: str,
        p_hat: float,  # Agent's estimate
        signal: float,  # Observed outcome
        r: float = 1.0,  # Reliability
        n: float = 1.0,  # Novelty
        q: float = 1.0,  # Quality
        provenance: Dict = None
    ) -> BeliefUpdate:
        """
        Tool única que combina:
        1. Update-on-Use (Cogito)
        2. Propagation (Baye)
        3. Training signal generation
        """
        # 1. Get current belief
        belief = self.graph.beliefs[belief_id]
        
        # 2. Update-on-Use (pseudo-counts)
        weight = r * n * q
        old_a, old_b = belief.a, belief.b
        
        belief.a += weight * signal
        belief.b += weight * (1 - signal)
        
        new_confidence = belief.a / (belief.a + belief.b)
        
        # 3. K-NN gradient estimation
        neighbors = self.graph.find_related_beliefs(belief, k=5)
        p_star_knn = np.mean([nb.confidence for nb in neighbors])
        p_star = 0.7 * signal + 0.3 * p_star_knn  # Mix
        
        # 4. Calculate loss for training
        loss = (p_hat - p_star) ** 2 * (old_a + old_b)  # Weighted Brier
        
        # 5. Log evidence
        evidence = Evidence(
            belief_id=belief_id,
            signal=signal,
            r=r, n=n, q=q,
            provenance=provenance,
            timestamp=datetime.now()
        )
        belief.evidence_log.append(evidence)
        
        # 6. Propagate to graph
        affected = self.graph.propagate_from(belief_id)
        
        # 7. Return training signal
        return BeliefUpdate(
            belief_id=belief_id,
            old_confidence=old_a / (old_a + old_b),
            new_confidence=new_confidence,
            p_hat=p_hat,
            p_star=p_star,
            loss=loss,
            affected_beliefs=affected,
            evidence=evidence
        )
```

#### 3. **CLI Renderer**

```python
class CLIRenderer:
    """Formata output bonito no terminal."""
    
    @staticmethod
    def render_response(response: AgentResponse) -> str:
        """Renderiza resposta do agent com belief updates."""
        output = []
        
        # Agent message
        output.append(f"\n[Agent] {response.icon} {response.message}\n")
        
        # Belief updates
        if response.updates:
            output.append("        📊 Belief Updates")
            output.append("        " + "═" * 40)
            for upd in response.updates:
                arrow = "↑" if upd.delta > 0 else "↓" if upd.delta < 0 else "="
                output.append(
                    f"        {arrow} {upd.belief_text} "
                    f"({upd.old_conf:.2f} → {upd.new_conf:.2f})"
                )
            output.append("")
        
        # Reasoning (if requested)
        if response.show_reasoning:
            output.append("        💭 Reasoning")
            output.append("        " + "═" * 40)
            output.append(f"        {response.reasoning}")
            output.append("")
        
        return "\n".join(output)
    
    @staticmethod
    def render_belief_table(beliefs: List[Belief]) -> str:
        """Renderiza tabela de crenças."""
        # Pretty table com rich ou tabulate
        ...
    
    @staticmethod
    def render_graph(root_belief: Belief, depth: int = 2) -> str:
        """Renderiza árvore ASCII do grafo."""
        # Similar ao tree command do Unix
        ...
```

### System Prompts

```python
CHAT_SYSTEM_PROMPT = """You are a belief-aware conversational agent.

Your beliefs about the user are stored in a justification graph.
When responding, you MUST:

1. Consult relevant beliefs from context
2. Update beliefs based on user input
3. Call update_belief_tool for significant updates
4. Explain your confidence when asked

Current active beliefs:
{active_beliefs}

Remember:
- Be helpful and honest
- Update beliefs when corrected
- Show uncertainty when appropriate
- Track user preferences and learning style

Available tools:
- update_belief: Update a belief with new evidence
- search_beliefs: Find relevant beliefs
- explain_reasoning: Show why you said something
"""

BELIEF_EXTRACTION_PROMPT = """Extract belief updates from this conversation turn.

User: {user_input}
Agent: {agent_response}

Identify:
1. Which beliefs were consulted (retrieved)
2. Which beliefs should be updated
3. What is the signal (0-1) from user feedback
4. Confidence the agent had (p_hat)

Return as JSON:
{
  "consulted": [{"id": "φ1", "p_hat": 0.8}],
  "updates": [
    {
      "id": "φ2",
      "p_hat": 0.7,
      "signal": 0.9,
      "reason": "User confirmed preference"
    }
  ]
}
"""
```

## 🎬 Exemplo de Conversação Completa

### Cenário: Tutoria de Programação

```
╔═══════════════════════════════════════════════════════════════╗
║  Baye Chat - Programming Tutor                                ║
║  Session: new  |  Beliefs: 0  |  Turns: 0                     ║
╚═══════════════════════════════════════════════════════════════╝

[System] Inicializando sistema de crenças...
         ✓ Loaded 25 domain beliefs (programming, learning)
         ✓ Ready to chat!

> Oi! Quero aprender web development

[Agent] 🌐 Legal! Web development tem várias áreas.

        📊 Belief Created
        ════════════════
        ✨ φ_001: User interested in web dev (P=0.85, NEW)
        
        Você já programa ou está começando do zero?

> Já sei um pouco de Python

[Agent] 🐍 Ótimo! Python abre várias portas no web dev.

        📊 Belief Updates
        ════════════════
        ✓ φ_002: User has prog background (0.30 → 0.88)
        ✓ φ_003: User knows Python (0.20 → 0.90)
        ↑ φ_004: Can learn Django/Flask (0.40 → 0.75)
        
        🔄 Propagation (3 affected)
        ═══════════════════════════
        ↑ φ_005: Skip basic syntax (0.45 → 0.72)
        ↑ φ_006: Focus on frameworks (0.50 → 0.78)
        ↓ φ_007: Need beginner tutorials (0.70 → 0.35)
        
        Sugiro Django ou Flask. Qual te interessa mais?

> Na verdade quero fazer frontend com React

[Agent] ⚛️ Ah, entendi! Frontend com React é diferente.

        📊 Belief Updates
        ════════════════
        ↓ φ_004: Django/Flask relevant (0.75 → 0.25) [CORRECTED]
        ✨ φ_008: User wants frontend (P=0.90, NEW)
        ✨ φ_009: User wants React (P=0.92, NEW)
        ↑ φ_010: Needs JavaScript first (0.35 → 0.88)
        
        🔄 Propagation (5 affected)
        ═══════════════════════════
        ↑ φ_011: Recommend JS basics (0.40 → 0.85)
        ↑ φ_012: Recommend Node/npm (0.30 → 0.70)
        ↓ φ_013: Backend-first approach (0.60 → 0.25)
        
        💡 Insight
        ══════════
        Based on: φ_003 (Python) + φ_009 (React)
        
        Você vai precisar de JavaScript primeiro. Como Python
        e JS têm sintaxes parecidas, a transição será suave.
        
        Quer que eu sugira um roadmap?

> Sim, por favor!

[Agent] 📋 Aqui está um roadmap personalizado:

        1️⃣ JavaScript Fundamentals (2-3 semanas)
           → ES6+ syntax, async/await, modules
           Confidence: φ_014 (0.85) - "JS before React"
        
        2️⃣ HTML/CSS Basics (1 semana)
           → Semantic HTML, Flexbox, Grid
           Confidence: φ_015 (0.92) - "Frontend needs HTML/CSS"
        
        3️⃣ React Basics (3-4 semanas)
           → Components, hooks, state management
           Confidence: φ_009 (0.92) - "User wants React"
        
        4️⃣ Build Projects (ongoing)
           → Portfolio, small apps
           Confidence: φ_016 (0.78) - "Learning by doing"
        
        Parece bom? Posso detalhar alguma parte?

> /explain

[📊 Análise da Última Resposta]

Pergunta: Criação do roadmap

Crenças consultadas (6):
  ✓ φ_003: User knows Python (P=0.90, weight=0.15)
  ✓ φ_009: User wants React (P=0.92, weight=0.30)
  ✓ φ_010: Needs JavaScript first (P=0.88, weight=0.25)
  ✓ φ_014: JS before React (P=0.85, weight=0.20)
  ✓ φ_015: Frontend needs HTML/CSS (P=0.92, weight=0.05)
  ✓ φ_016: Learning by doing (P=0.78, weight=0.05)

Raciocínio (weighted decision tree):
  1. Start with JS: φ_010 (0.88) × φ_014 (0.85) = HIGH
  2. Include HTML/CSS: φ_015 (0.92) = CRITICAL
  3. React after basics: φ_009 (0.92) + φ_014 (0.85) = CERTAIN
  4. Project-based: φ_016 (0.78) + φ_003 (0.90) = PREFERRED
     (Python bg suggests hands-on learner)

Alternativas consideradas:
  ❌ "Start directly with React" (rejected by φ_010, φ_014)
  ❌ "Do backend first with Django" (rejected by φ_008, φ_009)
  ✓ "JS → HTML/CSS → React → Projects" (chosen: 0.89 confidence)

Training signal:
  p_hat (agent confidence) = 0.89
  p_star (will be inferred from user feedback)
  If user says "perfect!": signal = 1.0 → reinforce
  If user says "too slow": signal = 0.4 → adjust φ_016

> Perfeito! Vamos começar

[Agent] 🚀 Ótimo! Vou registrar esse feedback.

        📊 Belief Updates (Reinforcement Learning)
        ══════════════════════════════════════════
        ✓ φ_014: JS before React (0.85 → 0.91) [REINFORCED]
        ✓ φ_016: Learning by doing (0.78 → 0.84) [REINFORCED]
        
        🎓 Training Signal Generated
        ════════════════════════════
        Sample added to buffer:
        {
          "beliefs_used": [φ_009, φ_010, φ_014, φ_015, φ_016],
          "p_hat": [0.92, 0.88, 0.85, 0.92, 0.78],
          "p_star": [0.95, 0.90, 0.91, 0.93, 0.84],
          "loss": 0.012,  # Low = good prediction
          "provenance": {
            "turn": 5,
            "feedback": "positive",
            "timestamp": "2025-11-08T18:15:30"
          }
        }
        
        Vamos começar com JavaScript! Primeira lição:
        Variáveis e tipos de dados...

> /beliefs top 5

╔═══════════════════════════════════════════════════════════════╗
║  Top 5 Beliefs (by Confidence × Recency)                      ║
╚═══════════════════════════════════════════════════════════════╝

ID       Belief                                P      Evidence  Used
──────────────────────────────────────────────────────────────────
φ_009  User wants React                    0.92   1 confirm   3x
φ_015  Frontend needs HTML/CSS             0.92   domain      1x
φ_014  JS before React                     0.91   1 reinf.    2x
φ_003  User knows Python                   0.90   1 confirm   4x
φ_010  Needs JavaScript first              0.88   inferred    2x

> /graph φ_014

    φ_009 (0.92) "User wants React"
      │
      │ REQUIRES (0.95)
      ▼
    φ_014 (0.91) "JS before React"
      │
      ├─ SUPPORTS (0.88) ─→ φ_011 (0.85) "Recommend JS basics"
      │
      └─ SUPPORTS (0.72) ─→ φ_017 (0.68) "Start with ES6"

> /history

Turn 1: User interest → Created φ_001 (web dev)
Turn 2: Python mention → Updated φ_002, φ_003
Turn 3: Frontend/React → Corrected φ_004, created φ_008, φ_009
Turn 4: Roadmap request → Consulted 6 beliefs
Turn 5: Positive feedback → Reinforced φ_014, φ_016

Timeline:
  18:10:00  Session start
  18:11:15  First beliefs created
  18:12:30  Major correction (backend → frontend)
  18:14:45  Roadmap generated (6 beliefs used)
  18:15:30  Positive feedback (training signal)
  18:16:00  [CURRENT]

> /export chat_session.md

✓ Exported to: chat_session.md
  - Conversation history (5 turns)
  - Belief evolution (17 beliefs)
  - Training samples (1 accumulated)
  - Graphs (3 shown)

> exit

[System] Salvando sessão...
         ✓ Beliefs persisted (17 total)
         ✓ Training buffer saved (1 sample)
         ✓ Session exported
         
         📊 Session Statistics
         ════════════════════
         Duration: 6m 15s
         Turns: 5
         Beliefs created: 9 new
         Beliefs updated: 8 existing
         Training samples: 1
         
         See you next time! 👋
```

## 🎨 Visualizações Adicionais

### 1. **Timeline de Confiança**

```
/timeline φ_014

Belief: "JS before React"

 1.0┤
    │
 0.9┤                        ●━━━━━●
    │                       ╱
 0.8┤                 ●━━━━╯
    │                ╱
 0.7┤          ●━━━━╯
    │         ╱
 0.6┤    ●━━━╯
    │   ╱
 0.5┤━━━╯
    │
 0.0┴───┬────┬────┬────┬────┬────┬────
       T1   T2   T3   T4   T5   T6   now
       
Events:
  T1: Created (P=0.50, prior)
  T3: First mention (P=0.70, +0.20)
  T4: Roadmap used (P=0.85, +0.15)
  T5: Reinforced (P=0.91, +0.06)
```

### 2. **Heatmap de Crenças**

```
/heatmap

Belief Activity (last 10 turns)

             T1  T2  T3  T4  T5  T6  T7  T8  T9  T10
φ_001 web    █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (1x)
φ_003 python ░░  █░░ █░░ █░░ █░░░░░░░░░░░░░░░░░░  (4x)
φ_009 react  ░░░░░░  █░░ ██░ █░░ █░░░░░░░░░░░░░  (5x)
φ_014 js     ░░░░░░░░░░  █░░ ██░ █░░ █░░░░░░░░  (5x)
φ_016 hands  ░░░░░░░░░░░░░░  █░░ █░░░░░░░░░░░░  (2x)

Legend: █ used  ░ not used
```

### 3. **Cluster de Crenças**

```
/cluster

Belief Clusters (semantic similarity)

┌─ Programming Languages ─────────────┐
│  φ_003: Python (0.90)                │
│  φ_018: JavaScript (0.85)            │
│  φ_023: TypeScript (0.45)            │
└──────────────────────────────────────┘
           │
           ├─ RELATED ─┐
           │           │
┌─ Frontend Tech ─────▼────────────────┐
│  φ_009: React (0.92)                 │
│  φ_015: HTML/CSS (0.92)              │
│  φ_021: Vue.js (0.30)                │
└──────────────────────────────────────┘
           │
           ├─ SUPPORTS ─┐
           │            │
┌─ Learning Preferences ▼──────────────┐
│  φ_016: Hands-on (0.84)              │
│  φ_025: Theory-first (0.25)          │
│  φ_027: Video tutorials (0.60)       │
└──────────────────────────────────────┘
```

## 🚀 Implementação em Fases

### Phase 1: MVP (1 semana)
- [x] CLI básico com prompt
- [x] Integração com Gemini (já temos)
- [x] Comandos /beliefs e /explain
- [x] Update-on-Use simples
- [ ] Rendering bonito

### Phase 2: Belief Tracking (1 semana)
- [ ] Propagação automática
- [ ] Visualização de grafos
- [ ] Timeline de mudanças
- [ ] Persistência de sessão

### Phase 3: Training Loop (2 semanas)
- [ ] K-NN gradient estimation
- [ ] Training buffer
- [ ] Periodic fine-tuning
- [ ] Métricas de calibração

### Phase 4: Polish (1 semana)
- [ ] Rich UI (cores, tabelas)
- [ ] Export formats (MD, JSON, HTML)
- [ ] Comandos avançados
- [ ] Documentação

## 📦 Dependências Adicionais

```toml
[project.dependencies]
# ... existing ...
rich = ">=13.0"           # Terminal UI
prompt-toolkit = ">=3.0"  # Interactive prompt
tabulate = ">=0.9"        # Pretty tables
click = ">=8.0"           # CLI framework
```

## 🎯 Métricas de Sucesso

1. **Usabilidade**: Usuário consegue conversar naturalmente
2. **Transparência**: Belief updates são claros e compreensíveis
3. **Aprendizado**: Sistema melhora ao longo da sessão
4. **Performance**: Resposta < 2s (incluindo LLM call)
5. **Auditabilidade**: Toda decisão é rastreável

## 🔮 Features Futuras

- 🌐 **Web UI**: Interface web com D3.js para grafos
- 📊 **Analytics**: Dashboard de métricas de calibração
- 👥 **Multi-user**: Perfis e crenças compartilhadas
- 🔄 **Sync**: Sincronização cross-device
- 🎮 **Gamification**: Badges por aprendizado
- 🔌 **Plugins**: Sistema de plugins para domínios

---

**Status**: Proposta completa ✅
**Next**: Implementação do MVP
**ETA**: 1-2 semanas
