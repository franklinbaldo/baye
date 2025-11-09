# 💬 Chat CLI with Belief Training System - Implementation Proposal

## 🎯 Concept

An interactive CLI chat that demonstrates the **Belief Training System V2.0** in action, allowing users to converse with an agent that:

1. **Maintains beliefs** about the world and the user
2. **Updates beliefs** in real-time during conversation
3. **Explains its reasoning** showing which beliefs were used
4. **Learns from feedback** from the user (implicit and explicit)
5. **Shows evolution** of beliefs over time

## 🏗️ Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CHAT CLI INTERFACE                       │
│  • Interactive prompt                                        │
│  • Special commands (/beliefs, /explain, /graph)            │
│  • Conversation history                                      │
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

## 🔑 Main Features

### 1. **Natural Conversation with Belief Tracking**

```
User: Hi! Help me learn Python?
Agent: Sure! I see you want to learn Python. 🐍

[🧠 Activated beliefs]
  φ1: "User is interested in programming" (P=0.50 → 0.82) ↑
  φ2: "User prefers hands-on learning" (P=0.60 → 0.65) ↑

[💭 Decision]
  Will suggest a practical project based on φ2...

Where do you want to start? Do you have a project in mind?
```

### 2. **Special Commands**

```bash
/beliefs          # Show all current beliefs
/explain          # Explain last response
/graph            # Visualize justification graph
/history          # View temporal evolution
/confidence φ1    # Detail specific belief
/feedback         # Give feedback on response
/reset            # Restart session
/export           # Export session (JSON/Markdown)
```

### 3. **Update-on-Use in Action**

Each interaction generates:

```python
# Example: User says "Actually, I prefer theory before practice"
{
  "belief_id": "φ2",
  "text": "User prefers hands-on learning",
  "p_hat": 0.65,  # Agent thought it was true
  "signal": 0.2,  # But evidence contradicts
  "update": {
    "a": 1.3 → 1.34,  # Adjustment in pseudo-counts
    "b": 0.7 → 1.16,
    "P": 0.65 → 0.54  # New confidence
  },
  "provenance": {
    "source": "user_correction",
    "timestamp": "2025-11-08T17:30:00",
    "conversation_id": "conv_123",
    "turn_number": 5
  }
}
```

### 4. **Visible Propagation**

```
User: I prefer books over videos

[🔄 Propagation]
  φ5: "User prefers visual content" (0.70 → 0.45) ↓ CONTRADICTED
    ├─ φ6: "Recommend YouTube tutorials" (0.80 → 0.55) ↓
    └─ φ7: "User learns faster with videos" (0.60 → 0.40) ↓

  φ8: "User prefers text-based resources" (0.40 → 0.72) ↑ SUPPORTED
    └─ φ9: "Recommend documentation reading" (0.50 → 0.75) ↑
```

### 5. **Explainability**

```
User: /explain

[📊 Last Response Analysis]

Question: "Where do you want to start?"

Beliefs consulted:
  ✓ φ1: User is interested in programming (P=0.82, used=3x)
  ✓ φ2: User prefers hands-on learning (P=0.65, used=1x)
  ✓ φ10: Beginners need simple projects (P=0.88, used=5x)

Reasoning:
  1. High confidence in φ10 (88%) → Suggest simple project
  2. Moderate in φ2 (65%) → Offer practice, but don't assume
  3. Ask preferences → Gathering more evidence

Alternatives considered:
  ❌ "Let's start with theory" (rejected by φ2)
  ❌ "Here's a book" (rejected by φ2)
  ✓ "Where do you want to start?" (chosen: gathering info)
```

## 📊 CLI Interface Mockup

```
╔═══════════════════════════════════════════════════════════════╗
║  Baye Chat - Belief-Aware Conversational Agent                ║
║  Session: conv_abc123  |  Beliefs: 47  |  Turns: 12           ║
╚═══════════════════════════════════════════════════════════════╝

[User] Hi! I want to learn machine learning

[Agent] 🤖 Cool! Machine learning is a fascinating area.

        📊 Belief Updates
        ═════════════════
        ↑ φ_tech_01: User interested in ML (0.50 → 0.85)
        ↑ φ_tech_02: User has programming bg (0.40 → 0.55)

        Do you already program? This helps choose the right path.

[User] Yes, I know Python

[Agent] 🐍 Great! Python is perfect for ML.

        📊 Belief Updates
        ═════════════════
        ✓ φ_tech_02: User has programming bg (0.55 → 0.92)
        ✓ φ_lang_01: User knows Python (0.30 → 0.90)
        ↑ φ_path_01: Can start with sklearn (0.45 → 0.78)

        🎯 Recommendation
        ═══════════════
        Based on: φ_lang_01 (0.90) + φ_path_01 (0.78)

        I suggest starting with scikit-learn. Want to see a
        classification example?
```

## 🔧 Technical Implementation

### File Structure

```
src/baye/
├── chat/                        # New module
│   ├── __init__.py
│   ├── cli.py                   # Main CLI interface
│   ├── session.py               # Session management
│   ├── belief_tracker.py        # Baye + Cogito integration
│   ├── commands.py              # Special commands
│   ├── renderer.py              # Output formatting
│   └── prompts.py               # System prompts
│
├── belief_types.py              # Already exists
├── llm_agents.py                # Already exists
├── justification_graph.py       # Already exists
└── cogito/                      # New: Cogito integration
    ├── __init__.py
    ├── update_on_use.py         # Update-on-Use logic
    ├── knn_gradient.py          # K-NN estimation
    └── training.py              # Fine-tuning loop
```

### Main Classes

#### 1. **ChatSession**

```python
class ChatSession:
    """Manages a chat session with belief tracking."""

    def __init__(self, session_id: str, user_profile: Optional[Dict] = None):
        self.session_id = session_id
        self.conversation_history: List[Turn] = []
        self.belief_system = BeliefSystem()  # Baye
        self.training_buffer = TrainingBuffer()  # Cogito
        self.start_time = datetime.now()

    async def process_message(self, user_input: str) -> AgentResponse:
        """Processes user message."""
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

## 🚀 Implementation in Phases

### Phase 1: MVP (1 week)
- [x] Basic CLI with prompt
- [x] Gemini integration (already have)
- [x] /beliefs and /explain commands
- [x] Simple Update-on-Use
- [ ] Nice rendering

### Phase 2: Belief Tracking (1 week)
- [ ] Automatic propagation
- [ ] Graph visualization
- [ ] Change timeline
- [ ] Session persistence

### Phase 3: Training Loop (2 weeks)
- [ ] K-NN gradient estimation
- [ ] Training buffer
- [ ] Periodic fine-tuning
- [ ] Calibration metrics

### Phase 4: Polish (1 week)
- [ ] Rich UI (colors, tables)
- [ ] Export formats (MD, JSON, HTML)
- [ ] Advanced commands
- [ ] Documentation

## 📦 Additional Dependencies

```toml
[project.dependencies]
# ... existing ...
rich = ">=13.0"           # Terminal UI
prompt-toolkit = ">=3.0"  # Interactive prompt
tabulate = ">=0.9"        # Pretty tables
click = ">=8.0"           # CLI framework
```

## 🎯 Success Metrics

1. **Usability**: User can converse naturally
2. **Transparency**: Belief updates are clear and understandable
3. **Learning**: System improves throughout session
4. **Performance**: Response < 2s (including LLM call)
5. **Auditability**: Every decision is traceable

## 🔮 Future Features

- 🌐 **Web UI**: Web interface with D3.js for graphs
- 📊 **Analytics**: Calibration metrics dashboard
- 👥 **Multi-user**: Shared profiles and beliefs
- 🔄 **Sync**: Cross-device synchronization
- 🎮 **Gamification**: Learning badges
- 🔌 **Plugins**: Plugin system for domains

---

**Status**: Complete proposal ✅
**Next**: MVP implementation
**ETA**: 1-2 weeks
