"""
ChatSession - Interactive belief tracking chat interface

Manages conversation flow, LLM interactions, and belief updates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import asyncio
import warnings

from pydantic_ai import Agent
from pydantic import BaseModel, Field

# Suppress PydanticAI warnings about additionalProperties
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic_ai')

from .belief_tracker import BeliefTracker, BeliefUpdate
from .belief_types import Belief
from .llm_agents import detect_relationship, check_gemini_api_key


@dataclass
class Message:
    """Chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class ExtractedBelief(BaseModel):
    """A single extracted belief"""
    content: str = Field(..., description="The belief statement")
    context: str = Field(default="general", description="Domain or context")
    confidence_estimate: float = Field(default=0.5, ge=0.0, le=1.0, description="Estimated confidence")


class BeliefExtraction(BaseModel):
    """Structured output for belief extraction from conversation"""
    beliefs: List[ExtractedBelief] = Field(default_factory=list, description="List of beliefs extracted")
    reasoning: str = Field(..., description="Explanation of why these beliefs were extracted")


@dataclass
class ToolCallStep:
    """Single step in a multi-step response"""
    text: str
    belief_id: str
    belief_value_guessed: float
    delta_requested: float
    applied_delta: float
    actual_confidence: float
    margin: float
    error: float


@dataclass
class AssistantReply:
    """Structured assistant reply returned to the CLI."""
    steps: List[ToolCallStep]  # Multiple tool calls

    @property
    def text(self) -> str:
        """Concatenate all step texts"""
        return "\n\n".join(step.text for step in self.steps)

    @property
    def belief_id(self) -> str:
        """Last belief ID"""
        return self.steps[-1].belief_id if self.steps else ""

    @property
    def belief_value_guessed(self) -> float:
        """Last guess"""
        return self.steps[-1].belief_value_guessed if self.steps else 0.0

    @property
    def delta_requested(self) -> float:
        """Total delta requested"""
        return sum(step.delta_requested for step in self.steps)

    @property
    def applied_delta(self) -> float:
        """Total delta applied"""
        return sum(step.applied_delta for step in self.steps)

    @property
    def actual_confidence(self) -> float:
        """Final confidence"""
        return self.steps[-1].actual_confidence if self.steps else 0.0

    @property
    def margin(self) -> float:
        """Margin from last step"""
        return self.steps[-1].margin if self.steps else 0.1


class JustificationBelief(BaseModel):
    """A belief that justifies a confidence change"""
    content: str = Field(..., description="The justification statement")
    context: str = Field(default="justification", description="Context/domain")
    confidence_estimate: float = Field(..., ge=0.0, le=1.0, description="Confidence in this justification")
    # Note: Removed sub_justifications to avoid recursive $ref (Gemini limitation)
    # Justifications are kept flat - deeper chains built via graph traversal


class ToolCallResult(BaseModel):
    """Payload returned by the update_belief_tool."""
    texto: str = Field(..., description="Mensagem que será exibida para o usuário final.")
    belief_value_guessed: float = Field(
        ..., ge=0.0, le=1.0,
        description="Palpite PRECISO da LLM para a confiança atual da crença ativa. Seja específico! Erro é esperado e será usado para aprendizado."
    )
    delta: float = Field(
        0.0, ge=-1.0, le=1.0,
        description="Ajuste solicitado para a crença (0 mantém o valor atual)."
    )
    justifications: List[JustificationBelief] = Field(
        default_factory=list,
        description="OBRIGATÓRIO quando delta != 0: Lista de crenças que justificam este ajuste. Mínimo 1 justificativa."
    )
    belief_id: str = Field(..., description="ID interno da crença manipulada.")
    actual_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confiança após o processamento do tool."
    )
    applied_delta: float = Field(
        0.0, description="Delta realmente aplicado à crença."
    )
    margin: float = Field(
        ..., ge=0.0, le=1.0,
        description="Margem de tolerância usada para esta crença."
    )


DEFAULT_CONFIDENCE_MARGIN = 0.10

class MultiStepResponse(BaseModel):
    """Multiple reasoning steps in one response"""
    steps: List[ToolCallResult] = Field(
        ...,
        min_length=1,
        description="Lista de passos de raciocínio. Cada passo é uma interação com uma crença."
    )


RESPONSE_SYSTEM_PROMPT = f"""
Você é um assistente prestativo com sistema de rastreamento de crenças.

**SUA PRIORIDADE PRINCIPAL**: Ajudar o usuário! Seja útil, claro e direto.

O sistema de crenças roda em **background** - ele serve para você aprender e melhorar,
mas NÃO deve ser o foco da conversa. O usuário não se importa com suas crenças internas,
ele quer **ajuda**, **respostas** e **valor**.

**Quando mencionar crenças**:
- ✅ Quando relevante para explicar sua resposta
- ✅ Quando o usuário explicitamente perguntar
- ❌ NÃO fique falando sobre "estou aprendendo", "minha confiança é X"
- ❌ NÃO faça meta-comentários desnecessários

**Sua saída** é um objeto JSON com lista de `steps`:

```json
{{
  "steps": [
    {{
      "texto": "Resposta útil e direta ao usuário (sem meta-talk!)",
      "belief_value_guessed": 0.73,
      "delta": 0,
      ...
    }}
  ]
}}
```

Cada passo (ToolCallResult) tem:

1. **`texto`** (string): Mensagem natural para o usuário. Explique sua resposta e raciocínio.

2. **`belief_value_guessed`** (float, 0-1):
   - Seu palpite PRECISO e ESPECÍFICO para a confiança atual da crença ativa
   - NÃO arredonde para 0.5, 0.6, etc. Use valores exatos: 0.73, 0.82, 0.47
   - **Erro é ESPERADO e ÚTIL** - será usado para calibração e meta-aprendizado
   - Quanto mais preciso você tentar ser, melhor o sistema aprende

3. **`delta`** (float, -1 a 1): Ajuste solicitado
   - Use 0 quando não quer mudar a crença
   - Positivo aumenta, negativo reduz

4. **`justifications`** (lista de objetos): **OBRIGATÓRIO quando delta != 0**
   - Cada justificativa é uma crença que SUPORTA esta mudança
   - Mínimo 1 justificativa se delta != 0
   - Cada justificativa tem:
     * `content`: Afirmação que justifica o delta
     * `context`: Domínio (e.g., "evidence", "observation", "inference")
     * `confidence_estimate`: Confiança nesta justificativa (0-1)

   Exemplo:
   ```json
   {{
     "delta": 0.2,
     "justifications": [
       {{
         "content": "Stripe API retornou 504 timeout às 14:32 UTC",
         "context": "observation",
         "confidence_estimate": 0.95
       }},
       {{
         "content": "Latência de rede está variando entre 200-500ms",
         "context": "measurement",
         "confidence_estimate": 0.85
       }},
       {{
         "content": "Logs mostram 3 timeouts nas últimas 24h",
         "context": "evidence",
         "confidence_estimate": 1.0
       }}
     ]
   }}
   ```

   **Nota**: Mantenha justificativas específicas e atômicas. Chains mais profundas
   são construídas automaticamente via grafo quando justificativas referenciam outras.

**Filosofia de Precisão**:
- Faça afirmações EXATAS, não genéricas
- 0.73 é melhor que 0.7
- Erro médio de 0.15 é esperado - isso é NORMAL e BOM!
- O sistema usa seus erros para aprender padrões de calibração

**REGRAS DE CALIBRAÇÃO FORÇADA** (CRÍTICO):

1. **Primeira tentativa**: Sempre use `delta=0` quando fazer uma estimativa
   - Você PRECISA ver o erro primeiro
   - Se seu palpite estiver fora da margem, você receberá feedback de erro

2. **Após ver erro**: Na PRÓXIMA resposta você pode usar `delta != 0`
   - Agora você sabe: "Eu chutei X, era Y"
   - Use esse conhecimento para justificar o ajuste
   - DEVE fornecer justificativas explicando por que está ajustando

3. **Ciclo**:
   ```
   Turno N: Chute (delta=0) → Erro mostrado
   Turno N+1: Ajuste (delta!=0) + Justificativas → Reset
   Turno N+2: Novo chute (delta=0) → ...
   ```

Isso força **meta-cognição**: você vê seus erros antes de poder agir!

**Margem padrão**: ±{DEFAULT_CONFIDENCE_MARGIN:.2f}
"""


class ChatSession:
    """
    Interactive chat session with belief tracking

    Features:
    - LLM-powered conversation with Gemini
    - Automatic belief extraction from conversation
    - Update-on-Use learning from user feedback
    - Command system for belief inspection
    """

    def __init__(
        self,
        tracker: Optional[BeliefTracker] = None,
        model: str = "google-gla:gemini-2.0-flash",
    ):
        # Check API key
        check_gemini_api_key()

        self.tracker = tracker or BeliefTracker()
        self.model = model
        self.messages: List[Message] = []
        self.confidence_margin = DEFAULT_CONFIDENCE_MARGIN
        self._belief_margins: Dict[str, float] = {}
        self._pending_belief_id: Optional[str] = None
        self._last_created_belief_id: Optional[str] = None

        # Track estimation errors per belief to enable calibrated deltas
        self._last_estimation_error: Dict[str, float] = {}  # belief_id → error
        self._can_use_delta: Dict[str, bool] = {}  # belief_id → can adjust?

        # Agent responsável apenas por extrair crenças implícitas
        self.extraction_agent = Agent(
            model,
            output_type=BeliefExtraction,
            system_prompt="""You are a helpful AI assistant with a belief tracking system.

Your job is to:
1. Have natural conversations with the user
2. Extract implicit beliefs, assumptions, and lessons from the conversation
3. Estimate confidence for each belief based on the conversation context

When extracting beliefs:
- Focus on actionable insights, patterns, and generalizable knowledge
- Avoid trivial facts or one-time observations
- Consider the domain/context (e.g., "api_calls", "security", "ux")
- Estimate confidence [0-1] based on evidence strength and user certainty

Example beliefs:
- "External APIs can fail unexpectedly" (context: infrastructure, confidence: 0.8)
- "Users prefer simple onboarding flows" (context: ux, confidence: 0.7)
- "Input validation prevents security issues" (context: security, confidence: 0.9)

Return beliefs as a list with: content, context, confidence_estimate.
Also provide reasoning explaining why you extracted these beliefs.
"""
        )

        # Agent separado para gerar respostas via múltiplos passos
        # Using output_type forces structured output
        self.response_agent = Agent(
            model,
            output_type=MultiStepResponse,
            system_prompt=RESPONSE_SYSTEM_PROMPT,
        )

        self._ensure_seed_belief()

    async def process_message(self, user_input: str) -> AssistantReply:
        """
        Process user message and update beliefs

        Args:
            user_input: User's message

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.messages.append(Message(role="user", content=user_input))

        # Build conversation context
        context = self._build_context()

        # Get LLM response with belief extraction
        result = await self.extraction_agent.run(context)
        extraction = result.output

        # Process extracted beliefs
        last_created = None
        for extracted_belief in extraction.beliefs:
            # Add belief to tracker (ExtractedBelief is already validated by Pydantic)
            belief = self.tracker.add_belief(
                content=extracted_belief.content,
                context=extracted_belief.context,
                initial_confidence=extracted_belief.confidence_estimate,
                auto_estimate=False,  # Use LLM's estimate
            )
            last_created = belief.id

            # Auto-link to similar beliefs
            similar = self.tracker._find_knn_neighbors(belief)
            for neighbor_id in similar:
                # Check relationship with LLM
                neighbor = self.tracker.graph.beliefs[neighbor_id]
                rel = await detect_relationship(belief, neighbor)

                if rel.relation_type == "supports":
                    self.tracker.graph.link_beliefs(
                        neighbor_id, belief.id, "supports"
                    )
                elif rel.relation_type == "contradicts":
                    self.tracker.graph.link_beliefs(
                        neighbor_id, belief.id, "contradicts"
                    )

        if last_created:
            self._last_created_belief_id = last_created

        active_belief = self._select_active_belief()
        response_context = self._build_response_context(
            user_input=user_input,
            active_belief=active_belief,
            extraction=extraction,
        )

        # Get structured output from response agent (multi-step)
        result = await self.response_agent.run(response_context)
        multi_step: MultiStepResponse = result.output

        # Process each step sequentially
        reply_steps = []

        for step_idx, llm_output in enumerate(multi_step.steps):
            step_result = await self._process_single_step(
                llm_output,
                active_belief,
                step_idx=step_idx,
                total_steps=len(multi_step.steps)
            )
            reply_steps.append(step_result)

        # Build final reply with all steps
        reply = AssistantReply(steps=reply_steps)

        # Add to message history (concatenated text)
        self.messages.append(Message(
            role="assistant",
            content=reply.text,
            metadata={
                "beliefs_extracted": len(extraction.beliefs),
                "num_steps": len(reply_steps),
                "total_delta": reply.delta_requested,
                "final_confidence": reply.actual_confidence,
            }
        ))

        return reply

    async def _process_single_step(
        self,
        llm_output: ToolCallResult,
        active_belief: Belief,
        step_idx: int = 0,
        total_steps: int = 1
    ) -> ToolCallStep:
        """Process a single step of multi-step response"""

        # Validate LLM's guess and calculate error
        belief = self.tracker.graph.beliefs[active_belief.id]
        margin = self._get_margin(active_belief.id)
        actual_conf = belief.confidence
        error = actual_conf - llm_output.belief_value_guessed
        abs_error = abs(error)

        # Check if guess is outside margin
        outside_margin = abs_error > margin

        # RULE 1: If outside margin, MUST use delta=0 (force calibration first)
        if outside_margin and abs(llm_output.delta) > 0:
            # This is an error - LLM tried to adjust before seeing the error
            error_msg = (
                f"❌ ERRO DE ESTIMAÇÃO:\n"
                f"Seu palpite: {llm_output.belief_value_guessed:.3f}\n"
                f"Valor real: {actual_conf:.3f}\n"
                f"Erro: {error:+.3f} (margem: ±{margin:.2f})\n\n"
                f"Você NÃO PODE usar delta != 0 na primeira tentativa quando erra a estimativa.\n"
                f"Use delta=0 AGORA. Na PRÓXIMA resposta você poderá ajustar com justificativas.\n\n"
                f"Isso força você a:\n"
                f"1. Ver seu erro de calibração\n"
                f"2. Pensar sobre POR QUE errou\n"
                f"3. Ajustar COM JUSTIFICATIVAS baseadas nesse aprendizado"
            )
            raise ValueError(error_msg)

        # RULE 2: Can only use delta != 0 AFTER seeing an estimation error
        can_adjust = self._can_use_delta.get(active_belief.id, False)

        if abs(llm_output.delta) > 0 and not can_adjust:
            raise ValueError(
                f"Você não pode usar delta != 0 ainda!\n"
                f"Primeiro você precisa fazer uma estimativa e VER o erro.\n"
                f"Use delta=0 neste turno."
            )

        # RULE 3: If using delta, MUST provide justifications
        if abs(llm_output.delta) > 0 and not llm_output.justifications:
            raise ValueError(
                f"Delta é {llm_output.delta} mas nenhuma justificativa foi fornecida.\n"
                f"Justificativas são OBRIGATÓRIAS quando delta != 0.\n"
                f"Explique por que você está fazendo este ajuste!"
            )

        # Apply delta if provided and allowed
        applied_delta = 0.0
        justification_ids = []

        if abs(llm_output.delta) > 0:
            # Process justifications recursively
            justification_ids = await self._process_justifications(
                llm_output.justifications,
                target_belief_id=active_belief.id
            )

            # Apply delta with justifications in provenance
            update = self.tracker.apply_manual_delta(
                active_belief.id,
                llm_output.delta,
                provenance={
                    "source": "llm_structured_output",
                    "texto": llm_output.texto,
                    "guess": llm_output.belief_value_guessed,
                    "error": error,
                    "justification_ids": justification_ids,
                },
            )
            applied_delta = update.new_confidence - update.old_confidence
            actual_conf = update.new_confidence

            # Reset: after using delta, need to see error again
            self._can_use_delta[active_belief.id] = False

        # Update state: if outside margin, enable delta for next turn
        if outside_margin:
            self._last_estimation_error[active_belief.id] = error
            self._can_use_delta[active_belief.id] = True

        # Return step result
        return ToolCallStep(
            text=llm_output.texto.strip(),
            belief_id=active_belief.id,
            belief_value_guessed=llm_output.belief_value_guessed,
            delta_requested=llm_output.delta,
            applied_delta=applied_delta,
            actual_confidence=actual_conf,
            margin=margin,
            error=error,
        )

    def _build_context(self) -> str:
        """Build conversation context for LLM"""
        # Include recent messages (last 10)
        recent = self.messages[-10:]
        context_parts = []

        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content}")

        # Add current beliefs summary
        if self.tracker.graph.beliefs:
            context_parts.append("\n--- Current Belief System ---")
            top_beliefs = sorted(
                self.tracker.graph.beliefs.values(),
                key=lambda b: abs(b.confidence),
                reverse=True
            )[:5]

            for b in top_beliefs:
                stats = self.tracker.get_belief_stats(b.id)
                context_parts.append(
                    f"[{b.confidence:.2f}] {b.content} "
                    f"(certainty: {stats['certainty']:.1f})"
                )

        return "\n".join(context_parts)

    def _build_response_context(
        self,
        user_input: str,
        active_belief: Belief,
        extraction: BeliefExtraction,
    ) -> str:
        """Compose instructions for the response agent."""
        margin = self._get_margin(active_belief.id)
        history_snippet = self._recent_history_snippet()

        if extraction.beliefs:
            extracted = "\n".join(
                f"- {b.content} (ctx: {b.context}, conf≈{b.confidence_estimate:.2f})"
                for b in extraction.beliefs[:5]
            )
            extracted_section = (
                "Novas crenças extraídas nesta entrada:\n"
                f"{extracted}\n"
            )
        else:
            extracted_section = "Nenhuma nova crença explícita foi extraída nesta entrada.\n"

        return (
            f"Última mensagem do usuário: {user_input}\n\n"
            f"Crença ativa (ID {active_belief.id}): {active_belief.content}\n"
            f"Contexto: {active_belief.context}\n"
            "O valor real da confiança NÃO está disponível para você. "
            "Você deve inferir e registrar o palpite em `belief_value_guessed`.\n"
            f"Margem de tolerância permitida: ±{margin:.2f}.\n"
            "Se o sistema disser que você excedeu a margem, repita o tool com o mesmo texto e informe um delta.\n\n"
            f"{extracted_section}"
            "Histórico recente:\n"
            f"{history_snippet}\n"
            "Lembre-se de que todo texto exibido ao usuário deve estar dentro do campo `texto`."
        )

    def _recent_history_snippet(self, limit: int = 4) -> str:
        """Return a short textual summary of the last few turns."""
        snippet = []
        for msg in self.messages[-limit:]:
            label = "User" if msg.role == "user" else "Assistant"
            snippet.append(f"{label}: {msg.content}")

        return "\n".join(snippet) if snippet else "Sem histórico relevante."

    def _ensure_seed_belief(self) -> Belief:
        """Make sure there is at least one belief to anchor tool calls."""
        if self.tracker.graph.beliefs:
            # Return the most recent belief
            return max(self.tracker.graph.beliefs.values(), key=lambda b: b.updated_at)

        # Seed belief: neutral, won't be shown to user unless they ask
        seed = self.tracker.add_belief(
            content="Usuários valorizam respostas úteis e diretas",
            context="interaction",
            initial_confidence=0.8,
            auto_estimate=False,
        )
        self._last_created_belief_id = seed.id
        return seed

    def _select_active_belief(self) -> Belief:
        """Choose which belief the response tool should operate on."""
        if self._last_created_belief_id:
            belief = self.tracker.graph.beliefs.get(self._last_created_belief_id)
            if belief:
                return belief

        if self.tracker.graph.beliefs:
            return max(self.tracker.graph.beliefs.values(), key=lambda b: b.updated_at)

        return self._ensure_seed_belief()

    def _get_margin(self, belief_id: str) -> float:
        """Return the confidence margin for a belief."""
        return self._belief_margins.get(belief_id, self.confidence_margin)

    async def _process_justifications(
        self,
        justifications: List[JustificationBelief],
        target_belief_id: str,
        depth: int = 0,
        max_depth: int = 5
    ) -> List[str]:
        """
        Recursively process justifications and build justification graph.

        Each justification becomes a belief that SUPPORTS the target belief.
        Sub-justifications create deeper chains.

        Returns:
            List of belief IDs created for justifications
        """
        if depth >= max_depth:
            print(f"WARNING: Max justification depth {max_depth} reached")
            return []

        created_ids = []

        for just in justifications:
            # Create belief for this justification
            just_belief = self.tracker.add_belief(
                content=just.content,
                context=just.context,
                initial_confidence=just.confidence_estimate,
                auto_estimate=False,
            )
            created_ids.append(just_belief.id)

            # Link justification → target (SUPPORTS relationship)
            self.tracker.graph.link_beliefs(
                just_belief.id,
                target_belief_id,
                "supports"
            )

            # Note: Sub-justifications removed due to Gemini recursive schema limitation
            # Deeper chains can be built by LLM creating new justifications
            # that reference existing ones in their content

        return created_ids

    async def handle_feedback(
        self,
        belief_id: str,
        outcome: str,  # "success" or "failure"
        context: Optional[Dict] = None,
    ) -> BeliefUpdate:
        """
        Handle user feedback on belief outcome

        This triggers Update-on-Use learning.

        Args:
            belief_id: ID of belief being validated
            outcome: "success" or "failure"
            context: Additional context (relevance, quality, etc.)

        Returns:
            BeliefUpdate with update details
        """
        belief = self.tracker.graph.beliefs.get(belief_id)
        if not belief:
            raise ValueError(f"Belief {belief_id} not found")

        # Convert outcome to signal
        signal = 1.0 if outcome == "success" else 0.0

        # Get current confidence as p_hat
        p_hat = belief.confidence

        # Extract weights from context
        r = context.get("relevance", 1.0) if context else 1.0
        n = context.get("narrative", 1.0) if context else 1.0
        q = context.get("quality", 1.0) if context else 1.0

        # Perform update
        update = await self.tracker.update_belief(
            belief_id=belief_id,
            p_hat=p_hat,
            signal=signal,
            r=r, n=n, q=q,
            provenance=context or {},
        )

        return update

    def list_beliefs(self, top_n: int = 10, sort_by: str = "confidence") -> List[Dict]:
        """
        List beliefs sorted by criteria

        Args:
            top_n: Number of beliefs to return
            sort_by: "confidence", "certainty", or "recent"

        Returns:
            List of belief statistics
        """
        beliefs = list(self.tracker.graph.beliefs.values())

        if sort_by == "confidence":
            beliefs.sort(key=lambda b: abs(b.confidence), reverse=True)
        elif sort_by == "certainty":
            beliefs.sort(
                key=lambda b: sum(self.tracker.pseudo_counts.get(b.id, (1, 1))),
                reverse=True
            )

        return [
            self.tracker.get_belief_stats(b.id)
            for b in beliefs[:top_n]
        ]

    def explain_belief(self, belief_id: str) -> Dict:
        """Get detailed explanation of belief"""
        return self.tracker.explain_confidence(belief_id)

    def get_history(self, limit: int = 20) -> List[Message]:
        """Get conversation history"""
        return self.messages[-limit:]

    def export_session(self) -> Dict:
        """Export full session state"""
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "beliefs": [
                self.tracker.get_belief_stats(b.id)
                for b in self.tracker.graph.beliefs.values()
            ],
            "training_summary": self.tracker.get_training_summary(),
        }
