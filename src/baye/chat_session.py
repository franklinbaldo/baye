"""
ChatSession - Interactive belief tracking chat interface

Manages conversation flow, LLM interactions, and belief updates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import asyncio

from pydantic_ai import Agent, Tool, RunContext
from pydantic import BaseModel, Field

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


class BeliefExtraction(BaseModel):
    """Structured output for belief extraction from conversation"""
    beliefs: List[Dict[str, Any]]  # Each: {content, context, confidence_estimate}
    reasoning: str


@dataclass
class AssistantReply:
    """Structured assistant reply returned to the CLI."""
    text: str
    belief_id: str
    belief_value_guessed: float
    delta_requested: float
    applied_delta: float
    actual_confidence: float
    margin: float


class ToolCallResult(BaseModel):
    """Payload returned by the update_belief_tool."""
    texto: str = Field(..., description="Mensagem que será exibida para o usuário final.")
    belief_value_guessed: float = Field(
        ..., ge=0.0, le=1.0,
        description="Palpite da LLM para a confiança atual da crença ativa."
    )
    delta: float = Field(
        0.0, ge=-1.0, le=1.0,
        description="Ajuste solicitado para a crença (0 mantém o valor atual)."
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

RESPONSE_SYSTEM_PROMPT = f"""
Você é o Cogito Belief Responder para o Baye Chat CLI.

Regras inegociáveis:
1. Você NUNCA pode responder com texto livre. Toda saída deve ser uma chamada ao tool `update_belief_tool`.
2. Cada chamada DEVE preencher os três inputs obrigatórios:
   - `texto`: mensagem completa destinada ao usuário (sem placeholders ou listas separadas).
   - `belief_value_guessed`: seu melhor palpite (0–1) para a confiança atual da crença ativa.
   - `delta`: ajuste solicitado. Use 0 quando seu palpite estiver dentro da margem permitida. Se o tool reclamar que o palpite está fora da margem, repita a chamada com o MESMO `texto` e um `delta` que aproxime o valor real.
3. Não use outros tools, não gere JSON manual, não crie respostas extras após chamar o tool.
4. Sempre reflita as crenças ativas e explique brevemente no `texto` como elas influenciaram sua resposta.
5. Delta positivo aumenta a confiança; delta negativo reduz. Limite-se ao intervalo [-1, 1].

O CLI exibirá diretamente o conteúdo de `texto`, portanto mantenha tom natural, claro e honesto.
Lembre-se: se estiver em dúvida, ainda assim chame o tool com `delta=0` e explicite a incerteza no `texto`.

Margem padrão: ±{DEFAULT_CONFIDENCE_MARGIN:.2f}. A margem específica de cada crença virá no prompt do usuário.
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

        # Agent separado para gerar respostas via tool único
        self.response_agent = Agent(
            model,
            system_prompt=RESPONSE_SYSTEM_PROMPT,
            tools=[
                Tool(
                    self._tool_update_belief_response,
                    name="update_belief_tool",
                    description=(
                        "Canal único para falar com o usuário e ajustar uma crença ativa. "
                        "Sempre exige `texto`, `belief_value_guessed` e `delta`."
                    ),
                    takes_ctx=True,
                )
            ],
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
        for belief_data in extraction.beliefs:
            # Add belief to tracker
            belief = self.tracker.add_belief(
                content=belief_data["content"],
                context=belief_data.get("context", "general"),
                initial_confidence=belief_data.get("confidence_estimate"),
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

        self._pending_belief_id = active_belief.id
        try:
            tool_run = await self.response_agent.run(response_context)
        finally:
            # Keep pending ID only while the tool is executing
            self._pending_belief_id = None

        # When using tools, PydanticAI returns the tool result in .data
        # The tool function's return value is accessible via result.data
        payload = tool_run.data

        # Validate we got the right type
        if not isinstance(payload, ToolCallResult):
            raise ValueError(
                f"Expected ToolCallResult but got {type(payload).__name__}: {payload}"
            )

        reply = AssistantReply(
            text=payload.texto.strip(),
            belief_id=payload.belief_id,
            belief_value_guessed=payload.belief_value_guessed,
            delta_requested=payload.delta,
            applied_delta=payload.applied_delta,
            actual_confidence=payload.actual_confidence,
            margin=payload.margin,
        )

        self.messages.append(Message(
            role="assistant",
            content=reply.text,
            metadata={
                "beliefs_extracted": len(extraction.beliefs),
                "belief_id": reply.belief_id,
                "belief_value_guessed": reply.belief_value_guessed,
                "delta": reply.delta_requested,
                "applied_delta": reply.applied_delta,
                "actual_confidence": reply.actual_confidence,
            }
        ))

        return reply

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
                f"- {b['content']} (ctx: {b.get('context', 'general')}, conf≈{b.get('confidence_estimate', 0.5):.2f})"
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

        seed = self.tracker.add_belief(
            content="Ainda estou aprendendo sobre o usuário e preciso coletar evidências.",
            context="meta",
            initial_confidence=0.5,
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

    async def _tool_update_belief_response(
        self,
        ctx: RunContext[None],
        texto: str,
        belief_value_guessed: float,
        delta: float = 0.0,
    ) -> ToolCallResult:
        """Single tool that the LLM must call to speak/update beliefs."""
        if not texto.strip():
            raise ValueError("O campo `texto` não pode ficar vazio.")

        if not 0.0 <= belief_value_guessed <= 1.0:
            raise ValueError("`belief_value_guessed` precisa estar no intervalo [0, 1].")

        if delta < -1.0 or delta > 1.0:
            raise ValueError("`delta` deve estar entre -1 e 1.")

        belief_id = self._pending_belief_id
        if not belief_id:
            raise ValueError("Nenhuma crença ativa foi disponibilizada para este turno.")

        belief = self.tracker.graph.beliefs.get(belief_id)
        if not belief:
            raise ValueError(f"Crença {belief_id} não encontrada.")

        margin = self._get_margin(belief_id)
        actual_conf = belief.confidence
        diff = abs(actual_conf - belief_value_guessed)

        if diff > margin and abs(delta) < 1e-4:
            raise ValueError(
                f"Sua estimativa ({belief_value_guessed:.2f}) está fora da margem ±{margin:.2f} "
                f"do valor real ({actual_conf:.2f}). "
                "Repita a chamada com o MESMO `texto` e forneça um `delta` que ajuste a crença."
            )

        applied_delta = 0.0
        if abs(delta) > 0:
            update = self.tracker.apply_manual_delta(
                belief_id,
                delta,
                provenance={
                    "source": "llm_tool",
                    "texto": texto,
                    "guess": belief_value_guessed,
                },
            )
            applied_delta = update.new_confidence - update.old_confidence
            actual_conf = update.new_confidence

        return ToolCallResult(
            texto=texto,
            belief_value_guessed=belief_value_guessed,
            delta=delta,
            belief_id=belief_id,
            actual_confidence=actual_conf,
            applied_delta=applied_delta,
            margin=margin,
        )

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
