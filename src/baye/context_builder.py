"""
Context pack builder for LLM consumption (US-09).

Composes concise belief cards within token budget.

Card format:
```
[BELIEF] proposition_text
Confidence: XX% | Updated: YYYY-MM-DD
Sources: source1 (r=0.9), source2 (r=0.8)
```

Features:
- Token budget enforcement
- Source prioritization (most reliable/recent)
- Tension annotations
- Factual tone (no modals)
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

from .belief_types import Belief
from .retrieval import CandidateBelief, TensionPair


# ============================================================================
# Token Estimation
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Simple heuristic: ~4 characters per token (conservative for English).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4 + 1


# ============================================================================
# Belief Card
# ============================================================================

@dataclass
class BeliefCard:
    """
    A formatted belief card for LLM context.

    Attributes:
        belief_id: Belief identifier
        text: Formatted card text
        token_count: Estimated tokens
        metadata: Additional context
    """
    belief_id: str
    text: str
    token_count: int
    metadata: Dict = None

    def __post_init__(self):
        """Calculate token count."""
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.text)


class BeliefCardFormatter:
    """
    Formats beliefs as concise cards.

    Example card:
    ```
    [BELIEF] APIs can experience unexpected downtime
    Confidence: 72% | Updated: 2025-01-08
    Sources: tool:api_monitor (r=0.9), human:sre_review (r=0.95)
    ```
    """

    def __init__(self,
                 max_sources_per_card: int = 2,
                 date_format: str = "%Y-%m-%d",
                 include_context: bool = False):
        """
        Initialize formatter.

        Args:
            max_sources_per_card: Max sources to show
            date_format: Date format string
            include_context: Include belief context/domain
        """
        self.max_sources_per_card = max_sources_per_card
        self.date_format = date_format
        self.include_context = include_context

    def format_card(self,
                   belief: Belief,
                   evidence_store=None,
                   language: str = "en") -> BeliefCard:
        """
        Format a belief as a card.

        Args:
            belief: Belief to format
            evidence_store: Optional evidence store for sources
            language: Target language (for US-11)

        Returns:
            BeliefCard
        """
        lines = []

        # Main content (factual, no modals)
        content = self._factual_tone(belief.content)
        lines.append(f"[BELIEF] {content}")

        # Confidence and date
        conf_pct = int((belief.confidence + 1.0) / 2.0 * 100)  # Map to [0, 100]
        date_str = belief.updated_at.strftime(self.date_format)
        lines.append(f"Confidence: {conf_pct}% | Updated: {date_str}")

        # Sources (if evidence available)
        if evidence_store:
            sources = self._format_sources(belief, evidence_store)
            if sources:
                lines.append(f"Sources: {sources}")

        # Optional context
        if self.include_context and belief.context != "general":
            lines.append(f"Context: {belief.context}")

        card_text = "\n".join(lines)

        return BeliefCard(
            belief_id=belief.id,
            text=card_text,
            token_count=0,  # Will be calculated in __post_init__
            metadata={'language': language}
        )

    def _factual_tone(self, content: str) -> str:
        """
        Convert to factual tone (remove modals like "might", "could").

        Args:
            content: Original content

        Returns:
            Factual version
        """
        # Simple replacements (can be expanded)
        modals = {
            "might ": "",
            "could ": "",
            "may ": "",
            "possibly ": "",
            "perhaps ": "",
            "probably ": "",
            "likely ": "",
        }

        result = content
        for modal, replacement in modals.items():
            result = result.replace(modal, replacement)

        return result.strip()

    def _format_sources(self, belief: Belief, evidence_store) -> str:
        """
        Format sources for card.

        Prioritizes by reliability and recency.

        Args:
            belief: Belief
            evidence_store: Evidence store

        Returns:
            Formatted sources string
        """
        evidences = evidence_store.get_evidence_for_belief(belief.id)

        if not evidences:
            return ""

        # Sort by reliability (from metadata) and recency
        scored_evidences = []
        for ev in evidences:
            reliability = ev.metadata.get('reliability', 0.5)
            recency = (datetime.now() - ev.created_at).total_seconds()
            # Higher score = better (high reliability, recent)
            score = reliability - (recency / 86400.0) * 0.01  # Small recency bonus
            scored_evidences.append((ev, score))

        scored_evidences.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        top_evidences = scored_evidences[:self.max_sources_per_card]

        # Format
        parts = []
        for ev, _ in top_evidences:
            reliability = ev.metadata.get('reliability', 0.5)
            parts.append(f"{ev.source} (r={reliability:.1f})")

        return ", ".join(parts)


# ============================================================================
# Context Pack Builder
# ============================================================================

class ContextPackBuilder:
    """
    Builds context packs with token budget (US-09).

    Includes:
    - Belief cards (prioritized by score)
    - Tension annotations (US-08)
    - Budget enforcement
    """

    def __init__(self,
                 formatter: Optional[BeliefCardFormatter] = None,
                 include_tensions: bool = True):
        """
        Initialize builder.

        Args:
            formatter: Card formatter (creates default if None)
            include_tensions: Include tension annotations
        """
        self.formatter = formatter or BeliefCardFormatter()
        self.include_tensions = include_tensions

    def build_pack(self,
                  ranked_beliefs: List[CandidateBelief],
                  tensions: Optional[List[TensionPair]] = None,
                  token_budget: int = 1000,
                  evidence_store=None,
                  language: str = "en") -> str:
        """
        Build context pack from ranked beliefs (US-09).

        Args:
            ranked_beliefs: Ranked candidate beliefs
            tensions: Optional tension pairs
            token_budget: Maximum tokens
            evidence_store: Evidence store for sources
            language: Target language

        Returns:
            Formatted context pack string
        """
        sections = []
        tokens_used = 0

        # Header
        header = "# Relevant Beliefs\n"
        sections.append(header)
        tokens_used += estimate_tokens(header)

        # Add belief cards in priority order
        cards_added = 0
        for candidate in ranked_beliefs:
            card = self.formatter.format_card(
                candidate.belief,
                evidence_store=evidence_store,
                language=language
            )

            # Check budget
            if tokens_used + card.token_count > token_budget:
                break  # Budget exceeded

            sections.append(card.text)
            sections.append("")  # Blank line
            tokens_used += card.token_count + 1
            cards_added += 1

        # Add tensions if any
        if self.include_tensions and tensions:
            tension_section = self._format_tensions(
                tensions,
                token_budget - tokens_used
            )

            if tension_section:
                sections.append("\n# ⚠️ Tensions (Contradictions)")
                sections.append(tension_section)

        return "\n".join(sections)

    def _format_tensions(self,
                        tensions: List[TensionPair],
                        remaining_budget: int) -> str:
        """
        Format tension annotations.

        Example:
        ```
        ⚠️ Belief A ⟷ Belief B
           A: APIs are reliable (conf: 70%)
           B: Payment API had downtime (conf: 85%)
        ```

        Args:
            tensions: Tension pairs
            remaining_budget: Remaining token budget

        Returns:
            Formatted tensions or empty string
        """
        if not tensions:
            return ""

        lines = []
        tokens_used = 0

        for tension in tensions:
            # Format tension
            a_conf = int((tension.belief_a.confidence + 1.0) / 2.0 * 100)
            b_conf = int((tension.belief_b.confidence + 1.0) / 2.0 * 100)

            tension_text = f"""
⚠️ {tension.belief_a.content[:50]}... ⟷ {tension.belief_b.content[:50]}...
   A: {tension.belief_a.content} (conf: {a_conf}%)
   B: {tension.belief_b.content} (conf: {b_conf}%)
""".strip()

            tokens = estimate_tokens(tension_text)

            if tokens_used + tokens > remaining_budget:
                break

            lines.append(tension_text)
            tokens_used += tokens

        return "\n\n".join(lines)

    def build_pack_with_metadata(self,
                                ranked_beliefs: List[CandidateBelief],
                                tensions: Optional[List[TensionPair]] = None,
                                token_budget: int = 1000,
                                evidence_store=None,
                                language: str = "en") -> Tuple[str, Dict]:
        """
        Build pack with metadata about what was included.

        Args:
            (same as build_pack)

        Returns:
            (context_pack, metadata)
            metadata includes: token_count, beliefs_included, tensions_included, etc.
        """
        pack = self.build_pack(
            ranked_beliefs,
            tensions,
            token_budget,
            evidence_store,
            language
        )

        metadata = {
            'token_count': estimate_tokens(pack),
            'token_budget': token_budget,
            'beliefs_total': len(ranked_beliefs),
            'beliefs_included': pack.count('[BELIEF]'),
            'tensions_included': len(tensions) if tensions else 0,
            'language': language,
            'timestamp': datetime.now().isoformat()
        }

        return pack, metadata


# ============================================================================
# Convenience Functions
# ============================================================================

def create_simple_context(beliefs: List[Belief],
                         max_beliefs: int = 8,
                         token_budget: int = 1000) -> str:
    """
    Create simple context pack from beliefs.

    Args:
        beliefs: Beliefs to include
        max_beliefs: Maximum number
        token_budget: Token limit

    Returns:
        Formatted context
    """
    # Mock candidates (no scoring)
    from .retrieval import CandidateBelief
    candidates = [
        CandidateBelief(belief=b, score=1.0)
        for b in beliefs[:max_beliefs]
    ]

    builder = ContextPackBuilder()
    return builder.build_pack(candidates, token_budget=token_budget)
