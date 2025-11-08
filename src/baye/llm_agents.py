"""
LLM-powered agents for belief graph intelligence using PydanticAI and Gemini.

This module provides AI-powered capabilities for:
1. Relationship detection: Identify if beliefs support/contradict each other
2. Conflict resolution: Generate nuanced beliefs to resolve contradictions
3. Semantic embeddings: Generate embeddings for similarity calculations
"""

import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from .belief_types import Belief, RelationType, BeliefID


# ============================================================================
# Structured Outputs for LLM Responses
# ============================================================================

class RelationshipAnalysis(BaseModel):
    """Analysis of relationship between two beliefs."""
    relationship: Literal["supports", "contradicts", "refines", "unrelated"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the relationship")
    explanation: str = Field(description="Brief explanation of the relationship")


class ConflictResolution(BaseModel):
    """Resolution for contradicting beliefs."""
    resolved_belief: str = Field(description="Nuanced belief that reconciles the conflict")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the resolution")
    reasoning: str = Field(description="Explanation of how this resolves the conflict")
    supports_first: bool = Field(description="Whether resolution leans toward first belief")
    supports_second: bool = Field(description="Whether resolution leans toward second belief")


class EmbeddingResult(BaseModel):
    """Semantic embedding for a belief."""
    embedding: List[float] = Field(description="Vector representation")
    dimension: int = Field(description="Embedding dimensionality")


# ============================================================================
# Relationship Detection Agent
# ============================================================================

relationship_agent = Agent(
    'google-gla:gemini-2.0-flash',
    output_type=RelationshipAnalysis,
    system_prompt="""You are an expert at analyzing logical relationships between beliefs.

Given two beliefs, determine if they:
- SUPPORT: One provides evidence or justification for the other
- CONTRADICT: They cannot both be true simultaneously
- REFINE: One is a more specific version of the other
- UNRELATED: No significant logical connection

Focus on semantic meaning, not just keyword overlap. Consider:
- Logical implications
- Causal relationships
- Contextual nuances

Be conservative: only mark as SUPPORT or CONTRADICT if the relationship is clear.
""",
)


async def detect_relationship(
    belief1: Belief,
    belief2: Belief
) -> RelationshipAnalysis:
    """
    Detect the relationship between two beliefs using LLM.

    Args:
        belief1: First belief
        belief2: Second belief

    Returns:
        RelationshipAnalysis with relationship type and confidence

    Example:
        >>> b1 = Belief("APIs can fail", 0.7, "reliability")
        >>> b2 = Belief("Always validate API responses", 0.8, "best_practices")
        >>> analysis = await detect_relationship(b1, b2)
        >>> print(analysis.relationship)  # "supports"
    """
    prompt = f"""Analyze the relationship between these two beliefs:

Belief 1:
  Content: {belief1.content}
  Context: {belief1.context}
  Confidence: {belief1.confidence}

Belief 2:
  Content: {belief2.content}
  Context: {belief2.context}
  Confidence: {belief2.confidence}

What is the logical relationship between them?
"""

    result = await relationship_agent.run(prompt)
    return result.output


# ============================================================================
# Conflict Resolution Agent
# ============================================================================

conflict_agent = Agent(
    'google-gla:gemini-2.0-flash',
    output_type=ConflictResolution,
    system_prompt="""You are an expert at resolving contradictions between beliefs.

When given contradicting beliefs, create a nuanced belief that:
1. Acknowledges valid aspects of both
2. Identifies conditions where each applies
3. Provides a balanced synthesis

Your resolution should be:
- Specific and actionable
- More sophisticated than either original belief
- Grounded in the context provided

Example:
  Belief 1: "Third-party services are reliable"
  Belief 2: "Payment APIs have unexpected downtime"
  Resolution: "Third-party services are generally reliable, but critical paths (like payments) need defensive programming"
""",
)


async def resolve_conflict(
    belief1: Belief,
    belief2: Belief,
    context: Optional[str] = None
) -> ConflictResolution:
    """
    Generate a nuanced belief that resolves contradiction.

    Args:
        belief1: First conflicting belief
        belief2: Second conflicting belief
        context: Optional additional context

    Returns:
        ConflictResolution with synthesized belief

    Example:
        >>> b1 = Belief("Services are reliable", 0.7, "infra")
        >>> b2 = Belief("APIs have downtime", 0.8, "infra")
        >>> resolution = await resolve_conflict(b1, b2)
        >>> print(resolution.resolved_belief)
    """
    prompt = f"""Resolve the contradiction between these beliefs:

Belief 1:
  Content: {belief1.content}
  Context: {belief1.context}
  Confidence: {belief1.confidence}
  Source: {belief1.source_task}

Belief 2:
  Content: {belief2.content}
  Context: {belief2.context}
  Confidence: {belief2.confidence}
  Source: {belief2.source_task}
"""

    if context:
        prompt += f"\nAdditional Context: {context}\n"

    prompt += "\nGenerate a nuanced belief that reconciles these contradictions."

    result = await conflict_agent.run(prompt)
    return result.output


# ============================================================================
# Semantic Embedding Agent
# ============================================================================

embedding_agent = Agent(
    'google-gla:gemini-2.0-flash',
    output_type=EmbeddingResult,
    system_prompt="""You generate semantic embeddings for text.

Convert the belief into a dense vector representation that captures:
- Core semantic meaning
- Contextual information
- Logical implications

Use a consistent dimensionality across all embeddings.
""",
)


async def generate_embedding(belief: Belief) -> List[float]:
    """
    Generate semantic embedding for a belief using Gemini.

    Args:
        belief: Belief to embed

    Returns:
        List of floats representing the embedding vector

    Note:
        In production, consider using dedicated embedding models
        like text-embedding-004 for better performance.
    """
    prompt = f"""Generate a semantic embedding for this belief:

Content: {belief.content}
Context: {belief.context}

Return a dense vector representation.
"""

    result = await embedding_agent.run(prompt)
    return result.output.embedding


# ============================================================================
# Batch Relationship Discovery
# ============================================================================

async def find_related_beliefs(
    new_belief: Belief,
    existing_beliefs: List[Belief],
    min_confidence: float = 0.7
) -> List[tuple[BeliefID, RelationType, float]]:
    """
    Find all beliefs related to a new belief.

    Args:
        new_belief: The belief to analyze
        existing_beliefs: Pool of existing beliefs
        min_confidence: Minimum confidence threshold for relationships

    Returns:
        List of (belief_id, relationship_type, confidence) tuples

    Example:
        >>> new = Belief("APIs timeout", 0.8, "reliability")
        >>> existing = [Belief("Validate responses", 0.7, "best_practices")]
        >>> related = await find_related_beliefs(new, existing)
        >>> print(related)  # [(belief_id, RelationType.SUPPORTS, 0.85)]
    """
    relationships = []

    for existing in existing_beliefs:
        analysis = await detect_relationship(new_belief, existing)

        if analysis.relationship != "unrelated" and analysis.confidence >= min_confidence:
            rel_type = RelationType.SUPPORTS if analysis.relationship == "supports" else RelationType.CONTRADICTS
            relationships.append((existing.id, rel_type, analysis.confidence))

    return relationships


# ============================================================================
# Configuration Check
# ============================================================================

def check_gemini_api_key() -> bool:
    """Check if Gemini API key is configured."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set. "
            "Please configure your Gemini API key."
        )
    return True
