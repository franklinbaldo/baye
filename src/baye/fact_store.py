"""
Fact Store - Complete provenance tracking for all LLM context

ALL content that enters the LLM context is automatically chunked and stored
as Facts with complete provenance tracking. This creates a perfect audit trail.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import uuid


class InputMode(str, Enum):
    """How the content entered the context"""
    USER_INPUT = "user_input"  # Direct user message
    TOOL_RETURN = "tool_return"  # Return value from a tool
    SYSTEM_PROMPT = "system_prompt"  # System prompt or instructions
    DOCUMENT = "document"  # External document/file
    API_RESPONSE = "api_response"  # API call response
    MANUAL = "manual"  # Manually added fact


@dataclass
class Fact:
    """
    Complete provenance-tracked fact

    Every piece of content that enters the LLM context becomes a Fact.
    """
    # Identity
    id: str  # UUID for global reference
    seq_id: int  # Sequential ID within session (1, 2, 3, ...)

    # Content
    content: str  # The actual content (chunk)
    chunk_index: int = 0  # Index if content was chunked (0 for single chunk)
    total_chunks: int = 1  # Total chunks from same source

    # Provenance
    input_mode: InputMode  # How it entered
    author_uuid: str  # UUID of who/what created this
    source_context_id: str  # UUID of the parent context

    # Temporal
    created_at: datetime = field(default_factory=datetime.now)

    # Confidence
    confidence: float = 1.0

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"Fact(seq={self.seq_id}, id={self.id[:8]}..., "
            f"mode={self.input_mode.value})"
        )

    def to_dict(self) -> Dict:
        """Export fact as dictionary"""
        return {
            "id": self.id,
            "seq_id": self.seq_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "input_mode": self.input_mode.value,
            "author_uuid": self.author_uuid,
            "source_context_id": self.source_context_id,
            "created_at": self.created_at.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def format_structured(self) -> str:
        """Format for display"""
        lines = [
            f"[Fact #{self.seq_id}] {self.id}",
            f"Content: \"{self.content[:80]}...\"" if len(self.content) > 80 else f"Content: \"{self.content}\"",
            f"Mode: {self.input_mode.value} | Author: {self.author_uuid[:16]}...",
            f"Source: {self.source_context_id[:16]}... | Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Confidence: {self.confidence:.2f} | Chunk: {self.chunk_index + 1}/{self.total_chunks}"
        ]
        if self.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            lines.append(f"Metadata: {meta_str}")
        return "\n".join(lines)


class FactStore:
    """
    Vector store for facts with complete provenance tracking

    Features:
    - Auto-increment sequential IDs
    - Automatic text chunking
    - Tool UUID registry
    - Complete provenance tracking
    """

    def __init__(
        self,
        estimator=None,
        chunk_size: int = 500,
        user_uuid: Optional[str] = None
    ):
        self.facts: Dict[str, Fact] = {}  # UUID → Fact
        self.facts_by_seq: Dict[int, Fact] = {}  # seq_id → Fact
        self.estimator = estimator
        self.chunk_size = chunk_size

        # Sequential ID counter
        self._next_seq_id = 1

        # Tool UUID registry
        self.tool_registry: Dict[str, str] = {}

        # User UUID
        self.user_uuid = user_uuid or str(uuid.uuid4())

        # System UUID
        self.system_uuid = "system_00000000-0000-0000-0000-000000000000"

    def register_tool(self, tool_name: str, tool_uuid: Optional[str] = None) -> str:
        """Register a tool and return its UUID"""
        if tool_name in self.tool_registry:
            return self.tool_registry[tool_name]

        if tool_uuid is None:
            tool_uuid = f"tool_{tool_name}_{str(uuid.uuid4())}"

        self.tool_registry[tool_name] = tool_uuid
        return tool_uuid

    def add_context(
        self,
        content: str,
        input_mode: InputMode,
        author_uuid: str,
        source_context_id: str,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
        auto_chunk: bool = True
    ) -> List[Fact]:
        """
        Add context content to fact store (with automatic chunking)

        ALL content entering LLM context should use this method.
        """
        # Chunk content if needed
        if auto_chunk and len(content) > self.chunk_size:
            chunks = self._chunk_text(content)
        else:
            chunks = [content]

        total_chunks = len(chunks)
        created_facts = []

        for chunk_idx, chunk_content in enumerate(chunks):
            fact_id = str(uuid.uuid4())
            seq_id = self._next_seq_id
            self._next_seq_id += 1

            fact = Fact(
                id=fact_id,
                seq_id=seq_id,
                content=chunk_content,
                chunk_index=chunk_idx,
                total_chunks=total_chunks,
                input_mode=input_mode,
                author_uuid=author_uuid,
                source_context_id=source_context_id,
                confidence=confidence,
                metadata=metadata or {},
            )

            self.facts[fact_id] = fact
            self.facts_by_seq[seq_id] = fact
            created_facts.append(fact)

        return created_facts

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text by character count"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get fact by UUID"""
        return self.facts.get(fact_id)

    def get_fact_by_seq(self, seq_id: int) -> Optional[Fact]:
        """Get fact by sequential ID"""
        return self.facts_by_seq.get(seq_id)

    def find_similar(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[Fact, float]]:
        """Find k most similar facts (simple word overlap for now)"""
        similarities = []

        for fact in self.facts.values():
            similarity = self._simple_similarity(query, fact.content)
            if similarity >= min_similarity:
                similarities.append((fact, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def find_contradicting(
        self,
        content: str,
        k: int = 3,
        include_beliefs: bool = True,
        belief_graph=None
    ) -> List[Tuple[str, str, float, str]]:
        """
        Find facts (and optionally beliefs) that contradict content

        Returns:
            List of (type, id, confidence, content) tuples
        """
        contradictions = []

        # Search facts
        for fact in self.facts.values():
            similarity = self._simple_similarity(content, fact.content)
            # Low similarity might indicate contradiction
            if similarity < 0.3:
                contradictions.append((
                    "fact",
                    fact.id,
                    fact.confidence,
                    fact.content
                ))

        # Search beliefs if requested
        if include_beliefs and belief_graph:
            for belief in belief_graph.beliefs.values():
                similarity = self._simple_similarity(content, belief.content)
                if similarity < 0.3:
                    contradictions.append((
                        "belief",
                        belief.id,
                        belief.confidence,
                        belief.content
                    ))

        # Sort by confidence and take top k
        contradictions.sort(key=lambda x: x[2], reverse=True)
        return contradictions[:k]

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def list_facts(self, limit: int = 20, reverse: bool = True) -> List[Dict]:
        """List facts for display"""
        facts_list = sorted(
            self.facts.values(),
            key=lambda f: f.seq_id,
            reverse=reverse
        )[:limit]

        return [f.to_dict() for f in facts_list]

    def format_for_context(self, max_facts: int = 10) -> str:
        """Format recent facts for LLM context"""
        if not self.facts:
            return ""

        recent_facts = sorted(
            self.facts.values(),
            key=lambda f: f.created_at,
            reverse=True
        )[:max_facts]

        lines = ["📌 **KNOWN FACTS** (Ground Truth):\n"]

        for fact in recent_facts:
            source_info = f"{fact.input_mode.value}"
            lines.append(
                f"  • [#{fact.seq_id}] {fact.content[:100]}\n"
                f"    ID: {fact.id[:8]}... | Source: {source_info} | "
                f"Created: {fact.created_at.strftime('%Y-%m-%d %H:%M')}"
            )

        return "\n".join(lines)

    def export_all(self) -> List[Dict]:
        """Export all facts as dictionaries"""
        return [f.to_dict() for f in sorted(self.facts.values(), key=lambda f: f.seq_id)]
