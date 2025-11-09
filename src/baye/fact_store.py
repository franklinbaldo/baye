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
import re

from baye.vector_store import VectorStore


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
    # Identity (required)
    id: str  # UUID for global reference
    seq_id: int  # Sequential ID within session (1, 2, 3, ...)

    # Content (required)
    content: str  # The actual content (chunk)

    # Provenance (required)
    input_mode: InputMode  # How it entered
    author_uuid: str  # UUID of who/what created this
    source_context_id: str  # UUID of the parent context

    # Chunking (optional with defaults)
    chunk_index: int = 0  # Index if content was chunked (0 for single chunk)
    total_chunks: int = 1  # Total chunks from same source

    # Temporal (optional with default) - ISO 8601 format
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Confidence (optional with default)
    confidence: float = 1.0

    # Metadata (optional with default)
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
        max_tokens_per_chunk: int = 1500,  # Conservative limit (text-embedding-004 max: 2048)
        chunk_overlap_tokens: int = 100,   # Overlap for context continuity
        user_uuid: Optional[str] = None,
        persist_directory: str = ".baye_data"
    ):
        self.facts: Dict[str, Fact] = {}  # UUID → Fact (in-memory cache)
        self.facts_by_seq: Dict[int, Fact] = {}  # seq_id → Fact
        self.estimator = estimator
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap_tokens = chunk_overlap_tokens

        # Sequential ID counter
        self._next_seq_id = 1

        # Tool UUID registry
        self.tool_registry: Dict[str, str] = {}

        # User UUID
        self.user_uuid = user_uuid or str(uuid.uuid4())

        # System UUID
        self.system_uuid = "system_00000000-0000-0000-0000-000000000000"

        # Persistent vector store
        self.vector_store = VectorStore(persist_directory=persist_directory)

        # Load existing facts from vector store on init
        self._load_existing_facts()

    def _load_existing_facts(self):
        """Load facts from persistent vector store into memory"""
        stored_facts = self.vector_store.get_all_facts()

        for fact_data in stored_facts:
            fact_id = fact_data['id']
            metadata = fact_data['metadata']

            # Reconstruct Fact object
            fact = Fact(
                id=fact_id,
                seq_id=metadata.get('seq_id', 0),
                content=fact_data['content'],
                chunk_index=metadata.get('chunk_index', 0),
                total_chunks=metadata.get('total_chunks', 1),
                input_mode=InputMode(metadata.get('input_mode', 'manual')),
                author_uuid=metadata.get('author_uuid', self.user_uuid),
                source_context_id=metadata.get('source_context_id', ''),
                confidence=metadata.get('confidence', 1.0),
                metadata=metadata.get('extra_metadata', {}),
                created_at=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat()))
            )

            # Add to in-memory dictionaries
            self.facts[fact_id] = fact
            self.facts_by_seq[fact.seq_id] = fact

            # Update seq_id counter
            if fact.seq_id >= self._next_seq_id:
                self._next_seq_id = fact.seq_id + 1

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
        # Chunk content if needed (based on token estimate)
        estimated_tokens = self._estimate_tokens(content)
        if auto_chunk and estimated_tokens > self.max_tokens_per_chunk:
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

            # Add to in-memory cache
            self.facts[fact_id] = fact
            self.facts_by_seq[seq_id] = fact
            created_facts.append(fact)

            # Persist to vector store
            # Flatten metadata (ChromaDB doesn't support nested dicts)
            flat_metadata = {
                'seq_id': seq_id,
                'chunk_index': chunk_idx,
                'total_chunks': total_chunks,
                'input_mode': input_mode.value,
            }
            # Add user metadata with prefix to avoid conflicts
            if metadata:
                for k, v in metadata.items():
                    # Only add simple types (str, int, float, bool)
                    if isinstance(v, (str, int, float, bool)):
                        flat_metadata[f'meta_{k}'] = v

            self.vector_store.add_fact(
                content=chunk_content,
                fact_id=fact_id,
                confidence=confidence,
                author_uuid=author_uuid,
                source_context_id=source_context_id,
                metadata=flat_metadata
            )

        return created_facts

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ≈ 4 chars)
        For more accuracy, use tiktoken, but this is good enough for chunking
        """
        return len(text) // 4

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text by token count with overlap (for embedding model context window)

        Based on ChromaDB best practices:
        - Chunks should fit within embedding model's context window (2048 tokens for text-embedding-004)
        - Use overlap to maintain context continuity
        - Split on sentence boundaries when possible
        """
        estimated_tokens = self._estimate_tokens(text)

        # If text fits in one chunk, return as-is
        if estimated_tokens <= self.max_tokens_per_chunk:
            return [text]

        chunks = []

        # Split by sentences first (better semantic boundaries)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # If single sentence exceeds max, split by words
            if sentence_tokens > self.max_tokens_per_chunk:
                # Flush current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                # Split long sentence by words
                words = sentence.split()
                for word in words:
                    word_tokens = self._estimate_tokens(word)
                    if current_tokens + word_tokens > self.max_tokens_per_chunk:
                        chunks.append(current_chunk.strip())
                        # Add overlap from end of previous chunk
                        overlap_words = current_chunk.split()[-self.chunk_overlap_tokens:]
                        current_chunk = " ".join(overlap_words) + " " + word
                        current_tokens = self._estimate_tokens(current_chunk)
                    else:
                        current_chunk += " " + word
                        current_tokens += word_tokens

            # Normal case: add sentence to current chunk
            elif current_tokens + sentence_tokens <= self.max_tokens_per_chunk:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens

            # Chunk is full, start new one with overlap
            else:
                chunks.append(current_chunk.strip())

                # Create overlap from end of previous chunk
                overlap_text = " ".join(current_chunk.split()[-self.chunk_overlap_tokens:])
                current_chunk = overlap_text + " " + sentence
                current_tokens = self._estimate_tokens(current_chunk)

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

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
        Find facts (and optionally beliefs) that MIGHT contradict content

        Uses vector store semantic search to find related facts.
        The caller must determine if they actually contradict.

        Returns:
            List of (type, id, confidence, content) tuples sorted by confidence
        """
        candidates = []

        # Search facts using vector store (semantic similarity)
        similar_facts = self.vector_store.find_contradicting_facts(
            query=content,
            k=k * 2,  # Get more to filter
            similarity_threshold=0.5  # Moderate similarity threshold
        )

        for fact_id, fact_content, confidence, metadata in similar_facts:
            candidates.append((
                "fact",
                fact_id,
                confidence,
                fact_content,
                1.0  # Placeholder for similarity (vector store uses distance)
            ))

        # Search beliefs if requested (still using simple similarity for now)
        if include_beliefs and belief_graph:
            for belief in belief_graph.beliefs.values():
                similarity = self._simple_similarity(content, belief.content)
                if similarity > 0.3:
                    candidates.append((
                        "belief",
                        belief.id,
                        belief.confidence,
                        belief.content,
                        similarity
                    ))

        # Sort by similarity first (most relevant), then by confidence
        candidates.sort(key=lambda x: (x[4], abs(x[2])), reverse=True)

        # Return without similarity score
        return [(t, i, c, txt) for t, i, c, txt, _ in candidates[:k]]

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
