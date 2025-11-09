"""
Vector store for persistent embeddings using ChromaDB
"""
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import uuid
from datetime import datetime
import requests
import os


class GoogleEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using Google Gemini API via HTTPS"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-embedding-001",
        output_dimensionality: int = 768  # Recommended dimension (supports 128-3072)
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:embedContent"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of documents"""
        embeddings = []

        for text in input:
            payload = {
                "model": self.model_name,
                "content": {
                    "parts": [{
                        "text": text
                    }]
                }
            }

            # Add output dimensionality if specified
            if self.output_dimensionality:
                payload["outputDimensionality"] = self.output_dimensionality

            response = requests.post(
                self.api_url,
                params={"key": self.api_key},
                json=payload
            )

            if response.status_code != 200:
                raise Exception(f"Embedding API error: {response.status_code} - {response.text}")

            result = response.json()
            embedding = result["embedding"]["values"]
            embeddings.append(embedding)

        return embeddings


class VectorStore:
    """
    Persistent vector store using ChromaDB for facts and beliefs
    """

    def __init__(
        self,
        persist_directory: str = ".baye_data",
        use_google_embeddings: bool = True
    ):
        """
        Initialize ChromaDB client with persistence

        Args:
            persist_directory: Directory to store ChromaDB data
            use_google_embeddings: Use Google Generative AI for embeddings
                                  (requires GOOGLE_API_KEY env var)
                                  If False, uses ChromaDB default embeddings
        """
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Choose embedding function
        embedding_fn = None
        embedding_model = "default"

        if use_google_embeddings:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                try:
                    embedding_fn = GoogleEmbeddingFunction(
                        api_key=api_key,
                        model_name="models/text-embedding-004"
                    )
                    embedding_model = "google-text-embedding-004"
                except Exception as e:
                    print(f"Warning: Could not initialize Google embeddings: {e}")
                    print("Falling back to default ChromaDB embeddings")
                    embedding_fn = None
            else:
                print("Warning: GOOGLE_API_KEY not found, using default embeddings")

        self.embedding_fn = embedding_fn
        self.embedding_model = embedding_model

        self.facts_collection = self.client.get_or_create_collection(
            name="facts",
            metadata={
                "description": "User-provided facts and statements",
                "embedding_model": embedding_model
            },
            embedding_function=embedding_fn
        )

        self.beliefs_collection = self.client.get_or_create_collection(
            name="beliefs",
            metadata={
                "description": "AI beliefs tracked over time",
                "embedding_model": embedding_model
            },
            embedding_function=embedding_fn
        )

    def add_fact(
        self,
        content: str,
        fact_id: str,
        confidence: float,
        author_uuid: str,
        source_context_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a fact to the vector store

        Args:
            content: Text content of the fact
            fact_id: Unique identifier for the fact
            confidence: Confidence level (-1 to 1)
            author_uuid: UUID of the author (user)
            source_context_id: Context where fact was stated
            metadata: Additional metadata

        Returns:
            The fact_id
        """
        meta = {
            "confidence": confidence,
            "author_uuid": author_uuid,
            "source_context_id": source_context_id,
            "created_at": datetime.now().isoformat(),
            "type": "fact"
        }
        if metadata:
            meta.update(metadata)

        self.facts_collection.add(
            documents=[content],
            ids=[fact_id],
            metadatas=[meta]
        )

        return fact_id

    def add_belief(
        self,
        content: str,
        belief_id: str,
        confidence: float,
        context: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a belief to the vector store

        Args:
            content: Text content of the belief
            belief_id: Unique identifier for the belief
            confidence: Confidence level (-1 to 1)
            context: Context where belief was formed
            metadata: Additional metadata (e.g., pseudo_counts)

        Returns:
            The belief_id
        """
        meta = {
            "confidence": confidence,
            "context": context,
            "created_at": datetime.now().isoformat(),
            "type": "belief"
        }
        if metadata:
            meta.update(metadata)

        self.beliefs_collection.add(
            documents=[content],
            ids=[belief_id],
            metadatas=[meta]
        )

        return belief_id

    def update_belief(
        self,
        belief_id: str,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Update an existing belief's confidence or metadata
        """
        if confidence is not None or metadata is not None:
            # Get current metadata
            result = self.beliefs_collection.get(ids=[belief_id])
            if not result['ids']:
                raise ValueError(f"Belief {belief_id} not found")

            current_meta = result['metadatas'][0]

            if confidence is not None:
                current_meta['confidence'] = confidence

            if metadata:
                current_meta.update(metadata)

            current_meta['updated_at'] = datetime.now().isoformat()

            self.beliefs_collection.update(
                ids=[belief_id],
                metadatas=[current_meta]
            )

    def search_facts(
        self,
        query: str,
        k: int = 5,
        min_confidence: Optional[float] = None
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Search for similar facts

        Args:
            query: Query text
            k: Number of results to return
            min_confidence: Optional minimum confidence filter

        Returns:
            List of (fact_id, content, distance, metadata) tuples
        """
        results = self.facts_collection.query(
            query_texts=[query],
            n_results=k
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        facts = []
        for i, fact_id in enumerate(results['ids'][0]):
            content = results['documents'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]

            # Filter by confidence if specified
            if min_confidence is not None:
                if metadata.get('confidence', 0) < min_confidence:
                    continue

            facts.append((fact_id, content, distance, metadata))

        return facts

    def search_beliefs(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Search for similar beliefs

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (belief_id, content, distance, metadata) tuples
        """
        results = self.beliefs_collection.query(
            query_texts=[query],
            n_results=k
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        beliefs = []
        for i, belief_id in enumerate(results['ids'][0]):
            content = results['documents'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]

            beliefs.append((belief_id, content, distance, metadata))

        return beliefs

    def find_contradicting_facts(
        self,
        query: str,
        k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Find facts that might contradict the query

        Uses high similarity (same topic) as indicator of potential contradiction

        Args:
            query: Query text
            k: Number of results to return
            similarity_threshold: Minimum similarity (1 - distance) to consider

        Returns:
            List of (fact_id, content, confidence, metadata) tuples
        """
        results = self.search_facts(query, k=k * 2)  # Get more to filter

        # Filter by similarity (distance < 1 - similarity_threshold)
        contradicting = []
        max_distance = 1.0 - similarity_threshold

        for fact_id, content, distance, metadata in results:
            if distance <= max_distance:
                confidence = metadata.get('confidence', 0.0)
                contradicting.append((fact_id, content, confidence, metadata))

        # Sort by confidence and take top k
        contradicting.sort(key=lambda x: abs(x[2]), reverse=True)
        return contradicting[:k]

    def get_all_facts(self) -> List[Dict]:
        """Get all facts from the collection"""
        result = self.facts_collection.get()

        facts = []
        if result['ids']:
            for i, fact_id in enumerate(result['ids']):
                facts.append({
                    'id': fact_id,
                    'content': result['documents'][i],
                    'metadata': result['metadatas'][i]
                })

        return facts

    def get_all_beliefs(self) -> List[Dict]:
        """Get all beliefs from the collection"""
        result = self.beliefs_collection.get()

        beliefs = []
        if result['ids']:
            for i, belief_id in enumerate(result['ids']):
                beliefs.append({
                    'id': belief_id,
                    'content': result['documents'][i],
                    'metadata': result['metadatas'][i]
                })

        return beliefs

    def reset(self):
        """Clear all data (for testing)"""
        self.client.reset()

        # Recreate collections
        self.facts_collection = self.client.get_or_create_collection(
            name="facts",
            metadata={"description": "User-provided facts and statements"}
        )

        self.beliefs_collection = self.client.get_or_create_collection(
            name="beliefs",
            metadata={"description": "AI beliefs tracked over time"}
        )
