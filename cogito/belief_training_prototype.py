"""
Belief Training System V2.0 - Minimal Prototype
Update-on-Use + K-NN Gradient Estimation

Demonstra:
1. Tool única update_belief com UoU
2. Estimação de alvo via K-NN
3. Loss calculation para treino
4. Propagação local
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import json


@dataclass
class Evidence:
    """Registro de evidência com proveniência"""
    evidence_id: str
    belief_id: str
    signal: float  # [0, 1]
    r: float  # Confiabilidade
    n: float  # Novidade
    q: float  # Qualidade
    source: str
    provenance: Dict[str, Any]
    timestamp: datetime
    
    @property
    def weight(self) -> float:
        return self.r * self.n * self.q


@dataclass
class BeliefState:
    """Crença com pseudo-contagens e memória justificatória"""
    belief_id: str
    text: str
    embedding: np.ndarray
    
    # Pseudo-contagens Beta(a, b)
    a: float = 1.0
    b: float = 1.0
    
    # Histórico
    evidence_log: List[Evidence] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Grafo de justificação
    supports: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    
    # Metadados
    context: str = ""
    
    @property
    def confidence(self) -> float:
        """P(φ) = a / (a + b)"""
        return self.a / (self.a + self.b)
    
    @property
    def uncertainty(self) -> float:
        """Incerteza: 1 / (a + b)"""
        return 1.0 / (self.a + self.b)


class BeliefSystem:
    """Sistema minimalista de crenças com treino UoU + K-NN"""
    
    def __init__(self, embedding_dim: int = 384):
        self.beliefs: Dict[str, BeliefState] = {}
        self.embedding_dim = embedding_dim
        self.training_buffer: List[Dict] = []
        
    def add_belief(
        self,
        belief_id: str,
        text: str,
        embedding: Optional[np.ndarray] = None,
        initial_confidence: float = 0.5
    ) -> BeliefState:
        """Adicionar nova crença"""
        
        if embedding is None:
            # Embedding fake para demo (normalizado)
            embedding = np.random.randn(self.embedding_dim)
            embedding /= np.linalg.norm(embedding)
        
        # Calcular (a, b) para confidence inicial
        # P = a/(a+b) = initial_confidence
        # Assumindo low uncertainty inicial: a+b = 2
        total = 2.0
        a = initial_confidence * total
        b = (1 - initial_confidence) * total
        
        belief = BeliefState(
            belief_id=belief_id,
            text=text,
            embedding=embedding,
            a=a,
            b=b,
            context=""
        )
        
        self.beliefs[belief_id] = belief
        return belief
    
    def get_belief(self, belief_id: str) -> Optional[BeliefState]:
        """Recuperar crença"""
        return self.beliefs.get(belief_id)
    
    def search_similar_beliefs(
        self,
        embedding: np.ndarray,
        k: int = 5,
        exclude: Optional[List[str]] = None
    ) -> List[BeliefState]:
        """Buscar K vizinhos mais próximos (cosine similarity)"""
        
        exclude = exclude or []
        
        # Calcular similaridades
        similarities = []
        for belief_id, belief in self.beliefs.items():
            if belief_id in exclude:
                continue
            
            # Cosine similarity
            sim = np.dot(embedding, belief.embedding)
            similarities.append((sim, belief))
        
        # Ordenar e pegar top-K
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [belief for _, belief in similarities[:k]]
    
    def estimate_belief_knn(
        self,
        belief: BeliefState,
        k: int = 5,
        uncertainty_weighting: bool = True
    ) -> float:
        """Estimar P(φ) via K-NN"""
        
        neighbors = self.search_similar_beliefs(
            embedding=belief.embedding,
            k=k,
            exclude=[belief.belief_id]
        )
        
        if not neighbors:
            return 0.5  # Prior neutro
        
        if uncertainty_weighting:
            # Peso inversamente proporcional à incerteza
            weights = []
            for nb in neighbors:
                w = 1.0 / (1.0 + nb.uncertainty)
                weights.append(w)
            
            # Normalizar
            weights = np.array(weights)
            weights /= weights.sum()
        else:
            weights = np.ones(len(neighbors)) / len(neighbors)
        
        # Média ponderada
        p_knn = sum(w * nb.confidence for w, nb in zip(weights, neighbors))
        
        return float(p_knn)
    
    def update_belief_tool(
        self,
        belief_id: str,
        p_hat: float,
        signal: Optional[float] = None,
        r: float = 0.7,
        n: float = 1.0,
        q: float = 0.5,
        provenance: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Tool única para update epistêmico
        
        Retorna:
            - p_star: Alvo para treino
            - loss: Brier loss para logging
            - uncertainties: Incerteza dos vizinhos
        """
        
        belief = self.get_belief(belief_id)
        if belief is None:
            raise ValueError(f"Belief {belief_id} not found")
        
        confidence_before = belief.confidence
        
        # 1. Update-on-Use (se houver signal)
        if signal is not None:
            w = r * n * q
            belief.a += w * signal
            belief.b += w * (1.0 - signal)
            
            # Registrar evidência
            evidence = Evidence(
                evidence_id=f"evid_{len(belief.evidence_log)}",
                belief_id=belief_id,
                signal=signal,
                r=r, n=n, q=q,
                source=provenance.get("source", "unknown") if provenance else "unknown",
                provenance=provenance or {},
                timestamp=datetime.now()
            )
            belief.evidence_log.append(evidence)
            belief.last_updated = datetime.now()
        
        # 2. Estimar alvo via K-NN
        p_knn = self.estimate_belief_knn(belief, k=5)
        
        # 3. Alvo misto (signal externo + consenso local)
        lambda_signal = 0.7
        if signal is not None:
            p_star = lambda_signal * signal + (1 - lambda_signal) * p_knn
        else:
            p_star = p_knn
        
        # 4. Suavização com temperatura (se consenso fraco)
        neighbors = self.search_similar_beliefs(belief.embedding, k=5, exclude=[belief_id])
        mean_uncertainty = np.mean([nb.uncertainty for nb in neighbors]) if neighbors else 0.5
        
        temperature = 0.3
        if mean_uncertainty > 0.5:
            p_star = temperature * 0.5 + (1 - temperature) * p_star
        
        # 5. Calcular loss (Brier ponderado)
        brier = (p_hat - p_star) ** 2
        confidence_weight = 1.0 - mean_uncertainty
        weighted_loss = brier * confidence_weight
        
        # 6. Adicionar ao buffer de treino
        self.training_buffer.append({
            "belief_id": belief_id,
            "context": f"Belief: {belief.text}",
            "p_hat": p_hat,
            "p_star": p_star,
            "uncertainties": mean_uncertainty,
            "signal": signal,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "belief_id": belief_id,
            "confidence_before": confidence_before,
            "confidence_after": belief.confidence,
            "uncertainty": belief.uncertainty,
            "p_star": p_star,
            "p_knn": p_knn,
            "mean_neighbor_uncertainty": mean_uncertainty,
            "brier_loss": brier,
            "weighted_loss": weighted_loss,
            "evidence_count": len(belief.evidence_log)
        }
    
    def propagate_local_update(
        self,
        belief_id: str,
        delta: float,
        max_hops: int = 2,
        threshold: float = 0.05
    ):
        """Propagar mudança para vizinhos via grafo"""
        
        visited = set()
        queue = [(belief_id, delta, 0)]
        
        while queue:
            current_id, current_delta, hop = queue.pop(0)
            
            if current_id in visited or hop > max_hops or abs(current_delta) < threshold:
                continue
            
            visited.add(current_id)
            belief = self.get_belief(current_id)
            
            # Propagar para supporters (atenuado)
            for neighbor_id in belief.supports:
                neighbor = self.get_belief(neighbor_id)
                if neighbor is None:
                    continue
                
                # Similaridade semântica
                similarity = np.dot(belief.embedding, neighbor.embedding)
                dampening = 0.5 * similarity
                
                neighbor_delta = current_delta * dampening
                
                # Aplicar update suave
                neighbor.a += neighbor_delta
                neighbor.b -= neighbor_delta
                
                # Garantir não-negatividade
                neighbor.a = max(0.1, neighbor.a)
                neighbor.b = max(0.1, neighbor.b)
                
                queue.append((neighbor_id, neighbor_delta, hop + 1))
            
            # Propagar para contradictors (invertido)
            for neighbor_id in belief.contradicts:
                neighbor = self.get_belief(neighbor_id)
                if neighbor is None:
                    continue
                
                similarity = np.dot(belief.embedding, neighbor.embedding)
                dampening = 0.3 * similarity
                
                neighbor_delta = -current_delta * dampening
                
                neighbor.a += neighbor_delta
                neighbor.b -= neighbor_delta
                
                neighbor.a = max(0.1, neighbor.a)
                neighbor.b = max(0.1, neighbor.b)
                
                queue.append((neighbor_id, neighbor_delta, hop + 1))
    
    def add_edge(self, src_id: str, dst_id: str, edge_type: str):
        """Adicionar aresta no grafo de justificação"""
        
        src = self.get_belief(src_id)
        dst = self.get_belief(dst_id)
        
        if src is None or dst is None:
            raise ValueError("Belief not found")
        
        if edge_type == "SUPPORTS":
            if dst_id not in src.supports:
                src.supports.append(dst_id)
        elif edge_type == "CONTRADICTS":
            if dst_id not in src.contradicts:
                src.contradicts.append(dst_id)
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")
    
    def calculate_training_metrics(self) -> Dict[str, float]:
        """Calcular métricas agregadas do buffer de treino"""
        
        if not self.training_buffer:
            return {}
        
        p_hats = [sample["p_hat"] for sample in self.training_buffer]
        p_stars = [sample["p_star"] for sample in self.training_buffer]
        
        # Brier score médio
        brier_scores = [(ph - ps) ** 2 for ph, ps in zip(p_hats, p_stars)]
        mean_brier = np.mean(brier_scores)
        
        # ECE aproximado (3 bins)
        ece = 0.0
        for i in range(3):
            bin_low = i / 3
            bin_high = (i + 1) / 3
            
            bin_samples = [
                (ph, ps) for ph, ps in zip(p_hats, p_stars)
                if bin_low <= ph < bin_high
            ]
            
            if bin_samples:
                avg_p_hat = np.mean([ph for ph, _ in bin_samples])
                avg_p_star = np.mean([ps for _, ps in bin_samples])
                ece += abs(avg_p_hat - avg_p_star) * len(bin_samples) / len(self.training_buffer)
        
        return {
            "mean_brier": mean_brier,
            "ece": ece,
            "buffer_size": len(self.training_buffer),
            "mean_p_hat": np.mean(p_hats),
            "mean_p_star": np.mean(p_stars),
            "std_p_hat": np.std(p_hats)
        }


def demo_scenario():
    """Demonstração completa do sistema"""
    
    print("=" * 60)
    print("Belief Training System V2.0 - Demo")
    print("=" * 60)
    
    system = BeliefSystem(embedding_dim=4)  # Dim baixa para demo
    
    # 1. Criar crenças iniciais
    print("\n1. Criando crenças iniciais...")
    
    beliefs_data = [
        ("φ1", "APIs externas podem falhar", 0.6),
        ("φ2", "Sempre validar input de usuários", 0.8),
        ("φ3", "Usar try-catch para operações I/O", 0.7),
        ("φ4", "Confiar em serviços autenticados", 0.5),
        ("φ5", "Logs ajudam no debugging", 0.9),
    ]
    
    for bid, text, conf in beliefs_data:
        belief = system.add_belief(bid, text, initial_confidence=conf)
        print(f"  ✓ {bid}: {text[:40]}... (conf={belief.confidence:.2f})")
    
    # 2. Adicionar arestas (grafo de justificação)
    print("\n2. Construindo grafo de justificação...")
    
    edges = [
        ("φ1", "φ3", "SUPPORTS"),    # Falhas → usar try-catch
        ("φ2", "φ3", "SUPPORTS"),    # Validar input → try-catch
        ("φ1", "φ4", "CONTRADICTS"), # Falhas ↔ confiar em serviços
        ("φ3", "φ5", "SUPPORTS"),    # Try-catch → logs
    ]
    
    for src, dst, edge_type in edges:
        system.add_edge(src, dst, edge_type)
        print(f"  ✓ {src} --[{edge_type}]--> {dst}")
    
    # 3. Simular aprendizado via tool
    print("\n3. Simulando atualizações epistêmicas...")
    
    # Cenário: Agent falhou por API timeout
    print("\n  Cenário: Timeout em API externa")
    
    result = system.update_belief_tool(
        belief_id="φ1",
        p_hat=0.75,  # Agent acha que ficou mais confiante
        signal=0.9,  # Evidência forte (houve falha)
        r=0.8,       # Alta confiabilidade (observação direta)
        n=1.0,       # Totalmente nova
        q=0.9,       # Alta qualidade (não ambígua)
        provenance={
            "source": "task_execution",
            "error": "TimeoutError",
            "task": "fetch_user_data"
        }
    )
    
    print(f"  ✓ Confidence: {result['confidence_before']:.3f} → {result['confidence_after']:.3f}")
    print(f"    p_hat={result['p_star']:.3f} (K-NN alvo)")
    print(f"    Brier loss={result['brier_loss']:.4f}")
    print(f"    Evidências={result['evidence_count']}")
    
    # 4. Propagar mudança
    print("\n4. Propagando atualização...")
    
    delta = result['confidence_after'] - result['confidence_before']
    system.propagate_local_update("φ1", delta=delta)
    
    print(f"  ✓ Propagado Δ={delta:.3f}")
    print(f"    φ3 (supports): {system.get_belief('φ3').confidence:.3f}")
    print(f"    φ4 (contradicts): {system.get_belief('φ4').confidence:.3f}")
    
    # 5. Mais uma rodada
    print("\n  Cenário: Validação de input salvou de crash")
    
    result2 = system.update_belief_tool(
        belief_id="φ2",
        p_hat=0.85,
        signal=0.95,
        r=0.9,
        n=1.0,
        q=0.95,
        provenance={
            "source": "task_execution",
            "prevented": "SQL injection attempt",
            "task": "process_search_query"
        }
    )
    
    print(f"  ✓ Confidence: {result2['confidence_before']:.3f} → {result2['confidence_after']:.3f}")
    print(f"    Brier loss={result2['brier_loss']:.4f}")
    
    # 6. Métricas de treino
    print("\n5. Métricas agregadas do buffer de treino...")
    
    metrics = system.calculate_training_metrics()
    print(f"  Buffer size: {metrics['buffer_size']}")
    print(f"  Mean Brier: {metrics['mean_brier']:.4f}")
    print(f"  ECE (3 bins): {metrics['ece']:.4f}")
    print(f"  Mean p_hat: {metrics['mean_p_hat']:.3f}")
    print(f"  Mean p_star: {metrics['mean_p_star']:.3f}")
    
    # 7. Estado final das crenças
    print("\n6. Estado final das crenças:")
    
    for bid in ["φ1", "φ2", "φ3", "φ4", "φ5"]:
        belief = system.get_belief(bid)
        print(f"  {bid}: conf={belief.confidence:.3f}, u={belief.uncertainty:.3f}, "
              f"evid={len(belief.evidence_log)}")
    
    # 8. Exemplo de dado de treino
    print("\n7. Exemplo de sample do training buffer:")
    print(json.dumps(system.training_buffer[0], indent=2, default=str))
    
    print("\n" + "=" * 60)
    print("✅ Demo completo!")
    print("=" * 60)


if __name__ == "__main__":
    demo_scenario()
