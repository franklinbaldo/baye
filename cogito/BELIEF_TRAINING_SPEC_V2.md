# Belief Training System V2.0 - Especificação Técnica Completa

**Status**: Design Specification  
**Data**: 2025-11-08  
**Versão**: 2.0-RC1  
**Base**: Sistema V1.5 (Justification Graph) + UoU Training Loop

---

## 1. Visão Geral

Transformar o sistema de crenças V1.5 em um **agente treinável** onde:

- ✅ **Toda interação epistêmica é uma tool call** (`update_belief`)
- ✅ **Update-on-Use (UoU)** mantém pseudo-contagens com proveniência
- ✅ **K-NN semântico** fornece alvos de gradiente locais
- ✅ **Fine-tuning direto** da LLM via MSE sobre `p_hat`
- ✅ **Memória justificatória auditável** com revisão dialética

**Resultado**: Agente que aprende com seus próprios atos epistêmicos de forma disciplinada e mensurável.

---

## 2. Arquitetura de Dados

### 2.1 Estado de Crenças (Memória Justificatória)

```python
@dataclass
class BeliefState:
    """Crença com pseudo-contagens e histórico de evidências"""
    
    belief_id: str
    text: str                    # Declaração em linguagem natural
    embedding: np.ndarray        # Vetor semântico (768-dim)
    
    # Pseudo-contagens (Beta prior)
    a: float = 1.0              # Evidências pró
    b: float = 1.0              # Evidências contra
    
    # Métricas derivadas
    @property
    def confidence(self) -> float:
        """P(φ) = a / (a + b)"""
        return self.a / (self.a + self.b)
    
    @property
    def uncertainty(self) -> float:
        """Incerteza epistêmica: 1 / (a + b)"""
        return 1.0 / (self.a + self.b)
    
    # Proveniência e auditabilidade
    evidence_log: List[Evidence] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Grafo de justificação (do V1.5)
    supports: List[str] = field(default_factory=list)      # belief_ids
    contradicts: List[str] = field(default_factory=list)   # belief_ids
    
    # Metadados
    context: str = ""           # Domínio/tópico
    tags: Set[str] = field(default_factory=set)
```

### 2.2 Evidência com Proveniência

```python
@dataclass
class Evidence:
    """Registro atômico de uma observação epistêmica"""
    
    evidence_id: str
    belief_id: str
    
    # Signal de observação
    signal: float               # ∈ [0, 1], onde 1 = forte pró, 0 = forte contra
    
    # Pesos UoU
    r: float                    # Confiabilidade da fonte ∈ [0, 1]
    n: float                    # Novidade (decay temporal) ∈ [0, 1]
    q: float                    # Qualidade da evidência ∈ [0, 1]
    
    @property
    def weight(self) -> float:
        """w = r * n * q"""
        return self.r * self.n * self.q
    
    # Auditabilidade
    source: str                 # "llm_tool_call", "web_search", "user_feedback"
    provenance: Dict[str, Any]  # Detalhes específicos da fonte
    hash: str                   # Hash do conteúdo (evita duplicatas)
    timestamp: datetime
```

### 2.3 Esquema de Banco de Dados

```sql
-- Tabela de crenças
CREATE TABLE beliefs (
    belief_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB,              -- Numpy array serializado
    a REAL DEFAULT 1.0,
    b REAL DEFAULT 1.0,
    context TEXT,
    tags TEXT,                   -- JSON array
    last_updated TIMESTAMP,
    meta JSON
);

CREATE INDEX idx_beliefs_context ON beliefs(context);
CREATE INDEX idx_beliefs_updated ON beliefs(last_updated);

-- Tabela de evidências
CREATE TABLE evidence (
    evidence_id TEXT PRIMARY KEY,
    belief_id TEXT NOT NULL,
    signal REAL NOT NULL CHECK(signal >= 0 AND signal <= 1),
    r REAL NOT NULL,
    n REAL NOT NULL,
    q REAL NOT NULL,
    source TEXT NOT NULL,
    provenance JSON,
    hash TEXT UNIQUE,
    timestamp TIMESTAMP,
    FOREIGN KEY (belief_id) REFERENCES beliefs(belief_id) ON DELETE CASCADE
);

CREATE INDEX idx_evidence_belief ON evidence(belief_id);
CREATE INDEX idx_evidence_timestamp ON evidence(timestamp);
CREATE INDEX idx_evidence_hash ON evidence(hash);

-- Tabela de arestas (grafo de justificação)
CREATE TABLE edges (
    edge_id TEXT PRIMARY KEY,
    src_belief TEXT NOT NULL,
    dst_belief TEXT NOT NULL,
    edge_type TEXT CHECK(edge_type IN ('SUPPORTS', 'CONTRADICTS')),
    weight REAL DEFAULT 1.0,
    FOREIGN KEY (src_belief) REFERENCES beliefs(belief_id) ON DELETE CASCADE,
    FOREIGN KEY (dst_belief) REFERENCES beliefs(belief_id) ON DELETE CASCADE,
    UNIQUE(src_belief, dst_belief, edge_type)
);

CREATE INDEX idx_edges_src ON edges(src_belief);
CREATE INDEX idx_edges_dst ON edges(dst_belief);
```

---

## 3. Tool Única: `update_belief`

### 3.1 Interface JSON

A LLM SEMPRE chama esta tool para qualquer ato epistêmico:

```json
{
  "tool": "update_belief",
  "parameters": {
    "belief_id": "φ_42",
    "p_hat": 0.73,
    "signal": 0.85,
    "r": 0.9,
    "n": 1.0,
    "q": 0.8,
    "provenance": {
      "source": "web_search",
      "query": "current best practices for API error handling",
      "url": "https://example.com/api-patterns",
      "retrieved_at": "2025-11-08T10:30:00Z"
    }
  }
}
```

**Campos obrigatórios**:
- `belief_id`: Identificador da crença
- `p_hat`: Probabilidade subjetiva do agente ∈ [0, 1]

**Campos opcionais** (se não fornecidos, usa defaults):
- `signal`: Observação externa ∈ [0, 1] (default: null)
- `r`: Confiabilidade (default: 0.7)
- `n`: Novidade (default: 1.0)
- `q`: Qualidade (default: 0.5)
- `provenance`: Metadados (default: {})

### 3.2 Lógica da Tool (Pseudocódigo)

```python
def update_belief_tool(
    belief_id: str,
    p_hat: float,
    signal: Optional[float] = None,
    r: float = 0.7,
    n: float = 1.0,
    q: float = 0.5,
    provenance: Dict = None
) -> Dict[str, Any]:
    """
    Tool única que:
    1. Atualiza (a, b) via UoU
    2. Estima alvo p* via K-NN
    3. Retorna gradiente para treino
    """
    
    # 1. Recuperar crença
    belief = db.get_belief(belief_id)
    
    # 2. Update-on-Use (se houver signal)
    if signal is not None:
        w = r * n * q
        belief.a += w * signal
        belief.b += w * (1 - signal)
        
        # Registrar evidência
        evidence = Evidence(
            evidence_id=generate_id(),
            belief_id=belief_id,
            signal=signal,
            r=r, n=n, q=q,
            source=provenance.get("source", "unknown"),
            provenance=provenance,
            hash=hash_evidence(signal, provenance),
            timestamp=datetime.now()
        )
        db.add_evidence(evidence)
        belief.evidence_log.append(evidence)
    
    # 3. Estimar alvo via K-NN (do V1.5)
    p_knn = estimate_belief_knn(
        belief=belief,
        k=5,
        distance_metric="cosine"
    )
    
    # 4. Alvo misto com temperatura
    if signal is not None:
        lambda_ = 0.7  # Peso do signal externo
        p_star = lambda_ * signal + (1 - lambda_) * p_knn
    else:
        p_star = p_knn
    
    # Suavização: puxar para 0.5 quando consenso é fraco
    neighbors = get_k_nearest_neighbors(belief, k=5)
    mean_uncertainty = np.mean([nb.uncertainty for nb in neighbors])
    
    temperature = 0.3
    if mean_uncertainty > 0.5:
        p_star = temperature * 0.5 + (1 - temperature) * p_star
    
    # 5. Calcular loss (para logging/debugging)
    brier = (p_hat - p_star) ** 2
    confidence_weight = 1 - mean_uncertainty
    weighted_loss = brier * confidence_weight
    
    # 6. Retornar resposta
    return {
        "success": True,
        "belief_id": belief_id,
        "updated_confidence": belief.confidence,
        "updated_uncertainty": belief.uncertainty,
        "p_star": p_star,          # Alvo de treino
        "p_knn": p_knn,
        "mean_neighbor_uncertainty": mean_uncertainty,
        "brier_loss": brier,
        "weighted_loss": weighted_loss,
        "evidence_count": len(belief.evidence_log)
    }
```

---

## 4. Estimação de Alvo via K-NN Semântico

### 4.1 Algoritmo de K-NN (Refinamento do V1.5)

```python
def estimate_belief_knn(
    belief: BeliefState,
    k: int = 5,
    distance_metric: str = "cosine",
    uncertainty_weighting: bool = True
) -> float:
    """
    Estima P(φ) baseado em vizinhos semânticos
    
    Returns:
        p_knn: Probabilidade estimada ∈ [0, 1]
    """
    
    # 1. Buscar K vizinhos mais próximos
    neighbors = db.search_similar_beliefs(
        embedding=belief.embedding,
        k=k,
        exclude=[belief.belief_id]
    )
    
    if not neighbors:
        return 0.5  # Prior neutro
    
    # 2. Calcular pesos por incerteza
    if uncertainty_weighting:
        weights = []
        for nb in neighbors:
            # Mais confiança = mais peso
            w_i = 1 / (1 + nb.uncertainty)
            weights.append(w_i)
        
        # Normalizar
        weights = np.array(weights)
        weights /= weights.sum()
    else:
        weights = np.ones(len(neighbors)) / len(neighbors)
    
    # 3. Média ponderada das probabilidades
    p_knn = sum(w_i * nb.confidence for w_i, nb in zip(weights, neighbors))
    
    return float(p_knn)
```

### 4.2 Buscas Vetoriais Eficientes

Para produção, usar **ChromaDB** ou **Qdrant**:

```python
import chromadb
from chromadb.config import Settings

class BeliefVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name="beliefs",
            metadata={"hnsw:space": "cosine"}  # Índice HNSW para cosine similarity
        )
    
    def add_belief(self, belief: BeliefState):
        self.collection.add(
            ids=[belief.belief_id],
            embeddings=[belief.embedding.tolist()],
            metadatas=[{
                "text": belief.text,
                "confidence": belief.confidence,
                "uncertainty": belief.uncertainty,
                "context": belief.context
            }]
        )
    
    def search_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        exclude: List[str] = None
    ) -> List[BeliefState]:
        """
        Busca K-NN via ChromaDB (sub-linear search)
        """
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k + len(exclude or []),
            include=["metadatas", "distances"]
        )
        
        # Filtrar excluídos e reconstruir BeliefState
        beliefs = []
        for i, belief_id in enumerate(results['ids'][0]):
            if exclude and belief_id in exclude:
                continue
            
            metadata = results['metadatas'][0][i]
            belief = db.get_belief(belief_id)  # Reconstruir do banco
            beliefs.append(belief)
            
            if len(beliefs) == k:
                break
        
        return beliefs
```

---

## 5. Pipeline de Treino da LLM

### 5.1 Arquitetura de Fine-Tuning

**Objetivo**: Treinar a LLM para outputar `p_hat` calibrado com `p_star`.

```python
class BeliefCalibrationHead(nn.Module):
    """Cabeçalho auxiliar para regressão de probabilidade"""
    
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output ∈ [0, 1]
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            p_hat: (batch, 1) probabilidades
        """
        # Usar último token (antes de </tool_call>)
        last_hidden = hidden_states[:, -1, :]
        p_hat = self.projection(last_hidden)
        return p_hat
```

### 5.2 Loss Function

```python
def belief_calibration_loss(
    p_hat: torch.Tensor,      # Predição da LLM: (batch,)
    p_star: torch.Tensor,     # Alvo K-NN: (batch,)
    uncertainties: torch.Tensor,  # Incerteza dos vizinhos: (batch,)
    tension_pairs: List[Tuple[int, int]],  # Índices de pares contraditórios
    beta: float = 0.1,        # Peso da tensão dialética
    gamma: float = 0.05       # Peso da calibração
) -> torch.Tensor:
    """
    Loss total = Brier + Tensão + ECE
    """
    
    # 1. Brier Score ponderado por certeza
    brier = (p_hat - p_star) ** 2
    confidence_weight = 1 - uncertainties
    weighted_brier = (brier * confidence_weight).mean()
    
    # 2. Tensão dialética (pares CONTRADICTS)
    tension_loss = 0.0
    if tension_pairs:
        for i, j in tension_pairs:
            # Forçar que p_i + p_j ≈ 1 (consistência lógica)
            margin = torch.relu(0.1 - torch.abs((p_hat[i] + p_hat[j]) - 1.0))
            tension_loss += margin
        tension_loss /= len(tension_pairs)
    
    # 3. ECE proxy (calibração em bins)
    ece = expected_calibration_error(p_hat, p_star, num_bins=10)
    
    # Loss total
    total_loss = weighted_brier + beta * tension_loss + gamma * ece
    
    return total_loss

def expected_calibration_error(
    p_hat: torch.Tensor,
    p_star: torch.Tensor,
    num_bins: int = 10
) -> torch.Tensor:
    """Proxy do ECE para calibração"""
    
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for i in range(num_bins):
        bin_mask = (p_hat >= bin_boundaries[i]) & (p_hat < bin_boundaries[i + 1])
        
        if bin_mask.sum() > 0:
            bin_p_hat = p_hat[bin_mask].mean()
            bin_p_star = p_star[bin_mask].mean()
            bin_weight = bin_mask.float().mean()
            
            ece += bin_weight * torch.abs(bin_p_hat - bin_p_star)
    
    return ece
```

### 5.3 Training Loop (Supervised Fine-Tuning)

```python
def train_belief_calibration(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    belief_system: JustificationGraph,
    dataset: List[Dict],  # Cada item = {context, belief_id, p_hat, p_star, ...}
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 8
):
    """
    Fine-tune LLM para calibração de crenças
    """
    
    # 1. Adicionar cabeçalho de calibração
    calibration_head = BeliefCalibrationHead(
        hidden_size=model.config.hidden_size
    ).to(model.device)
    
    # 2. Usar PEFT (LoRA) para eficiência
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 3. Otimizador
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": learning_rate},
        {"params": calibration_head.parameters(), "lr": learning_rate * 10}
    ])
    
    # 4. Training loop
    for epoch in range(num_epochs):
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            
            # Tokenizar contexto + belief
            inputs = tokenizer(
                batch["context"],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)
            
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            # Predição via calibration head
            p_hat = calibration_head(hidden_states).squeeze()
            
            # Alvos (já calculados pela tool)
            p_star = torch.tensor(batch["p_star"], device=model.device)
            uncertainties = torch.tensor(batch["uncertainties"], device=model.device)
            
            # Loss
            loss = belief_calibration_loss(
                p_hat=p_hat,
                p_star=p_star,
                uncertainties=uncertainties,
                tension_pairs=batch.get("tension_pairs", []),
                beta=0.1,
                gamma=0.05
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    return model, calibration_head
```

### 5.4 Inferência com Calibração

```python
def generate_calibrated_belief_update(
    model: PreTrainedModel,
    calibration_head: BeliefCalibrationHead,
    tokenizer: PreTrainedTokenizer,
    belief_id: str,
    context: str
) -> Dict[str, Any]:
    """
    Gerar p_hat calibrado para uma crença
    """
    
    # 1. Construir prompt estruturado
    prompt = f"""<context>
{context}
</context>

<belief_id>{belief_id}</belief_id>

Call the update_belief tool with your estimated probability p_hat for this belief.

<tool_call>
{{
  "tool": "update_belief",
  "parameters": {{
    "belief_id": "{belief_id}",
    "p_hat": """
    
    # 2. Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 3. Gerar até o valor numérico
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,  # Apenas "0.73}," por exemplo
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        
        # Extrair hidden states do último token gerado
        last_hidden = outputs.hidden_states[-1][-1][:, -1, :]
        
        # Predição via calibration head
        p_hat_calibrated = calibration_head(last_hidden.unsqueeze(1)).item()
    
    # 4. Parsear resposta textual (fallback)
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Extrair p_hat do JSON
    import re
    match = re.search(r'"p_hat":\s*(0\.\d+)', generated_text)
    p_hat_text = float(match.group(1)) if match else None
    
    return {
        "p_hat_text": p_hat_text,        # Do texto gerado
        "p_hat_calibrated": p_hat_calibrated,  # Do calibration head
        "generated_text": generated_text
    }
```

---

## 6. Propagação e Equilíbrio

### 6.1 Propagação Local via UoU

Quando `update_belief` é chamada para φ:

```python
def propagate_local_update(
    belief_system: JustificationGraph,
    updated_belief_id: str,
    delta_confidence: float,
    max_hops: int = 2,
    threshold: float = 0.05
):
    """
    Propagar mudança local para vizinhos via arestas do grafo
    """
    
    visited = set()
    queue = [(updated_belief_id, delta_confidence, 0)]  # (id, delta, hop)
    
    while queue:
        belief_id, delta, hop = queue.pop(0)
        
        if belief_id in visited or hop > max_hops or abs(delta) < threshold:
            continue
        
        visited.add(belief_id)
        belief = belief_system.get_belief(belief_id)
        
        # Propagar para vizinhos SUPPORTS (atenuado)
        for neighbor_id in belief.supports:
            neighbor = belief_system.get_belief(neighbor_id)
            
            # Peso proporcional à similaridade
            similarity = cosine_similarity(belief.embedding, neighbor.embedding)
            dampening = 0.5 * similarity  # Atenuar propagação
            
            neighbor_delta = delta * dampening
            
            # Aplicar update suave (não via UoU, apenas ajuste interno)
            neighbor.a += neighbor_delta
            neighbor.b -= neighbor_delta
            
            queue.append((neighbor_id, neighbor_delta, hop + 1))
        
        # Propagar para vizinhos CONTRADICTS (invertido)
        for neighbor_id in belief.contradicts:
            neighbor = belief_system.get_belief(neighbor_id)
            similarity = cosine_similarity(belief.embedding, neighbor.embedding)
            dampening = 0.3 * similarity
            
            neighbor_delta = -delta * dampening  # Inversão
            
            neighbor.a += neighbor_delta
            neighbor.b -= neighbor_delta
            
            queue.append((neighbor_id, neighbor_delta, hop + 1))
```

### 6.2 Detecção de Equilíbrio

```python
def check_equilibrium(
    belief_system: JustificationGraph,
    window_size: int = 10,
    threshold: float = 0.01
) -> bool:
    """
    Verificar se sistema atingiu equilíbrio
    
    Critérios:
    1. Nenhuma crença mudou > threshold nas últimas N atualizações
    2. Variância de mudanças < threshold
    """
    
    recent_updates = belief_system.get_recent_updates(window_size)
    
    if len(recent_updates) < window_size:
        return False
    
    # Calcular magnitude de mudanças
    deltas = [abs(update.confidence_after - update.confidence_before) 
              for update in recent_updates]
    
    max_delta = max(deltas)
    variance = np.var(deltas)
    
    return max_delta < threshold and variance < threshold / 10
```

---

## 7. Casos Extremos e Mitigação

### 7.1 Auto-Distilação Enviesada

**Problema**: K-NN pode criar câmaras de eco.

**Mitigação**:
```python
def diversified_knn_sampling(
    belief: BeliefState,
    k: int = 5,
    diversity_weight: float = 0.3
) -> List[BeliefState]:
    """
    Balancear similaridade com diversidade de perspectivas
    """
    
    # 1. Buscar K' > K candidatos
    candidates = db.search_similar_beliefs(belief.embedding, k=k*3)
    
    # 2. Re-ranquear com diversidade
    selected = [candidates[0]]  # Sempre incluir o mais similar
    
    for _ in range(k - 1):
        scores = []
        for cand in candidates:
            if cand in selected:
                scores.append(-np.inf)
                continue
            
            # Similaridade com query
            sim_query = cosine_similarity(belief.embedding, cand.embedding)
            
            # Dissimilaridade com já selecionados
            dissim_selected = min([
                1 - cosine_similarity(cand.embedding, s.embedding)
                for s in selected
            ])
            
            # Score híbrido
            score = (1 - diversity_weight) * sim_query + diversity_weight * dissim_selected
            scores.append(score)
        
        best_idx = np.argmax(scores)
        selected.append(candidates[best_idx])
    
    return selected
```

### 7.2 Overconfidence

**Problema**: Agent pode ficar overconfident sem evidência.

**Mitigação**:
```python
def uncertainty_regularization(belief: BeliefState, alpha: float = 0.1):
    """
    Adicionar ruído epistêmico para evitar colapso
    """
    
    # Decay suave das pseudo-contagens (envelhecimento)
    days_since_update = (datetime.now() - belief.last_updated).days
    decay_factor = np.exp(-days_since_update / 30)  # Meia-vida de 30 dias
    
    belief.a *= decay_factor
    belief.b *= decay_factor
    
    # Garantir mínimo
    belief.a = max(belief.a, 1.0)
    belief.b = max(belief.b, 1.0)
    
    # Entropia mínima: puxar levemente para prior
    if belief.uncertainty < 0.05:
        belief.a = 0.95 * belief.a + 0.05 * 1.0
        belief.b = 0.95 * belief.b + 0.05 * 1.0
```

### 7.3 Tópicos Raros (Cold Start)

**Problema**: Crenças isoladas sem vizinhos.

**Mitigação**:
```python
def fallback_estimation(
    belief: BeliefState,
    k_adaptive: int = 5
) -> float:
    """
    Estratégia de fallback para crenças isoladas
    """
    
    neighbors = db.search_similar_beliefs(belief.embedding, k=k_adaptive)
    
    # Se muito poucos vizinhos, aumentar k adaptativamente
    while len(neighbors) < 3 and k_adaptive < 50:
        k_adaptive *= 2
        neighbors = db.search_similar_beliefs(belief.embedding, k=k_adaptive)
    
    # Se ainda não há vizinhos, usar prior contextual
    if not neighbors:
        # Prior baseado no contexto
        context_beliefs = db.get_beliefs_by_context(belief.context)
        
        if context_beliefs:
            return np.mean([b.confidence for b in context_beliefs])
        else:
            # Prior neutro universal
            return 0.5
    
    # K-NN normal
    return estimate_belief_knn(belief, neighbors)
```

---

## 8. Exemplo Completo de Uso

### 8.1 Inicialização

```python
from belief_system import BeliefSystem
from llm_interface import CalibratedLLM

# 1. Criar sistema
system = BeliefSystem(
    db_path="./beliefs.db",
    vector_store_path="./chroma_beliefs",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

# 2. Carregar LLM calibrada
llm = CalibratedLLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    calibration_head_path="./calibration_head.pt",
    lora_weights_path="./lora_weights"
)

# 3. Registrar tool
system.register_tool("update_belief", update_belief_tool)
```

### 8.2 Loop de Tarefa com Aprendizado

```python
async def run_task_with_learning(task: str):
    """
    Executar tarefa e aprender com o resultado
    """
    
    # 1. Executar tarefa
    result = await execute_task(task, llm)
    
    # 2. Reflexão: extrair lições
    lesson = await llm.generate(f"""
    Task: {task}
    Result: {result}
    
    What belief should be updated based on this outcome?
    Provide:
    - belief_id (or new belief text)
    - Your confidence p_hat
    - Evidence signal (if measurable)
    """)
    
    # 3. Parsear resposta
    belief_update = parse_belief_update(lesson)
    
    # 4. Chamar tool (que retorna p_star)
    tool_result = system.call_tool("update_belief", {
        "belief_id": belief_update["belief_id"],
        "p_hat": belief_update["p_hat"],
        "signal": belief_update.get("signal"),
        "provenance": {
            "source": "task_reflection",
            "task": task,
            "result": result
        }
    })
    
    # 5. Registrar para treino
    training_buffer.append({
        "context": f"Task: {task}\nResult: {result}",
        "belief_id": belief_update["belief_id"],
        "p_hat": belief_update["p_hat"],
        "p_star": tool_result["p_star"],
        "uncertainties": tool_result["mean_neighbor_uncertainty"]
    })
    
    # 6. Propagar se mudança significativa
    if abs(tool_result["updated_confidence"] - belief_update["p_hat"]) > 0.1:
        system.propagate_local_update(
            belief_id=belief_update["belief_id"],
            delta_confidence=tool_result["updated_confidence"] - belief_update["p_hat"]
        )
    
    # 7. Verificar equilíbrio
    if system.check_equilibrium():
        print("✅ Belief system reached equilibrium")
    
    return result
```

### 8.3 Batch Training Periódico

```python
async def periodic_training(
    training_buffer: List[Dict],
    model: CalibratedLLM,
    min_buffer_size: int = 100
):
    """
    Treinar modelo periodicamente com buffer acumulado
    """
    
    if len(training_buffer) < min_buffer_size:
        return
    
    print(f"🔧 Training on {len(training_buffer)} examples...")
    
    # 1. Preparar dataset
    dataset = prepare_training_dataset(training_buffer)
    
    # 2. Fine-tune
    trained_model, calibration_head = train_belief_calibration(
        model=model.base_model,
        tokenizer=model.tokenizer,
        belief_system=system,
        dataset=dataset,
        num_epochs=1,  # Incremental
        learning_rate=1e-5
    )
    
    # 3. Atualizar modelo
    model.update_weights(trained_model, calibration_head)
    
    # 4. Limpar buffer
    training_buffer.clear()
    
    print("✅ Training complete")
```

---

## 9. Métricas de Avaliação

### 9.1 Calibração do Sistema

```python
def evaluate_calibration(
    belief_system: BeliefSystem,
    test_beliefs: List[BeliefState],
    ground_truth: Dict[str, float]
) -> Dict[str, float]:
    """
    Avaliar qualidade da calibração
    """
    
    metrics = {}
    
    # 1. Expected Calibration Error (ECE)
    ece = 0.0
    num_bins = 10
    
    for i in range(num_bins):
        bin_low = i / num_bins
        bin_high = (i + 1) / num_bins
        
        bin_beliefs = [
            b for b in test_beliefs
            if bin_low <= b.confidence < bin_high
        ]
        
        if bin_beliefs:
            avg_confidence = np.mean([b.confidence for b in bin_beliefs])
            avg_accuracy = np.mean([
                ground_truth[b.belief_id] for b in bin_beliefs
            ])
            
            ece += abs(avg_confidence - avg_accuracy) * len(bin_beliefs) / len(test_beliefs)
    
    metrics["ECE"] = ece
    
    # 2. Brier Score
    brier = np.mean([
        (b.confidence - ground_truth[b.belief_id]) ** 2
        for b in test_beliefs
    ])
    metrics["Brier"] = brier
    
    # 3. Log Loss (Cross-Entropy)
    log_loss = -np.mean([
        ground_truth[b.belief_id] * np.log(b.confidence + 1e-10) +
        (1 - ground_truth[b.belief_id]) * np.log(1 - b.confidence + 1e-10)
        for b in test_beliefs
    ])
    metrics["LogLoss"] = log_loss
    
    # 4. Incerteza média
    metrics["MeanUncertainty"] = np.mean([b.uncertainty for b in test_beliefs])
    
    return metrics
```

### 9.2 Auditabilidade

```python
def audit_belief(
    belief_system: BeliefSystem,
    belief_id: str
) -> Dict[str, Any]:
    """
    Auditar histórico completo de uma crença
    """
    
    belief = belief_system.get_belief(belief_id)
    
    return {
        "belief_id": belief_id,
        "text": belief.text,
        "current_confidence": belief.confidence,
        "current_uncertainty": belief.uncertainty,
        "total_evidence": len(belief.evidence_log),
        
        # Histórico temporal
        "confidence_over_time": [
            {
                "timestamp": e.timestamp,
                "confidence_before": compute_confidence(belief.a_before, belief.b_before),
                "confidence_after": compute_confidence(belief.a, belief.b),
                "signal": e.signal,
                "source": e.source
            }
            for e in sorted(belief.evidence_log, key=lambda x: x.timestamp)
        ],
        
        # Proveniências
        "evidence_sources": Counter([e.source for e in belief.evidence_log]),
        
        # Vizinhança semântica
        "k_nearest_neighbors": [
            {
                "belief_id": nb.belief_id,
                "text": nb.text,
                "confidence": nb.confidence,
                "similarity": cosine_similarity(belief.embedding, nb.embedding)
            }
            for nb in belief_system.get_k_nearest_neighbors(belief, k=5)
        ],
        
        # Grafo de justificação
        "supports": [belief_system.get_belief(bid).text for bid in belief.supports],
        "contradicts": [belief_system.get_belief(bid).text for bid in belief.contradicts]
    }
```

---

## 10. Roadmap de Implementação

### Fase 1: Core Infrastructure (2 semanas)
- [ ] Schema de dados SQL + migrations
- [ ] ChromaDB integration para vector search
- [ ] Tool `update_belief` com UoU
- [ ] K-NN estimation básico

### Fase 2: Training Pipeline (2 semanas)
- [ ] Calibration head architecture
- [ ] Loss functions (Brier + tension + ECE)
- [ ] Training loop com PEFT/LoRA
- [ ] Inference com calibração

### Fase 3: Propagação e Equilíbrio (1 semana)
- [ ] Propagação local via grafo
- [ ] Detecção de equilíbrio
- [ ] Dampening strategies

### Fase 4: Robustez (1 semana)
- [ ] Diversified K-NN sampling
- [ ] Uncertainty regularization
- [ ] Cold-start fallbacks
- [ ] Decay temporal de evidências

### Fase 5: Avaliação e Auditoria (1 semana)
- [ ] Métricas de calibração (ECE, Brier)
- [ ] Dashboard de auditoria
- [ ] Testes de stress (loops, contradições)

### Fase 6: Produção (1 semana)
- [ ] API REST completa
- [ ] Persistência e backup
- [ ] Monitoring e alertas
- [ ] Documentação final

---

## 11. Comparação com Sistemas Existentes

| Sistema | Update-on-Use | K-NN Training | Grafo Justificatório | LLM Trainable |
|---------|---------------|---------------|----------------------|---------------|
| **Nossa V2.0** | ✅ | ✅ | ✅ | ✅ |
| TMS (Truth Maintenance) | ❌ | ❌ | ✅ | ❌ |
| SOAR | ⚠️ (chunking) | ❌ | ⚠️ (production rules) | ❌ |
| ACT-R | ⚠️ (base-level learning) | ❌ | ❌ | ❌ |
| Probabilistic Logic | ❌ | ❌ | ⚠️ (Factor graphs) | ❌ |
| Neural Episodic Memory | ❌ | ⚠️ (implicit) | ❌ | ✅ |

**Vantagem competitiva**: Único sistema que une:
- Proveniência auditável (UoU)
- Treino on-policy (K-NN)
- Estrutura simbólica (grafo)
- Aprendizado neural (LLM fine-tuning)

---

## 12. Conclusão

Este sistema V2.0 fecha o ciclo completo de um **agente epistêmico treinável**:

1. **Disciplina**: Toda ação epistêmica é uma tool call estruturada
2. **Memória justificatória**: Cada crença mantém proveniência completa
3. **Aprendizado local**: K-NN fornece alvos de gradiente contextuais
4. **Revisão dialética**: Contradições forçam consistência lógica
5. **Auditabilidade**: Histórico completo de mudanças e suas causas

**Resultado prático**: Um agente que não apenas rastreia crenças, mas **aprende a calibrar suas estimações** através de:
- Observações externas (signal)
- Sabedoria coletiva local (K-NN)
- Tensão dialética (contradições)
- Gradiente direto sobre sua própria subjetividade (p_hat)

---

**Next Steps**: Implementar Fase 1 (Core Infrastructure) como MVP.

**Contato**: Para discussões técnicas, abrir issue no repositório.

**Licença**: Apache 2.0 (open-source)
