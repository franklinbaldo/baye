# Baye System Architecture

## Overview

Baye is an epistemic belief tracking system that simulates human memory and confidence through K-NN similarity search over a vectorstore of facts and beliefs.

## System Flow Diagram

```mermaid
graph TD
    Start([User Input]) --> Store1[Store as Fact with Provenance]
    Store1 --> |user_id, timestamp ISO 8601| VectorStore[(VectorStore<br/>ChromaDB)]

    Store1 --> LLM[LLM Processing<br/>gemini-flash-latest]

    SysPrompt[System Prompt] --> |Stored at Init<br/>UUIDv5 version| VectorStore

    LLM --> |Structured Output| Claims[Multiple Claims<br/>3-5+ atomic claims]

    Claims --> Validate{Validate Each Claim}

    Validate --> KNN[K-NN Search<br/>in VectorStore]

    KNN --> |Find Similar| SimFacts[Similar Facts<br/>& Beliefs]

    SimFacts --> Estimate[Estimate Confidence<br/>from Neighbors]

    Estimate --> Compare{Compare with<br/>LLM Confidence}

    Compare --> |Within Margin| Accept[✓ Accept Claim<br/>Award Points]
    Compare --> |Outside Margin<br/>+ Has Ground Truth| Reject[✗ Reject Claim<br/>Penalty]
    Compare --> |No Ground Truth<br/>KNN ≈ 0.0| Bootstrap[Bootstrap<br/>Trust LLM]

    Accept --> StoreBelief[Store as Belief]
    Bootstrap --> StoreBelief
    Reject --> Retry[Retry with Feedback<br/>Max 5 attempts]

    StoreBelief --> |belief_id, confidence| VectorStore

    Retry --> LLM

    Accept --> Response[Generate Response]
    Bootstrap --> Response

    Response --> Output([Response to User])

    Tools[Tools: Python, Query Facts, Query Beliefs] --> |Execute Before Claims| ToolResult[Tool Results]
    ToolResult --> |Store as Fact<br/>function_id| VectorStore
    ToolResult --> LLM

    style VectorStore fill:#e1f5ff
    style LLM fill:#fff4e1
    style Accept fill:#d4edda
    style Reject fill:#f8d7da
    style Bootstrap fill:#fff3cd
```

## Detailed Component Flow

```mermaid
sequenceDiagram
    participant User
    participant ChatSession
    participant FactStore
    participant VectorStore
    participant LLM
    participant BeliefTracker
    participant ToolRegistry

    Note over ChatSession: Initialization
    ChatSession->>FactStore: Create FactStore
    ChatSession->>VectorStore: Initialize ChromaDB
    ChatSession->>FactStore: Store System Prompt as Fact
    Note right of FactStore: UUIDv5 version<br/>InputMode.SYSTEM_PROMPT
    FactStore->>VectorStore: Add with embeddings

    Note over User,BeliefTracker: User Interaction
    User->>ChatSession: "Who is president of USA?"
    ChatSession->>FactStore: Store user input as Fact
    Note right of FactStore: user_id<br/>timestamp ISO 8601<br/>InputMode.USER_INPUT
    FactStore->>VectorStore: Add with embeddings

    ChatSession->>LLM: Send context + system prompt

    alt Tool Call Requested
        LLM->>ChatSession: tool_calls + claims
        ChatSession->>ToolRegistry: Execute tool(s)
        ToolRegistry->>FactStore: Store tool result as Fact
        Note right of FactStore: function_id<br/>InputMode.TOOL_RETURN
        FactStore->>VectorStore: Add with embeddings
        ToolRegistry-->>ChatSession: Tool results
        ChatSession->>LLM: Re-run with tool results
    end

    LLM-->>ChatSession: 5+ atomic claims with confidence

    loop For Each Claim
        ChatSession->>BeliefTracker: Get or create belief
        BeliefTracker->>VectorStore: K-NN search (similar beliefs)
        VectorStore-->>BeliefTracker: K nearest neighbors
        BeliefTracker->>BeliefTracker: Estimate confidence from neighbors
        BeliefTracker-->>ChatSession: Belief with KNN confidence

        ChatSession->>ChatSession: Compare LLM vs KNN confidence

        alt Within Margin OR No Ground Truth
            ChatSession->>ChatSession: ✓ Accept claim (+points)
            ChatSession->>BeliefTracker: Store/update belief
            BeliefTracker->>VectorStore: Add belief embedding
        else Outside Margin + Has Ground Truth
            ChatSession->>ChatSession: ✗ Reject claim (penalty)
            ChatSession->>LLM: Retry with error feedback
        end
    end

    ChatSession-->>User: Response + validation results
```

## Data Flow and Provenance

```mermaid
graph LR
    subgraph Input Sources
        UI[User Input]
        SP[System Prompt]
        TR[Tool Returns]
    end

    subgraph Fact Store
        F1[Fact 1<br/>user_id, timestamp]
        F2[Fact 2<br/>prompt_version UUIDv5]
        F3[Fact 3<br/>function_id]
    end

    subgraph VectorStore ChromaDB
        E1[Embedding 1]
        E2[Embedding 2]
        E3[Embedding 3]
        E4[...]
    end

    subgraph Beliefs
        B1[Belief 1<br/>confidence, pseudo_counts]
        B2[Belief 2<br/>confidence, pseudo_counts]
    end

    UI -->|InputMode.USER_INPUT| F1
    SP -->|InputMode.SYSTEM_PROMPT| F2
    TR -->|InputMode.TOOL_RETURN| F3

    F1 -->|gemini-embedding-001<br/>768 dims| E1
    F2 -->|gemini-embedding-001<br/>768 dims| E2
    F3 -->|gemini-embedding-001<br/>768 dims| E3

    E1 --> KNN[K-NN Search]
    E2 --> KNN
    E3 --> KNN
    E4 --> KNN

    KNN -->|Estimate Confidence| B1
    KNN -->|Estimate Confidence| B2

    B1 -->|Add to VectorStore| E4
    B2 -->|Add to VectorStore| E4

    style UI fill:#e3f2fd
    style SP fill:#fff9c4
    style TR fill:#f3e5f5
    style KNN fill:#c8e6c9
```

## Claim Validation Logic

```mermaid
flowchart TD
    Start([Claim from LLM]) --> GetBelief[Get/Create Belief]

    GetBelief --> KNN[K-NN Search<br/>Similar Facts + Beliefs]

    KNN --> Estimate[Estimate Confidence<br/>from K Neighbors]

    Estimate --> CheckGT{Has Ground Truth?}

    CheckGT -->|abs(actual) > 0.2<br/>OR certainty > 3.0| HasGT[Yes - Has Evidence]
    CheckGT -->|abs(actual) ≈ 0.0<br/>AND certainty ≤ 3.0| NoGT[No - No Similar Data]

    HasGT --> CheckMargin{Within Margin?}

    CheckMargin -->|abs(error) ≤ margin| Accept1[✓ Accept<br/>Award Points]
    CheckMargin -->|abs(error) > margin| Reject[✗ Reject<br/>Penalty + Retry]

    NoGT --> Bootstrap[Bootstrap Mode<br/>Trust LLM Confidence]

    Bootstrap --> Accept2[✓ Accept<br/>Initialize Belief]

    Accept1 --> Store[Store Belief<br/>Update Pseudo-Counts]
    Accept2 --> Store

    Store --> Vector[(Add to VectorStore<br/>For Future K-NN)]

    Reject --> Feedback[Add Error Feedback<br/>to Context]
    Feedback --> Retry{Retry < 5?}
    Retry -->|Yes| LLM[Re-run LLM]
    Retry -->|No| Fail[Max Retries Failed]

    LLM --> Start

    style Accept1 fill:#d4edda
    style Accept2 fill:#d4edda
    style Reject fill:#f8d7da
    style Bootstrap fill:#fff3cd
    style Vector fill:#e1f5ff
```

## Fact Provenance Structure

```mermaid
classDiagram
    class Fact {
        +string id (UUID)
        +int seq_id
        +string content
        +InputMode input_mode
        +string author_uuid
        +string source_context_id
        +int chunk_index
        +int total_chunks
        +string created_at (ISO 8601)
        +float confidence
        +dict metadata
    }

    class InputMode {
        <<enumeration>>
        USER_INPUT
        TOOL_RETURN
        SYSTEM_PROMPT
        DOCUMENT
        API_RESPONSE
        MANUAL
    }

    class UserInputFact {
        +author_uuid: user_id
        +metadata: {user_session, ...}
    }

    class ToolReturnFact {
        +author_uuid: function_id (tool.tool_uuid)
        +metadata: {tool_name, reasoning, ...}
    }

    class SystemPromptFact {
        +author_uuid: "system"
        +metadata: {prompt_version_uuid (UUIDv5), prompt_type, model}
    }

    Fact <|-- UserInputFact
    Fact <|-- ToolReturnFact
    Fact <|-- SystemPromptFact
    Fact --> InputMode

    note for SystemPromptFact "UUIDv5: Deterministic version ID\nbased on prompt content"
```

## Chunking for Large Facts

```mermaid
flowchart LR
    Input[Large Fact<br/>> 1500 tokens] --> Estimate[Estimate Tokens<br/>~4 chars = 1 token]

    Estimate --> Split[Split on Sentences<br/>. ! ?]

    Split --> Build{Build Chunks}

    Build --> Chunk1[Chunk 1<br/>~1500 tokens<br/>overlap: 100]
    Build --> Chunk2[Chunk 2<br/>~1500 tokens<br/>overlap: 100]
    Build --> Chunk3[Chunk 3<br/>~1500 tokens<br/>overlap: 100]

    Chunk1 --> Store1[Store as Fact<br/>chunk_index: 0<br/>total_chunks: 3]
    Chunk2 --> Store2[Store as Fact<br/>chunk_index: 1<br/>total_chunks: 3]
    Chunk3 --> Store3[Store as Fact<br/>chunk_index: 2<br/>total_chunks: 3]

    Store1 --> Embed1[Embed with<br/>gemini-embedding-001]
    Store2 --> Embed2[Embed with<br/>gemini-embedding-001]
    Store3 --> Embed3[Embed with<br/>gemini-embedding-001]

    Embed1 --> VectorStore[(ChromaDB)]
    Embed2 --> VectorStore
    Embed3 --> VectorStore

    note right of Chunk2
        100 token overlap
        ensures context
        continuity
    end note

    style VectorStore fill:#e1f5ff
```

## Key Design Principles

1. **Everything is a Fact**: All input to LLM (user input, system prompt, tool results) becomes a Fact with full provenance
2. **K-NN as Memory**: Confidence comes from similarity to existing facts/beliefs in vectorstore
3. **Bootstrap Trust**: When no similar data exists (KNN ≈ 0.0), trust LLM to initialize
4. **Granular Claims**: LLM produces 3-5+ atomic claims per response, each validated independently
5. **Gamification**: Points awarded for accurate confidence calibration
6. **Chunking**: Large facts automatically chunked with overlap for embedding model (2048 token limit)
7. **Provenance**: Complete tracking with timestamps (ISO 8601), user_id, function_id, prompt versions (UUIDv5)

## Technologies

- **LLM**: Google Gemini Flash Latest (gemini-flash-latest)
- **Embeddings**: Google Gemini Embedding 001 (768 dimensions)
- **Vector DB**: ChromaDB with persistent storage
- **Framework**: PydanticAI for structured outputs
- **Language**: Python 3.13+
