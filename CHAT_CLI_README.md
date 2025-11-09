# 🧠 Baye Chat CLI - Belief Tracking Chatbot

An interactive AI chat interface with **epistemic belief tracking** powered by Gemini and Update-on-Use learning.

## What is this?

A conversational AI that:
- **Extracts beliefs** automatically from conversations
- **Tracks confidence** using Bayesian pseudo-counts (Beta distribution)
- **Updates beliefs** via Update-on-Use when you provide feedback
- **Propagates changes** through a justification graph
- **Learns from neighbors** using K-NN gradient estimation

This combines:
- **Baye V1.5**: Justification graph with causal + semantic propagation
- **Cogito V2.0**: Update-on-Use learning with pseudo-counts
- **PydanticAI + Gemini**: Structured LLM outputs

## Quick Start

### 1. Setup

```bash
# Clone and enter directory
cd /home/frank/workspace/baye

# Install dependencies
uv sync

# Set API key
export GOOGLE_API_KEY="your-gemini-api-key"
# or
export GOOGLE_API_KEY="$GEMINI_API_KEY"  # if using workspace .envrc
```

### 2. Run

```bash
# Start the chat
uv run baye-chat

# Or use Python directly
uv run python -m baye.cli
```

### 3. Chat!

```
You: I tried calling the Stripe API and it timed out after 30 seconds

🤖 Assistant: I've learned 2 new beliefs:
  • External APIs can experience timeouts (confidence: 0.75)
  • Payment service reliability requires defensive coding (confidence: 0.80)

That's frustrating! Stripe's API is generally reliable, but timeouts do happen...
[Belief abc123de | Palpite: 0.75 | Valor registrado: 0.75 | Margem: ±0.10]
```

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/beliefs [N]` | List top N beliefs (default: 10) | `/beliefs 5` |
| `/explain <id>` | Explain belief confidence calculation | `/explain abc123` |
| `/feedback <id> <outcome>` | Update belief based on outcome | `/feedback abc123 success` |
| `/history [N]` | Show conversation history | `/history 20` |
| `/export` | Export session to JSON | `/export` |
| `/help` | Show help message | `/help` |
| `/quit` | Exit | `/quit` |

## How Beliefs Work

### Confidence Representation

Each belief has:
- **Confidence**: `p = α / (α + β)` where `(α, β)` are pseudo-counts
- **Certainty**: `α + β` (total pseudo-count = amount of evidence)
- **Variance**: How stable the confidence is

Example:
```python
Belief: "APIs can timeout"
α = 4.2, β = 1.8
→ confidence = 4.2 / 6.0 = 0.70
→ certainty = 6.0 (moderate evidence)
→ variance = 0.039 (low = stable)
```

### Update-on-Use

When you give feedback (`/feedback <id> success|failure`):

1. **Signal**: 1.0 for success, 0.0 for failure
2. **Update pseudo-counts**:
   ```python
   α_new = α + weight × signal
   β_new = β + weight × (1 - signal)
   ```
3. **K-NN gradient**: Blend signal with similar beliefs
   ```python
   p_star = 0.7 × signal + 0.3 × mean(neighbors)
   ```
4. **Training loss**: `(p_hat - p_star)² × certainty`
5. **Propagate**: Update dependent beliefs in graph

### Example Session

```
You: I need to implement user authentication