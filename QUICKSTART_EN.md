# 🚀 Quickstart: Running Baye on Your Machine

## Prerequisites

- Python 3.10+ installed
- `uv` installed ([instructions](https://docs.astral.sh/uv/getting-started/installation/))
- Google Gemini API key ([get it here](https://aistudio.google.com/app/apikey))

## Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/franklinbaldo/baye.git
cd baye

# Install dependencies with uv
uv sync
```

**Expected output:**
```
Using CPython 3.13.5
Creating virtual environment at: .venv
Resolved 144 packages in 2ms
Installed 132 packages in 1.33s
✓ baye==1.5.0
```

## Step 2: Configure API Key

```bash
# Option 1: Direct export
export GOOGLE_API_KEY="your-gemini-api-key"

# Option 2: Use direnv (if configured)
# source .envrc

# Option 3: Create .env in project
echo 'GOOGLE_API_KEY="your-key"' > .env
```

**Verify:**
```bash
echo $GOOGLE_API_KEY
# Should show your API key
```

## Step 3: Run the Example

```bash
# Run the complete example with LLM
export GOOGLE_API_KEY="your-key"
uv run python examples/example_llm_integration.py
```

**What you'll see:**

```
🧠 Belief Tracking with PydanticAI + Gemini
======================================================================

📖 Scenario: Stripe API Failure

Initial beliefs:
  B1: Third-party payment services are generally reliable (conf: 0.7)
  B2: Always validate and handle API responses gracefully (conf: 0.6)
  B3: Established services like Stripe don't need defensive programming (conf: 0.4)

💥 Incident: Stripe API returned 500 errors during checkout flow

🔍 Step 1: Detecting relationships with existing beliefs...

  • CONTRADICTS B1
    Confidence: 0.70
    → Third-party payment services are generally reliable...

  • SUPPORTS B2
    Confidence: 0.70
    → Always validate and handle API responses gracefully...

  • CONTRADICTS B3
    Confidence: 0.75
    → Established services like Stripe don't need defensive progra...

🔬 Step 2: Analyzing relationship with B1...

  Relationship: CONTRADICTS
  Confidence: 0.60
  Explanation: A 500 error from Stripe directly contradicts the general
               belief that third-party payment services are reliable.

🤝 Step 3: Resolving contradiction between lesson and B1...

  Resolved Belief:
    "While third-party payment services are generally reliable, specific
     incidents like Stripe API returning 500 errors during checkout flows
     can occur and severely impact revenue. Robust error handling and
     monitoring are essential in production environments."

  Confidence: 0.80
  Reasoning: Acknowledges general reliability while addressing specific
             failure. Proposes actionable steps.

  Supports lesson: True
  Supports original: True

✨ Step 4: Creating nuanced belief from resolution...

  New Belief ID: e77debff
  Content: While third-party payment services are generally reliable...
  Confidence: 0.80
  Context: learned_wisdom

🔗 Step 5: Analyzing support relationship...

  Lesson: Stripe API returned 500 errors during checkout flow
  Related: Network calls can fail at any time

  Relationship: SUPPORTS
  Confidence: 0.70
  Explanation: The 500 errors are a specific instance supporting the
               general belief that network calls can fail at any time.

======================================================================

✅ Demo Complete!

Key Takeaways:
  • LLM automatically detected contradictions and supports
  • Generated nuanced resolution instead of binary choice
  • Confidence scores guide propagation strength
  • Context-aware analysis considers incident severity
```

## Step 4: Test the Python REPL

```bash
# Start the Python REPL with the environment
uv run python
```

```python
# Import and use
from baye import Belief, detect_relationship
import asyncio

# Create beliefs
b1 = Belief("APIs are reliable", 0.8, "infra")
b2 = Belief("Stripe returned 500 error", 0.9, "incident")

# Detect relationship (asynchronous)
async def test():
    analysis = await detect_relationship(b1, b2)
    print(f"Relationship: {analysis.relationship}")
    print(f"Confidence: {analysis.confidence}")
    print(f"Explanation: {analysis.explanation}")

asyncio.run(test())
```

**Expected output:**
```
Relationship: contradicts
Confidence: 0.75
Explanation: A 500 error from Stripe contradicts the belief that APIs are reliable...
```

## Step 5: Your Own Script

Create `my_test.py`:

```python
"""My first test with Baye."""
import asyncio
from baye import Belief, detect_relationship, resolve_conflict

async def main():
    # Your beliefs
    b1 = Belief(
        content="Python is the best language for ML",
        confidence=0.9,
        context="programming"
    )

    b2 = Belief(
        content="Julia has better performance for scientific computing",
        confidence=0.85,
        context="programming"
    )

    # Analyze relationship
    print("🔍 Analyzing relationship...\n")
    analysis = await detect_relationship(b1, b2)

    print(f"Relationship: {analysis.relationship}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"Explanation: {analysis.explanation}\n")

    # If there's a conflict, resolve it
    if analysis.relationship == "contradicts":
        print("🤝 Resolving conflict...\n")
        resolution = await resolve_conflict(b1, b2)
        print(f"Resolution: {resolution.resolved_belief}")
        print(f"Confidence: {resolution.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run:**
```bash
export GOOGLE_API_KEY="your-key"
uv run python my_test.py
```

## 🔧 Troubleshooting

### Error: "GOOGLE_API_KEY environment variable not set"

```bash
# Check if it's set
echo $GOOGLE_API_KEY

# If not, export it
export GOOGLE_API_KEY="your-key"
```

### Error: "uv: command not found"

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

### Error: "ModuleNotFoundError: No module named 'baye'"

```bash
# Make sure you're in the correct directory
cd baye

# Reinstall
uv sync

# Always use uv run
uv run python examples/example_llm_integration.py
```

### Warning: "VIRTUAL_ENV does not match"

This warning is normal when you have a workspace venv activated. You can ignore it or:

```bash
# Deactivate workspace venv
deactivate

# Or use baye's local venv
source .venv/bin/activate
python examples/example_llm_integration.py
```

## 📚 Next Steps

1. **Explore examples:**
   ```bash
   ls examples/
   cat examples/example_llm_integration.py
   ```

2. **Read documentation:**
   ```bash
   cat README.md
   cat ARCHITECTURE.md
   ```

3. **Run tests:**
   ```bash
   uv run pytest tests/ -v
   ```

4. **Try the API:**
   - See `README.md` section "API Reference"
   - Create your own beliefs
   - Test relationship detection
   - Resolve conflicts

## 🎯 Use Cases

### 1. Recommendation System

```python
from baye import Belief, JustificationGraph

graph = JustificationGraph()

# User preferences
pref1 = graph.add_belief("User likes spicy food", 0.8, "preferences")
pref2 = graph.add_belief("User is vegetarian", 0.9, "preferences")

# System suggests restaurant
suggestion = graph.add_belief(
    "Recommend Thai vegetarian restaurant",
    confidence=0.85,
    supported_by=[pref1.id, pref2.id]
)
```

### 2. Autonomous Agent Learning

```python
from baye import Belief, detect_relationship, resolve_conflict
import asyncio

async def learn_from_failure(lesson_text):
    # Lesson from error
    lesson = Belief(lesson_text, confidence=0.9, context="incident")

    # Existing beliefs
    existing = [
        Belief("Timeouts should be 30s", 0.7, "config"),
        Belief("APIs are reliable", 0.6, "assumptions")
    ]

    # Detect conflicts
    for belief in existing:
        analysis = await detect_relationship(lesson, belief)
        if analysis.relationship == "contradicts":
            # Resolve automatically
            resolution = await resolve_conflict(lesson, belief)
            print(f"New rule: {resolution.resolved_belief}")

asyncio.run(learn_from_failure("API timeout after 10s caused failure"))
```

### 3. Medical Diagnosis (Educational Example)

```python
symptoms = [
    Belief("Patient has fever", 0.95, "symptoms"),
    Belief("Patient has cough", 0.8, "symptoms"),
    Belief("Patient has fatigue", 0.7, "symptoms")
]

# System infers diagnosis based on beliefs
# (simplified for example)
```

## 🆘 Need Help?

- **GitHub Issues**: https://github.com/franklinbaldo/baye/issues
- **Complete documentation**: `README.md`
- **Architecture**: `ARCHITECTURE.md`
- **Example code**: `examples/`

---

**Tip**: Use `uv run python -i examples/example_llm_integration.py` to run the example and drop into an interactive REPL afterwards, where you can explore the created objects!
