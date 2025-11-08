# 🚀 Quickstart: Rodando Baye na sua máquina

## Pré-requisitos

- Python 3.10+ instalado
- `uv` instalado ([instruções](https://docs.astral.sh/uv/getting-started/installation/))
- API key do Google Gemini ([obtenha aqui](https://aistudio.google.com/app/apikey))

## Passo 1: Clone e Instale

```bash
# Clone o repositório
cd /home/frank/workspace
git clone https://github.com/franklinbaldo/baye.git
cd baye

# Instale dependências com uv
uv sync
```

**Saída esperada:**
```
Using CPython 3.13.5
Creating virtual environment at: .venv
Resolved 144 packages in 2ms
Installed 132 packages in 1.33s
✓ baye==1.5.0
```

## Passo 2: Configure API Key

Você já tem a chave em `/home/frank/workspace/.envrc`:

```bash
# Opção 1: Export direto
export GOOGLE_API_KEY="AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

# Opção 2: Usar direnv (se já configurado)
source /home/frank/workspace/.envrc

# Opção 3: Criar .env no projeto
echo 'GOOGLE_API_KEY="AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"' > .env
```

**Verificar:**
```bash
echo $GOOGLE_API_KEY
# Deve mostrar: AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic
```

## Passo 3: Rode o Exemplo

```bash
# Execute o exemplo completo com LLM
export GOOGLE_API_KEY="AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"
uv run python examples/example_llm_integration.py
```

**O que você vai ver:**

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

## Passo 4: Teste o Python REPL

```bash
# Inicie o Python REPL com o ambiente
uv run python
```

```python
# Importe e use
from baye import Belief, detect_relationship
import asyncio

# Crie beliefs
b1 = Belief("APIs são confiáveis", 0.8, "infra")
b2 = Belief("Stripe retornou erro 500", 0.9, "incident")

# Detecte relacionamento (assíncrono)
async def test():
    analysis = await detect_relationship(b1, b2)
    print(f"Relação: {analysis.relationship}")
    print(f"Confiança: {analysis.confidence}")
    print(f"Explicação: {analysis.explanation}")

asyncio.run(test())
```

**Saída esperada:**
```
Relação: contradicts
Confiança: 0.75
Explicação: Um erro 500 do Stripe contradiz a crença de que APIs são confiáveis...
```

## Passo 5: Seu Próprio Script

Crie `meu_teste.py`:

```python
"""Meu primeiro teste com Baye."""
import asyncio
from baye import Belief, detect_relationship, resolve_conflict

async def main():
    # Suas beliefs
    b1 = Belief(
        content="Python é a melhor linguagem para ML",
        confidence=0.9,
        context="programming"
    )

    b2 = Belief(
        content="Julia tem melhor performance para computação científica",
        confidence=0.85,
        context="programming"
    )

    # Analise relacionamento
    print("🔍 Analisando relacionamento...\n")
    analysis = await detect_relationship(b1, b2)

    print(f"Relação: {analysis.relationship}")
    print(f"Confiança: {analysis.confidence:.2f}")
    print(f"Explicação: {analysis.explanation}\n")

    # Se houver conflito, resolva
    if analysis.relationship == "contradicts":
        print("🤝 Resolvendo conflito...\n")
        resolution = await resolve_conflict(b1, b2)
        print(f"Resolução: {resolution.resolved_belief}")
        print(f"Confiança: {resolution.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Execute:**
```bash
export GOOGLE_API_KEY="AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"
uv run python meu_teste.py
```

## 🔧 Troubleshooting

### Erro: "GOOGLE_API_KEY environment variable not set"

```bash
# Verifique se está setada
echo $GOOGLE_API_KEY

# Se não estiver, exporte
export GOOGLE_API_KEY="AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"
```

### Erro: "uv: command not found"

```bash
# Instale uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ou use pip
pip install uv
```

### Erro: "ModuleNotFoundError: No module named 'baye'"

```bash
# Certifique-se de estar no diretório correto
cd /home/frank/workspace/baye

# Reinstale
uv sync

# Use sempre uv run
uv run python examples/example_llm_integration.py
```

### Warning: "VIRTUAL_ENV does not match"

Esse warning é normal quando você tem um venv do workspace ativado. Pode ignorar ou:

```bash
# Desative o venv do workspace
deactivate

# Ou use o venv local do baye
source .venv/bin/activate
python examples/example_llm_integration.py
```

## 📚 Próximos Passos

1. **Explore os exemplos:**
   ```bash
   ls examples/
   cat examples/example_llm_integration.py
   ```

2. **Leia a documentação:**
   ```bash
   cat README.md
   cat ARCHITECTURE.md
   ```

3. **Rode os testes:**
   ```bash
   uv run pytest tests/ -v
   ```

4. **Experimente a API:**
   - Veja `README.md` seção "API Reference"
   - Crie suas próprias beliefs
   - Teste detecção de relacionamentos
   - Resolva conflitos

## 🎯 Casos de Uso

### 1. Sistema de Recomendação

```python
from baye import Belief, JustificationGraph

graph = JustificationGraph()

# Preferências do usuário
pref1 = graph.add_belief("User likes spicy food", 0.8, "preferences")
pref2 = graph.add_belief("User is vegetarian", 0.9, "preferences")

# Sistema sugere restaurante
suggestion = graph.add_belief(
    "Recommend Thai vegetarian restaurant",
    confidence=0.85,
    supported_by=[pref1.id, pref2.id]
)
```

### 2. Agente Autônomo Aprendendo

```python
from baye import Belief, detect_relationship, resolve_conflict
import asyncio

async def learn_from_failure(lesson_text):
    # Lição do erro
    lesson = Belief(lesson_text, confidence=0.9, context="incident")

    # Beliefs existentes
    existing = [
        Belief("Timeouts should be 30s", 0.7, "config"),
        Belief("APIs are reliable", 0.6, "assumptions")
    ]

    # Detecta conflitos
    for belief in existing:
        analysis = await detect_relationship(lesson, belief)
        if analysis.relationship == "contradicts":
            # Resolve automaticamente
            resolution = await resolve_conflict(lesson, belief)
            print(f"Nova regra: {resolution.resolved_belief}")

asyncio.run(learn_from_failure("API timeout after 10s caused failure"))
```

### 3. Diagnóstico Médico (Exemplo Educacional)

```python
symptoms = [
    Belief("Patient has fever", 0.95, "symptoms"),
    Belief("Patient has cough", 0.8, "symptoms"),
    Belief("Patient has fatigue", 0.7, "symptoms")
]

# Sistema infere diagnóstico baseado em beliefs
# (simplificado para exemplo)
```

## 🆘 Precisa de Ajuda?

- **Issues GitHub**: https://github.com/franklinbaldo/baye/issues
- **Documentação completa**: `README.md`
- **Arquitetura**: `ARCHITECTURE.md`
- **Código dos exemplos**: `examples/`

---

**Dica**: Use `uv run python -i examples/example_llm_integration.py` para rodar o exemplo e cair no REPL interativo depois, onde você pode explorar os objetos criados!
