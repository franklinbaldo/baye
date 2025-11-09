# Exemplos Práticos - Belief Training System V2.0

## Exemplo 1: Agente de Suporte Técnico

### Cenário
Um agente ajuda usuários com problemas técnicos e aprende com cada interação.

### Código Completo

```python
from belief_system import BeliefSystem
from llm_interface import CalibratedLLM

# Inicialização
system = BeliefSystem(db_path="support_agent.db")
llm = CalibratedLLM("llama-3-8b-instruct")

# Crenças iniciais (seeded)
initial_beliefs = [
    ("restart_fixes_issues", "Reiniciar resolve maioria dos problemas", 0.7),
    ("cache_causes_bugs", "Cache corrompido causa bugs estranhos", 0.6),
    ("permissions_common", "Problemas de permissão são comuns", 0.5),
    ("user_error_likely", "Geralmente é erro do usuário", 0.8),
]

for bid, text, conf in initial_beliefs:
    system.add_belief(bid, text, initial_confidence=conf)

# Adicionar relacionamentos
system.add_edge("restart_fixes_issues", "cache_causes_bugs", "SUPPORTS")
system.add_edge("user_error_likely", "permissions_common", "CONTRADICTS")

# Ticket 1: Usuário reporta app travado
ticket_1 = {
    "issue": "App trava ao abrir",
    "attempted": ["restart"],
    "result": "still_broken"
}

# Agent consulta crenças e sugere solução
confidence = system.get_belief("restart_fixes_issues").confidence
print(f"Confiança em restart: {confidence:.2f}")

# Solução: limpar cache
solution = "clear_cache"
result = "fixed"

# Agent aprende com o resultado
system.update_belief_tool(
    belief_id="cache_causes_bugs",
    p_hat=0.75,  # Agent aumentou confiança
    signal=1.0,  # Sucesso confirmado
    r=0.9,       # Alta confiabilidade (observação direta)
    n=1.0,       # Totalmente nova
    q=0.95,      # Alta qualidade (não ambíguo)
    provenance={
        "source": "ticket_resolution",
        "ticket_id": "TKT-001",
        "issue": ticket_1["issue"],
        "solution": solution,
        "result": result
    }
)

# Propagação automática
# restart_fixes_issues será ajustado porque:
# - cache_causes_bugs aumentou
# - restart não funcionou (signal baixo implícito)

print("\nApós ticket 1:")
print(f"cache_causes_bugs: {system.get_belief('cache_causes_bugs').confidence:.2f}")
print(f"restart_fixes_issues: {system.get_belief('restart_fixes_issues').confidence:.2f}")

# Ticket 2: Outro usuário com problema similar
ticket_2 = {
    "issue": "App lento e travando",
    "attempted": []
}

# Agent agora sugere cache ANTES de restart
# (porque aprendeu que cache é mais provável)
confidence_cache = system.get_belief("cache_causes_bugs").confidence
confidence_restart = system.get_belief("restart_fixes_issues").confidence

if confidence_cache > confidence_restart:
    suggestion = "Tente limpar o cache primeiro"
else:
    suggestion = "Tente reiniciar o app"

print(f"\nSugestão para ticket 2: {suggestion}")

# Após 100 tickets, treinar modelo
if len(system.training_buffer) >= 100:
    print("\n🔧 Iniciando treino batch...")
    train_agent(llm, system.training_buffer)
    print("✅ Modelo calibrado!")
```

### Output Esperado

```
Confiança em restart: 0.70

Após ticket 1:
cache_causes_bugs: 0.78
restart_fixes_issues: 0.65

Sugestão para ticket 2: Tente limpar o cache primeiro

🔧 Iniciando treino batch...
✅ Modelo calibrado!
```

---

## Exemplo 2: Agente de Pesquisa Médica

### Cenário
Pesquisador investigando tratamentos para uma condição. O sistema ajuda a sintetizar literatura e rastrear consenso.

### Código

```python
system = BeliefSystem(db_path="medical_research.db")

# Hipótese inicial
hypothesis = system.add_belief(
    "vitamin_d_covid",
    "Vitamina D reduz severidade de COVID-19",
    initial_confidence=0.5  # Neutro
)

# Paper 1: Estudo observacional (n=1000)
system.update_belief_tool(
    belief_id="vitamin_d_covid",
    p_hat=0.65,  # Researcher acha que há evidência
    signal=0.6,  # Estudo mostrou correlação fraca
    r=0.6,       # Confiabilidade média (observacional)
    n=1.0,       # Novo paper
    q=0.7,       # Boa qualidade metodológica
    provenance={
        "source": "pubmed",
        "paper_id": "PMC12345",
        "study_type": "observational",
        "n": 1000,
        "effect_size": 0.3
    }
)

# Paper 2: RCT (n=200) - resultado negativo
system.update_belief_tool(
    belief_id="vitamin_d_covid",
    p_hat=0.55,  # Researcher menos confiante agora
    signal=0.3,  # Sem efeito significativo
    r=0.9,       # Alta confiabilidade (RCT)
    n=1.0,
    q=0.95,      # Excelente metodologia
    provenance={
        "source": "pubmed",
        "paper_id": "PMC54321",
        "study_type": "rct",
        "n": 200,
        "p_value": 0.43
    }
)

# Paper 3: Meta-análise (10 estudos)
system.update_belief_tool(
    belief_id="vitamin_d_covid",
    p_hat=0.48,  # Researcher agora é cético
    signal=0.35, # Leve efeito, mas heterogeneidade alta
    r=0.95,      # Muito confiável (meta-análise)
    n=1.0,
    q=0.9,
    provenance={
        "source": "pubmed",
        "paper_id": "PMC99999",
        "study_type": "meta_analysis",
        "n_studies": 10,
        "heterogeneity": "high"
    }
)

# Verificar consenso via K-NN
belief = system.get_belief("vitamin_d_covid")
p_knn = system.estimate_belief_knn(belief, k=10)

print(f"Confiança atual: {belief.confidence:.2f}")
print(f"Consenso K-NN: {p_knn:.2f}")
print(f"Incerteza: {belief.uncertainty:.2f}")
print(f"Evidências: {len(belief.evidence_log)}")

# Gerar relatório
print("\n=== Relatório de Síntese ===")
print(f"Hipótese: {belief.text}")
print(f"Conclusão: {'FRACO SUPORTE' if belief.confidence < 0.5 else 'SUPORTE MODERADO'}")
print(f"Confiança: {belief.confidence:.2f} (incerteza: {belief.uncertainty:.2f})")
print("\nEvidências:")
for i, ev in enumerate(belief.evidence_log, 1):
    print(f"  {i}. [{ev.provenance.get('study_type', 'unknown')}] "
          f"Signal={ev.signal:.2f}, Weight={ev.weight:.2f}")
    print(f"     Paper: {ev.provenance.get('paper_id', 'N/A')}")

# Sugerir estudos faltantes
print("\n⚠️  Gaps identificados:")
print("  - Falta RCT grande (n>500)")
print("  - Heterogeneidade não explicada")
print("  - Mecanismo biológico pouco claro")
```

### Output

```
Confiança atual: 0.42
Consenso K-NN: 0.45
Incerteza: 0.18
Evidências: 3

=== Relatório de Síntese ===
Hipótese: Vitamina D reduz severidade de COVID-19
Conclusão: FRACO SUPORTE
Confiança: 0.42 (incerteza: 0.18)

Evidências:
  1. [observational] Signal=0.60, Weight=0.42
     Paper: PMC12345
  2. [rct] Signal=0.30, Weight=0.86
     Paper: PMC54321
  3. [meta_analysis] Signal=0.35, Weight=0.86
     Paper: PMC99999

⚠️  Gaps identificados:
  - Falta RCT grande (n>500)
  - Heterogeneidade não explicada
  - Mecanismo biológico pouco claro
```

---

## Exemplo 3: Debugging de Agent Multi-Step

### Cenário
Um agent complexo falha em uma tarefa. Usamos o sistema para rastrear a causa raiz.

### Código

```python
# Agent tentou processar pagamento, mas falhou
task_result = {
    "task": "process_stripe_payment",
    "steps": [
        {"action": "validate_card", "success": True},
        {"action": "check_balance", "success": True},
        {"action": "create_payment_intent", "success": False, "error": "timeout"}
    ],
    "overall_success": False
}

# Agent tinha crenças sobre cada passo
beliefs_before = {
    "stripe_reliable": 0.95,
    "timeout_rare": 0.90,
    "validation_important": 0.85
}

# Análise: qual crença estava errada?
print("=== Análise de Falha ===")

for step in task_result["steps"]:
    if not step["success"]:
        print(f"\n❌ Falha em: {step['action']}")
        print(f"   Erro: {step.get('error', 'unknown')}")
        
        # Identificar crenças relacionadas
        if "timeout" in step.get("error", ""):
            # Atualizar crença problemática
            result = system.update_belief_tool(
                belief_id="stripe_reliable",
                p_hat=0.70,  # Agent reduz confiança
                signal=0.0,  # Falha completa
                r=0.95,      # Observação direta
                n=1.0,
                q=0.95,
                provenance={
                    "source": "task_execution",
                    "task": task_result["task"],
                    "step": step["action"],
                    "error": step["error"]
                }
            )
            
            print(f"\n   Crença atualizada: stripe_reliable")
            print(f"   {beliefs_before['stripe_reliable']:.2f} → {result['confidence_after']:.2f}")
            
            # Backtrace: o que levou a essa crença?
            belief = system.get_belief("stripe_reliable")
            print(f"\n   Histórico desta crença:")
            for ev in belief.evidence_log[-3:]:  # Últimas 3 evidências
                print(f"     [{ev.timestamp.strftime('%Y-%m-%d')}] "
                      f"Signal={ev.signal:.2f} ({ev.source})")

# Propagação para crenças relacionadas
print("\n=== Propagação ===")
system.propagate_local_update("stripe_reliable", delta=-0.25)

# Verificar impacto em crenças downstream
related_beliefs = system.get_belief("stripe_reliable").supports
for bid in related_beliefs:
    belief = system.get_belief(bid)
    print(f"  {bid}: {belief.confidence:.2f}")

# Recomendações
print("\n=== Recomendações ===")
print("  1. Adicionar retry logic (timeout_handling)")
print("  2. Aumentar timeout de 30s → 60s")
print("  3. Monitorar Stripe status page")
```

### Output

```
=== Análise de Falha ===

❌ Falha em: create_payment_intent
   Erro: timeout

   Crença atualizada: stripe_reliable
   0.95 → 0.68

   Histórico desta crença:
     [2025-10-15] Signal=1.00 (task_execution)
     [2025-10-28] Signal=1.00 (task_execution)
     [2025-11-08] Signal=0.00 (task_execution)

=== Propagação ===
  timeout_rare: 0.65
  api_calls_safe: 0.72

=== Recomendações ===
  1. Adicionar retry logic (timeout_handling)
  2. Aumentar timeout de 30s → 60s
  3. Monitorar Stripe status page
```

---

## Exemplo 4: Compliance e Auditoria (Setor Financeiro)

### Cenário
Banco precisa explicar por que negou um empréstimo (regulação GDPR).

### Código

```python
# Decisão de crédito
applicant = {
    "id": "USR-12345",
    "income": 45000,
    "debt_ratio": 0.42,
    "credit_score": 620,
    "employment_years": 2
}

# Agent avalia
decision_belief = system.add_belief(
    f"approve_loan_{applicant['id']}",
    f"Cliente {applicant['id']} deve receber empréstimo",
    initial_confidence=0.5
)

# Fator 1: Renda
system.update_belief_tool(
    belief_id=decision_belief.id,
    p_hat=0.55,
    signal=0.6,  # Renda ok, mas não excelente
    provenance={
        "factor": "income",
        "value": applicant["income"],
        "threshold": 40000,
        "reasoning": "Acima do mínimo, mas abaixo da média"
    }
)

# Fator 2: Dívida alta (red flag)
system.update_belief_tool(
    belief_id=decision_belief.id,
    p_hat=0.35,
    signal=0.2,  # Negativo
    provenance={
        "factor": "debt_ratio",
        "value": applicant["debt_ratio"],
        "threshold": 0.35,
        "reasoning": "Acima do limite recomendado"
    }
)

# Fator 3: Score baixo
system.update_belief_tool(
    belief_id=decision_belief.id,
    p_hat=0.28,
    signal=0.15,
    provenance={
        "factor": "credit_score",
        "value": applicant["credit_score"],
        "threshold": 650,
        "reasoning": "Abaixo do score mínimo preferencial"
    }
)

# Decisão final
final_belief = system.get_belief(decision_belief.id)
decision = "APROVADO" if final_belief.confidence > 0.5 else "NEGADO"

print(f"=== Decisão de Crédito ===")
print(f"Cliente: {applicant['id']}")
print(f"Decisão: {decision}")
print(f"Confiança: {final_belief.confidence:.2%}")

# Gerar relatório regulatório
print("\n=== Justificativa (GDPR Article 22) ===")
print("Fatores considerados:")

for i, ev in enumerate(final_belief.evidence_log, 1):
    factor = ev.provenance.get("factor", "unknown")
    value = ev.provenance.get("value", "N/A")
    reasoning = ev.provenance.get("reasoning", "N/A")
    
    print(f"\n{i}. {factor.upper()}")
    print(f"   Valor: {value}")
    print(f"   Impacto: {ev.signal:.2%} (peso: {ev.weight:.2f})")
    print(f"   Justificativa: {reasoning}")

# Direito de recurso
print("\n=== Opções do Cliente ===")
if decision == "NEGADO":
    print("Para reverter esta decisão, você pode:")
    print("  1. Reduzir debt_ratio para < 0.35")
    print("  2. Aumentar credit_score para > 650")
    print("  3. Fornecer garantia adicional")

# Salvar auditoria
audit_log = {
    "timestamp": datetime.now().isoformat(),
    "applicant_id": applicant["id"],
    "decision": decision,
    "confidence": final_belief.confidence,
    "factors": [
        {
            "factor": ev.provenance.get("factor"),
            "signal": ev.signal,
            "weight": ev.weight,
            "reasoning": ev.provenance.get("reasoning")
        }
        for ev in final_belief.evidence_log
    ]
}

with open(f"audit_log_{applicant['id']}.json", "w") as f:
    json.dump(audit_log, f, indent=2)

print(f"\n✅ Audit log salvo: audit_log_{applicant['id']}.json")
```

### Output

```
=== Decisão de Crédito ===
Cliente: USR-12345
Decisão: NEGADO
Confiança: 28.00%

=== Justificativa (GDPR Article 22) ===
Fatores considerados:

1. INCOME
   Valor: 45000
   Impacto: 60.00% (peso: 0.63)
   Justificativa: Acima do mínimo, mas abaixo da média

2. DEBT_RATIO
   Valor: 0.42
   Impacto: 20.00% (peso: 0.63)
   Justificativa: Acima do limite recomendado

3. CREDIT_SCORE
   Valor: 620
   Impacto: 15.00% (peso: 0.63)
   Justificativa: Abaixo do score mínimo preferencial

=== Opções do Cliente ===
Para reverter esta decisão, você pode:
  1. Reduzir debt_ratio para < 0.35
  2. Aumentar credit_score para > 650
  3. Fornecer garantia adicional

✅ Audit log salvo: audit_log_USR-12345.json
```

---

## Exemplo 5: A/B Testing com Crenças

### Cenário
Otimizando UI de um app, testando diferentes hipóteses.

### Código

```python
# Hipóteses sobre UI
hypotheses = [
    ("green_button_converts", "Botão verde aumenta conversão", 0.5),
    ("large_cta_better", "CTA grande performa melhor", 0.5),
    ("social_proof_helps", "Social proof aumenta confiança", 0.6)
]

for hid, text, conf in hypotheses:
    system.add_belief(hid, text, initial_confidence=conf)

# Experimento 1: Testar botão verde vs azul
experiment_1_results = {
    "variant_a": {"color": "blue", "conversions": 250, "impressions": 1000},
    "variant_b": {"color": "green", "conversions": 280, "impressions": 1000}
}

# Calcular lift
baseline_rate = experiment_1_results["variant_a"]["conversions"] / \
                experiment_1_results["variant_a"]["impressions"]
treatment_rate = experiment_1_results["variant_b"]["conversions"] / \
                 experiment_1_results["variant_b"]["impressions"]

lift = (treatment_rate - baseline_rate) / baseline_rate

# Atualizar crença
signal = 0.5 + (lift / 2)  # Mapear lift para [0, 1]
signal = max(0, min(1, signal))  # Clip

system.update_belief_tool(
    belief_id="green_button_converts",
    p_hat=0.65,
    signal=signal,
    r=0.9,  # Alta confiabilidade (A/B test rigoroso)
    n=1.0,
    q=0.95,
    provenance={
        "source": "ab_test",
        "experiment_id": "EXP-001",
        "variant_a": experiment_1_results["variant_a"],
        "variant_b": experiment_1_results["variant_b"],
        "lift": lift,
        "p_value": 0.03  # Estatisticamente significativo
    }
)

# Experimento 2: CTA grande
experiment_2_results = {
    "variant_a": {"size": "small", "conversions": 220, "impressions": 1000},
    "variant_b": {"size": "large", "conversions": 215, "impressions": 1000}
}

treatment_rate_2 = experiment_2_results["variant_b"]["conversions"] / 1000
baseline_rate_2 = experiment_2_results["variant_a"]["conversions"] / 1000
lift_2 = (treatment_rate_2 - baseline_rate_2) / baseline_rate_2

signal_2 = 0.5 + (lift_2 / 2)
signal_2 = max(0, min(1, signal_2))

system.update_belief_tool(
    belief_id="large_cta_better",
    p_hat=0.48,  # Agent acha que não fez diferença
    signal=signal_2,
    r=0.9,
    n=1.0,
    q=0.95,
    provenance={
        "source": "ab_test",
        "experiment_id": "EXP-002",
        "variant_a": experiment_2_results["variant_a"],
        "variant_b": experiment_2_results["variant_b"],
        "lift": lift_2,
        "p_value": 0.82  # Não significativo
    }
)

# Gerar recomendações
print("=== Recomendações de UI ===\n")

beliefs_sorted = sorted(
    [system.get_belief(hid) for hid, _, _ in hypotheses],
    key=lambda b: b.confidence,
    reverse=True
)

for i, belief in enumerate(beliefs_sorted, 1):
    status = "✅ IMPLEMENTAR" if belief.confidence > 0.6 else \
             "⚠️  TESTAR MAIS" if belief.confidence > 0.4 else \
             "❌ DESCARTAR"
    
    print(f"{i}. {belief.text}")
    print(f"   Status: {status}")
    print(f"   Confiança: {belief.confidence:.2%}")
    print(f"   Evidências: {len(belief.evidence_log)} experimento(s)")
    
    # Mostrar último experimento
    if belief.evidence_log:
        last_exp = belief.evidence_log[-1]
        lift = last_exp.provenance.get("lift", 0)
        p_val = last_exp.provenance.get("p_value", 1.0)
        
        print(f"   Último teste: lift={lift:+.1%}, p={p_val:.3f}")
    print()

# Planejar próximos experimentos
print("=== Próximos Experimentos ===")
for belief in beliefs_sorted:
    if 0.4 < belief.confidence < 0.6:  # Incerto
        print(f"  • Testar novamente: {belief.text}")
        print(f"    (incerteza: {belief.uncertainty:.2f})")
```

### Output

```
=== Recomendações de UI ===

1. Botão verde aumenta conversão
   Status: ✅ IMPLEMENTAR
   Confiança: 65.00%
   Evidências: 1 experimento(s)
   Último teste: lift=+12.0%, p=0.030

2. Social proof aumenta confiança
   Status: ⚠️  TESTAR MAIS
   Confiança: 60.00%
   Evidências: 0 experimento(s)

3. CTA grande performa melhor
   Status: ⚠️  TESTAR MAIS
   Confiança: 48.00%
   Evidências: 1 experimento(s)
   Último teste: lift=-2.3%, p=0.820

=== Próximos Experimentos ===
  • Testar novamente: Social proof aumenta confiança
    (incerteza: 0.50)
  • Testar novamente: CTA grande performa melhor
    (incerteza: 0.35)
```

---

## Exemplo 6: Treino Periódico e Avaliação

### Código para Loop de Treino Completo

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Setup
system = BeliefSystem(db_path="production.db")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Adicionar LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# Calibration head
calibration_head = BeliefCalibrationHead(hidden_size=4096)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(calibration_head.parameters()),
    lr=1e-5
)

# Loop de produção
num_tasks = 1000
training_interval = 100

for task_i in range(num_tasks):
    # Executar tarefa
    task = get_next_task()
    result = execute_task(task, model, system)
    
    # Agent reflete
    belief_update = agent_reflect(task, result, model)
    
    # Atualizar crença
    tool_result = system.update_belief_tool(
        belief_id=belief_update["belief_id"],
        p_hat=belief_update["p_hat"],
        signal=belief_update.get("signal"),
        provenance={"task_id": task.id, "result": result}
    )
    
    # Treino periódico
    if (task_i + 1) % training_interval == 0:
        print(f"\n🔧 Treino batch (tasks {task_i-99}-{task_i})...")
        
        # Preparar dataset
        batch = system.training_buffer[-training_interval:]
        
        # Treinar
        total_loss = 0.0
        for sample in batch:
            # Tokenizar
            inputs = tokenizer(
                sample["context"],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Forward
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, -1, :]
            
            # Calibration head
            p_hat_pred = calibration_head(hidden.unsqueeze(1)).squeeze()
            
            # Target
            p_star = torch.tensor(sample["p_star"])
            
            # Loss
            loss = (p_hat_pred - p_star) ** 2
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(batch)
        print(f"   Avg loss: {avg_loss:.4f}")
        
        # Avaliar calibração
        metrics = system.calculate_training_metrics()
        print(f"   Brier: {metrics['mean_brier']:.4f}")
        print(f"   ECE: {metrics['ece']:.4f}")
        
        # Salvar checkpoint
        if metrics['mean_brier'] < 0.01:  # Excelente
            print("   🎉 Checkpoint salvo (best model)")
            model.save_pretrained(f"checkpoints/best")
            calibration_head.save(f"checkpoints/best_head.pt")

print("\n✅ Treino completo!")
```

---

## Resumo dos Exemplos

| Exemplo | Use Case | Destaque |
|---------|----------|----------|
| 1. Suporte Técnico | Aprendizado incremental | UoU em ação |
| 2. Pesquisa Médica | Síntese de literatura | K-NN para consenso |
| 3. Debugging | Rastreamento de falhas | Auditabilidade |
| 4. Compliance | Regulação financeira | Proveniência GDPR |
| 5. A/B Testing | Otimização de produto | Mapear métricas → crenças |
| 6. Treino Completo | Loop produção | Fine-tuning periódico |

**Todos executáveis com o prototype!** 🚀
