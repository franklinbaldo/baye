# LLM ↔ Belief Tool Contract

Este documento descreve como o chat CLI deve operar para que a LLM interaja **apenas** através da tool `update_belief_tool`, garantindo que toda saída ao usuário e todo ajuste de crenças passem pelo mesmo canal.

## Fluxo Geral
1. **Entrada do usuário** chega ao chat.
2. A LLM recebe o histórico e o estado das crenças, mas só pode responder chamando `update_belief_tool`.
3. O CLI renderiza para o usuário **exatamente** o texto que a LLM escreveu no input da tool, junto com os dados de crença.

## Campos obrigatórios do tool input
Cada chamada deve enviar um payload com três campos:

| Campo | Obrigatório? | Descrição |
| --- | --- | --- |
| `texto` | Sim | Mensagem completa que será mostrada ao usuário. Não é permitido emitir prose fora deste campo. |
| `belief_value_guessed` | Sim | Valor (0–1) que a LLM acredita ser a confiança atual da crença que está manipulando. Serve como chute explícito antes de qualquer ajuste. |
| `delta` | Sim (default 0) | Ajuste solicitado para a crença. Zero significa “não quero alterar o valor interno”; qualquer outro valor representa o deslocamento desejado. |

### Por que cada campo existe?
- **texto** garante que a LLM “fale” apenas através do tool input.
- **belief_value_guessed** força a LLM a expor seu entendimento atual da crença, permitindo comparar com o valor real e medir precisão.
- **delta** permite que a própria LLM proponha updates quando necessário, mantendo o canal único da tool (sem APIs adicionais).

## Margem de confiança e retries
- O backend mantém uma margem admissível entre `belief_value_guessed` e o valor real da crença.
- Se o chute estiver fora da margem, a tool deve responder exigindo que a LLM refaça a chamada **com o mesmo texto** e um `delta` apropriado para aproximar o valor real do alvo pretendido.
- Enquanto a margem não for respeitada, nenhuma outra tool pode ser chamada.

## Regras para o System Prompt
O prompt do agente precisa deixar claro que:
1. Só existe `update_belief_tool`; qualquer outra saída é proibida.
2. Toda resposta deve preencher `texto`, `belief_value_guessed` e `delta`.
3. `delta` fica em `0` até que o sistema peça um ajuste explícito.
4. Todo raciocínio, explicação ou “fala” deve ser escrito dentro de `texto`.
5. Mesmo sem evidências suficientes, a LLM deve chamar a tool, sinalizando incerteza via `belief_value_guessed` e usando `delta=0`.

## Renderização no CLI
- O CLI mostra `texto` diretamente ao usuário.
- Pode opcionalmente exibir o par `belief_value_guessed` / `delta` para transparência (ex.: “Palpite atual: 0.72 · Delta aplicado: +0.05”).
- Como não há prose fora da tool, basta ler o payload da chamada mais recente para montar a UI.

## Benefícios
- **Traço auditável**: todo conteúdo e ajuste passa pelo mesmo registro.
- **Feedback contínuo**: comparar `belief_value_guessed` com o valor real ajuda a calibrar a LLM.
- **Simplicidade operacional**: um único mecanismo atende fala, tracking e updates.

Com este contrato documentado, os próximos passos são atualizar o system prompt e o wrapper do agent para aplicarem exatamente essas regras.
