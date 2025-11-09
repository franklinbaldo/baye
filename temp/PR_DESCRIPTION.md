# PR: Enforce Tool-Only Cogito Responses in Baye Chat

## Summary
- replace the legacy single-agent flow with a dual-agent pipeline where belief extraction and response generation are decoupled but share the same tracker state
- introduce the `update_belief_tool` contract (texto, belief_value_guessed, delta) plus enforcement logic, margin checks, and delta retries inside `ChatSession`
- render assistant output directly from the tool input and expose the guessed belief value + adjustment telemetry in the CLI, while adding documentation of the protocol in `LLM_TOOL_CONTRACT.md`
- expand the tracker with deterministic delta application and safer propagation accounting so manual adjustments preserve pseudo-count semantics

## Scientific & Philosophical Rationale
- **Bayesian cognition**: forcing the LLM to state `belief_value_guessed` each turn mirrors Bayesian agents that must publish their posterior before observing feedback; this deters hindsight bias and keeps Update-on-Use faithful to Beta-Bernoulli conjugacy.
- **Epistemic humility**: the mandatory margin-check/delta loop operationalizes virtue epistemology principles—agents must confess uncertainty (texto + guess) and only adjust beliefs when evidence pushes them beyond the tolerated credence interval.
- **Single-channel signaling**: by collapsing all speech + updates into one tool, we respect information-theoretic cleanliness (no side channels) and keep the justification graph auditable, echoing cogito’s focus on explainable belief revision.
- **Human factors**: surfacing the guess/delta telemetry in the CLI gives users phenomenological insight into the agent’s “belief stream,” aligning with reflective practice traditions (e.g., Schön) where practitioners externalize intermediate confidence to stay calibratable.

## Implementation Notes
- `ChatSession.process_message` now returns an `AssistantReply` dataclass consumed by the CLI; we seed at least one belief so the tool always has a target.
- `_tool_update_belief_response` gates every response, enforces margin errors, and calls the new `BeliefTracker.apply_manual_delta` when the LLM decides a delta is needed.
- `LLM_TOOL_CONTRACT.md` documents the operational rules so future agents and reviewers understand the invariant.

## Validation
- `uv run python -m py_compile src/baye/chat_session.py src/baye/cli.py src/baye/belief_tracker.py`

