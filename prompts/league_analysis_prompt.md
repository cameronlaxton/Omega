# League Daily Analysis Prompt

**Status:** Reference template for manual / ad-hoc local runtime sessions. Not wired to any scheduler — scheduling is deferred to a later phase.

**Substitute:** `{league}` (e.g. `NBA`, `MLB`), `{slate_date}` (ISO `YYYY-MM-DD`; default = today in the workspace's local date).

---

Execute from the active Omega workspace directory. Follow [`OMEGA_RUNTIME.md`](../OMEGA_RUNTIME.md) and the runtime contract in [`prompts/system_prompt.txt`](system_prompt.txt).

Run the Omega {league} daily analysis session for {slate_date}.

## Boundaries (non-negotiable)

- Typed execution only. MCP tools first, then `omega.core.contracts.service.analyze`. Never invent probabilities, edge, EV, Kelly, units, confidence tiers, fair/no-vig prices, or `trace_id`s in prose.
- Sidecar audit state goes to `var/inbox/sessions/<session_id>.json` via `omega.trace.session_sidecar.append_audit_events` (atomic). Do **not** hand-edit the sidecar JSON.
- Do **not** create `RUN_AUDIT.md` or `RUN_TRACE.jsonl`. Both are retired.

## Pipeline

1. **Preflight.** Mint a new `session_id` of the form `sess-YYYYMMDD-{league_lower}1`. Bootstrap the sidecar via `append_audit_events(..., bootstrap=bootstrap_payload(...))` and append the first event:
   - `event_type=preflight`, `step=cowork_preflight`, `status=ok|warn|fail`, `notes` summarizing repo / engine readiness.

2. **Resolve odds.** Use BetMGM via Omega's typed odds path (`omega_resolve_odds` or `omega-resolve-odds`). Append `data_provenance` events naming each source.

3. **Gather evidence.** Per the league:
   - **NBA-flavored vocab:** `pace_up`, `pace_down`, `rest_advantage`, `b2b_fatigue`, `def_matchup_weak`, `def_matchup_strong`, `usage_role_change`, `blowout_risk`.
   - **MLB-flavored vocab:** probable pitcher, park factor, weather (wind, temp), lineup/injury status, bullpen rest.
   - Express material evidence as typed `EvidenceSignal` objects on the analyze request — never as free text inside protected fields.
   - Cite each source. Record missing inputs as `assumptions`/`bugs` on the relevant audit event.

4. **Run engine.** Call analyze for each eligible game and selected prop candidate. Ensure that both `home_context` and `away_context` are fully populated with their required team keys (e.g. `off_rating`, `def_rating`, `pace` for basketball/NBA/WNBA; `off_rating`, `def_rating` for MLB) to guarantee `context_source="provided"` and satisfy the trace calibration eligibility gate. Reuse the canonical analyze output verbatim — no rewriting of numeric fields.

5. **Export traces.** Write `{"trace": <full analyze return>, "bet_record": null}` to `var/inbox/traces/<trace_id>.json`. Nest `reasoning_inputs`, `reasoning_narrative`, `reasoning_downgrade_rationale`, and `trace_quality` **inside** the inner `trace` block. (Top-level siblings still work via ingest's compatibility merge but are deprecated.)

6. **Append audit events.** For each major step, append a structured `AuditEvent`:
   - `engine_run` per analyze call, with `trace_ids=[<id>]` and notes summarizing what was provided/missing.
   - `candidate_rejected` for any candidate dropped before recommendation, with reasoning.
   - `downgrade` whenever the engine or LLM lowered confidence; include the `rationale`.
   - `rationale` for final slate-level decisions.
   - `bug` for any anomaly observed (be specific; reference file paths and trace_ids).
   - Never put protected quant values into event `inputs`/`outputs`/`notes`. The writer will raise `ProtectedValueError`.

7. **Bet confirmation (optional).** If the user confirms a bet, re-export the same trace with the populated `bet_record` and `selection_descriptor`. Single-trace policy.

8. **Close the session.** Set `closed_at`, finalize `agent_notes` (freeform summary), and stop. Do not run `ingest_traces` or `render_audit` from inside this session — those are deterministic follow-up steps.

## Post-session (deterministic, run separately)

After the session is closed, the operator (or scheduler) runs:

```
omega-run-action-plan fixtures/action_plans/daily_trace_intake.json
omega-run-action-plan fixtures/action_plans/render_session_audits.json
```

The rendered audit markdown lands at `var/reports/run_audits/<session_id>.audit.md`.
