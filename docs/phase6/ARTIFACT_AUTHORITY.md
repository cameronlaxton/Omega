# Artifact Authority (Phase 6)

The single source-of-truth map for Omega's tracking artifacts. When two
artifacts disagree, this table decides which one wins. For background see
[`LOGGING_TRACING_QA_BET_TRACKING_REDESIGN.md`](LOGGING_TRACING_QA_BET_TRACKING_REDESIGN.md);
export-shape detection/validation lives in `omega/trace/export_validator.py` and
DB-path policy in `omega/trace/store.py` (`_resolve_db_path`, `db_status`).

## The one rule

> **TraceStore (`var/omega_traces.db`) is the only source of truth for numbers.**
> Sidecars are the source of truth for session narrative/process. Everything in
> `var/reports/` is derived and reproducible. Trace export files are import-only.
> Per-session JSONL is a recovery mirror.

## Authority table

| Artifact | Role | Canonical? | Append-only? | Regenerable? | If missing/malformed |
|---|---|---|---|---|---|
| `var/omega_traces.db` (TraceStore) | model predictions, outcomes, grading, calibration state, optional wager/CLV | **canonical (numeric)** | no (idempotent writes) | no | guard refuses corrupt/ambiguous-empty DB; `db_status.py` diagnoses |
| `var/inbox/traces/*.json` (trace export) | transfer analyze() output → ledger | no (import-only) | n/a (consumed) | re-export from analyze() output | bad shape → `failed/` + `.error.txt`; **re-wrap, never re-run analyze()** |
| `var/inbox/sessions/<sid>.json` (sidecar) | session narrative: `exec_stats`, `agent_notes`, `audit_events` | source of truth for **narrative only** | events appended atomically | from JSONL mirror (events) | `load_sidecar_safe`→None; opt-in quarantine; numeric fields never live here |
| `var/inbox/sessions/<sid>.events.jsonl` | recovery mirror of audit events | no (derived mirror) | yes | derived from sidecar writes | best-effort; write failure is warn-only; **not promoted to canonical** |
| `var/reports/latest.md` | calibration health report | **derived** | no | `report_calibration.py` | regenerate; never edit by hand |
| `var/reports/run_audits/<sid>.audit.md` | human session audit | **derived** (numbers from DB, prose from sidecar) | no | `render_session_audits.py` | regenerate; degraded mode if sidecar absent |
| `var/inbox/action_plans/*.json` | prescriptive maintenance directive | no (ephemeral) | n/a | re-author | strict allowlist; unknown action → exit 2 |
| `bet_ledger` / `closing_lines` tables | optional wager / CLV metadata (`bet_ledger` absorbed the retired `bet_records` table at schema V14) | optional | no | no | **absence is normal — never gates grading/calibration/output mode** |

## What to trust after a run

- **Automated scripts** trust the ledger (`var/omega_traces.db`).
- **A human after a failed run** reads `var/reports/run_audits/<sid>.audit.md`, then
  the `*.events.jsonl` mirror if the sidecar was quarantined.

## Decoupling invariants (do not regress)

- Calibration eligibility depends on model predictions + outcome + trace_quality
  flags — **never** on `bet_record`, `bet_taken`, `closing_line`, or CLV.
- Output mode (`RESEARCH_CANDIDATE` vs actionable) is a model-evaluation decision
  (fitted profile + eligible coverage + valid sidecar) — **never** on wager logs.
- CLV / `closing_line` are optional review/market-quality metadata, not grading
  requirements.

## Generated-file labeling

Every derived markdown report carries a front-matter header
(`omega/trace/report_header.py`) with `canonical: false`, `generated_at`,
`source_db_path`, `db_path_source`, `trace_count_at_generation`, and
`source_artifacts`, so a derived file can never be mistaken for source state.

## Player props are league-scoped

Props are **not** a separate top-level domain. A league owns context, schedule,
identity, outcome source, and calibration surface. Conceptually:

- `league` — NBA, MLB, WNBA, NFL, …
- `market_family` — `game` | `team` | `player_prop` (today carried as `trace.kind`)
- `stat_key` — points, rebounds, strikeouts, pass_yds, … (`MarketQuote.stat_key`)
- `entity_type` — `team` | `player` (`Entity.entity_type`)

The separate prop scripts/tables (`fetch_outcomes_props.py`, `analyze_player_prop`,
`prop_outcomes`, the prop calibration plane, `omega_analyze_prop`) are **accepted
league-scoped implementation adapters** — the grading shape genuinely differs from
games. They are adapters, not a parallel domain. Unifying storage/calibration
under an explicit `market_family` field is deferred until a second league's prop
+ outcome flow proves the abstraction is needed.

Current DB truth remains split by grading shape:

- `outcomes` is the active game-outcome table.
- `prop_outcomes` is the active player-stat outcome table.

Do not merge these tables, rename the prop calibration plane, or replace
`trace.kind = "prop"` in this hardening phase. Treat those names as current
adapter vocabulary until a future migration explicitly introduces
`market_family`.
