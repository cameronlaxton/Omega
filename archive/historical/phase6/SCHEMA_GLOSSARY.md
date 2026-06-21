> [!NOTE]
> This document is from a legacy phase that has been implemented and merged to `main`. It is retained here for historical reference.

# Phase 6 Schema Glossary

This glossary names the active concepts without changing runtime schema. It is
documentation for agents and operators, not a migration plan.

| concept | source of truth | persisted where | derived where | action |
|---|---|---|---|---|
| trace | `PersistableTrace` and `TraceStore` | `traces.full_trace` plus query columns | reports and audits | leave unchanged |
| candidate | model-issued trace with predictions | `traces.predictions`, trace JSON | calibration/report queries | document only |
| prediction | deterministic engine output (`GamePredictions` model) | trace JSON and `traces.predictions` | calibration pairs | retyped to `GamePredictions` Pydantic model |
| recommendations | list/dict of `Recommendation` models | `traces.recommendations`, trace JSON | calibration/report queries | retyped to `list[Recommendation]` (game) or singular prop `Recommendation` dict (prop) |
| market | request/odds/bet contract | `MarketQuote`, `bet_records.market`, trace JSON | CLV/reporting | document field ownership |
| market_family | current proxy is `trace.kind` | trace JSON | prompt/report vocabulary | defer explicit field |
| league | request schema and trace metadata | `traces.league`, trace JSON | var/reports/outcome adapters | leave unchanged |
| entity_type | `Entity.entity_type` | request/context JSON | prompt routing | document only |
| stat_key | `MarketQuote.stat_key` / prop `prop_type` | trace JSON, evidence rows, prop outcomes | prop grading/reporting | document accepted aliases |
| outcome | game score or player stat result | `outcomes`, `prop_outcomes` | grading and calibration | keep both active |
| grade | result derived from outcome vs prediction/line | outcome tables plus report code | var/reports/calibration | do not store in sidecars |
| calibration_eligible | `trace_quality.calibration_eligible` | trace JSON | calibration SQL/reporting | keep JSON-based |
| bet_record | confirmed wager metadata | `bet_records` | CLV/session reports | optional only |
| closing_line / CLV | market-close metadata | `closing_lines` | CLV report | optional only |
| audit_event | session process event | sidecar JSON, JSONL mirror | audit markdown | current session event shape |
| quality_gate / null_data_audit | trace quality and sidecar events | trace JSON, sidecar JSON | ingest/report gates | standardize vocabulary |
| evidence_signal | structured reasoning signal | trace JSON, `evidence_signals` | signal performance | query aid, not canonical over trace |
| provenance/source | provider/source fields | trace JSON, evidence tables, market tables | reports | keep source labels explicit |
| report metadata | derived frontmatter | generated markdown | operator views | every generated report marks `canonical: false` |

Player props are league-scoped player-stat markets. The separate
`prop_outcomes` table, prop outcome fetcher, prop analyzer, and prop calibration
plane remain active adapters because their grading shape differs from game
outcomes. Do not use this glossary as justification for a table unification
refactor.

