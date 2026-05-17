# Phase 6 Logical Infrastructure Redesign

**Date:** 2026-05-16  
**Status:** proposed operating contract, with adapter groundwork implemented

**Phase 6g implementation update:** Current local odds resolution now uses
The Odds API with BetMGM as the default bookmaker. Multi-book requests are
explicit line-shopping/consensus/audit operations. `market_snapshots` store
provider observations for line-movement audit; `closing_lines` remains the CLV
join table for user-confirmed bets.

## Roles Used

- System Architect
- Data Pipeline Designer
- Prompt/Agent Systems Designer
- QA/Red Team Reviewer
- DevOps/Runtime Agent

## Current Repo Truth

Omega already has the right high-level boundaries:

- `omega/trace/*` owns durable trace, bet-record, closing-line, outcome, and CLV persistence primitives.
- `scripts/ingest_traces.py` owns trace export ingestion from `inbox/traces/*.json`.
- `scripts/ingest_closing_lines.py` owns file-to-DB closing-line ingestion.
- `scripts/resolve_odds.py` owns BetMGM-default pre-decision odds resolution.
- `scripts/fetch_closing_lines.py` owns post-decision current odds capture for pending bets.
- `omega/integrations/odds_api.py` is the only Odds API client.
- `omega/skills/*` contains older logging/QA helpers, but these are not currently wired as the source of truth for trace persistence or benchmark evaluation.
- `prompts/system_prompt.txt` is correct for no-local-access agents, but it must not be treated as the Cowork automation contract.
- `OMEGA_COWORK.md` is the correct place for local automation instructions.

The main drift was conceptual: paid historical Odds API access was added before the repo had a pre-decision market resolver. Phase 6g makes BetMGM via The Odds API the default local odds source while keeping WebFetch as the no-local fallback.

## Design Recommendation

Use a two-ledger model with one deterministic source of truth:

1. **Decision ledger:** every engine run becomes a trace row, optionally with user-confirmed `bet_records`.
2. **Market ledger:** every closing or historical odds snapshot becomes a `closing_lines` row tied to `(trace_id, market, selection_descriptor)`.

The LLM can orchestrate and explain these ledgers, but it never computes the protected fields. The engine computes recommendations; trace/strategy code computes QA, CLV, grading, and calibration metrics.

For odds sourcing:

- **Current decision-time odds:** sourced locally by `omega_resolve_odds` / `scripts/resolve_odds.py`, defaulting to BetMGM.
- **Closing-line CLV:** Cowork should prefer `OMEGA_ODDS_API_KEY` through `omega.integrations.odds_api` over WebFetch.
- **Historical replay/backfill:** paid historical Odds API endpoints should create frozen market snapshots for replay/backtests, not mutate original trace inputs.
- **Manual WebFetch capture:** remains a fallback for no-local-access Project agents and for books/markets the API cannot resolve.

## Files To Create Or Modify

Implemented now:

- `omega/__init__.py` - load `.env` without requiring `python-dotenv`.
- `omega/integrations/odds_api.py` - make Odds API an active current/historical adapter.
- `omega/integrations/__init__.py` - update adapter description.
- `tests/integrations/test_odds_api.py` - parser and endpoint tests.
- `requirements.txt` - compatibility install path for Cowork/CI tools that do not read optional pyproject extras.

Recommended next:

- `scripts/backfill_closing_lines.py` - use historical Odds API snapshots to fill missing `closing_lines` for pending/graded bets.
- `omega/trace/market_snapshot.py` - typed, versioned artifact model if historical snapshots become benchmark inputs.
- `tests/scripts/test_backfill_closing_lines.py` - no-network fixture tests for event and market matching.
- `docs/phase6/HISTORICAL_ODDS_ARTIFACTS.md` - freeze policy for benchmark artifacts.

## Logical Ownership

| Concern | Owner | Contract |
|---|---|---|
| Runtime engine trace | `omega_lite.run` / `omega_lite_standalone.py` | Mints `sandbox-` trace IDs and protected numeric outputs |
| Trace persistence | `omega/trace/store.py` | Idempotent write/read; no direct DB writes from agents |
| Bet tracking | `omega/trace/bet_record.py` + `TraceStore.record_bet()` | User-confirmed wager metadata only |
| Closing lines | `TraceStore.attach_closing_line()` | One row per exact trace/market/selection descriptor |
| CLV math | `omega/trace/clv.py` | Deterministic conversion and line-value math |
| Outcome grading | `scripts/fetch_outcomes_nba.py` today; `omega/strategy/*` long-term | Outcomes attach after trace persistence |
| QA/reporting | `scripts/report_calibration.py`, future QA report helpers | Read-only summaries, never new recommendations |
| Historical odds source | `omega/integrations/odds_api.py` | HTTP + parsing only; no persistence or business logic |
| Prompt/runtime automation | `OMEGA_COWORK.md` | Cowork local automation contract |

## Why This Is Better

- Avoids a parallel CLV pipeline: all close snapshots still enter `closing_lines`.
- Lets paid historical odds repair missed JIT windows without inventing closes.
- Keeps API-key use inside local repo automation, not LLM prompt output.
- Preserves replay reproducibility by freezing historical snapshots as artifacts before they enter quant evaluation.
- Keeps no-local Project agents usable via manual export/WebFetch fallback.

## Failure Modes And Risks

- **Descriptor mismatch:** CLV joins are exact. Bad `selection_descriptor` strings remain the highest-risk failure mode.
- **Historical timestamp ambiguity:** The Odds API returns the closest snapshot at or before the requested timestamp. Scripts must store `timestamp`, `previous_timestamp`, and `next_timestamp` for audit.
- **API coverage gaps:** Player props and some books may require event-specific endpoints or may not exist for a historical timestamp.
- **Budget surprise:** Historical endpoints can cost more than live odds. The local budget file should remain conservative and configurable via `OMEGA_ODDS_API_MONTHLY_BUDGET`.
- **Prompt drift:** `system_prompt.txt` and `OMEGA_COWORK.md` must stay deployment-specific. Do not paste both into one project.

## Verification Plan

- Unit-test parsers with fixture payloads from current and historical Odds API shapes.
- Unit-test endpoint construction without making network calls.
- For the future backfill script, fixture-test exact matching from bet records to historical event/market/outcome rows.
- Run `python -m pytest tests/integrations/test_odds_api.py -q`.
- Before any real paid call, run scripts with `--dry-run` and log request counts without printing API keys.

## Rollback Plan

- Revert `omega/integrations/odds_api.py` to live-only behavior if historical API coverage is unreliable.
- Keep `scripts/ingest_closing_lines.py` as the stable fallback; it can ingest manual WebFetch captures independently of the API client.
- Do not roll back the DB schema unless absolutely necessary; additive trace schema keeps old data readable.

## Ordered Implementation Steps

1. Keep the existing trace/bet/close/outcome schema as the canonical ledger.
2. Promote Odds API to active BetMGM-default pre-decision and post-decision market adapter.
3. Add current event, event-market, event-odds, and historical endpoint parsing/tests.
4. Update Cowork instructions: use BetMGM by default; use multi-book requests only for line shopping/consensus/audit.
5. Add market snapshots for line movement and historical backfill.
6. Add versioned historical market artifacts for benchmark replay.
7. Expand QA reports to include trace coverage, bet-record coverage, close coverage, CLV coverage, outcome coverage, and descriptor mismatch counts.
