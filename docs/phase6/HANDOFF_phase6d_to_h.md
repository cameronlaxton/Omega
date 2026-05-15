# Phase 6 Handoff — Steps 1–3 Done, Steps 4 + 5 Next

**Branch:** `main` (no PR opened yet — commit when ready)
**Date:** 2026-05-14
**Owner:** Cam
**Pickup model:** any Omega-aware session (Claude Code or Claude.ai)

This doc captures *exactly* what landed in this session and what the next session
needs to do. Read this first; the original plan lives at
`~/.claude/plans/the-engine-seems-to-gleaming-puzzle.md` for full context.

---

## What's done

### Step 1 — Sandbox→Local trace bridge ✅

- `omega/trace/schema.py` — `CURRENT_VERSION=3`, additive `bet_records` (v2) and
  `closing_lines` (v3) tables.
- `omega/trace/bet_record.py` — `BetRecord` Pydantic model + `from_export_block()`.
- `omega/trace/store.py` — added `record_bet()`, `get_bet_records()`,
  `update_bet_status()`, `attach_closing_line()`, `get_closing_lines()`.
  Schema-migration is forward-additive via `_record_version()`.
- `scripts/ingest_traces.py` — scans `inbox/traces/*.json`, adapts engine
  `analyze()` output, persists trace + optional bet, moves to processed/failed.
  Accepts both export-block shape and raw `analyze()` output. `--dry-run` flag.
- `inbox/traces/` + `processed/` + `failed/` scaffold with `.gitkeep` and README.

### Step 2 — System prompt update ✅

- `prompts/system_prompt.txt` — added §10 Trace Export Protocol (LLM emits fenced
  JSON block per turn) and §11 CLV Capture Discipline (book/line/odds/timestamp
  required on every Bet Card).
- `OMEGA_HANDBOOK.md` — stale "estimated lean" reference deleted on line 90.

### Step 3 — Outcome + CLV resolution ✅

- `omega/integrations/__init__.py` + `espn_nba.py` — ESPN public scoreboard
  client, 30-team NBA alias resolver, `parse_scoreboard()` for unit tests.
  `canonical_team()` uses exact match first, then substring fallback gated to
  aliases ≥4 chars to prevent false matches.
- `omega/integrations/odds_api.py` — the-odds-api client with monthly-budget
  bookkeeping (`omega_odds_api_budget.json` at cwd, default cap 450/500).
  Reads `OMEGA_ODDS_API_KEY` env var. Raises `OddsApiKeyMissing` /
  `OddsApiBudgetExceeded`.
- `omega/trace/clv.py` — `compute_clv()` returning `CLVResult` dataclass.
  American↔decimal↔implied-prob helpers. Line-value scoring for over/under and
  home/away spread.
- `scripts/fetch_outcomes_nba.py` — ESPN scoreboard → `TraceStore.attach_outcome()`.
  Accepts `--since today|yesterday|YYYY-MM-DD --until ...`. Searches game-date
  AND prior-day windows for decisions made the night before. Logs unmatched
  traces so the alias table can be extended.
- `scripts/fetch_closing_lines.py` — sport-agnostic; joins `bet_records` × `traces`
  × `closing_lines` to find pending bets with no close. Groups pending bets by
  `trace.league` and issues one the-odds-api call per league via
  `OddsApiClient.fetch_event_odds(league)` (which resolves through
  `SPORT_KEY_MAP` in `omega/integrations/odds_api.py`). Maps `bet.market`
  (`moneyline|spread|total`) → the-odds-api keys (`h2h|spreads|totals`). Skips
  `player_prop:*` markets (not in free-tier base). Accepts `--league` to restrict
  to one league; default iterates every league present in pending bets.
  `scripts/fetch_closing_lines_nba.py` remains as a deprecation shim that
  forwards to the generalized script with `--league NBA`.

### Tests ✅

**268 tests passing.** New this session:
- `tests/scripts/test_ingest_traces.py` — 7 tests (shape A + shape B + error
  path + idempotency + dry-run).
- `tests/integrations/test_espn_alias_resolver.py` — alias resolution + ESPN
  fixture parsing.
- `tests/trace/test_clv.py` — odds conversions + CLV computation + line value.
- `tests/trace/test_closing_lines.py` — `attach_closing_line` round-trip +
  unique-key idempotency + multi-market.

### What got smoke-tested manually

- `python scripts/ingest_traces.py --dry-run` and full run against a synthetic
  `sandbox-smoke001.json` — file moved to `processed/`, row inserted with
  `schema_version=2` (and after step 3, `=3`).
- Both new scripts respond to `--help` with their expected arg surface.

### What did NOT get tested

- **Real network call to ESPN.** Fixture parsing is tested; the live HTTP path
  has not been exercised. First real run should be against a known game date so
  you can verify scores match.
- **Real network call to the-odds-api.** Same — needs `OMEGA_ODDS_API_KEY` set
  to a real free-tier key. Run against a current NBA game.
- **End-to-end loop on a real Claude.ai session.** The §10 trace export
  emission has not been seen in practice yet.

---

## What's next — Step 4 (Calibration fit + hybrid promotion)

The plan calls for two scripts and a hybrid auto/manual gate. References to
existing code already present in the repo are in **bold** — these are the
building blocks, do not rebuild them.

**Files to create:**
- `scripts/fit_calibration.py --league NBA`
  1. Load graded NBA traces: **`TraceStore.get_graded_traces(league="NBA")`**.
  2. Train/holdout split (last ~30 days = holdout). Be deterministic.
  3. **`CalibrationFitter.extract_pairs(traces)`** → (prediction, outcome) lists.
     Verify what `extract_pairs` actually returns today by reading
     `omega/core/calibration/fitter.py`.
  4. Fit two candidates: **`fit_isotonic()`**, **`fit_shrinkage()`**.
  5. **`evaluate()`** each on holdout.
  6. Persist via **`CalibrationRegistry.register()`** (status=CANDIDATE).
  7. Emit `reports/calibration_fit_{YYYYMMDD}.md` with metrics table + candidate IDs.

- `scripts/promote_profile.py --candidate-id <id> [--auto] [--manual-override]`
  Hybrid gate per user's answer in plan:
  - **Hard gates:** sample_size ≥ 100, Brier improvement ≥ 1pp absolute over
    incumbent on holdout, no log-loss regression.
  - **Backtest-replay parity gate:** rerun **`BacktestEngine.run()`** on the
    same holdout artifacts with `calibration_policy=<candidate.profile_id>`.
    Candidate must produce ROI within ±0.5% of holdout-implied ROI AND Brier
    consistent with the standalone evaluation.
  - **CLV gate:** mean CLV cents on holdout must not regress by > 0.5
    (this is *new* — bring in `compute_clv` from `omega.trace.clv`, joining
    `bet_records` × `closing_lines` for each holdout trace).
  - `--auto` + all gates pass → **`CalibrationRegistry.promote()`** and append
    to `reports/promotion_audit.jsonl`.
  - Any gate fails → write `reports/pending_review_{candidate_id}.md`, exit
    non-zero. Manual approval via `--manual-override --reason "..."`.

**Scheduled task** (don't create until script works manually):
- `omega-fit-and-promote-nba`: weekly Sunday 09:00 ET via the `scheduled-tasks`
  MCP. Chain: `fit_calibration.py` → `promote_profile.py --auto`.

**Tests to add:**
- `tests/scripts/test_fit_and_promote.py` — synthetic graded-trace fixture →
  fit → assert candidate registered → simulate gate pass/fail.
- Parity check: `test_calibration_parity_service_and_backtest` must continue
  to pass after a promotion (existing test, do not break).

**Risks for Step 4:**
- Step 4 cannot meaningfully run until ≥100 graded NBA traces exist in
  `omega_traces.db`. The scripts will be written and unit-tested with synthetic
  data; the first real fit will need 4–8 weeks of NBA usage.
- the-odds-api free tier excludes historical odds, so closing-line capture
  for any trace is one-shot at scheduled-task time. Missed windows = no CLV
  for those bets. The CLV gate must handle missing closing_lines gracefully
  (compute on the subset that has them, surface the coverage % in the report).

---

## What's next — Step 5 (Drift / health report)

**Files to create:**
- `scripts/report_calibration.py --league NBA` — emits `reports/calibration_health_{YYYYMMDD}.md`:
  - Trace counts: total / graded / with-bet-record / with-closing-line
    (last 7d, last 30d, all-time).
  - Current production profile (`CalibrationRegistry.get_production("NBA")`):
    id, method, age, training window.
  - Last 30d realized: Brier, ECE, log-loss, mean CLV (in cents), betslip ROI
    by tier (A/B/C).
  - Pending candidates from `CalibrationRegistry.list_profiles(status=CANDIDATE)`.
  - Top 5 unmapped team strings flagged by `fetch_outcomes_nba.py` so the
    alias table can be extended (parse log output or capture in DB).

**Scheduled task:**
- `omega-calibration-report-nba`: weekly Sunday 09:30 ET (after fit+promote).

---

## How to pick this up (next session)

```bash
cd C:/Users/camer/OneDrive/Documents/GitHub/Omega
git status                                  # confirm clean
python -m pytest tests/ -q                  # confirm 268 passing
cat docs/phase6/HANDOFF_phase6d_to_h.md     # this file
```

Then:
1. Open `omega/core/calibration/fitter.py` and `omega/core/calibration/registry.py`
   to confirm method signatures (`extract_pairs`, `fit_isotonic`, `evaluate`,
   `compare`, `register`, `promote`, `get_production`).
2. Write `scripts/fit_calibration.py` to exercise that surface. Keep it under
   200 lines.
3. Write `scripts/promote_profile.py`. Pay attention to the backtest-replay
   parity gate — the existing **`BacktestEngine.run(strategy, artifacts)`**
   may need a `calibration_policy` kwarg or env override; check `omega/strategy/backtest/engine.py`
   and verify before writing.
4. Add `tests/scripts/test_fit_and_promote.py` with a synthetic fixture.
5. Run full suite. Commit Step 4 (one PR).
6. Write `scripts/report_calibration.py` + a small markdown formatter helper.
7. Run full suite. Commit Step 5 (one PR).
8. Schedule the two recurring tasks via the `scheduled-tasks` MCP.

---

## Open questions / decisions deferred

1. **CLV gate threshold.** The plan says "must not regress mean CLV by > 0.5
   cents." This is a guess. If the first few fits show real-world CLV variance
   is much larger, loosen it. Document the chosen value in the audit log.
2. **Holdout split window.** "Last 30 days = holdout" is reasonable for NBA's
   tempo, but for low-volume weeks (post-season, off-season) we may need a
   rolling-N approach instead of a calendar window. Decide on first real fit.
3. **Player props in the closing-line path.** Skipped today. The-odds-api free
   tier doesn't cover player props consistently. Need a sourcing decision before
   props get a CLV signal — either upgrade the API tier or scrape a single book.
4. **Auto-ingest of `inbox/traces/`.** Plan flagged this as a future enhancement
   (Claude Code `/loop` or scheduled task). Worth doing once Steps 4+5 land so
   you don't have to remember `python scripts/ingest_traces.py` after each
   Claude.ai session.

---

## Files touched this session

**Created:**
- `omega/trace/bet_record.py`
- `omega/trace/clv.py`
- `omega/integrations/__init__.py`
- `omega/integrations/espn_nba.py`
- `omega/integrations/odds_api.py`
- `scripts/ingest_traces.py`
- `scripts/fetch_outcomes_nba.py`
- `scripts/fetch_closing_lines_nba.py` (now a deprecation shim — see `scripts/fetch_closing_lines.py`)
- `inbox/README.md`
- `inbox/traces/.gitkeep` + `processed/.gitkeep` + `failed/.gitkeep`
- `tests/scripts/__init__.py` + `tests/scripts/test_ingest_traces.py`
- `tests/integrations/__init__.py` + `tests/integrations/test_espn_alias_resolver.py`
- `tests/trace/test_clv.py`
- `tests/trace/test_closing_lines.py`
- `docs/phase6/HANDOFF_phase6d_to_h.md` (this file)

**Modified:**
- `omega/trace/schema.py` — CURRENT_VERSION → 3, added `SCHEMA_V2`, `SCHEMA_V3`
- `omega/trace/store.py` — migration pipeline, `record_bet`, `get_bet_records`,
  `update_bet_status`, `attach_closing_line`, `get_closing_lines`
- `prompts/system_prompt.txt` — §10 and §11 added
- `OMEGA_HANDBOOK.md` — "estimated lean" reference deleted

**Test count:** 203 → 268 (+65 new tests).

---

## Gotchas to remember

- The Claude.ai sandbox **cannot reach localhost** — that's why the bridge is
  copy/paste, not HTTP. Don't propose a FastAPI ingest without solving the
  reachability problem (Cloudflare tunnel or hosted).
- `TraceStore.attach_outcome()` is NOT idempotent on `(trace_id)` — the schema
  technically allows multiple outcome rows per trace. `fetch_outcomes_nba.py`
  filters with `has_outcome=False` to avoid duplicates in practice. If you ever
  re-grade, delete the existing outcome row first.
- The system_prompt.txt is the source of truth for what the LLM emits. Don't
  rewrite the trace adapter in `scripts/ingest_traces.py` to accept a different
  shape — extend §10 instead, or you'll have skewed history.
- Schema migration applies `CREATE TABLE IF NOT EXISTS` for every version on
  every connection. Old DBs converge to current. New tables only — never
  ALTER an existing one without writing a real migration.
