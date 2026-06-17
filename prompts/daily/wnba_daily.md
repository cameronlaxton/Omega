# WNBA Daily League Session

## Session Guard

Apply the `omega-failure-budget` skill before setup, slate building, odds resolution, analysis, trace export, ingest, or outcome attachment.
If setup/preflight/DB/trace/outcome checks exceed the failure budget, stop and produce the required failure report. Do not continue into analysis or candidate generation after a hard setup failure.

Use this prompt for the complete WNBA betting surface: moneylines, spreads,
team totals where supported, and player props. WNBA game totals are currently
quarantined by the engine and must not be presented as formal Bet Cards.
Do not run a separate props prompt.

Follow [OMEGA_COWORK.md](../../OMEGA_COWORK.md) and
[prompts/system_prompt.txt](../system_prompt.txt) throughout.

---

## Step 0 - Calibration Snapshot

Run:

```bash
omega-report-calibration --league WNBA --window-days 30
```

**Mandatory:** read [`prompts/reference/output_modes.md`](../reference/output_modes.md) before
producing any output, and read the machine-readable `output_modes` map from the
`var/reports/latest.md` **frontmatter** — it carries a mode **per market** (`output_modes.game` and
`output_modes.prop`, each `research_candidate` or `actionable`). Authorize **per market**: a
`RESEARCH_CANDIDATE` game market and an `ACTIONABLE` prop market (or vice versa) can coexist, so
suppress or release Bet Cards for games and props off their own market's mode. `RESEARCH_CANDIDATE`
restricts only what you show the user for that market — it never skips the engine. (The scalar
`output_mode` is a conservative fallback only; prefer the per-market map.)

Read `var/reports/latest.md` before analysis. Expect small samples; when the
calibration report says `insufficient_n`, treat that as unproven, not as
positive evidence.

---

## Step 1 - Formal Gate And Session

Run:

```bash
omega-cowork-preflight --formal-output-gate
```

If this does not print `cowork_preflight_ready`, do not emit Bet Cards.

Mint one league session ID: `sess-YYYYMMDD-wnb1`.

Bootstrap or update `var/inbox/sessions/<session_id>.json` and append a preflight
audit event with status and bankroll confirmation.

---

## Step 2 - Build One League Slate

Create one candidate list containing both game markets and player props:

```text
candidate_id | market_family | matchup | player | market | line | source | status
```

Market families:
- `game`: moneyline and spread; team_total only if sourced and engine-supported.
- `suppressed`: WNBA total. Keep as audit metadata only.
- `prop`: player points, rebounds, assists, 3pm, PRA, steals, blocks, and other
  supported basketball prop types from
  [prop_stat_keys.md](../reference/prop_stat_keys.md).

Do not discard props just because game markets are thin. WNBA prop edges may be
the primary actionable surface, but they still need the same league context.

---

## Step 3 - Resolve Odds

Game markets:

```bash
omega-resolve-odds --kind game --league WNBA --home-team "Team A" --away-team "Team B"
```

Player props:

```bash
omega-resolve-odds --kind prop --league WNBA --player "Player Name" --prop-type pts --line 15.5
```

WNBA prop rule:
- Query direct sportsbook/Odds API prop boards first through the typed odds
  path.
- Standard O/U player props are valid when returned by the typed path or a
  direct sportsbook board.
- Milestone props may be recorded as context, but manually estimated lines are
  research-only and must not be passed to `analyze()` as formal odds.

Never infer a standard O/U line from a milestone threshold.

Default book is BetMGM unless the user asks for broader line shopping. The
source book is recorded on every persisted bet (`bet_ledger.bookmaker`). When
you line-shop (`--line-shopping` / `--all-books`), the resolver payload includes
a `best_prices` block — surface it as the advisory "Best available" line on the
Bet Card per
[`prompts/reference/output_modes.md`](../reference/output_modes.md#book-provenance--line-shopping-in-the-bet-card).
It is advisory only: never recompute edge/EV/Kelly against a shopped price.

---

## Step 4 - League Context And Injury Translation

Gather one shared WNBA context pack:
- injury statuses and late scratches
- minute limits and role changes
- starting lineup / rotation notes
- rest days and travel
- pace environment
- defensive matchup notes
- sportsbook line source and timestamp

Injury/news protocol:
1. Noticing news is not enough.
2. Translate news into structured model inputs:
   - game plane: `off_rating`, `def_rating`, `pace`, typed evidence
   - prop plane: `minutes`, `usage_rate`, `{stat}_mean`, `{stat}_std`, or
     explicit `injury_impact`
3. If the impact cannot be quantified from cited pre-decision sources, append
   `quality_gate/null_data_audit` and downgrade to research-only.

Basketball team contexts require possession-adjusted ratings:

```python
home_context={"off_rating": 105.0, "def_rating": 102.0, "pace": 82.0}
away_context={"off_rating": 108.0, "def_rating": 101.0, "pace": 82.0}
```

Never pass raw FG%, opponent FG%, eFG%, or other fractional proxies as
`off_rating` or `def_rating`.

Both `home_context` and `away_context` must contain all required team context keys (e.g., `off_rating`, `def_rating`, `pace` for WNBA) to guarantee `context_source="provided"` and satisfy the calibration eligibility gate.

`game_context` is mandatory for every WNBA game and prop trace:

```python
game_context={
    "is_playoff": False,
    "rest_days": 1,
    "injury_impact": 1.0,  # include when relevant; omit only if not applicable
}
```

---

## Step 5 - WNBA Market Protections

WNBA `total` markets are suppressed in `omega/core/contracts/service.py`.
If a total line is sourced, the response should carry:

```python
metadata.suppressed_markets == ["WNBA:total"]
```

Presentation must say "no actionable total market" from metadata. Do not
create a prose warning attached to an actionable total edge row.

Actionable WNBA candidates may include:
- moneyline
- spread
- player props with direct O/U lines

Research-only WNBA candidates include:
- game totals
- estimated prop lines
- milestone-only interpretations without a standard O/U line
- injury-sensitive props without quantified minutes/usage/stat impact

---

## Step 6 - Typed Evidence

Express material evidence as `EvidenceSignal` objects. Do not hide predictive
metrics in prose.

Markov-approved game-plane signal types and the ±15% cap are listed in the canonical
[`prompts/reference/markov_evidence_vocab.md`](../reference/markov_evidence_vocab.md). Use the exact
`signal_type` keys from that file. All other signal types are audit-only unless the engine maps
them.

---

## Step 7 - Run Game Engine

**Always run the engine when it is available, regardless of the Step 0 output mode.**
`RESEARCH_CANDIDATE` only restricts what you present to the user; it never means "skip
`analyze()`". Running the engine and persisting traces is how calibration data accumulates.

Use the default WNBA backend unless there is a documented reason to override.

```python
analyze({
    "league": "WNBA",
    "home_team": "...",
    "away_team": "...",
    "game_date": "YYYY-MM-DD",
    "home_context": {"off_rating": ..., "def_rating": ..., "pace": ...},
    "away_context": {"off_rating": ..., "def_rating": ..., "pace": ...},
    "game_context": {"is_playoff": False, "rest_days": 1},
    "odds": {...},
    "evidence": [...],
    "n_iterations": 10000,
    "seed": <sha256_seed>,
}, session_id="sess-YYYYMMDD-wnb1", bankroll=1000.0)
```

If only a total market is supplied, expect `best_bet=None` and
`metadata.suppressed_markets=["WNBA:total"]`.

---

## Step 8 - Run Prop Engine

Every prop trace must include `home_team`, `away_team`, and `game_date`.

```python
analyze({
    "player_name": "Player Name",
    "league": "WNBA",
    "prop_type": "pts",
    "line": 15.5,
    "home_team": "Dallas Wings",
    "away_team": "Las Vegas Aces",
    "game_date": "YYYY-MM-DD",
    "odds_over": -110,
    "odds_under": -110,
    "player_context": {
        "pts_mean": 16.2,
        "pts_std": 4.8,
        "sample_size": 10,
        "minutes": 30,
        "usage_rate": 0.24
    },
    "game_context": {"is_playoff": False, "rest_days": 1, "injury_impact": 1.0},
    "evidence": [...],
    "n_iterations": 10000,
    "seed": <sha256_seed>,
}, session_id="sess-YYYYMMDD-wnb1", bankroll=1000.0)
```

If sample size is thin, opponent context is missing, or the injury/minutes
translation is not quantifiable, downgrade or keep research-only.

---

## Step 8b - Engine Output Nullability Check

**Execute immediately after each `analyze()` call returns, before any other processing.**

Follow the canonical procedure in
[`prompts/reference/engine_output_validation.md`](../reference/engine_output_validation.md).
For WNBA, the sport-specific input-context fields to verify are `home_context.off_rating`,
`def_rating`, `pace`, and injury-impact quantification (`minutes`, `usage_rate`, or explicit
`injury_impact`). Downgrades here are user-facing only — the engine already ran and the trace still
persists (see [`output_modes.md`](../reference/output_modes.md)).

---

## Step 9 - Pre-Export Quality Gate

Before exporting any trace or presenting a Bet Card:
- Confirm `game_context.is_playoff` and `game_context.rest_days` were populated and validated (Step 8b).
- Confirm injury impacts were translated or explicitly audited as missing (Step 8b).
- Confirm WNBA total suppression metadata is honored.
- Confirm engine output passed nullability check (Step 8b).

If critical missing data was found and logged in Step 8b, the trace is already
downgraded to research-only. Do not emit Bet Cards for research-only traces.

---

## Step 10 - Export, Confirm, Close

After each successful `analyze()` call, write `var/inbox/traces/<trace_id>.json`.
Nest `reasoning_inputs`, `reasoning_narrative`,
`reasoning_downgrade_rationale`, and `trace_quality` inside the inner `trace`
block.

If the user confirms a bet, re-export the same trace with `bet_record`
populated. Reuse the original `trace_id`; do not rerun `analyze()`.

Close the session by setting `closed_at` and a short `agent_notes` summary.
Do not run ingest or audit rendering inside the live betting session.

---

## Post-Session

Run separately after session close:

```bash
omega-run-action-plan fixtures/action_plans/daily_trace_intake.json
omega-run-action-plan fixtures/action_plans/render_session_audits.json
```

After games are final, run the outcome loop described in
[ops/fetch_outcomes.md](../ops/fetch_outcomes.md).
