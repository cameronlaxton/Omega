# Session Bug Report — 2026-05-26

Session: `sess-20260526-auto` — MLB daily slate analysis (automated scheduled task); bankroll $1,000

Scope: 7 MLB game-line analyses (fast_score), 5 MLB player prop analyses. 12 traces emitted and ingested. No formal Bet Cards issued (see BUG-SPREAD-1); 2 engine-confirmed prop recommendations (Strider K Over, Turner TB Under).

---

## BUG-SIM-2 (CONFIRMED AGAIN): MLB `draw_prob` ~13–14% on fast_score

**Severity:** High — first reported 2026-05-25; still unpatched

**Affected file:** `omega/core/simulation/engine.py` — winner-attribution path; `omega/core/simulation/archetypes.py` (`BASEBALL` declares `supports_draw=False`)

**Reproduction:** Every MLB game trace this session shows ~13–14% draw probability:

| Trace | Matchup | home_win | away_win | draw |
|---|---|---|---|---|
| sandbox-eea964a2 | CIN @ NYM | 48.2% | 38.0% | 13.9% |
| sandbox-e47ef200 | ATL @ BOS | 48.1% | 38.7% | 13.2% |
| sandbox-c27100bd | MIN @ CWS | 45.8% | 40.4% | 13.8% |
| sandbox-bd37a280 | WSH @ CLE | 48.5% | 37.4% | 14.1% |

Home + away sum to ~86%, leaving both sides systematically below market-implied probabilities. This inflates apparent run-line edge by deflating true_prob relative to any plus-money market (feeds directly into BUG-SPREAD-1 below).

**Expected behavior:** For any archetype with `supports_draw=False`, post-sim tie resolution must convert all drawn outcomes to a home or away win via Bernoulli(0.5) or a strength-weighted coin. `draw_prob` must be 0.0 and `home_win_prob + away_win_prob` must equal 1.0 (within float tolerance).

**Test:** Assert `home_win_prob + away_win_prob == pytest.approx(1.0)` for all MLB and NHL traces.

**Fix priority:** Critical — every MLB game-line edge on this slate is inflated by the draw leak.

---

## BUG-SPREAD-1 (NEW): fast_score computes run-line edge against ML win probability

**Severity:** High — invalidates all spread market Bet Cards from fast_score

**Affected file:** `omega/core/contracts/service.py` (or edge construction layer); fast_score backend

**Symptom:** Every game trace this session shows `spread_coverage_prob: null` in the edges array. The `edge_pct` for the spread market is computed as:

```
edge = true_prob (ML win probability) − market_implied (run-line implied probability)
```

Example from sandbox-bd37a280 (WSH @ CLE):
```json
{
  "side": "home",
  "true_prob": 0.485,
  "market_implied": 0.392,
  "edge_pct": 9.28,
  "market_odds": 155.0,
  "spread_coverage_prob": null
}
```

CLE's ML simulation win probability is 48.5%. The edge is computed against the +155 run-line market (implied 39.2%). But a team with a 48.5% straight ML win probability covers −1.5 runs substantially less often — plausibly 28–35% depending on score distribution. The run-line market at 39.2% implied is likely correct or even offering the model side, not the bettor side. The apparent 9.28% edge is an artifact of the wrong numerator.

**Root cause:** `fast_score` does not compute `spread_coverage_prob` from the simulated score distribution. The edge builder falls back to `home_win_prob` as a proxy for all home-side markets regardless of market type.

**Fix:**
1. After scoring the 10,000 iterations, compute `spread_coverage_prob = P(home_score − away_score > spread_home)` directly from the simulated score pairs.
2. Use this value as `true_prob` when building the spread market edge row.
3. If score pairs are discarded after win/loss attribution, retain them until after the edge layer runs.
4. Add a unit test: for a team projected to win 50% of games, assert `spread_coverage_prob` for −1.5 is materially lower than `home_win_prob`.

**Interaction with BUG-SIM-2:** The draw leak independently depresses both win probabilities. BUG-SIM-2 must be fixed first; after that, BUG-SPREAD-1 is still a separate and independent error.

**Impact this session:** All 4 game Bet Cards (CLE RL +155, BOS RL +145, NYM RL +140, SD RL +165) carry inflated edge figures and were treated as research-only leans in the session output.

---

## BUG-FUSE-2 (NEW): SQLite fallback to DELETE journal mode also fails on FUSE mount

**Severity:** High — escalation of BUG-FUSE-1 (2026-05-25)

**Affected file:** `omega/trace/store.py` lines ~65–73

**2026-05-25 behavior:** `PRAGMA journal_mode=WAL` raised `OperationalError: unable to open database file`. Code caught this and attempted `PRAGMA journal_mode=DELETE` as a fallback.

**2026-05-26 behavior:** The DELETE fallback also raises the same `OperationalError`. The connection never completes in either mode.

```
SQLite WAL mode unsupported on this mount. Falling back to DELETE mode.
sqlite3.OperationalError: unable to open database file   ← fallback also fails
```

The database file exists and is readable (`-rwx------ 737280`). The mount is otherwise writable — JSON files were written to `inbox/traces/` without issue. The failure appears specific to SQLite's requirement to create and unlink lock sidecar files in the same directory as the `.db` file.

**Workaround used this session:**
```bash
cp omega_traces.db /sessions/<name>/tmp/omega_traces.db
python scripts/ingest_traces.py --db /sessions/<name>/tmp/omega_traces.db --verbose
cp /sessions/<name>/tmp/omega_traces.db omega_traces.db   # sync back
```

This required a round-trip write-back. A crash between ingest and copy-back loses the writes.

**Fix options (same as BUG-FUSE-1; escalating urgency — workaround is now more complex):**

- **Option A (preferred):** Add `COWORK_DB_PATH` env var that auto-resolves to a local `/tmp`-equivalent path in the cowork sandbox, with a sync-back hook at session end in `run_action_plan.py` or `ingest_traces.py`.
- **Option B:** Detect FUSE mount in `TraceStore.__init__` via a probe `unlink` attempt; redirect to local temp and register `atexit` sync-back.
- **Option C:** Adjust FUSE mount config to allow `unlink` on `*.db-wal`, `*.db-shm`, `*.db-journal` only.

---

## BUG-PREFLIGHT-3 (NEW): `--repair-from-git` blocked by dirty test files, masking core engine status

**Severity:** Medium

**Affected file:** `scripts/cowork_preflight.py`

**Reproduction:** Preflight reported 3 divergent files this session, all test files:
```
tests/core/test_contracts_service.py
tests/core/test_evidence_to_modifier.py
tests/integrations/test_odds_api.py
```

Running `--repair-from-git` returned:
```
Refusing to --repair-from-git because tracked Python files outside repair targets are dirty.
Use --force-repair to clobber.
```

The repair guard treats all divergent tracked files identically, blocking repair of any file when any other tracked file is dirty — even when all dirty files are tests and core engine source is intact. This session the engine was fully functional (smoke test passed immediately); the `cowork_preflight_failed` banner was noise that would mislead an agent into skipping engine output.

**Fix:** Separate repair targets into two tiers:
- **Core tier** (`omega/`, `scripts/` engine utilities, `omega/mcp/`): eligible for `--repair-from-git` regardless of test file state.
- **Test/integration tier** (`tests/`): blocked by `--repair-from-git` if there are uncommitted changes in the same tier; require `--force-repair`.

Preflight should emit `cowork_preflight_core_ready` (engine clean, tests may diverge) as a distinct status from `cowork_preflight_failed` (core source corrupt). Automated runs should not hard-fail on test-file divergence when the smoke test passes.

---

## BUG-TOTALS-1 (NEW): fast_score does not emit total edges

**Severity:** Medium — totals are the most liquid MLB market after moneyline

**Affected file:** Edge construction layer in `omega/core/contracts/service.py` or fast_score result adapter

**Symptom:** Every game trace this session has `result.edges` containing only `side=home` and `side=away` rows. No `market=total` row is emitted even though the request supplies `over_under` and the simulation reports `predicted_total` and a full score distribution (p10/p50/p90/sample_mean).

**Impact:** The engine projects:
- CIN @ NYM predicted total 8.34 vs line 7.5 (+0.84 Over lean)
- PHI @ SD predicted total 8.75 vs line 7.5 (+1.25 Over lean)

Neither produces a formal Bet Card. These are reported as qualitative leans without engine-backed edge, EV, Kelly, or units — the exact output the engine should be generating from its own simulation.

**Fix:**
1. After computing the simulated score distribution, compute `over_prob = P(home_score + away_score > over_under)` and `under_prob = P(home_score + away_score < over_under)` from the simulated pairs.
2. Build total edge rows using `total_over_price` / `total_under_price` from the odds block (default to `over_under_price` if a single price is provided).
3. Emit them in `result.edges` with `market="total"` and include them in `best_bet` selection if they clear the edge threshold.
4. Test: run a game where predicted total exceeds the line by ≥0.5 and assert a total `market="total"` row appears in edges.

---

## Summary and fix order

| Bug ID | Severity | Status |
|---|---|---|
| BUG-SIM-2: MLB draw prob leak | High | Open (regression from 20260525) |
| BUG-SPREAD-1: ML win prob used for run-line edge numerator | High | New |
| BUG-FUSE-2: DELETE journal fallback also fails on FUSE mount | High | New (escalation of FUSE-1) |
| BUG-PREFLIGHT-3: repair blocked by dirty test files | Medium | New |
| BUG-TOTALS-1: total edges not emitted by fast_score | Medium | New |

**Suggested fix order:**
1. **BUG-SIM-2** — valid MLB win probabilities are a prerequisite for everything below
2. **BUG-SPREAD-1** — run-line Bet Cards are structurally invalid until spread_coverage_prob uses sim pairs
3. **BUG-TOTALS-1** — highest marginal value; totals are liquid and the engine already has the distribution
4. **BUG-FUSE-2** — ingest workaround functions but is fragile; one crash = silent data loss
5. **BUG-PREFLIGHT-3** — low-cost; eliminates false `preflight_failed` noise in automated runs
