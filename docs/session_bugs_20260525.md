# Session Bug Report — 2026-05-25

Session: sess-20260525-18b5 — NBA + MLB best-bets scan; user bankroll $1000

Scope: 13 MLB game-line analyses, 1 NBA ECF G4 game-line (Markov), 7 NBA player props (Normal). Final output: no Bet Cards; research-only leans on a subset of NBA props. MLB output suppressed end-to-end due to BUG-SIM-1.

---

## BUG-SIM-1 (CRITICAL): `_sim_baseball` inverted `def_rating` math

**Severity:** Critical — every MLB game-line trace this session is corrupted; all underdog edges are inverted

**Affected file:** `omega/core/simulation/engine.py`, function `_sim_baseball`, lines ~648–649

**Current code:**
```python
home_lambda = (home_off * (league_avg_rpg / away_def)) * park_factor
away_lambda = (away_off * (league_avg_rpg / home_def)) * park_factor
```

**Bug:** The archetype declares `def_rating = runs allowed per game` (lower = better defense). The current divisor inverts the effect: a strong pitching staff (low `def_rating`) produces a LARGER `(league_avg_rpg / def)` multiplier, increasing the opponent's expected runs. Elite pitching makes the model project the opponent to score MORE, not fewer.

**Observed impact this session:**
| Matchup | Real market | Engine edge call | What engine "saw" |
|---|---|---|---|
| COL +260 @ LAD -325 | LAD heavy chalk | Rockies +26.47% edge, Tier A, 5u | LAD's 3.17 RA/G blew Rockies' expected runs UP |
| WSH +145 @ CLE -175 | CLE chalk | Nationals +31.98% edge, Tier A, 5u | CLE's 3.80 RA/G inflated WSH expected runs |
| STL +190 @ MIL -235 | MIL chalk | Cardinals +14.92%, A, 5u | Same pattern |
| HOU +105 @ TEX -130 | Slight TEX edge | Astros +14.72%, A, 5u | Same pattern |

**Correct formula:**
```python
home_lambda = (home_off * (away_def / league_avg_rpg)) * park_factor
away_lambda = (away_off * (home_def / league_avg_rpg)) * park_factor
```
With this, low `away_def` (good opposing pitching) reduces `home_lambda`, which is the intended direction.

**Probable companion bug (same pattern):** `_sim_hockey` lines ~684–686 use the same `(league_avg / opp_def)` form. Confirm and fix together.

**Fix steps:**
1. Patch `_sim_baseball` and `_sim_hockey` divisor → multiplier flip.
2. Add regression test in `tests/core/test_engine.py`: assert that holding `home_off` fixed, INCREASING `away_def` (worse opposing defense) INCREASES `home_lambda`. The existing test suite passed with the inverted code, so it has no asymmetric-defense assertion.
3. Backfill: re-grade or invalidate every MLB trace older than the patch date in `var/omega_traces.db` so calibration learning doesn't anchor on broken predictions.

**Rollback:** Trivial — single-line revert per simulator.

---

## BUG-SIM-2: MLB sim emits `draw_prob` despite `supports_draw=False`

**Severity:** High — deflates home/away win probabilities by ~10–13 percentage points

**Affected files:** `omega/core/simulation/engine.py` (post-sim winner attribution), `omega/core/simulation/archetypes.py` (BASEBALL archetype declares `supports_draw=False`)

**Reproduction:** sess-20260525-18b5, TB @ BAL trace `sandbox-d1d34ac2-491a`:
```
home_win_prob=60.0  away_win_prob=27.3  draw_prob=12.7   (sum=100.0)
```
Baseball does not allow regulation draws — extra innings always resolve.

**Fix:** In the winner-attribution path, after Poisson sampling home/away scores, for sports with `supports_draw=False` resolve ties by Bernoulli(0.5) (or by team-strength weighting). Do not emit a `draw_prob` field on traces whose archetype declares `supports_draw=False`.

**Test:** Assert `home_win_prob + away_win_prob == 1.0` (within float tolerance) for any trace whose archetype declares `supports_draw=False`.

**Interaction with BUG-SIM-1:** Fixing only the draw leak without fixing the def inversion will produce more confident WRONG bets. Fix BUG-SIM-1 first.

---

## BUG-PREFLIGHT-1: `cowork_preflight.py --repair-from-git` is a silent no-op

**Severity:** High — masks Pattern C mount corruption

**Affected file:** `omega-cowork-preflight`

**Reproduction:** Six tracked files were truncated on mount at session start:
- `omega-cowork-preflight` (283 of 330 lines)
- `omega/core/contracts/evidence.py` (corrupt at line 470)
- `omega/core/contracts/service.py` (corrupt at line 1252)
- `omega-run-champion-challenger` (corrupt at line 265)
- `tests/core/test_engine.py` (corrupt at line 408)
- `tests/mcp/test_mcp_tools.py` (corrupt at line 197)

Running `omega-cowork-preflight --repair-from-git` exited without modifying any of the six files; re-running preflight immediately reported the same six errors.

**Suspected root cause:** The `--repair-from-git` code path likely writes via Python `open(..., 'w')` or `Path.write_text(...)` through the FUSE mount cache, which does not invalidate on Windows-side writes (this is the same pathology the docstring warns about for the Write tool). The repair must shell out to `git checkout HEAD -- <file>` followed by `os.sync()` so the kernel re-reads.

**Workaround used this session:**
```bash
for f in omega-cowork-preflight omega/core/contracts/evidence.py omega/core/contracts/service.py omega-run-champion-challenger tests/core/test_engine.py tests/mcp/test_mcp_tools.py; do
  git cat-file -p "HEAD:$f" > "/tmp/_fix_$(basename $f)"
  cp "/tmp/_fix_$(basename $f)" "$f"
done
```
This restored all six and preflight printed `cowork_preflight_ready`.

**Fix:** Rewrite `--repair-from-git` to use `subprocess.run(["git", "checkout", "HEAD", "--", path], check=True)` then `os.sync()`. Add a self-test: after repair, re-AST-parse the file and report success/failure per file.

---

## BUG-FUSE-1: SQLite WAL mode fails on cowork FUSE mount

**Severity:** High — blocks `report_calibration.py`, `ingest_traces.py`, `fit_calibration.py`, any `TraceStore` consumer that runs in cowork

**Affected file:** `omega/trace/store.py` line ~63 — `self._conn.execute("PRAGMA journal_mode=WAL")`

**Symptom:**
```
sqlite3.OperationalError: unable to open database file
```
The `var/omega_traces.db` itself is readable and the directory accepts `touch` (file creation). But the mount denies `rm`, and SQLite WAL requires the ability to manage `-wal` and `-shm` sidecar files. The journal-mode handshake fails on its own cleanup path.

**Workaround used this session:**
```bash
cp var/omega_traces.db /tmp/omega_traces.db
omega-report-calibration --db /tmp/omega_traces.db ...
omega-ingest-traces --db /tmp/omega_traces.db ...
```
Writes against `/tmp/omega_traces.db` are NOT persisted back to the mount, so this is a READ-only workaround. Production ingest from cowork still needs a real fix.

**Fix (two options):**
1. Add a `TraceStore(journal_mode=...)` arg defaulting to `"WAL"` but settable to `"DELETE"` (no sidecars). Detect cowork via `os.environ.get("COWORK_SANDBOX")` or a mount probe and auto-downgrade.
2. Loosen the FUSE mount config to allow `unlink` on `*.db-wal` and `*.db-shm` only.

Option 1 is simpler and self-contained.

---

## BUG-INGEST-1: `ingest_traces.py` crashes at end with `TraceStore.close` AttributeError

**Severity:** Low — cosmetic; ingest completes before the crash

**Affected file:** `omega-ingest-traces` line ~373 — `store.close()` is called but `TraceStore` does not define `close()`

**Reproduction:**
```
2026-05-25 18:23:40,476 INFO ingest_traces: Done. 34 ingested, 0 failed.
Traceback (most recent call last):
  File ".../omega-ingest-traces", line 378, in <module>
    sys.exit(main())
  File ".../omega-ingest-traces", line 373, in main
    store.close()
AttributeError: 'TraceStore' object has no attribute 'close'
```

**Fix:** Either add `def close(self): self._conn.close()` to `TraceStore`, or wrap the call in `if hasattr(store, 'close'): store.close()`. Prefer adding the method — explicit close is correct.

---

## BUG-SIDECAR-1: SessionSidecar schema rejects `outcomes` key

**Severity:** Medium — recurrence of pattern from BUG-SS-1 (2026-05-22)

**Affected file:** `var/inbox/sessions/sess-20260524-nba1.json` (extra `outcomes` key)

**Symptom:** `validate_session_sidecars.py` and `report_calibration.py` both reject the sidecar with:
```
SessionSidecar: outcomes — Extra inputs are not permitted
```

**Why this is a fresh bug, not a re-occurrence of BUG-SS-1:** BUG-SS-1 (5/22) was about migrating historical pre-schema sidecars. The 5/24 sidecar is post-schema and was likely written by a prior agent run that wanted to attach trace-level outcome summaries inline.

**Fix:** Decide whether session-level outcome attachment is a real product need:
- If yes: add `outcomes: dict[str, OutcomeSummary] | None = None` to `SessionSidecar` schema.
- If no: amend `OMEGA_COWORK.md` §8 to explicitly state that `outcomes` belongs in the per-trace `bet_record`, not in the sidecar; correct the offending sidecar.

---

## BUG-MARKOV-1: NBA Markov backend emits no spread/total edges

**Severity:** Medium — silently reduces NBA game-line surface area

**Affected files:** `omega/core/simulation/engine.py` Markov backend path, `omega/core/contracts/service.py` edge construction

**Reproduction:** sess-20260525-18b5 NBA G4 trace `sandbox-38a29cfb-a73c`:
- Input request supplied `spread_home=2.5`, `spread_home_price=-110`, `spread_away_price=-110`, `over_under=218.5`, `total_over_price=-110`, `total_under_price=-110`
- Output `result.edges` contained only moneyline rows (`side=home, market=None` and `side=away, market=None`). No `market="total"` or `market="spread"` rows.

The fast_score backend on the same kind of request produces spread and total edges in MLB traces (confirmed in BUG-SIM-1 examples). The difference is the Markov code path.

**Fix:** Audit Markov result construction to ensure it builds spread and total edges from the simulated final scores the same way fast_score does. Add an integration test that runs the same NBA request through both backends and asserts the same edge MARKETS appear (not the same values — same set of markets).

---

## ENGINE-INVOCATION-1: No MCP tool surface in cowork client; no CLI wrapper for `analyze()`

**Severity:** Process — pushed the agent to write a one-off batch script in `scripts/` (since removed; stub left as a tombstone)

**Context:** OMEGA_COWORK §2 specifies tier order: (1) local MCP server (`python -m omega.mcp.server`), (2) direct repo imports. The Cowork client this session does NOT expose `omega_resolve_odds`, `omega_analyze_game`, `omega_analyze_prop`, `omega_markov_evidence_guide` as callable MCP tools — the only tools visible in the client are generic shell/file/web tools. So tier 1 was effectively unavailable.

For a multi-game slate, tier 2 (direct `analyze()` calls) requires either:
- N hand-typed `python -c '<inline>'` invocations (one per game/prop), each repeating bankroll/session/trace-export boilerplate
- A batch loop file

The agent chose the batch loop and put it in `scripts/`, which is wrong: `scripts/` is for canonical, version-controlled, repeatable utilities. Session scratch belongs in `/tmp/`.

**Recommended fixes (pick one or both):**

1. **Add a thin CLI wrapper for `analyze()`:** `omega-run-analyze --kind {game,prop} --league NBA --home-team ... --away-team ... --player-name ... --prop-type ... --line ... --odds-file <json> --session-id ... --bankroll ... --trace-out var/inbox/traces/`. Emits one trace per call. The agent then loops via shell; no per-session Python scratch needed. Wraps the engine and centralizes seed derivation, session_id propagation, reasoning_inputs/evidence collection, trace export, and aggregate_quality stamping.

2. **Surface MCP tools in cowork:** Add the omega MCP server to the cowork tool registry (or document the connection procedure) so tier 1 actually works. Without this, every multi-game session forces tier 2 batching.

If neither is done, the next agent will hit the same fork: write throwaway Python in `scripts/` (wrong) or do N inline `python -c` calls (clumsy and unobservable). Both are worse than fixing the tool surface.

---

## MEMORY-NOTE: `MEMORY.md` index entries added this session

For traceability between this report and the agent's persistent memory:
- `bug_baseball_def_rating_inverted.md` — BUG-SIM-1 above
- `bug_mlb_draw_prob_leak.md` — BUG-SIM-2 above
- `bug_repair_from_git_noop.md` — BUG-PREFLIGHT-1 above
- `bug_sqlite_wal_fuse.md` — BUG-FUSE-1 above

---

## Suggested fix order

1. **BUG-SIM-1** then **BUG-SIM-2** — re-enables MLB output
2. **BUG-PREFLIGHT-1** — every session that starts dirty wastes time without it
3. **BUG-FUSE-1** — unblocks calibration and ingest in cowork
4. **ENGINE-INVOCATION-1** — prevents this whole class of session-script mistakes
5. **BUG-MARKOV-1** — NBA game lines stay narrow until fixed
6. **BUG-INGEST-1**, **BUG-SIDECAR-1** — low-cost cleanups
