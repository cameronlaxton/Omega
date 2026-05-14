# Phase 6 Red Team — Two-Plane Evaluation Architecture

## Context

CURRENT_HANDOFF.md proposes a two-plane evaluation model:
- **Quant plane** (standalone backtest engine): benchmark source of truth for forecast, calibration, EV, staking quality
- **Replay plane** (orchestrator replay with live fetching disabled): sampled audit for routing, evidence selection, downgrade discipline

This document answers the 6 red-team questions required before implementation.

---

## Question 1: What are the strongest arguments for and against two-plane evaluation versus unified orchestrator replay?

### For two-plane (current recommendation)

1. **Speed.** The standalone backtest engine processes thousands of games in seconds. Orchestrator replay must run 7 pipeline stages per game. A 2,000-game backtest takes ~2 seconds standalone; orchestrator replay at ~200ms per game takes ~400 seconds. This is not a marginal difference — it determines whether calibration iteration is practical.

2. **Determinism.** The backtest engine has no LLM calls, no evidence gathering, no intent parsing. Its outputs are fully reproducible from frozen inputs. Orchestrator replay adds nondeterminism from LLM routing (unless the LLM is mocked, which creates a separate fidelity problem).

3. **Responsibility separation.** The quant plane tests the deterministic engine (sim → calibrate → edge → stake → grade). The replay plane tests the LLM-controlled layer (route → plan → gather → evaluate). These are different quality dimensions. Mixing them into one path produces metrics that conflate model quality with agent quality.

4. **Failure isolation.** When backtest metrics degrade, you know the deterministic engine changed. When replay metrics degrade, you know the orchestration logic changed. A unified path cannot distinguish these.

### Against two-plane (arguments for unified replay)

1. **Drift risk.** The standalone engine and the orchestrator can diverge. The calibration drift bug (just fixed) is evidence this has already happened. Any behavior that exists in production but not in the backtest engine is invisible to evaluation.

2. **Feature engineering gap.** The orchestrator assembles team contexts from gathered facts. The backtest engine receives pre-assembled contexts. If the context assembly logic introduces errors (e.g., wrong field mapping, entity mismatch), the backtest never sees them.

3. **Reduced surface area.** One path is simpler to maintain than two. Every feature change must be verified against both paths.

### Verdict

**Keep two-plane, but close the drift gap explicitly.**

The speed and determinism advantages are real and significant. The drift risk is real but manageable if:
- Calibration policy is shared (now fixed)
- Edge, staking, and grading functions are shared (already true — both import from `core/betting/`)
- Frozen artifacts are derived from orchestrator outputs, not hand-constructed (Phase 6 task)
- A periodic "parity check" validates that production-path outputs match backtest-path outputs on the same inputs

---

## Question 2: What blind spots does the two-plane design introduce?

### Blind spot 1: Context assembly errors

The orchestrator builds `home_context` and `away_context` from `GatheredFact` objects. The backtest receives pre-built contexts. If the orchestrator maps `off_rating` to the wrong field, or silently drops a stat, the backtest will never see it.

**Mitigation:** The frozen artifact converter (Phase 6) must extract contexts from persisted traces, not from a separate pipeline. This ensures that what the orchestrator actually used is what the backtest evaluates.

### Blind spot 2: Quality gate interaction with predictions

The orchestrator's quality gate may downgrade a sim to research mode. The trace records this downgrade. But the backtest engine always runs the sim because it receives pre-validated contexts. It never exercises the "should we even simulate?" decision.

**Mitigation:** This is correct behavior — the quant plane is testing forecast quality, not routing quality. Routing quality is the replay plane's job. Document this boundary explicitly.

### Blind spot 3: Odds format inconsistencies

The orchestrator gathers odds from live providers and normalizes them. The backtest receives pre-normalized odds. If the normalization has bugs, the backtest doesn't catch them.

**Mitigation:** Frozen artifacts must capture the raw odds as gathered by the orchestrator. The backtest should consume the same odds the orchestrator used, not re-normalized values.

### Blind spot 4: Seed derivation differences

The orchestrator derives seeds from `sha256(prompt + date)`. The backtest engine uses whatever seed is in the strategy params or no explicit seed. These are different derivation paths.

**Mitigation:** Phase 6 frozen artifacts must include the exact seed used. The backtest must use that seed, not derive a new one.

---

## Question 3: Where could the standalone quant engine drift from the real production path?

### Already drifted (now fixed)
- Calibration method: backtest used "shrinkage", service used "combined"
- Calibration gate: backtest always calibrated, service checked `should_apply_calibration()`
- Calibration parameters: different cap_max/cap_min values

### Currently aligned
- Edge calculation: both import `edge_percentage()` from `core/betting/odds.py`
- Implied probability: both import `implied_probability()` from same module
- Kelly staking: both import `recommend_stake()` from `core/betting/kelly.py`
- Simulation engine: both use `OmegaSimulationEngine`

### At risk of future drift
- **Simulation context validation.** The orchestrator calls `validate_sim_context()` before sim. The backtest engine does not. If contexts in frozen artifacts have invalid fields, the backtest won't reject them the way production would.
- **3-way markets.** Service handles `moneyline_draw` for soccer/hockey. Backtest engine only handles home/away ML. A backtest for soccer would miss draw edges.
- **Over/under markets.** Service has a `_analyze_totals()` path. Backtest engine only evaluates moneyline sides.

### Recommended action
Add a `DRIFT_LOG.md` to `omega/strategy/` documenting known differences. Each entry should have: what differs, why, risk level, and planned resolution.

---

## Question 4: What exactly must be frozen to make replay and quant evaluation historically valid?

### For quant backtest (frozen artifact):
1. **Event identity** — home_team, away_team, league, date (unique key)
2. **Team contexts** — the exact `home_context` and `away_context` dicts as used in the sim (extracted from persisted trace, not re-gathered)
3. **Odds snapshot** — exact odds at decision time (from trace.odds_snapshot)
4. **Simulation seed** — the exact seed used (from trace.simulation_seed)
5. **Calibration policy reference** — which calibration profile was active (for Phase 6 learned profiles; for now, "static_v1")
6. **Actual outcome** — home_score, away_score (attached after the fact)
7. **Schema version** — integer version for format evolution
8. **Source trace ID** — trace_id linking back to the original execution trace

### For replay audit (replay fixture):
1. **Original prompt** — what the user asked
2. **Evidence bundle** — all `GatheredFact` objects as returned at decision time
3. **Timestamp** — when the query ran (affects seed derivation)
4. **Replay mode flag** — to disable live fetching
5. **Expected outputs** — the trace's routing decision, downgrades, and output packages (for comparison)

### What must NOT be included in pre-decision artifacts:
- Actual outcome (only attached at grading time)
- Post-game stats (e.g., final box scores in the contexts)
- Closing line (stored separately for CLV, not used in decision)

---

## Question 5: Which path is the source of truth for model quality versus agent quality, and why?

### Model quality → Quant plane (standalone backtest)

**What it measures:** Given correct inputs, does the engine produce good forecasts, well-calibrated probabilities, real edges, and sound stakes?

**Why this path:** The backtest engine isolates the deterministic pipeline (simulate → calibrate → edge → stake → grade). No routing, no evidence gathering, no LLM. If this path shows poor Brier scores or negative ROI, the mathematical model needs improvement regardless of how good the agent is at gathering data.

**Metrics:** Brier score, ROI, CLV, win rate, calibration curve, max drawdown.

### Agent quality → Replay plane (orchestrator replay)

**What it measures:** Given a historical prompt and knowable-at-the-time evidence, does the agent make good routing decisions, gather the right data, apply appropriate downgrades, and refuse when it should?

**Why this path:** The orchestrator's LLM-controlled decisions (intent classification, evidence arbitration, quality gate application) are not exercised by the standalone backtest. Replay with frozen evidence bundles tests whether the agent's reasoning layer is disciplined.

**Metrics:** Routing accuracy (did it pick the right execution mode?), downgrade discipline (did it downgrade when it should have?), refusal accuracy (did it refuse unsupported requests?), trace completeness.

### Why they cannot be merged

A unified path would produce aggregate metrics where agent routing errors and model calibration errors are confounded. A bad Brier score could mean the sim is mis-calibrated OR the agent fed it bad contexts. Separating them makes diagnosis possible.

---

## Question 6: What is the recommended Phase 6 implementation plan after considering those tradeoffs?

See `docs/phase6/DESIGN_PLAN.md` (companion document).

The plan follows the order specified in CURRENT_HANDOFF.md:
1. Trace persistence (SQLite in omega/trace/)
2. Backtest integration (frozen artifacts derived from traces)
3. Calibration learning (profile fitting, versioning, promotion)

Each step closes one of the blind spots identified above.
