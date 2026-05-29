# DRIFT_LOG — Standalone Quant Engine vs. Production Orchestrator

Tracks known behavioral differences between the **standalone quant/backtest engine**
(`omega/strategy/backtest/engine.py`) and the **production analyze path**
(`omega/core/contracts/service.py`). Per `CLAUDE.md`, the two evaluation planes must
not duplicate edge/calibration/staking/grading logic and must share the calibration
selection policy.

Each entry: what differs · why it matters · risk · status. Entries are verified
against current code, not copied from design docs. Last verified: 2026-05-29.

---

## Currently aligned (no drift)

| Concern | Shared implementation | Evidence |
|---|---|---|
| Edge calculation | `edge_percentage()` | both import from `omega/core/betting/odds.py` |
| Implied probability | `implied_probability()` | same module, both paths |
| Kelly staking | `recommend_stake()` | `omega/core/betting/kelly.py`, both paths |
| Simulation engine | `OmegaSimulationEngine` | both paths |

## Resolved drift

| # | What differed | Resolution | Status |
|---|---------------|------------|--------|
| R1 | Calibration method: backtest used "shrinkage", service used "combined"; backtest always calibrated while service checked `should_apply_calibration()`; different cap_min/cap_max | Unified onto a shared calibration selection policy (`omega/core/calibration/*`) | **Resolved** — keep covered by parity tests; re-open if the two paths diverge on profile selection |

## Open / at-risk drift (verified present 2026-05-29)

| # | What differs | Why it matters | Risk | Planned resolution |
|---|--------------|----------------|------|--------------------|
| D1 | **Sim-context validation.** Production calls `validate_sim_context()` before simulating (`service.py`, `omega/core/simulation/validation.py`); the backtest engine does not. | Invalid fields in frozen artifacts pass through the backtest but would be rejected in production, inflating apparent coverage. | Medium | Call the same `validate_sim_context()` in the backtest engine, or pre-validate artifacts at freeze time. |
| D2 | **Market coverage.** The backtest engine evaluates **moneyline home/away only** (`engine.py:259-294`). Production also handles spread, totals (`_analyze_totals()`), and 3-way `moneyline_draw`. | Backtests silently omit spread/total/draw edges, so benchmark ROI/CLV is not representative for those markets or for soccer/hockey. | Medium | Extend the backtest engine to the same market set as `service.analyze_game`, reusing the shared edge/staking helpers. Until then, scope backtest claims to moneyline explicitly. |

---

### Maintenance

- Re-verify each open entry whenever `engine.py` or `service.py` market/validation/calibration logic changes.
- When an entry is resolved, move it to **Resolved drift** with the resolving change and a parity test reference — do not delete it.
- New drift is discovered fastest by diffing the market/validation/calibration seams of the two files; add an entry before merging the divergence.
