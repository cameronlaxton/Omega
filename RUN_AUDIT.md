# RUN_AUDIT — OKC Thunder vs San Antonio Spurs, WCF Game 5
**Session:** `sess-20260526-nba1`  
**Date:** 2026-05-26  
**Model:** `claude-sonnet-4-6` / `omega-core-phase6h`  
**Bankroll:** $1,000.00  
**Sidecar:** `RUN_TRACE.jsonl` (21 events)

---

## 1. Pipeline Steps (Sequential)

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Preflight | ✅ `cowork_preflight_ready` | Python 3.10.12; 7 stale .pyc skipped safely |
| 2 | Game identification | ✅ | WCF G5: OKC vs SA, 8:30 PM ET, Paycom Center |
| 3 | Live odds resolution | ✅ | BetMGM via `the-odds-api`; confirmed at 22:18Z |
| 4 | Injury research | ✅ | Williams questionable; Mitchell OUT; Fox/Harper available |
| 5 | Team/player stats | ✅ | Series averages, G4 box score, RS h2h trend |
| 6 | Evidence schema correction | ✅ | 3 category fixes + 1 window fix applied before engine run |
| 7 | Engine batch run (10 candidates) | ✅ 9/10 | PRA initially skipped → fixed + rerun |
| 8 | Candidate filtering | ✅ | 7 recommended; 3 rejected (rec=pass) |
| 9 | Trace export (10 files) | ✅ | `inbox/traces/*.json` |
| 10 | RUN_TRACE.jsonl + RUN_AUDIT.md | ✅ | This file |

---

## 2. Data Provenance

| Data Type | Source | Confidence |
|-----------|--------|------------|
| Game line (ML, spread, total) | `the-odds-api:betmgm` — live API call at 22:18Z | High (confirmed) |
| Prop lines | DraftKings/FanDuel web consensus (no props API call) | Medium (estimated) |
| Injury report | FanSided, DK Network, Fadeaway World (Williams), Heavy.com | High |
| Series averages (Wemby, SGA) | nba.com, Yahoo Sports, Bleacher Report | High |
| Fox since-return stats | nba.com box scores (2 games) | High (small n=2) |
| Team ratings (off/def) | Basketball-Reference 2026 playoffs | Medium (pre-series-weighted) |
| RS under trend | CBS Sports ("3 of 5 RS meetings") | Medium |
| SGA usage bump without Williams | PrizePicks / SI (+8 pts/75 possessions) | Medium |

---

## 3. Validation Outcomes

| Candidate | Engine Status | Rec | Edge % | EV % | Tier | Kelly | Units |
|-----------|--------------|-----|--------|------|------|-------|-------|
| Game Total UNDER 217.5 (-110) | success | **under** | **+22.83** | +54.80 | A | 0.196 | 5.0† |
| SA +4.5 (-115) | success | **away** | **+23.13** | +55.52 | A | 0.198 | 5.0† |
| SA ML +140 | success | **away** | **+22.53** | +54.08 | A | 0.193 | 5.0† |
| Wemby pts UNDER 29.5 (-105) | success | **under** | +5.55 | — | A | 0.057 | 5.0 |
| Wemby reb O/U 12.5 | success | pass | — | — | — | — | — |
| Wemby blk O/U 2.5 | success | pass | — | — | — | — | — |
| SGA pts UNDER 26.5 (-110) | success | **under** | **+14.69** | — | A | 0.154 | 5.0 |
| Wemby PRA UNDER 46.5 (-105) | success (rerun) | **under** | +4.60 | — | A | 0.047 | 4.72 |
| Chet pts O/U 17.5 | success | pass | — | — | — | — | — |
| Fox pts UNDER 14.5 (-105) | success | **under** | **+20.12** | — | A | 0.206 | 5.0 |

†Game bets 1/2/3 are correlated (same Markov simulation). Do not stake all three at face value.

---

## 4. Degradation Decisions

| Decision | Reason | Action |
|----------|--------|--------|
| PRA skipped → rerun | `pra_mean` required; component means not accepted | Fixed input; rerun succeeded |
| Evidence schema fixes | `category` values 'game_tempo'/'defense'/'personnel' invalid | Corrected to 'matchup'/'situational' per registry |
| Prop line caveat applied | BetMGM props API not queried | Lines estimated from web sources; user must confirm |
| No full downgrade to qualitative | Input quality ≥ 0.74 on all runs; no critical missing fields post-fix | Bet cards produced |

---

## 5. LLM Orchestration Choices

| Choice | Rationale |
|--------|-----------|
| Markov backend for game | Possessions-level model appropriate for playoff, slow-pace game |
| 10,000 iterations | Production-quality; sufficient for stable probability estimates |
| SA off_rating estimated at 115.5 | SA defensive data confirmed; offensive rating interpolated from G4 score (103 points) |
| OKC off_rating stepped down to 121.0 from 124.3 season average | Series-specific suppression by SA defense; avoid overconfident OKC offense projection |
| Used `blowout_risk=0.20` | G4 was a blowout (103-82); OKC is home with must-stop pressure; modest risk applied |
| Fox prop set as UNDER 14.5 | n=2 sample (ankle return); mean below line; high-ankle sprain limits explosiveness |
| Wemby reb/blk rejected despite positive series avg | Both sides had <4% absolute edge; rec=pass; thin market |

---

## 6. Rejected Candidates

| Candidate | Engine Edge | Reason |
|-----------|-------------|--------|
| Wemby reb Over 12.5 (-115) | edge_over=-3.81%; edge_under=-0.90% | Proj mean 12.47 — dead-center on line; both sides negative or near-zero |
| Wemby blk Over 2.5 (-120) | edge_over=-1.47%; edge_under=-3.08% | Market -120 implies 54.5%; over_prob only 53.1%; thin hold both sides |
| Chet pts Over 17.5 (-110) | edge_over=-6.91%; edge_under=+2.15% | Proj mean 16.95; weak under edge only (2.15%); below threshold |

---

## 7. Final Recommended Bets

**Hard rule:** Bet IDs 1/2/3 share the same Markov seed. Treat as one bet thesis; do not triple-stake.

| Priority | Bet | Odds | Edge | Proj vs Line | Thesis |
|----------|-----|------|------|--------------|--------|
| 1 | **Under 217.5** | -110 | +22.83% | 212.5 vs 217.5 | SA #1 playoff D; G4=185; pace suppressed |
| 2 | **SA +4.5** | -115 | +23.13% | Spread proj 2.2 OKC | Same simulation; SA covers if Williams limited |
| 3 | **Fox pts UNDER 14.5** | -105 | +20.12% | Proj 12.6 | Ankle-limited; n=2 sample; away game |
| 4 | **SGA pts UNDER 26.5** | -110 | +14.69% | Proj 23.9 | SA schemes disrupt SGA; 12-32 FG in G3+G4 |
| 5 | **Wemby pts UNDER 29.5** | -105 | +5.55% | Proj 28.3 | Positive edge; high variance; confirm line |
| 6 | **Wemby PRA UNDER 46.5** | -105 | +4.60% | Proj 45.2 | Correlated with pts Under; confirm line |
| 7 | **SA ML +140** | +140 | +22.53% | Win prob 43.6% | Correlated with #1/#2; underdog value |

**Not recommended (rec=pass):** Wemby reb Over 12.5, Wemby blk Over 2.5, Chet pts Over 17.5.

---

## 8. Calibration Eligibility

| Trace | Eligible | Slice | Exclusion |
|-------|---------|-------|-----------|
| All 10 traces | ✅ Yes | `playoff` | None |
| Calibration fitter slice | `is_playoff=True, rest_days=2` | — | calibration profile: static_identity (no fit profile resolved) |

**Action:** Run `scripts/fetch_outcomes_nba.py` after game completes. Run `scripts/fetch_outcomes_props.py` for props. Ingest with `scripts/ingest_traces.py --verbose` tomorrow AM.

---

## 9. Bugs / Issues Observed This Session

| ID | Bug | Impact | Action |
|----|-----|--------|--------|
| Schema | EvidenceSignal category enum mismatch: 'game_tempo','defense','personnel' not in registry | Blocked first engine run | Fixed inline; schema corrected |
| Schema | `window='game'` not a valid literal | Validation error | Corrected to 'matchup' |
| Engine | PRA prop requires `pra_mean` not component means | Skipped on first pass | Fixed; rerun succeeded |
| Calibration | `calibration_audit.path='static_identity'` on all game traces | No profile fitted | No active calibration profile for NBA playoff; run fit after outcomes attached |

