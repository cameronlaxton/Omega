# Markov Backend — Approved Evidence Signal Vocabulary (Canonical Reference)

**Canonical source for the Markov `signal_type` vocabulary and the ±15% cap.** `OMEGA_COWORK.md`,
`prompts/system_prompt.txt`, and the league daily prompts link here instead of restating the table.

When calling `omega_analyze_game` with `simulation_backend="markov_state"`, **only these 8
`signal_type` values affect the possession-level transition matrix.** All other signal types are
audited and scored but have no Markov effect (silently ignored by the modifier engine). Use the
exact string keys below:

| signal_type | effect | direction required? |
|---|---|---|
| `pace_up` | +6% game pace | no |
| `pace_down` | -8% game pace | no |
| `rest_advantage` | +4% scoring rate for rested team | yes (`home`/`away`) |
| `b2b_fatigue` | -6% scoring rate for fatigued team | yes (`home`/`away`) |
| `def_matchup_weak` | +5% offense vs. weak defender | yes (`home`/`away`) |
| `def_matchup_strong` | -5% offense vs. strong defender | yes (`home`/`away`) |
| `usage_role_change` | -7% team rate when key player restricted/elevated | yes (`home`/`away`) |
| `blowout_risk` | -2% momentum acceleration; suppresses variance | no |

## Rules

- **Cumulative cap:** no single modifier attribute shifts by more than **±15%**, regardless of
  stacked signals.
- Do **not** pre-adjust `home_context`/`away_context` ratings by hand to bake in these effects — the
  engine applies them from the signal.
- Do **not** emit the same logical signal on both `plane="game"` and `plane="player"` in one
  request. The service suppresses player-plane duplicates when a matching game-plane signal is
  present.
- Call `omega_markov_evidence_guide` (MCP prompt) for the full modifier table with scalar values.
- **Evidence routing:** Markov transition modifiers are the Markov evidence path. Handler-based
  shadow/live mode applies to fast-score game and player-prop adjustments.

## Example (evidence on an analyze() request)

```python
analyze({
    "player_name": "Donovan Mitchell", "league": "NBA", "prop_type": "pts",
    "line": 26.5, "odds_over": -110, "odds_under": -110,
    "player_context": {"pts_mean": 28.4, "pts_std": 6.9},
    "home_team": "Cleveland Cavaliers", "away_team": "Detroit Pistons",
    "game_date": "2026-05-22",
    "game_context": {"is_playoff": True, "rest_days": 2},
    "evidence": [
        {"signal_type": "series_avg", "category": "player_form", "plane": "player",
         "value": 30.6, "source": "nba.com", "confidence": 0.9,
         "window": "series", "direction": "over", "stat_key": "pts"},
        {"signal_type": "last_game_outlier", "category": "player_form", "plane": "player",
         "value": True, "source": "agent_reasoning", "confidence": 0.6,
         "window": "last_3", "stat_key": "pts"},
    ],
}, session_id=session_id, bankroll=bankroll)
```
