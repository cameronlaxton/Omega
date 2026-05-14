# OMEGA_DATA_SOURCES

How the Omega agent sources live data now that the FastAPI service, all external-API collectors, and `[CLAUDE-ESTIMATED]` fallbacks have been retired. The full self-heal loop is specified in [prompts/system_prompt.txt](prompts/system_prompt.txt) §6; this document is the per-slot reference card.

## Sourcing model

The sandbox engine (`omega_lite_standalone.py`) performs **no** network calls. Live data is sourced by the **agent** using its built-in WebSearch tool — there are no API keys, no `ODDS_API_KEY`, no Anthropic/Perplexity backends, no Redis. Every numeric value injected into a request must be cited with a source URL and an access timestamp; the agent echoes both in its `Inputs used` section.

If a slot cannot be sourced after 3 distinct searches, the agent declines the Bet Card and produces a Research Report (or a "market not available" refusal). See [prompts/system_prompt.txt](prompts/system_prompt.txt) §6.3.

## Per-slot search recipe

| Missing slot                                  | Search target                                                                                  | Preferred domains                                                                                       |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `odds_over` / `odds_under` / `odds.markets`   | Today's prop or game line                                                                      | `draftkings.com`, `fanduel.com`, `betmgm.com`, `caesars.com`, `espnbet.com`, `hardrock.bet`             |
| `player_context.{prop}_mean`                  | Player season + last-10-game rolling average for that stat                                     | `basketball-reference.com`, `baseball-reference.com`, `pro-football-reference.com`, `hockey-reference.com`, `nba.com`, `mlb.com`, `nhl.com`, `nfl.com`, `espn.com` |
| `player_context.{prop}_std`                   | Last 5–10 game logs; compute `statistics.stdev(values)` in Python                              | Same as above. Fallback: `std = 0.25 * mean` (coefficient-of-variation default — disclose to the user). |
| `home_context.off_rating` / `def_rating`      | Team offensive/defensive efficiency rating                                                     | `basketball-reference.com`, `pro-football-reference.com`, `mlb.com`, `hockey-reference.com`             |
| `home_context.pace`                           | Team pace / possessions-per-48                                                                 | `basketball-reference.com`, `nba.com`                                                                   |
| `serve_win_pct` / `return_win_pct` (tennis)   | ATP/WTA player profile                                                                         | `atptour.com`, `wtatennis.com`                                                                          |
| `strokes_gained_total` (golf)                 | Strokes Gained season totals                                                                   | `datagolf.com`, `pgatour.com`                                                                           |
| `win_pct` / `finish_rate` (fighting)          | Fighter record + finish breakdown                                                              | `ufc.com`, `tapology.com`, `sherdog.com`                                                                |
| `map_win_rate` (esports)                      | Team map win rate                                                                              | `hltv.org` (CS2), `liquipedia.net`                                                                      |

The sportsbook domains above are scraped only via the agent's WebSearch tool — Omega does not maintain HTTP clients or API integrations against them.

## Source-credibility hierarchy

When two sources disagree:

1. **Official league sites and `*-reference.com`** — highest weight for player and team stats.
2. **Major sportsbooks (DraftKings / FanDuel / BetMGM / Caesars)** — highest weight for current market lines.
3. **ESPN / CBSSports / TheAthletic** — secondary stat and news sources.
4. **Aggregators (vegasinsider.com / oddshark.com / actionnetwork.com)** — useful for cross-book consensus but not first-choice.
5. **Twitter/X / Reddit / podcast clips** — never an Omega-cited source for a numeric input. Use only as qualitative context.

If sources within the same tier disagree by ≤ 5 cents (American odds) or ≤ 5% (rate stats), use the median and flag the spread. Beyond that, the agent surfaces both and picks the most recent or most authoritative.

## Freshness windows

The engine and quality gate continue to consume freshness rules from `omega/core/models.py:FRESHNESS_RULES` for per-slot decay (still used by the quality gate's `aggregate_quality` calculation if facts include `freshness_max`). The default windows:

- `team_stat`: 24 hours
- `player_stat`: 24 hours
- `player_game_log`: 24 hours
- `odds`: 15 minutes
- `injury`: 2 hours
- `schedule`: 1 hour
- `environment`: 4 hours

When the agent injects a value older than its window, it labels it as `stale` in `Inputs used` and the user can decide to re-fetch.

## Sources that are NOT Omega data sources

- `dimers.com`, `rotowire.com`, `covers.com` — third-party analytics; cite at most as supplementary context, never as the authoritative input for a numeric slot.
- Twitter/X, Reddit, sports talk podcasts — qualitative context only.
- "Omega's X collector says..." — there are no Omega collectors. Every external value is a WebSearch citation, sourced and timestamped by the agent.

## Code anchors

- `omega/core/models.py:FRESHNESS_RULES` — per-type freshness windows still consumed by the quality gate.
- `omega/core/simulation/archetypes.py` — `prop_stat_keys` and `required_team_keys` per archetype define which slots the agent must source per request.
- `omega_lite_standalone.py` — the sandbox engine. No network code.

The previous code anchors at `omega/evidence/collectors/*` have been deleted as part of retiring the external-API surface.
