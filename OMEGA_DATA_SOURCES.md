# OMEGA_DATA_SOURCES

How the Omega agent sources live and historical data. The full self-heal loop is specified in [prompts/system_prompt.txt](prompts/system_prompt.txt) Section 6; this document is the per-slot reference card.

## Sourcing Model

The sandbox engine (`omega_lite_standalone.py`) performs **no** network calls.

Pre-decision live data is sourced by the **agent** using cited WebSearch/WebFetch evidence. Every numeric value injected into an engine request must be cited with a source URL and an access timestamp; the agent echoes both in its `Inputs used` section.

Post-decision market tracking is local automation. Cowork may use `OMEGA_ODDS_API_KEY` through `omega.integrations.odds_api` to capture current and historical closing-line snapshots for CLV, replay, and backtest artifacts. The API key stays local in `.env` or the runtime environment and is never copied into prompts, traces, reports, or frontend code.

## Per-Slot Search Recipe

| Missing slot | Search target | Preferred domains |
|---|---|---|
| `odds_over` / `odds_under` / `odds.markets` | Today's prop or game line | `draftkings.com`, `fanduel.com`, `betmgm.com`, `caesars.com`, `espnbet.com`, `hardrock.bet` |
| `player_context.{prop}_mean` | Player season + last-10-game rolling average for that stat | `basketball-reference.com`, `baseball-reference.com`, `pro-football-reference.com`, `hockey-reference.com`, `nba.com`, `mlb.com`, `nhl.com`, `nfl.com`, `espn.com` |
| `player_context.{prop}_std` | Last 5-10 game logs; compute `statistics.stdev(values)` in Python | Same as above. Fallback: `std = 0.25 * mean`; disclose to the user. |
| `home_context.off_rating` / `def_rating` | Team offensive/defensive efficiency rating | `basketball-reference.com`, `pro-football-reference.com`, `mlb.com`, `hockey-reference.com` |
| `home_context.pace` | Team pace / possessions-per-48 | `basketball-reference.com`, `nba.com` |
| `serve_win_pct` / `return_win_pct` | ATP/WTA player profile | `atptour.com`, `wtatennis.com` |
| `strokes_gained_total` | Strokes Gained season totals | `datagolf.com`, `pgatour.com` |
| `win_pct` / `finish_rate` | Fighter record + finish breakdown | `ufc.com`, `tapology.com`, `sherdog.com` |
| `map_win_rate` | Team map win rate | `hltv.org`, `liquipedia.net` |

The sportsbook domains above are scraped only by the agent for pre-decision evidence. Omega's maintained HTTP odds client is the post-decision/historical The Odds API adapter in `omega/integrations/odds_api.py`.

## Source-Credibility Hierarchy

When two sources disagree:

1. Official league sites and `*-reference.com` - highest weight for player and team stats.
2. Major sportsbooks - highest weight for current market lines.
3. ESPN / CBSSports / TheAthletic - secondary stat and news sources.
4. Aggregators such as VegasInsider, OddShark, and Action Network - useful for cross-book consensus but not first-choice.
5. Twitter/X, Reddit, podcast clips - qualitative context only.

If sources within the same tier disagree by <= 5 cents (American odds) or <= 5% (rate stats), use the median and flag the spread. Beyond that, surface both and pick the most recent or most authoritative.

## Freshness Windows

The engine and quality gate continue to consume freshness rules from `omega/core/models.py:FRESHNESS_RULES` for per-slot decay:

- `team_stat`: 24 hours
- `player_stat`: 24 hours
- `player_game_log`: 24 hours
- `odds`: 15 minutes
- `injury`: 2 hours
- `schedule`: 1 hour
- `environment`: 4 hours

When the agent injects a value older than its window, it labels it as `stale` in `Inputs used`.

## Sources That Are Not Omega Data Sources

- `dimers.com`, `rotowire.com`, `covers.com` - third-party analytics; cite at most as supplementary context, never as the authoritative input for a numeric slot.
- Twitter/X, Reddit, sports talk podcasts - qualitative context only.
- "Omega's X collector says..." - there are no general-purpose Omega collectors. Pre-decision values remain WebSearch citations. Post-decision/historical market snapshots may cite `the-odds-api` through the local adapter.

## Code Anchors

- `omega/core/models.py:FRESHNESS_RULES` - per-type freshness windows still consumed by the quality gate.
- `omega/core/simulation/archetypes.py` - `prop_stat_keys` and `required_team_keys` per archetype define which slots the agent must source per request.
- `omega/integrations/odds_api.py` - post-decision and historical odds snapshots for CLV/replay/backtest artifacts.
- `omega_lite_standalone.py` - the sandbox engine. No network code.

The previous code anchors at `omega/evidence/collectors/*` have been deleted as part of retiring the general external-API surface.
