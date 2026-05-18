# OMEGA_DATA_SOURCES

How Omega sources live and historical data. The full no-local self-heal loop is specified in [prompts/system_prompt.txt](prompts/system_prompt.txt) Section 6; this document is the per-slot reference card.

## Sourcing Model

The deterministic core service and MCP analyze tools perform no network calls.

In local Cowork/Codex operation, pre-decision current odds are resolved through The Odds API with BetMGM as the default bookmaker. Use `scripts/resolve_odds.py` or the `omega_resolve_odds` MCP tool. Multi-book requests are reserved for explicit line shopping, consensus checks, or audit/backfill work.

No-local Project/API agents cannot read `.env` or call local tooling. They still use cited WebSearch/WebFetch evidence, or they ask the user to run the local resolver and paste/save the output.

The API key stays local in `.env` or the runtime environment and is never copied into prompts, traces, reports, or frontend code.

## Per-Slot Recipe

| Missing slot | Local Cowork/Codex source | No-local fallback |
|---|---|---|
| `odds_over` / `odds_under` / `odds.markets` | `omega_resolve_odds` / `scripts/resolve_odds.py` using BetMGM | Cited sportsbook WebFetch/WebSearch |
| Line-shopping / consensus odds | `resolve_odds.py --line-shopping` or `all_books=true` | Cited multi-book search |
| Closing line | `scripts/fetch_closing_lines.py` through The Odds API, then `scripts/ingest_closing_lines.py` | WebFetch closing snapshot block |
| Historical odds / replay market artifact | The Odds API historical endpoints through `omega.integrations.odds_api` | Not available without local access |
| `player_context.{prop}_mean` | Official/reference stats or local curated inputs | Same |
| `player_context.{prop}_std` | Last 5-10 game logs; compute `statistics.stdev(values)` in Python | Same |
| `home_context.off_rating` / `def_rating` | Official/reference team stats | Same |
| Injury/schedule/environment | Official league/team/weather sources | Same |

## Odds Defaults

- Default bookmaker: `betmgm`.
- Default game markets: `h2h`, `spreads`, `totals`.
- Player props: event-level The Odds API markets mapped from Omega `prop_type` keys where available.
- If BetMGM does not list an exact market, the resolver returns `status: "unavailable"` unless the caller explicitly requested line shopping or all-book comparison.

## Source-Credibility Hierarchy

When sources disagree:

1. Official league sites and `*-reference.com` for player and team stats.
2. BetMGM via The Odds API for default current market lines.
3. Explicit multi-book The Odds API line-shopping output for price comparison.
4. Major sportsbook pages via WebFetch when local Odds API access is unavailable.
5. ESPN / CBSSports / TheAthletic for secondary stat and news context.
6. Aggregators such as VegasInsider, OddShark, and Action Network for supplementary context only.
7. Twitter/X, Reddit, podcast clips for qualitative context only.

If line-shopping sources disagree, preserve each book's price instead of collapsing to a fabricated consensus. The engine should receive the selected book/line actually used for the analysis.

## Freshness Windows

The engine and quality gate continue to consume freshness rules from `omega/core/models.py:FRESHNESS_RULES`:

- `team_stat`: 24 hours
- `player_stat`: 24 hours
- `player_game_log`: 24 hours
- `odds`: 15 minutes
- `injury`: 2 hours
- `schedule`: 1 hour
- `environment`: 4 hours

When an injected value is older than its window, label it as `stale` in `Inputs used`.

## Sources That Are Not Omega Data Sources

- `dimers.com`, `rotowire.com`, `covers.com`: third-party analytics; cite at most as supplementary context, never as authoritative numeric input.
- Twitter/X, Reddit, sports talk podcasts: qualitative context only.
- LLM-generated odds, estimated leans, or ballpark prices: forbidden.

## Code Anchors

- `omega/integrations/odds_api.py`: The Odds API client.
- `scripts/resolve_odds.py`: BetMGM-default pre-decision odds resolver.
- `omega/mcp/server.py`: `omega_resolve_odds` MCP tool.
- `omega/trace/market_snapshot.py`: line-movement snapshot model.
- `omega/core/simulation/archetypes.py`: supported `prop_type` keys per sport archetype.
- `omega/core/contracts/service.py`: canonical local analyze entry point. No network code.
