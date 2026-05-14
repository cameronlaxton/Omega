# OMEGA_DATA_SOURCES

Authoritative list of what Omega's evidence layer actually integrates with. If a source is not on this list, Omega does **not** have a structured collector for it — you may cite it as raw context only, never as an Omega data source.

## The 7 evidence collectors

Each implements the `Collector` protocol in `omega/evidence/collectors/base.py`. Trust tier 1 is highest (direct structured API); tier 3 is lowest (LLM-extracted from web search). Freshness rules from `FRESHNESS_RULES` in `omega/core/models.py`.

| # | Collector | Source | Evidence types | Leagues | Trust | Auth | Status |
|---|---|---|---|---|---|---|---|
| 1 | **EspnCollector** | `site.api.espn.com` public API | `schedule`, `team_stat` | NBA, NFL, MLB, NHL, NCAAB, NCAAF, WNBA, MLS | 1 | None | Implemented |
| 2 | **OddsApiCollector** | The Odds API v4 (`api.the-odds-api.com`) | `odds` (moneyline, spreads, totals; consensus across books) | NBA, NFL, MLB, NHL, NCAAB, NCAAF, WNBA, MLS, EPL, UFC | 1 | `ODDS_API_KEY` env | Implemented |
| 3 | **TeamFormCollector** | ESPN standings | `team_stat` | NBA, NFL, MLB, NHL, NCAAB, NCAAF, WNBA | 2 | None | Implemented |
| 4 | **ContextCollector** | ESPN game details (venue, weather, rest) | `environment` | NBA, NFL, MLB, NHL, NCAAB, NCAAF, WNBA, MLS | 2 | None | **Stub** — returns None; future iteration |
| 5 | **InjuryCollector** | Web search via FallbackSearchCollector | `injury` | NBA, NFL, MLB, NHL, NCAAB, NCAAF, WNBA | 3 | None | **Stub** — returns None; pipeline falls through to FallbackSearchCollector |
| 6 | **NewsSignalCollector** | Web search via FallbackSearchCollector | `news_signal` | NBA, NFL, MLB, NHL, NCAAB, NCAAF, WNBA, MLS, EPL, UFC, ATP, PGA | 3 | None | **Stub** — returns None; pipeline falls through to FallbackSearchCollector |
| 7 | **FallbackSearchCollector** | Perplexity Sonar (primary) or Anthropic web_search (fallback) | All types — last resort | All supported leagues | 3 (capped) | `PERPLEXITY_API_KEY` or `ANTHROPIC_API_KEY` | Implemented; confidence capped at 0.75, dropped to 0.30 if <2 numeric values extracted |

## Freshness windows by data type

Defined in `FRESHNESS_RULES`:

- `team_stat`: 24 hours
- `player_stat`: 24 hours
- `player_game_log`: 24 hours
- `odds`: 15 minutes
- `injury`: 2 hours
- `schedule`: 1 hour
- `environment`: 4 hours

When the LLM consumer is asked for stale data, surface the freshness window so the user knows the cache horizon.

## Web search domain priority

`FallbackSearchCollector` prioritizes these domains when prompting Perplexity / web_search:

`espn.com`, `basketball-reference.com`, `baseball-reference.com`, `pro-football-reference.com`, `hockey-reference.com`, `covers.com`, `vegasinsider.com`, `actionnetwork.com`, `oddshark.com`, `rotowire.com`, `cbssports.com`, `nba.com`, `nfl.com`, `mlb.com`, `nhl.com`.

**Important distinction:** these are *web-search citation targets*, not Omega data collectors. A citation to `rotowire.com` from a Perplexity result is fine — it travels through `FallbackSearchCollector` at trust tier 3. Claiming "Omega's rotowire collector says X" is false; there is no such collector.

## NOT INTEGRATED — do not cite these as Omega data sources

- **dimers.com** — third-party analytics. Not a collector.
- **betmgm.com / DraftKings / FanDuel / Caesars / Pinnacle** as data sources — Omega does not scrape book sites. Use `OddsApiCollector` for normalized odds across many books. The user may paste market lines from a book and you may treat the *book* as the market-quote source, but never as an Omega collector.
- **Twitter/X / Reddit / sports talk podcasts** — not collectors; only valid as raw context the user has manually supplied.
- **`covers.com` / `rotowire.com` as analytic authorities** — they appear in the web-search priority list as citation targets but are tier-3 sources, not Omega collectors.

## API key checklist

For a full-quality Omega run you need:

- `ODDS_API_KEY` — without this, no live market odds; the caller must pass odds in the request payload.
- `PERPLEXITY_API_KEY` or `ANTHROPIC_API_KEY` — without one, `FallbackSearchCollector` returns nothing and any slot it would have served goes unfilled, which trips the quality gate.

When a Claude.ai Project user reports "Omega didn't have any odds for me," the most common cause is a missing `ODDS_API_KEY` on the run machine, not a bug in Omega.

## Source code anchors

- `omega/evidence/collectors/base.py` — `Collector` protocol, `CollectorResult`.
- `omega/evidence/collectors/espn.py` — public ESPN endpoints and `EspnCollector`.
- `omega/evidence/collectors/odds_api.py` — `OddsApiCollector` and consensus extractor.
- `omega/evidence/collectors/search.py` — `FallbackSearchCollector`, Perplexity + Anthropic backends.
- `omega/evidence/collectors/{team_form,context,injury,news_signal}.py` — current implementations.
- `omega/core/models.py:FRESHNESS_RULES` — per-type freshness windows.
