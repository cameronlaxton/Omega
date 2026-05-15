"""omega.integrations — thin clients for external data sources.

Each module here is a narrow adapter: HTTP fetch + minimal parsing + alias
resolution. No business logic, no calibration math, no persistence — those
belong in `omega/trace/`, `omega/core/`, and `scripts/`.

Existing modules:
- espn_nba: ESPN public scoreboard (no auth) for NBA final scores
- odds_api: the-odds-api free-tier client for closing-line snapshots
"""
