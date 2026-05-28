# Deprecated: Standalone Props Daily Session

Do not use this as a live operating prompt for NBA, WNBA, or MLB daily betting
work.

Props and game bets must be considered in the same breadth of a league-specific
request because they share the same injury map, rest context, matchup context,
line movement, source provenance, session sidecar, and null-data audit. The
0528 QA failure showed that separating props into a different prompt lets the
agent miss cross-market context and treat estimated/milestone lines too loosely.

Use the league prompt instead:
- [NBA daily league session](nba_daily.md)
- [WNBA daily league session](wnba_daily.md)
- [MLB daily league session](mlb_daily.md)

This file remains only as a routing guard for older references. If a user asks
for "props only", still start from the league prompt and simply mark game
markets as scanned/pass/research-only rather than opening a separate props
session.

For prop type keys, use [prop_stat_keys.md](../reference/prop_stat_keys.md).
