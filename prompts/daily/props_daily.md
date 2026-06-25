# Deprecated: Standalone Props Daily Session

```yaml
status: deprecated_redirect
runtime_allowed: false
reason: player props are league-scoped markets, not a separate top-level prompt
canonical_prompts:
  NBA: nba_daily.md
  WNBA: wnba_daily.md
  MLB: mlb_daily.md
```

For a full daily sweep across all leagues and sports, use
[`daily_all_sports.md`](daily_all_sports.md).

For deep-dive prop + game analysis within a single league, use the league prompt.
Mark non-prop markets as scanned, pass, or research-only as appropriate; do not
open a separate props session.
