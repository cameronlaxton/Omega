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

Use the league prompt instead. Do not run analysis, setup, trace export, ingest,
outcome attachment, or report generation from this file.

If a request mentions props only, choose the matching league prompt and scan the
full league slate. Mark non-prop markets as scanned, pass, or research-only as
appropriate; do not open a separate props session.
