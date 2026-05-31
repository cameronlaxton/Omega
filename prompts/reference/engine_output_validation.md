# Engine Output Nullability Check — Canonical Reference

**Canonical source for the post-`analyze()` null-data audit.** The league daily prompts link here
instead of inlining this procedure, so the checks stay identical across NBA / WNBA / MLB.

**Execute immediately after each `analyze()` call returns, before any other processing.**

Evaluate the engine response payload for fields returning NULL, `0.0`, `"undefined"`, empty
collections, or otherwise missing values. This is a structural validation step, not a data-quality
judgment.

## Mandatory checks

1. **Request-level identity fields** (must be echoed):
   - `league`, `player_name` (for props), `home_team`, `away_team`, `game_date`
2. **Result object fields** (engine-owned; must exist if `status != "skipped"`):
   - `model_prob` (or `over_prob`/`under_prob` for player props)
   - `fair_price` / `no_vig_price`
   - `edge_pct`
   - `recommended_units`
   - `confidence_tier`
   - `trace_id`
3. **Input context fields** (your responsibility; must be populated before calling `analyze()`):
   - `game_context.is_playoff` and `game_context.rest_days` (non-null integers) — always.
   - **Player props:** `{prop_type}_mean`, `{prop_type}_std`, `sample_size` (≥ 5 if available).
   - **Game analysis:** `home_context.off_rating`, `def_rating`, and any **sport-specific** rating
     keys named by the league daily prompt (NBA/WNBA: `pace`; MLB: `starter_era`, `park_factor`,
     `weather_wind_mph`). Map the sport-specific keys where material and available.

## Capture strategy

If any of the above are NULL or missing:

- Build a **null_fields** list with clean field paths, e.g.
  `["result.recommended_units", "game_context.rest_days"]`.
- Append a `quality_gate` event with `event_type="quality_gate/null_data_audit"` and
  `notes="Null fields: " + ", ".join(null_fields)`.
- Do **not** include numeric protected values in the event notes; only field names.

## Decision logic

- Result-level fields NULL and `status != "skipped"` → **engine error → downgrade to
  research-only**.
- Input context fields NULL → **your input was incomplete → downgrade to research-only**; log which
  context field(s) were missing.
- `sample_size < 5` for a player prop → **research-only** unless explicitly backfilled from a
  reliable source.
- A sport-specific field (e.g. `park_factor`, `weather_wind_mph`) unavailable where expected →
  **log as missing and downgrade if material**.
- All required fields present → proceed to trace export.

> Downgrade here means **user-facing** output is research-only. The engine already ran and the
> trace still persists with its `sandbox-` trace_id for calibration — see
> [`output_modes.md`](output_modes.md). Do not skip trace export just because the user-facing
> output was downgraded.

## Example null_fields audit event

```json
{
  "ts": "2026-05-28T20:15:00Z",
  "event_type": "quality_gate/null_data_audit",
  "step": "engine_output_validation",
  "status": "warn",
  "notes": "Null fields: ['game_context.rest_days']. Downgraded to research-only.",
  "inputs": [],
  "outputs": [],
  "trace_ids": ["sandbox-xxxxx"]
}
```
