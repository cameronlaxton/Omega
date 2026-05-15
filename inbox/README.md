# inbox/

Drop zone for artifacts produced outside this machine that need to be ingested
into the local Omega DB.

## inbox/traces/

The LLM in a Claude.ai project chat emits a JSON block per system prompt §10 at
the end of every analysis. Save the inner JSON object to
`inbox/traces/{trace_id}.json` and run:

```bash
python scripts/ingest_traces.py
```

The script will:
- Persist each `trace` to `omega_traces.db` via `TraceStore.persist()` (idempotent).
- Persist each `bet_record` (if present) to the `bet_records` table.
- Move successfully ingested files to `inbox/traces/processed/`.
- Move malformed files to `inbox/traces/failed/` with a sibling `.error.txt`.

Re-running is safe: idempotent on `trace_id` and `(trace_id, market, selection_descriptor)`.

## Expected JSON shape

```json
{
  "trace": { "trace_id": "sandbox-...", "model_version": "...", ... },
  "bet_record": null,
  "clv_capture_instructions": { "...": "..." }
}
```

See `prompts/system_prompt.txt` §10 for the exact emission format.
