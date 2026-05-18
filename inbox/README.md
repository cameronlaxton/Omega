# inbox/

Drop zone for artifacts produced outside this machine that need to be ingested into the local Omega DB.

## inbox/traces/

After every local MCP or direct-core analysis, save a JSON file to `inbox/traces/{trace_id}.json` and run:

```bash
python scripts/ingest_traces.py
```

The script will:

- Persist each `trace` to `omega_traces.db` through `TraceStore.persist()` idempotently.
- Persist each `bet_record` when present.
- Move successfully ingested files to `inbox/traces/processed/`.
- Move malformed files to `inbox/traces/failed/` with a sibling `.error.txt`.

Re-running is safe: idempotent on `trace_id` and `(trace_id, market, selection_descriptor)`.

## Expected JSON Shape

```json
{
  "trace": { "trace_id": "sandbox-...", "model_version": "omega-core-phase6h" },
  "bet_record": null
}
```

If a bet was actually taken, `bet_record.selection_descriptor` is required. The retired closing-line instruction block is not emitted in Phase 6h.
