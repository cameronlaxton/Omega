# inbox/

Drop zone for artifacts produced outside this machine that need to be ingested into the local Omega DB.

## inbox/traces/

`inbox/traces/` is an import and transfer surface, not canonical storage.
After every local MCP or direct-core analysis, save a JSON file to
`inbox/traces/{trace_id}.json` and run:

```bash
python scripts/ingest_traces.py
```

The script will:

- Persist each `trace` to `omega_traces.db` through `TraceStore.persist()` idempotently.
- Persist each `bet_record` when present.
- Move successfully ingested files to `inbox/traces/processed/`.
- Move malformed files to `inbox/traces/failed/` with a sibling `.error.txt`.

Re-running is safe: idempotent on `trace_id` and `(trace_id, market, selection_descriptor)`.

Processed, failed, and backfill exports are forensic history only after ingest.
They must not be searched as current architecture truth; the canonical numeric
state is `omega_traces.db`. Failed JSON stays in `failed/` with its error
sidecar for diagnostic review. Do not delete failed or malformed trace exports
as part of routine cleanup.

## Expected JSON Shape

```json
{
  "trace": { "trace_id": "sandbox-...", "model_version": "omega-core-phase6h" },
  "bet_record": null
}
```

If a bet was actually taken, `bet_record.selection_descriptor` is required. The retired closing-line instruction block is not emitted in Phase 6h.

## inbox/sessions/

Session sidecars are strict, typed JSON summaries. Add new top-level fields only
by updating `omega.trace.session_sidecar.SessionSidecar`; ad hoc keys are
rejected by Pydantic. Current optional actionability fields are:

- `league`
- `window`
- `effective_db_path`
- `runtime_db_status`
- `pipeline_status`
- `next_required_action`

The sibling `<session_id>.events.jsonl` file is a recovery mirror only and is
not canonical.
