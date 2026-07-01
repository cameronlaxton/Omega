# Trace Query Cookbook — typed read-only inspection

**Purpose.** When you want to *inspect* trace history — "what did we analyze," "how
many graded traces this week," "show me one full trace" — use `TraceStore`'s typed
methods instead of opening `var/omega_traces.db` with raw `sqlite3.connect` and
guessing the schema. This doc exists because audits found agents hand-rolling raw
SQL for read-only diagnosis (which is low-risk but schema-fragile) and, worse,
occasionally for **writes** (which is not allowed — see the hard rule at the end).

This is the Python-API companion to [`docs/LLM_MCP_INTERFACE.md`](LLM_MCP_INTERFACE.md)
(the MCP tool surface) and [`OMEGA_RUNTIME.md`](../OMEGA_RUNTIME.md) (the ownership
boundary). It is **read-only inspection** only.

## Open a read-only handle

```python
from omega.trace.store import TraceStore

store = TraceStore(read_only=True)   # mode=ro + PRAGMA query_only=ON at the DB layer
print(store.db_path, store.db_path_source)   # which DB, and how it was resolved
```

`read_only=True` enforces no-mutation at the SQLite layer, so an accidental write
raises rather than corrupting the ledger. Use it for every inspection script.
`db_path_source` is one of `requested` / `env_override` / `auto_redirect_network_fs`
/ `default` / `database_url` — the same value now recorded in session sidecars'
`runtime_db_status`.

## Fetch one full trace by ID

```python
trace = store.get_trace("nba-20260701-lal-bos-a1b2")   # -> dict | None
if trace is None:
    print("no such trace")
else:
    print(trace["kind"], trace["league"], trace.get("trace_quality", {}).get("aggregate_quality"))
```

## Query recent traces with filters

`query_traces()` is the workhorse. All filters are optional; `limit` defaults to 100.

```python
# Last 20 NBA traces (any grading state)
recent = store.query_traces(league="NBA", limit=20)

# Only graded traces in a date window (has_outcome=True == a game or prop outcome attached)
graded = store.query_traces(
    league="MLB",
    start="2026-06-24T00:00:00Z",
    end="2026-07-01T00:00:00Z",
    has_outcome=True,
)

# Only ungraded (awaiting outcomes)
pending = store.query_traces(has_outcome=False, limit=200)

# Only calibration-eligible traces (respects the trace_quality gate)
eligible = store.query_traces(league="NBA", calibration_eligible_only=True)

# Count by league for a date range, without pulling full rows into your own SQL:
from collections import Counter
rows = store.query_traces(start="2026-06-24T00:00:00Z", end="2026-07-01T00:00:00Z", limit=1000)
print(Counter(r.get("league") for r in rows))
```

Signature (see `src/omega/trace/store.py`):

```python
query_traces(
    league=None, start=None, end=None,
    has_outcome=None,             # True=graded, False=ungraded, None=all
    execution_mode=None,          # e.g. "native_sim"
    limit=100,
    calibration_eligible_only=False,
) -> list[dict]
```

## Per-session trace counts

```python
# Aggregate counts grouped by session_id (NULL session_ids excluded)
for s in store.get_session_summary(league="NBA", limit=50):
    print(s["session_id"], s.get("trace_count"))
```

## Escape hatch: genuinely ad-hoc read-only queries

If your question isn't covered by a typed method (e.g. a one-off "which traces have
a NULL `model_prob` in `predictions`"), a **read-only** SQLite connection is an
acceptable last resort for diagnosis — the schema is documented at the top of
[`OMEGA_RUNTIME.md`](../OMEGA_RUNTIME.md). Open it read-only so you cannot mutate:

```python
import sqlite3
from omega.trace.store import TraceStore

# Reuse the store's resolved path so you never guess where the DB lives.
db_path = TraceStore(read_only=True).db_path
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT trace_id, league FROM traces ORDER BY timestamp DESC LIMIT 5").fetchall()
```

Prefer a typed method when one fits; the escape hatch is for the long tail.

## Hard rule: never hand-write mutations

Do **not** `INSERT` / `UPDATE` / `DELETE` against the trace DB, and do **not**
hand-set quality fields like `aggregate_quality`. Every state change flows through
an authoritative typed path so protected numeric values stay engine-owned and
idempotent:

| To do this | Use | Never |
|---|---|---|
| Persist a trace | `omega_run_batch` / `omega-run-batch` / `omega-run-analyze` → `omega-ingest-traces` | raw `INSERT INTO traces` |
| Attach an outcome | `omega_trace_attach_outcome` / `omega-fetch-outcomes-*` | raw `INSERT INTO outcomes` |
| Void a prop (DNP) | `omega_trace_void_prop` | manual quality patch |
| Settle bets | `omega_settle_bets` / `omega-settle` | raw `UPDATE bet_ledger` |
| Open / append a session sidecar | `create_sidecar` / `append_audit_events` | hand-edit the JSON |

See [`AGENTS.md`](../AGENTS.md) and [`OMEGA_RUNTIME.md`](../OMEGA_RUNTIME.md) for the
full ownership boundary.
