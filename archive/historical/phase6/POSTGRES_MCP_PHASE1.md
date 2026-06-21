> [!NOTE]
> This document is from a legacy phase that has been implemented and merged to `main`. It is retained here for historical reference.

# Postgres + MCP HTTP Phase 1

Phase 1 adds opt-in Postgres persistence and browser-reachable MCP transports
without changing Omega's deterministic model behavior.

## Scope

- SQLite remains the default backend.
- Setting `DATABASE_URL=postgresql+psycopg://...` routes `TraceStore` public APIs
  to the Postgres repository.
- `python -m omega.mcp.server` remains stdio for local MCP clients. The stdio
  entrypoint **ignores an inherited `DATABASE_URL`** unless
  `OMEGA_MCP_ALLOW_DB_BACKEND=1` is also set, so an exported Postgres URL can never
  silently flip Antigravity to Postgres. The HTTP server, CLI tools, the migration
  tool, and tests still honor `DATABASE_URL`.
- `python -m omega.mcp.http` and `omega-mcp-http` expose the same FastMCP
  registry over `/sse` and `/mcp`, bound to `127.0.0.1` by default.
- Portfolio summaries keep all existing keys and add artifact-friendly fields.

## Setup

```powershell
python -m pip install -e ".[postgres,mcp]"
docker compose up -d postgres
$env:DATABASE_URL = "postgresql+psycopg://omega:omega@localhost:5432/omega"
alembic upgrade head
python tools/migrate_sqlite_to_postgres.py --dry-run
```

`tools/migrate_sqlite_to_postgres.py` is idempotent on real runs and preserves
trace IDs, ledger IDs, provenance, and V14 row identity.

## Running the live Postgres test suite

The Postgres tests are skipped unless **both** of these are set, and they
**reset the `public` schema** (`DROP SCHEMA ... CASCADE`) on every run:

```powershell
$env:OMEGA_TEST_DATABASE_URL = "postgresql+psycopg://omega:omega@localhost:5432/omega"
$env:OMEGA_TEST_DB_ALLOW_DESTROY = "1"
pytest tests/trace/test_postgres_repository.py -q
```

> **⚠️ `OMEGA_TEST_DATABASE_URL` MUST point at a disposable, throwaway database.**
> The fixture drops and recreates the schema; never aim it at a developer,
> staging, or production database. `OMEGA_TEST_DB_ALLOW_DESTROY=1` is the required
> explicit confirmation — without it the destructive fixture refuses to run. Tests
> use `OMEGA_TEST_DATABASE_URL` only and never read the runtime `DATABASE_URL`.

## Deferred To Phase 2

These raw-SQL maintenance paths are guarded as SQLite-only in Phase 1:

- `omega-backfill-bets`
- `omega-backfill-closing-lines`
- `omega-fetch-closing-lines`
- `omega-ingest-closing-lines`
- `omega-backfill-trace-quality`
- `omega-backfill-evidence-signals`
- `omega-report-calibration`
- `omega.strategy.anchor.tracker.AnchorBetTracker`

Phase 2 should port or retire those raw `.conn` paths before flipping the
default backend to Postgres.

## Known Limitations

- `unrealized_pnl` is always `0.0`; Omega does not mark open wagers to market.
- JSON blobs and timestamps remain text for SQLite/Postgres round-trip parity.
- Alembic is the canonical Postgres schema path. `bootstrap_create_all()` is
  guarded by `OMEGA_DB_DEV_BOOTSTRAP=1` and is for scratch databases only.

