# Current Handoff - Phase 6h

For product doctrine, canonical source-of-truth rules, and artifact authority, refer to [PROJECT_STATE.md](PROJECT_STATE.md).

Omega now runs through the local MCP server or the canonical core service.

## Runtime

- MCP entry point: `python -m omega.mcp.server`
- Direct Python entry point: `omega.core.contracts.service.analyze`
- Trace envelope model version: `omega-core-phase6h`
- Analyze calls require explicit `session_id` and `bankroll`.
- Live analyze calls should receive a deterministic integer seed derived from `sha256(prompt + date)`.

## Retired

The previous standalone bridge has been removed. Do not rebuild or route through it.

## Key Files

| Path | Purpose |
|---|---|
| `omega/core/contracts/service.py` | Canonical analyze wrapper and deterministic service |
| `omega/mcp/server.py` | Typed MCP tools |
| `scripts/resolve_odds.py` | BetMGM-first current odds resolver |
| `scripts/ingest_traces.py` | Trace export ingestion |
| `scripts/fetch_closing_lines.py` | Closing-line capture |
| `scripts/fetch_outcomes_props.py` | Player-prop outcome attachment |
| `OMEGA_COWORK.md` | Local VM runtime protocol |

## Next Pickup

Continue hardening the MCP-native workflow: improve seed provenance, add richer trace-quality metadata when needed, and keep replay/benchmark boundaries separate.

---

## Known Operational Gap: Scheduled Tasks Write to Ephemeral DB (2026-05-28)

**Problem:** `TraceStore` detects that `omega_traces.db` is on a FUSE/network mount and redirects all SQLite writes to a local sandbox path (`~/.omega/runtime/omega_traces.db`). This is correct behavior — SQLite on FUSE is unreliable (BUG-FUSE-2). But it means every scheduled task run writes to a path that is destroyed when the Linux sandbox session expires. The Windows-side `omega_traces.db` is never updated.

**Root cause:** The scheduled task is running against `C:\repos\Omega` (the network mount), not a local clone. The supported steady-state per OMEGA_COWORK.md §2c is to run from `%USERPROFILE%\.omega\workspace\Omega` (bootstrapped via `cowork_bootstrap.ps1`), where the DB is local and writes are durable.

**Mitigation applied (2026-05-28):** The 13 MLB traces + outcomes from the 2026-05-27 daily loop were exported to `inbox/traces/backfill_20260528/` as ingest-ready JSON. To restore them: move those files to `inbox/traces/` and run `python scripts/ingest_traces.py`. The `_export_meta.outcome_attached` field on each file documents the already-fetched ESPN outcome — skip `fetch_outcomes` after re-ingest to avoid a no-op second pass (idempotent anyway).

**To fix permanently:** Run `scripts/cowork_bootstrap.ps1` on the Windows host to set up the local workspace. Future scheduled tasks must be pointed at that local clone, not `C:\repos\Omega`.

**Also flagged this session:** FUSE mount truncation of four source files (`espn_nba.py`, `espn_mlb.py`, `store.py`, `session_sidecar.py`) — trailing bytes dropped silently at the mount cache boundary. Per OMEGA_COWORK.md §2c, repair path is `python scripts/cowork_preflight.py --repair-from-git` (runs `git checkout HEAD` through bash to bypass the cache). The `Write` tool does NOT fix this — Windows-side writes don't invalidate the Linux mount cache.
