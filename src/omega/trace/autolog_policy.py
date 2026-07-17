"""Single autolog policy for the engine_auto bet-ledger dual-write.

Both persistence backends (SQLite ``TraceStore`` and ``PostgresRepository``)
MUST route their autolog decision through :func:`engine_auto_autolog_decision`
so they cannot drift (tests enforce parity).

Policy (Matchup Intelligence Phase 0 — fail closed):

- ``engine_auto_ledger_mode`` on the trace payload is the per-run authority.
  Missing or unrecognized values coerce to ``disabled`` — no engine_auto row.
- ``shadow`` mode additionally requires the operator environment gate
  ``OMEGA_ENABLE_ENGINE_SHADOW=1``. A trace stamped ``shadow`` without the env
  gate (or vice versa) writes nothing.
- The legacy ``OMEGA_BET_LEDGER_AUTOLOG`` variable is a KILL SWITCH ONLY: an
  explicit falsy value still denies the write, but no value of it can enable
  one. (Its historical default-on enabling behavior is retired.)
- Scoped suppression (``TraceStore.autolog_suppressed()``) wins over
  everything — historical replay must never autolog.

Explicit user-confirmed wager recording (``record_ledger_bet`` and the MCP
``omega_record_flat_bet`` path) does not pass through this policy and is
unaffected.
"""

from __future__ import annotations

import os
from typing import Any

from omega.core.contracts.schemas import coerce_engine_auto_ledger_mode

ENV_ENGINE_SHADOW = "OMEGA_ENABLE_ENGINE_SHADOW"
ENV_LEGACY_AUTOLOG_KILL_SWITCH = "OMEGA_BET_LEDGER_AUTOLOG"

_FALSY = {"0", "false", "False", "no", "off"}


def engine_auto_autolog_decision(
    trace: dict[str, Any], *, suppressed: bool = False
) -> tuple[bool, str]:
    """Decide whether an engine_auto ledger row may be written for this trace.

    Returns ``(allowed, reason)``; the reason string is stable vocabulary for
    logs and tests: ``scoped_suppression``, ``legacy_kill_switch``,
    ``engine_auto_ledger_mode_disabled``, ``shadow_env_not_enabled``,
    ``shadow_enabled``.
    """
    if suppressed:
        return False, "scoped_suppression"
    legacy = os.environ.get(ENV_LEGACY_AUTOLOG_KILL_SWITCH)
    if legacy is not None and legacy in _FALSY:
        return False, "legacy_kill_switch"
    mode = coerce_engine_auto_ledger_mode(trace.get("engine_auto_ledger_mode"))
    if mode != "shadow":
        return False, "engine_auto_ledger_mode_disabled"
    if os.environ.get(ENV_ENGINE_SHADOW) != "1":
        return False, "shadow_env_not_enabled"
    return True, "shadow_enabled"
