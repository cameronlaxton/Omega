"""Historical replay + walk-forward backtest module.

Repo-native system that ingests local historical datasets, reconstructs
pre-game-only feature snapshots, replays historical slates through Omega's normal
simulation/analyze path, persists replayed predictions as normal TraceStore
traces, attaches outcomes and closing lines, runs chronological walk-forward
calibration, and reports raw-vs-calibrated probability performance and optional
betting performance.

Design invariants (see plan + AGENTS.md):
- The existing ``TraceStore`` is the only trace sink (pointed at an *isolated*
  backtest DB file so production calibration stays clean).
- No parallel calibration fitter or registry: walk-forward uses
  ``omega.core.calibration`` directly and freezes fold profiles in memory.
- Closing lines are CLV-only and never influence bet selection.
- No post-game data may enter a pre-game feature snapshot; unsafe rows fail
  closed or are skipped with an explicit reason.
- Replay is deterministic from dataset manifest + replay config + code version.
"""

from __future__ import annotations
