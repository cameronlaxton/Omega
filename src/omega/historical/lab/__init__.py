"""Historical Validation Lab — orchestration + ledger + report over the existing engine.

This package is a *thin* layer. It does not own a fitter, registry, grading, or
staking system — every modeling primitive is reused from elsewhere:

* replay            → ``omega.ops.replay_history`` / ``omega.historical.replay``
* fitting           → ``omega.core.calibration.fitter`` / ``omega.ops.fit_calibration``
* registry/promote  → ``omega.core.calibration.registry`` (the single fail-closed gate)
* metrics           → ``omega.historical.metrics`` (Brier/ECE/log-loss, ROI/CLV)
* walk-forward       → ``omega.historical.walk_forward``
* parity            → ``omega.ops.backtest_parity`` / ``omega.ops.historical_live_parity``

What this package adds is the glue that nothing else provides: a single
orchestrated run, an Attempted Variants ledger (every fit attempt, including the
rejected ones, for winner's-curse accounting), a holdout-sealing guard across the
variant grid, git provenance, and a stitched lab report.
"""

from __future__ import annotations

from omega.historical.lab.schemas import (
    AttemptedVariant,
    AttemptedVariantLedger,
    HistoricalLabRun,
    PromotionEvidenceBundle,
    Window,
    WinnersCurse,
    assert_consistent,
    windows_overlap,
)

__all__ = [
    "AttemptedVariant",
    "AttemptedVariantLedger",
    "HistoricalLabRun",
    "PromotionEvidenceBundle",
    "Window",
    "WinnersCurse",
    "assert_consistent",
    "windows_overlap",
]
