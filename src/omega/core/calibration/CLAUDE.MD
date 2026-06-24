# omega/core/calibration/CLAUDE.md

## Purpose

This package owns probability calibration policy and learning.

It does not own orchestration, persistence, or backtest execution orchestration.

## Phase 6 mission in this package

Add:
- versioned calibration profiles
- fitting from historical outcomes
- profile selection policy
- evaluation and promotion criteria

Do not add:
- direct database persistence concerns unless profile storage is local and minimal
- orchestrator-specific branching
- alternative duplicate calibration call sites

## Calibration philosophy

Calibration is a deterministic correction layer between raw simulated probabilities and downstream edge/stake logic.

Goals:
- reduce overconfidence
- improve observed probability realism
- preserve reproducibility
- allow controlled promotion of better profiles

## Policy rules

Calibration must be policy-driven.

Selection should be based on:
- league
- market type
- engine version
- artifact schema version

Do not scatter hard-coded defaults across service and backtest code.

There should be one shared path for choosing a profile.

## Training data rules

Allowed training inputs:
- frozen historical artifacts
- attached outcomes
- evaluation splits derived from those artifacts

Disallowed training inputs:
- live mutable requests
- replay-only evidence bundles
- post-hoc hand-edited probability tables without provenance

## Profile rules

Each calibration profile must be versioned and attributable.

Minimum metadata:
- profile id
- version
- method
- training window
- sample size
- metrics
- status
- created timestamp

## Method rules

Supported methods may include:
- none
- shrinkage
- cap
- combined
- isotonic
- market_aware — blends the model probability toward the closing-line implied
  probability: `(1 - w)·model + w·market`. The deference `w` is fitted by
  `CalibrationFitter.fit_market_aware` (grid search; a positive weight is adopted
  only on strict held-out Brier improvement over the pure-model baseline) and
  scaled at application time by `liquidity_deference_factor(league, market)` — a
  coarse proxy (market type + per-league liquidity tier) since real limit sizes
  are not captured. Requires `market_prob` at apply time; absent it, the method is
  identity (fail-safe). `params = {"market_weight": w}`. (issue #28 WS4)

Do not introduce more methods unless there is clear evidence they improve held-out calibration.

## Promotion rules

A candidate calibration profile should not be promoted merely because it improves one metric on one sample.

Promotion should require:
- improvement on held-out calibration quality
- no material degradation to broader benchmark outcomes
- explicit comparison against incumbent

## Drift rules

Backtest and production must use the same calibration selection logic.

If one path uses profile lookup and another path uses hard-coded defaults, that is a bug.

## File ownership

Likely touched files:
- `omega/core/calibration/probability.py`
- new profile, learner, and registry files under this package
- shared selection utilities consumed by service and backtest paths

## Test expectations

Must add tests for:
- fit reproducibility
- profile serialization
- profile selection by policy
- parity between service and backtest application
- incumbent versus candidate comparison