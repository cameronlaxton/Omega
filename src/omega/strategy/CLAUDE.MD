# omega/strategy/CLAUDE.md

## Purpose

This package owns:
- strategy models
- backtest engine
- frozen historical artifacts
- historical grading
- strategy versioning and promotion linkage

This package is the primary home of the quant evaluation plane.

## Phase 6 mission in this package

Add:
- frozen artifact models
- artifact-driven backtest execution
- trace-to-backtest linkage
- outcome grading improvements where needed

Do not add:
- live provider access
- orchestrator-owned reasoning logic
- calibration fitting internals

## Quant plane rules

The quant plane is the benchmark source of truth for:
- forecast quality
- calibration quality
- EV quality
- staking quality

The quant plane must remain:
- deterministic
- batchable
- outcome-blind until grading
- network-free

## Frozen artifact rules

Backtests should consume typed frozen artifacts, not loose ad hoc dicts.

Artifacts must include:
- stable event identity
- normalized team contexts
- exact odds snapshot
- calibration policy reference
- deterministic seed
- source trace linkage
- schema version

Do not:
- rebuild artifacts differently in multiple places
- mix post-outcome information into pre-decision artifacts
- let mutable caches serve as historical truth

## Drift control rules

The backtest path must share with production:
- calibration selection policy
- edge logic
- staking logic
- grading assumptions where applicable

If production and backtest differ, document why explicitly.

Do not introduce a second implementation of:
- implied probability
- edge percentage
- Kelly or stake sizing
- calibration application policy

## Historical validity rules

A backtest record is valid only if:
- the odds reflect the decision-time snapshot
- the contexts exclude post-game fields
- the seed is reproducible
- the outcome is attached only at grading time

## Registry and promotion rules

Strategy promotion remains separate from calibration profile promotion.

Do not overload strategy registry with unrelated calibration artifacts unless a shared registry abstraction is intentionally introduced and justified.

## File ownership

Likely touched files:
- `omega/strategy/models.py`
- `omega/strategy/backtest/engine.py`
- `omega/strategy/versioning/registry.py`
- new artifact and replay-support files in this package

## Test expectations

Must add tests for:
- artifact round-trip serialization
- deterministic reruns with same artifact
- grading parity
- no-network backtest execution
- trace-to-artifact conversion integrity