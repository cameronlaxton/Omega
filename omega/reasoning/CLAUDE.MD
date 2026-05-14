# omega/reasoning/CLAUDE.md

## Purpose

This package owns orchestration, intent handling, planning, gathering coordination, quality gating, and replay-mode control.

It does not own simulation logic, calibration logic, staking logic, or backtest grading.

## Phase 6 mission in this package

Add:
- trace persistence hook integration
- replay-safe execution mode
- frozen evidence export boundaries

Do not add:
- duplicate quant benchmarking logic
- alternative edge calculators
- calibration fitting code

## Hard boundaries

Reasoning may decide:
- what the user is asking
- what data is needed
- whether the answer should downgrade
- whether simulation should run
- what output package should be composed

Reasoning may not decide:
- probability calibration formula
- edge math
- stake sizing math
- grading against outcomes

Those belong elsewhere.

## Replay mode rules

Replay mode exists to audit:
- routing quality
- evidence quality
- downgrade discipline
- trace completeness

Replay mode must:
- disable live data fetching
- consume frozen evidence bundles only
- preserve normal stage ordering as much as possible
- emit replay metadata and audit results

Replay mode must not:
- become the default benchmark path
- silently substitute missing evidence with live calls
- mutate frozen evidence
- use wall-clock time for deterministic decisions

## Trace rules

Execution traces are audit artifacts.

Requirements:
- every completed orchestrator run should be persistable
- trace fields should be additive and stable
- trace should be serializable without request/response wrapper objects
- trace linkage to replay and frozen artifacts must be explicit

Do not:
- make trace shape depend on UI response formatting
- store transient objects that cannot be serialized
- hide downgrade reasons

## Seed rules

Historical replay and artifact-derived execution must not derive seeds from the current date.

Allowed seed basis:
- event identity
- artifact identity
- frozen market identity

Disallowed seed basis:
- current day
- current timestamp
- session randomness

## File ownership

Likely touched files:
- `omega/reasoning/orchestrator.py`
- replay-specific helpers in this package
- trace export or adapter hooks only if orchestration-specific

Avoid introducing new top-level abstractions unless existing flow cannot support them.

## Test expectations

Must add tests for:
- replay mode disables live fetches
- traces remain populated in replay
- deterministic seed behavior when using frozen inputs
- downgrade behavior survives replay