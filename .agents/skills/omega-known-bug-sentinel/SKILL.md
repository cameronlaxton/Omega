---
name: omega-known-bug-sentinel
description: Run the Omega known-bug sentinel before any formal analysis. Checks live engine state against the bug catalog, reports which bugs are present or fixed, enforces Bet Card suppression for active critical bugs, and writes results to the session sidecar.
---

# Omega Known-Bug Sentinel

Run at session start (after `omega-session-bootstrap` opens the sidecar), when
switching sports, or whenever a suspected regression occurs. This skill was
previously referenced by name in agent tooling with no repo-level definition
backing it — this file, [`docs/bugs/README.md`](../../../docs/bugs/README.md)
(the catalog), and the plugin thin-pointer are that definition.

## What this is not

This is not a substitute for `omega-trace-qa` (post-analysis trace
completeness) or `omega-replay-qa` (replay/persistence boundaries). It is a
narrow, fast check: *is any filed critical bug still live in this session's
engine state?*

## Step 1 — Read the catalog

Read [`docs/bugs/README.md`](../../../docs/bugs/README.md). It is a table of
every filed bug with `status` (`active` | `investigating` | `closed`) and
`severity` (S1 critical … S4 low). As of this skill's authoring the table has
zero `active`/`investigating` rows — that is the expected steady state, not a
sign the sentinel has nothing to do. Report it explicitly rather than silently
passing:

```
Known-bug sentinel: 0 active/investigating bugs in docs/bugs/README.md. Proceeding.
```

## Step 2 — For each active S1/S2 row, run its live check

A bug row filed as `active`/`investigating` at S1 or S2 **must** carry a "Live
check" section in its own `BUG-*.md` — a concrete, runnable command or grep
that determines whether the defect is still present in this session's engine
state (not "ask the LLM to guess"). Run every such check. Do not invent a
check for a bug doc that doesn't have one; flag the missing check as a gap in
the bug doc itself instead of skipping silently.

## Step 3 — Enforce suppression

- Any **critical (S1)** bug whose live check confirms it is still present:
  suppress formal Bet Card / BetSlip / EdgeDetail output for the affected
  scope (session-wide if the bug's own doc doesn't specify a narrower one).
  Downgrade to `RESEARCH_CANDIDATE` per
  [`prompts/reference/output_modes.md`](../../../prompts/reference/output_modes.md)
  and say why, citing the bug ID.
- **S2 and below:** note the finding in the session sidecar; do not suppress
  output solely on this basis unless the bug's own doc says formal output
  must be blocked.

## Step 4 — Record to the sidecar

```python
from omega.trace.session_sidecar import append_audit_events
from pathlib import Path

append_audit_events(
    Path(f"var/inbox/sessions/{session_id}.json"),
    [{
        "ts": now_iso,
        "event_type": "quality_gate",
        "step": "known_bug_sentinel",
        "status": "ok",  # "fail" if a live critical bug forced suppression
        "notes": "0 active bugs" ,  # or "BUG-<id> confirmed live; Bet Cards suppressed"
    }],
)
```

## Filing a new bug the sentinel should watch

Follow [`docs/bugs/README.md`](../../../docs/bugs/README.md)'s filing
convention, add a "Live check" section to the bug doc, and add the row to the
catalog table with `status: active`. The next sentinel run will pick it up —
there is no separate registration step.
