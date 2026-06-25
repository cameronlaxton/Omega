# Calibration Fitting — Scheduled Operator Prompt

**Cadence:** periodic (not daily)
**Purpose:** review calibration status and promote/flag profiles through the standing gate.

Copy the block below into the scheduled task. This repo copy is the source of truth — keep the
scheduler in sync with this file.

---

Use the **omega-session-bootstrap** skill first.

Check all calibration status across markets and leagues and report it concisely. Promote any
profile that clears the promotion gate; flag any that warrant review or demotion.

Follow the calibration method and promotion rules in `src/omega/core/calibration/CLAUDE.md` and the
continuous CLV loop in `docs/issue28_clv_loop.md`. Do not relax the promotion gate.

This is a maintenance task — present a short, readable promotion/flag summary. No betting
recommendations or protected values are required.
