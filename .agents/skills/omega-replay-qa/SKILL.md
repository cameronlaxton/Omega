---
name: omega-replay-qa
description: Audit Omega replay, trace persistence, and calibration dry-run boundaries.
---

# Omega Replay QA

Replay is sampled audit only, not the quant benchmark path.

Before accepting replay evidence:
- Confirm live fetching is disabled.
- Confirm all facts were knowable at decision time.
- Confirm replay output is used for routing/evidence/downgrade QA only.
- Do not use orchestrator replay as the quant benchmark path.
- Keep calibration dry-runs tied to frozen artifacts, odds snapshots, and known outcomes.
