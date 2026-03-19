Skills package for Omega.

Purpose
-------
Lightweight, opt-in "skills" that observe assistant behavior and emit
summaries/suggestions. Keep skills out of core deterministic engine logic.

Quickstart
---------
- Toggle skills in `omega/skills/config.json`
- Use `omega/skills/logger.write_event(...)` to emit events
- Skills register via the decorator in `omega/skills/__init__.py`

Next steps
---------
- Add tests under `tests/skills/`
- Integrate skill enablement checks into orchestrator/session manager
