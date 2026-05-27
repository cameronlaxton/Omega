"""Guard against the retired RUN_AUDIT.md / RUN_TRACE.jsonl scratch files
being regenerated at the repo root. Their replacements are:

  - inbox/sessions/<session_id>.json  (audit_events)
  - reports/run_audits/<session_id>.audit.md  (rendered)

If either file reappears, the LLM or a script has reverted to muscle-memory
behavior. Fix the source rather than the symptom.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_audit_md_not_at_repo_root():
    assert not (_REPO_ROOT / "RUN_AUDIT.md").exists(), (
        "RUN_AUDIT.md is retired. Append structured audit_events to the "
        "session sidecar via omega.trace.session_sidecar.append_audit_events "
        "and render via the render_audit action."
    )


def test_run_trace_jsonl_not_at_repo_root():
    assert not (_REPO_ROOT / "RUN_TRACE.jsonl").exists(), (
        "RUN_TRACE.jsonl is retired. Append structured audit_events to the "
        "session sidecar via omega.trace.session_sidecar.append_audit_events."
    )
