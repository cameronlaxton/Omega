# Retired legacy artifacts

These files are **retired historical artifacts, not a source of truth.** They are
kept here for historical/debug reference only. Do not read them programmatically
and do not recreate them at the repo root.

| File | Replaced by |
|---|---|
| `RUN_TRACE.jsonl` | Per-session JSONL mirror (`var/inbox/sessions/<sid>.events.jsonl`) + the ledger (`var/omega_traces.db`) |
| `RUN_AUDIT.md` | `var/reports/run_audits/<sid>.audit.md`, rendered by `omega/trace/audit_renderer.py` |

The audit renderer (`omega/trace/audit_renderer.py`) explicitly refuses to read
either file. The agent runtime prompts (`prompts/system_prompt.txt`,
`OMEGA_COWORK.md`) instruct the agent never to create them.

A root-anchored `.gitignore` guard (`/RUN_TRACE.jsonl`, `/RUN_AUDIT.md`) prevents
them from reappearing at the repo root. See
[`docs/phase6/ARTIFACT_AUTHORITY.md`](../../docs/phase6/ARTIFACT_AUTHORITY.md)
for the current artifact authority map.
