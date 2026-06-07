# QA Audit — Session Sidecars & Logging

**Date:** 2026-06-07
**Auditor:** multi-agent QA pass (5 parallel read-only workstreams + lead consolidation)
**Scope:** session sidecars (`var/inbox/sessions/<sid>.json`) + the `<sid>.events.jsonl`
recovery mirror + `audit_events`, **and** the full logging surface — python `logging`
config, trace/bet-ledger/evidence DB persistence logging, ingest file-routing,
audit renderer, ETL provenance.
**Method:** every finding is repro-confirmed (executed repro or exact code/line citation);
unconfirmed observations were dropped. Severity: **S1** critical (data loss / silent
corruption / wrong calibration gating), **S2** high (real bug, bounded/recoverable),
**S3** medium (fragility / missing guard / observability gap), **S4** low
(legacy / dead / doc-drift / cosmetic).

---

## 1. Executive summary

The subsystem is **fundamentally sound**: the atomic-write primitive is correct, the
authority boundary (numbers in the DB, narrative in the sidecar) holds in the audit
renderer and ingest paths, no secrets leak into logs, the live DB is intact (V14,
integrity OK, 416 traces), and all 16 live sidecars pass strict validation. The defects
are concentrated in three areas: **(a) the QA gate that is supposed to validate sidecars
was pointed at the wrong directory**, **(b) the append/mirror machinery loses or
mis-mirrors events under concurrency, pre-existing drift, or non-append writes**, and
**(c) accumulated doc/skill drift** around schema version, table names, and the
protected-field contract.

**19 findings** (after de-duplicating across workstreams): **2 S1, 3 S2, 7 S3, 7 S4.**

**Top risks**
1. **F1 (S1, FIXED):** `omega-validate-session-sidecars` / `omega-validate-all` defaulted to
   `inbox/sessions` (no `var/`), so the routine session-close gate validated a stale
   **2-file** directory while the **16** live sidecars went unchecked — a false-green QA gate.
2. **F2 (S1, backlog):** `append_audit_events` is an unlocked read-modify-write that also
   strict-re-validates the whole sidecar; concurrent appends silently drop events and any
   pre-existing drift permanently blocks (and loses) further appends. A dropped
   `quality_gate=fail` event would let an ineligible trace look calibration-clean.
3. **F3 (S2, backlog):** the "recovery mirror" is not a faithful superset — `write_sidecar`
   bypasses it and a `create_sidecar` re-call mis-mirrors; **7 of 16 live sessions already
   show JSON↔JSONL drift.**

**Overall health verdict: GOOD with one critical gate bug (now fixed) and a structural
mirror/append hardening backlog.** No live data corruption, no protected-value leak, no
wrong calibration gating observed in the current ledger.

### Counts

| Severity | Count | Fixed this pass | Backlog |
|---|---|---|---|
| S1 | 2 | 1 | 1 |
| S2 | 3 | 1 | 2 |
| S3 | 7 | 1 | 6 |
| S4 | 7 | 1 (+docs) | 6 |
| **Total** | **19** | **4 families** | **15** |

---

## 2. Findings

IDs cross-reference the workstream finding IDs (A=sidecar contract, B=logging, C=on-disk
data, D=harness/tests, E=integration paths).

| ID | Title | Cat | Sev | Location | Evidence (confirmed) | Disposition |
|----|-------|-----|-----|----------|----------------------|-------------|
| **F1** | Validator default inbox drops `var/` → routine gate validates stale 2-file dir, 16 live sidecars unchecked (false green) | correctness | **S1** | `ops/validate_session_sidecars.py:49`, `ops/validate_all.py:73,78` | No-arg `omega-validate-session-sidecars` validated **2** files in `inbox/sessions`; `var/inbox/sessions` holds **16**. `validate_all.py:73` help text literally said "default: var/inbox/sessions" while the code used `inbox/sessions`. `validate_all` SKIPs (not fails) on empty dir → compound false green. [C1/D1/D2] | **FIXED** |
| **F2** | `append_audit_events` unlocked read-modify-write + strict re-validation → lost/blocked events | concurrency | **S1** | `trace/session_sidecar.py:226-237` (RMW), `:234` (strict `from_path`) | Interleaved two appends sharing a pre-append snapshot → final JSON kept only the 2nd; 1st lost. Injected one extra top-level key → every subsequent `append_audit_events` raises `ValidationError` **before** the mirror write, so the event is lost from JSON *and* mirror. A lost `quality_gate=fail` → wrong calibration gating. [A1/A2/E2] | backlog |
| **F3** | "Recovery mirror" is not a faithful superset — `write_sidecar` bypasses it; `create_sidecar` re-call mis-mirrors; 7/16 live sessions drift | concurrency/atomicity | **S2** | `trace/session_sidecar.py:207-211` (`write_sidecar`, no mirror), `:214-223` (`create_sidecar` re-call) | `inspect` confirms `_mirror_events_jsonl` is NOT called by `write_sidecar`; repro: `write_sidecar` w/ 2 events → JSON=2, mirror=0 (C4 root cause). Live drift table: 7/16 sessions diverge — incl. `sess-20260603-mlb1` mirror duplicated 2× (C3), and `session_close`/`engine_run` events present in JSON but absent from mirror in 3 sessions (C4). [E1/E4/C2/C3/C4] | backlog |
| **F4** | `rebuild_sidecar_from_jsonl` returns extra keys → fails `extra="forbid"` re-validation; a test masked it | schema | **S2** | `trace/session_sidecar.py:532-572`; test `tests/trace/test_session_sidecar_durability.py:236,199` | Repro: raw return has `event_count`+`source_jsonl`; `SessionSidecar.model_validate(rebuilt)` → "2 validation errors … Extra inputs". The durability test stripped both keys before validating, green-lighting a non-compliant contract. No production caller (recovery path is WIP). [A4/D4/A10] | **FIXED** |
| **F5** | Protected-field scan covers only `inputs`/`outputs`, not `notes`/`assumptions`/`bugs`; doc claimed `notes` too | authority-boundary | **S2** | `trace/session_sidecar.py:170-179`; doc `OMEGA_COWORK.md:667` | Appended `notes="edge_pct=5.2 kelly_fraction=0.03"` and `assumptions=["edge_pct is 5.2"]` → both ACCEPTED; `inputs={"edge_pct":5.2}` → BLOCKED. A naive notes key-scan would false-positive on legitimate null-audits (`append_null_data_audit` itself writes `notes="NULL detected: result.edge_pct"`). [A3/D5/E3/C8] | **doc FIXED**; code value-scan = backlog |
| **F6** | No `NullHandler` on the `omega` root logger → library-context logs hit lastResort / silently dropped | observability | **S3** | `src/omega/__init__.py` (no logging setup) | grep `addHandler\|NullHandler\|propagate` across all `__init__.py` = 0. When omega is imported without a CLI `basicConfig` (MCP server, tests, notebooks), `logger.info/debug` from trace/integrations/core are discarded and warnings emit unformatted via lastResort. [B1] | **FIXED** |
| **F7** | `log_effective_db` not called by ~half of TraceStore entrypoints (incl. MCP server, audit_renderer, closing-line writers) → silent wrong-DB risk | observability | **S3** | `trace/store.py:450-463` (def); 24 `TraceStore(` sites, only 12 log | MCP `server.py` opens TraceStore at 8 sites with no logging config and never logs effective DB; `audit_renderer.py:72` opens a writable store without it. Combined with the FUSE/network auto-redirect, a silently-redirected/empty DB is invisible to those flows. [B3/B4/E7/E8] | backlog |
| **F8** | JSONL event mirror grows unbounded — no cap/rotation/retention | resource | **S3** | `trace/session_sidecar.py:187-204` | `_mirror_events_jsonl` appends per event with no size/age/count cap; one file per session accumulates in `var/inbox/sessions/` forever; nothing prunes. [B5] | backlog |
| **F9** | Unscoped failed `quality_gate` condemns the whole session; 300s window matcher never actually spares a trace | correctness | **S3** | `trace/session_sidecar.py:414,424-438,472-482`; writer `ops/mark_session_qa_failed.py:60` | A trace inside 300s of an unscoped fail → `fail/timestamp_window`; outside → still `fail/session_fallback`. The window branch only relabels scope, never changes the verdict, because `unscoped_fails` catches all. `mark_session_qa_failed` writes exactly this empty-`trace_ids` gate → session-wide calibration-ineligibility. Conservative-by-design, but the matcher implies protection it doesn't provide. [A5] | backlog |
| **F10** | `quarantine_sidecar` reason-file write is unguarded after the (durable) move; collision branch untested | error-handling | **S3** | `trace/session_sidecar.py:524-527` | `shutil.move` then `reason_path.write_text` with no try/except — a reason-write failure after the move leaves the file quarantined with no reason and the op non-repeatable (source gone). Collision-rename path has no test. [A7/D9] | backlog |
| **F11** | `_emit_provenance_event` swallows audit-write failures with bare `except: pass` (no log) | error-handling | **S3** | `integrations/_etl.py:195-196` | Unlike the bet-ledger autolog (which warns), a total failure of the provenance audit emits no log line — a silently lost QA signal. [E task2] | backlog |
| **F12** | Test-coverage gaps in sidecar+logging behaviors | test-coverage | **S3** | `tests/trace/*`, `tests/ops/*` | Uncovered: append concurrency/lost-event (D7), append-blocked-by-drift / mirror reconciliation (D8), protected-field rejection in `notes`/`assumptions`/`bugs` (D6), `quarantine_sidecar` collision (D9), 1 of 7 `quality_gate_verdict_for_trace` scope branches — `unrelated_session_failure` via `scoped_other_fails` (D10), `log_effective_db` invocation (D11). Baseline before fixes: 483 passed / 5 skipped. [D6-D12] | backlog (F1/F4 gaps closed this pass) |
| **F13** | Doc/code drift: §8 event_type list omits `quality_gate`; stale `V6`/`V10`/`bet_records` table refs | schema/doc | **S4** | `OMEGA_COWORK.md:5,666,691`; `docs/phase6/ARTIFACT_AUTHORITY.md:27` | Code `_VALID_EVENT_TYPES` includes `quality_gate` (and `append_null_data_audit` emits it) but §8 list omitted it. `bet_records` table dropped at V14 (confirmed gone in DB) yet docs said "SQLite V10 … bet_records" / "SQLite V6". [D13/D14] | **FIXED** |
| **F14** | Recovery helper fabricates session state (`bankroll=1000.0`, `model_version="unknown"`, `purpose="recovered_session"`) | on-disk/error-handling | **S4** | `trace/session_sidecar.py:563-565` | Hardcoded placeholders. Latent (no production caller) but if wired, injects a fake $1000 bankroll into the narrative artifact. Docstring now flags them as placeholders (this pass); gating recovery behind explicit operator action is backlog. [A4b] | backlog |
| **F15** | Logging-config hygiene: no central config (28 duplicated `basicConfig`); `backfill_trace_quality` module-level config ignores `--verbose`; engine inline lazy logger; no structured/file logging | observability/legacy | **S4** | `ops/*` (28 sites), `ops/backfill_trace_quality.py:62`, `core/simulation/engine.py:976-977` | Format/level are consistent across 27/28; `backfill_trace_quality` configures at import (hardcoded INFO) and re-`setLevel` as a workaround; `engine.py` re-imports logging inside a function vs the module-level convention used elsewhere. No `FileHandler`/rotation/JSON formatter anywhere. [B2/B8/B9/B10] | backlog |
| **F16** | Live data hygiene: 2 sessions never closed; 3 `exec_stats` empty; `Z` vs `+00:00` ts inconsistency; stale `inbox/sessions` orphan dir | on-disk-data | **S4** | `var/inbox/sessions/*`; stale `inbox/sessions/` | `closed_at=null` on `sess-20260601-mlb1`, `sess-20260604-wnb1`; empty `exec_stats` on 3 sidecars; mixed ts suffixes; the stale `inbox/sessions/` (the old F1 default target) holds 9 orphan `.events.jsonl`. No corruption — hygiene only. [C5/C6/C7/C9] | backlog |
| **F17** | Skill/prompt drift: `append_audit_events(bootstrap=…)` and an MCP sidecar wrapper referenced but nonexistent; trace-qa `bet_record` rejection wording | legacy/doc | **S4** | `prompts/league_analysis_prompt.md:21`, `prompts/system_prompt.txt:66`, `.agents/skills/omega-trace-qa/SKILL.md:43,86` | `signature(append_audit_events)` = `(path, events)` — no `bootstrap` kwarg (would `TypeError`, pushing agents to hand-write JSON → a fresh F3 vector). `server.py` has no sidecar wrapper. trace-qa §2/§10 describe a rejection reason that doesn't match ingest. [E5/E6/D15] | backlog |
| **F18** | `quality_gate_status` (blunt session-wide) has zero production callers | dead-code | **S4** | `trace/session_sidecar.py:262-273` | grep: callers = own module + tests only. Retained intentionally (comment :286-287); flagged so a future caller doesn't re-introduce session-wide over-blocking. [A9] | backlog (note) |
| **F19** | Verified-healthy (no defect) | — | — | — | No secret/PII leakage in integration logs — `odds_api.py:275` logs only `path`, never the keyed URL (B6). Authority boundary holds in `audit_renderer` — all numbers from DB, sidecar only narrative (E task4). `ingest_traces` treats `load_sidecar_safe`→None as `unknown`, not clean (E task1). `_atomic.atomic_write_text` is correct (temp+fsync+replace, cleanup-on-error). | n/a |

---

## 3. Fixes applied this pass (inline-safe)

All applied with regression tests; `pytest tests/trace tests/ops` → **487 passed, 5 skipped**
(was 483/5; +4 new tests), ruff clean, **zero `var/` data mutation**.

1. **F1 — validator default paths (S1).** Added testable constants
   `_DEFAULT_SESSIONS_INBOX` / `_DEFAULT_TRACES = <repo>/var/inbox/…` and used them as the
   argparse defaults in `ops/validate_session_sidecars.py` and `ops/validate_all.py`.
   *Verified:* no-arg `omega-validate-session-sidecars` now validates **16** sidecars (0
   invalid); `omega-validate-all --skip-tests` runs the session-sidecars step against
   `var/inbox/sessions` and PASSES. New test: `tests/ops/test_validate_paths.py` (pins the
   defaults under `var/inbox`, asserts orchestrator/sub-validator agree, behavioral
   valid/invalid count smoke).
2. **F4 — rebuild_sidecar schema compliance (S2).** Dropped the redundant `event_count` /
   `source_jsonl` keys from `rebuild_sidecar_from_jsonl` (derivable from
   `len(audit_events)` / the input path) so the recovery dict validates as-is; corrected the
   docstring. Rewrote the two durability tests to validate the **raw** return (no key
   stripping) — closing the masking gap.
3. **F6 — NullHandler (S3).** Added `logging.getLogger("omega").addHandler(logging.NullHandler())`
   to `src/omega/__init__.py` (standard library convention) so library-context logs no
   longer hit lastResort.
4. **F5 (doc) + F13 — contract/doc reconciliation (S2/S4).** OMEGA_COWORK §8: added
   `quality_gate` to the `event_type` allowlist; rewrote the protected-field rule to state
   enforcement is mechanical for `inputs`/`outputs` and discipline-only for free-text
   `notes`/`assumptions`/`bugs` (with the null-audit false-positive rationale). Fixed stale
   DB header (`V10`→`V14`, `bet_records`→`bet_ledger` + current table list) at
   `OMEGA_COWORK.md:5` and `:691`; updated the `ARTIFACT_AUTHORITY.md:27` row to `bet_ledger`.

---

## 4. Prioritized remediation backlog (report-only)

Effort: **S** ≤1h · **M** ≤½day · **L** multi-day/design.

| Pri | ID | Item | Sev | Effort |
|----|----|------|-----|--------|
| 1 | **F2** | Serialize/lock `append_audit_events` (file lock or compare-and-swap retry); decouple append from full-sidecar strict re-validation; mirror-first so a parse failure can't drop the event | S1 | L |
| 2 | **F3** | Make the JSONL mirror a true superset: have `write_sidecar` mirror missing events (dedup-aware — naive mirroring would double-append per C3), guard `create_sidecar` against re-call on an existing path; add a json-count == mirror-line-count invariant test; investigate/repair the 7 drifted live sessions | S2 | M |
| 3 | **F5** | Add a careful `name[=:]value` value-pattern scan to `notes`/`assumptions`/`bugs` that does NOT reject null-audit field-name notes; tests pinning both directions | S2 | M |
| 4 | **F7** | Call `log_effective_db` at every write-capable TraceStore open (MCP server, audit_renderer, closing-line/capture/backfill/fit paths) | S3 | S |
| 5 | **F9** | Have quarantine writers populate `trace_ids`, or drop the dead `timestamp_window` branch that implies non-existent protection | S3 | S |
| 6 | **F11** | Replace `_etl` bare `except: pass` with a warning log | S3 | S |
| 7 | **F8** | Add a retention/rotation sweep (or `--prune-mirrors`) for `var/inbox/sessions/*.events.jsonl` | S3 | M |
| 8 | **F10** | Guard the `quarantine_sidecar` reason-file write (or write it before the move); add the collision test | S3 | S |
| 9 | **F12** | Add the missing tests: append concurrency, drift reconciliation, notes/assumptions protection, quarantine collision, the `unrelated_session_failure` scope branch, `log_effective_db` | S3 | M |
| 10 | **F15** | Extract a shared `configure_logging(verbose)`; fix `backfill_trace_quality` import-time config; hoist `engine.py` logger; optional `--log-file`/JSON opt-in | S4 | M |
| 11 | **F14** | Gate `rebuild_sidecar_from_jsonl` behind explicit operator action; mark reconstructed values UNVERIFIED instead of fabricating a $1000 bankroll | S4 | S |
| 12 | **F16** | Stale-open-session sweep on close; soft `exec_stats` key conformance warning; normalize ts to `Z`; retire the stale `inbox/sessions/` tree | S4 | S |
| 13 | **F17** | Fix the prompt `append_audit_events(bootstrap=…)` example and the system-prompt MCP-wrapper reference; reword trace-qa §2/§10 | S4 | S |
| 14 | **F18** | Mark/deprecate `quality_gate_status` as off the ingest path | S4 | S |

---

## 5. Coverage map

**Audited (in scope):**
- Sidecar contract & lifecycle — `trace/session_sidecar.py`, `trace/_atomic.py` (full read + repros).
- Logging surface — 58 `getLogger` + 145 `print` sites enumerated; 28 `basicConfig`;
  `log_effective_db`; mirror growth; secret-leakage scan of integrations.
- On-disk data — all 16 live sidecars validated; JSON↔JSONL drift table; `var/inbox/traces`
  processed/failed; DB read-only (schema V14, `bet_records` gone, bet_ledger provenance,
  `trace_qa_verdicts`, evidence/signal scoring, sidecar↔trace cross-check).
- Harness & tests — `validate_all`, `validate_session_sidecars`, `validate_trace_export`,
  `db_status`; the three QA skills vs code; `tests/trace` + `tests/ops` coverage map.
- Integration paths — `ingest_traces`, `_etl._emit_provenance_event`,
  `mark_session_qa_failed`, `audit_renderer`, `report_calibration`, `store` autolog/redirect.

**Explicitly out of scope (not audited this pass):** calibration fit/promotion math,
simulation engine internals, odds-provider fetch correctness, the Postgres repository
backend (tests env-gated/skipped), and non-sidecar report rendering beyond the authority
boundary check.

---

*Spec baseline used: `docs/phase6/ARTIFACT_AUTHORITY.md`, `OMEGA_COWORK.md` §8, `AGENTS.md`,
`prompts/reference/output_modes.md`, and the `omega-trace-qa` / `omega-session-bootstrap` /
`omega-replay-qa` skills.*
