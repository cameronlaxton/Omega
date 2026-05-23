# Session Bug Report — 2026-05-21
## OneDrive Mount Corruption: Truncated and Null-Padded Source Files

**Discovered during:** Cowork session startup preflight  
**Severity:** Critical — engine unbootable until resolved  
**Status:** Partially mitigated in-session; permanent fix needed

---

## Root Cause

The Cowork Linux sandbox mounts the OneDrive repo at `/sessions/<id>/mnt/Omega/`. During this session the mount presented **stale, incomplete versions of 22+ source files**. Two distinct corruption patterns were observed:

### Pattern A — Trailing Null Bytes (20 files)
Files were fully allocated on disk but only partially written. The unwritten tail was filled with `\x00` null bytes. Python's `compile()` refuses source containing null bytes, producing:

```
ValueError: source code string cannot contain null bytes
```

These files were identifiable because:
- `content.rstrip(b'\x00') != content`
- No internal null bytes; all nulls were contiguous at the end
- File size matched the allocated block size, not the actual content size

### Pattern B — Hard Truncation (2 files)
Files were cut off mid-expression with no null padding. Python's parser hit the truncation point and raised `SyntaxError`:

```
SyntaxError: unterminated string literal (detected at line 874)
SyntaxError: expected ':'
```

These were NOT detectable by null-byte scan — the files appeared syntactically valid up to the cut point.

### Pattern C — Silent semantic truncation (observed 2026-05-23)
Files arrive cut off at a function/class boundary, so the remaining bytes still parse as valid Python. Neither the null-byte scan nor `ast.parse()` catches this. The corruption surfaces only when the missing function is called at runtime:

```
NameError: name '_extract_calibration_audit' is not defined
  File "/sessions/<id>/mnt/Omega/omega/trace/persistable.py", line 84, in from_analyze_output
    calibration_audit=_extract_calibration_audit(result),
```

Observed instance: `omega/trace/persistable.py` arrived at 153 lines instead of 191; the trailing `_extract_calibration_audit` definition was missing, and `scripts/ingest_traces.py` crashed mid-pipeline with `NameError`.

Detection requires **content equality with the git blob** — same SHA-1 hash as the index. `scripts/cowork_preflight.py:verify_against_git()` (added 2026-05-23) does this for every tracked `.py` file using `git hash-object` versus `git rev-parse HEAD:<path>`. When divergence is reported, run `python scripts/cowork_preflight.py --repair-from-git` to restore via `git checkout`, which writes through the mount cache.

---

## Affected Files

### Pattern A — Null-padded (stripped in-session)
| File | Original Size | Nulls Removed | Syntax After Fix |
|------|-------------|---------------|-----------------|
| `omega/core/models.py` | 13,178 | 158 | ok |
| `omega/core/betting/kelly.py` | 2,384 | 3 | ok |
| `omega/core/betting/parlay.py` | 6,096 | 149 | ok |
| `omega/core/calibration/profiles.py` | 3,108 | 30 | ok |
| `omega/core/config/leagues.py` | 12,684 | 25 | ok |
| `omega/core/simulation/validation.py` | 6,772 | 18 | ok |
| `omega/integrations/espn_boxscore.py` | 11,157 | 204 | ok |
| `omega/integrations/espn_mlb.py` | 8,151 | 102 | ok |
| `omega/integrations/espn_nba.py` | 7,812 | 84 | ok |
| `omega/skills/data_quality_grader.py` | 3,443 | 22 | ok |
| `omega/skills/evidence_validator.py` | 2,819 | 26 | ok |
| `omega/skills/evolution_tracker.py` | 1,146 | 18 | ok |
| `omega/strategy/models.py` | 5,037 | 30 | ok |
| `omega/strategy/anchor/formatter.py` | 3,115 | 9 | ok |
| `omega/strategy/backtest/engine.py` | 13,221 | 22 | ok |
| `omega/strategy/versioning/promotion.py` | 4,133 | 44 | ok |
| `omega/synthesis/composer.py` | 8,112 | 1 | ok |
| `omega/trace/bet_record.py` | 3,581 | 12 | ok |
| `omega/trace/clv.py` | 4,444 | 58 | ok |
| `scripts/cowork_preflight.py` | 4,428 | 36 | ok |
| `scripts/ingest_traces.py` | 14,792 | 3,509 | ok |

### Pattern B — Hard truncation (repaired by appending missing tail)
| File | Truncated At | Missing Content |
|------|-------------|-----------------|
| `omega/core/contracts/service.py` | line 874, byte 32,655 | Lines 874–922 (`analyze_slate` loop tail + `_extract_team`) |
| `omega/core/calibration/registry.py` | line 205, byte 8,040 | Lines 206–210 (context_slice filter + return) |

---

## What Was Attempted and Outcome

### Attempt 1 — Write tool to Windows path (service.py)
- **Action:** Used the `Write` tool to overwrite the complete 922-line `service.py` to the Windows path `C:\Users\camer\OneDrive\Documents\GitHub\Omega\omega\core\contracts\service.py`
- **Outcome:** FAILED. The Linux mount did not pick up the change. The mount returned the old 32,655-byte stale version on all subsequent reads. The `Write` tool writes to the Windows filesystem; the Linux sandbox mount does not invalidate its cache on Windows-side writes.

### Attempt 2 — Bash append for Pattern B files
- **Action:** Used `python3` via bash to open the file in append-binary mode (`'ab'`) and write the known-missing tail bytes directly to the mount path.
- **Outcome:** SUCCESS for both truncated files. `ast.parse()` confirmed valid syntax post-repair.

### Attempt 3 — Null-byte strip for Pattern A files
- **Action:** Walk all `.py` files under the repo, detect trailing null bytes, strip with `content.rstrip(b'\x00')`, write back via bash Python, verify with `ast.parse()`.
- **Outcome:** SUCCESS. All 21 Pattern A files passed syntax check after stripping.

---

## What Was NOT Completed

The session was interrupted before:
- Running `cowork_preflight.py` to completion (was still blocked on Pattern B registry.py at time of stoppage)
- `fetch_outcomes_all.py` — no outcomes gathered this session
- `report_calibration.py` — no calibration health check run
- Inbox inspection — pending traces, action plans, and session sidecars not reviewed
- Session sidecar write

---

## Bugs to Fix in Claude Code

### BUG-MOUNT-1: Stale Linux mount does not reflect Windows-side writes
The `Write` tool (which writes to the Windows OneDrive path) produces no visible effect in the bash sandbox mount. Any workflow that relies on Write-then-bash-execute is broken in this environment.

**Recommended fix:** When repairing or writing source files that bash must subsequently execute, always use bash Python `open(..., 'wb').write(...)` directly on the mount path. Never rely on the Write tool to propagate changes that bash needs to see in the same session.

**Longer-term:** Add a note to `OMEGA_COWORK.md` — "file repairs must be written via bash, not the Write tool, due to mount cache isolation."

### BUG-MOUNT-2: OneDrive sync left 22 Python source files corrupt on the mount
Multiple files arrived at the sandbox with either trailing null bytes or hard truncation. This is an OneDrive partial-sync artifact: the file allocation table reserved the full block but sync did not finish writing all bytes before the mount snapshot was taken.

**Recommended fix for Claude Code:** Add a preflight step to `scripts/cowork_preflight.py` that scans all repo `.py` files for:
1. Trailing null bytes (`content != content.rstrip(b'\x00')`)
2. Syntax errors (`ast.parse()` failure)

Report failures explicitly rather than letting them surface as cryptic import errors mid-session. Suggested check function:

```python
import ast, os

def scan_source_integrity(repo_root: str) -> list[dict]:
    issues = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d != '__pycache__' and not d.startswith('.')]
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(root, f)
            raw = open(path, 'rb').read()
            null_count = len(raw) - len(raw.rstrip(b'\x00'))
            if null_count:
                issues.append({'file': path, 'issue': 'trailing_nulls', 'count': null_count})
                continue
            try:
                ast.parse(raw.decode('utf-8', errors='replace'))
            except SyntaxError as e:
                issues.append({'file': path, 'issue': 'syntax_error', 'detail': str(e)})
    return issues
```

If any issues are found, print them and exit non-zero before attempting any imports.

### BUG-MOUNT-3: Stale Python 3.13 `.pyc` files in `__pycache__`
Several `__pycache__` directories contained `.cpython-313.pyc` files (compiled on Windows with Python 3.13), which cannot be used or deleted by the Python 3.10 sandbox. These are harmless to execution (Python falls back to source) but the permission errors produced noise in preflight output and may mask real issues.

**Recommended fix:** Add `.gitignore` entries to exclude `__pycache__/` and `*.pyc` if not already present. Prevents Windows-compiled bytecode from polluting the mount.

---

## Recommended Action for Next Session

Before running any analysis, execute this repair script:

```bash
cd /path/to/Omega
python3 -c "
import ast, os
omega_dir = '.'
fixed = 0
for root, dirs, files in os.walk(omega_dir):
    dirs[:] = [d for d in dirs if d != '__pycache__' and not d.startswith('.')]
    for f in files:
        if not f.endswith('.py'): continue
        path = os.path.join(root, f)
        raw = open(path, 'rb').read()
        stripped = raw.rstrip(b'\x00')
        if stripped != raw and b'\x00' not in stripped:
            open(path, 'wb').write(stripped)
            fixed += 1
            print(f'stripped: {path}')
print(f'Done — {fixed} files repaired')
"
python scripts/cowork_preflight.py --direct-only
```

Then verify syntax on any file that still fails with a hard-truncation error and append the missing tail manually using the Read tool (Windows path) as the source of truth.
