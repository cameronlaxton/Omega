# BUG: sess-20260524-nba1.json — null-byte tail causes sidecar parse failure

**Reported:** 2026-05-26  
**Detected by:** weekly shadow calibration run (`run_action_plan.py weekly_shadow_review.json`)  
**Severity:** Low (data not lost; session excluded from calibration report only)  
**File:** `inbox/sessions/sess-20260524-nba1.json`

---

## Symptom

`report_calibration.py` emits:

```
WARNING report_calibration: Skipping invalid session sidecar sess-20260524-nba1.json: Extra data: line 16 column 1 (char 1079)
```

The session appears in `reports/latest.md` with `?` for model version, closing-line count, and webfetch failures. Its exec stats and agent notes are absent from the report.

## Root cause

The file is 1,334 bytes. Valid JSON ends at byte 1,079 (the closing `}` on line 15). Bytes 1,079–1,333 are 255 contiguous null bytes (`\x00`).

```
...trace."\n}\n\x00\x00\x00\x00\x00\x00\x00\x00...  (255 null bytes)
```

`json.load` raises `JSONDecodeError: Extra data` when it encounters the null tail after a complete parse. The SessionSidecar loader (`omega/trace/session_sidecar.py:from_path`) does not strip null bytes before parsing, so the exception propagates to `_load_session_sidecars`, which catches it, logs the warning, and skips the file.

This pattern — valid JSON followed by a fixed-length null pad — is consistent with a pre-allocated write buffer that was not truncated to the actual content length on flush. Likely cause: the sidecar was written with a fixed-size buffer (e.g., `file.write(content.ljust(1334, '\x00'))`) or a file was opened with `O_CREAT` without a subsequent `truncate()`.

The JSON content itself is valid and complete. No data is missing.

## Reproduction

```python
import json
with open("inbox/sessions/sess-20260524-nba1.json", "rb") as f:
    raw = f.read()
print(len(raw))          # 1334
print(raw[1075:1090])    # b'trace."\n}\n\x00\x00\x00\x00'
json.loads(raw)          # raises JSONDecodeError: Extra data: line 16 column 1 (char 1079)
json.loads(raw.rstrip(b"\x00"))  # parses OK
```

## Impact

- Session `sess-20260524-nba1` is excluded from all calibration report sections that join on sidecar data (model version, exec stats, agent notes, closing-line count).
- The session's 3 traces and their grades are still present in the trace store and are counted in coverage metrics; only the sidecar join is missing.
- No calibration fits, bet records, or trace outcomes are affected.

## Fix options

**Option A — Fix the file (immediate):** Strip the null tail in place.

```bash
python - <<'EOF'
from pathlib import Path
p = Path("inbox/sessions/sess-20260524-nba1.json")
raw = p.read_bytes()
p.write_bytes(raw.rstrip(b"\x00"))
print(f"Trimmed to {len(p.read_bytes())} bytes")
EOF
```

**Option B — Harden the loader (preferred, prevents recurrence):** Update `SessionSidecar.from_path` to strip null bytes before parsing.

```python
# omega/trace/session_sidecar.py
@classmethod
def from_path(cls, path: Path) -> SessionSidecar:
    raw = path.read_bytes().rstrip(b"\x00")
    return cls.model_validate(json.loads(raw))
```

This is safe: valid JSON never ends with `\x00`, so stripping cannot corrupt a well-formed file.

**Option C — Fix the writer:** Identify the sidecar write path and ensure it calls `file.truncate()` after writing, or uses `Path.write_text()` rather than a pre-allocated buffer.

Recommended sequence: apply Option B now (one-line change, zero risk), then investigate the writer to prevent new occurrences.

## Bugs also logged in this sidecar's agent_notes

The `agent_notes` field records two additional issues from the 2026-05-24 session that warrant separate tickets:

1. `markov_state backend fails AttributeError on evidence_signals at service.py:619`
2. `fetch_closing_lines skipping 4 prior-session bets with no league on trace`

These are unrelated to the null-padding defect and should be filed independently.
