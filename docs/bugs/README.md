# Omega Bug Catalog

Index of filed bugs. One `BUG-<id>.md` per entry; this file is the queryable
index the `omega-known-bug-sentinel` skill reads before formal analysis.

**Filing a bug:** copy the shape of an existing `BUG-*.md` (symptom, root
cause, reproduction, impact, fix options) and add a row below. Severity
follows the sidecar-audit convention: **S1** critical (data loss / silent
corruption / wrong calibration gating), **S2** high, **S3** medium, **S4** low.

**Closing a bug:** update the row's status to `closed`, and add a `**Status:**
CLOSED <date>` line at the top of the bug doc itself explaining what was done.
Do not delete closed bug docs — they are the record of what broke and why.

## Active / historical bugs

| ID | Status | Severity | Summary |
|---|---|---|---|
| [sess-20260524-nba1-null-padding](BUG-sess-20260524-nba1-null-padding.md) | closed | S4 | Fixed-size write buffer left a session sidecar with a trailing null-byte pad, causing a JSON parse failure that excluded the session from calibration reports (data intact, only the sidecar join was affected). |

## Sentinel behavior

The sentinel checks this table for rows where `status=active` (or
`investigating`) **and** `severity` is S1/S2 before formal analysis. As of
this writing there are none — the table above is empty of active entries by
design, not by omission; the sentinel should report "no active critical bugs"
rather than silently passing. If a future critical bug is filed here while
still active, `omega-known-bug-sentinel` enforces Bet Card suppression until
its row is marked `closed`.
