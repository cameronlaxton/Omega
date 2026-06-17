"""
omega.ops.validate_trace_export â€” validate exported trace files before ingest.

Checks the canonical export shape expected by ingest_traces.py and reports
clear, per-file PASS/FAIL with reasons. Use it before ingest to avoid the
failure mode where the agent re-runs analyze() because an export didn't ingest:
**if the raw analyze() output is valid but the wrapper is wrong, re-wrap and
re-export â€” do not re-run analyze().**

Modes:
  --strict  (default) : extra export-quality checks (session_id, result.status,
                        prediction fields, identity, NBA game_context) are errors.
  --lenient           : those become warnings; errors then mirror exactly what
                        ingest_traces.py would actually reject (good for legacy
                        / backfill inspection).

Usage:
    omega-validate-trace-export var/inbox/traces
    omega-validate-trace-export var/inbox/traces/foo.json --lenient
    omega-validate-trace-export var/inbox/traces --json

Exit codes:
    0 â€” all validated files pass (no errors)
    1 â€” at least one file has an error, or a path was unreadable
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.export_validator import validate_export_block  # noqa: E402


def _iter_files(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.glob("*.json") if p.is_file())
    return [target]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate trace export files before ingest.")
    parser.add_argument("target", type=Path, help="A .json file or a directory of *.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict", action="store_true", help="(default) export-quality checks are errors")
    mode.add_argument("--lenient", action="store_true", help="export-quality checks are warnings")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    strict = not args.lenient  # strict is the default

    target: Path = args.target
    if not target.exists():
        print(f"path not found: {target}", file=sys.stderr)
        return 1

    files = _iter_files(target)
    if not files:
        print(f"no .json files in {target}")
        return 0

    reports = []
    n_error = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            n_error += 1
            reports.append((path, None, str(exc)))
            continue
        report = validate_export_block(payload, strict=strict)
        if not report.ok:
            n_error += 1
        reports.append((path, report, None))

    if args.json:
        out = []
        for path, report, read_err in reports:
            if read_err is not None:
                out.append({"file": path.name, "ok": False, "read_error": read_err})
            else:
                out.append(
                    {
                        "file": path.name,
                        "ok": report.ok,
                        "trace_id": report.trace_id,
                        "kind": report.kind,
                        "issues": [asdict(i) for i in report.issues],
                    }
                )
        print(json.dumps({"mode": "strict" if strict else "lenient", "results": out}, indent=2))
    else:
        for path, report, read_err in reports:
            if read_err is not None:
                print(f"REJECT {path.name}: unreadable JSON: {read_err}")
                continue
            print(f"{path.name}: {report.summary()}")
        print(f"\n{len(files)} file(s), {n_error} with errors (mode={'strict' if strict else 'lenient'}).")

    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())




