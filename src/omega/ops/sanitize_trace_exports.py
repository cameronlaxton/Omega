"""Remove retired keys from trace export JSON files.

Phase 6h retires the top-level clv_capture_instructions block from trace
exports. This utility recursively scans JSON files and removes that key
wherever it appears, preserving all other content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.paths import trace_inbox_dir  # noqa: E402

RETIRED_KEYS = {"clv_capture_instructions"}


def _strip_retired_keys(value: Any) -> tuple[Any, int]:
    if isinstance(value, dict):
        removed = 0
        sanitized = {}
        for key, item in value.items():
            if key in RETIRED_KEYS:
                removed += 1
                continue
            new_item, count = _strip_retired_keys(item)
            removed += count
            sanitized[key] = new_item
        return sanitized, removed
    if isinstance(value, list):
        removed = 0
        sanitized_items = []
        for item in value:
            new_item, count = _strip_retired_keys(item)
            removed += count
            sanitized_items.append(new_item)
        return sanitized_items, removed
    return value, 0


def sanitize_file(path: Path, dry_run: bool = False) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sanitized, removed = _strip_retired_keys(payload)
    if removed and not dry_run:
        path.write_text(json.dumps(sanitized, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove retired keys from trace export JSON")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=trace_inbox_dir(),
        help="Root directory to scan recursively (default: var/inbox/traces)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    changed_files = 0
    removed_keys = 0
    for path in sorted(args.root.rglob("*.json")):
        removed = sanitize_file(path, dry_run=args.dry_run)
        if removed:
            changed_files += 1
            removed_keys += removed
            action = "would remove" if args.dry_run else "removed"
            print(f"{action} {removed} retired key(s): {path}")

    print(f"files_changed={changed_files} keys_removed={removed_keys} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
