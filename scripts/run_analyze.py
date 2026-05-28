"""Thin CLI wrapper for omega.core.contracts.service.analyze().

This is the sanctioned direct-engine path when an MCP client cannot expose the
Omega tools. It parses a request JSON file, injects a deterministic seed when
missing, delegates to analyze(), and optionally writes the raw trace JSON for
scripts/ingest_traces.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.core.contracts.schemas import GameAnalysisRequest, PlayerPropRequest  # noqa: E402
from omega.core.contracts.service import analyze  # noqa: E402
from scripts import cowork_preflight  # noqa: E402


def _load_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("request JSON must contain an object")
    return payload


def _derive_seed(request: dict[str, Any], session_id: str) -> int:
    encoded = json.dumps(
        {"request": request, "session_id": session_id},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _typed_request(kind: str, request: dict[str, Any]) -> GameAnalysisRequest | PlayerPropRequest:
    if kind == "game":
        return GameAnalysisRequest(**request)
    if kind == "prop":
        return PlayerPropRequest(**request)
    raise ValueError(f"unsupported kind={kind!r}")


def run(
    *,
    kind: str,
    request_json: Path,
    session_id: str,
    bankroll: float,
    seed: int | None = None,
    trace_out: Path | None = None,
) -> dict[str, Any]:
    request = _load_request(request_json)
    if seed is None:
        seed = _derive_seed(request, session_id)
    request.setdefault("seed", seed)
    typed = _typed_request(kind, request)
    trace = analyze(typed, session_id=session_id, bankroll=bankroll)

    if trace_out is not None:
        trace_out.mkdir(parents=True, exist_ok=True)
        out_path = trace_out / f"{trace['trace_id']}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(trace, fh, indent=2, sort_keys=False)
            fh.write("\n")
        trace["_trace_out_path"] = str(out_path)
    return trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one canonical Omega analyze() call")
    parser.add_argument("--kind", choices=("game", "prop"), required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--bankroll", type=float, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trace-out", type=Path)
    args = parser.parse_args(argv)

    gate_failures = cowork_preflight.run_formal_output_gate(require_mcp=False)
    if gate_failures:
        print("formal_output_blocked:")
        for failure in gate_failures:
            print(f"- {failure}")
        print("- Do not emit formal Omega numeric outputs until the gate passes.")
        return 2

    trace = run(
        kind=args.kind,
        request_json=args.request_json,
        session_id=args.session_id,
        bankroll=args.bankroll,
        seed=args.seed,
        trace_out=args.trace_out,
    )
    print(json.dumps(trace, indent=2, sort_keys=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
