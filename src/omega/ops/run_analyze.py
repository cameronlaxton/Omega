"""Thin CLI wrapper for omega.core.contracts.service.analyze().

This is the sanctioned direct-engine path when an MCP client cannot expose the
Omega tools. It parses a request JSON file, injects a deterministic seed when
missing, delegates to analyze(), and optionally writes the raw trace JSON for
omega-ingest-traces.
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

from omega.core.contracts.schemas import GameAnalysisRequest, PlayerPropRequest  # noqa: E402
from omega.core.contracts.seeding import derive_seed_from_request  # noqa: E402
from omega.core.contracts.service import analyze  # noqa: E402
from omega.ops import cowork_preflight  # noqa: E402

_PREFLIGHT_SCRIPT = _REPO_ROOT / "src" / "omega" / "ops" / "cowork_preflight.py"


def _check_preflight_sentinel() -> str | None:
    """Read cowork_preflight.py as raw text and verify the EOF sentinel is present.

    If the sentinel is missing the file was truncated (Pattern C). This check is
    intentionally performed by the calling layer â€” the sentinel logic cannot protect
    itself when the truncation removes the logic along with it.
    """
    try:
        text = _PREFLIGHT_SCRIPT.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Cannot read preflight script: {exc}"
    if cowork_preflight._PREFLIGHT_SENTINEL not in text:
        return (
            f"cowork_preflight.py is missing its EOF sentinel "
            f"({cowork_preflight._PREFLIGHT_SENTINEL!r}). "
            "The script is likely truncated (Pattern C mount corruption). "
            "Restore from git before running the engine: "
            "git checkout HEAD -- src/omega/ops/cowork_preflight.py"
        )
    return None


def _load_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("request JSON must contain an object")
    return payload


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
        seed = derive_seed_from_request(request)
    request.setdefault("seed", seed)
    if kind == "game":
        # Gatherer seam: merge league dynamic priors (Dixon-Coles rho, ...)
        # and record provenance when this session has a sidecar. Best-effort —
        # a backend that requires a missing prior fails closed in the engine.
        try:
            from omega.trace.priors import inject_game_priors

            request, prior_event = inject_game_priors(request)
            if prior_event:
                sidecar = _REPO_ROOT / "var" / "inbox" / "sessions" / f"{session_id}.json"
                if sidecar.exists():
                    from omega.trace.session_sidecar import append_audit_events

                    append_audit_events(sidecar, [prior_event])
        except Exception:  # noqa: BLE001 - injection must never block analysis
            pass
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

    sentinel_error = _check_preflight_sentinel()
    if sentinel_error:
        print("preflight_sentinel_missing:")
        print(f"- {sentinel_error}")
        print("- Do not emit formal Omega numeric outputs until the gate passes.")
        return 2

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



