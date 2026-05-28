"""Local MCP server for Omega's LLM-facing tool interface.

The MCP layer is intentionally thin: it validates inputs, delegates to Omega's
existing contracts, and returns versioned JSON-friendly dictionaries. It does
not duplicate simulation, calibration, edge, staking, backtest, or grading
logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from omega.mcp.schemas import (
    MCP_SCHEMA_VERSION,
    CalibrationFitPreviewRequest,
    EvidenceRetrieveRequest,
    ReplayBundle,
    ReplayToolRequest,
    TraceAttachOutcomeRequest,
    TraceQueryRequest,
)

TOOL_NAMES = (
    "omega_analyze_game",
    "omega_analyze_prop",
    "omega_analyze_slate",
    "omega_chat_orchestrate",
    "omega_replay_bundle",
    "omega_trace_get",
    "omega_trace_query",
    "omega_trace_attach_outcome",
    "omega_calibration_fit_preview",
    "omega_evidence_retrieve",
    "omega_resolve_odds",
)

RESOURCE_URIS = (
    "omega://docs/llm-mcp-interface",
    "omega://schemas/contracts",
    "omega://calibration/universal-latest",
)

PROMPT_NAMES = (
    "omega_runtime_prompt",
    "omega_missing_input_repair",
    "omega_trace_audit",
    "omega_replay_review",
    "omega_markov_evidence_guide",
)


def _formal_output_gate_failures() -> list[str]:
    """Return formal-output preflight failures for MCP analysis tools.

    Checks the EOF sentinel in cowork_preflight.py before invoking the gate
    so that a Pattern C truncation of the preflight script is detected here
    in the calling layer rather than silently producing an empty response.
    """
    try:
        from scripts import cowork_preflight
    except Exception as exc:  # noqa: BLE001
        return [f"Could not import cowork_preflight formal gate: {exc}"]

    # Sentinel check must run in the calling layer (VUL-1 fix: a truncated
    # preflight script cannot verify its own completeness).
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    preflight_path = _REPO_ROOT / "scripts" / "cowork_preflight.py"
    try:
        preflight_text = preflight_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"Cannot read cowork_preflight.py for sentinel check: {exc}"]
    if cowork_preflight._PREFLIGHT_SENTINEL not in preflight_text:
        return [
            f"cowork_preflight.py is missing its EOF sentinel "
            f"({cowork_preflight._PREFLIGHT_SENTINEL!r}). "
            "The script is likely truncated (Pattern C). "
            "Restore from git: git checkout HEAD -- scripts/cowork_preflight.py"
        ]

    return cowork_preflight.run_formal_output_gate(require_mcp=False)


def _formal_output_blocked(tool: str, failures: list[str]) -> dict[str, Any]:
    return _error(
        tool,
        "formal_output_blocked",
        {
            "failures": failures,
            "message": "Formal Omega outputs require clean preflight plus deterministic smoke.",
        },
    )


def omega_analyze_game(
    request: dict[str, Any],
    bankroll: float,
    session_id: str,
    trace_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic single-game analysis through canonical core service.

    Key request fields
    ------------------
    simulation_backend : str, default "fast_score"
        "fast_score"   — Normal/Poisson model, fast, default.
        "markov_state" — Possession-level Markov simulator. Produces empirical
                         score distributions with provenance. Preferred when
                         structured evidence signals are available.

    evidence : list[EvidenceSignal]
        Structured reasoning signals. All signal types are valid for audit and
        fast_score paths. Only the 8 Markov-eligible types below affect the
        transition matrix when simulation_backend="markov_state".

        Markov-eligible signal_type values (use exactly these strings):
          pace_up            +6% pace; matchup faster than league baseline
          pace_down          -8% pace; matchup slower than league baseline
          rest_advantage     +4% scoring rate advantage (use direction="home"/"away")
          b2b_fatigue        -6% scoring rate for the fatigued side (directional)
          def_matchup_weak   +5% offense vs. weak defender (directional)
          def_matchup_strong -5% offense vs. strong defender (directional)
          usage_role_change  -7% team rate when key player is restricted (directional)
          blowout_risk       -2% momentum acceleration; suppresses variance runaway

        All other signal types from the full SIGNAL_REGISTRY taxonomy are
        captured, audited, and scored retrospectively — they do NOT adjust the
        Markov transition matrix. Do not fabricate signal types outside the
        registry.

        Cumulative cap: no single transition attribute shifts by more than ±15%
        regardless of stacked signals.
    """
    from omega.core.contracts.schemas import GameAnalysisRequest
    from omega.core.contracts.service import analyze

    try:
        typed = GameAnalysisRequest(**request)
        gate_failures = _formal_output_gate_failures()
        if gate_failures:
            return _formal_output_blocked("omega_analyze_game", gate_failures)
        trace = analyze(typed, bankroll=bankroll, session_id=session_id, trace_quality=trace_quality)
        return _ok("omega_analyze_game", trace=trace, result=trace.get("result"))
    except ValidationError as exc:
        return _error("omega_analyze_game", "invalid_request", exc.errors())
    except ValueError as exc:
        return _error("omega_analyze_game", "invalid_request", str(exc))
    except Exception as exc:
        return _error("omega_analyze_game", "analysis_failed", str(exc))


def omega_analyze_prop(
    request: dict[str, Any],
    bankroll: float,
    session_id: str,
    trace_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic player-prop analysis through canonical core service."""
    from omega.core.contracts.schemas import PlayerPropRequest
    from omega.core.contracts.service import analyze

    try:
        typed = PlayerPropRequest(**request)
        gate_failures = _formal_output_gate_failures()
        if gate_failures:
            return _formal_output_blocked("omega_analyze_prop", gate_failures)
        trace = analyze(typed, bankroll=bankroll, session_id=session_id, trace_quality=trace_quality)
        return _ok("omega_analyze_prop", trace=trace, result=trace.get("result"))
    except ValidationError as exc:
        return _error("omega_analyze_prop", "invalid_request", exc.errors())
    except ValueError as exc:
        return _error("omega_analyze_prop", "invalid_request", str(exc))
    except Exception as exc:
        return _error("omega_analyze_prop", "analysis_failed", str(exc))


def omega_analyze_slate(
    request: dict[str, Any],
    bankroll: float,
    session_id: str,
    trace_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic slate analysis through canonical core service."""
    from omega.core.contracts.schemas import SlateAnalysisRequest
    from omega.core.contracts.service import analyze

    try:
        typed = SlateAnalysisRequest(**request)
        gate_failures = _formal_output_gate_failures()
        if gate_failures:
            return _formal_output_blocked("omega_analyze_slate", gate_failures)
        trace = analyze(typed, bankroll=bankroll, session_id=session_id, trace_quality=trace_quality)
        return _ok("omega_analyze_slate", trace=trace, result=trace.get("result"))
    except ValidationError as exc:
        return _error("omega_analyze_slate", "invalid_request", exc.errors())
    except ValueError as exc:
        return _error("omega_analyze_slate", "invalid_request", str(exc))
    except Exception as exc:
        return _error("omega_analyze_slate", "analysis_failed", str(exc))


def omega_chat_orchestrate(prompt: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a safe orchestration boundary response.

    The current source-of-truth repo does not have the stale orchestrator module
    that existed in Desktop/Omega. This tool therefore refuses to invent a
    parallel pipeline and points callers to the typed deterministic tools.
    """
    return _error(
        "omega_chat_orchestrate",
        "unsupported_current_repo",
        {
            "prompt": prompt,
            "context": context or {},
            "message": (
                "The current repo has no MCP chat orchestrator implementation. "
                "Use omega_analyze_game, omega_analyze_prop, omega_analyze_slate, "
                "trace tools, or Standard Text."
            ),
        },
    )


def omega_replay_bundle(bundle: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    """Audit a frozen replay bundle with live evidence fetching disabled."""
    try:
        req = ReplayToolRequest(bundle=ReplayBundle(**bundle), strict=strict)
    except ValidationError as exc:
        return _error("omega_replay_bundle", "invalid_replay_bundle", exc.errors())
    except ValueError as exc:
        return _error("omega_replay_bundle", "invalid_replay_bundle", str(exc))

    replay_bundle = req.bundle
    if req.strict and not replay_bundle.facts:
        return _error(
            "omega_replay_bundle",
            "empty_replay_bundle",
            "strict replay requires at least one frozen fact",
        )

    fact_count = len(replay_bundle.facts)
    filled_count = sum(1 for fact in replay_bundle.facts if fact.get("filled", True))
    missing_count = fact_count - filled_count
    response = {
        "mode": "replay_audit",
        "live_fetch_enabled": False,
        "benchmark_plane": "replay",
        "quant_benchmark": False,
        "prompt": replay_bundle.prompt,
        "trace": {
            "schema_version": replay_bundle.schema_version,
            "source_trace_id": replay_bundle.source_trace_id,
            "decision_date": replay_bundle.decision_date,
            "simulation_seed": replay_bundle.simulation_seed,
            "facts_summary": {
                "replay_mode": True,
                "live_fetch_enabled": False,
                "source_trace_id": replay_bundle.source_trace_id,
                "fact_count": fact_count,
                "filled_count": filled_count,
                "missing_count": missing_count,
            },
            "expected_outputs": replay_bundle.expected_outputs,
        },
        "audit": {
            "checks": [
                "live_fetch_disabled",
                "post_outcome_information_rejected",
                "replay_plane_only",
            ],
            "downgrades": ["missing_frozen_facts"] if missing_count else [],
        },
    }
    return _ok("omega_replay_bundle", response=response)


def omega_trace_get(trace_id: str, db_path: str | None = None) -> dict[str, Any]:
    """Retrieve a persisted trace via TraceStore."""
    from omega.trace.store import TraceStore

    store = TraceStore(db_path=db_path)
    try:
        trace = store.get_trace(trace_id)
        if trace is None:
            return _error("omega_trace_get", "trace_not_found", trace_id)
        return _ok("omega_trace_get", trace=trace)
    except Exception as exc:
        return _error("omega_trace_get", "trace_get_failed", str(exc))
    finally:
        store.close()


def omega_trace_query(
    db_path: str | None = None,
    league: str | None = None,
    start: str | None = None,
    end: str | None = None,
    has_outcome: bool | None = None,
    execution_mode: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Query persisted traces with versioned filters."""
    from omega.trace.store import TraceStore

    try:
        req = TraceQueryRequest(
            db_path=db_path,
            league=league,
            start=start,
            end=end,
            has_outcome=has_outcome,
            execution_mode=execution_mode,
            limit=limit,
        )
    except ValidationError as exc:
        return _error("omega_trace_query", "invalid_request", exc.errors())

    store = TraceStore(db_path=req.db_path)
    try:
        traces = store.query_traces(
            league=req.league,
            start=req.start,
            end=req.end,
            has_outcome=req.has_outcome,
            execution_mode=req.execution_mode,
            limit=req.limit,
        )
        return _ok(
            "omega_trace_query",
            schema_version=store.schema_version(),
            traces=traces,
        )
    except Exception as exc:
        return _error("omega_trace_query", "trace_query_failed", str(exc))
    finally:
        store.close()


def omega_trace_attach_outcome(
    trace_id: str,
    home_score: int,
    away_score: int,
    source: str = "mcp",
    db_path: str | None = None,
) -> dict[str, Any]:
    """Attach an outcome after initial trace persistence."""
    from omega.trace.store import TraceStore

    try:
        req = TraceAttachOutcomeRequest(
            trace_id=trace_id,
            home_score=home_score,
            away_score=away_score,
            source=source,
            db_path=db_path,
        )
    except ValidationError as exc:
        return _error("omega_trace_attach_outcome", "invalid_request", exc.errors())

    store = TraceStore(db_path=req.db_path)
    try:
        outcome_id = store.attach_outcome(
            req.trace_id,
            req.home_score,
            req.away_score,
            source=req.source,
        )
        return _ok("omega_trace_attach_outcome", outcome_id=outcome_id)
    except Exception as exc:
        return _error("omega_trace_attach_outcome", "outcome_attach_failed", str(exc))
    finally:
        store.close()


def omega_calibration_fit_preview(
    db_path: str | None = None,
    league: str | None = None,
    plane: str = "game",
    method: str = "isotonic",
    limit: int = 1000,
) -> dict[str, Any]:
    """Dry-run a calibration fit from graded traces without writing profiles."""
    from omega.core.calibration.fitter import CalibrationFitter
    from omega.core.calibration.registry import CalibrationRegistry
    from omega.trace.store import TraceStore

    try:
        req = CalibrationFitPreviewRequest(
            db_path=db_path,
            league=league,
            plane=plane,
            method=method,
            limit=limit,
        )
    except ValidationError as exc:
        return _error("omega_calibration_fit_preview", "invalid_request", exc.errors())

    store = TraceStore(db_path=req.db_path)
    try:
        graded = store.get_graded_traces(league=req.league, limit=req.limit)
        fitter = CalibrationFitter()
        if req.plane == "prop":
            predictions, outcomes = fitter.extract_prop_pairs(graded)
            pair_label = "prop probability/outcome"
        else:
            predictions, outcomes = fitter.extract_pairs(graded)
            pair_label = "home_win_prob/outcome"
        if len(predictions) < 30:
            return _ok(
                "omega_calibration_fit_preview",
                result={
                    "status": "skipped",
                    "dry_run": True,
                    "sample_size": len(predictions),
                    "reason": "insufficient_graded_samples",
                    "minimum_samples": 30,
                    "league": req.league,
                    "plane": req.plane,
                    "pair_type": pair_label,
                    "method": req.method,
                },
            )

        fit_league = (req.league or "UNIVERSAL").upper()
        candidate = (
            fitter.fit_isotonic(predictions, outcomes, league=fit_league)
            if req.method == "isotonic"
            else fitter.fit_shrinkage(predictions, outcomes, league=fit_league)
        )
        metrics = fitter.evaluate(candidate, predictions, outcomes)
        candidate.metrics = metrics

        incumbent = CalibrationRegistry().get_production(fit_league)
        comparison = (
            fitter.compare(candidate, incumbent, predictions, outcomes)
            if incumbent is not None
            else None
        )
        return _ok(
            "omega_calibration_fit_preview",
            result={
                "status": "success",
                "dry_run": True,
                "sample_size": len(predictions),
                "plane": req.plane,
                "pair_type": pair_label,
                "candidate": candidate.model_dump(mode="json"),
                "metrics": metrics,
                "comparison": comparison,
            },
        )
    except Exception as exc:
        return _error("omega_calibration_fit_preview", "fit_preview_failed", str(exc))
    finally:
        store.close()


def omega_evidence_retrieve(slots: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an explicit no-live-fetch skipped response for evidence retrieval."""
    try:
        req = EvidenceRetrieveRequest(slots=slots)
    except ValidationError as exc:
        return _error("omega_evidence_retrieve", "invalid_request", exc.errors())
    return _ok(
        "omega_evidence_retrieve",
        result={
            "status": "skipped",
            "reason": "no_live_fetch_in_mcp_adapter",
            "slots": req.slots,
            "filled": [],
        },
    )


def omega_resolve_odds(
    kind: str,
    league: str,
    home_team: str | None = None,
    away_team: str | None = None,
    player_name: str | None = None,
    prop_type: str | None = None,
    line: float | None = None,
    event_id: str | None = None,
    bookmaker: str = "betmgm",
    line_shopping: bool = False,
    all_books: bool = False,
) -> dict[str, Any]:
    """Resolve BetMGM-first Odds API markets into engine-ready odds inputs.

    This is an input-prep tool only. It does not compute protected Omega
    outputs such as probability, edge, EV, Kelly, units, tiers, or trace IDs.
    """
    if kind not in {"game", "prop"}:
        return _error("omega_resolve_odds", "invalid_request", "kind must be 'game' or 'prop'")
    try:
        from scripts.resolve_odds import resolve_odds

        result = resolve_odds(
            kind=kind,
            league=league,
            home_team=home_team,
            away_team=away_team,
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            event_id=event_id,
            bookmaker=bookmaker,
            line_shopping=line_shopping,
            all_books=all_books,
        )
        return _ok("omega_resolve_odds", result=result)
    except Exception as exc:  # noqa: BLE001
        return _error("omega_resolve_odds", "odds_resolution_failed", str(exc))


def build_server():
    """Build the optional FastMCP server.

    Importing the MCP SDK is deferred so tests and direct imports work without
    installing the optional dependency group.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "Omega MCP server requires the optional dependency group: "
            "python -m pip install -e .[mcp]"
        ) from exc

    mcp = FastMCP("Omega", json_response=True)
    for tool in (
        omega_analyze_game,
        omega_analyze_prop,
        omega_analyze_slate,
        omega_chat_orchestrate,
        omega_replay_bundle,
        omega_trace_get,
        omega_trace_query,
        omega_trace_attach_outcome,
        omega_calibration_fit_preview,
        omega_evidence_retrieve,
        omega_resolve_odds,
    ):
        mcp.tool()(tool)

    mcp.resource(
        "omega://docs/llm-mcp-interface",
        name="omega_llm_mcp_interface_doc",
        description="Omega LLM/MCP interface design document.",
    )(lambda: _read_repo_file("docs/LLM_MCP_INTERFACE.md"))
    mcp.resource(
        "omega://schemas/contracts",
        name="omega_contracts_schemas",
        description="Source of truth for Omega deterministic contract schemas.",
    )(lambda: _read_repo_file("omega/core/contracts/schemas.py"))
    mcp.resource(
        "omega://calibration/universal-latest",
        name="omega_calibration_universal_latest",
        description="Currently active universal calibration profile.",
    )(lambda: _read_repo_file("config/calibration/universal_latest.json"))

    mcp.prompt()(omega_runtime_prompt)
    mcp.prompt()(omega_missing_input_repair)
    mcp.prompt()(omega_trace_audit)
    mcp.prompt()(omega_replay_review)
    mcp.prompt()(omega_markov_evidence_guide)
    return mcp


def omega_runtime_prompt() -> str:
    """Prompt template for safe Omega operation through MCP."""
    return (
        "Use Omega MCP tools before shell scripts or prose math. The LLM owns "
        "routing, evidence arbitration, downgrade decisions, and explanation. "
        "The deterministic Omega tools own simulation, calibration, edge, EV, "
        "Kelly, staking, backtesting, and grading. If no engine tool can run, "
        "respond with Standard Text only."
    )


def omega_markov_evidence_guide() -> str:
    """Reference guide for structuring evidence signals on Markov-backend game requests.

    Returns the authoritative vocabulary table derived directly from the
    evidence_to_modifier module — never hand-edited here.
    """
    from omega.core.simulation.evidence_to_modifier import build_markov_vocabulary_table  # noqa: PLC0415

    header = (
        "=== Omega Markov Evidence Guide ===\n\n"
        "Use simulation_backend='markov_state' when you have game-level structural\n"
        "evidence (pace, rest, matchup) that should shift the possession-level model.\n\n"
    )
    footer = (
        "\n\nWorkflow:\n"
        "  1. Assess which of the 8 types above apply to this matchup.\n"
        "  2. Set direction='home' or direction='away' for all directional signals.\n"
        "  3. Set confidence honestly (scored retrospectively against outcomes).\n"
        "  4. Call omega_analyze_game with simulation_backend='markov_state'.\n"
        "  5. The engine applies modifiers automatically -- do NOT pre-adjust\n"
        "     home_context or away_context ratings by hand.\n\n"
        "Shadow-mode note: evidence modifiers are currently in shadow mode.\n"
        "They are applied and recorded but do not yet affect live predictions\n"
        "until the champion/challenger promotion gate is cleared."
    )
    return header + build_markov_vocabulary_table() + footer


def omega_missing_input_repair(missing_requirements: str = "") -> str:
    return (
        "Inspect missing_requirements, skip_reason, and trace downgrades, "
        "and trace facts. Retrieve only missing pre-decision inputs with "
        "provenance, then rerun the same Omega MCP analyze tool. Missing: "
        f"{missing_requirements}"
    )


def omega_trace_audit(trace_id: str = "") -> str:
    return (
        "Audit the persisted trace for schema version, prompt, execution mode, "
        "seed, odds snapshot, recommendations, downgrades, and post-decision "
        "outcome attachment. Separate current truth from recommendations. "
        f"Trace ID: {trace_id}"
    )


def omega_replay_review() -> str:
    return (
        "Review a ReplayBundle as sampled replay-plane audit only. Confirm live "
        "fetching is disabled, facts are knowable at decision time, and replay "
        "is not being used as the quant benchmark path."
    )


def _ok(tool: str, **payload: Any) -> dict[str, Any]:
    return {
        "schema_version": MCP_SCHEMA_VERSION,
        "tool": tool,
        "status": "success",
        **payload,
    }


def _error(tool: str, code: str, detail: Any) -> dict[str, Any]:
    return {
        "schema_version": MCP_SCHEMA_VERSION,
        "tool": tool,
        "status": "error",
        "error_code": code,
        "detail": detail,
    }


def _read_repo_file(path: str) -> str:
    root = Path(__file__).resolve().parents[2]
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"Refusing to read outside repo: {path}")
    return target.read_text(encoding="utf-8")


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
