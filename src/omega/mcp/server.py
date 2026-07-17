"""Local MCP server for Omega's LLM-facing tool interface.

The MCP layer is intentionally thin: it validates inputs, delegates to Omega's
existing contracts, and returns versioned JSON-friendly dictionaries. It does
not duplicate simulation, calibration, edge, staking, backtest, or grading
logic.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from omega.core.config.leagues import get_league_config
from omega.mcp.schemas import (
    MCP_SCHEMA_VERSION,
    CalibrationFitPreviewRequest,
    EvidenceRetrieveRequest,
    FetchOutcomesRequest,
    FlatBetRequest,
    GameContextRequest,
    PortfolioSummaryRequest,
    ReplayBundle,
    ReplayToolRequest,
    SettleBetsRequest,
    TraceAttachOutcomeRequest,
    TraceQueryRequest,
    TraceVoidPropRequest,
)
from omega.paths import repo_root

logger = logging.getLogger(__name__)

TOOL_NAMES = (
    "omega_analyze_game",
    "omega_analyze_prop",
    "omega_analyze_slate",
    "omega_run_batch",
    "omega_chat_orchestrate",
    "omega_replay_bundle",
    "omega_trace_get",
    "omega_get_matchup_brief",
    "omega_trace_query",
    "omega_trace_attach_outcome",
    "omega_trace_void_prop",
    "omega_fetch_outcomes",
    "omega_settle_bets",
    "omega_calibration_fit_preview",
    "omega_evidence_retrieve",
    "omega_list_events",
    "omega_resolve_odds",
    "omega_record_flat_bet",
    "omega_get_portfolio_summary",
    "omega_get_game_context",
    "omega_propose_signal",
)

RESOURCE_URIS = (
    "omega://docs/llm-mcp-interface",
    "omega://schemas/contracts",
)

PROMPT_NAMES = (
    "omega_runtime_prompt",
    "omega_missing_input_repair",
    "omega_trace_audit",
    "omega_replay_review",
    "omega_markov_evidence_guide",
)

_MCP_EXPLORATORY_DOWNGRADE = "mcp_exploratory_iterations"


def _commence_window_for_game_date(game_date: str, league: str | None = None) -> tuple[str, str]:
    """Return the UTC window for a YYYY-MM-DD slate date, adjusted for the league's timezone.

    When game_date is in a local timezone (e.g., EST), this expands the window to capture
    games that start late in that timezone, whose commence_time falls after midnight UTC
    on the following calendar day. Default timezone is America/New_York (EST/EDT).
    """
    league_tz = "America/New_York"  # Default to EST
    if league:
        config = get_league_config(league)
        league_tz = config.get("timezone", "America/New_York")

    day = datetime.strptime(game_date[:10], "%Y-%m-%d").date()

    # Midnight on this day in the league's local timezone, to the next local midnight.
    local_tz = ZoneInfo(league_tz)
    local_midnight = datetime.combine(day, datetime.min.time()).replace(tzinfo=local_tz)
    local_next_midnight = local_midnight + timedelta(days=1)

    # Convert the local-day window to UTC so late local games (post-00:00Z next day) are kept.
    start_utc = local_midnight.astimezone(timezone.utc)
    end_utc = local_next_midnight.astimezone(timezone.utc)
    return start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_with_mcp_defaults(
    request: dict[str, Any],
    *,
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(request)
    if payload.get("n_iterations") is not None:
        return payload, {}

    if kind == "game":
        backend = str(payload.get("simulation_backend") or "fast_score")
        n_iterations = 100 if backend == "markov_state" else 300
        payload["n_iterations"] = n_iterations
        return payload, {
            "n_iterations": n_iterations,
            "reason": "mcp_exploratory_default",
            "simulation_backend": backend,
        }
    if kind == "prop":
        payload["n_iterations"] = 500
        return payload, {
            "n_iterations": 500,
            "reason": "mcp_exploratory_default",
        }
    return payload, {}


def _maybe_inject_game_priors(payload: dict[str, Any], session_id: str | None) -> dict[str, Any]:
    """Gatherer seam: merge league dynamic priors (e.g. Dixon-Coles rho) into
    the request and record provenance in the session sidecar.

    Best-effort by design — on any failure the payload passes through
    unchanged and a backend that requires the prior fails closed
    (status="skipped"), which is the honest outcome.
    """
    try:
        from omega.trace.priors import inject_game_priors

        payload, event = inject_game_priors(payload)
        if event and session_id:
            sidecar = repo_root() / "var" / "inbox" / "sessions" / f"{session_id}.json"
            if sidecar.exists():
                from omega.trace.session_sidecar import append_audit_events

                append_audit_events(sidecar, [event])
    except Exception as exc:  # noqa: BLE001 - injection must never block analysis
        logger.warning("prior injection failed for session_id=%s: %s", session_id, exc)
        if session_id:
            sidecar = repo_root() / "var" / "inbox" / "sessions" / f"{session_id}.json"
            if sidecar.exists():
                try:
                    from omega.trace.session_sidecar import append_audit_events

                    append_audit_events(
                        sidecar,
                        [
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "event_type": "data_provenance",
                                "step": "mcp:prior_injection",
                                "status": "skipped",
                                "notes": f"prior_injection_failed: {exc}",
                                "trace_ids": [],
                            }
                        ],
                    )
                except Exception as audit_exc:  # noqa: BLE001
                    logger.warning(
                        "prior injection audit event failed for session_id=%s: %s",
                        session_id,
                        audit_exc,
                    )
    return payload


def _maybe_inject_prop_priors(payload: dict[str, Any], session_id: str | None) -> dict[str, Any]:
    """Gatherer seam for player props: merge the fitted NB dispersion ``k``
    (NFL yardage) into player_context and record provenance in the sidecar.

    Best-effort by design — on any failure the payload passes through unchanged
    and the service falls back to the per-request std-derived ``k`` (fail-open).
    """
    try:
        from omega.trace.priors import inject_prop_priors

        payload, event = inject_prop_priors(payload)
        if event and session_id:
            sidecar = repo_root() / "var" / "inbox" / "sessions" / f"{session_id}.json"
            if sidecar.exists():
                from omega.trace.session_sidecar import append_audit_events

                append_audit_events(sidecar, [event])
    except Exception as exc:  # noqa: BLE001 - injection must never block analysis
        logger.warning("prop prior injection failed for session_id=%s: %s", session_id, exc)
    return payload


def _reject_protected_trace_quality(tool: str, trace_quality: dict[str, Any] | None) -> None:
    """Fail closed when caller-supplied trace_quality carries engine-owned quant keys.

    ``trace_quality`` is orchestrator QUALITY metadata (aggregate_quality,
    downgrades, exclusion reasons). The engine owns edge/EV/Kelly/units/
    confidence-tier/probability values; an LLM-facing caller must never smuggle
    them into the persisted quality block. Mirrors the sidecar's
    ``ProtectedValueError`` discipline at the MCP seam.
    """
    from omega.core.contracts.protected_fields import find_protected_key

    found = find_protected_key(trace_quality)
    if found:
        raise ValueError(
            f"{tool}: caller-supplied trace_quality contains protected engine field "
            f"{found!r}. trace_quality carries quality metadata only "
            "(e.g. aggregate_quality, downgrades); engine-owned values are computed "
            "by analyze() and live in the trace result."
        )


def _reject_protected_reasoning_inputs(tool: str, reasoning_inputs: dict[str, Any]) -> None:
    """Fail closed when batch provenance/context metadata carries engine-owned keys."""
    from omega.core.contracts.protected_fields import find_protected_key

    found = find_protected_key(reasoning_inputs)
    if found:
        raise ValueError(
            f"{tool}: synthesized reasoning_inputs contains protected engine field "
            f"{found!r}. reasoning_inputs carries source/context provenance only; "
            "engine-owned values are computed by analyze() and stay in the trace result."
        )


def _merge_mcp_trace_quality(
    trace_quality: dict[str, Any] | None,
    mcp_defaults: dict[str, Any],
) -> dict[str, Any] | None:
    if not mcp_defaults:
        return trace_quality

    merged = dict(trace_quality or {})
    merged["downgrades"] = sorted({*merged.get("downgrades", []), _MCP_EXPLORATORY_DOWNGRADE})
    merged["calibration_exclusion_reasons"] = sorted(
        {
            *merged.get("calibration_exclusion_reasons", []),
            _MCP_EXPLORATORY_DOWNGRADE,
        }
    )
    return merged


def _formal_output_gate_failures() -> list[str]:
    """Return formal-output preflight failures for MCP analysis tools.

    Checks the EOF sentinel in cowork_preflight.py before invoking the gate
    so that a Pattern C truncation of the preflight script is detected here
    in the calling layer rather than silently producing an empty response.
    """
    try:
        from omega.ops import cowork_preflight
    except Exception as exc:  # noqa: BLE001
        return [f"Could not import cowork_preflight formal gate: {exc}"]

    # Sentinel check must run in the calling layer (VUL-1 fix: a truncated
    # preflight script cannot verify its own completeness).
    root = repo_root()
    preflight_path = root / "src" / "omega" / "ops" / "cowork_preflight.py"
    try:
        preflight_text = preflight_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"Cannot read cowork_preflight.py for sentinel check: {exc}"]
    if cowork_preflight._PREFLIGHT_SENTINEL not in preflight_text:
        return [
            f"cowork_preflight.py is missing its EOF sentinel "
            f"({cowork_preflight._PREFLIGHT_SENTINEL!r}). "
            "The script is likely truncated (Pattern C). "
            "Restore from git: git checkout HEAD -- src/omega/ops/cowork_preflight.py"
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

        EvidenceSignal schema:
          category  one of: player_form, matchup, situational, team_form
          plane     one of: player, game
          window    one of: last_1, last_3, last_5, last_10, season, series, h2h, matchup
          direction optional; one of: over, under, home, away, neutral

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
        regardless of stacked signals. Canonical vocabulary reference:
        prompts/reference/markov_evidence_vocab.md (that file wins on conflict).

    trace_quality : dict, optional
        Orchestrator QUALITY metadata only (e.g. aggregate_quality, downgrades).
        Engine-owned values (edge/EV/Kelly/units/confidence tier/probabilities)
        are rejected — analyze() computes those.
    """
    from omega.core.contracts.schemas import GameAnalysisRequest
    from omega.core.contracts.service import analyze

    try:
        _reject_protected_trace_quality("omega_analyze_game", trace_quality)
        request_payload, mcp_defaults = _request_with_mcp_defaults(request, kind="game")
        request_payload = _maybe_inject_game_priors(request_payload, session_id)
        effective_trace_quality = _merge_mcp_trace_quality(trace_quality, mcp_defaults)
        typed = GameAnalysisRequest(**request_payload)
        gate_failures = _formal_output_gate_failures()
        if gate_failures:
            return _formal_output_blocked("omega_analyze_game", gate_failures)
        trace = analyze(
            typed,
            bankroll=bankroll,
            session_id=session_id,
            trace_quality=effective_trace_quality,
        )
        tq = trace.get("trace_quality") or {}
        extra: dict[str, Any] = {}
        if tq.get("evidence_quality") == "missing":
            extra["evidence_warning"] = (
                "No evidence signals were submitted despite full provided context. "
                "Include at least one EvidenceSignal in evidence=[] to enable "
                "retrospective signal scoring and the evidence-learning feedback loop."
            )
        return _ok(
            "omega_analyze_game",
            trace=trace,
            result=trace.get("result"),
            trace_quality=tq,
            mcp_defaults=mcp_defaults,
            **extra,
        )
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
    """Run deterministic player-prop analysis through canonical core service.

    evidence : list[EvidenceSignal]
        Structured reasoning signals for player prop analysis.

        EvidenceSignal schema:
          category  one of: player_form, matchup, situational, team_form
          plane     one of: player, game
          window    one of: last_1, last_3, last_5, last_10, season, series, h2h, matchup
          direction optional; one of: over, under, home, away, neutral

    trace_quality : dict, optional
        Orchestrator QUALITY metadata only (e.g. aggregate_quality, downgrades).
        Engine-owned values (edge/EV/Kelly/units/confidence tier/probabilities)
        are rejected — analyze() computes those.
    """
    from omega.core.contracts.schemas import PlayerPropRequest
    from omega.core.contracts.service import analyze

    try:
        _reject_protected_trace_quality("omega_analyze_prop", trace_quality)
        request_payload, mcp_defaults = _request_with_mcp_defaults(request, kind="prop")
        effective_trace_quality = _merge_mcp_trace_quality(trace_quality, mcp_defaults)
        request_payload = _maybe_inject_prop_priors(request_payload, session_id)
        typed = PlayerPropRequest(**request_payload)
        gate_failures = _formal_output_gate_failures()
        if gate_failures:
            return _formal_output_blocked("omega_analyze_prop", gate_failures)
        trace = analyze(
            typed,
            bankroll=bankroll,
            session_id=session_id,
            trace_quality=effective_trace_quality,
        )
        tq = trace.get("trace_quality") or {}
        extra: dict[str, Any] = {}
        if tq.get("evidence_quality") == "missing":
            extra["evidence_warning"] = (
                "No evidence signals were submitted despite full provided context. "
                "Include at least one EvidenceSignal in evidence=[] to enable "
                "retrospective signal scoring and the evidence-learning feedback loop."
            )
        return _ok(
            "omega_analyze_prop",
            trace=trace,
            result=trace.get("result"),
            trace_quality=tq,
            mcp_defaults=mcp_defaults,
            **extra,
        )
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
        _reject_protected_trace_quality("omega_analyze_slate", trace_quality)
        typed = SlateAnalysisRequest(**request)
        gate_failures = _formal_output_gate_failures()
        if gate_failures:
            return _formal_output_blocked("omega_analyze_slate", gate_failures)
        trace = analyze(
            typed, bankroll=bankroll, session_id=session_id, trace_quality=trace_quality
        )
        return _ok("omega_analyze_slate", trace=trace, result=trace.get("result"))
    except ValidationError as exc:
        return _error("omega_analyze_slate", "invalid_request", exc.errors())
    except ValueError as exc:
        return _error("omega_analyze_slate", "invalid_request", str(exc))
    except Exception as exc:
        return _error("omega_analyze_slate", "analysis_failed", str(exc))


def omega_run_batch(
    entries: list[dict[str, Any]],
    bankroll: float,
    session_id: str,
    presentation_mode: str = "decision_support",
    engine_auto_ledger_mode: str = "disabled",
) -> dict[str, Any]:
    """Run N game/prop analyses in one call: resolve odds, analyze, write export blocks.

    Each entry is a BatchAnalysisEntry dict.  If odds fields are absent the tool
    resolves them via omega_resolve_odds before calling analyze().  For props,
    prop_type may be a list — the first market that resolves successfully is used.

    ``presentation_mode`` ('decision_support' default | 'recommendation_lab')
    frames how authorized values are presented downstream; it never widens what
    ``output_mode`` authorizes. ``engine_auto_ledger_mode`` ('disabled' default |
    'shadow') is the per-run wager-autolog authority: 'disabled' writes NO
    engine_auto bet-ledger rows; 'shadow' additionally requires the operator
    env gate OMEGA_ENABLE_ENGINE_SHADOW=1 at persist time. Both are stamped
    onto every trace (schema v2). Invalid values are rejected, not coerced.

    Per entry, supply the structured reasoning fields rather than prose-only output:

    - ``evidence`` — typed EvidenceSignal dicts (see omega/core/contracts/evidence.py);
      the engine applies known types deterministically and persists all for scoring.
    - ``reasoning_presentation`` — analyst-note prose keyed thesis/market_read/why/
      risks/verdict. Qualitative only; protected engine values are rejected.
    - ``roster_context`` — RSVG (Roster & Situational Verification Gate) payload, a
      RosterContextPayload dict (omega/core/gates/rsvg.py). The gate runs BEFORE odds
      resolution/analyze(): ``blocked`` entries are skipped, ``research_candidate``
      stamps ``reasoning_downgrade_rationale`` + ``trace_quality.rsvg`` on the trace,
      and verified key-absence ``usage_role_change`` signals merge into evidence.

    Export blocks are written to ``var/inbox/traces/<trace_id>.json`` in the
    standard shape required by ``omega-ingest-traces``.

    Returns a summary envelope with per-entry status (ok/skipped/error) and the
    list of trace_ids and export paths for successfully completed entries.
    """
    import json as _json
    from datetime import datetime, timezone

    from omega.core.contracts.schemas import (
        BatchAnalysisEntry,
        EventIdentityV1,
        coerce_engine_auto_ledger_mode,
        coerce_presentation_mode,
    )
    from omega.core.contracts.seeding import derive_seed_from_request
    from omega.core.contracts.service import analyze
    from omega.integrations.odds_resolver import EVENT_PROVIDER, resolve_odds
    from omega.paths import repo_root

    # Explicit API parameters are validated, not silently coerced — a caller
    # that names a mode must name a real one. (Persist-time reads of missing/
    # malformed values still fail closed via the coerce_* helpers.)
    if coerce_presentation_mode(presentation_mode) != presentation_mode:
        return _error(
            "omega_run_batch",
            "INVALID_INPUT",
            f"invalid presentation_mode {presentation_mode!r}; "
            "expected 'decision_support' or 'recommendation_lab'",
        )
    if coerce_engine_auto_ledger_mode(engine_auto_ledger_mode) != engine_auto_ledger_mode:
        return _error(
            "omega_run_batch",
            "INVALID_INPUT",
            f"invalid engine_auto_ledger_mode {engine_auto_ledger_mode!r}; "
            "expected 'disabled' or 'shadow'",
        )

    # Gate check runs once for the entire batch.
    gate_failures = _formal_output_gate_failures()
    if gate_failures:
        return _formal_output_blocked("omega_run_batch", gate_failures)

    from omega.trace._atomic import atomic_write_text

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    inbox_dir = repo_root() / "var" / "inbox" / "traces"
    inbox_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    trace_ids: list[str] = []
    export_paths: list[str] = []
    errors: list[dict[str, Any]] = []
    # RSVG content-quality: evaluate_roster_context sees only one matchup and
    # cannot itself notice the same source_summaries text reused verbatim
    # across different entries in this batch (a live QA audit found exactly
    # this: identical boilerplate injected into 68 entries to mechanically
    # satisfy the non-empty-list check). Track signatures across the batch
    # here, where cross-entry visibility actually exists. The signature is
    # keyed to the matchup it was first seen for: a prop entry legitimately
    # sharing its own game's roster_context (same matchup) is NOT boilerplate;
    # only the same text reused for a DIFFERENT matchup is. (A 2026-07 audit
    # found 240 prop traces spuriously downgraded by the matchup-blind check.)
    _seen_rsvg_summary_signatures: dict[
        tuple[tuple[str, str], ...], tuple[str, str, str]
    ] = {}

    for idx, raw in enumerate(entries):
        try:
            entry = BatchAnalysisEntry(**raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "index": idx,
                    "identifier": str(raw.get("player_name") or raw.get("home_team") or idx),
                    "error": str(exc),
                }
            )
            results.append({"index": idx, "status": "error", "error": str(exc)})
            continue

        game_date = entry.game_date or today
        try:
            commence_time_from, commence_time_to = _commence_window_for_game_date(
                game_date, league=entry.league
            )
        except ValueError as exc:
            identifier = (
                entry.player_name
                if entry.kind == "prop"
                else f"{entry.away_team} @ {entry.home_team}"
            )
            errors.append({"index": idx, "identifier": identifier, "error": str(exc)})
            results.append(
                {"index": idx, "status": "error", "identifier": identifier, "error": str(exc)}
            )
            continue
        identifier = (
            entry.player_name if entry.kind == "prop" else f"{entry.away_team} @ {entry.home_team}"
        )

        # --- RSVG pre-analysis gate (roster & situational verification) ---
        # Runs BEFORE odds resolution so blocked entries never touch providers or
        # analyze(). The gate only validates structured facts the operator/LLM
        # already gathered; it computes no engine values.
        rsvg = None
        if entry.roster_context is not None:
            from omega.core.gates.rsvg import RosterContextPayload, evaluate_roster_context

            try:
                payload = RosterContextPayload(**entry.roster_context)
                # Identity agreement (Phase 0): the RSVG payload must describe
                # THIS entry's matchup. A verified-looking context for a
                # different league/game must never validate an analysis — this
                # is the cross-event/cross-league contamination boundary.
                mismatches = [
                    f"{field}: entry={entry_val!r} rsvg={payload_val!r}"
                    for field, entry_val, payload_val in (
                        ("league", entry.league, payload.league),
                        ("home_team", entry.home_team, payload.home_team),
                        ("away_team", entry.away_team, payload.away_team),
                    )
                    if str(entry_val or "").casefold().strip()
                    != str(payload_val or "").casefold().strip()
                ]
                if payload.game_date and payload.game_date != game_date:
                    mismatches.append(
                        f"game_date: entry={game_date!r} rsvg={payload.game_date!r}"
                    )
                if mismatches:
                    msg = "rsvg_identity_mismatch: " + "; ".join(mismatches)
                    errors.append({"index": idx, "identifier": identifier, "error": msg})
                    results.append(
                        {
                            "index": idx,
                            "status": "error",
                            "identifier": identifier,
                            "error": msg,
                        }
                    )
                    continue
                summaries = entry.roster_context.get("source_summaries") or []
                signature = tuple(
                    sorted(
                        (str(s.get("source", "")), str(s.get("summary", ""))) for s in summaries
                    )
                )
                matchup_key = (
                    str(entry.league or "").casefold(),
                    str(entry.home_team or "").casefold(),
                    str(entry.away_team or "").casefold(),
                )
                is_duplicate = False
                if signature:
                    first_matchup = _seen_rsvg_summary_signatures.get(signature)
                    if first_matchup is None:
                        _seen_rsvg_summary_signatures[signature] = matchup_key
                    elif first_matchup != matchup_key:
                        is_duplicate = True
                rsvg = evaluate_roster_context(
                    payload,
                    duplicate_summary_detected=is_duplicate,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {"index": idx, "identifier": identifier, "error": f"rsvg_invalid: {exc}"}
                )
                results.append(
                    {
                        "index": idx,
                        "status": "error",
                        "identifier": identifier,
                        "error": f"rsvg_invalid: {exc}",
                    }
                )
                continue
            if rsvg.status == "blocked":
                results.append(
                    {
                        "index": idx,
                        "status": "skipped",
                        "identifier": identifier,
                        "reason": "rsvg_blocked",
                        "rsvg": rsvg.gate_audit.model_dump(),
                    }
                )
                continue

        # Typed usage_role_change signals for verified key absences merge into the
        # entry's evidence so the engine can apply them deterministically.
        entry_evidence: list[dict[str, Any]] = list(entry.evidence)
        if rsvg is not None:
            entry_evidence.extend(rsvg.evidence_dicts())

        # --- Odds resolution ---
        # Provider event identity captured from the resolver result (Phase 1):
        # the same the-odds-api event id groups this entry's trace with every
        # other market on the same real-world event.
        resolved_event_id: str | None = None
        resolved_commence_time: str | None = None
        if entry.kind == "prop":
            prop_types = (
                entry.prop_type
                if isinstance(entry.prop_type, list)
                else ([entry.prop_type] if entry.prop_type else [])
            )
            odds_patch: dict[str, Any] | None = None
            resolved_prop_type: str | None = None
            if (
                entry.odds_over is not None
                and entry.odds_under is not None
                and entry.line is not None
            ):
                # Pre-supplied
                odds_patch = {
                    "line": entry.line,
                    "odds_over": entry.odds_over,
                    "odds_under": entry.odds_under,
                }
                resolved_prop_type = prop_types[0] if prop_types else None
            else:
                fallback_books = ["betmgm", "draftkings", "fanduel", "williamhill_us", "espnbet"]
                for pt in prop_types:
                    for book in fallback_books:
                        try:
                            res = resolve_odds(
                                kind="prop",
                                league=entry.league,
                                player_name=entry.player_name,
                                prop_type=pt,
                                home_team=entry.home_team,
                                away_team=entry.away_team,
                                commence_time_from=commence_time_from,
                                commence_time_to=commence_time_to,
                                bookmaker=book,
                            )
                        except Exception:  # noqa: BLE001
                            continue
                        if res.get("status") == "success" and res.get("request_patch"):
                            odds_patch = res["request_patch"]
                            resolved_prop_type = pt
                            resolved_event_id = res.get("event_id") or None
                            resolved_commence_time = res.get("commence_time") or None
                            break
                    if odds_patch is not None and "odds_over" in odds_patch and "odds_under" in odds_patch:
                        break
                    else:
                        odds_patch = None
                        resolved_event_id = None
                        resolved_commence_time = None
            if odds_patch is None or "odds_over" not in odds_patch or "odds_under" not in odds_patch:
                results.append(
                    {
                        "index": idx,
                        "status": "skipped",
                        "identifier": identifier,
                        "reason": "odds_unavailable",
                    }
                )
                continue

            request_dict: dict[str, Any] = {
                "player_name": entry.player_name,
                "league": entry.league,
                "prop_type": resolved_prop_type,
                "line": odds_patch["line"],
                "home_team": entry.home_team,
                "away_team": entry.away_team,
                "game_date": game_date,
                "odds_over": odds_patch["odds_over"],
                "odds_under": odds_patch["odds_under"],
                "player_context": entry.player_context or {},
                "game_context": entry.game_context or {},
                "n_iterations": entry.n_iterations,
                "evidence": entry_evidence,
            }

        else:  # game
            game_odds = entry.odds
            if game_odds is None:
                fallback_books = ["betmgm", "draftkings", "fanduel", "williamhill_us", "espnbet"]
                for book in fallback_books:
                    try:
                        res = resolve_odds(
                            kind="game",
                            league=entry.league,
                            home_team=entry.home_team,
                            away_team=entry.away_team,
                            commence_time_from=commence_time_from,
                            commence_time_to=commence_time_to,
                            bookmaker=book,
                        )
                        if res.get("status") == "success" and res.get("request_patch"):
                            game_odds = res["request_patch"].get("odds") or res["request_patch"]
                            resolved_event_id = res.get("event_id") or None
                            resolved_commence_time = res.get("commence_time") or None
                            break
                    except Exception:  # noqa: BLE001
                        continue
                if game_odds is None:
                    results.append(
                        {
                            "index": idx,
                            "status": "skipped",
                            "identifier": identifier,
                            "reason": "odds_unavailable",
                        }
                    )
                    continue

            request_dict = {
                "home_team": entry.home_team,
                "away_team": entry.away_team,
                "league": entry.league,
                "home_context": entry.home_context or {},
                "away_context": entry.away_context or {},
                "game_context": entry.game_context or {},
                "odds": game_odds,
                "n_iterations": entry.n_iterations,
                "evidence": entry_evidence,
            }

        # --- Event identity agreement (Phase 1) ---
        # A caller-supplied event id that disagrees with the id the resolver
        # matched means the odds belong to a different event than the caller
        # believes — never analyze under a wrong identity.
        if (
            entry.event_id
            and resolved_event_id
            and entry.event_id != resolved_event_id
        ):
            msg = (
                f"event_identity_mismatch: entry event_id={entry.event_id!r} but odds "
                f"resolved to event_id={resolved_event_id!r}"
            )
            errors.append({"index": idx, "identifier": identifier, "error": msg})
            results.append(
                {"index": idx, "status": "error", "identifier": identifier, "error": msg}
            )
            continue
        provider_event_id = resolved_event_id or entry.event_id or None

        # --- Seed derivation ---
        if entry.seed is not None:
            request_dict["seed"] = entry.seed
        else:
            request_dict["seed"] = derive_seed_from_request(request_dict, date=game_date)

        # --- Dynamic priors (after seeding so seeds stay payload-stable) ---
        if entry.kind != "prop":
            request_dict = _maybe_inject_game_priors(request_dict, session_id)
        else:
            request_dict = _maybe_inject_prop_priors(request_dict, session_id)

        # --- Reasoning/provenance payload (validated BEFORE analyze) ---
        market_context: dict[str, Any] = {}
        if entry.kind == "prop":
            market_context = {
                "player": entry.player_name,
                "prop_type": resolved_prop_type,
                "line": odds_patch["line"],
                "odds_over": odds_patch["odds_over"],
                "odds_under": odds_patch["odds_under"],
            }
        elif game_odds:
            market_context = {"odds": game_odds}

        reasoning_sources = list(entry.reasoning_sources)
        if rsvg is not None:
            reasoning_sources.extend(
                s for s in rsvg.reasoning_sources if s not in reasoning_sources
            )
        reasoning_inputs = {
            "sources": reasoning_sources,
            "fields_gathered": list(request_dict.keys()),
            "missing_fields": [],
            "market_context": market_context,
        }
        try:
            _reject_protected_reasoning_inputs("omega_run_batch", reasoning_inputs)
        except ValueError as exc:
            errors.append({"index": idx, "identifier": identifier, "error": str(exc)})
            results.append(
                {"index": idx, "status": "error", "identifier": identifier, "error": str(exc)}
            )
            continue

        # --- Analyze ---
        try:
            trace = analyze(request_dict, bankroll=bankroll, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            errors.append({"index": idx, "identifier": identifier, "error": str(exc)})
            results.append(
                {"index": idx, "status": "error", "identifier": identifier, "error": str(exc)}
            )
            continue

        trace = dict(trace)
        trace_id = trace["trace_id"]

        # --- Matchup Intelligence stamps (trace schema v2) ---
        # Identity is provider-anchored: the resolver-confirmed the-odds-api
        # event id when odds were resolved live, else the caller-supplied id
        # (same namespace, e.g. from omega_list_events). Game and prop traces
        # for the same event therefore share one event_key.
        trace["schema_version"] = 2
        trace["presentation_mode"] = presentation_mode
        trace["engine_auto_ledger_mode"] = engine_auto_ledger_mode
        if provider_event_id:
            trace["event_identity"] = EventIdentityV1(
                provider=EVENT_PROVIDER,
                provider_event_id=provider_event_id,
                event_key=EventIdentityV1.derive_event_key(
                    entry.league, EVENT_PROVIDER, provider_event_id
                ),
                league=entry.league,
                home_team=entry.home_team,
                away_team=entry.away_team,
                game_date=game_date,
                commence_time=resolved_commence_time,
            ).model_dump(mode="json")
        else:
            trace["event_identity"] = None
        if entry.decision_support_presentation is not None:
            trace["decision_support_presentation"] = (
                entry.decision_support_presentation.model_dump(mode="json")
            )

        # --- Build export block in the current nested trace-export shape. ---
        trace["reasoning_inputs"] = reasoning_inputs
        trace["reasoning_downgrade_rationale"] = (
            rsvg.reasoning_downgrade_rationale if rsvg is not None else None
        )
        trace["reasoning_narrative"] = entry.reasoning_narrative or f"Batch analysis: {identifier}"
        # Copy before mutating: the analyze() return may share nested dicts.
        trace_quality = dict(trace.get("trace_quality") or {})
        if rsvg is not None:
            trace_quality.update(rsvg.trace_quality_fragment())
        trace["trace_quality"] = trace_quality
        if entry.reasoning_presentation is not None:
            trace["reasoning_presentation"] = entry.reasoning_presentation.model_dump(exclude_none=True)
        elif rsvg is not None:
            trace["reasoning_presentation"] = rsvg.reasoning_presentation.model_dump(
                exclude_none=True
            )

        export_block = {
            "export_schema_version": 2,
            # Top-level session_id so the prediction->session link survives outside
            # the DB and the export-export validator's strict session_id check passes.
            "session_id": session_id,
            "trace": trace,
            "bet_record": None,
        }

        # --- Write to inbox (atomic: a crash mid-write must not leave a truncated
        # export in var/inbox/traces/ — truncation is the failure class the rest of
        # the pipeline already hardened against). ---
        dest = inbox_dir / f"{trace_id}.json"
        try:
            atomic_write_text(dest, _json.dumps(export_block, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {"index": idx, "identifier": identifier, "error": f"export_write_failed: {exc}"}
            )
            results.append(
                {
                    "index": idx,
                    "status": "error",
                    "identifier": identifier,
                    "trace_id": trace_id,
                    "error": str(exc),
                }
            )
            continue

        trace_ids.append(trace_id)
        export_paths.append(str(dest))
        ok_row: dict[str, Any] = {
            "index": idx,
            "status": "ok",
            "identifier": identifier,
            "trace_id": trace_id,
            "export_path": str(dest),
        }
        if rsvg is not None:
            ok_row["rsvg_status"] = rsvg.status
        results.append(ok_row)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")
    overall = "ok" if error_count == 0 else ("partial" if ok_count > 0 else "error")

    return _ok(
        "omega_run_batch",
        status=overall,
        entries_total=len(entries),
        entries_ok=ok_count,
        entries_skipped=skipped_count,
        entries_error=error_count,
        trace_ids=trace_ids,
        export_paths=export_paths,
        errors=errors,
        results=results,
    )


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
        "trace_quality": {
            "calibration_eligible": False,
            "calibration_exclusion_reasons": ["replay_plane_only"],
        },
    }
    return _ok(
        "omega_replay_bundle",
        result=response,
        response=response,
        trace_quality=response["trace_quality"],
    )


def omega_trace_get(trace_id: str, db_path: str | None = None) -> dict[str, Any]:
    """Retrieve a persisted trace via TraceStore."""
    from omega.trace.store import TraceStore, log_effective_db

    store = TraceStore(db_path=db_path)
    log_effective_db(store, logger)
    try:
        trace = store.get_trace(trace_id)
        if trace is None:
            return _error("omega_trace_get", "trace_not_found", trace_id)
        return _ok("omega_trace_get", trace=trace)
    except Exception as exc:
        return _error("omega_trace_get", "trace_get_failed", str(exc))
    finally:
        store.close()


def omega_get_matchup_brief(event_key: str, db_path: str | None = None) -> dict[str, Any]:
    """Safe decision-support brief for one event — the primary matchup view.

    ``event_key`` is either an ``EventIdentityV1.event_key`` (groups every game
    and prop trace stamped with it) or ``trace:<trace_id>`` for a legacy trace
    without provider identity (rendered as a singleton group with an identity
    warning). The response contains only the allowlisted decision-support DTO:
    listed lines, simulation distribution summaries, symmetric outcome
    probability sets where the market's own output_mode authorizes them, data
    quality and provenance notes. It never contains edge/EV/Kelly/stake/tier or
    any recommendation framing.
    """
    from omega.trace.decision_support import brief_for_group_key
    from omega.trace.store import TraceStore, log_effective_db

    store = TraceStore(db_path=db_path, read_only=True)
    log_effective_db(store, logger)
    try:
        brief = brief_for_group_key(store, event_key)
        if brief is None:
            return _error("omega_get_matchup_brief", "matchup_not_found", event_key)
        return _ok("omega_get_matchup_brief", brief=brief.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        return _error("omega_get_matchup_brief", "matchup_brief_failed", str(exc))
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
    from omega.trace.store import TraceStore, log_effective_db

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
    log_effective_db(store, logger)
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
    from omega.trace.store import TraceStore, log_effective_db

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
    log_effective_db(store, logger)
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


def omega_trace_void_prop(
    trace_id: str,
    player_name: str,
    stat_type: str,
    side: str = "over",
    reason: str = "dnp",
    source: str = "mcp",
    db_path: str | None = None,
) -> dict[str, Any]:
    """Record a DNP / no-action void for a player prop absent from the box score.

    Use when the player did not play (injury, scratch, ejection) so the prop has
    no gradeable stat line. Records a ``void`` prop outcome so settlement returns
    VOID (stake returned, net 0) rather than leaving the bet pending or grading
    it as a loss. Post-decision, like outcome attachment — computes no protected
    betting output.
    """
    from omega.trace.store import TraceStore, log_effective_db

    try:
        req = TraceVoidPropRequest(
            trace_id=trace_id,
            player_name=player_name,
            stat_type=stat_type,
            side=side,
            reason=reason,
            source=source,
            db_path=db_path,
        )
    except ValidationError as exc:
        return _error("omega_trace_void_prop", "invalid_request", exc.errors())

    store = TraceStore(db_path=req.db_path)
    log_effective_db(store, logger)
    try:
        # attach_prop_outcome is idempotent on (trace_id, player_name, stat_type):
        # if a row already exists it returns that row's id WITHOUT changing its
        # result. Detect a pre-existing non-void outcome first so we never report
        # a "void" that did not actually take effect (which would mislead the
        # caller into thinking a graded loss/win was converted to a no-action).
        existing = next(
            (
                r
                for r in store.get_prop_outcomes(req.trace_id)
                if r.get("player_name") == req.player_name and r.get("stat_type") == req.stat_type
            ),
            None,
        )
        if existing is not None and existing.get("result") != "void":
            return _error(
                "omega_trace_void_prop",
                "outcome_exists",
                {
                    "message": (
                        "A non-void prop outcome is already attached; refusing to "
                        "silently overwrite it. Detach/correct it first if the "
                        "player truly did not play."
                    ),
                    "existing_result": existing.get("result"),
                    "prop_outcome_id": existing.get("prop_outcome_id"),
                },
            )

        prop_outcome_id = store.attach_prop_outcome(
            trace_id=req.trace_id,
            player_name=req.player_name,
            stat_type=req.stat_type,
            stat_value=0.0,
            line=0.0,
            side=req.side,
            source=f"{req.source}:void:{req.reason}",
            void=True,
        )
        return _ok(
            "omega_trace_void_prop",
            prop_outcome_id=prop_outcome_id,
            result="void",
            reason=req.reason,
        )
    except ValueError as exc:
        return _error("omega_trace_void_prop", "trace_not_found", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error("omega_trace_void_prop", "void_failed", str(exc))
    finally:
        store.close()


def omega_fetch_outcomes(
    leagues: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Batch-gather outcomes across leagues (wraps fetch_outcomes_all).

    Dispatches the idempotent per-league outcome fetchers and returns per-league
    status. Defaults to all leagues; to exclude soccer (future-dated fixtures),
    pass ``leagues`` without ``"soccer"``. ``dry_run=True`` reports what would run
    without attaching anything. Delegates entirely to the deterministic ops layer
    — computes no protected betting output.
    """
    try:
        req = FetchOutcomesRequest(
            leagues=leagues,
            since=since,
            until=until,
            dry_run=dry_run,
            db_path=db_path,
        )
    except ValidationError as exc:
        return _error("omega_fetch_outcomes", "invalid_request", exc.errors())

    try:
        from omega.ops.fetch_outcomes_all import run_fetch_outcomes

        outcome = run_fetch_outcomes(
            leagues=req.leagues,
            db=req.db_path,
            since=req.since,
            until=req.until,
            dry_run=req.dry_run,
            capture_output=True,
        )
        return _ok("omega_fetch_outcomes", **outcome)
    except Exception as exc:  # noqa: BLE001
        return _error("omega_fetch_outcomes", "fetch_outcomes_failed", str(exc))


def omega_settle_bets(
    apply: bool = False,
    league: str | None = None,
    sport: str | None = None,
    provenance: str = "user_confirmed",
    start: str | None = None,
    end: str | None = None,
    limit: int = 100000,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Settle pending bet_ledger rows with attached outcomes (wraps settle_bets).

    ``apply=False`` (default) is a dry run that scans and reports counts/PnL
    without writing. Delegates to the deterministic settlement op — owns no
    grading math itself.
    """
    from omega.trace.ledger_settlement import settle_pending_ledger
    from omega.trace.store import TraceStore, log_effective_db

    try:
        req = SettleBetsRequest(
            apply=apply,
            league=league,
            sport=sport,
            provenance=provenance,
            start=start,
            end=end,
            limit=limit,
            db_path=db_path,
        )
    except ValidationError as exc:
        return _error("omega_settle_bets", "invalid_request", exc.errors())

    provenance_filter = None if req.provenance == "all" else req.provenance
    store = TraceStore(db_path=req.db_path)
    log_effective_db(store, logger)
    try:
        summary = settle_pending_ledger(
            store,
            apply=req.apply,
            league=req.league,
            sport=req.sport,
            provenance=provenance_filter,
            start=req.start,
            end=req.end,
            limit=req.limit,
        )
        staked = summary.total_staked
        roi = (summary.total_net / staked * 100.0) if staked else 0.0
        return _ok(
            "omega_settle_bets",
            applied=req.apply,
            provenance=req.provenance,
            pending_scanned=summary.pending_scanned,
            settled=dict(summary.settled),
            settled_total=sum(summary.settled.values()),
            ungradeable=summary.ungradeable,
            total_staked=round(staked, 2),
            total_net=round(summary.total_net, 2),
            roi_pct=round(roi, 2),
        )
    except Exception as exc:  # noqa: BLE001
        return _error("omega_settle_bets", "settle_failed", str(exc))
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
    from omega.core.calibration.market import calibration_market_for_plane
    from omega.core.calibration.registry import CalibrationRegistry
    from omega.trace.store import TraceStore, log_effective_db

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
    log_effective_db(store, logger)
    try:
        graded = store.get_graded_traces(league=req.league, limit=req.limit)
        fitter = CalibrationFitter()
        if req.plane == "prop":
            predictions, outcomes = fitter.extract_prop_pairs(graded)
            pair_label = "prop probability/outcome"
            calibration_market = calibration_market_for_plane(req.plane)
        else:
            predictions, outcomes = fitter.extract_pairs(graded)
            pair_label = "home_win_prob/outcome"
            calibration_market = calibration_market_for_plane(req.plane)
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
                    "market": calibration_market,
                    "pair_type": pair_label,
                    "method": req.method,
                },
            )

        fit_league = (req.league or "UNIVERSAL").upper()
        candidate = (
            fitter.fit_isotonic(predictions, outcomes, league=fit_league, market=calibration_market)
            if req.method == "isotonic"
            else fitter.fit_shrinkage(
                predictions,
                outcomes,
                league=fit_league,
                market=calibration_market,
                eligible_sample_size=len(predictions),
            )
        )
        metrics = fitter.evaluate(candidate, predictions, outcomes)
        candidate.metrics = metrics

        incumbent = CalibrationRegistry().get_production(
            fit_league,
            market=calibration_market,
        )
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
                "market": calibration_market,
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
        from omega.integrations.odds_resolver import resolve_odds

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


def omega_list_events(
    league: str,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
) -> dict[str, Any]:
    """List today's events for a league from the Odds API.

    Returns event IDs, home/away team names, and commence times suitable
    for use as inputs to omega_resolve_odds or omega_run_batch. For tennis
    tours (ATP, WTA, GRAND_SLAM) provider sport keys are resolved dynamically
    per active tournament.
    Does not compute probabilities, edge, EV, Kelly, units, tiers, or trace IDs.
    """
    try:
        from omega.integrations.odds_resolver import list_events

        result = list_events(
            league=league,
            commence_time_from=commence_time_from,
            commence_time_to=commence_time_to,
        )
        return _ok("omega_list_events", result=result)
    except Exception as exc:  # noqa: BLE001
        return _error("omega_list_events", "event_listing_failed", str(exc))


def omega_record_flat_bet(
    trace_id: str,
    market: str,
    side: str,
    odds: float,
    line: float | None = None,
    bookmaker: str = "betmgm",
    stake_amount: float = 25.0,
    selection: str | None = None,
    player_name: str | None = None,
    prop_type: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Log a flat dollar wager into bet_ledger, tied to an existing trace.

    Bookkeeping / input-prep only — this does NOT compute probability, edge, EV,
    Kelly, units, tiers, or grades. The bet is written with provenance
    'user_confirmed' and lands 'pending'; the outcome + regrade pipelines settle
    it (dollar PnL) later.

    `market`   moneyline | spread | total | player_prop (or player_prop:<stat>)
    `side`     home | away | draw | over | under (required so the bet is gradeable)
    `line`     point/total; None for moneyline
    """
    from omega.trace.bet_settlement import build_selection_descriptor, coerce_american_odds
    from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
    from omega.trace.store import TraceStore, log_effective_db

    try:
        req = FlatBetRequest(
            trace_id=trace_id,
            market=market,
            side=side,
            odds=odds,
            line=line,
            bookmaker=bookmaker,
            stake_amount=stake_amount,
            selection=selection,
            player_name=player_name,
            prop_type=prop_type,
            db_path=db_path,
        )
    except ValidationError as exc:
        return _error("omega_record_flat_bet", "invalid_request", exc.errors())
    except ValueError as exc:
        return _error("omega_record_flat_bet", "invalid_request", str(exc))

    american = coerce_american_odds(req.odds)
    if american is None:
        return _error(
            "omega_record_flat_bet",
            "invalid_request",
            f"odds={req.odds!r} is not valid American odds",
        )

    descriptor, label = build_selection_descriptor(
        req.market,
        req.side,
        line=req.line,
        player=req.player_name,
        stat=req.prop_type,
    )

    store = TraceStore(db_path=req.db_path)
    log_effective_db(store, logger)
    try:
        trace = store.get_trace(req.trace_id)
        if trace is None:
            return _error("omega_record_flat_bet", "trace_not_found", req.trace_id)
        ts = str(trace.get("timestamp") or "")
        league = trace.get("league")
        sport = None
        if league:
            try:
                from omega.core.config.leagues import get_league_config

                sport = get_league_config(str(league)).get("sport")
            except Exception:  # noqa: BLE001 - sport is a slice nicety, never fatal
                sport = None
        bet = LedgerBet(
            ledger_id=uuid.uuid4().hex[:12],
            trace_id=req.trace_id,
            bet_date=ts[:10] if len(ts) >= 10 else None,
            league=league,
            sport=sport,
            matchup=trace.get("matchup") or "",
            market=req.market,
            bookmaker=req.bookmaker,
            selection=req.selection or label,
            selection_descriptor=descriptor,
            line=req.line,
            odds=american,
            stake_amount=req.stake_amount,
            status=LedgerStatus.PENDING,
            provenance=BetProvenance.USER_CONFIRMED,
            decision_timestamp=ts or datetime.now(timezone.utc).isoformat(),
        )
        store.record_ledger_bet(bet)
        # Idempotent insert: resolve the stored row by its idempotency key so the
        # response reflects reality whether this call inserted or matched an existing row.
        rows = store.get_ledger_bets(req.trace_id)
        recorded = next(
            (
                r
                for r in rows
                if r.get("market") == req.market and r.get("selection_descriptor") == descriptor
            ),
            None,
        )
        ledger_id = recorded["ledger_id"] if recorded else bet.ledger_id
        return _ok(
            "omega_record_flat_bet",
            ledger_id=ledger_id,
            already_existed=bool(recorded and recorded["ledger_id"] != bet.ledger_id),
            bet=recorded,
        )
    except ValueError as exc:
        return _error("omega_record_flat_bet", "trace_not_found", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error("omega_record_flat_bet", "record_failed", str(exc))
    finally:
        store.close()


def omega_get_portfolio_summary(
    league: str | None = None,
    sport: str | None = None,
    start: str | None = None,
    end: str | None = None,
    base_bankroll: float = 1000.0,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Return the financial state of the bet_ledger: bankroll, pending stakes,
    ROI, net PnL, win%. Read-only aggregation over stored dollar PnL — computes
    no protected betting output."""
    from omega.trace.portfolio import summarize_ledger
    from omega.trace.store import TraceStore, log_effective_db

    try:
        req = PortfolioSummaryRequest(
            league=league,
            sport=sport,
            start=start,
            end=end,
            base_bankroll=base_bankroll,
            db_path=db_path,
        )
    except ValidationError as exc:
        return _error("omega_get_portfolio_summary", "invalid_request", exc.errors())

    store = TraceStore(db_path=req.db_path)
    log_effective_db(store, logger)
    try:
        rows = store.query_ledger(
            league=req.league,
            sport=req.sport,
            start=req.start,
            end=req.end,
            limit=100000,
        )
        summary = summarize_ledger(rows, base_bankroll=req.base_bankroll)
        return _ok(
            "omega_get_portfolio_summary",
            as_of=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            filters={
                "league": req.league,
                "sport": req.sport,
                "start": req.start,
                "end": req.end,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _error("omega_get_portfolio_summary", "summary_failed", str(exc))
    finally:
        store.close()


def omega_get_game_context(
    league: str,
    home_team: str,
    away_team: str,
    game_date: str,
    lookback_days: int = 5,
) -> dict[str, Any]:
    """Resolve the situational context pack for a matchup.

    Returns deterministic game_context keys analyze() consumes (rest_days /
    is_b2b_* / is_playoff / park_factor where derivable), the applicable
    EvidenceSignal worksheet for the sport, and suggested evidence for semantic
    context with no wired data source. Input-prep only — splice
    result['game_context'] into an analyze request. Computes no protected output.
    """
    try:
        req = GameContextRequest(
            league=league,
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            lookback_days=lookback_days,
        )
    except ValidationError as exc:
        return _error("omega_get_game_context", "invalid_request", exc.errors())

    try:
        from omega.integrations.game_context import resolve_game_context

        result = resolve_game_context(
            league=req.league,
            home_team=req.home_team,
            away_team=req.away_team,
            game_date=req.game_date,
            lookback_days=req.lookback_days,
        )
        return _ok("omega_get_game_context", result=result)
    except Exception as exc:  # noqa: BLE001
        return _error("omega_get_game_context", "game_context_failed", str(exc))


def omega_propose_signal(
    name: str,
    feature_combo: dict[str, Any],
    thesis: str = "",
    plane: str = "both",
    direction_rule: str | None = None,
    source: str = "llm",
    db_path: str | None = None,
) -> dict[str, Any]:
    """Propose a new evidence-signal hypothesis (issue #28 WS3).

    A proposal is a typed hypothesis over a WHITELISTED feature vocabulary — never
    code. It is stored in ``lifecycle='probation'``: scored by CLV like an active
    signal but NEVER applied to a prediction, and graduates to ``active`` only
    through the operator-gated CLV + marginal bar. ``feature_combo`` is a dict
    spec — a ``predicate`` (boolean AND/OR/NOT over threshold comparisons -> a
    factor) or a ``linear`` (bias + sum of weight*feature); see
    ``feature_combo_eval.FEATURE_WHITELIST`` for the allowed feature names.
    Returns the stored proposal, or a validation error for a bad spec / off-
    whitelist feature / name collision with a built-in signal_type.
    """
    from omega.core.simulation.feature_combo_eval import FeatureComboError
    from omega.ops.propose_signal import propose_signal as _propose

    try:
        result = _propose(
            name=name,
            feature_combo=feature_combo or {},
            thesis=thesis,
            plane=plane,
            direction_rule=direction_rule,
            source=source,
            db_path=db_path,
        )
    except (ValueError, FeatureComboError) as exc:
        return _error("omega_propose_signal", "invalid_proposal", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error("omega_propose_signal", "proposal_failed", str(exc))
    return _ok("omega_propose_signal", **result)


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
        omega_run_batch,
        omega_chat_orchestrate,
        omega_replay_bundle,
        omega_trace_get,
        omega_get_matchup_brief,
        omega_trace_query,
        omega_trace_attach_outcome,
        omega_trace_void_prop,
        omega_fetch_outcomes,
        omega_settle_bets,
        omega_calibration_fit_preview,
        omega_evidence_retrieve,
        omega_list_events,
        omega_resolve_odds,
        omega_record_flat_bet,
        omega_get_portfolio_summary,
        omega_get_game_context,
        omega_propose_signal,
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
    from omega.core.simulation.evidence_to_modifier import (
        build_markov_vocabulary_table,  # noqa: PLC0415
    )

    # Operator-approved lifecycle overrides (issue #28 WS3) filter the vocabulary:
    # deprecated/rejected signals drop out so the agent stops emitting them. Best-
    # effort — the guide must still render if no production policy exists.
    overrides: dict[str, str] | None = None
    prod_mode: str | None = None
    try:
        from omega.core.calibration.adjustment_policy import (  # noqa: PLC0415
            AdjustmentPolicyRegistry,
        )

        prod = AdjustmentPolicyRegistry().get_production_policy()
        if prod is not None:
            prod_mode = prod.mode
            if prod.signal_lifecycle:
                overrides = prod.signal_lifecycle
    except Exception:  # noqa: BLE001 — never let registry access break the guide
        overrides = None

    header = (
        "=== Omega Markov Evidence Guide ===\n\n"
        "Use simulation_backend='markov_state' when you have game-level structural\n"
        "evidence (pace, rest, matchup) that should shift the possession-level model.\n\n"
    )
    if prod_mode in ("bounded_live", "live"):
        mode_note = (
            f"Evidence-application note: the production adjustment policy mode is\n"
            f"'{prod_mode}', so these modifiers DO affect live predictions. Each is\n"
            "applied under hard per-signal/family/plane caps and scaled by the\n"
            "signal's reliability weight (an unproven signal moves the prediction\n"
            "only a sliver). Evidence still cannot lift confidence to A until the\n"
            "policy's evidence metrics pass the promotion gate."
        )
    else:
        mode_note = (
            f"Evidence-application note: the production adjustment policy mode is\n"
            f"'{prod_mode or 'score_only'}', so these modifiers are computed and\n"
            "recorded for audit/learning but do NOT affect live predictions until an\n"
            "operator advances the mode (e.g. to 'bounded_live')."
        )
    footer = (
        "\n\nWorkflow:\n"
        "  1. Assess which of the 8 types above apply to this matchup.\n"
        "  2. Set direction='home' or direction='away' for all directional signals.\n"
        "  3. Set confidence honestly (scored retrospectively against outcomes).\n"
        "  4. Call omega_analyze_game with simulation_backend='markov_state'.\n"
        "  5. The engine applies modifiers automatically -- do NOT pre-adjust\n"
        "     home_context or away_context ratings by hand.\n\n" + mode_note
    )
    return header + build_markov_vocabulary_table(overrides) + footer


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
    root = repo_root()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"Refusing to read outside repo: {path}")
    return target.read_text(encoding="utf-8")


def _insulate_stdio_backend() -> None:
    """Keep the Antigravity stdio server on SQLite unless explicitly opted in.

    The stdio server is spawned by ``.mcp.json`` and inherits the parent shell's
    environment. A developer who exported ``DATABASE_URL`` for Postgres work would
    otherwise silently flip Antigravity's stdio server to Postgres. We require a
    positive opt-in (``OMEGA_MCP_ALLOW_DB_BACKEND=1``) for the stdio entrypoint;
    the HTTP server, CLI tools, migration tool, and tests still honor DATABASE_URL.
    """
    if os.environ.get("OMEGA_MCP_ALLOW_DB_BACKEND") == "1":
        return
    if os.environ.pop("DATABASE_URL", None):
        print(
            "omega.mcp.server: ignoring inherited DATABASE_URL for the stdio "
            "server (set OMEGA_MCP_ALLOW_DB_BACKEND=1 to honor it).",
            file=sys.stderr,
        )


def main() -> None:
    _insulate_stdio_backend()
    build_server().run()


if __name__ == "__main__":
    main()
