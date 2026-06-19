"""Persistence + fail-closed promotion for backend parameter profiles (Phase 8).

Storage and lifecycle for :class:`~omega.core.simulation.parameter_profile.BackendParameterProfile`,
mirroring how :class:`~omega.core.calibration.registry.CalibrationRegistry` owns
calibration-profile storage and promotion. The model stays pure; SQL lives here.

The single promotion path, :func:`promote_parameter_profile`, composes the shared
:func:`omega.core.governance.promotion_gates.evaluate_promotion_gates` engine — the
*same* gates, thresholds, and evidence discipline that gate calibration profiles —
so backend-parameter promotion is fail-closed and cannot drift from calibration
promotion. There is no ``--force`` bypass: a profile reaches PRODUCTION only when
every gate passes, and exactly one profile per ``(backend_name,
competition_bucket)`` is PRODUCTION at a time (enforced by the partial unique
index ``uq_parameter_profiles_production``).

These helpers only touch ``store.conn``; they add no state to ``TraceStore``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from omega.core.governance.promotion_gates import (
    DEFAULT_BRIER_IMPROVEMENT,
    DEFAULT_ECE_FLOOR,
    DEFAULT_LOG_LOSS_TOL,
    DEFAULT_MIN_SAMPLES,
    GateReport,
    PromotionGateError,
    evaluate_promotion_gates,
)
from omega.core.simulation.parameter_profile import (
    BackendParameterProfile,
    ParameterProfileStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omega.trace.store import TraceStore

UTC = timezone.utc

_COLUMNS = (
    "profile_id, schema_version, version, backend_name, backend_component_version, "
    "competition_bucket, params_json, priors_as_of_date, dataset_manifest_id, "
    "dataset_hash, sample_size, metrics_json, status, incumbent_id, "
    "promotion_gate_report, created_at, promoted_at, rejected_at, reject_reason"
)


def _row_to_profile(row: Any) -> BackendParameterProfile:
    return BackendParameterProfile(
        profile_id=row[0],
        schema_version=row[1],
        version=row[2],
        backend_name=row[3],
        backend_component_version=row[4],
        competition_bucket=row[5],
        params=json.loads(row[6]) if row[6] else {},
        priors_as_of_date=row[7],
        dataset_manifest_id=row[8],
        dataset_hash=row[9],
        sample_size=row[10],
        metrics=json.loads(row[11]) if row[11] else {},
        status=ParameterProfileStatus(row[12]),
        incumbent_id=row[13],
        promotion_gate_report=json.loads(row[14]) if row[14] else None,
        created_at=row[15],
        promoted_at=row[16],
        rejected_at=row[17],
        reject_reason=row[18],
    )


def register_parameter_profile(store: TraceStore, profile: BackendParameterProfile) -> None:
    """Insert (or refresh) a CANDIDATE parameter profile.

    Promotion is the only path to PRODUCTION, so a candidate is all this writes.
    Re-registering an existing ``profile_id`` refreshes it ONLY while it is still
    a candidate; a promoted/archived/rejected profile is immutable (fails loud)
    so history and the frozen-production invariant cannot be clobbered.
    """
    if profile.status != ParameterProfileStatus.CANDIDATE:
        raise ValueError(
            f"register_parameter_profile writes candidates only; got status="
            f"{profile.status.value!r}. Use promote_parameter_profile() to reach production."
        )
    existing = get_parameter_profile(store, profile.profile_id)
    if existing is not None and existing.status != ParameterProfileStatus.CANDIDATE:
        raise ValueError(
            f"parameter profile {profile.profile_id!r} is {existing.status.value!r}; "
            "promoted/archived/rejected profiles are immutable"
        )
    store.conn.execute(
        f"""INSERT INTO parameter_profiles ({_COLUMNS})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (profile_id) DO UPDATE SET
                schema_version = excluded.schema_version,
                version = excluded.version,
                backend_name = excluded.backend_name,
                backend_component_version = excluded.backend_component_version,
                competition_bucket = excluded.competition_bucket,
                params_json = excluded.params_json,
                priors_as_of_date = excluded.priors_as_of_date,
                dataset_manifest_id = excluded.dataset_manifest_id,
                dataset_hash = excluded.dataset_hash,
                sample_size = excluded.sample_size,
                metrics_json = excluded.metrics_json""",
        (
            profile.profile_id,
            profile.schema_version,
            profile.version,
            profile.backend_name,
            profile.backend_component_version,
            profile.competition_bucket,
            json.dumps(profile.params, sort_keys=True),
            profile.priors_as_of_date,
            profile.dataset_manifest_id,
            profile.dataset_hash,
            profile.sample_size,
            json.dumps(profile.metrics, sort_keys=True),
            ParameterProfileStatus.CANDIDATE.value,
            profile.incumbent_id,
            None,
            profile.created_at,
            None,  # promoted_at — a freshly registered candidate carries no lifecycle stamps
            None,  # rejected_at
            None,  # reject_reason
        ),
    )
    store.conn.commit()


def get_parameter_profile(store: TraceStore, profile_id: str) -> BackendParameterProfile | None:
    """Return one profile by id, or None."""
    row = store.conn.execute(
        f"SELECT {_COLUMNS} FROM parameter_profiles WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()
    return _row_to_profile(row) if row is not None else None


def get_production_parameter_profile(
    store: TraceStore, backend_name: str, competition_bucket: str
) -> BackendParameterProfile | None:
    """Return the production profile for a backend+bucket, or None (fail closed).

    The gatherer calls this to inject ``params`` into ``request.prior_payload``;
    None means the backend runs without an injected profile (fail-closed skip for
    backends that require one, exactly like a missing rho prior today).
    """
    row = store.conn.execute(
        f"""SELECT {_COLUMNS} FROM parameter_profiles
            WHERE backend_name = ? AND competition_bucket = ? AND status = ?""",
        (backend_name, competition_bucket, ParameterProfileStatus.PRODUCTION.value),
    ).fetchone()
    return _row_to_profile(row) if row is not None else None


def list_parameter_profiles(
    store: TraceStore,
    backend_name: str | None = None,
    competition_bucket: str | None = None,
    status: ParameterProfileStatus | str | None = None,
) -> list[BackendParameterProfile]:
    """Return profiles matching the optional filters, newest first."""
    clauses: list[str] = []
    params: list[Any] = []
    if backend_name is not None:
        clauses.append("backend_name = ?")
        params.append(backend_name)
    if competition_bucket is not None:
        clauses.append("competition_bucket = ?")
        params.append(competition_bucket)
    if status is not None:
        clauses.append("status = ?")
        params.append(status.value if isinstance(status, ParameterProfileStatus) else status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = store.conn.execute(
        f"SELECT {_COLUMNS} FROM parameter_profiles{where} ORDER BY created_at DESC",
        params,
    ).fetchall()
    return [_row_to_profile(r) for r in rows]


def promote_parameter_profile(
    store: TraceStore,
    profile_id: str,
    *,
    confirm_backtest_parity: bool = False,
    confirm_clv_non_regression: bool = False,
    parity_evidence: dict[str, Any] | None = None,
    clv_evidence: dict[str, Any] | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    brier_improvement: float = DEFAULT_BRIER_IMPROVEMENT,
    log_loss_tol: float = DEFAULT_LOG_LOSS_TOL,
    ece_floor: float = DEFAULT_ECE_FLOOR,
) -> GateReport:
    """Promote a candidate parameter profile to PRODUCTION, fail-closed.

    Evaluates the shared promotion gates against the current production profile
    for the same ``(backend_name, competition_bucket)`` (the incumbent). Raises
    :class:`PromotionGateError` unless every gate passes — there is no bypass. On
    success the incumbent is archived and the candidate flips to PRODUCTION with a
    full ``promotion_gate_report`` recorded for audit.

    Gates evaluate the candidate's RAW (pre-calibration) metrics: a backend's
    structural parameters must clear the ECE floor on their *uncalibrated* output.
    """
    candidate = get_parameter_profile(store, profile_id)
    if candidate is None:
        raise ValueError(f"no parameter profile {profile_id!r}; register it first")
    if candidate.status == ParameterProfileStatus.PRODUCTION:
        raise ValueError(f"parameter profile {profile_id!r} is already production")
    if candidate.status != ParameterProfileStatus.CANDIDATE:
        raise ValueError(
            f"parameter profile {profile_id!r} is {candidate.status.value!r}; "
            "only candidates can be promoted"
        )

    incumbent = get_production_parameter_profile(
        store, candidate.backend_name, candidate.competition_bucket
    )
    report = evaluate_promotion_gates(
        candidate,
        incumbent,
        min_samples=min_samples,
        brier_improvement=brier_improvement,
        log_loss_tol=log_loss_tol,
        ece_floor=ece_floor,
        confirm_backtest_parity=confirm_backtest_parity,
        confirm_clv_non_regression=confirm_clv_non_regression,
        parity_evidence=parity_evidence,
        clv_evidence=clv_evidence,
    )
    if not report.passed:
        raise PromotionGateError(report)

    now = datetime.now(UTC).isoformat()
    # Archive the incumbent FIRST so the single-production partial unique index is
    # never momentarily violated when the candidate flips to production.
    if incumbent is not None and incumbent.profile_id != candidate.profile_id:
        store.conn.execute(
            "UPDATE parameter_profiles SET status = ? WHERE profile_id = ?",
            (ParameterProfileStatus.ARCHIVED.value, incumbent.profile_id),
        )
    store.conn.execute(
        """UPDATE parameter_profiles
           SET status = ?, promoted_at = ?, incumbent_id = ?, promotion_gate_report = ?
           WHERE profile_id = ?""",
        (
            ParameterProfileStatus.PRODUCTION.value,
            now,
            incumbent.profile_id if incumbent else None,
            json.dumps(report.to_dict()),
            candidate.profile_id,
        ),
    )
    store.conn.commit()
    return report


def reject_parameter_profile(
    store: TraceStore, profile_id: str, reason: str
) -> BackendParameterProfile:
    """Mark a candidate profile REJECTED (failed eval / manual). Idempotent-safe."""
    candidate = get_parameter_profile(store, profile_id)
    if candidate is None:
        raise ValueError(f"no parameter profile {profile_id!r}")
    if candidate.status == ParameterProfileStatus.REJECTED:
        return candidate  # idempotent: keep the original rejected_at + reason
    if candidate.status != ParameterProfileStatus.CANDIDATE:
        raise ValueError(
            f"parameter profile {profile_id!r} is {candidate.status.value!r}; only candidates "
            "can be rejected"
        )
    store.conn.execute(
        "UPDATE parameter_profiles SET status = ?, rejected_at = ?, reject_reason = ? "
        "WHERE profile_id = ?",
        (ParameterProfileStatus.REJECTED.value, datetime.now(UTC).isoformat(), reason, profile_id),
    )
    store.conn.commit()
    result = get_parameter_profile(store, profile_id)
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Trace-level parameter-profile provenance (P8.0.3)
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _candidate_result_dicts(trace: dict[str, Any]):
    """Yield dicts within a trace that may carry simulation provenance fields.

    The persisted simulation result can sit at a few nesting points depending on
    the caller (top level, ``result``, ``simulation``, ``result.simulation``), so
    the extractor searches each rather than assuming one shape.
    """
    yield trace
    for key in ("result", "simulation"):
        val = trace.get(key)
        if isinstance(val, dict):
            yield val
            inner = val.get("simulation")
            if isinstance(inner, dict):
                yield inner


def extract_parameter_profile_ref(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the governed parameter-profile provenance ref from a trace dict.

    Resolution order:

    1. An explicit ``parameter_profile_ref`` dict stamped by the gatherer when it
       injects a governed :class:`BackendParameterProfile` (P8.2+), at the top
       level or on the simulation result.
    2. Legacy soccer Dixon-Coles provenance the soccer backend already echoes onto
       its result (``rho_profile_id`` / ``rho_as_of_date``), synthesized into a
       partial ref so today's rho provenance becomes queryable with no backend
       change.

    Returns None when a trace carries no governed-parameter provenance (the
    ``freshness=unpinned`` case).
    """
    if not isinstance(trace, dict):
        return None
    for d in _candidate_result_dicts(trace):
        ref = d.get("parameter_profile_ref")
        if isinstance(ref, dict) and ref:
            return ref
    for d in _candidate_result_dicts(trace):
        rho_profile = d.get("rho_profile_id")
        if rho_profile:
            return {
                "backend_name": d.get("backend_name"),
                "backend_component_version": d.get("component_version"),
                "param_profile_id": rho_profile,
                "priors_as_of_date": d.get("rho_as_of_date"),
                "source": "legacy_dc_rho",
            }
    return None


def get_parameter_profile_ref(store: TraceStore, trace_id: str) -> dict[str, Any] | None:
    """Return the stored parameter_profile_ref for a trace, or None (unpinned)."""
    row = store.conn.execute(
        "SELECT parameter_profile_ref FROM traces WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return json.loads(row[0])


def parameter_pin_status(
    store: TraceStore, trace_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return ``(ref, data_provenance_event)`` for a replay/lab pin attempt.

    A trace with a stored ref is *pinned*: any re-derivation of its parameters must
    use the ref, not a live-table read, so replay stays reproducible. A trace
    without one is *unpinned* (it predates parameter-profile governance or used an
    ungoverned backend): the event is a loud ``freshness=unpinned`` warning so a
    non-reproducible live re-read is never silent. Consumed by the lab (P8.1) and
    the soccer pilot (P8.2); the event shape matches the gatherer's other
    ``data_provenance`` sidecar events.
    """
    ref = get_parameter_profile_ref(store, trace_id)
    if ref is not None:
        return ref, {
            "ts": _utc_now_iso(),
            "event_type": "data_provenance",
            "step": "parameter_profile:pin",
            "status": "ok",
            "notes": (
                f"pinned trace {trace_id} to param_profile_id={ref.get('param_profile_id')!r}"
            ),
            "outputs": {"parameter_profile_ref": ref},
        }
    return None, {
        "ts": _utc_now_iso(),
        "event_type": "data_provenance",
        "step": "parameter_profile:pin",
        "status": "warn",
        "notes": (
            f"trace {trace_id} has no parameter_profile_ref (freshness=unpinned); it "
            "predates parameter-profile governance or used an ungoverned backend, so any "
            "re-derivation reads live priors tables and is not guaranteed reproducible"
        ),
        "outputs": {"freshness": "unpinned"},
    }
