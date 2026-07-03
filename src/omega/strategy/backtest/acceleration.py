"""Issue #27 — backtest acceleration lane (replay rows, output cache, CRN).

This module is a *view layer over* the FrozenArtifact boundary, never a second
replay source: every :class:`HistoricalReplayRow` is derived FROM a
:class:`FrozenArtifact` (``replay_rows_from_artifact``) and converts back into
one (``artifact_from_replay_rows``) so the one BacktestEngine keeps consuming
the one artifact family. Three primitives land in this slice:

1. **HistoricalReplayRow** — one market-side decision flattened out of an
   artifact, carrying decision-time provenance (line/odds, contexts, seed,
   backend/substrate identity, calibration policy) with fail-closed lookahead
   guards: rows refuse to build without decision-time odds/context provenance,
   and post-outcome fields can never enter the pre-decision model inputs.
2. **SimulationOutputCache** — an in-process, deterministic-only cache of RAW
   simulation outputs. The key covers backend/component/model version, league,
   market, line, the full input-context hash, evidence policy, and calibration
   ref, so nothing is ever reused across a backend, calibration, or
   evidence-policy change. In-memory only: a mutable cache must never serve as
   historical truth (see strategy CLAUDE.md), so nothing persists past the run.
3. **crn_seed** — common-random-number seeds for MC-only paths: stable per
   replay row, shared across paired candidates that pass the same ``model_key``
   and decorrelated across different model/backend versions. Exact evaluation
   stays the preferred substrate — CRN only shapes the sampling fallback and
   never displaces ``exact_eval``.

Backlog (deliberately NOT implemented here): importance sampling over replay
rows and surrogate model paths. Both are expected to build on these primitives
(rows as the unit of work, the cache for reuse, CRN for paired variance
reduction) rather than introducing a new input family.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from omega.strategy.artifacts import FrozenArtifact


class ReplayIntegrityError(ValueError):
    """A replay row could not be derived without violating historical validity."""


class MissingDecisionProvenance(ReplayIntegrityError):
    """The artifact lacks decision-time odds/context provenance the row requires."""


class LookaheadContamination(ReplayIntegrityError):
    """Post-outcome data was found inside pre-decision model inputs."""


# Keys that can only exist after the event settled. Finding one inside a
# decision-time context means the artifact was contaminated upstream; rows
# refuse to build rather than let a lookahead leak into the model inputs.
_POST_OUTCOME_KEYS = frozenset(
    {"home_score", "away_score", "result", "outcome", "final_score", "winner", "settled"}
)


def _assert_pre_decision(mapping: dict[str, Any], where: str) -> None:
    """Recursively reject post-outcome keys inside a pre-decision mapping."""
    for key, value in mapping.items():
        if str(key).lower() in _POST_OUTCOME_KEYS:
            raise LookaheadContamination(
                f"post-outcome field {key!r} found in pre-decision {where}; "
                "refusing to build replay rows from a contaminated artifact"
            )
        if isinstance(value, dict):
            _assert_pre_decision(value, f"{where}.{key}")


class HistoricalReplayRow(BaseModel):
    """One market-side decision derived from a :class:`FrozenArtifact`.

    The flat unit of work for the acceleration lane (caching, CRN pairing, and
    later importance sampling): everything needed to reprice ONE offered side is
    on the row, and rows regroup losslessly into the artifact they came from.
    Derived only via :func:`replay_rows_from_artifact` — never built from traces
    or provider data directly (that would be a parallel replay source).
    """

    schema_version: int = 1

    # Identity (the parent artifact's event identity)
    artifact_id: str = Field(description="FrozenArtifact.artifact_id this row derives from")
    source_trace_id: str | None = None
    home_team: str
    away_team: str
    league: str
    date: str = Field(description="YYYY-MM-DD decision date (artifact.date)")

    # The offered decision
    market: str = Field(description="moneyline | spread | total")
    side: str = Field(description="home | away | draw | over | under")
    line: float | None = Field(
        default=None, description="Offered line for spread/total sides; None for moneyline"
    )
    offered_odds: float = Field(description="Decision-time American odds for this side")
    closing_odds: float | None = Field(
        default=None, description="Closing American odds for this side (CLV), when captured"
    )

    # Decision-time model inputs (pre-decision ONLY; guarded on derivation)
    model_inputs: dict[str, Any] = Field(
        description="home_context / away_context / game_context exactly as the sim saw them"
    )
    # Full decision-time snapshots so ``artifact_from_replay_rows`` is lossless
    # even for markets this slice does not enumerate (soccer exotics etc.).
    odds_snapshot: dict[str, Any] = Field(description="The artifact's full decision-time odds")
    closing_snapshot: dict[str, Any] | None = None

    # Determinism + substrate identity
    simulation_seed: int
    simulation_backend: str | None = None
    prior_payload: dict[str, Any] | None = Field(
        default=None,
        description="Decision-time priors incl. parameter_profile_ref (substrate identity)",
    )
    substrate_unresolved: bool = False
    calibration_policy: str = "static_v1"

    # Grading data — attached alongside, NEVER part of model_inputs
    outcome: dict[str, Any] | None = None


# (market, side, price key, line key, line sign) — the same core market/side set
# the BacktestEngine evaluates. Exotic soccer markets are not enumerated as rows
# in this slice; they survive round-trips via ``odds_snapshot``.
_ROW_SPECS: tuple[tuple[str, str, str, str | None, float], ...] = (
    ("moneyline", "home", "moneyline_home", None, 1.0),
    ("moneyline", "away", "moneyline_away", None, 1.0),
    ("moneyline", "draw", "moneyline_draw", None, 1.0),
    ("spread", "home", "spread_home_price", "spread_home", 1.0),
    ("spread", "away", "spread_away_price", "spread_home", -1.0),
    ("total", "over", "total_over_price", "over_under", 1.0),
    ("total", "under", "total_under_price", "over_under", 1.0),
)


def replay_rows_from_artifact(
    artifact: FrozenArtifact,
    *,
    require_decision_provenance: bool = True,
) -> list[HistoricalReplayRow]:
    """Derive the priced market-side rows from one FrozenArtifact.

    Fail-closed guards (historical validity):

    - ``require_decision_provenance`` (default True): raises
      :class:`MissingDecisionProvenance` when the artifact carries no
      decision-time odds or no team contexts — a row without decision-time
      provenance is not a valid replay unit. Pass False only for diagnostic
      flows that tolerate empty output (an artifact with no priced sides then
      yields ``[]``, never a guessed row).
    - Post-outcome keys (scores, results, winners) found inside the artifact's
      contexts raise :class:`LookaheadContamination`; ``outcome`` itself rides
      the row as a separate grading-time field and is never copied into
      ``model_inputs``.
    """
    if require_decision_provenance:
        if not artifact.odds:
            raise MissingDecisionProvenance(
                f"artifact {artifact.artifact_id} has no decision-time odds snapshot"
            )
        if not artifact.home_context and not artifact.away_context:
            raise MissingDecisionProvenance(
                f"artifact {artifact.artifact_id} has no decision-time team contexts"
            )

    model_inputs = {
        "home_context": dict(artifact.home_context),
        "away_context": dict(artifact.away_context),
        "game_context": dict(artifact.game_context),
    }
    _assert_pre_decision(model_inputs, f"artifact {artifact.artifact_id} model_inputs")

    closing = artifact.closing_odds or {}
    rows: list[HistoricalReplayRow] = []
    for market, side, price_key, line_key, line_sign in _ROW_SPECS:
        price = artifact.odds.get(price_key)
        if price is None:
            continue
        line = None
        if line_key is not None:
            raw_line = artifact.odds.get(line_key)
            if raw_line is None:
                continue  # a spread/total price without its line is not replayable
            line = float(raw_line) * line_sign
        rows.append(
            HistoricalReplayRow(
                artifact_id=artifact.artifact_id,
                source_trace_id=artifact.source_trace_id,
                home_team=artifact.home_team,
                away_team=artifact.away_team,
                league=artifact.league,
                date=artifact.date,
                market=market,
                side=side,
                line=line,
                offered_odds=float(price),
                closing_odds=(float(closing[price_key]) if closing.get(price_key) is not None else None),
                model_inputs=model_inputs,
                odds_snapshot=dict(artifact.odds),
                closing_snapshot=dict(artifact.closing_odds) if artifact.closing_odds else None,
                simulation_seed=artifact.simulation_seed,
                simulation_backend=artifact.simulation_backend,
                prior_payload=copy.deepcopy(artifact.prior_payload),
                substrate_unresolved=artifact.substrate_unresolved,
                calibration_policy=artifact.calibration_policy,
                outcome=dict(artifact.outcome) if artifact.outcome else None,
            )
        )
    if require_decision_provenance and not rows:
        raise MissingDecisionProvenance(
            f"artifact {artifact.artifact_id} prices no replayable market side"
        )
    return rows


def artifact_from_replay_rows(rows: list[HistoricalReplayRow]) -> FrozenArtifact:
    """Regroup rows of ONE artifact back into the FrozenArtifact the engine eats.

    The inverse of :func:`replay_rows_from_artifact` — the acceleration lane
    hands work back to the existing ``BacktestEngine`` through this, so there is
    no second engine input family. All rows must share one ``artifact_id``; the
    reconstruction is lossless (full odds/closing snapshots ride every row).
    """
    if not rows:
        raise ValueError("cannot rebuild an artifact from zero replay rows")
    ids = {r.artifact_id for r in rows}
    if len(ids) > 1:
        raise ValueError(f"rows span multiple artifacts: {sorted(ids)}")
    first = rows[0]
    return FrozenArtifact(
        artifact_id=first.artifact_id,
        source_trace_id=first.source_trace_id,
        home_team=first.home_team,
        away_team=first.away_team,
        league=first.league,
        date=first.date,
        home_context=dict(first.model_inputs.get("home_context") or {}),
        away_context=dict(first.model_inputs.get("away_context") or {}),
        game_context=dict(first.model_inputs.get("game_context") or {}),
        odds=dict(first.odds_snapshot),
        closing_odds=dict(first.closing_snapshot) if first.closing_snapshot else None,
        simulation_seed=first.simulation_seed,
        simulation_backend=first.simulation_backend,
        prior_payload=copy.deepcopy(first.prior_payload),
        substrate_unresolved=first.substrate_unresolved,
        calibration_policy=first.calibration_policy,
        outcome=dict(first.outcome) if first.outcome else None,
    )


# ---------------------------------------------------------------------------
# Simulation output cache (deterministic raw outputs only)
# ---------------------------------------------------------------------------


def _canonical_hash(payload: Any) -> str:
    """sha256 of canonical (sorted-key) JSON. Unlike
    ``omega.core.contracts.seeding.stable_analysis_hash`` this excludes NOTHING —
    lines and odds are part of cache identity, and dropping them (as the
    analysis-identity sanitizer deliberately does) would alias distinct
    simulations onto one key."""
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SimulationOutputCache:
    """In-process cache of RAW simulation outputs + their audit metadata.

    Scaffold for the Issue #27 acceleration lane. Discipline:

    - **Key covers every substrate axis.** ``make_key`` requires backend name,
      component version, model version, league, market, line, the full
      input-context hash, evidence policy, and calibration ref — change any one
      and the key changes, so nothing is ever reused across a backend,
      calibration, or evidence-policy change. Over-keying is safe (a miss);
      under-keying would be silent reuse of stale probabilities.
    - **Raw outputs only.** Calibrated/staked values are derived downstream on
      every read through the shared production policy; caching them would fork
      the calibration path.
    - **Deterministic entries only.** Callers must not ``put`` outputs of an
      unseeded MC run; the BacktestEngine hook enforces this at its call site.
    - **Never historical truth.** In-memory, dies with the process; entries are
      deep-copied on both put and get so no caller can mutate a cached value.

    ``exact_eval`` remains the preferred substrate where available — the cache
    accelerates repeated evaluation of EITHER path, it does not compete with
    exactness.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        *,
        backend_name: str,
        component_version: str,
        model_version: str,
        league: str,
        market: str,
        line: Any,
        context_hash: str,
        evidence_policy: str,
        calibration_ref: str,
    ) -> str:
        """Deterministic cache key. Every argument is identity — no defaults."""
        return _canonical_hash(
            {
                "backend_name": backend_name,
                "component_version": component_version,
                "model_version": model_version,
                "league": league,
                "market": market,
                "line": line,
                "context_hash": context_hash,
                "evidence_policy": evidence_policy,
                "calibration_ref": calibration_ref,
            }
        )

    @staticmethod
    def context_hash(payload: dict[str, Any]) -> str:
        """Canonical hash of the simulation input payload (contexts, seed, lines...)."""
        return _canonical_hash(payload)

    def get(self, key: str) -> dict[str, Any] | None:
        """Cached raw outputs (a private copy), or None. Counts hits/misses."""
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return copy.deepcopy(entry[0])

    def get_audit(self, key: str) -> dict[str, Any] | None:
        """Audit metadata stored beside a cached entry (a private copy), or None."""
        entry = self._entries.get(key)
        return copy.deepcopy(entry[1]) if entry is not None else None

    def put(
        self,
        key: str,
        raw_outputs: dict[str, Any],
        audit: dict[str, Any] | None = None,
    ) -> None:
        self._entries[key] = (copy.deepcopy(raw_outputs), copy.deepcopy(audit))

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Common random numbers (MC fallback only; exact eval unaffected)
# ---------------------------------------------------------------------------


def crn_seed(event_key: str, model_key: str | None = None) -> int:
    """Stable 64-bit seed for common-random-number MC replay.

    ``event_key`` is the replay row/artifact identity (``artifact_id``);
    ``model_key`` salts the stream space. Paired model comparisons that want the
    sampling noise to cancel call this with the SAME ``model_key`` (or none) for
    both candidates — identical random streams per row; independent evaluations
    pass their model/backend version so streams decorrelate. Deterministic:
    the same inputs always yield the same seed.
    """
    raw = f"{event_key}|{model_key or ''}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
