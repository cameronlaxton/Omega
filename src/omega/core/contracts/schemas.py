"""
Strict Pydantic schemas for the OmegaSportsAgent service contract.

These models define the JSON interface between:
  - Backend <-> Frontend (GameAnalysisResponse)
  - Agent <-> Backend (GameAnalysisRequest)
  - External callers <-> API endpoints
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omega.core.contracts.evidence import EvidenceSignal
from omega.core.contracts.language import blocked_language
from omega.core.contracts.protected_fields import find_protected_value_leak


class SoccerDerivativeMarket(str, Enum):
    """Soccer derivative markets evaluated by omega/core/edge/soccer_derivatives.py."""

    asian_handicap = "asian_handicap"
    total_goals_over_under = "total_goals_over_under"
    both_teams_to_score = "both_teams_to_score"
    correct_score = "correct_score"
    first_half_total = "first_half_total"


# -- Request Models ----------------------------------------------------------


class MarketQuote(BaseModel):
    """A single normalized market line from any sportsbook."""

    market_type: str = Field(
        description="e.g. moneyline, spread, total, moneyline_3way, puck_line, run_line, team_total, method_of_victory, set_spread, total_games, map_spread, outright_winner"
    )
    selection: str = Field(description="Human label, e.g. 'Home', 'Over 224.5', 'KO/TKO', 'Top 10'")
    price: float = Field(description="American odds")
    line: float | None = Field(default=None, description="Spread/total line value if applicable")
    segment: str = Field(
        default="full_game", description="full_game, 1h, 1q, 1p, regulation, first_5_innings, etc."
    )
    player: str | None = Field(default=None, description="Player name for player props")
    stat_key: str | None = Field(
        default=None, description="Prop stat key, e.g. pts, pass_yds, aces, kills"
    )
    bookmaker: str | None = Field(default=None, description="Source sportsbook")
    source: str | None = Field(
        default=None, description="Provider/source label, e.g. the-odds-api:betmgm"
    )
    event_id: str | None = Field(default=None, description="Provider event id")
    provider_market_key: str | None = Field(default=None, description="Provider-native market key")
    last_update: str | None = Field(
        default=None, description="Provider market last update timestamp"
    )
    snapshot_timestamp: str | None = Field(default=None, description="Provider snapshot timestamp")


class OddsInput(BaseModel):
    """User-supplied or agent-scraped odds for a single game.

    For backward compatibility the flat fields (spread_home, moneyline_home, etc.)
    are preserved.  New callers should prefer the ``markets`` list which can
    represent any market type across all sports.
    """

    # Legacy flat fields (2-way)
    spread_home: float | None = Field(default=None, description="Home spread line (e.g., -3.5)")
    spread_home_price: float | None = Field(
        default=-110, description="Juice on the home spread (American odds)"
    )
    spread_away_price: float | None = Field(
        default=-110, description="Juice on the away spread (American odds)"
    )
    moneyline_home: float | None = Field(default=None, description="Home moneyline (American odds)")
    moneyline_away: float | None = Field(default=None, description="Away moneyline (American odds)")
    over_under: float | None = Field(default=None, description="Total line (e.g., 224.5)")
    total_over_price: float | None = Field(default=-110, description="Over price (American odds)")
    total_under_price: float | None = Field(default=-110, description="Under price (American odds)")

    # 3-way moneyline (hockey regulation, soccer)
    moneyline_draw: float | None = Field(
        default=None, description="Draw moneyline (American odds) for 3-way markets"
    )

    # Exotic 3-way markets (soccer). All additive and optional — absent price
    # means no edge is built for that market.
    dc_home_draw: float | None = Field(
        default=None, description="Double chance Home-or-Draw (1X) price (American odds)"
    )
    dc_home_away: float | None = Field(
        default=None, description="Double chance Home-or-Away (12) price (American odds)"
    )
    dc_away_draw: float | None = Field(
        default=None, description="Double chance Away-or-Draw (X2) price (American odds)"
    )
    dnb_home: float | None = Field(
        default=None, description="Draw-no-bet Home price (American odds); draw voids"
    )
    dnb_away: float | None = Field(
        default=None, description="Draw-no-bet Away price (American odds); draw voids"
    )
    btts_yes: float | None = Field(
        default=None, description="Both teams to score - Yes price (American odds)"
    )
    btts_no: float | None = Field(
        default=None, description="Both teams to score - No price (American odds)"
    )
    correct_score: dict[str, float] | None = Field(
        default=None,
        description="Correct-score prices keyed by 'home-away' scoreline, e.g. {'1-0': 650}",
    )

    # Asian handicap + first-half total (soccer derivatives, Phase 7 M2).
    # Quarter-ball lines (-0.25, +0.75, 2.25, ...) are supported: the stake is
    # split across the two adjacent half/integer lines with push/half-stake
    # semantics evaluated in omega/core/edge/soccer_derivatives.py.
    asian_handicap_home: float | None = Field(
        default=None,
        description="Home Asian-handicap line (e.g. -0.75); away line is the negation",
    )
    ah_home_price: float | None = Field(
        default=None, description="Asian-handicap Home price (American odds)"
    )
    ah_away_price: float | None = Field(
        default=None, description="Asian-handicap Away price (American odds)"
    )
    first_half_total: float | None = Field(
        default=None, description="First-half total goals line (e.g. 1.0 or 1.25)"
    )
    fh_over_price: float | None = Field(
        default=None, description="First-half total Over price (American odds)"
    )
    fh_under_price: float | None = Field(
        default=None, description="First-half total Under price (American odds)"
    )

    # Normalized market list (preferred path for all new integrations)
    markets: list[MarketQuote] | None = Field(
        default=None, description="Normalized list of all scraped markets"
    )


class GameAnalysisRequest(BaseModel):
    """Request to analyze a single game matchup.

    home_context / away_context carry team performance stats (archetype-specific).
    game_context carries situational signals used for calibration slice selection
    and any future game-level adjustments (is_playoff, rest_days, etc.).

    Required keys by archetype (see omega/core/simulation/archetypes.py for full lists):
      Basketball (NBA, NCAAB): off_rating, def_rating, pace
      American football (NFL): off_rating, def_rating
      Baseball (MLB): off_rating, def_rating
      Hockey (NHL): off_rating, def_rating
      Soccer (EPL, MLS, ...): off_rating, def_rating
      Tennis (ATP, WTA): serve_win_pct, return_win_pct
      Golf (PGA): strokes_gained_total
      Combat sports (UFC, Boxing): win_pct, finish_rate
      Esports (CS2): map_win_rate, recent_form
    """

    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    league: str = Field(
        description="League identifier: NBA, NFL, MLB, NHL, NCAAB, EPL, UFC, ATP, PGA, CS2, ..."
    )
    odds: OddsInput | None = Field(default=None, description="Market odds (if available)")
    n_iterations: int = Field(default=1000, ge=100, le=100000, description="Simulation iterations")
    allow_baseline: bool = Field(
        default=False,
        description=(
            "Explicitly allow league-average baseline contexts when home_context or "
            "away_context is absent. Baseline runs are audit-only and are not "
            "calibration eligible."
        ),
    )
    simulation_backend: str = Field(
        default="fast_score",
        description="Deterministic game simulator backend: 'fast_score' or 'markov_state'.",
    )
    home_context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Pre-fetched home team stats. Required keys depend on league archetype "
            "(e.g. off_rating, def_rating, pace for basketball). "
            "Omitting or passing None returns status='skipped' with missing_requirements listing exact keys."
        ),
    )
    away_context: dict[str, Any] | None = Field(
        default=None,
        description=("Pre-fetched away team stats. Same required keys as home_context."),
    )
    game_context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Situational game context for calibration slice selection. Open-ended dict; "
            "supply any applicable signals. "
            "Universal: is_playoff (bool), rest_days (int; 0=B2B), "
            "opponent_def_rank (int), blowout_risk (float 0-1), "
            "pace_adjustment_factor (float). "
            "MLB: park_factor (float), weather_wind_mph (float). "
            "NFL: is_dome (bool), weather_temp_f (float), week_of_season (int). "
            "Any additional matchup context (e.g. matchup_weakness, scheme_advantage) "
            "is preserved in context_labels for calibration fitting."
        ),
    )
    prior_payload: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Game-level dynamic priors for the simulation backend that are not "
            "team-scoped — e.g. Dixon-Coles 'rho' (soccer) or 'pressure_coefficients' "
            "(tennis). Supplied by the gatherer; flows verbatim into "
            "GameSimulationInput.prior_payload. A backend that requires a prior fails "
            "closed (status='skipped', missing_requirements) when it is absent."
        ),
    )
    seed: int | None = Field(default=None, description="RNG seed for reproducible simulations")
    evidence: list[EvidenceSignal] = Field(
        default_factory=list,
        description=(
            "Structured reasoning signals (game/matchup/situational/team_form). "
            "The engine applies known signal types deterministically before simulation "
            "and persists every signal for retrospective scoring; unknown types are "
            "ignored. Defaults to empty — supplying none preserves prior behavior."
        ),
    )

    @model_validator(mode="after")
    def _enforce_missing_game_context_keys(self) -> GameAnalysisRequest:
        gc = self.game_context or {}
        missing = [k for k in ("is_playoff", "rest_days") if k not in gc]
        if missing:
            raise ValueError(
                f"GameAnalysisRequest missing game_context keys: {missing}. "
                "Context-slice calibration fitting requires is_playoff and rest_days."
            )
        return self


class SlateAnalysisRequest(BaseModel):
    """Request to analyze all games for a league on a given date.
    Caller (e.g. agent) should supply games list; service does not fetch schedule."""

    league: str = Field(description="League identifier")
    date: str | None = Field(
        default=None, description="Date in YYYY-MM-DD format; defaults to today"
    )
    bankroll: float = Field(default=1000.0, gt=0, description="Bankroll for stake sizing")
    edge_threshold: float = Field(default=0.03, ge=0.0, le=0.5, description="Minimum edge to flag")
    games: list[dict[str, Any]] | None = Field(
        default=None, description="Pre-fetched games (home/away/odds); required for analysis"
    )


class ReasoningPresentation(BaseModel):
    """Qualitative analyst-note prose persisted with a batch trace."""

    model_config = ConfigDict(extra="forbid")

    thesis: str | None = None
    market_read: str | None = None
    why: str | None = None
    risks: str | None = None
    verdict: str | None = None


# -- Presentation & ledger policy (Matchup Intelligence, Phase 0) -------------
#
# ``output_mode`` (omega.ops.output_modes) governs WHICH engine values may be
# disclosed; ``presentation_mode`` governs HOW authorized values are framed.
# Effective visibility is the intersection of the two policies. Both new modes
# fail closed: a missing/unrecognized value is treated as the restrictive
# default (decision_support / disabled) — recommendation_lab never elevates a
# restrictive output_mode.

PresentationMode = Literal["decision_support", "recommendation_lab"]
EngineAutoLedgerMode = Literal["disabled", "shadow"]

PRESENTATION_MODE_DEFAULT: PresentationMode = "decision_support"
ENGINE_AUTO_LEDGER_MODE_DEFAULT: EngineAutoLedgerMode = "disabled"

_PRESENTATION_MODES: frozenset[str] = frozenset({"decision_support", "recommendation_lab"})
_ENGINE_AUTO_LEDGER_MODES: frozenset[str] = frozenset({"disabled", "shadow"})


def coerce_presentation_mode(value: object) -> PresentationMode:
    """Fail-closed coercion: anything but a recognized mode is decision_support."""
    if isinstance(value, str) and value in _PRESENTATION_MODES:
        return value  # type: ignore[return-value]
    return PRESENTATION_MODE_DEFAULT


def coerce_engine_auto_ledger_mode(value: object) -> EngineAutoLedgerMode:
    """Fail-closed coercion: anything but a recognized mode is disabled."""
    if isinstance(value, str) and value in _ENGINE_AUTO_LEDGER_MODES:
        return value  # type: ignore[return-value]
    return ENGINE_AUTO_LEDGER_MODE_DEFAULT


class EventIdentityV1(BaseModel):
    """Versioned provider-anchored event identity persisted in the full trace JSON.

    Game and prop traces for the same real-world event must share ``event_key``
    so read models can group them without heuristics. Legacy traces without a
    trustworthy provider id stay ungrouped (identity warning) rather than being
    merged by team-name matching.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    provider: str = Field(min_length=1, description="Odds/schedule provider, e.g. the-odds-api")
    provider_event_id: str = Field(min_length=1, description="Provider-native event id")
    event_key: str = Field(min_length=1, description="Stable cross-market grouping key")
    league: str = Field(min_length=1)
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    game_date: str = Field(min_length=10, description="ISO date YYYY-MM-DD")
    commence_time: str | None = Field(
        default=None, description="ISO 8601 scheduled start, when known"
    )

    @staticmethod
    def derive_event_key(league: str, provider: str, provider_event_id: str) -> str:
        """Canonical event_key derivation shared by every stamping site."""
        return f"{league.strip().upper()}::{provider.strip()}::{provider_event_id.strip()}"

    @model_validator(mode="after")
    def _validate_identity(self) -> EventIdentityV1:
        for field_name in (
            "provider",
            "provider_event_id",
            "event_key",
            "league",
            "home_team",
            "away_team",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"EventIdentityV1.{field_name} must not be blank")
        expected_key = self.derive_event_key(self.league, self.provider, self.provider_event_id)
        if self.event_key != expected_key:
            raise ValueError(
                f"EventIdentityV1.event_key {self.event_key!r} does not match the canonical "
                f"derivation {expected_key!r} for league/provider/provider_event_id — every "
                "stamping site must derive event_key via derive_event_key(), never construct it "
                "independently."
            )
        return self


def _normalize_outcome_component(value: str) -> str:
    """Case/whitespace-fold a market_key or outcome_key before comparison so
    " Home " and "home" are recognized as the same side."""
    return value.strip().casefold()


# Known symmetric outcome_key vocabularies (per OutcomeCase's own doc: "e.g.
# home, away, draw, over, under"). A market whose observed outcome_keys
# intersect one of these families must report the corresponding complete set;
# unrecognized vocabulary falls back to a minimum-cardinality check.
_TOTAL_STYLE_OUTCOMES = frozenset({"over", "under"})
_MONEYLINE_STYLE_OUTCOMES = frozenset({"home", "away", "draw"})


def _expected_complete_outcome_set(observed: frozenset[str]) -> frozenset[str] | None:
    """Canonical complete outcome set implied by ``observed`` vocabulary, or
    None when the vocabulary isn't a recognized symmetric family."""
    if observed & _TOTAL_STYLE_OUTCOMES:
        return _TOTAL_STYLE_OUTCOMES
    if observed & _MONEYLINE_STYLE_OUTCOMES:
        return (
            frozenset({"home", "away", "draw"})
            if "draw" in observed
            else frozenset({"home", "away"})
        )
    return None


class OutcomeCase(BaseModel):
    """Balanced case for one outcome of one market (decision-support unit)."""

    model_config = ConfigDict(extra="forbid")

    market_key: str = Field(min_length=1, description="Stable market identity, e.g. moneyline")
    outcome_key: str = Field(min_length=1, description="e.g. home, away, draw, over, under")
    label: str = Field(min_length=1, description="Human outcome label")
    supporting: list[str] = Field(default_factory=list)
    challenging: list[str] = Field(default_factory=list)
    data_status: Literal["complete", "partial", "insufficient"]

    @model_validator(mode="after")
    def _validate_case(self) -> OutcomeCase:
        if self.data_status == "complete" and (not self.supporting or not self.challenging):
            raise ValueError(
                "OutcomeCase data_status='complete' requires both supporting and "
                "challenging evidence; downgrade data_status instead of fabricating a case."
            )
        for text in (self.label, *self.supporting, *self.challenging):
            found = blocked_language(text)
            if found:
                raise ValueError(f"OutcomeCase contains blocked language {found}: {text!r}")
            leak = find_protected_value_leak(text)
            if leak:
                raise ValueError(f"OutcomeCase contains a protected engine value {leak!r}: {text!r}")
        return self


class DecisionSupportPresentationV1(BaseModel):
    """Primary presentation contract for decision-support briefs.

    A separate typed field from the legacy ``ReasoningPresentation`` — the two
    contracts never share a schema. Qualitative only; recommendation vocabulary
    and protected engine values are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    matchup_summary: str = Field(min_length=1)
    market_context: str = Field(min_length=1)
    outcome_cases: list[OutcomeCase] = Field(default_factory=list)
    scenario_triggers: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    decision_conditions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_presentation(self) -> DecisionSupportPresentationV1:
        seen: set[tuple[str, str]] = set()
        by_market: dict[str, set[str]] = {}
        for case in self.outcome_cases:
            market_key = _normalize_outcome_component(case.market_key)
            outcome_key = _normalize_outcome_component(case.outcome_key)
            key = (market_key, outcome_key)
            if key in seen:
                raise ValueError(f"Duplicate outcome case for {key}")
            seen.add(key)
            by_market.setdefault(market_key, set()).add(outcome_key)
        for market_key, observed in by_market.items():
            expected = _expected_complete_outcome_set(frozenset(observed))
            if expected is not None:
                if observed != expected:
                    raise ValueError(
                        f"decision-support market {market_key!r} has an incomplete outcome "
                        f"set {sorted(observed)!r}; expected the complete symmetric set "
                        f"{sorted(expected)!r} (mark a missing side data_status="
                        "'insufficient' rather than omitting its OutcomeCase)."
                    )
            elif len(observed) < 2:
                raise ValueError(
                    f"decision-support market {market_key!r} has a single-sided outcome "
                    f"set {sorted(observed)!r}; every quoted market must report the "
                    "complete symmetric outcome set."
                )
        for text in (
            self.matchup_summary,
            self.market_context,
            *self.scenario_triggers,
            *self.uncertainties,
            *self.decision_conditions,
            *self.questions,
        ):
            found = blocked_language(text)
            if found:
                raise ValueError(
                    f"DecisionSupportPresentationV1 contains blocked language {found}: {text!r}"
                )
            leak = find_protected_value_leak(text)
            if leak:
                raise ValueError(
                    f"DecisionSupportPresentationV1 contains a protected engine value {leak!r}: {text!r}"
                )
        return self


class BatchAnalysisEntry(BaseModel):
    """One entry in an omega_run_batch call — either a game or a player prop.

    If odds fields are absent the batch tool resolves them via omega_resolve_odds.
    For props, prop_type may be a list to express a fallback chain (first available
    market wins).
    """

    kind: Literal["game", "prop"] = Field(description="'game' or 'prop'")
    league: str = Field(description="League identifier, e.g. MLB, NBA, NFL")
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    game_date: str | None = Field(default=None, description="YYYY-MM-DD; defaults to today")
    event_id: str | None = Field(
        default=None,
        description=(
            "Odds-provider (the-odds-api) event id for this matchup, e.g. from "
            "omega_list_events. Game and prop entries for the same real-world event "
            "must carry the same id so their traces share an EventIdentityV1.event_key. "
            "When the batch tool resolves odds live it confirms this id against the "
            "resolver's match and errors the entry on disagreement. None = identity "
            "resolved from odds resolution alone, or legacy/unknown (trace stays "
            "ungrouped with an identity warning)."
        ),
    )
    game_context: dict[str, Any] = Field(
        default_factory=dict, description="Calibration context (is_playoff, rest_days, …)"
    )
    n_iterations: int = Field(default=10000, ge=100, le=100000)
    seed: int | None = Field(
        default=None, description="RNG seed; auto-derived from content hash if None"
    )
    evidence: list[dict[str, Any]] = Field(default_factory=list, description="EvidenceSignal dicts")
    reasoning_narrative: str | None = Field(
        default=None, description="2–4 sentence summary of reasoning"
    )
    reasoning_presentation: ReasoningPresentation | None = Field(
        default=None,
        description=(
            "Optional analyst-note prose keyed thesis/market_read/why/risks/verdict. "
            "Qualitative only; no protected engine values."
        ),
    )
    decision_support_presentation: DecisionSupportPresentationV1 | None = Field(
        default=None,
        description=(
            "Optional decision-support presentation (matchup summary, symmetric "
            "outcome cases, uncertainties, decision conditions). The primary "
            "presentation contract — qualitative only; recommendation vocabulary "
            "and protected engine values are rejected at validation."
        ),
    )
    reasoning_sources: list[str] = Field(
        default_factory=list, description="Sources consulted, e.g. espn.com"
    )
    roster_context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "RSVG (Roster & Situational Verification Gate) payload — a "
            "RosterContextPayload dict (see omega/core/gates/rsvg.py). When present "
            "the batch tool runs the gate BEFORE odds resolution/analyze(): "
            "'blocked' skips the entry, 'research_candidate' stamps "
            "reasoning_downgrade_rationale + trace_quality.rsvg on the trace, and "
            "emitted usage_role_change signals are merged into evidence. Structured "
            "facts only — the gate never browses and never computes engine values."
        ),
    )
    # prop-only
    player_name: str | None = Field(default=None, description="Player name (kind='prop' only)")
    prop_type: str | list[str] | None = Field(
        default=None,
        description="Stat key (kind='prop'). List = fallback chain — first available market wins.",
    )
    player_context: dict[str, Any] = Field(default_factory=dict, description="Player stat context")
    # game-only
    home_context: dict[str, Any] = Field(default_factory=dict, description="Home team context")
    away_context: dict[str, Any] = Field(default_factory=dict, description="Away team context")
    # pre-supplied odds (absent → batch tool resolves via resolve_odds)
    odds_over: float | None = Field(default=None, description="American odds for Over (prop)")
    odds_under: float | None = Field(default=None, description="American odds for Under (prop)")
    line: float | None = Field(default=None, description="Prop line; auto-resolved if absent")
    odds: dict[str, Any] | None = Field(
        default=None, description="Game odds dict (kind='game'); auto-resolved if absent"
    )


class PlayerPropRequest(BaseModel):
    """Request to analyze a single player prop."""

    player_name: str
    league: str
    prop_type: str = Field(description="Stat key, e.g. pts, pass_yds, aces, kills, goals")
    line: float = Field(description="The prop line, e.g. 22.5")
    odds_over: float | None = Field(default=None, description="American odds for Over")
    odds_under: float | None = Field(default=None, description="American odds for Under")
    bookmaker: str | None = Field(
        default=None,
        description=(
            "Source sportsbook for odds_over/odds_under (provenance only; does not "
            "affect simulation). Recorded into the bet ledger. Leave None when the "
            "prices were line-shopped across books."
        ),
    )
    player_context: dict[str, Any] | None = Field(
        default=None, description="Player statistical context"
    )
    game_context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Game-level context used to adjust player_context means before simulation. "
            "All keys optional; supply only what applies for the sport. "
            "Universal: is_playoff (bool), rest_days (int; 0=B2B), "
            "opponent_def_rank (int 1-30), blowout_risk (float 0-1), "
            "pace_adjustment_factor (float ratio vs league baseline). "
            "MLB: park_factor (float; >1=hitter-friendly), weather_wind_mph (float), "
            "pitcher_k_rate (float), is_starter_probable (bool). "
            "NFL: is_dome (bool), weather_temp_f (float), week_of_season (int)."
        ),
    )
    n_iterations: int = Field(default=5000, ge=100, le=100000)
    seed: int | None = Field(default=None, description="RNG seed for reproducible simulations")
    evidence: list[EvidenceSignal] = Field(
        default_factory=list,
        description=(
            "Structured reasoning signals (player_form/matchup/situational). "
            "The engine applies known signal types deterministically to player_context "
            "means before simulation and persists every signal for retrospective "
            "scoring; unknown types are ignored. Defaults to empty — supplying none "
            "preserves prior behavior."
        ),
    )

    @model_validator(mode="after")
    def _enforce_missing_game_context_keys(self) -> PlayerPropRequest:
        gc = self.game_context or {}
        missing = [k for k in ("is_playoff", "rest_days") if k not in gc]
        if missing:
            raise ValueError(
                f"PlayerPropRequest missing game_context keys: {missing}. "
                "Context-slice calibration fitting requires is_playoff and rest_days."
            )
        return self

    # Required because persisted prop traces are graded by
    # (game_date, home_team, away_team). Without these fields the outcome
    # resolver cannot safely attach an ESPN box-score result.
    home_team: str = Field(description="Home team of the game the prop is on", min_length=1)
    away_team: str = Field(description="Away team of the game the prop is on", min_length=1)
    game_date: str = Field(description="ISO date (YYYY-MM-DD) of the game", min_length=10)


# -- Response Sub-Models -----------------------------------------------------


class SimulationResult(BaseModel):
    """Core simulation output."""

    iterations: int
    home_win_prob: float = Field(description="Home/Player-A win probability (0-100)")
    away_win_prob: float = Field(description="Away/Player-B win probability (0-100)")
    draw_prob: float | None = Field(
        default=None, description="Draw probability (0-100), for 3-way markets"
    )
    predicted_spread: float = Field(description="Predicted spread (negative = home favored)")
    predicted_total: float = Field(description="Predicted combined score")
    predicted_home_score: float
    predicted_away_score: float
    context_source: str = Field(default="provided", description="'provided' or 'league_default'")
    baseline_used: bool = Field(default=False)
    simulation_backend: str | None = None
    component_version: str | None = None
    competition_strength_adjustment: dict[str, Any] | None = Field(
        default=None,
        description="Structural soccer competition-strength index debug payload "
        "(raw/adjusted attack-concede rates, applied index by side, final "
        "home/away lambdas); None when the index was not applied. Issue #22.",
    )


class EdgeDetail(BaseModel):
    """Edge analysis for one side of a matchup."""

    side: str = Field(description="'home', 'away', or 'draw'")
    team: str
    market: str = Field(default="moneyline", description="moneyline, spread, total, or draw")
    line: float | None = Field(default=None, description="Market line for spread/total edges")
    true_prob: float = Field(description="Raw model probability (0-1)")
    calibrated_prob: float = Field(description="Calibrated probability (0-1)")
    market_implied: float = Field(description="Market implied probability (0-1)")
    edge_pct: float = Field(description="Edge in percentage points")
    ev_pct: float = Field(description="Expected value percentage")
    market_odds: float = Field(description="American odds used for this side")
    confidence_tier: str = Field(description="A (high), B (medium), C (low), or Pass")
    confidence_cap_reason: str | None = Field(
        default=None,
        description="Why confidence was capped below A (e.g. 'no_production_profile_calibration', "
        "'trace_quality_cap', 'static_identity', 'insufficient_iterations'); None when uncapped.",
    )
    recommended_units: float = Field(default=0.0, description="Kelly-sized stake in units")
    spread_coverage_prob: float | None = Field(
        default=None,
        description="P(covers spread) when market is run-line/puck-line (0-1); None for moneyline edges",
    )
    calibration_audit: CalibrationAudit | None = Field(
        default=None,
        description="Calibration provenance for this edge: which path/profile was used",
    )


class BetSlip(BaseModel):
    """A single actionable bet recommendation."""

    selection: str = Field(
        description="e.g., 'Lakers -3.5', 'Over 2.5 Goals', 'Fighter A by KO/TKO'"
    )
    odds: float = Field(description="American odds")
    edge_pct: float
    ev_pct: float
    confidence_tier: str
    recommended_units: float
    kelly_fraction: float


class CalibrationAudit(BaseModel):
    """Per-edge calibration provenance. Records which path was taken, not inferred."""

    raw_prob: float
    calibrated_prob: float
    league: str | None = None
    plane: str = Field(description="'game' or 'prop'")
    market: str = Field(description="'home', 'away', 'draw', 'over', 'under', 'cover'")
    method_resolved: str | None = None
    profile_id: str | None = None
    context_slice: str | None = None
    resolved_slice: str | None = None
    path: str = Field(
        description=(
            "'profile': learned profile applied; "
            "'base_profile_fallback': fell back from context_slice to base profile; "
            "'static_calibrated': no profile, static policy changed the value; "
            "'static_identity': no profile, within threshold, returned raw unchanged"
        )
    )
    # Profile maturity + quality provenance (None on static paths). Surfaced so
    # the confidence layer and reports can be honest about how trustworthy the
    # applied calibration was.
    profile_status: str | None = Field(
        default=None, description="Lifecycle status of the applied profile (e.g. 'production')."
    )
    profile_maturity: str | None = Field(
        default=None,
        description="Trust level: none|provisional|probation|production|retired. "
        "provisional/probation apply capped corrections and cap confidence.",
    )
    sample_size: int | None = Field(
        default=None, description="Training sample size of the applied profile."
    )
    ece: float | None = Field(
        default=None, description="Expected Calibration Error of the applied profile, if recorded."
    )
    brier: float | None = Field(
        default=None, description="Brier score of the applied profile, if recorded."
    )
    fallback_level: str | None = Field(
        default=None,
        description="Hierarchical level the profile came from: league|sport_family|global. "
        "None on static paths.",
    )
    binding_status: str | None = Field(
        default=None,
        description=(
            "Backend-binding status of the APPLIED profile (P8.3): 'bound' (its "
            "recorded substrate matched the live one), 'unpinned' (fit dataset "
            "carried no substrate identity), 'legacy' (profile predates binding). "
            "None on static paths."
        ),
    )
    binding_mismatch: str | None = Field(
        default=None,
        description=(
            "Why a fitted profile was SKIPPED during selection because its backend "
            "binding rejected the live raw-probability substrate (P8.3 fail-closed), "
            "e.g. 'iso_nfl_prop_v2_...: param_profile_id_mismatch:fit=...,live=...'. "
            "None when nothing was skipped."
        ),
    )

    @property
    def static_identity_used(self) -> bool:
        """True when no calibration moved the probability (raw returned unchanged)."""
        return self.path == "static_identity"


class AnalysisMetadata(BaseModel):
    """Metadata about how the analysis was produced."""

    engine_version: str = "2.0-dse"
    data_sources: list[str] = Field(default_factory=lambda: ["simulation"])
    archetype: str | None = Field(default=None, description="Sport archetype used for simulation")
    suppressed_markets: list[str] = Field(
        default_factory=list,
        description="Non-quant list of markets intentionally suppressed by service policy.",
    )


# -- Top-Level Response Models -----------------------------------------------


class GameAnalysisResponse(BaseModel):
    """Complete analysis for a single game. The primary JSON contract for the frontend."""

    matchup: str = Field(description="'Away @ Home' display string")
    league: str
    analyzed_at: str = Field(description="ISO 8601 timestamp")
    status: str = Field(description="'success', 'skipped', or 'error'")
    skip_reason: str | None = None
    missing_requirements: list[str] | None = Field(
        default=None,
        description="Machine-readable list of missing inputs the agent should fetch, e.g. ['home_context.off_rating', 'away_context.serve_win_pct']",
    )

    simulation: SimulationResult | None = None
    edges: list[EdgeDetail] = Field(default_factory=list)
    best_bet: BetSlip | None = None
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)
    context_source: str | None = None
    baseline_used: bool = False
    simulation_backend: str | None = None
    component_version: str | None = None
    simulation_distributions: list[dict[str, Any]] = Field(default_factory=list)


class SlateAnalysisResponse(BaseModel):
    """Analysis for a full slate of games."""

    league: str
    date: str
    total_games: int
    games_analyzed: int
    games_with_edge: int
    analyses: list[GameAnalysisResponse] = Field(default_factory=list)


class PlayerPropResponse(BaseModel):
    """Analysis for a single player prop."""

    player_name: str
    league: str
    prop_type: str
    line: float
    status: str = Field(description="'success', 'skipped', or 'error'")
    skip_reason: str | None = None
    missing_requirements: list[str] | None = None

    over_prob: float | None = Field(default=None, description="Probability of Over (0-1)")
    under_prob: float | None = Field(default=None, description="Probability of Under (0-1)")
    projection_mean: float | None = Field(default=None, description="Simulated distribution mean")
    projection_std: float | None = Field(default=None, description="Simulated distribution std dev")
    projection_p10: float | None = None
    projection_p50: float | None = None
    projection_p90: float | None = None
    distribution_type: str | None = Field(default=None, description="Resolved generator family")
    edge_over: float | None = Field(default=None, description="Edge on Over in pct points")
    edge_under: float | None = Field(default=None, description="Edge on Under in pct points")
    recommendation: str | None = Field(default=None, description="'over', 'under', or 'pass'")
    confidence_tier: str | None = None
    confidence_cap_reason: str | None = Field(
        default=None,
        description="Why confidence was capped below A; None when uncapped.",
    )
    kelly_fraction: float | None = Field(
        default=None,
        description="Scaled Kelly fraction for the recommended prop side; null for pass or unsourced odds",
    )
    recommended_units: float | None = Field(
        default=None,
        description="Kelly-sized stake in units for the recommended prop side; null for pass or unsourced odds",
    )
    bet_side_odds: float | None = Field(
        default=None,
        description="American odds used for the recommended prop side; null for pass or unsourced odds",
    )
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Machine-readable annotations: 'odds_unsourced_over', "
            "'odds_unsourced_under', 'tier_capped_imputation', "
            "'insufficient_real_observations', 'imputed_keys_provided_without_sample_size', "
            "'distribution_override:<family>'."
        ),
    )
    imputed_fraction: float | None = Field(
        default=None,
        description="Fraction of observations marked as imputed (0..1); None when not reported.",
    )
    over_calibration_audit: CalibrationAudit | None = Field(
        default=None,
        description="Calibration provenance for the over probability",
    )
    under_calibration_audit: CalibrationAudit | None = Field(
        default=None,
        description="Calibration provenance for the under probability",
    )
    context_source: str | None = None
    baseline_used: bool = False
    simulation_distributions: list[dict[str, Any]] = Field(default_factory=list)
    parameter_profile_ref: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Provenance ref of the promoted BackendParameterProfile whose structural "
            "knobs priced this prop (echoed by the backend from prior_payload); "
            "None when the pair is ungoverned. Persisted into the V20 "
            "traces.parameter_profile_ref column so replay/lab runs can pin params."
        ),
    )


class ErrorResponse(BaseModel):
    """Structured error returned by the API."""

    error_code: str = Field(
        description="Machine-readable code: SIM_FAILED, DATA_MISSING, INVALID_INPUT"
    )
    message: str = Field(description="Human-readable error description")
    context: dict[str, Any] | None = None
    fallback_hint: str | None = Field(default=None, description="Suggestion for the caller")
    missing_requirements: list[str] | None = Field(
        default=None,
        description="Machine-readable missing inputs for agent self-healing",
    )
