"""
Structured evidence signals for the Omega service contract.

The LLM agent does genuine multi-sport reasoning (defensive matchup, recent form,
elimination-game usage spikes, outlier regression). Historically that reasoning
landed in open ``dict[str, Any]`` context fields and was silently discarded by the
deterministic engine. ``EvidenceSignal`` makes each piece of reasoning a typed,
attributable, retrospectively-scoreable unit.

Design contract:
  - Evidence VALUES come from the LLM (reasoning / evidence arbitration).
  - How the engine APPLIES them is deterministic and versioned (see
    ``omega/core/simulation/evidence_handlers.py`` and
    ``omega/core/calibration/adjustment_policy.py``).
  - The engine ignores any ``signal_type`` it does not have a handler for, so
    adding a new signal type is always backward-compatible.

This module owns only the request-side contract: the typed model plus a
sport-tagged taxonomy registry. It must stay import-light.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omega.core.simulation.archetypes import ARCHETYPE_REGISTRY, get_archetype_name

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

SignalCategory = Literal["player_form", "matchup", "situational", "team_form"]
SignalPlane = Literal["player", "game"]
SignalWindow = Literal[
    "last_1", "last_3", "last_5", "last_10", "season", "series", "h2h", "matchup"
]
SignalDirection = Literal["over", "under", "home", "away", "neutral"]
ValueKind = Literal["scalar", "bool", "series", "categorical"]

# Archetype name used by SignalSpec.applies_to_sports to mean "every sport".
UNIVERSAL = "*"

_ALL_ARCHETYPES: frozenset[str] = frozenset(ARCHETYPE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Typed evidence model
# ---------------------------------------------------------------------------


class EvidenceSignal(BaseModel):
    """A single structured reasoning signal supplied by the agent.

    The engine applies known ``signal_type`` values deterministically before
    simulation; unknown types are persisted (for audit) but never applied.
    """

    model_config = ConfigDict(extra="forbid")

    signal_type: str = Field(
        description="Registry key, e.g. 'usage_spike', 'def_matchup_weak', 'recent_form'."
    )
    category: SignalCategory = Field(
        description="player_form | matchup | situational | team_form"
    )
    plane: SignalPlane = Field(
        description="'player' adjusts player_context means; 'game' adjusts team context."
    )
    value: bool | int | float | str | list[float] = Field(
        description="Raw signal payload (scalar, bool, categorical label, or series of floats)."
    )
    source: str = Field(
        description="Provenance, used as a retrospective-scoring key "
        "(e.g. 'agent_reasoning', 'boxscore_derived', 'injury_report', 'nba.com')."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Agent's stated confidence in this signal (0-1)."
    )
    window: SignalWindow = Field(
        description="Observation window: last_3 | last_5 | last_10 | season | series | h2h | matchup."
    )
    direction: SignalDirection | None = Field(
        default=None,
        description="Side the signal favors (over/under for props, home/away for games); "
        "None or 'neutral' if the signal only adjusts magnitude.",
    )
    stat_key: str | None = Field(
        default=None,
        description="Prop stat the signal bears on (e.g. 'pts', 'pass_yds'); "
        "None for game-plane signals.",
    )
    note: str | None = Field(
        default=None,
        description="Optional free-text rationale. Preserves the legacy matchup_note content; "
        "never consumed by the engine.",
    )

    @model_validator(mode="after")
    def _warn_on_registry_mismatch(self) -> EvidenceSignal:
        """Warn (never raise) when a signal disagrees with the taxonomy registry.

        Raising would break forward-compatibility: a newer agent may emit a
        ``signal_type`` this engine build does not yet know. The trace still
        persists the signal so the disagreement is auditable.
        """
        spec = SIGNAL_REGISTRY.get(self.signal_type)
        if spec is None:
            warnings.warn(
                f"EvidenceSignal unknown signal_type {self.signal_type!r}; "
                "it will be persisted for audit but the engine will not apply it.",
                stacklevel=3,
            )
            return self
        if spec.category != self.category:
            warnings.warn(
                f"EvidenceSignal {self.signal_type!r}: category {self.category!r} "
                f"disagrees with registry ({spec.category!r}).",
                stacklevel=3,
            )
        if spec.plane != "both" and spec.plane != self.plane:
            warnings.warn(
                f"EvidenceSignal {self.signal_type!r}: plane {self.plane!r} "
                f"disagrees with registry ({spec.plane!r}).",
                stacklevel=3,
            )
        if spec.requires_stat_key and self.plane == "player" and not self.stat_key:
            warnings.warn(
                f"EvidenceSignal {self.signal_type!r} requires a stat_key but none was supplied.",
                stacklevel=3,
            )
        return self


# ---------------------------------------------------------------------------
# Sport-tagged taxonomy registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalSpec:
    """Canonical definition of one signal type.

    ``applies_to_sports`` holds archetype names (see
    ``omega/core/simulation/archetypes.py``) or ``{UNIVERSAL}`` for all sports.
    ``plane`` may be "player", "game", or "both".
    """

    signal_type: str
    category: SignalCategory
    plane: Literal["player", "game", "both"]
    applies_to_sports: frozenset[str] = field(default_factory=lambda: frozenset({UNIVERSAL}))
    requires_stat_key: bool = False
    default_window: SignalWindow = "season"
    value_kind: ValueKind = "scalar"
    description: str = ""


def _spec(*args, **kwargs) -> tuple[str, SignalSpec]:
    s = SignalSpec(*args, **kwargs)
    return s.signal_type, s


# Seed taxonomy. Multi-sport by construction: a signal declares which archetypes
# it is meaningful for, and the engine gates application on that declaration.
SIGNAL_REGISTRY: dict[str, SignalSpec] = dict(
    [
        # --- Universal (all sports) ---
        _spec(
            "recent_form",
            "player_form",
            "player",
            requires_stat_key=True,
            default_window="last_5",
            value_kind="series",
            description="Recent per-game stat values vs the season baseline.",
        ),
        _spec(
            "series_avg",
            "player_form",
            "player",
            requires_stat_key=True,
            default_window="series",
            description="Player's average for this stat in the current playoff series.",
        ),
        _spec(
            "last_game_outlier",
            "player_form",
            "player",
            requires_stat_key=True,
            default_window="last_3",
            value_kind="bool",
            description="Agent judgment that the previous game does not reflect the true level.",
        ),
        _spec(
            "home_away_split",
            "situational",
            "player",
            requires_stat_key=True,
            description="Player's home-vs-away performance delta for this stat.",
        ),
        _spec(
            "opponent_stat_rank",
            "matchup",
            "player",
            requires_stat_key=True,
            description="Opponent's defensive rank against this stat (1=toughest).",
        ),
        _spec(
            "rest_advantage",
            "situational",
            "both",
            default_window="matchup",
            description="Net rest-days advantage relative to the opponent.",
        ),
        _spec(
            "elimination_game",
            "situational",
            "both",
            default_window="matchup",
            value_kind="bool",
            description="Must-win / elimination context.",
        ),
        _spec(
            "motivation_edge",
            "situational",
            "game",
            default_window="matchup",
            description="Net motivational edge (tanking, seeding locked, rivalry, etc.).",
        ),
        _spec(
            "blowout_risk",
            "situational",
            "both",
            default_window="matchup",
            description="Probability the game is non-competitive; compresses prop floors.",
        ),
        _spec(
            "win_streak",
            "team_form",
            "game",
            default_window="last_10",
            description="Length of the current win streak for the directional team.",
        ),
        _spec(
            "series_lead",
            "matchup",
            "game",
            default_window="series",
            description="Games-ahead lead in the current playoff series for the directional team.",
        ),
        _spec(
            "season_record",
            "team_form",
            "game",
            default_window="season",
            description="Season win percentage for the directional team. Audit-only: "
            "team ratings already encode this; no handler registered to avoid double-counting.",
        ),
        _spec(
            "season_baseline",
            "player_form",
            "player",
            requires_stat_key=True,
            default_window="season",
            description="Player's season baseline for the stat. Audit-only: recent_form/series_avg "
            "already drive the live blend; no handler registered.",
        ),
        _spec(
            "defensive_scheme",
            "matchup",
            "game",
            applies_to_sports=frozenset({"basketball", "american_football"}),
            default_window="matchup",
            value_kind="categorical",
            description="Defensive scheme/coverage label (e.g. 'blitz_primary_handler'). "
            "Audit-only: no scheme-to-effect mapping yet.",
        ),
        # --- Basketball / hockey ---
        _spec(
            "usage_spike",
            "player_form",
            "player",
            applies_to_sports=frozenset({"basketball", "hockey"}),
            description="Expected usage increase (injury absence, new role).",
        ),
        _spec(
            "usage_role_change",
            "situational",
            "both",
            applies_to_sports=frozenset({"basketball", "hockey", "american_football", "soccer"}),
            value_kind="categorical",
            description="Starting status / minutes restriction / new lineup role.",
        ),
        _spec(
            "b2b_fatigue",
            "situational",
            "both",
            applies_to_sports=frozenset({"basketball", "hockey"}),
            default_window="matchup",
            value_kind="bool",
            description="Back-to-back fatigue (played the previous night).",
        ),
        _spec(
            "pace_up",
            "matchup",
            "both",
            applies_to_sports=frozenset({"basketball", "hockey", "soccer"}),
            default_window="matchup",
            description="Matchup expected to run faster than league baseline.",
        ),
        _spec(
            "pace_down",
            "matchup",
            "both",
            applies_to_sports=frozenset({"basketball", "hockey", "soccer"}),
            default_window="matchup",
            description="Matchup expected to run slower than league baseline.",
        ),
        # --- Defensive matchup (team sports) ---
        _spec(
            "def_matchup_weak",
            "matchup",
            "both",
            applies_to_sports=frozenset(
                {"basketball", "american_football", "hockey", "soccer"}
            ),
            requires_stat_key=True,
            description=(
                "Opponent is a weak defender of this stat for props, or weak against "
                "the directional team offense for Markov game analysis."
            ),
        ),
        _spec(
            "def_matchup_strong",
            "matchup",
            "both",
            applies_to_sports=frozenset(
                {"basketball", "american_football", "hockey", "soccer"}
            ),
            requires_stat_key=True,
            description=(
                "Opponent is a strong defender of this stat for props, or strong against "
                "the directional team offense for Markov game analysis."
            ),
        ),
        # --- Baseball ---
        _spec(
            "park_factor_evidence",
            "situational",
            "both",
            applies_to_sports=frozenset({"baseball"}),
            default_window="matchup",
            description="Ballpark hitter/pitcher friendliness for this matchup.",
        ),
        _spec(
            "weather_wind",
            "situational",
            "both",
            applies_to_sports=frozenset({"baseball"}),
            default_window="matchup",
            description="Wind speed/direction effect on the run environment.",
        ),
        _spec(
            "pitcher_matchup",
            "matchup",
            "player",
            applies_to_sports=frozenset({"baseball"}),
            requires_stat_key=True,
            description="Batter-vs-pitcher handedness / history edge.",
        ),
        _spec(
            "starter_era",
            "matchup",
            "game",
            applies_to_sports=frozenset({"baseball"}),
            default_window="season",
            description="Starting pitcher's ERA for the directional team.",
        ),
        # --- American football ---
        _spec(
            "weather_cold",
            "situational",
            "both",
            applies_to_sports=frozenset({"american_football"}),
            default_window="matchup",
            description="Cold/adverse weather suppressing passing output.",
        ),
        _spec(
            "dome_effect",
            "situational",
            "game",
            applies_to_sports=frozenset({"american_football"}),
            default_window="matchup",
            value_kind="bool",
            description="Indoor venue removing weather variance.",
        ),
        # --- Soccer / tennis / golf / combat / esports ---
        _spec(
            "formation_mismatch",
            "matchup",
            "game",
            applies_to_sports=frozenset({"soccer"}),
            default_window="matchup",
            description="Tactical formation advantage for one side.",
        ),
        _spec(
            "surface_edge",
            "matchup",
            "game",
            applies_to_sports=frozenset({"tennis"}),
            default_window="matchup",
            description="Player's edge on the current court surface.",
        ),
        _spec(
            "course_fit",
            "matchup",
            "game",
            applies_to_sports=frozenset({"golf"}),
            default_window="matchup",
            description="Player's historical fit for this course profile.",
        ),
        _spec(
            "stylistic_matchup",
            "matchup",
            "game",
            applies_to_sports=frozenset({"fighting"}),
            default_window="matchup",
            description="Striker-vs-grappler style advantage.",
        ),
        _spec(
            "map_pool_edge",
            "matchup",
            "game",
            applies_to_sports=frozenset({"esports"}),
            default_window="matchup",
            description="Team's map-pool advantage for this series.",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_archetype(league: str) -> str | None:
    """Return the sport archetype name for a league, reusing the engine map.

    Thin wrapper over ``archetypes.get_archetype_name`` so callers do not
    duplicate the league->archetype table.
    """
    return get_archetype_name(league)


def signal_applies(signal_type: str, archetype: str | None) -> bool:
    """Return True when ``signal_type`` is meaningful for ``archetype``.

    Unknown signal types and unmapped archetypes return False — the engine
    gates handler dispatch on this so a signal never adjusts a sport it does
    not fit.
    """
    spec = SIGNAL_REGISTRY.get(signal_type)
    if spec is None or archetype is None:
        return False
    if UNIVERSAL in spec.applies_to_sports:
        return True
    return archetype in spec.applies_to_sports


def signal_applies_to_league(signal_type: str, league: str) -> bool:
    """Convenience: ``signal_applies`` resolved straight from a league code."""
    return signal_applies(signal_type, resolve_archetype(league))
