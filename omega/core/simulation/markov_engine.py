"""Deterministic Markov state-transition simulator.

This module owns possession-level stochastic state transitions. It is deliberately
input-driven: callers provide contexts, players, and optional scalar modifiers;
the simulator performs no network or evidence gathering.

Momentum note: momentum is a single-possession hot-hand signal, not a cumulative
streak counter. After each possession it is reset to ±1 (scored) or 0 (no score).
Callers should not expect multi-possession momentum accumulation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from omega.core.config.leagues import get_league_config
from omega.core.simulation.archetypes import get_archetype, get_archetype_name

COMPONENT_VERSION = "markov_state_v1"

# Transition modifiers outside these bounds are clamped and logged.
# This prevents extreme scalars from corrupting terminal score distributions
# while maintaining a clear audit trail.
_MODIFIER_BOUNDS: tuple[float, float] = (0.05, 10.0)

_log = logging.getLogger(__name__)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weighted_choice(weights: dict[str, float]) -> str:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        return next(iter(weights))
    pick = random.random() * total
    running = 0.0
    for key, weight in weights.items():
        running += max(0.0, weight)
        if pick <= running:
            return key
    return next(reversed(weights))


def _points_probabilities(expected_ppp: float) -> dict[int, float]:
    """Convert expected points per possession into 0/1/2/3-point probabilities."""
    mean = _clamp(expected_ppp, 0.15, 2.4)
    p3 = _clamp(0.10 + (mean - 1.0) * 0.08, 0.02, 0.24)
    p1 = _clamp(0.07 + (mean - 1.0) * 0.03, 0.02, 0.16)
    p2 = _clamp((mean - 3.0 * p3 - p1) / 2.0, 0.02, 0.82)
    p0 = max(0.02, 1.0 - p1 - p2 - p3)
    total = p0 + p1 + p2 + p3
    return {0: p0 / total, 1: p1 / total, 2: p2 / total, 3: p3 / total}


@dataclass
class MarkovGameState:
    """Terminal game state returned by one Markov simulation run."""

    home_score: float = 0.0
    away_score: float = 0.0
    player_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def get_player_stat(self, player_name: str, stat_key: str) -> float:
        return float(self.player_stats.get(player_name, {}).get(stat_key, 0.0))


class MarkovSimulator:
    """Possession-level Markov simulator with modifier-ready transitions."""

    component_version = COMPONENT_VERSION

    def __init__(
        self,
        *,
        league: str,
        players: list[dict[str, Any]] | None = None,
        home_context: dict[str, Any] | None = None,
        away_context: dict[str, Any] | None = None,
        transition_modifiers: dict[str, float] | None = None,
    ) -> None:
        self.league = league.upper()
        self.players = players or []
        self.home_context = home_context or {}
        self.away_context = away_context or {}
        self.transition_modifiers = self._clamp_modifiers(transition_modifiers or {})
        self.config = get_league_config(self.league)
        self.archetype = get_archetype(self.league)
        self.archetype_name = get_archetype_name(self.league)
        self._base_n_possessions = self._resolve_base_possessions()
        self.transition_matrix_ids = {
            "home": f"{COMPONENT_VERSION}:{self.league}:home_score_outcome",
            "away": f"{COMPONENT_VERSION}:{self.league}:away_score_outcome",
            "momentum": f"{COMPONENT_VERSION}:{self.league}:momentum_state",
            "player": f"{COMPONENT_VERSION}:{self.league}:player_stat_allocation",
        }

    @staticmethod
    def _clamp_modifiers(raw: dict[str, float]) -> dict[str, float]:
        lo, hi = _MODIFIER_BOUNDS
        clamped: dict[str, float] = {}
        for key, val in raw.items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                _log.warning("transition_modifier %r has non-numeric value %r; skipping", key, val)
                continue
            if not (lo <= fval <= hi):
                clamped_val = max(lo, min(hi, fval))
                _log.warning(
                    "transition_modifier %r=%r clamped to %r (bounds [%s, %s])",
                    key, fval, clamped_val, lo, hi,
                )
                clamped[key] = clamped_val
            else:
                clamped[key] = fval
        return clamped

    def _resolve_base_possessions(self) -> int:
        """Compute total simulation loop iterations.

        ``pace`` in team context (and ``avg_pace`` in league config) represents
        **per-team possessions per game** (e.g. NBA ≈ 100 per team).  The
        simulation loop alternates home / away, so the total iteration count
        must equal ``home_pace + away_pace`` to give each team its full
        possession allocation.

        American football: NFL play counts are roughly half of basketball pace
        values in absolute terms; the ``/ 2.0`` preserves the existing
        calibration until dedicated NFL sweep data is available.
        """
        cfg_pace = _as_float(self.config.get("avg_pace"), 100.0)
        home_pace = _as_float(self.home_context.get("pace"), cfg_pace)
        away_pace = _as_float(self.away_context.get("pace"), cfg_pace)
        # Sum both teams' per-team possession counts so the alternating loop
        # gives each team its full allocation (fixes the previous /2 halving bug).
        possessions = home_pace + away_pace
        if self.archetype_name == "american_football":
            # NFL context paces are expressed in a different unit; halve to
            # avoid doubling the existing calibration until NFL sweep validates.
            possessions = possessions / 2.0
        return max(1, int(round(possessions)))

    def _expected_ppp(self, side: str, momentum: int) -> float:
        if side == "home":
            off = _as_float(self.home_context.get("off_rating"), 100.0)
            opp_def = _as_float(self.away_context.get("def_rating"), 100.0)
            scalar = self.transition_modifiers.get("home_score_rate_scalar", 1.0)
        else:
            off = _as_float(self.away_context.get("off_rating"), 100.0)
            opp_def = _as_float(self.home_context.get("def_rating"), 100.0)
            scalar = self.transition_modifiers.get("away_score_rate_scalar", 1.0)

        if self.archetype_name in {"basketball", "american_football"}:
            expected = (off + opp_def) / 200.0
        elif self.archetype_name in {"baseball", "hockey", "soccer"}:
            expected = max(0.02, (off + opp_def) / 200.0)
        else:
            expected = _as_float(self.config.get("avg_total"), 2.0) / max(
                1.0,
                self._base_n_possessions,
            )

        if side == "home":
            expected += _as_float(self.config.get("home_advantage"), 0.0) / 100.0
        if momentum > 0 and side == "home":
            expected *= self.transition_modifiers.get("home_momentum_scalar", 1.03)
        elif momentum < 0 and side == "away":
            expected *= self.transition_modifiers.get("away_momentum_scalar", 1.03)
        return max(0.01, expected * scalar)

    def _sample_points(self, side: str, momentum: int) -> int:
        probs = _points_probabilities(self._expected_ppp(side, momentum))
        outcome = _weighted_choice({str(points): prob for points, prob in probs.items()})
        return int(outcome)

    def _player_stat(self, player: dict[str, Any], stat_key: str) -> float:
        mean = _as_float(player.get(f"{stat_key}_mean"), 0.0)
        std = _as_float(player.get(f"{stat_key}_std"), max(1.0, mean**0.5))
        if mean <= 0:
            return 0.0
        if stat_key in {"pts", "reb", "ast", "3pm", "stl", "blk"}:
            return max(0.0, round(random.gauss(mean, std), 1))
        return max(0.0, random.gauss(mean, std))

    def _simulate_player_stats(self) -> dict[str, dict[str, float]]:
        player_stats: dict[str, dict[str, float]] = {}
        stat_keys = self.archetype.prop_stat_keys if self.archetype is not None else ("pts",)
        for player in self.players:
            name = str(player.get("name") or "").strip()
            if not name:
                continue
            stats: dict[str, float] = {}
            for stat_key in stat_keys:
                if f"{stat_key}_mean" in player:
                    stats[stat_key] = self._player_stat(player, stat_key)
            player_stats[name] = stats
        return player_stats

    def run_diagnostics(self, n_games: int = 500, *, seed: int | None = 0) -> dict[str, float]:
        """Return calibration statistics from a sample run of n_games.

        Intended for sweep scripts and calibration notebooks, not production.
        Sets the RNG to a fixed seed so diagnostics are reproducible.

        Returns:
            Dict with keys:
              base_possessions, mean_home_score, mean_away_score, mean_total,
              std_total, mean_spread, expected_ppp_home, expected_ppp_away
        """
        if seed is not None:
            import random as _r

            _r.seed(seed)
        home_scores: list[float] = []
        away_scores: list[float] = []
        for _ in range(n_games):
            state = self.simulate_game()
            home_scores.append(state.home_score)
            away_scores.append(state.away_score)
        totals = [h + a for h, a in zip(home_scores, away_scores)]
        spreads = [h - a for h, a in zip(home_scores, away_scores)]
        n = len(home_scores)
        mean_h = sum(home_scores) / n
        mean_a = sum(away_scores) / n
        mean_t = sum(totals) / n
        mean_s = sum(spreads) / n
        var_t = sum((t - mean_t) ** 2 for t in totals) / n
        return {
            "base_possessions": self._base_n_possessions,
            "mean_home_score": round(mean_h, 2),
            "mean_away_score": round(mean_a, 2),
            "mean_total": round(mean_t, 2),
            "std_total": round(var_t ** 0.5, 2),
            "mean_spread": round(mean_s, 2),
            "expected_ppp_home": round(self._expected_ppp("home", 0), 4),
            "expected_ppp_away": round(self._expected_ppp("away", 0), 4),
        }

    def simulate_game(self, n_possessions: int | None = None) -> MarkovGameState:
        possessions = max(1, int(n_possessions or self._base_n_possessions))
        home_score = 0.0
        away_score = 0.0
        momentum = 0
        side = "home" if random.random() < 0.5 else "away"

        for _ in range(possessions):
            points = self._sample_points(side, momentum)
            if side == "home":
                home_score += points
                momentum = 1 if points > 0 else 0
                side = "away"
            else:
                away_score += points
                momentum = -1 if points > 0 else 0
                side = "home"

        return MarkovGameState(
            home_score=home_score,
            away_score=away_score,
            player_stats=self._simulate_player_stats(),
        )
