"""
Sport Archetype Registry

Defines simulation model archetypes for all supported sports. Each archetype
encapsulates the mathematical model, required inputs, supported markets,
and stat taxonomy for a category of sports.

Archetypes:
    basketball   - ORtg/DRtg/pace possession model (Normal)
    american_football - Points/drives efficiency model (Normal + Poisson hybrid)
    baseball     - Run environment model (Poisson), pitcher-aware
    hockey       - Goal/shot model with goalie (Poisson), 3-way regulation
    soccer       - Goal model (Poisson), 3-way result, xG-based
    tennis       - Point-level win probability, best-of-N sets
    golf         - Field probability model, finishing position distribution
    fighting     - Win probability + method-of-victory (KO/TKO/decision/sub)
    esports      - Map-based win probability, best-of-N maps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Archetype definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SportArchetype:
    """Immutable definition of a sport simulation archetype."""

    name: str
    display_name: str
    score_distribution: str  # "normal", "poisson", "bernoulli", "custom"
    supports_draw: bool
    result_type: str  # "team_score", "individual_matchup", "field", "combat"

    # Required context keys the agent must supply for team/participant A and B
    required_team_keys: Tuple[str, ...]
    optional_team_keys: Tuple[str, ...] = ()

    # Critical subset of required keys: missing ANY one prevents formal edge output.
    # Important keys (required minus critical) allow sim but cap confidence at C.
    critical_team_keys: Tuple[str, ...] = ()

    # Required player/participant context keys
    required_player_keys: Tuple[str, ...] = ()
    optional_player_keys: Tuple[str, ...] = ()

    # Stat keys available for props
    prop_stat_keys: Tuple[str, ...] = ()

    # Market types this archetype supports
    supported_markets: Tuple[str, ...] = (
        "moneyline", "spread", "total",
    )

    # Scoring semantics
    score_unit: str = "points"
    avg_total: float = 100.0
    default_std: float = 12.0

    # Tempo / opportunity semantics
    tempo_unit: str = "possessions"
    avg_tempo: float = 100.0

    # For best-of-N formats (tennis, esports)
    best_of: Optional[int] = None
    segment_name: str = "period"


# ---------------------------------------------------------------------------
# Archetype instances
# ---------------------------------------------------------------------------

BASKETBALL = SportArchetype(
    name="basketball",
    display_name="Basketball",
    score_distribution="normal",
    supports_draw=False,
    result_type="team_score",
    required_team_keys=("off_rating", "def_rating", "pace"),
    critical_team_keys=("off_rating", "def_rating"),
    optional_team_keys=("fg_pct", "three_pt_pct", "ft_pct", "turnover_rate",
                        "off_reb_pct", "home_court_adj"),
    required_player_keys=("name",),
    optional_player_keys=("usage_rate", "minutes", "pts_mean", "pts_std",
                          "reb_mean", "ast_mean", "three_pm_mean"),
    prop_stat_keys=("pts", "reb", "ast", "3pm", "pra", "stl", "blk",
                    "pts_reb", "pts_ast", "reb_ast"),
    supported_markets=("moneyline", "spread", "total", "team_total",
                       "1h_spread", "1h_total", "1q_spread", "1q_total"),
    score_unit="points",
    avg_total=224.0,
    default_std=12.0,
    tempo_unit="possessions",
    avg_tempo=100.0,
    segment_name="quarter",
)

AMERICAN_FOOTBALL = SportArchetype(
    name="american_football",
    display_name="American Football",
    score_distribution="normal",
    supports_draw=False,
    result_type="team_score",
    required_team_keys=("off_rating", "def_rating"),
    critical_team_keys=("off_rating", "def_rating"),
    optional_team_keys=("pace", "pass_eff", "rush_eff", "turnover_diff",
                        "red_zone_pct", "third_down_pct"),
    required_player_keys=("name", "position"),
    optional_player_keys=("pass_yards_mean", "rush_yards_mean", "rec_yards_mean",
                          "receptions_mean", "td_mean", "target_share",
                          "carry_share", "snap_pct"),
    prop_stat_keys=("pass_yds", "rush_yds", "rec_yds", "receptions",
                    "pass_td", "rush_td", "rec_td", "completions",
                    "interceptions", "longest_reception", "longest_rush"),
    supported_markets=("moneyline", "spread", "total", "team_total",
                       "1h_spread", "1h_total", "1q_spread", "1q_total"),
    score_unit="points",
    avg_total=45.0,
    default_std=10.0,
    tempo_unit="plays",
    avg_tempo=130.0,
    segment_name="quarter",
)

BASEBALL = SportArchetype(
    name="baseball",
    display_name="Baseball",
    score_distribution="poisson",
    supports_draw=False,
    result_type="team_score",
    required_team_keys=("off_rating", "def_rating"),
    critical_team_keys=("off_rating", "def_rating"),
    optional_team_keys=("pace", "batting_avg", "obp", "slg", "era",
                        "whip", "bullpen_era", "park_factor"),
    required_player_keys=("name", "role"),  # role: "pitcher" or "batter"
    optional_player_keys=("batting_avg", "hr_rate", "k_rate", "bb_rate",
                          "era", "whip", "k_per_9", "outs_recorded_mean",
                          "hits_mean", "total_bases_mean", "rbis_mean"),
    prop_stat_keys=("hits", "total_bases", "runs", "rbis", "hrs",
                    "stolen_bases", "strikeouts_pitched", "outs_recorded",
                    "hits_allowed", "walks_allowed", "earned_runs"),
    supported_markets=("moneyline", "spread", "total", "team_total",
                       "1h_moneyline", "1h_spread", "1h_total",
                       "first_5_innings_ml", "first_5_innings_total"),
    score_unit="runs",
    avg_total=8.5,
    default_std=3.0,
    tempo_unit="plate_appearances",
    avg_tempo=38.0,
    segment_name="inning",
)

HOCKEY = SportArchetype(
    name="hockey",
    display_name="Hockey",
    score_distribution="poisson",
    supports_draw=True,  # regulation can draw → OT decides
    result_type="team_score",
    required_team_keys=("off_rating", "def_rating"),
    critical_team_keys=("off_rating", "def_rating"),
    optional_team_keys=("pace", "shots_per_game", "save_pct",
                        "pp_pct", "pk_pct", "xgf_per_60", "xga_per_60",
                        "goalie_sv_pct", "goalie_gsax"),
    required_player_keys=("name",),
    optional_player_keys=("goals_mean", "assists_mean", "shots_mean",
                          "toi_share", "pp_toi_share", "saves_mean"),
    prop_stat_keys=("goals", "assists", "points", "shots_on_goal",
                    "saves", "blocked_shots", "power_play_points"),
    supported_markets=("moneyline", "puck_line", "total", "team_total",
                       "regulation_ml", "1p_total", "1p_ml"),
    score_unit="goals",
    avg_total=6.0,
    default_std=2.0,
    tempo_unit="shots",
    avg_tempo=62.0,
    segment_name="period",
)

SOCCER = SportArchetype(
    name="soccer",
    display_name="Soccer",
    score_distribution="poisson",
    supports_draw=True,
    result_type="team_score",
    required_team_keys=("off_rating", "def_rating"),
    critical_team_keys=("off_rating", "def_rating"),
    optional_team_keys=("pace", "xg_for", "xg_against", "possession_pct",
                        "shots_per_game", "shots_on_target", "corners_per_game"),
    required_player_keys=("name",),
    optional_player_keys=("goals_mean", "assists_mean", "shots_mean",
                          "shots_on_target_mean", "minutes_mean"),
    prop_stat_keys=("goals", "assists", "shots", "shots_on_target",
                    "tackles", "fouls_committed", "corners"),
    supported_markets=("moneyline_3way", "spread", "total", "team_total",
                       "double_chance", "draw_no_bet", "both_teams_to_score",
                       "1h_moneyline_3way", "1h_total", "correct_score"),
    score_unit="goals",
    avg_total=2.5,
    default_std=1.3,
    tempo_unit="possessions",
    avg_tempo=50.0,
    segment_name="half",
)

TENNIS = SportArchetype(
    name="tennis",
    display_name="Tennis",
    score_distribution="bernoulli",  # point-by-point probability
    supports_draw=False,
    result_type="individual_matchup",
    required_team_keys=("serve_win_pct", "return_win_pct"),
    critical_team_keys=("serve_win_pct", "return_win_pct"),
    optional_team_keys=("ace_rate", "double_fault_rate", "first_serve_pct",
                        "break_point_conversion", "surface_adj",
                        "elo_rating", "fatigue_factor"),
    required_player_keys=("name",),
    optional_player_keys=("aces_mean", "double_faults_mean",
                          "games_won_mean", "tiebreaks_won_pct"),
    prop_stat_keys=("aces", "double_faults", "total_games",
                    "sets_won", "tiebreaks"),
    supported_markets=("moneyline", "set_spread", "total_games",
                       "total_sets", "set_betting", "first_set_winner"),
    score_unit="sets",
    avg_total=3.0,  # best-of-3 average
    default_std=0.5,
    tempo_unit="points",
    avg_tempo=180.0,
    best_of=3,
    segment_name="set",
)

GOLF = SportArchetype(
    name="golf",
    display_name="Golf",
    score_distribution="normal",  # strokes relative to par
    supports_draw=False,
    result_type="field",
    required_team_keys=("strokes_gained_total",),
    critical_team_keys=("strokes_gained_total",),
    optional_team_keys=("sg_off_tee", "sg_approach", "sg_around_green",
                        "sg_putting", "course_fit", "recent_form",
                        "elo_rating", "driving_distance", "gir_pct"),
    required_player_keys=("name",),
    optional_player_keys=("scoring_avg", "top_10_pct", "made_cut_pct",
                          "win_pct"),
    prop_stat_keys=("finishing_position", "top_5", "top_10", "top_20",
                    "make_cut", "first_round_leader", "round_score"),
    supported_markets=("outright_winner", "top_5", "top_10", "top_20",
                       "make_cut", "matchup", "first_round_leader",
                       "round_3_balls"),
    score_unit="strokes",
    avg_total=280.0,  # 4-round tournament total
    default_std=4.0,  # per-round std
    tempo_unit="holes",
    avg_tempo=72.0,
    segment_name="round",
)

FIGHTING = SportArchetype(
    name="fighting",
    display_name="Fighting",
    score_distribution="bernoulli",  # win/loss outcome
    supports_draw=True,  # draws possible in boxing, rare in MMA
    result_type="combat",
    required_team_keys=("win_pct", "finish_rate"),
    critical_team_keys=("win_pct",),
    optional_team_keys=("ko_tko_rate", "submission_rate", "decision_rate",
                        "sig_strikes_per_min", "sig_strike_accuracy",
                        "takedown_avg", "takedown_defense",
                        "reach", "stance", "age", "activity_rate",
                        "rounds_scheduled"),
    required_player_keys=("name",),
    optional_player_keys=("ko_tko_rate", "decision_rate", "submission_rate",
                          "avg_fight_time", "sig_strikes_landed_mean",
                          "takedowns_mean"),
    prop_stat_keys=("method_of_victory", "round_of_finish",
                    "total_rounds", "sig_strikes", "takedowns",
                    "fight_to_go_distance"),
    supported_markets=("moneyline", "method_of_victory",
                       "total_rounds", "round_betting",
                       "fight_to_go_distance", "round_group"),
    score_unit="rounds",
    avg_total=3.0,  # 3-round fight
    default_std=1.0,
    tempo_unit="minutes",
    avg_tempo=15.0,  # 3 x 5min rounds
    segment_name="round",
)

ESPORTS = SportArchetype(
    name="esports",
    display_name="Esports",
    score_distribution="bernoulli",  # map win probability
    supports_draw=False,
    result_type="individual_matchup",  # best-of-N maps
    required_team_keys=("map_win_rate", "recent_form"),
    critical_team_keys=("map_win_rate",),
    optional_team_keys=("win_rate_by_map", "avg_round_diff",
                        "first_blood_rate", "side_win_rates",
                        "elo_rating", "roster_stability"),
    required_player_keys=("name",),
    optional_player_keys=("kills_mean", "deaths_mean", "assists_mean",
                          "kd_ratio", "adr", "rating"),
    prop_stat_keys=("kills", "deaths", "assists", "headshots",
                    "total_rounds", "total_maps"),
    supported_markets=("moneyline", "map_spread", "total_maps",
                       "map_winner", "first_map_winner",
                       "total_rounds"),
    score_unit="maps",
    avg_total=2.5,
    default_std=0.5,
    tempo_unit="rounds",
    avg_tempo=25.0,
    best_of=3,
    segment_name="map",
)


# ---------------------------------------------------------------------------
# Archetype registry & league mapping
# ---------------------------------------------------------------------------

ARCHETYPE_REGISTRY: Dict[str, SportArchetype] = {
    "basketball": BASKETBALL,
    "american_football": AMERICAN_FOOTBALL,
    "baseball": BASEBALL,
    "hockey": HOCKEY,
    "soccer": SOCCER,
    "tennis": TENNIS,
    "golf": GOLF,
    "fighting": FIGHTING,
    "esports": ESPORTS,
}

# Maps every known league code → archetype name
LEAGUE_TO_ARCHETYPE: Dict[str, str] = {
    # Basketball
    "NBA": "basketball",
    "WNBA": "basketball",
    "NCAAB": "basketball",
    "NCAAM": "basketball",
    "FIBA": "basketball",
    "EUROLEAGUE": "basketball",
    "NBL": "basketball",
    # American Football
    "NFL": "american_football",
    "NCAAF": "american_football",
    "CFL": "american_football",
    "XFL": "american_football",
    "UFL": "american_football",
    # Baseball
    "MLB": "baseball",
    "NPB": "baseball",
    "KBO": "baseball",
    "NCAA_BASEBALL": "baseball",
    # Hockey
    "NHL": "hockey",
    "KHL": "hockey",
    "SHL": "hockey",
    "AHL": "hockey",
    "IIHF": "hockey",
    # Soccer
    "MLS": "soccer",
    "EPL": "soccer",
    "PREMIER_LEAGUE": "soccer",
    "LA_LIGA": "soccer",
    "LALIGA": "soccer",
    "BUNDESLIGA": "soccer",
    "SERIE_A": "soccer",
    "LIGUE_1": "soccer",
    "LIGA_MX": "soccer",
    "CHAMPIONS_LEAGUE": "soccer",
    "EUROPA_LEAGUE": "soccer",
    "WORLD_CUP": "soccer",
    "NWSL": "soccer",
    "A_LEAGUE": "soccer",
    "EREDIVISIE": "soccer",
    "PRIMEIRA_LIGA": "soccer",
    "SUPER_LIG": "soccer",
    "SCOTTISH_PREMIERSHIP": "soccer",
    "CHAMPIONSHIP": "soccer",
    "LEAGUE_ONE": "soccer",
    "LEAGUE_TWO": "soccer",
    "COPA_LIBERTADORES": "soccer",
    "COPA_AMERICA": "soccer",
    "EURO": "soccer",
    "NATIONS_LEAGUE": "soccer",
    # Tennis
    "ATP": "tennis",
    "WTA": "tennis",
    "GRAND_SLAM": "tennis",
    "AUSTRALIAN_OPEN": "tennis",
    "FRENCH_OPEN": "tennis",
    "WIMBLEDON": "tennis",
    "US_OPEN_TENNIS": "tennis",
    "DAVIS_CUP": "tennis",
    # Golf
    "PGA": "golf",
    "PGA_TOUR": "golf",
    "LPGA": "golf",
    "EUROPEAN_TOUR": "golf",
    "DP_WORLD_TOUR": "golf",
    "LIV": "golf",
    "MASTERS": "golf",
    "US_OPEN_GOLF": "golf",
    "OPEN_CHAMPIONSHIP": "golf",
    "PGA_CHAMPIONSHIP": "golf",
    # Fighting
    "UFC": "fighting",
    "MMA": "fighting",
    "BOXING": "fighting",
    "BELLATOR": "fighting",
    "PFL": "fighting",
    "ONE_FC": "fighting",
    # Esports
    "ESPORTS": "esports",
    "CS2": "esports",
    "CSGO": "esports",
    "LOL": "esports",
    "DOTA2": "esports",
    "VALORANT": "esports",
    "OVERWATCH": "esports",
    "ROCKET_LEAGUE": "esports",
    "COD": "esports",
    "STARCRAFT": "esports",
}


def get_archetype(league: str) -> Optional[SportArchetype]:
    """Return the archetype for a given league, or None if unmapped."""
    archetype_name = LEAGUE_TO_ARCHETYPE.get(league.upper())
    if archetype_name is None:
        return None
    return ARCHETYPE_REGISTRY.get(archetype_name)


def get_archetype_name(league: str) -> Optional[str]:
    """Return just the archetype name string for a league."""
    return LEAGUE_TO_ARCHETYPE.get(league.upper())


def get_required_inputs(league: str) -> List[str]:
    """Return the list of required team context keys for a league.

    Useful for building missing_requirements responses.
    """
    archetype = get_archetype(league)
    if archetype is None:
        return []
    return list(archetype.required_team_keys)


def get_critical_inputs(league: str) -> List[str]:
    """Return the critical team context keys for a league.

    Missing any critical input should prevent formal edge output.
    """
    archetype = get_archetype(league)
    if archetype is None:
        return []
    return list(archetype.critical_team_keys)


def get_important_inputs(league: str) -> List[str]:
    """Return important (required but non-critical) team context keys.

    These are required_team_keys minus critical_team_keys.
    Missing these caps confidence at tier C but allows simulation.
    """
    archetype = get_archetype(league)
    if archetype is None:
        return []
    critical = set(archetype.critical_team_keys)
    return [k for k in archetype.required_team_keys if k not in critical]


def get_supported_markets(league: str) -> List[str]:
    """Return the list of market types an archetype supports."""
    archetype = get_archetype(league)
    if archetype is None:
        return []
    return list(archetype.supported_markets)


def get_prop_stat_keys(league: str) -> List[str]:
    """Return available prop stat keys for a league."""
    archetype = get_archetype(league)
    if archetype is None:
        return []
    return list(archetype.prop_stat_keys)
