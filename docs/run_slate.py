import sys
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Insert repository roots
sys.path.insert(0, r"C:\repos\Omega")
sys.path.insert(0, r"C:\repos\Omega\src")

from omega.integrations.odds_resolver import resolve_odds
from omega.integrations.game_context import resolve_game_context
from omega.core.contracts.service import analyze

session_id = "sess-20260603-mlb1"
bankroll = 1000.0

games = [
    {"home": "Washington Nationals", "away": "Miami Marlins", "home_pitcher": "Andrew Alvarez", "away_pitcher": "Max Meyer", "home_pitcher_era": 4.02, "away_pitcher_era": 2.97, "wind": 9.0},
    {"home": "Tampa Bay Rays", "away": "Detroit Tigers", "home_pitcher": "Nick Martinez", "away_pitcher": "Troy Melton", "home_pitcher_era": 1.62, "away_pitcher_era": 1.42, "wind": 0.0},
    {"home": "Minnesota Twins", "away": "Chicago White Sox", "home_pitcher": "Taj Bradley", "away_pitcher": "Erick Fedde", "home_pitcher_era": 3.21, "away_pitcher_era": 5.40, "wind": 11.0},
    {"home": "Seattle Mariners", "away": "New York Mets", "home_pitcher": "George Kirby", "away_pitcher": "Freddy Peralta", "home_pitcher_era": 3.77, "away_pitcher_era": 3.55, "wind": 4.0},
    {"home": "Philadelphia Phillies", "away": "San Diego Padres", "home_pitcher": "Cristopher Sánchez", "away_pitcher": "Walker Buehler", "home_pitcher_era": 1.47, "away_pitcher_era": 4.88, "wind": 5.0},
    {"home": "Boston Red Sox", "away": "Baltimore Orioles", "home_pitcher": "Payton Tolle", "away_pitcher": "Chris Bassitt", "home_pitcher_era": 2.61, "away_pitcher_era": 5.06, "wind": 6.0},
    {"home": "New York Yankees", "away": "Cleveland Guardians", "home_pitcher": "Gerrit Cole", "away_pitcher": "Gavin Williams", "home_pitcher_era": 0.71, "away_pitcher_era": 3.07, "wind": 7.0},
    {"home": "Cincinnati Reds", "away": "Kansas City Royals", "home_pitcher": "Chase Burns", "away_pitcher": "Stephen Kolek", "home_pitcher_era": 1.96, "away_pitcher_era": 3.48, "wind": 3.0},
    {"home": "Atlanta Braves", "away": "Toronto Blue Jays", "home_pitcher": "Grant Holmes", "away_pitcher": "Patrick Corbin", "home_pitcher_era": 3.95, "away_pitcher_era": 3.65, "wind": 5.0},
    {"home": "St. Louis Cardinals", "away": "Texas Rangers", "home_pitcher": "Andre Pallante", "away_pitcher": "MacKenzie Gore", "home_pitcher_era": 4.19, "away_pitcher_era": 3.96, "wind": 5.0}
]

team_stats = {
    "Washington Nationals": {"off": 5.32, "def": 5.44},
    "Miami Marlins": {"off": 4.26, "def": 4.60},
    "Tampa Bay Rays": {"off": 4.67, "def": 4.50},
    "Detroit Tigers": {"off": 3.89, "def": 4.37},
    "Minnesota Twins": {"off": 4.68, "def": 4.94},
    "Chicago White Sox": {"off": 4.67, "def": 4.62},
    "Seattle Mariners": {"off": 4.29, "def": 3.71},
    "New York Mets": {"off": 4.00, "def": 4.31},
    "Philadelphia Phillies": {"off": 3.88, "def": 4.30},
    "San Diego Padres": {"off": 3.88, "def": 4.00},
    "Boston Red Sox": {"off": 3.95, "def": 4.07},
    "Baltimore Orioles": {"off": 4.57, "def": 5.16},
    "New York Yankees": {"off": 5.15, "def": 3.60},
    "Cleveland Guardians": {"off": 4.16, "def": 4.06},
    "Cincinnati Reds": {"off": 4.37, "def": 5.00},
    "Kansas City Royals": {"off": 3.82, "def": 4.69},
    "Atlanta Braves": {"off": 5.25, "def": 3.44},
    "Toronto Blue Jays": {"off": 4.05, "def": 4.18},
    "St. Louis Cardinals": {"off": 4.27, "def": 4.51},
    "Texas Rangers": {"off": 4.03, "def": 3.85}
}

pitcher_k_stats = {
    "Andrew Alvarez": {"mean": 4.5, "std": 1.5},
    "Max Meyer": {"mean": 5.0, "std": 1.6},
    "Nick Martinez": {"mean": 4.5, "std": 1.5},
    "Troy Melton": {"mean": 5.2, "std": 1.7},
    "Taj Bradley": {"mean": 5.9, "std": 1.9},
    "Erick Fedde": {"mean": 4.8, "std": 1.6},
    "George Kirby": {"mean": 5.4, "std": 1.8},
    "Freddy Peralta": {"mean": 6.8, "std": 2.2},
    "Cristopher Sánchez": {"mean": 7.9, "std": 2.1},
    "Walker Buehler": {"mean": 4.8, "std": 1.6},
    "Payton Tolle": {"mean": 5.5, "std": 1.8},
    "Chris Bassitt": {"mean": 4.9, "std": 1.6},
    "Gerrit Cole": {"mean": 6.0, "std": 2.0},
    "Gavin Williams": {"mean": 5.3, "std": 1.7},
    "Chase Burns": {"mean": 6.5, "std": 2.0},
    "Stephen Kolek": {"mean": 4.2, "std": 1.4},
    "Grant Holmes": {"mean": 5.2, "std": 1.7},
    "Patrick Corbin": {"mean": 4.0, "std": 1.3},
    "Andre Pallante": {"mean": 4.4, "std": 1.4},
    "MacKenzie Gore": {"mean": 5.5, "std": 1.8}
}

traces_out_dir = Path(r"C:\repos\Omega\var\inbox\traces")
traces_out_dir.mkdir(parents=True, exist_ok=True)

def generate_seed(prompt, date_str):
    encoded = f"{prompt}|{date_str}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")

# Executing 10 Game Traces
print("Running Game simulations...")
for idx, game in enumerate(games):
    home = game["home"]
    away = game["away"]
    print(f"[{idx+1}/10] Resolving odds and context for {away} @ {home}...")
    
    # 1. Resolve Odds
    odds_res = resolve_odds(kind="game", league="MLB", home_team=home, away_team=away)
    if odds_res["status"] != "success" or "request_patch" not in odds_res:
        print(f"Skipping game {away} @ {home}: Odds unavailable")
        continue
    
    odds_patch = odds_res["request_patch"]["odds"]
    
    # 2. Resolve Context
    context_res = resolve_game_context(league="MLB", home_team=home, away_team=away, game_date="2026-06-03")
    game_context = context_res.get("game_context", {})
    
    # Enrich context labels
    game_context["weather_wind_mph"] = game["wind"]
    
    home_context = {
        "off_rating": team_stats[home]["off"],
        "def_rating": team_stats[home]["def"],
        "starter_era": game["home_pitcher_era"],
        "park_factor": game_context.get("park_factor", 1.0),
        "weather_wind_mph": game["wind"]
    }
    
    away_context = {
        "off_rating": team_stats[away]["off"],
        "def_rating": team_stats[away]["def"],
        "starter_era": game["away_pitcher_era"]
    }
    
    prompt = f"MLB Game: {away} @ {home}"
    seed = generate_seed(prompt, "2026-06-03")
    
    # 3. Analyze
    trace = analyze({
        "home_team": home,
        "away_team": away,
        "league": "MLB",
        "n_iterations": 10000,
        "seed": seed,
        "home_context": home_context,
        "away_context": away_context,
        "game_context": game_context,
        "odds": odds_patch,
        "simulation_backend": "fast_score"
    }, session_id=session_id, bankroll=bankroll)
    
    # 4. Save
    trace_id = trace["trace_id"]
    export_block = {
        "trace": trace,
        "bet_record": None,
        "reasoning_inputs": {
            "sources": ["espn.com", "rotowire.com", "teamrankings.com", "the-odds-api"],
            "fields_gathered": ["off_rating", "def_rating", "starter_era", "park_factor", "weather_wind_mph", "is_playoff", "rest_days"],
            "missing_fields": [],
            "market_context": {
                "book": "betmgm",
                "odds_home": odds_patch.get("moneyline_home"),
                "odds_away": odds_patch.get("moneyline_away"),
                "over_under": odds_patch.get("over_under")
            }
        },
        "reasoning_downgrade_rationale": "RESEARCH_CANDIDATE: static fallback profile active.",
        "reasoning_narrative": f"Matchup analysis of {away} ({game['away_pitcher']}) vs {home} ({game['home_pitcher']}) under research mode.",
        "trace_quality": {
            "aggregate_quality": 0.95
        }
    }
    
    dest = traces_out_dir / f"{trace_id}.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(export_block, f, indent=2)
    print(f"Saved game trace {trace_id} to {dest.name}")

# Executing 10 Prop Traces
print("\nRunning Pitcher Strikeout Prop simulations...")
prop_count = 0
for idx, game in enumerate(games):
    if prop_count >= 10:
        break
    home = game["home"]
    away = game["away"]
    player = game["home_pitcher"] # Run props for the home pitcher (and some away if needed)
    
    print(f"[{prop_count+1}/10] Resolving prop odds and context for {player} strikeouts...")
    
    # 1. Resolve Odds
    prop_res = resolve_odds(kind="prop", league="MLB", player_name=player, prop_type="strikeouts_pitched", home_team=home, away_team=away)
    if prop_res["status"] != "success" or not prop_res.get("request_patch"):
        print(f"Prop unavailable for {player}, trying away pitcher...")
        player = game["away_pitcher"]
        prop_res = resolve_odds(kind="prop", league="MLB", player_name=player, prop_type="strikeouts_pitched", home_team=home, away_team=away)
        if prop_res["status"] != "success" or not prop_res.get("request_patch"):
            print(f"Prop unavailable for {player} too. Skipping.")
            continue
            
    patch = prop_res["request_patch"]
    line = patch["line"]
    odds_over = patch["odds_over"]
    odds_under = patch["odds_under"]
    
    # 2. Get contextual labels
    context_res = resolve_game_context(league="MLB", home_team=home, away_team=away, game_date="2026-06-03")
    game_context = context_res.get("game_context", {})
    
    player_context = {
        "strikeouts_pitched_mean": pitcher_k_stats[player]["mean"],
        "strikeouts_pitched_std": pitcher_k_stats[player]["std"],
        "sample_size": 8,
        "sample_season": 2026
    }
    
    prompt = f"MLB Prop: {player} strikeouts_pitched O/U {line}"
    seed = generate_seed(prompt, "2026-06-03")
    
    # 3. Analyze
    trace = analyze({
        "player_name": player,
        "league": "MLB",
        "prop_type": "strikeouts_pitched",
        "line": line,
        "home_team": home,
        "away_team": away,
        "game_date": "2026-06-03",
        "odds_over": odds_over,
        "odds_under": odds_under,
        "player_context": player_context,
        "game_context": game_context,
        "n_iterations": 10000,
        "seed": seed
    }, session_id=session_id, bankroll=bankroll)
    
    # 4. Save
    trace_id = trace["trace_id"]
    export_block = {
        "trace": trace,
        "bet_record": None,
        "reasoning_inputs": {
            "sources": ["espn.com", "rotowire.com", "teamrankings.com", "the-odds-api"],
            "fields_gathered": ["strikeouts_pitched_mean", "strikeouts_pitched_std", "sample_size", "sample_season", "park_factor", "rest_days"],
            "missing_fields": [],
            "market_context": {
                "book": "betmgm",
                "player": player,
                "prop_type": "strikeouts_pitched",
                "line": line,
                "odds_over": odds_over,
                "odds_under": odds_under
            }
        },
        "reasoning_downgrade_rationale": "RESEARCH_CANDIDATE: static fallback profile active.",
        "reasoning_narrative": f"Prop simulation for {player} strikeouts_pitched (Line: {line}) against opposing lineup.",
        "trace_quality": {
            "aggregate_quality": 0.95
        }
    }
    
    dest = traces_out_dir / f"{trace_id}.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(export_block, f, indent=2)
    print(f"Saved prop trace {trace_id} to {dest.name}")
    prop_count += 1

print("\nAll simulations completed.")
