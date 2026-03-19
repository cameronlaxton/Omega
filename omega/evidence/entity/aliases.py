"""
Static alias database for team/player name canonicalization.

Each entry maps a canonical name to its known aliases, abbreviation,
and league.  The resolver uses this database plus fuzzy matching as
a fallback for user typos.

Canonical names are chosen to match the most common API representation
(usually the full city + mascot form used by ESPN and The Odds API).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class TeamRecord:
    canonical: str
    abbreviation: str
    aliases: tuple[str, ...]  # lowercased


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------

NBA_TEAMS: List[TeamRecord] = [
    TeamRecord("Atlanta Hawks", "ATL", ("hawks", "atl", "atlanta")),
    TeamRecord("Boston Celtics", "BOS", ("celtics", "bos", "boston", "celts")),
    TeamRecord("Brooklyn Nets", "BKN", ("nets", "bkn", "brooklyn")),
    TeamRecord("Charlotte Hornets", "CHA", ("hornets", "cha", "charlotte")),
    TeamRecord("Chicago Bulls", "CHI", ("bulls", "chi", "chicago")),
    TeamRecord("Cleveland Cavaliers", "CLE", ("cavaliers", "cavs", "cle", "cleveland")),
    TeamRecord("Dallas Mavericks", "DAL", ("mavericks", "mavs", "dal", "dallas")),
    TeamRecord("Denver Nuggets", "DEN", ("nuggets", "den", "denver")),
    TeamRecord("Detroit Pistons", "DET", ("pistons", "det", "detroit")),
    TeamRecord("Golden State Warriors", "GSW", ("warriors", "gsw", "golden state", "gs", "dubs")),
    TeamRecord("Houston Rockets", "HOU", ("rockets", "hou", "houston")),
    TeamRecord("Indiana Pacers", "IND", ("pacers", "ind", "indiana")),
    TeamRecord("Los Angeles Clippers", "LAC", ("clippers", "lac", "la clippers")),
    TeamRecord("Los Angeles Lakers", "LAL", ("lakers", "lal", "la lakers", "los angeles lakers")),
    TeamRecord("Memphis Grizzlies", "MEM", ("grizzlies", "grizz", "mem", "memphis")),
    TeamRecord("Miami Heat", "MIA", ("heat", "mia", "miami")),
    TeamRecord("Milwaukee Bucks", "MIL", ("bucks", "mil", "milwaukee")),
    TeamRecord("Minnesota Timberwolves", "MIN", ("timberwolves", "twolves", "wolves", "min", "minnesota")),
    TeamRecord("New Orleans Pelicans", "NOP", ("pelicans", "pels", "nop", "new orleans")),
    TeamRecord("New York Knicks", "NYK", ("knicks", "nyk", "ny knicks", "new york knicks", "new york")),
    TeamRecord("Oklahoma City Thunder", "OKC", ("thunder", "okc", "oklahoma city")),
    TeamRecord("Orlando Magic", "ORL", ("magic", "orl", "orlando")),
    TeamRecord("Philadelphia 76ers", "PHI", ("76ers", "sixers", "phi", "philly", "philadelphia")),
    TeamRecord("Phoenix Suns", "PHX", ("suns", "phx", "phoenix")),
    TeamRecord("Portland Trail Blazers", "POR", ("trail blazers", "blazers", "por", "portland")),
    TeamRecord("Sacramento Kings", "SAC", ("kings", "sac", "sacramento")),
    TeamRecord("San Antonio Spurs", "SAS", ("spurs", "sas", "san antonio")),
    TeamRecord("Toronto Raptors", "TOR", ("raptors", "raps", "tor", "toronto")),
    TeamRecord("Utah Jazz", "UTA", ("jazz", "uta", "utah")),
    TeamRecord("Washington Wizards", "WAS", ("wizards", "wiz", "was", "washington")),
]

# ---------------------------------------------------------------------------
# NFL
# ---------------------------------------------------------------------------

NFL_TEAMS: List[TeamRecord] = [
    TeamRecord("Arizona Cardinals", "ARI", ("cardinals", "ari", "arizona")),
    TeamRecord("Atlanta Falcons", "ATL", ("falcons", "atl falcons", "atlanta falcons")),
    TeamRecord("Baltimore Ravens", "BAL", ("ravens", "bal", "baltimore")),
    TeamRecord("Buffalo Bills", "BUF", ("bills", "buf", "buffalo")),
    TeamRecord("Carolina Panthers", "CAR", ("panthers", "car", "carolina")),
    TeamRecord("Chicago Bears", "CHI", ("bears", "chi bears", "chicago bears")),
    TeamRecord("Cincinnati Bengals", "CIN", ("bengals", "cin", "cincinnati")),
    TeamRecord("Cleveland Browns", "CLE", ("browns", "cle browns", "cleveland browns")),
    TeamRecord("Dallas Cowboys", "DAL", ("cowboys", "dal cowboys", "dallas cowboys")),
    TeamRecord("Denver Broncos", "DEN", ("broncos", "den broncos", "denver broncos")),
    TeamRecord("Detroit Lions", "DET", ("lions", "det lions", "detroit lions")),
    TeamRecord("Green Bay Packers", "GB", ("packers", "gb", "green bay")),
    TeamRecord("Houston Texans", "HOU", ("texans", "hou texans", "houston texans")),
    TeamRecord("Indianapolis Colts", "IND", ("colts", "ind colts", "indianapolis")),
    TeamRecord("Jacksonville Jaguars", "JAX", ("jaguars", "jags", "jax", "jacksonville")),
    TeamRecord("Kansas City Chiefs", "KC", ("chiefs", "kc", "kansas city")),
    TeamRecord("Las Vegas Raiders", "LV", ("raiders", "lv", "las vegas")),
    TeamRecord("Los Angeles Chargers", "LAC", ("chargers", "lac chargers", "la chargers")),
    TeamRecord("Los Angeles Rams", "LAR", ("rams", "lar", "la rams")),
    TeamRecord("Miami Dolphins", "MIA", ("dolphins", "mia dolphins", "miami dolphins")),
    TeamRecord("Minnesota Vikings", "MIN", ("vikings", "min vikings", "minnesota vikings")),
    TeamRecord("New England Patriots", "NE", ("patriots", "pats", "ne", "new england")),
    TeamRecord("New Orleans Saints", "NO", ("saints", "no saints", "new orleans saints")),
    TeamRecord("New York Giants", "NYG", ("giants", "nyg", "ny giants")),
    TeamRecord("New York Jets", "NYJ", ("jets", "nyj", "ny jets")),
    TeamRecord("Philadelphia Eagles", "PHI", ("eagles", "phi eagles", "philadelphia eagles")),
    TeamRecord("Pittsburgh Steelers", "PIT", ("steelers", "pit", "pittsburgh")),
    TeamRecord("San Francisco 49ers", "SF", ("49ers", "niners", "sf", "san francisco")),
    TeamRecord("Seattle Seahawks", "SEA", ("seahawks", "sea", "seattle")),
    TeamRecord("Tampa Bay Buccaneers", "TB", ("buccaneers", "bucs", "tb", "tampa bay", "tampa")),
    TeamRecord("Tennessee Titans", "TEN", ("titans", "ten", "tennessee")),
    TeamRecord("Washington Commanders", "WAS", ("commanders", "was commanders", "washington commanders")),
]

# ---------------------------------------------------------------------------
# MLB
# ---------------------------------------------------------------------------

MLB_TEAMS: List[TeamRecord] = [
    TeamRecord("Arizona Diamondbacks", "ARI", ("diamondbacks", "dbacks", "ari dbacks", "arizona")),
    TeamRecord("Atlanta Braves", "ATL", ("braves", "atl braves", "atlanta braves")),
    TeamRecord("Baltimore Orioles", "BAL", ("orioles", "os", "bal", "baltimore orioles")),
    TeamRecord("Boston Red Sox", "BOS", ("red sox", "sox", "bos", "boston red sox")),
    TeamRecord("Chicago Cubs", "CHC", ("cubs", "chc", "chicago cubs")),
    TeamRecord("Chicago White Sox", "CHW", ("white sox", "chw", "chicago white sox")),
    TeamRecord("Cincinnati Reds", "CIN", ("reds", "cin", "cincinnati reds")),
    TeamRecord("Cleveland Guardians", "CLE", ("guardians", "cle", "cleveland guardians")),
    TeamRecord("Colorado Rockies", "COL", ("rockies", "col", "colorado")),
    TeamRecord("Detroit Tigers", "DET", ("tigers", "det tigers", "detroit tigers")),
    TeamRecord("Houston Astros", "HOU", ("astros", "hou astros", "houston astros")),
    TeamRecord("Kansas City Royals", "KC", ("royals", "kc royals", "kansas city royals")),
    TeamRecord("Los Angeles Angels", "LAA", ("angels", "laa", "la angels", "anaheim")),
    TeamRecord("Los Angeles Dodgers", "LAD", ("dodgers", "lad", "la dodgers")),
    TeamRecord("Miami Marlins", "MIA", ("marlins", "mia marlins", "miami marlins")),
    TeamRecord("Milwaukee Brewers", "MIL", ("brewers", "mil brewers", "milwaukee brewers")),
    TeamRecord("Minnesota Twins", "MIN", ("twins", "min twins", "minnesota twins")),
    TeamRecord("New York Mets", "NYM", ("mets", "nym", "ny mets")),
    TeamRecord("New York Yankees", "NYY", ("yankees", "yanks", "nyy", "ny yankees")),
    TeamRecord("Oakland Athletics", "OAK", ("athletics", "as", "oak", "oakland")),
    TeamRecord("Philadelphia Phillies", "PHI", ("phillies", "phi phillies", "philadelphia phillies")),
    TeamRecord("Pittsburgh Pirates", "PIT", ("pirates", "pit pirates", "pittsburgh pirates")),
    TeamRecord("San Diego Padres", "SD", ("padres", "sd", "san diego")),
    TeamRecord("San Francisco Giants", "SF", ("sf giants", "san francisco giants")),
    TeamRecord("Seattle Mariners", "SEA", ("mariners", "ms", "sea mariners", "seattle mariners")),
    TeamRecord("St. Louis Cardinals", "STL", ("cardinals", "stl", "st louis", "st. louis")),
    TeamRecord("Tampa Bay Rays", "TB", ("rays", "tb rays", "tampa bay rays")),
    TeamRecord("Texas Rangers", "TEX", ("rangers", "tex", "texas")),
    TeamRecord("Toronto Blue Jays", "TOR", ("blue jays", "jays", "tor", "toronto blue jays")),
    TeamRecord("Washington Nationals", "WSH", ("nationals", "nats", "wsh", "washington nationals")),
]

# ---------------------------------------------------------------------------
# NHL
# ---------------------------------------------------------------------------

NHL_TEAMS: List[TeamRecord] = [
    TeamRecord("Anaheim Ducks", "ANA", ("ducks", "ana", "anaheim ducks")),
    TeamRecord("Arizona Coyotes", "ARI", ("coyotes", "yotes", "ari coyotes")),
    TeamRecord("Boston Bruins", "BOS", ("bruins", "bos bruins", "boston bruins")),
    TeamRecord("Buffalo Sabres", "BUF", ("sabres", "buf", "buffalo sabres")),
    TeamRecord("Calgary Flames", "CGY", ("flames", "cgy", "calgary")),
    TeamRecord("Carolina Hurricanes", "CAR", ("hurricanes", "canes", "car", "carolina")),
    TeamRecord("Chicago Blackhawks", "CHI", ("blackhawks", "hawks", "chi blackhawks")),
    TeamRecord("Colorado Avalanche", "COL", ("avalanche", "avs", "col", "colorado avalanche")),
    TeamRecord("Columbus Blue Jackets", "CBJ", ("blue jackets", "cbj", "columbus")),
    TeamRecord("Dallas Stars", "DAL", ("stars", "dal stars", "dallas stars")),
    TeamRecord("Detroit Red Wings", "DET", ("red wings", "det red wings", "detroit red wings")),
    TeamRecord("Edmonton Oilers", "EDM", ("oilers", "edm", "edmonton")),
    TeamRecord("Florida Panthers", "FLA", ("florida panthers", "fla", "florida")),
    TeamRecord("Los Angeles Kings", "LAK", ("kings", "lak", "la kings")),
    TeamRecord("Minnesota Wild", "MIN", ("wild", "min wild", "minnesota wild")),
    TeamRecord("Montreal Canadiens", "MTL", ("canadiens", "habs", "mtl", "montreal")),
    TeamRecord("Nashville Predators", "NSH", ("predators", "preds", "nsh", "nashville")),
    TeamRecord("New Jersey Devils", "NJD", ("devils", "njd", "new jersey")),
    TeamRecord("New York Islanders", "NYI", ("islanders", "isles", "nyi")),
    TeamRecord("New York Rangers", "NYR", ("rangers", "nyr", "ny rangers")),
    TeamRecord("Ottawa Senators", "OTT", ("senators", "sens", "ott", "ottawa")),
    TeamRecord("Philadelphia Flyers", "PHI", ("flyers", "phi flyers", "philadelphia flyers")),
    TeamRecord("Pittsburgh Penguins", "PIT", ("penguins", "pens", "pit penguins")),
    TeamRecord("San Jose Sharks", "SJS", ("sharks", "sjs", "san jose")),
    TeamRecord("Seattle Kraken", "SEA", ("kraken", "sea kraken", "seattle kraken")),
    TeamRecord("St. Louis Blues", "STL", ("blues", "stl blues", "st louis blues")),
    TeamRecord("Tampa Bay Lightning", "TBL", ("lightning", "bolts", "tbl", "tampa bay lightning")),
    TeamRecord("Toronto Maple Leafs", "TOR", ("maple leafs", "leafs", "tor leafs", "toronto maple leafs")),
    TeamRecord("Vancouver Canucks", "VAN", ("canucks", "nucks", "van", "vancouver")),
    TeamRecord("Vegas Golden Knights", "VGK", ("golden knights", "knights", "vgk", "vegas")),
    TeamRecord("Washington Capitals", "WSH", ("capitals", "caps", "wsh capitals")),
    TeamRecord("Winnipeg Jets", "WPG", ("jets", "wpg", "winnipeg")),
]

# ---------------------------------------------------------------------------
# Aggregate: league → team list
# ---------------------------------------------------------------------------

LEAGUE_TEAMS: Dict[str, List[TeamRecord]] = {
    "NBA": NBA_TEAMS,
    "NFL": NFL_TEAMS,
    "MLB": MLB_TEAMS,
    "NHL": NHL_TEAMS,
}
