#!/usr/bin/env python
"""build_slate_entries.py - turn a researched slate spec into validated batch entries.

Replaces the dated scratch runners (write_entries.py, run_mlb_batch.py,
run_slate_YYYYMMDD.py, ...) that re-created the same entry-building Python every
day with hardcoded matchups. The per-day researched facts now live in a JSON
*slate spec* (see ``--init`` for a template); this tool merges them with
extracted context packs (tools/extract_contexts.py) and emits a list of
``BatchAnalysisEntry`` dicts ready for tools/run_slate.py.

Design rules carried over from the honesty doctrine:

- **No fabricated data.** A game or prop whose team/player context cannot be
  found (inline or in a pack) is an error — the tool lists exactly what is
  missing so the operator can research it. ``--skip-missing`` drops such
  entries instead (reported), it never invents them.
- **No silent neutral game_context.** The scratch runners swallowed resolver
  failures into ``{"is_playoff": False, "rest_days": 1}``. Here that fallback
  exists only behind ``--allow-default-game-context`` and is reported per entry.
- **No hand-rolled seeds.** ``BatchAnalysisEntry.seed`` auto-derives from a
  content hash when absent; duplicating that here would be a second source of
  truth. A slate may still pin an explicit ``seed`` per item.
- **Contract-validated output.** Every entry is validated against
  ``BatchAnalysisEntry`` before writing, so a malformed slate fails at build
  time, not mid-batch. Unknown keys in slate items are rejected (typo guard).

Pipeline:

    tools/extract_contexts.py --league MLB
        -> var/context_packs/mlb_{team,player}_contexts.json
    (operator writes/edits the slate spec with fresh research)
    tools/build_slate_entries.py --slate slate.json --team-contexts ... --player-contexts ...
        -> var/slates/<league>-<date>.entries.json
    tools/run_slate.py --entries ... --session-id ...
    tools/query_session_results.py --session-id ...

Usage
-----
    python tools/build_slate_entries.py --init > slate.json     # template to fill in
    python tools/build_slate_entries.py --slate slate.json \
        --team-contexts var/context_packs/mlb_team_contexts.json \
        --player-contexts var/context_packs/mlb_player_contexts.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from omega.core.contracts.schemas import BatchAnalysisEntry
except ImportError:  # running outside an installed env — fall back to src layout
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from omega.core.contracts.schemas import BatchAnalysisEntry

NEUTRAL_GAME_CONTEXT = {"is_playoff": False, "rest_days": 1}

# Slate-item keys copied through to the entry verbatim when present.
COMMON_KEYS = (
    "game_context",
    "n_iterations",
    "seed",
    "evidence",
    "reasoning_narrative",
    "reasoning_presentation",
    "reasoning_sources",
    "roster_context",
)
GAME_KEYS = COMMON_KEYS + ("home_context", "away_context", "odds")
PROP_KEYS = COMMON_KEYS + (
    "player_name",
    "prop_type",
    "player_context",
    "line",
    "odds_over",
    "odds_under",
)
# Keys that identify the item but are not copied blindly.
STRUCTURAL_KEYS = ("home_team", "away_team")

SLATE_TEMPLATE: dict[str, Any] = {
    "league": "MLB",
    "game_date": "2026-07-04",
    "defaults": {"n_iterations": 10000, "reasoning_sources": ["mlb.com"]},
    "games": [
        {
            "home_team": "Seattle Mariners",
            "away_team": "Toronto Blue Jays",
            "home_context": {"off_rating": 4.2, "def_rating": 3.9, "starter_era": 3.42},
            "away_context": {"off_rating": 4.4, "def_rating": 4.1, "starter_era": 6.0},
            "evidence": [],
            "roster_context": {
                "home_team": "Seattle Mariners",
                "away_team": "Toronto Blue Jays",
                "league": "MLB",
                "game_date": "2026-07-04",
                "source_summaries": [{"source": "mlb.com", "summary": "..."}],
                "home_status": {"lineup_status": "confirmed", "injury_report_checked": True},
                "away_status": {"lineup_status": "confirmed", "injury_report_checked": True},
                "absences": [],
                "roster_context_complete": True,
                "gathered_at": "2026-07-04T18:00:00Z",
            },
            "reasoning_narrative": "2-4 sentence summary of the researched read.",
            "reasoning_presentation": {
                "thesis": "...",
                "market_read": "...",
                "why": "...",
                "risks": "...",
                "verdict": "...",
            },
        }
    ],
    "props": [
        {
            "player_name": "Logan Gilbert",
            "prop_type": "strikeouts_pitched",
            "home_team": "Seattle Mariners",
            "away_team": "Toronto Blue Jays",
            "player_context": {
                "strikeouts_pitched_mean": 6.1,
                "strikeouts_pitched_std": 2.3,
                "sample_size": 15,
                "sample_season": 2026,
            },
            "evidence": [],
        }
    ],
}


def _fold_name(name: str) -> str:
    """Accent-insensitive, case-insensitive key for player matching."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _load_pack(path: str | None, label: str) -> dict[str, Any]:
    """Load a context pack — extract_contexts.py envelope or a plain mapping."""
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} pack {path} is not a JSON object")
    contexts = payload.get("contexts", payload)
    if payload.get("stale"):
        print(
            f"WARNING: {label} pack {path} flags {len(payload['stale'])} stale entries "
            "- re-verify or re-extract before trusting them."
        )
    return contexts


def _check_unknown_keys(item: dict[str, Any], allowed: tuple[str, ...], where: str) -> list[str]:
    unknown = sorted(set(item) - set(allowed) - set(STRUCTURAL_KEYS))
    return [f"{where}: unknown key(s) {unknown} (typo?)"] if unknown else []


def _check_required(item: dict[str, Any], fields: tuple[str, ...], where: str) -> list[str]:
    missing = [f for f in fields if not item.get(f)]
    return [f"{where}: missing required field(s) {missing}"] if missing else []


def _matchup(item: dict[str, Any]) -> str:
    return f"{item.get('away_team', '?')} @ {item.get('home_team', '?')}"


class SlateBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        slate = json.loads(Path(args.slate).read_text(encoding="utf-8"))
        for field in ("league", "game_date"):
            if not slate.get(field):
                raise ValueError(f"slate is missing required field {field!r}")
        if not slate.get("games") and not slate.get("props"):
            raise ValueError("slate has neither 'games' nor 'props'")
        self.slate = slate
        self.league: str = slate["league"]
        self.game_date: str = slate["game_date"]
        self.defaults: dict[str, Any] = slate.get("defaults") or {}
        self.team_pack = _load_pack(args.team_contexts, "team")
        player_pack = _load_pack(args.player_contexts, "player")
        self.player_pack_folded = {_fold_name(k): v for k, v in player_pack.items()}
        # Final game_context / roster_context per matchup, for prop inheritance.
        self.game_ctx_by_matchup: dict[str, dict[str, Any]] = {}
        self.roster_ctx_by_matchup: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.skipped: list[str] = []
        self.notes: list[str] = []

    # -- context resolution -------------------------------------------------

    def _team_context(self, item: dict[str, Any], side: str, where: str) -> dict[str, Any] | None:
        inline = item.get(f"{side}_context")
        if inline:
            return inline
        team = item.get(f"{side}_team")
        from_pack = self.team_pack.get(team)
        if from_pack:
            self.notes.append(f"{where}: {side}_context for {team} from team pack")
            return from_pack
        self.errors.append(f"{where}: no {side}_context inline or in team pack for {team!r}")
        return None

    def _game_context(self, item: dict[str, Any], where: str) -> dict[str, Any] | None:
        inline = item.get("game_context")
        if inline:
            return inline
        inherited = self.game_ctx_by_matchup.get(_matchup(item))
        if inherited:
            return inherited
        if self.args.resolve_game_context:
            try:
                from omega.integrations.game_context import resolve_game_context

                resolved = resolve_game_context(
                    self.league, item["home_team"], item["away_team"], self.game_date
                )
                needs_manual = resolved.get("needs_manual") or []
                if needs_manual:
                    self.notes.append(f"{where}: resolver needs_manual={needs_manual}")
                ctx = resolved.get("game_context") or {}
                if ctx:
                    self.notes.append(f"{where}: game_context via resolver")
                    return ctx
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"{where}: resolve_game_context failed: {exc}")
        if self.args.allow_default_game_context:
            self.notes.append(f"{where}: NEUTRAL DEFAULT game_context (operator-approved)")
            return dict(NEUTRAL_GAME_CONTEXT)
        self.errors.append(
            f"{where}: no game_context (supply inline, or use --resolve-game-context / "
            "--allow-default-game-context)"
        )
        return None

    def _player_context(self, item: dict[str, Any], where: str) -> dict[str, Any] | None:
        inline = item.get("player_context")
        if inline:
            return inline
        player, prop_type = item.get("player_name"), item.get("prop_type")
        by_prop = self.player_pack_folded.get(_fold_name(player or ""))
        # prop_type may be a fallback chain (list) — first market with a context wins.
        for pt in prop_type if isinstance(prop_type, list) else [prop_type]:
            if by_prop and pt in by_prop:
                self.notes.append(f"{where}: player_context from player pack ({pt})")
                return by_prop[pt]
        self.errors.append(
            f"{where}: no player_context inline or in player pack for "
            f"{player!r} / {prop_type!r} - research it; this tool does not impute"
        )
        return None

    # -- entry assembly -----------------------------------------------------

    def _base_entry(self, kind: str, item: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "kind": kind,
            "league": self.league,
            "home_team": item["home_team"],
            "away_team": item["away_team"],
            "game_date": self.game_date,
        }
        for key in keys:
            if key in item:
                entry[key] = item[key]
        for key in ("n_iterations", "reasoning_sources"):
            if key in self.defaults:
                entry.setdefault(key, self.defaults[key])
        return entry

    def build(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        for i, item in enumerate(self.slate.get("games") or []):
            where = f"games[{i}] {_matchup(item)}"
            errs_before = len(self.errors)
            self.errors.extend(_check_unknown_keys(item, GAME_KEYS, where))
            self.errors.extend(_check_required(item, ("home_team", "away_team"), where))
            if len(self.errors) == errs_before:
                entry = self._base_entry("game", item, GAME_KEYS)
                home_ctx = self._team_context(item, "home", where)
                away_ctx = self._team_context(item, "away", where)
                game_ctx = self._game_context(item, where)
                if len(self.errors) == errs_before:
                    entry.update(
                        home_context=home_ctx, away_context=away_ctx, game_context=game_ctx
                    )
                    self.game_ctx_by_matchup[_matchup(item)] = game_ctx
                    if entry.get("roster_context"):
                        self.roster_ctx_by_matchup[_matchup(item)] = entry["roster_context"]
                    entries.append(entry)
                    continue
            if self.args.skip_missing:
                self.skipped.extend(self.errors[errs_before:])
                del self.errors[errs_before:]

        for i, item in enumerate(self.slate.get("props") or []):
            where = f"props[{i}] {item.get('player_name', '?')} {item.get('prop_type', '?')}"
            errs_before = len(self.errors)
            self.errors.extend(_check_unknown_keys(item, PROP_KEYS, where))
            self.errors.extend(
                _check_required(item, ("player_name", "prop_type", "home_team", "away_team"), where)
            )
            if len(self.errors) == errs_before:
                entry = self._base_entry("prop", item, PROP_KEYS)
                player_ctx = self._player_context(item, where)
                game_ctx = self._game_context(item, where)
                if "roster_context" not in entry:
                    inherited = self.roster_ctx_by_matchup.get(_matchup(item))
                    if inherited:
                        entry["roster_context"] = inherited
                        self.notes.append(f"{where}: roster_context inherited from game entry")
                if len(self.errors) == errs_before:
                    entry.update(player_context=player_ctx, game_context=game_ctx)
                    entries.append(entry)
                    continue
            if self.args.skip_missing:
                self.skipped.extend(self.errors[errs_before:])
                del self.errors[errs_before:]

        self._check_matchup_specific_summaries(entries)

        # Contract gate: a slate that produces an invalid entry fails the build.
        for idx, entry in enumerate(entries):
            try:
                BatchAnalysisEntry.model_validate(entry)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"entry[{idx}] failed BatchAnalysisEntry validation: {exc}")
        return entries

    def _check_matchup_specific_summaries(self, entries: list[dict[str, Any]]) -> None:
        """Fail the build when source_summaries are reused across matchups.

        RSVG requires roster/situational summaries independently verified per
        matchup; omega_run_batch downgrades any entry whose exact summary text
        was first seen for a DIFFERENT matchup (research_candidate — formal
        output suppressed). Catch that boilerplate here, at build time, so the
        operator researches each matchup instead of shipping a slate that can
        never earn formal output. Same-matchup reuse (a prop inheriting its
        game's roster_context) is legitimate and not flagged.
        """
        first_matchup_for_sig: dict[tuple[tuple[str, str], ...], str] = {}
        flagged: set[tuple[str, str]] = set()
        for entry in entries:
            rc = entry.get("roster_context")
            if not isinstance(rc, dict):
                continue
            summaries = rc.get("source_summaries") or []
            signature = tuple(
                sorted(
                    (str(s.get("source", "")), str(s.get("summary", "")))
                    for s in summaries
                    if isinstance(s, dict)
                )
            )
            if not signature:
                continue
            matchup = _matchup(entry)
            first = first_matchup_for_sig.setdefault(signature, matchup)
            if first != matchup and (first, matchup) not in flagged:
                flagged.add((first, matchup))
                self.errors.append(
                    f"{matchup}: roster_context.source_summaries reused verbatim from "
                    f"{first} — research each matchup separately (RSVG downgrades "
                    "cross-matchup boilerplate to research_candidate)"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build validated omega_run_batch entries from a slate spec."
    )
    parser.add_argument("--slate", help="Slate spec JSON (see --init)")
    parser.add_argument("--init", action="store_true", help="Print a slate template and exit")
    parser.add_argument("--team-contexts", help="Team context pack (tools/extract_contexts.py)")
    parser.add_argument("--player-contexts", help="Player context pack")
    parser.add_argument(
        "--output", help="Entries path (default: var/slates/<league>-<date>.entries.json)"
    )
    parser.add_argument(
        "--resolve-game-context",
        action="store_true",
        help="Resolve missing game_context via omega.integrations.game_context (network)",
    )
    parser.add_argument(
        "--allow-default-game-context",
        action="store_true",
        help=f"Permit the neutral fallback {NEUTRAL_GAME_CONTEXT} when unresolved (reported per entry)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Drop entries with unresolvable contexts instead of failing the build",
    )
    args = parser.parse_args(argv)

    if args.init:
        print(json.dumps(SLATE_TEMPLATE, indent=2))
        return 0
    if not args.slate:
        parser.error("--slate is required (or --init for a template)")

    try:
        builder = SlateBuilder(args)
        entries = builder.build()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    for note in builder.notes:
        print(f"note: {note}")
    for skip in builder.skipped:
        print(f"SKIPPED: {skip}")
    if builder.errors:
        print(f"\nBUILD FAILED — {len(builder.errors)} problem(s):")
        for err in builder.errors:
            print(f"  - {err}")
        return 1
    if not entries:
        print("BUILD FAILED - slate produced zero entries.")
        return 1

    out_path = Path(
        args.output
        or REPO_ROOT
        / "var"
        / "slates"
        / f"{builder.league.lower()}-{builder.game_date}.entries.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    n_games = sum(1 for e in entries if e["kind"] == "game")
    n_props = len(entries) - n_games
    print(f"\nBuilt {len(entries)} entries ({n_games} games, {n_props} props) -> {out_path}")
    if builder.skipped:
        print(f"({len(builder.skipped)} entr(y/ies) skipped — see SKIPPED lines above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
