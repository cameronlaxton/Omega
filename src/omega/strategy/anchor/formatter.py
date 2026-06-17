"""
Output formatter for anchor parlay scan results.

Produces human-readable ranked tables of parlay recommendations.
"""

from __future__ import annotations

from omega.strategy.anchor.scanner import AnchorParlay, ScanResult


def format_scan_result(result: ScanResult) -> str:
    """Format a ScanResult into a readable report.

    Returns:
        Multi-line string with scan summary and ranked parlays.
    """
    lines: list[str] = []

    lines.append(f"=== Anchor Parlay Scan: {result.league} ({result.date}) ===")
    lines.append(
        f"Games scanned: {result.games_scanned} | "
        f"Players scanned: {result.players_scanned} | "
        f"Anchors found: {result.anchors_found} | "
        f"Parlays built: {result.parlays_built}"
    )
    lines.append("")

    if not result.parlays:
        lines.append("No qualifying parlays found.")
        return "\n".join(lines)

    for i, parlay in enumerate(result.parlays, 1):
        lines.append(format_parlay(parlay, rank=i))
        lines.append("")

    return "\n".join(lines)


def format_parlay(parlay: AnchorParlay, rank: int = 0) -> str:
    """Format a single AnchorParlay into a readable block.

    Args:
        parlay: The parlay to format.
        rank: Display rank (0 = no rank shown).

    Returns:
        Multi-line string for one parlay.
    """
    lines: list[str] = []

    header = f"#{rank} " if rank > 0 else ""
    lines.append(
        f"{header}{parlay.game} | "
        f"{len(parlay.legs)}-leg parlay | "
        f"{parlay.combined_decimal_odds:.2f}x"
    )
    lines.append(
        f"  Combined hit rate: {parlay.combined_hit_rate:.0%} | "
        f"Edge: {parlay.combined_edge_pct:+.1f}% | "
        f"EV: {parlay.ev_pct:+.1f}%"
    )
    lines.append(
        f"  Recommended: {parlay.recommended_units:.1f} units | "
        f"Confidence: {parlay.confidence_tier}"
    )

    lines.append("  Legs:")
    for leg in parlay.legs:
        hit_display = f"{int(leg.hit_rate * leg.games_checked)}/{leg.games_checked}"
        odds_display = f"({leg.odds_over:+.0f})" if leg.odds_over is not None else "(no odds)"
        edge_display = f"edge {leg.edge_pct:+.1f}%" if leg.odds_over is not None else ""
        lines.append(
            f"    - {leg.player_name} {leg.threshold:.0f}+ {leg.stat_key} "
            f"[{hit_display} = {leg.hit_rate:.0%}] {odds_display} {edge_display}"
        )

    if parlay.correlation_warnings:
        lines.append("  Warnings:")
        for w in parlay.correlation_warnings:
            lines.append(f"    ! {w}")

    return "\n".join(lines)


def format_parlay_brief(parlay: AnchorParlay) -> str:
    """One-line summary of a parlay for quick display."""
    legs_str = " + ".join(
        f"{leg.player_name} {leg.threshold:.0f}+ {leg.stat_key}" for leg in parlay.legs
    )
    return (
        f"{parlay.game}: {legs_str} | "
        f"{parlay.combined_decimal_odds:.2f}x | "
        f"Hit: {parlay.combined_hit_rate:.0%} | "
        f"EV: {parlay.ev_pct:+.1f}%"
    )
