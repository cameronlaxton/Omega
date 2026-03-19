"""
Response composer — builds structured responses from execution results.

Assembles output packages (bet_card, game_breakdown, key_factors, etc.)
and generates human-readable text summaries from simulation and edge data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from omega.core.models import (
    AnswerPlan,
    ExecutionResult,
    GatheredFact,
    OutputPackage,
    QueryUnderstanding,
)


def compose_response(
    understanding: QueryUnderstanding,
    plan: AnswerPlan,
    facts: List[GatheredFact],
    execution: ExecutionResult,
) -> Dict[str, Any]:
    """Build structured response from execution results."""
    sections: Dict[str, Any] = {}
    narrative_parts: List[str] = []

    for pkg in plan.output_packages:
        if pkg == OutputPackage.BET_CARD and execution.best_bet:
            sections["bet_card"] = {
                "edges": execution.edges,
                "best_bet": execution.best_bet,
                "data_completeness": execution.data_completeness,
            }

        elif pkg == OutputPackage.GAME_BREAKDOWN and execution.simulation:
            sections["game_breakdown"] = {
                "simulation": execution.simulation,
                "key_factors": _extract_key_factors(facts),
            }

        elif pkg == OutputPackage.KEY_FACTORS:
            sections["key_factors"] = _extract_key_factors(facts)

        elif pkg == OutputPackage.RESEARCH_REPORT:
            sections["research"] = {
                "facts": [
                    {
                        "key": f.slot.key,
                        "data": f.result.data if f.result else None,
                        "source": f.result.source if f.result else None,
                        "quality": f.quality_score,
                    }
                    for f in facts if f.filled
                ],
            }

        elif pkg == OutputPackage.PLAIN_EXPLANATION:
            narrative_parts.append(
                f"Analysis of {understanding.raw_prompt}"
            )

        elif pkg == OutputPackage.COMPACT_SUMMARY:
            filled = [f for f in facts if f.filled]
            sections["compact_summary"] = {
                "filled_facts": len(filled),
                "total_facts": len(facts),
                "data_quality": execution.data_quality_score,
                "mode": execution.mode.value,
            }
            if execution.simulation:
                narrative_parts.append(
                    _build_text_summary(understanding, execution, []),
                )
            elif filled:
                narrative_parts.append(
                    f"Quick look at {understanding.raw_prompt}: "
                    f"{len(filled)} data points gathered."
                )

        elif pkg == OutputPackage.LIMITED_CONTEXT_ANSWER:
            filled = [f for f in facts if f.filled]
            sections["limited_context"] = {
                "filled_facts": len(filled),
                "total_facts": len(facts),
                "data_quality": execution.data_quality_score,
                "caveat": "Limited data available — treat conclusions with caution.",
            }
            narrative_parts.append(
                f"I found limited data for this query. "
                f"({len(filled)}/{len(facts)} data points available)"
            )

        elif pkg == OutputPackage.BANKROLL_GUIDANCE:
            sections["bankroll_guidance"] = {
                "mode": execution.mode.value,
                "edges": execution.edges,
                "best_bet": execution.best_bet,
                "data_quality": execution.data_quality_score,
            }
            if execution.best_bet:
                kelly = execution.best_bet.get("kelly_fraction", 0)
                narrative_parts.append(
                    f"Bankroll guidance: suggested Kelly fraction {kelly:.1%}"
                )

        elif pkg == OutputPackage.NEWS_DIGEST:
            news_facts = [
                {
                    "key": f.slot.key,
                    "data": f.result.data if f.result else None,
                    "source": f.result.source if f.result else None,
                }
                for f in facts if f.filled
            ]
            sections["news_digest"] = {"items": news_facts}

        elif pkg == OutputPackage.SCENARIO_ANALYSIS:
            sections["scenario_analysis"] = {
                "simulation": execution.simulation,
                "edges": execution.edges,
                "data_quality": execution.data_quality_score,
            }

        elif pkg == OutputPackage.ALTERNATIVE_BETS:
            alt_edges = execution.edges[1:] if len(execution.edges) > 1 else []
            sections["alternative_bets"] = {"edges": alt_edges}

    # Build text summary
    text = _build_text_summary(understanding, execution, narrative_parts)

    return {
        "type": "answer",
        "text": text,
        "narrative": text,
        "sections": sections,
        "simulation": execution.simulation,
        "edges": execution.edges,
        "best_bet": execution.best_bet,
        "metadata": {
            "execution_mode": execution.mode.value,
            "data_quality": execution.data_quality_score,
            "sources_used": len([f for f in facts if f.filled]),
            "total_slots": len(facts),
            "league": understanding.league,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def _build_text_summary(
    understanding: QueryUnderstanding,
    execution: ExecutionResult,
    narrative_parts: List[str],
) -> str:
    """Build a human-readable text summary."""
    parts: List[str] = []

    if execution.simulation:
        sim = execution.simulation
        home = sim.get("home_team", "Home")
        away = sim.get("away_team", "Away")
        parts.append(
            f"**{away} @ {home}** — "
            f"Model: {home} {sim.get('home_win_prob', '?')}% / "
            f"{away} {sim.get('away_win_prob', '?')}%"
        )
        if sim.get("predicted_spread"):
            parts.append(
                f"Projected spread: {sim['predicted_spread']:+.1f} | "
                f"Total: {sim.get('predicted_total', '?')}"
            )

    if execution.edges:
        best = execution.edges[0] if execution.edges else None
        if best:
            parts.append(
                f"Best edge: {best.get('side', '?')} "
                f"({best.get('edge_pct', 0):+.1f}% edge, "
                f"{best.get('confidence_tier', '?')} confidence)"
            )

    if not parts:
        if narrative_parts:
            parts.extend(narrative_parts)
        else:
            parts.append(
                f"Analysis complete for: {understanding.raw_prompt}"
            )

    return "\n\n".join(parts)


def _extract_key_factors(facts: List[GatheredFact]) -> List[Dict[str, Any]]:
    """Extract key factors from gathered facts."""
    factors = []
    for fact in facts:
        if fact.filled and fact.result:
            factors.append({
                "key": fact.slot.key,
                "importance": fact.slot.importance.value,
                "data": fact.result.data,
                "source": fact.result.source,
                "quality": fact.quality_score,
            })
    return factors
