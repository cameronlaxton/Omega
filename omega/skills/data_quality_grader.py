"""
omega.skills.data_quality_grader — audit quality gate decisions post-hoc.

Observes stage 5 (after apply_quality_gate). Checks three concern groups:

sanity:
  - aggregate quality score is outside [0.0, 1.0]

under_downgrade (gate missed something it should have caught):
  - bet_card present in revised plan when critical inputs were not filled
  - very low quality (< 0.3) with zero downgrades applied

over_downgrade (gate fired when it should not have):
  - bet_card absent from revised plan when critical inputs were filled
    AND quality exceeds the bet_card threshold (default 0.7)

These checks detect misconfiguration or bugs in the quality gate that would
silently let bad-evidence queries produce betting recommendations, or
unnecessarily suppress good-evidence queries.
"""

from __future__ import annotations

from typing import Any

from omega.core.models import AnswerPlan, GatheredFact, InputImportance, OutputPackage
from omega.skills import register
from omega.skills.base import SkillBase, SkillObservation

_DEFAULT_BET_CARD_THRESHOLD = 0.7
_LOW_QUALITY_THRESHOLD = 0.3


def _critical_inputs_filled(facts: list[GatheredFact]) -> bool:
    critical = [f for f in facts if f.slot.importance == InputImportance.CRITICAL]
    if not critical:
        return False
    return all(f.filled for f in critical)


@register("data-quality-grader")
class DataQualityGrader(SkillBase):
    name = "data-quality-grader"
    stage = "quality_gate"

    def _run(  # type: ignore[override]
        self,
        *,
        facts: list[GatheredFact],
        quality: float,
        revised_plan: AnswerPlan,
        **_: Any,
    ) -> SkillObservation:
        findings: list[str] = []

        # --- sanity ---
        if not (0.0 <= quality <= 1.0):
            findings.append(f"sanity.quality_out_of_range:{quality:.4f}")

        has_critical = _critical_inputs_filled(facts)
        has_bet_card = OutputPackage.BET_CARD in revised_plan.output_packages
        has_downgrades = len(revised_plan.downgrades) > 0
        bet_card_threshold = revised_plan.quality_thresholds.get(
            OutputPackage.BET_CARD.value, _DEFAULT_BET_CARD_THRESHOLD
        )

        # --- under_downgrade ---
        if has_bet_card and not has_critical:
            findings.append("under_downgrade.bet_card_without_critical_inputs")

        if quality < _LOW_QUALITY_THRESHOLD and not has_downgrades:
            findings.append(f"under_downgrade.no_downgrades_on_low_quality:{quality:.2f}")

        # --- over_downgrade ---
        if not has_bet_card and has_critical and quality > bet_card_threshold:
            # Only flag this when bet_card was originally in scope for this query.
            # We infer intent from whether the original plan included betting modes.
            if revised_plan.betting_recommendations_included is False:
                # Check if this was genuinely dropped (downgrade present)
                if "dropped_bet_card" in revised_plan.downgrades:
                    findings.append(
                        f"over_downgrade.bet_card_dropped_above_threshold"
                        f":quality={quality:.2f},threshold={bet_card_threshold:.2f}"
                    )

        ok = len(findings) == 0
        return SkillObservation(skill=self.name, stage=self.stage, ok=ok, findings=findings)
