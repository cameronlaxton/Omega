"""
omega.skills.evidence_validator — validate gathered evidence integrity.

Observes stage 4 (after gather_facts). Checks two concern groups:

evidence_integrity:
  - filled=True but result is None (silent null injection)
  - confidence < 0.3 on a CRITICAL slot (low-confidence critical input)
  - result.data is empty on a CRITICAL filled slot (no usable values)

source_attribution:
  - result.source is empty or "unknown" on a CRITICAL filled fact
    (indicates no real provider returned data)
  - result.method is "unknown" on a CRITICAL filled fact
    (indicates the retrieval path is unidentified)

These checks catch the primary path by which bad sim inputs enter: a slot
reports filled=True but the underlying data is absent, synthetic, or unattributed.
"""
from __future__ import annotations

from typing import Any, List

from omega.skills import register
from omega.skills.base import SkillBase, SkillObservation
from omega.core.models import GatheredFact, InputImportance

_LOW_CONFIDENCE_THRESHOLD = 0.3


@register("evidence-validator")
class EvidenceValidator(SkillBase):
    name = "evidence-validator"
    stage = "gathering"

    def _run(self, *, facts: List[GatheredFact], **_: Any) -> SkillObservation:
        findings: list[str] = []

        for fact in facts:
            slot_key = fact.slot.key
            is_critical = fact.slot.importance == InputImportance.CRITICAL

            # --- evidence integrity ---
            if fact.filled and fact.result is None:
                findings.append(f"evidence_integrity.null_result:{slot_key}")
                continue  # further checks require result

            if fact.filled and fact.result is not None:
                if is_critical and fact.result.confidence < _LOW_CONFIDENCE_THRESHOLD:
                    findings.append(
                        f"evidence_integrity.low_confidence:{slot_key}"
                        f":{fact.result.confidence:.2f}"
                    )
                if is_critical and not fact.result.data:
                    findings.append(f"evidence_integrity.empty_data:{slot_key}")

                # --- source attribution ---
                if is_critical:
                    source = (fact.result.source or "").strip()
                    if not source or source.lower() in ("", "unknown"):
                        findings.append(f"source_attribution.no_source:{slot_key}")

                    method = (fact.result.method or "").strip()
                    if method.lower() == "unknown":
                        findings.append(f"source_attribution.unknown_method:{slot_key}")

        ok = len(findings) == 0
        return SkillObservation(skill=self.name, stage=self.stage, ok=ok, findings=findings)
