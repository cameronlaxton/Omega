from __future__ import annotations

from omega.core.models import (
    AnswerPlan,
    ExecutionMode,
    ExecutionResult,
    GatheredFact,
    GatherSlot,
    InputImportance,
    OutputPackage,
    QueryUnderstanding,
)
from omega.synthesis.composer import compose_response


def test_limited_context_answer_renders_structured_section():
    fact = GatheredFact(
        slot=GatherSlot(
            key="home_team.off_rating",
            data_type="team_stat",
            entity="Boston Celtics",
            league="NBA",
            importance=InputImportance.CRITICAL,
        ),
        filled=False,
        quality_score=0.0,
    )
    plan = AnswerPlan(
        execution_modes=[ExecutionMode.RESEARCH],
        output_packages=[OutputPackage.LIMITED_CONTEXT_ANSWER],
        simulation_required=False,
        betting_recommendations_included=False,
        downgrades=["ultra_low_data"],
    )
    execution = ExecutionResult(
        mode=ExecutionMode.RESEARCH,
        data_quality_score=0.0,
    )
    understanding = QueryUnderstanding(
        subjects=[],
        league="NBA",
        raw_prompt="Celtics vs Pacers",
    )

    response = compose_response(understanding, plan, [fact], execution)

    assert response["type"] == "answer"
    assert "limited_context" in response["sections"]
    assert "bet_card" not in response["sections"]
    assert response["sections"]["limited_context"]["filled_facts"] == 0
    assert response["sections"]["limited_context"]["total_facts"] == 1
    assert "limited data" in response["text"].lower()


def test_catastrophic_empty_execution_renders_fallback_text():
    plan = AnswerPlan(
        execution_modes=[ExecutionMode.RESEARCH],
        output_packages=[],
        simulation_required=False,
        betting_recommendations_included=False,
    )
    execution = ExecutionResult(
        mode=ExecutionMode.RESEARCH,
        data_quality_score=0.0,
    )
    understanding = QueryUnderstanding(
        subjects=[],
        league="NBA",
        raw_prompt="Celtics vs Pacers",
    )

    response = compose_response(understanding, plan, [], execution)

    assert response["type"] == "answer"
    assert "fallback" in response["sections"]
    assert "bet_card" not in response["sections"]
    assert response["text"]
    assert "not enough verified input data" in response["text"].lower()
