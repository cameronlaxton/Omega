"""Trace Replay timeline — a read-only reconstruction of the decision flow.

It only reorders/labels fields already persisted on the trace; nothing is
recomputed. These tests pin the step set and the decision ordering.
"""

from __future__ import annotations


def test_trace_detail_renders_decision_replay(client):
    html = client.get("/traces/sandbox-aaa").text
    assert "Decision Replay" in html
    for label in (
        "Inputs &amp; context",
        "Odds snapshot",
        "Evidence signals",
        "Model probability",
        "LLM narrative",
        "Deterministic engine / simulation",
        "Gates &amp; QA",
        "Final recommendation",
        "Outcome",
    ):
        assert label in html, f"missing timeline step: {label}"


def test_decision_replay_is_ordered(client):
    html = client.get("/traces/sandbox-aaa").text
    # The timeline runs inputs → recommendation → outcome.
    assert html.index("Inputs &amp; context") < html.index("Final recommendation") < html.index("Outcome")


def test_decision_replay_outcome_states(client):
    # sandbox-aaa has an attached outcome; sandbox-bbb does not.
    graded = client.get("/traces/sandbox-aaa").text
    ungraded = client.get("/traces/sandbox-bbb").text
    assert "graded" in graded.split("Outcome", 1)[1][:120]
    assert "not graded yet" in ungraded
