"""Shared guards for omega.integrations live-data modules."""

from __future__ import annotations

import os


class OmegaReplayModeError(RuntimeError):
    """Raised when a live network fetch is attempted with OMEGA_REPLAY_MODE=1."""


def assert_not_replay_mode(fetch_description: str = "network fetch") -> None:
    """Raise OmegaReplayModeError if OMEGA_REPLAY_MODE=1 in the environment.

    Set OMEGA_REPLAY_MODE=1 when running historical trace evaluations or replay
    sessions where live data must never be fetched. Callers that supply a mock
    url_opener in tests do not trip this guard — the guard is env-var-driven and
    is intended to protect production replay sessions, not unit tests.
    """
    if os.environ.get("OMEGA_REPLAY_MODE") == "1":
        raise OmegaReplayModeError(
            f"Live {fetch_description} blocked: OMEGA_REPLAY_MODE=1. "
            "Unset OMEGA_REPLAY_MODE for live sessions, or use frozen fixtures "
            "for deterministic replay."
        )
