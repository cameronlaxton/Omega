"""EXPERIMENTAL writing-style skill: a tiny style profile and optional rewrites.

Status: experimental, **disabled by default** (see omega/skills/config.json →
``enabled["writing-style"] == false``). The implementation is intentionally a
naive heuristic (regex/character counts), not an LLM/ML model.

Scope guard: this skill is narrative-only. It must never touch formal Omega
outputs — no Bet Card / EdgeDetail field, no probability, EV%, edge%, Kelly,
units, confidence tier, or trace_id passes through here (see CLAUDE.md hard
rule). It is an observer over free text, kept behind the disabled flag until a
real implementation is justified; do not enable it in a formal pipeline.
"""

from . import is_enabled, register
from .logger import write_event


@register("writing-style")
class WritingStyle:
    def __init__(self):
        self.name = "writing-style"

    def profile(self, text: str) -> dict:
        # naive heuristics
        profile = {
            "length": len(text),
            "avg_sentence_len": _avg_sentence_len(text),
            "tone": _guess_tone(text),
        }
        return profile

    def suggest_rewrite(self, text: str) -> str:
        # trivial rewrite: trim whitespace and collapse spaces
        return " ".join(text.strip().split())

    def handle_event(self, event: dict):
        if not is_enabled(self.name):
            return
        evt = {"skill": self.name, "event": event}
        write_event(evt)


def _avg_sentence_len(text: str) -> float:
    s = [s for s in text.replace("!", ".").split(".") if s.strip()]
    if not s:
        return 0.0
    return sum(len(x.split()) for x in s) / len(s)


def _guess_tone(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["please", "thank you", "appreciate"]):
        return "polite"
    if len(text) < 60:
        return "concise"
    return "neutral"
