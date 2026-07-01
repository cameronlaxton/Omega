"""Omega trace enrichment — the opt-in LLM "Deep Dive" narrative subsystem.

This package is the ONE place in the console stack that writes. It is deliberately
kept OUT of :mod:`omega.ui` (which the read-only static guard scans) and writes
only to a separate sidecar database (``var/omega_enrichments.db``), never to the
canonical trace store. The read-only console is never imported into a mutation
path here, and this package is never imported by ``omega.ops.console_server`` —
the Deep Dive surface is reached only through the separate ``omega-enrich``
entry point that mounts this app beside the read-only console.

Doctrine (enforced by ``tests/enrichment/test_enrich_writes_only_sidecar.py``):

* the deterministic engine owns probability / edge / EV / grade / stake; the
  enrichment layer only *explains, challenges, and contextualizes* them;
* the LLM never asserts a protected number — the result schema has no numeric
  fields and the context pack carries qualitative trust factors, not raw edges;
* every write targets the enrichment sidecar DB; ``omega_traces.db`` is opened
  read-only (through the console service) for context only.
"""

from __future__ import annotations

PROMPT_VERSION = "enrich-v1"

__all__ = ["PROMPT_VERSION"]
