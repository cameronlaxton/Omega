"""Omega MCP interface.

This package exposes Omega's existing deterministic contracts as LLM-callable
tools. It is intentionally an adapter layer: analysis, calibration, trace
persistence, staking, grading, and replay policy stay in their owning packages.
"""

from omega.mcp.schemas import MCP_SCHEMA_VERSION

__all__ = ["MCP_SCHEMA_VERSION"]
