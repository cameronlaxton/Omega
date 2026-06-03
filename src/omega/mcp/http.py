"""Module entry point for the Omega MCP HTTP transport."""

from __future__ import annotations

from omega.mcp.http_app import run_http


def main() -> None:
    run_http()


if __name__ == "__main__":
    main()
