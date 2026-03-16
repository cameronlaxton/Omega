"""
Global error handling middleware.

Catches unhandled exceptions and returns structured JSON errors
instead of raw 500 tracebacks.
"""

from __future__ import annotations

import logging
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("omega.api.middleware")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        try:
            return await call_next(request)
        except Exception as exc:
            logger.error(
                "Unhandled error on %s %s: %s",
                request.method,
                request.url.path,
                exc,
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "INTERNAL_ERROR",
                    "message": str(exc),
                    "path": str(request.url.path),
                },
            )
