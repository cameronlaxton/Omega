"""
Omega FastAPI application.

Single entry point. Wire routes, middleware, session manager, and frontend.
Run with: uvicorn omega.api.app:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from omega.api.middleware.error_handler import ErrorHandlerMiddleware
from omega.api.routes import analysis, chat, health
from omega.api.session.manager import SessionManager
from omega.research.agent.orchestrator import Orchestrator, OrchestratorConfig

app = FastAPI(
    title="Omega",
    description="Quantitative sports analytics API",
    version="0.1.0",
)

# --- Middleware ---
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session manager ---
_redis_url = os.getenv("REDIS_URL")
session_manager = SessionManager(redis_url=_redis_url)
chat.set_session_manager(session_manager)

# --- Orchestrator ---
orchestrator = Orchestrator(OrchestratorConfig())
chat.set_orchestrator(orchestrator)

# --- API Routes ---
app.include_router(health.router)
app.include_router(analysis.router)
app.include_router(chat.router)

# --- Frontend static files ---
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir / "static")), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend SPA."""
        return FileResponse(str(_frontend_dir / "index.html"))
