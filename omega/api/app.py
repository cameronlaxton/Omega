"""
Omega FastAPI application.

Single entry point. Wire routes, middleware, and session manager.
Run with: uvicorn omega.api.app:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omega.api.middleware.error_handler import ErrorHandlerMiddleware
from omega.api.routes import analysis, chat, health
from omega.api.session.manager import SessionManager

app = FastAPI(
    title="Omega",
    description="Quantitative sports analytics API",
    version="0.1.0",
)

# --- Middleware ---
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session manager ---
_redis_url = os.getenv("REDIS_URL")
session_manager = SessionManager(redis_url=_redis_url)
chat.set_session_manager(session_manager)

# --- Routes ---
app.include_router(health.router)
app.include_router(analysis.router)
app.include_router(chat.router)
