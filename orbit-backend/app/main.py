"""ORBIT FastAPI application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import _cleanup_stale_uploads, router
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ORBIT — Multi-Agent AI Orchestration Platform",
        version="1.0.0",
        description=(
            "Deterministic multi-agent orchestration: AI, Data, ML, and Research "
            "agents behind a single Freebuff-powered Orchestrator."
        ),
        # Production: set ENABLE_API_DOCS=0 to stop exposing the schema/UIs.
        docs_url="/docs" if settings.enable_api_docs else None,
        redoc_url="/redoc" if settings.enable_api_docs else None,
        openapi_url="/openapi.json" if settings.enable_api_docs else None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    _cleanup_stale_uploads()

    frontend_dist = Path(__file__).resolve().parent.parent.parent / "orbit-frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()
