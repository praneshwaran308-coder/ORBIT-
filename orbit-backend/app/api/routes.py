"""ORBIT API routes."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    HealthResponse,
    RunAccepted,
    RunRequest,
    RunSummary,
    StatusResponse,
)
from app.config import describe_config
from app.core.files import FileContext, FileValidationError
from app.core.results import AGENT_LABELS, AgentResult
from app.core.routing import SUPPORTED_AGENTS, route_task
from app.core.tasks import get_task_registry

logger = logging.getLogger("orbit.api")

VERSION = "1.0.0"
router = APIRouter()


def _cleanup_stale_uploads(max_age_hours: int = 6) -> None:
    """Delete upload copies older than max_age_hours (startup hygiene)."""
    from app.config import DATA_DIR

    uploads = DATA_DIR / "uploads"
    if not uploads.is_dir():
        return
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for f in uploads.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("Startup cleanup: removed %d stale upload(s).", removed)


@router.get("/", response_model=AgentResult)
def root() -> AgentResult:
    """Platform descriptor in the standard envelope."""
    return AgentResult(
        agent="orchestrator",
        status="completed",
        message="ORBIT — Multi-Agent AI Orchestration Platform. "
                "See /docs for the interactive API reference.",
        data={
            "version": VERSION,
            "agents": [AGENT_LABELS[a] for a in SUPPORTED_AGENTS],
            "endpoints": ["/", "/health", "/run", "/run/file", "/status/{task_id}", "/runs", "/docs"],
        },
        metadata={"served_at": time.time()},
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, version=VERSION, config=describe_config())


@router.post("/run", response_model=RunAccepted, status_code=202)
async def run_task(request: RunRequest) -> RunAccepted:
    """Accept a task (JSON body, no file)."""
    return _accept_task(task=request.task, agent_override=request.agent,
                        history=request.history, upload=None)


@router.post("/run/file", response_model=RunAccepted, status_code=202)
async def run_task_with_file(
    task: str = Form(..., min_length=1, max_length=8000),
    agent: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> RunAccepted:
    """Accept a task with an optional CSV upload (multipart form)."""
    return _accept_task(task=task, agent_override=agent, history=None, upload=file)


@router.get("/status/{task_id}", response_model=StatusResponse)
async def status(task_id: str) -> StatusResponse:
    entry = get_task_registry().get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    return StatusResponse(task_id=task_id, result=entry.result, status=entry.status)


@router.get("/runs", response_model=list[RunSummary])
async def runs() -> list[RunSummary]:
    """Recent task executions from the in-memory registry (current session).

    Read-only observability view over the SAME bounded registry that backs
    /status — no new storage, newest first.
    """
    items = [RunSummary.from_entry(e) for e in get_task_registry().list_recent()]
    return items


def _accept_task(*, task: str, agent_override: str | None,
                 history: list[dict[str, Any]] | None,
                 upload: UploadFile | None) -> RunAccepted:
    file_context: FileContext | None = None
    if upload is not None:
        try:
            raw = upload.file.read()
            file_context = FileContext.from_upload(upload.filename or "upload.csv", raw)
        except FileValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    try:
        decision = route_task(task, has_file=file_context is not None,
                              agent_override=agent_override)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    orchestrator = _get_orchestrator()

    def job(task_id: str) -> AgentResult:
        return orchestrator.run(task, file_context=file_context,
                                agent_override=agent_override, history=history)

    task_id = get_task_registry().submit(
        job, agent=decision.agent, agent_label=decision.agent_label,
        routing_reason=decision.reason, task=task,
    )
    # The validated CSV has been fully read into memory for the run; the disk
    # copy is only a crash-surrogate. Remove it so uploads don't accumulate.
    if file_context is not None:
        try:
            Path(file_context.path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove upload copy %s", file_context.path)
    return RunAccepted(
        task_id=task_id,
        agent=decision.agent,
        agent_label=decision.agent_label,
        status="queued",
        routing_reason=decision.reason,
        poll_url=f"/status/{task_id}",
    )


def _get_orchestrator():
    from app.agents.orchestrator import Orchestrator

    return Orchestrator()
