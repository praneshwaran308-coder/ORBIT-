"""Pydantic request/response schemas for the ORBIT API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.results import AgentResult


class RunRequest(BaseModel):
    """Body for POST /run (JSON mode)."""

    task: str = Field(min_length=1, max_length=8000, description="The task for ORBIT.")
    agent: str | None = Field(
        default=None, description="Optional explicit agent: ai | data | ml | research."
    )
    history: list[dict[str, Any]] | None = Field(
        default=None, description="Optional recent conversation turns for the AI Agent."
    )


class RunAccepted(BaseModel):
    """Response for POST /run."""

    task_id: str
    agent: str
    agent_label: str
    status: str
    routing_reason: str
    poll_url: str


class StatusResponse(BaseModel):
    """Response for GET /status/{task_id}."""

    task_id: str
    result: AgentResult | None
    # Registry-level lifecycle status (queued/running/terminal). Additive field
    # so observers can see live progress while `result` is still null.
    status: str = "unknown"


class RunSummary(BaseModel):
    """One row of the Runs view: a previous task execution."""

    task_id: str
    task: str
    agent: str
    agent_label: str
    status: str
    routing_reason: str
    created_at: float
    duration_seconds: float | None = None

    @classmethod
    def from_entry(cls, entry) -> "RunSummary":
        result = entry.result
        duration = None
        if result is not None:
            duration = result.metadata.get("duration_seconds")
        return cls(
            task_id=entry.task_id,
            task=(entry.task or (result.message if result else ""))[:120],
            agent=entry.agent or (result.agent if result else ""),
            agent_label=entry.agent_label or (result.label() if result else ""),
            status=entry.status,
            routing_reason=entry.routing_reason,
            created_at=entry.created_at,
            duration_seconds=duration,
        )


class HealthResponse(BaseModel):
    ok: bool
    version: str
    config: dict[str, Any]
