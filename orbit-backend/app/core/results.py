"""Standardized agent result envelope.

Every agent returns, and every endpoint responds with, this exact shape:
    { agent, status, message, data, metadata }
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["queued", "running", "completed", "failed", "partial", "insufficient_data"]

QUEUED: Status = "queued"
RUNNING: Status = "running"
COMPLETED: Status = "completed"
FAILED: Status = "failed"
PARTIAL: Status = "partial"
INSUFFICIENT_DATA: Status = "insufficient_data"

AGENT_LABELS = {
    "orchestrator": "Orchestrator",
    "ai": "AI Agent",
    "data": "Data Agent",
    "ml": "ML Agent",
    "research": "Research Agent",
}


class AgentResult(BaseModel):
    """Canonical result envelope for all agents."""

    agent: str = "orchestrator"
    status: Status = "queued"
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def create(agent: str, status: Status, message: str = "", data: dict[str, Any] | None = None,
               metadata: dict[str, Any] | None = None) -> "AgentResult":
        return AgentResult(
            agent=agent,
            status=status,
            message=message,
            data=data or {},
            metadata={"started_at": time.time(), **(metadata or {})},
        )

    def label(self) -> str:
        return AGENT_LABELS.get(self.agent, self.agent.replace("_", " ").title())
