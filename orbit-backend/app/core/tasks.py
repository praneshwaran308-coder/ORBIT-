"""In-memory task registry + bounded worker pool.

Tasks transition queued -> running -> a terminal status (completed | failed |
partial | insufficient_data). Completed tasks are kept up to MAX_HISTORY;
older terminal tasks are dropped. Worker threads run agent code so pandas /
scikit-learn never block the FastAPI event loop.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from app.core.results import AgentResult, QUEUED, RUNNING

logger = logging.getLogger("orbit.tasks")

MAX_HISTORY = 100


@dataclass
class TaskEntry:
    task_id: str
    status: str
    result: AgentResult | None = None
    error: str | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())
    # Execution-observability metadata (additive; safe defaults keep the
    # registry backward compatible with existing callers/tests).
    task: str = ""
    agent: str = ""
    agent_label: str = ""
    routing_reason: str = ""


class TaskRegistry:
    """Thread-safe, bounded, in-memory task store."""

    def __init__(self, max_workers: int = 4,
                 task_timeout_seconds: float | None = None) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskEntry] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="orbit-worker")
        if task_timeout_seconds is None:
            from app.config import get_settings

            task_timeout_seconds = get_settings().task_timeout_seconds
        self._task_timeout = float(task_timeout_seconds)

    # -- public API -----------------------------------------------------------
    def submit(self, fn: Callable[[str], AgentResult], *, agent: str,
               agent_label: str, routing_reason: str, task: str = "") -> str:
        task_id = uuid.uuid4().hex
        entry = TaskEntry(task_id=task_id, status=QUEUED, task=task,
                          agent=agent, agent_label=agent_label,
                          routing_reason=routing_reason)
        with self._lock:
            self._tasks[task_id] = entry
        self._executor.submit(self._execute, task_id, fn)
        return task_id

    def get(self, task_id: str) -> TaskEntry | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_recent(self) -> list["TaskEntry"]:
        """All known tasks, newest first (for the read-only Runs view)."""
        with self._lock:
            return sorted(self._tasks.values(),
                          key=lambda e: e.created_at, reverse=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- internals ------------------------------------------------------------
    def _execute(self, task_id: str, fn: Callable[[str], AgentResult]) -> None:
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            entry.status = RUNNING
        # Watchdog: mark the task failed after the configured timeout. The
        # worker thread is never killed (unsafe); if it eventually finishes,
        # its late result is discarded because the entry is no longer RUNNING.
        timer: threading.Timer | None = None
        if self._task_timeout > 0:
            timer = threading.Timer(self._task_timeout, self._on_timeout,
                                    args=(task_id, self._task_timeout))
            timer.daemon = True
            timer.start()
        try:
            result = fn(task_id)
        except Exception as exc:  # noqa: BLE001 - registry must never crash
            logger.exception("Task %s crashed", task_id)
            from app.core.results import FAILED

            result = AgentResult.create("orchestrator", FAILED,
                                        f"Internal error: {type(exc).__name__}: {exc}")
        finally:
            if timer is not None:
                timer.cancel()
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            if entry.status == RUNNING:
                entry.result = result
                entry.status = result.status
            else:
                # A terminal status was set elsewhere (watchdog): discard.
                logger.info("Discarding late result for timed-out task %s", task_id)
            # Bound memory: drop oldest terminal tasks beyond MAX_HISTORY.
            terminal = [t for t in self._tasks.values()
                        if t.status not in (QUEUED, RUNNING)]
            if len(terminal) > MAX_HISTORY:
                for stale in terminal[: len(terminal) - MAX_HISTORY]:
                    self._tasks.pop(stale.task_id, None)

    def _on_timeout(self, task_id: str, timeout: float) -> None:
        from app.core.results import FAILED

        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None or entry.status != RUNNING:
                return
            entry.result = AgentResult.create(
                "orchestrator", FAILED,
                f"Task timed out after {timeout:g}s and was stopped. "
                "Try a smaller task or a smaller file.",
            )
            entry.status = FAILED
        logger.warning("Task %s timed out after %gs", task_id, timeout)


# Process-wide registry (FastAPI dependency).
_registry: TaskRegistry | None = None


def get_task_registry() -> TaskRegistry:
    global _registry
    if _registry is None:
        _registry = TaskRegistry()
    return _registry


def reset_task_registry() -> None:
    global _registry
    if _registry is not None:
        _registry.shutdown()
    _registry = None
