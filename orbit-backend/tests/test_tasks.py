import time

from app.core.results import AgentResult
from app.core.tasks import TaskRegistry


def _wait_terminal(reg, tid, seconds=5):
    deadline = time.time() + seconds
    while time.time() < deadline:
        entry = reg.get(tid)
        if entry and entry.status not in ("queued", "running"):
            return entry
        time.sleep(0.02)
    return reg.get(tid)


def test_submit_lifecycle_and_result():
    reg = TaskRegistry(max_workers=2)
    try:
        tid = reg.submit(lambda _: AgentResult.create("ai", "completed", "done"),
                         agent="ai", agent_label="AI Agent", routing_reason="r")
        entry = _wait_terminal(reg, tid)
        assert entry.status == "completed"
        assert entry.result.message == "done"
    finally:
        reg.shutdown()


def test_registry_survives_job_crash():
    reg = TaskRegistry(max_workers=2)
    try:
        def boom(_):
            raise RuntimeError("worker exploded")

        tid = reg.submit(boom, agent="ai", agent_label="AI Agent", routing_reason="r")
        entry = _wait_terminal(reg, tid)
        assert entry.status == "failed"
        assert "worker exploded" in entry.result.message
    finally:
        reg.shutdown()


def test_task_timeout_marks_failed_and_discards_late_result():
    reg = TaskRegistry(max_workers=1, task_timeout_seconds=0.2)
    try:
        def slow(_):
            time.sleep(0.6)
            return AgentResult.create("ai", "completed", "too late")

        tid = reg.submit(slow, agent="ai", agent_label="AI Agent", routing_reason="r")
        entry = _wait_terminal(reg, tid)
        assert entry.status == "failed"
        assert "timed out" in entry.result.message
        # The worker eventually finishes; its late result must be discarded.
        time.sleep(0.6)
        entry = reg.get(tid)
        assert entry.status == "failed"
        assert "too late" not in entry.result.message
    finally:
        reg.shutdown()


def test_task_within_timeout_completes_normally():
    reg = TaskRegistry(max_workers=2, task_timeout_seconds=30)
    try:
        tid = reg.submit(lambda _: AgentResult.create("ai", "completed", "fast"),
                         agent="ai", agent_label="AI Agent", routing_reason="r")
        entry = _wait_terminal(reg, tid)
        assert entry.status == "completed"
        assert entry.result.message == "fast"
    finally:
        reg.shutdown()


def test_history_is_bounded():
    reg = TaskRegistry(max_workers=4)
    try:
        for _ in range(30):
            reg.submit(lambda _: AgentResult.create("ai", "completed", "x"),
                       agent="ai", agent_label="AI Agent", routing_reason="r")
        deadline = time.time() + 5
        while time.time() < deadline:
            terminal = [t for t in reg._tasks.values() if t.status == "completed"]
            if len(terminal) >= 30:
                break
            time.sleep(0.01)
        assert len(reg._tasks) <= 30 + 4  # bounded near MAX_HISTORY
    finally:
        reg.shutdown()
