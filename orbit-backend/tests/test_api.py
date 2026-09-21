from fastapi.testclient import TestClient
import pytest

from app.core.results import AgentResult
from app.main import create_app
from tests.conftest import FakeLLMClient


@pytest.fixture()
def client(monkeypatch):
    """TestClient with the shared LLM swapped for a fake and jobs run inline."""
    from app.core import tasks as tasks_mod
    from app.agents.orchestrator import Orchestrator

    monkeypatch.setattr("app.llm.client.get_llm_client", lambda: FakeLLMClient())

    app = create_app()
    registry = tasks_mod.get_task_registry()

    # Run jobs synchronously so tests are deterministic.
    real_submit = registry.submit  # captured BEFORE monkeypatching

    def inline_submit(fn, **kwargs):
        task_id = real_submit(fn, **kwargs)
        entry = registry.get(task_id)
        entry.result = fn(task_id)
        entry.status = entry.result.status
        return task_id

    monkeypatch.setattr(registry, "submit", inline_submit)
    monkeypatch.setattr(
        Orchestrator, "run",
        lambda self, task, file_context=None, agent_override=None, history=None:
        AgentResult.create(
            "ai", "completed", "ok", data={"response": "FAKE-REPLY"},
            metadata={"routing": {"agent": "ai"}},
        ),
    )

    with TestClient(app) as c:
        yield c
    tasks_mod.reset_task_registry()


def test_root_and_health(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["agent"] == "orchestrator"
    assert "ORBIT" in body["message"]
    assert set(body["data"]["agents"]) == {"AI Agent", "Data Agent", "ML Agent", "Research Agent"}

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["ok"] is True
    assert h.json()["config"]["api_key_configured"] in (True, False)


def test_run_json_and_poll(client):
    r = client.post("/run", json={"task": "Explain gravity"})
    assert r.status_code == 202
    body = r.json()
    assert body["agent"] == "ai"
    assert body["status"] == "queued"
    assert body["poll_url"] == f"/status/{body['task_id']}"

    s = client.get(body["poll_url"])
    assert s.status_code == 200
    assert s.json()["result"]["status"] == "completed"
    assert s.json()["result"]["data"]["response"] == "FAKE-REPLY"


def test_run_with_csv_upload(client):
    csv = "a,b\n1,2\n3,4\n"
    r = client.post(
        "/run/file",
        data={"task": "analyze this csv", "agent": "data"},
        files={"file": ("data.csv", csv.encode(), "text/csv")},
    )
    assert r.status_code == 202
    assert r.json()["agent"] == "data"

    s = client.get(r.json()["poll_url"])
    assert s.json()["result"]["status"] == "completed"


def test_run_rejects_non_csv_upload(client):
    r = client.post(
        "/run/file",
        data={"task": "analyze this"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 422


def test_unknown_agent_override_422(client):
    r = client.post("/run", json={"task": "hi", "agent": "gpt"})
    assert r.status_code == 422


def test_unknown_task_404(client):
    assert client.get("/status/does-not-exist").status_code == 404


# -- deployment-prep behavior -------------------------------------------------


def test_api_docs_enabled_by_default(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_api_docs_disabled_in_production_mode(monkeypatch):
    """ENABLE_API_DOCS=0 removes /docs, /redoc and /openapi.json entirely."""
    from app.main import create_app as _create
    from app.config import get_settings

    monkeypatch.setenv("ENABLE_API_DOCS", "0")
    get_settings.cache_clear()
    try:
        with TestClient(_create()) as c:
            assert c.get("/docs").status_code == 404
            assert c.get("/redoc").status_code == 404
            assert c.get("/openapi.json").status_code == 404
            # Core API and health are unaffected.
            assert c.get("/health").status_code == 200
    finally:
        get_settings.cache_clear()


def test_cors_allows_configured_origin(client):
    r = client.options(
        "/run",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert r.status_code == 200  # request itself is served
    assert "access-control-allow-origin" not in r.headers  # but never blessed


def test_upload_size_limit_enforced(monkeypatch):
    """MAX_UPLOAD_BYTES is honored end-to-end with a clean 422."""
    from app.config import get_settings

    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1000")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as c:
            big = b"a,b\n" + b"1,2\n" * 900  # ~2.7 KB > 1000 B limit
            r = c.post(
                "/run/file",
                data={"task": "analyze this csv", "agent": "data"},
                files={"file": ("big.csv", big, "text/csv")},
            )
            assert r.status_code == 422
            assert "limit" in r.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_blank_task_422(client):
    assert client.post("/run", json={"task": ""}).status_code == 422


def test_runs_lists_recent_tasks(client):
    """GET /runs exposes the in-memory registry, newest first."""
    r1 = client.post("/run", json={"task": "Explain something"}).json()
    r2 = client.post(
        "/run/file",
        data={"task": "Analyze this CSV"},
        files={"file": ("s.csv", b"a,b\n1,2\n", "text/csv")},
    ).json()
    body = client.get("/runs").json()
    ids = [item["task_id"] for item in body]
    assert r1["task_id"] in ids and r2["task_id"] in ids
    by_id = {item["task_id"]: item for item in body}
    assert by_id[r1["task_id"]]["task"] == "Explain something"
    assert by_id[r1["task_id"]]["status"] == "completed"
    assert by_id[r1["task_id"]]["agent"] == "ai"
    assert by_id[r2["task_id"]]["agent_label"]
    assert by_id[r2["task_id"]]["routing_reason"]
    # Newest first (r2 submitted after r1).
    assert ids.index(r2["task_id"]) < ids.index(r1["task_id"])
    # No secrets in run summaries.
    assert all("api_key" not in str(item).lower() for item in body)
