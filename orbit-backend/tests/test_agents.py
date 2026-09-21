import pandas as pd
import pytest

from app.agents.ai_agent import AIAgent
from app.agents.base import InsufficientDataError
from app.agents.data_agent import DataAgent
from app.core.files import FileContext, FileValidationError
from app.core.results import AgentResult
from tests.conftest import FakeLLMClient, make_csv_context


# -- BaseAgent envelope ---------------------------------------------------------


class BoomAgent(AIAgent):
    name = "boom"

    def _run(self, task, file_context=None, history=None):
        raise ValueError("kaboom")


class StarveAgent(AIAgent):
    name = "starve"

    def _run(self, task, file_context=None, history=None):
        raise InsufficientDataError("not enough rows")


def test_base_agent_maps_exceptions(fake_llm):
    result = BoomAgent(llm=fake_llm).run("task")
    assert result.status == "failed"
    assert "kaboom" in result.message
    assert "duration_seconds" in result.metadata

    result2 = StarveAgent(llm=fake_llm).run("task")
    assert result2.status == "insufficient_data"
    assert "not enough rows" in result2.message


# -- AI Agent ---------------------------------------------------------------------


def test_ai_agent_grounded_reply(fake_llm):
    result = AIAgent(llm=fake_llm).run("Explain the meaning of life")
    assert result.agent == "ai"
    assert result.status == "completed"
    assert result.data["response"] == "FAKE-REPLY"
    assert result.metadata["model"] == "fake-model"


def test_ai_agent_includes_file_preview(fake_llm):
    ctx = make_csv_context(rows=12)
    AIAgent(llm=fake_llm).run("Summarize the uploaded file", file_context=ctx)
    sent = fake_llm.calls[-1]
    blob = " ".join(m["content"] for m in sent)
    assert "sample.csv" in blob and "Shape:" in blob


# -- Data Agent ---------------------------------------------------------------------


def test_data_agent_requires_file(fake_llm):
    result = DataAgent(llm=fake_llm).run("analyze this")
    assert result.status == "failed"
    assert "CSV" in result.message


def test_data_agent_local_stats(fake_llm):
    ctx = make_csv_context(rows=40)
    result = DataAgent(llm=fake_llm).run("Analyze this csv", file_context=ctx)
    assert result.status == "completed"
    report = result.data["report"]
    assert report["shape"]["rows"] == 40
    assert report["shape"]["columns"] == 4
    summary = report["numeric_summary"]["y"]
    # y = 2x + (i%3), so mean of y for rows 0..39 is ~ 2*19.5 + 1 = 40.0
    assert summary["mean"] == pytest.approx(40.0, abs=0.6)
    assert summary["median"] == pytest.approx(40.0, abs=1.0)
    assert summary["min"] == 0.0
    assert "x" in [c["name"] for c in report["columns"]]
    # x and y should be almost perfectly correlated
    pair = next(p for p in report["correlations"] if {p["a"], p["b"]} == {"x", "y"})
    assert pair["r"] == pytest.approx(0.999, abs=0.01)
    # LLM received only computed stats (narration role)
    assert result.data["narrative"] == "FAKE-REPLY"
    assert result.metadata["computed_locally"] is True


def test_data_agent_handles_all_text_csv(fake_llm):
    raw = "name,city\nAnn,Oslo\nBob,Rome\n".encode()
    ctx = FileContext.from_upload("text.csv", raw)
    result = DataAgent(llm=fake_llm).run("analyze", file_context=ctx)
    assert result.status == "completed"
    assert result.data["report"]["numeric_summary"] == {}
    assert "No numeric columns" in " ".join(result.data["report"]["insights"])


def test_data_agent_narration_fallback_on_llm_error():
    from app.llm.client import LLMError

    class BrokenLLM(FakeLLMClient):
        def chat_sync(self, messages, **kwargs):
            raise LLMError("freebuff down")

    ctx = make_csv_context(rows=40)
    result = DataAgent(llm=BrokenLLM()).run("analyze", file_context=ctx)
    assert result.status == "completed"
    assert "narration unavailable" in result.data["narrative"].lower()
    # Stats are still present even when the LLM narration fails.
    assert result.data["report"]["shape"]["rows"] == 40


# -- FileContext -------------------------------------------------------------------


def test_file_context_rejects_non_csv():
    with pytest.raises(FileValidationError):
        FileContext.from_upload("notes.txt", b"hello")


def test_file_context_rejects_bad_csv():
    with pytest.raises(FileValidationError):
        FileContext.from_upload("bad.csv", b"a,b\n\"unclosed")
