import pytest

from app.core.routing import route_task


@pytest.mark.parametrize(
    "task,expected",
    [
        ("Explain quantum computing simply", "ai"),
        ("Summarize this article for me", "ai"),
        ("What is the latest news about space telescopes?", "research"),
        ("Research the current price of copper", "research"),
        ("Analyze this CSV: average and correlation of columns", "data"),
        ("Show mean, median and missing values for my dataset", "data"),
        ("Train a model to predict price from this CSV", "ml"),
        ("Run classification on my dataset and report accuracy", "ml"),
        ("Do a regression analysis and forecast sales", "ml"),
    ],
)
def test_keyword_routing(task, expected):
    assert route_task(task).agent == expected


def test_ml_beats_data_when_both_match():
    # "csv" and "analyze" appear, but "train"/"model"/"predict" are more specific.
    decision = route_task("analyze this csv and train a model to predict price")
    assert decision.agent == "ml"
    assert decision.matched_keywords


def test_file_default_routes_to_data():
    decision = route_task("look at this", has_file=True)
    assert decision.agent == "data"


def test_file_referencing_task_routes_to_data():
    decision = route_task("summarize the dataset", has_file=True)
    assert decision.agent == "data"


def test_general_ai_task_with_file_stays_ai():
    # A general question must not be hijacked by an incidental attachment.
    decision = route_task("Explain how transformers work in simple terms", has_file=True)
    assert decision.agent == "ai"
    assert "context" in decision.reason


def test_general_default_is_ai():
    assert route_task("hello there").agent == "ai"


def test_explicit_override_wins():
    decision = route_task("what is the latest news", agent_override="ai")
    assert decision.agent == "ai"
    assert decision.reason == "Explicit agent override supplied with the request."


def test_unknown_override_raises():
    with pytest.raises(ValueError):
        route_task("anything", agent_override="gpt")


def test_research_priority_over_data():
    assert route_task("latest csv news").agent == "research"
