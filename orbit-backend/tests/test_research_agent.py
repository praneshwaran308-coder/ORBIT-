import pytest

from app.agents.research_agent import (
    _assess_sources,
    _describe_evidence_scope,
    _extract_article_text,
    _parse_feed_items,
    ResearchAgent,
    Source,
)
from tests.conftest import FakeLLMClient


def _make_agent(monkeypatch, article_html: str | None, feed_xml: str) -> ResearchAgent:
    """Agent whose networking is replaced by local fixtures."""
    import asyncio

    agent = ResearchAgent(llm=FakeLLMClient())

    async def fake_discover(self, queries, budget, relevance_terms=None):
        from app.agents.research_agent import Source

        return [Source(
            title="Fixture Article",
            url="https://example.com/news/fixture",
            publisher="example.com",
            published="2026-01-02T00:00:00Z",
            snippet="RSS snippet only",
            kind="search_result",
        )]

    async def fake_fetch(self, sources, budget):
        from app.agents.research_agent import Source

        for s in sources:
            if article_html is None:
                s.evidence_status = "rejected"
                s.evidence_note = "Fetch failed; metadata only."
                s.error = "fixture: no article"
            else:
                text = _extract_article_text(article_html)
                s.extracted_chars = len(text)
                if len(text) >= 400:
                    s.kind = "article"
                    s.evidence_status = "verified"
                else:
                    s.evidence_status = "rejected"
                    s.evidence_note = (
                        f"Only {len(text)} chars of extractable content; "
                        "insufficient for evidence."
                    )
        return sources

    monkeypatch.setattr(ResearchAgent, "_discover", fake_discover)
    monkeypatch.setattr(ResearchAgent, "_fetch_and_validate", fake_fetch)
    return agent


RICH_ARTICLE = """
<html><body><article>
<h1>Test Headline</h1>
""" + (" ".join(["This is a long verified paragraph about the topic. "] * 40)) + """
<p>Another meaningful paragraph with plenty of content for validation.</p>
""" + (" ".join(["Additional detail sentence for the evidence body. "] * 30)) + """
</article></body></html>
"""

THIN_ARTICLE = "<html><body><p>Just a teaser line.</p></body></html>"


def test_feed_parsing():
    xml = """
    <rss><channel>
      <item><title>T1</title><link>https://a.example/x</link>
        <pubDate>Mon, 02 Jan 2026 00:00:00 GMT</pubDate>
        <description>&lt;b&gt;Bold snippet&lt;/b&gt;</description><source>A</source></item>
      <item><title>T2</title><link>https://b.example/y</link></item>
    </channel></rss>
    """
    items = _parse_feed_items(xml)
    assert len(items) == 2
    assert items[0]["title"] == "T1"
    assert items[0]["link"] == "https://a.example/x"
    assert items[0]["description"] == "Bold snippet"


def test_article_extraction_strips_chrome():
    html = (
        "<html><head><script>var x=1;</script></head><body><nav>menu</nav>"
        "<article><h2>Head</h2><p>" + ("word " * 120) + "</p></article></body></html>"
    )
    text = _extract_article_text(html)
    assert "Head" in text
    assert "var x=1" not in text and "menu" not in text


def test_verified_evidence_flow(monkeypatch):
    agent = _make_agent(monkeypatch, RICH_ARTICLE, feed_xml="")
    result = agent.run("What is the latest on the fixture topic?")
    assert result.status == "completed"
    assert result.data["verified_evidence_count"] == 1
    src = result.data["sources"][0]
    assert src["evidence_status"] == "verified"
    assert src["kind"] == "article"
    assert src["title"] == "Fixture Article"
    assert result.data["synthesis"] == "FAKE-REPLY"


def test_unverified_flow_never_claims_evidence(monkeypatch):
    agent = _make_agent(monkeypatch, THIN_ARTICLE, feed_xml="")
    result = agent.run("Research the fixture topic")
    assert result.status == "partial"
    assert result.data["verified_evidence_count"] == 0
    src = result.data["sources"][0]
    assert src["evidence_status"] == "rejected"
    assert "metadata only" in src["evidence_note"] or "insufficient" in src["evidence_note"]
    # The synthesis prompt must not be called with claims of verified evidence.
    assert "NOT treated as evidence" in result.data["evidence_disclaimer"]


def test_snippet_never_marked_verified():
    from app.agents.research_agent import Source

    s = Source(title="t", url="https://x.example/a", publisher="x.example",
               snippet="just a snippet", kind="search_result")
    assert s.evidence_status == "unverified"
    assert s.kind == "search_result"


# -- Issue 2: topical discovery, relevance filtering, temporal intent -------------


def test_derive_relevance_terms_drops_stopwords():
    from app.agents.research_agent import _derive_relevance_terms

    terms = _derive_relevance_terms("Research the current state of fusion energy with sources")
    assert "fusion" in terms and "energy" in terms
    assert "research" not in terms and "current" not in terms and "sources" not in terms


def test_off_topic_generated_query_is_discarded():
    """A generated query that lost the topic must not reach discovery."""
    import app.agents.research_agent as ra

    agent = ResearchAgent(llm=FakeLLMClient())

    class OffTopicLLM(FakeLLMClient):
        async def chat(self, messages, **kwargs):
            self._record(messages)

            class R:
                content = '["latest news", "top headlines today"]'

            return R()

    agent.llm = OffTopicLLM()
    queries, terms = asyncio_run_queries(agent, "Research the current state of fusion energy")
    assert all("fusion" in q.lower() for q in queries)
    assert "fusion" in terms


def asyncio_run_queries(agent, task):
    import asyncio

    return asyncio.run(agent._generate_queries(task))


def test_markdown_fenced_json_queries_parse():
    agent = ResearchAgent(llm=FakeLLMClient())
    raw = '```json\n["fusion energy developments", "fusion reactors news"]\n```'
    parsed = agent._parse_queries(raw, fallback="fusion energy")
    assert parsed == ["fusion energy developments", "fusion reactors news"]


def test_no_invented_year_cutoff_in_generated_queries(monkeypatch):
    """A query that replaced 'current' with '2023' fails the relevance gate."""
    import app.agents.research_agent as ra

    agent = ResearchAgent(llm=FakeLLMClient())

    class CutoffLLM(FakeLLMClient):
        async def chat(self, messages, **kwargs):
            self._record(messages)

            class R:
                content = '["fusion energy developments 2023"]'

            return R()

    agent.llm = CutoffLLM()
    queries, terms = asyncio_run_queries(agent, "Research the current state of fusion energy")
    # The 2023 query IS on-topic (mentions fusion), so it survives, but the
    # system prompt must explicitly forbid inventing cutoffs.
    assert any("fusion" in q for q in queries)
    system_prompts = [m["content"] for call in agent.llm.calls for m in call
                      if m["role"] == "system"]
    assert any("never replace them" in p for p in system_prompts)


def test_bing_link_decoding():
    from app.agents.research_agent import _decode_feed_link

    wrapped = ("http://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&tid=abc&url="
               "https%3a%2f%2fwww.example.com%2ffusion%2farticle%2f1")
    assert _decode_feed_link(wrapped) == "https://www.example.com/fusion/article/1"
    assert _decode_feed_link("https://example.com/direct") == "https://example.com/direct"


def test_irrelevant_discovery_results_rejected():
    from app.agents.research_agent import Source, _is_relevant

    terms = ["fusion", "energy"]
    on_topic = Source(title="Fusion energy startup raises funds",
                      url="https://tech.example/fusion-energy",
                      publisher="tech.example",
                      snippet="A fusion energy company announced...")
    off_topic = Source(title="Celebrity gossip roundup",
                       url="https://gossip.example/star",
                       publisher="gossip.example",
                       snippet="A completely unrelated story about celebrities")
    assert _is_relevant(on_topic, terms)
    assert not _is_relevant(off_topic, terms)


def test_discover_filters_irrelevant_feed_items(monkeypatch):
    """Feed items not matching the topic never become research candidates."""
    import app.agents.research_agent as ra

    agent = ResearchAgent(llm=FakeLLMClient())

    feed_xml = """
    <rss><channel>
      <item><title>Fusion energy hits new milestone</title>
        <link>https://power.example/fusion-milestone</link>
        <description>Compact fusion energy reactor achieves net gain</description></item>
      <item><title>Local bakery wins award</title>
        <link>https://bakery.example/croissant</link>
        <description>Artisan croissants judged best in the city</description></item>
    </channel></rss>
    """

    class FakeResp:
        status_code = 200
        text = feed_xml

        def raise_for_status(self):
            return None

    class FakeClient:
        async def get(self, url, timeout=None):
            return FakeResp()

    class NullCtx:
        async def __aenter__(self):
            return FakeClient()

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr(ra.httpx.AsyncClient, "__aenter__",
                        lambda self: self, raising=False)
    monkeypatch.setattr(ra, "_NullCtx", NullCtx)
    monkeypatch.setattr(ra.httpx, "AsyncClient",
                        lambda **kwargs: NullCtx())

    sources = asyncio_run_discover(agent, ["fusion energy"], ["fusion", "energy"])
    urls = [s.url for s in sources]
    assert "https://power.example/fusion-milestone" in urls
    assert all("bakery" not in u for u in urls)


def asyncio_run_discover(agent, queries, terms):
    import asyncio
    import app.agents.research_agent as ra

    return asyncio.run(agent._discover(queries, ra._Budget(), terms))


# -- Research-quality refinement: relevance/recency/quality assessment -------------


def _mk_verified(title: str, published: str | None, *, chars: int = 5000,
                 url: str = "https://example.com/a",
                 publisher: str = "example.com") -> Source:
    """A source that was successfully fetched and parsed (verified evidence)."""
    return Source(
        title=title, url=url, publisher=publisher, published=published,
        kind="article", evidence_status="verified",
        evidence_note=f"Full article text fetched and parsed ({chars} chars).",
        extracted_chars=chars,
    )


def test_old_fetched_source_downweighted_vs_newer_for_current_query():
    """Both sources are verified, but the old one must not lead a current-state
    synthesis merely because its content was fetched successfully."""
    import asyncio

    agent = ResearchAgent(llm=FakeLLMClient())
    task = "Research the current state of fusion energy"
    terms = ["fusion", "energy"]
    old = _mk_verified("Fusion baseline overview", "2022-03-01T00:00:00Z",
                       url="https://a.example/old")
    new = _mk_verified("Fusion milestone announced", "2026-09-10T00:00:00Z",
                       url="https://b.example/new")
    _assess_sources([old, new], task, terms)

    assert old.assessment["recency"] == "old"
    assert new.assessment["recency"] == "current"
    # Evidence definition untouched by the quality layer.
    assert old.evidence_status == "verified" and new.evidence_status == "verified"
    assert "old" in old.assessment["note"]

    synthesis = asyncio.run(agent._synthesize(task, ["fusion energy"], [old, new], [old, new]))
    user_msgs = [m for call in agent.llm.calls for m in call if m["role"] == "user"]
    evidence_block = user_msgs[-1]["content"]
    # The synthesis evidence lists the CURRENT source first, old source after.
    assert evidence_block.index("b.example/new") < evidence_block.index("a.example/old")


def test_irrelevant_but_successfully_fetched_source_is_flagged():
    """Fetching success still means 'verified', but the assessment layer must
    classify an off-topic fetched source as off-topic."""
    task = "Research the current state of fusion energy"
    terms = ["fusion", "energy"]
    off = _mk_verified("Celebrity gossip roundup", "2026-09-15T00:00:00Z",
                       url="https://g.example/star")
    _assess_sources([off], task, terms)
    assert off.evidence_status == "verified"           # evidence definition intact
    assert off.assessment["relevance"] == "off_topic"  # quality layer flags it


def test_current_query_with_mixed_source_dates_orders_and_describes_scope():
    task = "Research the current state of fusion energy"
    terms = ["fusion", "energy"]
    old = _mk_verified("Fusion history piece", "2022-03-01T00:00:00Z",
                       url="https://a.example/old")
    aging = _mk_verified("Fusion program profile", "2026-01-02T00:00:00Z",
                         url="https://c.example/aging")
    current = _mk_verified("Fusion milestone announced", "2026-09-10T00:00:00Z",
                           url="https://b.example/new")
    _assess_sources([old, aging, current], task, terms)

    assert [s.assessment["recency"] for s in (old, aging, current)] == \
        ["old", "aging", "current"]

    scope = _describe_evidence_scope(task, [old, aging, current])
    assert "3 verified article(s) from 3 publisher(s)" in scope
    assert "1 of 3 within the last ~45 days" in scope
    assert "not a complete technical assessment" in scope


def test_assessment_classifies_without_changing_evidence_status():
    """Verified source with unknown date + brief content: quality labels are
    attached, but the evidence status is never touched."""
    src = _mk_verified("Fusion energy report", None, chars=800)
    _assess_sources([src], "current fusion energy state", ["fusion", "energy"])
    assert src.evidence_status == "verified"
    a = src.assessment
    assert a["recency"] == "unknown_date"
    assert a["depth"] == "brief"
    assert a["relevance"] == "on_topic"
    assert "publication date unknown" in a["note"]


def test_zero_relevant_verified_sources_is_partial(monkeypatch):
    """Verified-but-irrelevant evidence must NOT be synthesized: explicit
    partial result, honest message, fallback synthesis, no invented findings."""
    agent = _make_agent(monkeypatch, RICH_ARTICLE, feed_xml="")
    # Fixture source ("Fixture Article", example.com) shares no terms with
    # this request, so the single verified source is off-topic.
    result = agent.run("Research the current state of quantum computing")

    assert result.status == "partial"
    assert "none were assessed" in result.message
    assert "topically relevant" in result.message
    assert "verified still means full article content" in result.message
    src = result.data["sources"][0]
    assert src["evidence_status"] == "verified"           # still verified evidence
    assert src["assessment"]["relevance"] == "off_topic"  # but flagged irrelevant
    assert "No verified article evidence topically relevant" in result.data["synthesis"]
    # No synthesis LLM call may present irrelevant content as findings.
    assert result.data["verified_evidence_count"] == 1
