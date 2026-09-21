"""Research Agent — bounded research pipeline.

Pipeline: query generation (Freebuff) -> relevance filtering -> bounded source
discovery (topical RSS/search feeds) -> article fetch + extraction -> evidence
validation -> relevance/recency/depth ASSESSMENT -> grounded synthesis (Freebuff).
Search snippets and RSS metadata are NEVER presented as verified article
evidence; only fetched article content can become 'verified' evidence. The
assessment layer is a separate quality classification of verified sources —
it never changes evidence status.

Networking is strictly bounded: per-request timeout, capped retries, limited
source count, and a wall-clock budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from app.agents.base import BaseAgent
from app.core.results import AgentResult
from app.llm.client import LLMConfigError, LLMError, LLMMessage

if TYPE_CHECKING:  # pragma: no cover
    from app.core.files import FileContext

logger = logging.getLogger("orbit.research")

MAX_QUERIES = 3
MAX_SOURCES = 6
MAX_CONCURRENCY = 3
FETCH_TIMEOUT_SECONDS = 10.0
MAX_FETCH_RETRIES = 1
RETRY_BACKOFF_SECONDS = 1.0
WALL_CLOCK_BUDGET_SECONDS = 55.0
MIN_ARTICLE_CHARS = 400
MAX_ARTICLE_CHARS = 12_000
MAX_SNIPPET_CHARS = 300

# Relevance filtering: a discovery candidate must mention the topic's
# distinctive terms before it is treated as a research candidate at all.
RELEVANCE_MIN_HITS = 1

# Recency tiers for the assessment layer (in DAYS): sources newer than this
# count as current for a "current/latest" request; older ones are judged by
# content, never by date alone.
CURRENT_REQUEST_WINDOW_DAYS = 45
CURRENT_REQUEST_WARN_DAYS = 2 * 365
CURRENT_REQUEST_WINDOW_SECONDS = CURRENT_REQUEST_WINDOW_DAYS * 24 * 3600

# Wording that marks a request as asking for current/latest information.
_CURRENT_INTENT_RE = re.compile(
    r"\b(current|latest|now|today|recent|newest|up.to.date|state of)\b", re.IGNORECASE
)

# Browser-like client identity. Feed/article servers aggressively reject
# bot-style agents (some return HTTP 405/403 for them), so the pipeline
# presents a standard browser User-Agent like the desktop app does.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Topical search feeds. Every feed MUST actually filter by the query:
# the previous secondary feed ignored its query parameter entirely and
# returned the same unrelated latest-headlines for every topic.
FEED_BING_NEWS = "https://www.bing.com/news/search?q={query}&format=RSS"
FEED_GOOGLE_NEWS = "https://news.google.com/rss/search?q={query}"
SEARCH_FEEDS = (FEED_BING_NEWS, FEED_GOOGLE_NEWS)

# Aggregator/redirect hosts are never fetched as article sources; their links
# are resolved to the underlying publisher URL first (best effort).
SKIP_HOSTS = ("google.com", "news.google.com", "bing.com", "duckduckgo.com", "microsoft.com")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Stopwords excluded when deriving relevance terms from the user request.
_STOPWORDS = frozenset(
    "a an the and or of to in on for with about from as by at into over after "
    "is are was were be been being this that these those it its their there "
    "current currently latest recent state status news research sources source "
    "cited cite references please give show tell me what why how when who "
    "today now present up-to-date upto-date modern new newest".split()
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class Source:
    title: str
    url: str
    publisher: str
    published: str | None = None
    snippet: str = ""
    kind: str = "search_result"            # search_result | article
    evidence_status: str = "unverified"    # unverified | verified | rejected
    evidence_note: str = ""
    extracted_chars: int = 0
    error: str | None = None
    # Source-quality assessment (separate layer; NEVER changes evidence_status).
    assessment: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "publisher": self.publisher,
            "published": self.published,
            "snippet": self.snippet[:MAX_SNIPPET_CHARS],
            "kind": self.kind,
            "evidence_status": self.evidence_status,
            "evidence_note": self.evidence_note,
            "extracted_chars": self.extracted_chars,
            "error": self.error,
            "assessment": self.assessment,
        }


@dataclass
class _Budget:
    deadline: float = field(default_factory=lambda: time.monotonic() + WALL_CLOCK_BUDGET_SECONDS)

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0


def _clean(text: str | None) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub("", text or "")).strip()


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _publisher_from_url(url: str) -> str:
    host = _host(url)
    return host or "unknown"


def _derive_relevance_terms(text: str) -> list[str]:
    """Distinctive terms from the user's request, for relevance gating."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", (text or "").lower())
    seen: list[str] = []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            seen.append(w)
    return seen


def _query_is_on_topic(query: str, relevance_terms: list[str]) -> bool:
    if not relevance_terms:
        return True
    q = query.lower()
    return any(t in q for t in relevance_terms)


def _text_relevance_hits(text: str, relevance_terms: list[str]) -> int:
    """How many distinctive topic terms appear in the text."""
    if not relevance_terms:
        return 0
    t = text.lower()
    return sum(1 for term in relevance_terms if term in t)


def _decode_feed_link(link: str) -> str:
    """Resolve aggregator feed links to the underlying publisher URL.

    Bing News wraps targets in apiclick.aspx?...url=<encoded>; Google News
    uses redirect pages. Anything else is returned unchanged.
    """
    link = (link or "").strip()
    if "bing.com" in link and "url=" in link:
        match = re.search(r"[?&]url=(https?%[0-9a-fA-F]{2}.+?)(?:&|$)", link)
        if match:
            return unquote(match.group(1))
        match = re.search(r"[?&]url=(https?://[^&]+)", link)
        if match:
            return unquote(match.group(1))
    return link


# Hosts/IPs that must never be fetched as article sources (SSRF guard):
# loopback, link-local, private, and reserved ranges.
_PRIVATE_HOST_RE = re.compile(
    r"^(localhost|.*\.localhost|.*\.local|metadata\.google\.internal)$"
    r"|^(127\.|10\.|192\.168\.|169\.254\.|0\.|22[4-9]\.|2[3-5][0-9]\.)"
)


def _url_allowed(url: str) -> bool:
    """Only public https:// pages may be fetched as evidence sources."""
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    if parts.scheme != "https" or not parts.hostname:
        return False
    host = parts.hostname.lower().rstrip(".")
    host = host.removeprefix("www.")
    if _PRIVATE_HOST_RE.search(host):
        return False
    return True


def _is_relevant(source: "Source", relevance_terms: list[str]) -> bool:
    """Reject obviously off-topic discovery results before fetching."""
    if not relevance_terms:
        return True
    hits = _text_relevance_hits(
        f"{source.title} {source.snippet} {_host(source.url)}", relevance_terms
    )
    return hits >= RELEVANCE_MIN_HITS


# -- stage 3.5: source-quality assessment (separate from evidence status) ----------


def _parse_source_date(published: str | None) -> datetime | None:
    """Best-effort date parse: RFC-2822 (RSS pubDate) or ISO-8601."""
    if not published:
        return None
    try:
        return parsedate_to_datetime(published)
    except (TypeError, ValueError, IndexError):
        pass
    try:
        dt = datetime.fromisoformat((published or "").replace("Z", "+00:00").strip())
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _request_wants_current(task: str) -> bool:
    """Does the request ask for current/latest information?"""
    return bool(_CURRENT_INTENT_RE.search(task or ""))


def _assess_source(source: "Source", relevance_terms: list[str]) -> dict[str, Any]:
    """Grade a VERIFIED source for synthesis ordering — never re-verify it.

    Dimensions: topical relevance, recency relative to the request's temporal
    intent, and content depth. This layer NEVER changes evidence_status:
    'verified' continues to mean full article content was fetched and parsed.
    """
    if not relevance_terms:
        relevance, hits = "on_topic", 0
    else:
        hits = _text_relevance_hits(
            f"{source.title} {source.snippet} {_host(source.url)}", relevance_terms
        )
        relevance = "on_topic" if hits >= RELEVANCE_MIN_HITS else "off_topic"

    published_dt = _parse_source_date(source.published)
    recency, age_days = "unknown_date", None
    if published_dt is not None:
        age_days = max(0, int((datetime.now(timezone.utc) - published_dt).total_seconds() // 86400))
        if age_days <= CURRENT_REQUEST_WINDOW_DAYS:
            recency = "current"
        elif age_days <= CURRENT_REQUEST_WARN_DAYS:
            recency = "aging"
        else:
            recency = "old"

    depth = "substantial" if source.extracted_chars >= 2000 else "brief"
    if recency == "old":
        recency_note = "old — down-weighted for current-state conclusions"
    elif recency == "aging":
        recency_note = "not recent"
    elif recency == "current":
        recency_note = "recent"
    else:
        recency_note = "publication date unknown"
    note = (
        f"{'Topical' if relevance == 'on_topic' else 'Off-topic'}; {recency_note}; "
        f"{depth} coverage"
    )
    return {
        "relevance": relevance,
        "relevance_hits": hits,
        "recency": recency,
        "age_days": age_days,
        "depth": depth,
        "note": note,
    }


_RECENCY_ORDER = {"current": 0, "aging": 1, "unknown_date": 2, "old": 3}


def _assess_sources(sources: list["Source"], task: str,
                    relevance_terms: list[str]) -> list["Source"]:
    """Assess all verified sources; for current-state requests, order them so
    recent, on-topic, substantial sources lead. Old-but-relevant sources stay
    available as evidence — they are simply not allowed to dominate."""
    for s in sources:
        if s.evidence_status == "verified":
            s.assessment = _assess_source(s, relevance_terms)
    if _request_wants_current(task):
        def rank(s: "Source") -> tuple[int, int, int]:
            if s.evidence_status != "verified":
                return (2, 0, 0)  # rejected/unverified rows last
            a = s.assessment or {}
            return (
                0 if a.get("relevance") == "on_topic" else 1,
                _RECENCY_ORDER.get(a.get("recency", "unknown_date"), 2),
                -(s.extracted_chars or 0),
            )
        sources.sort(key=rank)
    return sources


def _describe_evidence_scope(task: str, verified: list["Source"]) -> str:
    """Factual description of what the evidence base covers — so the synthesis
    can state its scope instead of implying a complete field assessment."""
    if not verified:
        return "No verified article evidence was retrieved."
    dated = [d for s in verified if (d := _parse_source_date(s.published)) is not None]
    now = datetime.now(timezone.utc)
    recent = sum(1 for d in dated if (now - d).total_seconds() <= CURRENT_REQUEST_WINDOW_SECONDS)
    publishers = sorted({_host(s.url) or s.publisher for s in verified})
    if dated:
        span = f"{min(dated):%b %Y} to {max(dated):%b %Y}"
    else:
        span = "publication dates unknown"
    parts = [
        f"{len(verified)} verified article(s) from {len(publishers)} publisher(s)",
        f"publication dates: {span}",
    ]
    if _request_wants_current(task):
        parts.append(f"{recent} of {len(verified)} within the last ~45 days")
    parts.append(
        "predominantly recent news coverage, not a complete technical assessment "
        "of the entire field"
    )
    return "; ".join(parts)


class ResearchAgent(BaseAgent):
    name = "research"

    def __init__(self, *args: Any, http_client: httpx.AsyncClient | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._http_client = http_client

    # -- entry point -------------------------------------------------------------
    def _run(self, task: str, file_context: "FileContext | None" = None,
             history: list[dict] | None = None) -> AgentResult:
        return asyncio.run(self._run_async(task))

    async def _run_async(self, task: str) -> AgentResult:
        budget = _Budget()
        try:
            queries, relevance_terms = await self._generate_queries(task)
        except (LLMConfigError, LLMError) as exc:
            return AgentResult.create(
                self.name, "failed",
                f"Freebuff LLM call failed while generating search queries: {exc}",
                metadata={"stage": "query_generation"},
            )
        queries = queries[:MAX_QUERIES]
        if not relevance_terms:
            relevance_terms = _derive_relevance_terms(task)

        discoveries = await self._discover(queries, budget, relevance_terms)
        articles = await self._fetch_and_validate(discoveries, budget)
        # Stage 3.5 — source-quality assessment of VERIFIED sources only.
        # Relevance/recency/depth classification; evidence statuses are final
        # after validation and are never changed here.
        articles = _assess_sources(articles, task, relevance_terms)
        verified = [a for a in articles if a.evidence_status == "verified"]

        try:
            synthesis = await self._synthesize(task, queries, articles, verified)
        except (LLMConfigError, LLMError) as exc:
            return AgentResult.create(
                self.name, "partial",
                f"Retrieved sources but Freebuff synthesis failed: {exc}",
                data=self._data_payload(task, queries, articles, verified, synthesis=None),
                metadata={"stage": "synthesis"},
            )

        on_topic_verified = [
            s for s in verified
            if (s.assessment or {}).get("relevance", "on_topic") == "on_topic"
        ]
        status = "completed" if on_topic_verified else "partial"
        if on_topic_verified:
            message = (
                f"Synthesized findings from {len(on_topic_verified)} relevant verified "
                f"source(s) (of {len(verified)} verified / {len(articles)} fetched)."
            )
        elif verified:
            message = (
                f"Retrieved {len(verified)} verified source(s), but none were assessed "
                "as topically relevant to the request — no synthesis was produced. "
                "Evidence definition unchanged: verified still means full article "
                "content was fetched and parsed."
            )
        else:
            # Be explicit that this is about THIS attempt, not a claim that no
            # evidence exists anywhere; discovery metadata stays clearly labeled.
            message = (
                "No verified evidence could be retrieved during this attempt "
                f"({len(articles)} on-topic candidate(s) found; article fetch or "
                "parsing failed for all of them). Listed rows are discovery "
                "metadata only — NOT evidence. Try re-running: fetch "
                "availability varies by source."
            )
        return AgentResult.create(
            self.name, status, message,
            data=self._data_payload(task, queries, articles, verified, synthesis),
            metadata={"queries": queries, "verified_count": len(verified),
                      "fetched_count": len(articles),
                      "evidence_attempt": "this_attempt"},
        )

    # -- stage 1: queries ----------------------------------------------------------
    async def _generate_queries(self, task: str) -> tuple[list[str], list[str]]:
        """Generate search queries that preserve the user's topic and intent.

        Returns (queries, relevance_terms). The LLM is explicitly told to keep
        temporal words like "current" as-is (no inventing a year cutoff), and
        the returned queries are validated: each must contain at least one of
        the request's distinctive terms, otherwise the user's own request (or
        a derived topic query) is used instead.
        """
        response = await self.llm.chat(
            [
                LLMMessage(role="system", content=(
                    "Generate 2-3 diverse web-search queries for the user's "
                    "research request. RULES: keep the user's actual topic in "
                    "every query; keep temporal intent exactly (words like "
                    "'current', 'latest', 'now' stay as-is — never replace them "
                    "with a specific year, date, or cutoff); do not narrow the "
                    "topic. Prefer queries likely to match news/RSS feeds. Reply "
                    "with ONLY a JSON array of strings, no commentary."
                )),
                LLMMessage(role="user", content=task),
            ],
            temperature=0.2,
        )
        queries = self._parse_queries(response.content, fallback=task)
        relevance_terms = _derive_relevance_terms(task)

        # Relevance gate: a generated query that dropped the topic entirely
        # (e.g. generic "latest news") is replaced by the user's own request.
        checked: list[str] = []
        for q in queries:
            if _query_is_on_topic(q, relevance_terms):
                checked.append(q)
            else:
                logger.info("Discarding off-topic generated query: %r", q)
        if not checked:
            checked = [task]
            derived = " ".join(relevance_terms[:6])
            if derived and derived.lower() not in task.lower():
                checked.append(derived)
        return checked, relevance_terms

    def _parse_queries(self, content: str, *, fallback: str) -> list[str]:
        # Models may wrap JSON in markdown fences — strip any prose around it.
        content = re.sub(r"^.*?```(?:json)?", "", content, flags=re.DOTALL).strip()
        content = content.split("```", 1)[0].strip() or content.strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()][:MAX_QUERIES]
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(q).strip() for q in parsed if str(q).strip()][:MAX_QUERIES]
            except json.JSONDecodeError:
                pass
        return [fallback]

    # -- stage 2: discovery ----------------------------------------------------------
    async def _discover(self, queries: list[str], budget: _Budget,
                        relevance_terms: list[str] | None = None) -> list[Source]:
        seen: dict[str, Source] = {}
        relevance_terms = relevance_terms or []

        async def worker(client: httpx.AsyncClient, feed_url: str) -> None:
            try:
                resp = await client.get(feed_url, timeout=FETCH_TIMEOUT_SECONDS)
                resp.raise_for_status()
            except (httpx.HTTPError, httpx.StreamError) as exc:
                logger.info("Discovery feed failed (%s): %s", feed_url, exc)
                return
            for item in _parse_feed_items(resp.text):
                url = _decode_feed_link(item.get("link", ""))
                if not url or _host(url) in SKIP_HOSTS or url in seen:
                    continue
                source = Source(
                    title=item.get("title") or "(untitled)",
                    url=url,
                    publisher=item.get("source") or _publisher_from_url(url),
                    published=item.get("pubDate"),
                    snippet=_clean(item.get("description")),
                    kind="search_result",
                )
                # Relevance gate: reject obviously irrelevant results before
                # they ever become research candidates.
                if relevance_terms and not _is_relevant(source, relevance_terms):
                    logger.info("Rejected irrelevant discovery result: %s",
                                source.url[:100])
                    continue
                seen[url] = source

        feeds = [f.format(query=_url_quote(q)) for q in queries for f in SEARCH_FEEDS]
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        client_ctx = (
            _NullCtx(self._http_client) if self._http_client is not None
            else httpx.AsyncClient(
                headers=BROWSER_HEADERS,
                follow_redirects=True,
                timeout=FETCH_TIMEOUT_SECONDS,
            )
        )
        async with client_ctx as client:
            async def guarded(feed: str) -> None:
                if budget.expired() or len(seen) >= MAX_SOURCES * 3:
                    return
                async with semaphore:
                    await worker(client, feed)

            await asyncio.gather(*(guarded(f) for f in feeds))

        ordered = sorted(seen.values(), key=lambda s: s.published or "", reverse=True)
        return ordered[:MAX_SOURCES]

    # -- stage 3: fetch + validate ----------------------------------------------------
    async def _fetch_and_validate(self, sources: list[Source], budget: _Budget) -> list[Source]:
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def fetch_one(client: httpx.AsyncClient, source: Source) -> None:
            if not _url_allowed(source.url):
                source.error = "URL blocked (not a public HTTPS page)"
                return
            for attempt in range(MAX_FETCH_RETRIES + 1):
                if budget.expired():
                    source.error = source.error or "wall-clock budget exhausted"
                    return
                try:
                    resp = await client.get(source.url, timeout=FETCH_TIMEOUT_SECONDS)
                    if not _url_allowed(str(resp.url)):
                        source.error = "URL blocked (redirected to a non-public host)"
                        return
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_FETCH_RETRIES:
                        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                        continue
                    resp.raise_for_status()
                    text = _extract_article_text(resp.text)
                    source.extracted_chars = len(text)
                    if len(text) >= MIN_ARTICLE_CHARS:
                        source.kind = "article"
                        source.evidence_status = "verified"
                        source.evidence_note = (
                            f"Full article text fetched and parsed "
                            f"({len(text)} chars)."
                        )
                        if not source.published:
                            source.published = _extract_date(resp.text) or source.published
                    else:
                        source.evidence_status = "rejected"
                        source.evidence_note = (
                            f"Only {len(text)} chars of extractable content; "
                            "insufficient for evidence."
                        )
                    return
                except (httpx.HTTPError, httpx.StreamError) as exc:
                    if attempt < MAX_FETCH_RETRIES:
                        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                        continue
                    source.error = f"{type(exc).__name__}: {exc}"
                    source.evidence_status = "rejected"
                    source.evidence_note = "Fetch failed; metadata only."
                    return

        async with httpx.AsyncClient(
            headers=BROWSER_HEADERS, follow_redirects=True, timeout=FETCH_TIMEOUT_SECONDS,
        ) as client:
            async def guarded(source: Source) -> None:
                async with semaphore:
                    await fetch_one(client, source)

            await asyncio.gather(*(guarded(s) for s in sources))
        return sources

    # -- stage 4: grounded synthesis ----------------------------------------------------
    async def _synthesize(self, task: str, queries: list[str], all_sources: list[Source],
                          verified: list[Source]) -> str:
        # Synthesis consumes only RELEVANT verified sources, most current first
        # (assessment ordering). Old-but-fetched sources are not given equal
        # weight for a current-state conclusion; off-topic fetched sources are
        # excluded entirely from the evidence the LLM sees.
        relevant = [s for s in verified
                    if (s.assessment or {}).get("relevance", "on_topic") == "on_topic"]
        ordered = relevant
        if _request_wants_current(task):
            ordered = sorted(
                relevant,
                key=lambda s: (
                    _RECENCY_ORDER.get((s.assessment or {}).get("recency", "unknown_date"), 2),
                    -(s.extracted_chars or 0),
                ),
            )
        if not ordered:
            return (
                "No verified article evidence topically relevant to this request "
                "could be retrieved during this attempt, so no findings are "
                "reported. Listed rows are discovery metadata only — NOT evidence."
            )

        evidence_blocks: list[str] = []
        for i, s in enumerate(ordered, 1):
            a = s.assessment or {}
            evidence_blocks.append(
                f"[{i}] {s.title} — {s.publisher} — {s.published or 'date unknown'}"
                f" | quality: {a.get('note', 'n/a')}\n"
                f"URL: {s.url}"
            )
        evidence_text = "\n\n".join(evidence_blocks)
        scope = _describe_evidence_scope(task, ordered)

        response = await self.llm.chat(
            [
                LLMMessage(role="system", content=(
                    "You are the Research Agent inside ORBIT. Synthesize an answer "
                    "to the user's research request using ONLY the numbered verified "
                    "sources provided. RULES: cite sources as [1], [2]...; substantive "
                    "findings must come only from those sources; preserve the user's "
                    "temporal intent (if they asked for the current state, write about "
                    "the current state — NEVER substitute an invented cutoff like a "
                    "past year); each source's quality line describes its recency and "
                    "topical fit — weight the MOST CURRENT, most relevant sources "
                    "highest and do not present an old source as current; begin with a "
                    "one-line 'Evidence scope:' note describing what the evidence base "
                    "covers (it is predominantly recent news coverage, not a complete "
                    "technical assessment of the entire field); if the evidence is "
                    "insufficient, say exactly what is missing — never invent facts or "
                    "treat search snippets as evidence. Markdown output."
                )),
                LLMMessage(role="user", content=(
                    f"Research request: {task}\n\nSearch queries used: "
                    f"{json.dumps(queries)}\n\nEvidence scope: {scope}\n\n"
                    f"Verified evidence (assessed, most current first):\n{evidence_text}"
                )),
            ],
            temperature=0.3,
        )
        return response.content

    # -- payload ---------------------------------------------------------------------
    def _data_payload(self, task: str, queries: list[str], articles: list[Source],
                      verified: list[Source], synthesis: str | None) -> dict[str, Any]:
        return {
            "format": "research_report",
            "request": task,
            "queries": queries,
            "synthesis": synthesis,
            "verified_evidence_count": len(verified),
            "evidence_scope": _describe_evidence_scope(task, verified),
            "sources": [s.to_dict() for s in articles],
            "evidence_disclaimer": (
                "Only sources marked verified were fetched and parsed as full article "
                "text. Other rows are discovery metadata (search/RSS) and are NOT "
                "treated as evidence. Quality/assessment labels are a separate layer "
                "and never change evidence status."
            ),
        }


# -- feed / article parsing helpers ----------------------------------------------------


def _parse_feed_items(xml: str) -> list[dict[str, str]]:
    """Minimal RSS/Atom item extraction without extra dependencies."""
    items: list[dict[str, str]] = []
    for block in re.findall(r"<item>(.*?)</item>", xml, re.DOTALL | re.IGNORECASE):
        item = {
            "title": _clean(_first_tag(block, "title")),
            "link": (_first_tag(block, "link") or "").strip(),
            "pubDate": _clean(_first_tag(block, "pubDate") or _first_tag(block, "updated")),
            "description": _first_tag(block, "description"),
            "source": _clean(_first_tag(block, "source")),
        }
        if item["link"]:
            items.append(item)
    return items


def _first_tag(block: str, tag: str) -> str | None:
    import html as _html

    match = re.search(
        rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", block, re.DOTALL | re.IGNORECASE
    )
    if not match:
        return None
    raw = _html.unescape(match.group(1))
    if "<" in raw:  # embedded HTML (e.g. description) — strip tags
        return _clean(raw)
    return raw.strip()


def _extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    article = soup.find("article") or soup.find("main") or soup.body or soup
    headings = [_clean(h.get_text(" ")) for h in article.find_all(["h1", "h2", "h3"])]
    paragraphs = [_clean(p.get_text(" ")) for p in article.find_all(["p", "li"])]
    parts = [h for h in headings if h] + [p for p in paragraphs if len(p) > 40]
    text = "\n\n".join(parts)
    if len(text) < MIN_ARTICLE_CHARS:
        # Sparse markup: fall back to the container's visible text.
        text = _clean(article.get_text(" "))
    return text[:MAX_ARTICLE_CHARS]


def _extract_date(html: str) -> str | None:
    match = re.search(
        r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|og:published_time|'
        r'date)["\'][^>]+content=["\']([^"\']+)', html, re.IGNORECASE
    )
    return match.group(1) if match else None


def _url_quote(query: str) -> str:
    from urllib.parse import quote_plus

    return quote_plus(query)


class _NullCtx:
    """Allows `async with X if cond else Y` symmetry for an existing client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._client

    async def __aexit__(self, *exc: Any) -> None:
        return None
