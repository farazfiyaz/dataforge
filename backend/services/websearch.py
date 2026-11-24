# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Web search for the local model (opt-in, Model settings → "Allow web search").

Uses DuckDuckGo's plain-HTML results page: no API key, no account. Only the
search query leaves the machine, never the user's data. (In cloud mode Claude's
own web search tool is used instead; see services/claude.py.)
"""

import html
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

SEARCH_URL = "https://html.duckduckgo.com/html/"
MAX_RESULTS = 5

_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAGS = re.compile(r"<[^>]+>")

TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current or general information (facts, definitions, recent "
            "events, documentation). Returns titles, links and snippets. Use it for questions "
            "you can't answer reliably from memory; mention the links you used in your answer. "
            "Never put the user's data in the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for."}},
            "required": ["query"],
        },
    },
}


class WebSearchError(RuntimeError):
    """A search failure, worded for the user."""


async def search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 (DataForge; +local app)"}) as client:
            resp = await client.post(SEARCH_URL, data={"q": query})
        resp.raise_for_status()
    except httpx.HTTPError:
        raise WebSearchError("Web search isn't reachable right now (no internet connection?).") from None
    return parse_results(resp.text)[:max_results]


def parse_results(page: str) -> list[dict]:
    results = []
    for m in _RESULT.finditer(page):
        url = _real_url(html.unescape(m["href"]))
        if not url:
            continue
        results.append({
            "title": _text(m["title"]),
            "url": url,
            "snippet": _text(m["snippet"]),
        })
    return results


def _real_url(href: str) -> str | None:
    """DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<url>; unwrap them, skip ads."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc:
        if parsed.path.startswith("/y.js"):          # sponsored result
            return None
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        return unquote(target) if target else None
    return href if parsed.scheme in ("http", "https") else None


def _text(fragment: str) -> str:
    return html.unescape(_TAGS.sub("", fragment)).strip()
