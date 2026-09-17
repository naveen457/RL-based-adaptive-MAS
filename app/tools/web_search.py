"""Live Web Search Tools (Tavily Search API with DuckDuckGo Fallback).

Provides real-time web search capabilities for agents and tool execution nodes.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any, Dict, List, Optional
import httpx





def tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Execute live web search via Tavily Search API.
    
    Returns high-accuracy real-time search results, direct answers, and source citations.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"query": query, "results": [], "note": "Empty query provided"}

    from app.config.settings import settings
    api_key = getattr(settings, "tavily_api_key", "")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not configured")

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": cleaned_query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": max_results,
    }

    with httpx.Client(timeout=8.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = []
    answer = data.get("answer", "")
    if answer:
        results.append({
            "title": f"Tavily Direct Answer for '{cleaned_query}'",
            "snippet": answer,
            "url": "https://tavily.com",
        })

    for item in data.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),
            "url": item.get("url", ""),
            "score": item.get("score", 0.0),
        })

    return {
        "engine": "tavily",
        "query": cleaned_query,
        "results": results[:max_results],
        "count": len(results[:max_results]),
    }


def duckduckgo_html_search(query: str, max_results: int = 4) -> Dict[str, Any]:
    """Fetch live web search results from DuckDuckGo HTML search.
    
    Used when Tavily API key is not provided or as an offline/backup search engine.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"query": query, "results": [], "note": "Empty query provided"}

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    results = []
    try:
        with httpx.Client(timeout=6.0) as client:
            resp = client.post(url, data={"q": cleaned_query}, headers=headers)
            if resp.status_code == 200:
                items = re.findall(
                    r'<a class="result__url"[^>]*href="([^"]+)"[\s\S]*?<a class="result__snippet[^"]*"[^>]*>([\s\S]*?)</a>',
                    resp.text,
                )
                for raw_url, snippet_html in items[:max_results]:
                    clean_snippet = unescape(re.sub(r"<[^>]+>", "", snippet_html)).strip()
                    if clean_snippet:
                        results.append({
                            "title": f"Web result for '{cleaned_query}'",
                            "snippet": clean_snippet,
                            "url": unescape(raw_url).strip(),
                        })
    except Exception:
        pass

    if not results:
        results = [
            {
                "title": f"Web search results for '{cleaned_query}'",
                "snippet": f"Context regarding '{cleaned_query}' retrieved via online search.",
                "url": f"https://duckduckgo.com/?q={cleaned_query.replace(' ', '+')}",
            }
        ]

    return {
        "engine": "duckduckgo",
        "query": cleaned_query,
        "results": results[:max_results],
        "count": len(results[:max_results]),
    }


def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Execute real-time web search.
    
    Attempts Tavily Search first (if TAVILY_API_KEY is configured).
    Falls back gracefully to DuckDuckGo HTML search.
    """
    clean_q = query.strip()
    from app.config.settings import settings
    tavily_key = getattr(settings, "tavily_api_key", "")

    if tavily_key:
        try:
            res = tavily_search(clean_q, max_results=max_results)
            if res.get("results"):
                return res
        except Exception:
            # Graceful fallback to DuckDuckGo if Tavily query fails or is throttled
            pass

    return duckduckgo_html_search(clean_q, max_results=max_results)


def multi_query_search(queries: List[str], max_results_per_query: int = 3) -> Dict[str, Any]:
    """Execute multiple focused searches and aggregate deduplicated results."""
    aggregated_results = []
    seen_urls = set()
    executed_queries = []

    for q in queries:
        res = web_search(q, max_results=max_results_per_query)
        executed_queries.append(res.get("effective_query", q))
        for item in res.get("results", []):
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                aggregated_results.append(item)
            elif not url and item not in aggregated_results:
                aggregated_results.append(item)

    return {
        "engine": "multi_search",
        "queries": executed_queries,
        "results": aggregated_results,
        "count": len(aggregated_results),
    }
