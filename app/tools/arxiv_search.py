"""ArXiv scientific paper search tool.

Provides querying and retrieval of research papers, preprints, authors,
abstracts, and citations directly from the official arXiv API without third-party API keys.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_API_URL = "https://export.arxiv.org/api/query"


def clean_text(text: Optional[str]) -> str:
    """Clean whitespace and newlines from XML text."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def arxiv_search(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> Dict[str, Any]:
    """Search arXiv for research papers, preprints, and scientific literature.

    Args:
        query: Search keywords, author, topic, or title.
        max_results: Maximum number of papers to retrieve (default: 5).
        sort_by: Sorting field ('relevance', 'lastUpdatedDate', 'submittedDate').
        sort_order: 'descending' or 'ascending'.

    Returns:
        Dict containing query, count, list of paper dicts, and a clean markdown summary.
    """
    clean_q = query.strip()
    if not clean_q:
        return {
            "query": query,
            "total_results": 0,
            "papers": [],
            "summary": "No search query provided.",
        }

    # Format search query for arXiv: 'all:terms'
    if not any(clean_q.startswith(p) for p in ["all:", "ti:", "au:", "abs:"]):
        search_query = f"all:{clean_q}"
    else:
        search_query = clean_q

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max(1, min(max_results, 25)),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(ARXIV_API_URL, params=params)
            resp.raise_for_status()
            xml_content = resp.text
    except Exception as e:
        return {
            "query": query,
            "total_results": 0,
            "papers": [],
            "error": f"arXiv API request failed: {type(e).__name__}: {e}",
            "summary": f"Could not retrieve arXiv papers: {e}",
        }

    papers: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_content)
        entries = root.findall(f"{ATOM_NS}entry")

        for entry in entries:
            title_elem = entry.find(f"{ATOM_NS}title")
            title = clean_text(title_elem.text if title_elem is not None else "")

            # If entry is an error/empty indicator
            if not title or title.lower() == "error":
                continue

            id_elem = entry.find(f"{ATOM_NS}id")
            entry_id = clean_text(id_elem.text if id_elem is not None else "")
            arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id

            summary_elem = entry.find(f"{ATOM_NS}summary")
            abstract = clean_text(summary_elem.text if summary_elem is not None else "")

            published_elem = entry.find(f"{ATOM_NS}published")
            published = clean_text(published_elem.text if published_elem is not None else "")[:10]

            authors: List[str] = []
            for author_elem in entry.findall(f"{ATOM_NS}author"):
                name_elem = author_elem.find(f"{ATOM_NS}name")
                if name_elem is not None and name_elem.text:
                    authors.append(clean_text(name_elem.text))

            pdf_url = ""
            for link_elem in entry.findall(f"{ATOM_NS}link"):
                if link_elem.attrib.get("title") == "pdf" or link_elem.attrib.get("type") == "application/pdf":
                    pdf_url = link_elem.attrib.get("href", "")
                    break

            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            paper_data = {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "summary": abstract,
                "entry_url": entry_id,
                "pdf_url": pdf_url,
            }
            papers.append(paper_data)

    except Exception as parse_err:
        return {
            "query": query,
            "total_results": 0,
            "papers": [],
            "error": f"Failed to parse arXiv response: {parse_err}",
            "summary": f"Error parsing arXiv feed: {parse_err}",
        }

    # Build markdown summary for downstream synthesis
    summary_lines = [f"### arXiv Research Papers for '{clean_q}' ({len(papers)} results):"]
    for idx, p in enumerate(papers, 1):
        authors_str = ", ".join(p["authors"][:4])
        if len(p["authors"]) > 4:
            authors_str += " et al."
        summary_lines.append(f"\n{idx}. **{p['title']}** ({p['published']})")
        summary_lines.append(f"   - Authors: {authors_str}")
        summary_lines.append(f"   - arXiv ID: [{p['arxiv_id']}]({p['entry_url']}) | [PDF]({p['pdf_url']})")
        abstract_preview = p["summary"][:350] + ("..." if len(p["summary"]) > 350 else "")
        summary_lines.append(f"   - Abstract: {abstract_preview}")

    return {
        "query": query,
        "total_results": len(papers),
        "papers": papers,
        "summary": "\n".join(summary_lines),
    }
