"""Whitelisted paper retrieval (ARCHITECTURE.md §8.4 / §9.3 `paper.*`).

`paper_search(query)` / `paper_fetch(arxiv_id)` over the arXiv API (a whitelisted host, §8.3).
Each result carries the **canonical citation string** so the agent can record attribution
frictionlessly in both `thesis.md` and the memory file's `papers_cited` (§8.4) — the audit in
`observability/attribution.py` later checks that this was done.

The arXiv API returns an Atom feed; parsing is pure (`parse_atom`) and unit-tested against a
canned response, so no network is needed in tests. The `Transport` (default `UrllibTransport`)
is injected and whitelist-gated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from darwin.sources.transport import Transport, UrllibTransport

_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


@dataclass
class PaperRef:
    """A retrieved paper + its canonical citation (§8.4)."""

    arxiv_id: str  # bare id, e.g. "2401.01234" (version stripped)
    title: str
    authors: list[str] = field(default_factory=list)
    summary: str = ""
    published: str = ""
    url: str = ""

    @property
    def year(self) -> str:
        return self.published[:4] if self.published else ""

    def citation(self) -> str:
        """A canonical citation string to record in `papers_cited` / inline (§8.4)."""
        who = ", ".join(self.authors) if self.authors else "Unknown"
        year = f" ({self.year})" if self.year else ""
        return f"{who}. {self.title}. arXiv:{self.arxiv_id}{year}."

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": list(self.authors),
            "summary": self.summary,
            "published": self.published,
            "year": self.year,
            "url": self.url,
            "citation": self.citation(),
        }


def normalize_arxiv_id(raw: str) -> str:
    """Extract the bare arXiv id (strip a version suffix / abs URL), or return raw stripped."""
    m = _ARXIV_ID_RE.search(raw)
    return m.group(1) if m else raw.strip()


def _text(entry: ET.Element, tag: str) -> str:
    el = entry.find(f"{_ATOM}{tag}")
    return (el.text or "").strip() if el is not None and el.text else ""


def parse_atom(xml: str) -> list[PaperRef]:
    """Parse an arXiv Atom feed into PaperRefs (pure; the network-free core)."""
    root = ET.fromstring(xml)
    refs: list[PaperRef] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = _text(entry, "id")
        authors = [
            (a.find(f"{_ATOM}name").text or "").strip()
            for a in entry.findall(f"{_ATOM}author")
            if a.find(f"{_ATOM}name") is not None
        ]
        # the <link rel="alternate"> abs page, falling back to the id URL
        url = raw_id
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("rel") == "alternate":
                url = link.get("href") or url
        refs.append(
            PaperRef(
                arxiv_id=normalize_arxiv_id(raw_id),
                title=" ".join(_text(entry, "title").split()),
                authors=[a for a in authors if a],
                summary=" ".join(_text(entry, "summary").split()),
                published=_text(entry, "published"),
                url=url,
            )
        )
    return refs


class PaperSource:
    """Whitelisted paper retrieval over the arXiv API (§9.3 `paper.*`)."""

    def __init__(self, transport: Transport | None = None):
        self.transport = transport or UrllibTransport()

    def search(self, query: str, *, limit: int = 5) -> list[PaperRef]:
        if not query.strip():
            return []
        xml = self.transport.get_text(
            _ARXIV_API,
            {
                "search_query": f"all:{query}",
                "start": "0",
                "max_results": str(max(1, limit)),
            },
        )
        return parse_atom(xml)

    def fetch(self, arxiv_id: str) -> PaperRef | None:
        bare = normalize_arxiv_id(arxiv_id)
        xml = self.transport.get_text(_ARXIV_API, {"id_list": bare})
        refs = parse_atom(xml)
        return refs[0] if refs else None
