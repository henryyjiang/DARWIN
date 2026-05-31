"""Whitelisted external retrieval (ARCHITECTURE.md §8.3 / §8.4 / §9.3).

The backing subsystem for the MCP `paper.*` and `data.*` tools — the agent's *only* web access,
mediated through a default-deny egress whitelist (`whitelist.py`):

- `papers.py`   — arXiv retrieval (`PaperSource`); returns the canonical citation string for
  frictionless attribution (§8.4).
- `datasets.py` — Hugging Face Hub retrieval (`DataSource`); returns the dataset card + license
  string so provenance travels with the data (§8.3). No scraping tool exists by design.
- `transport.py`— the injectable, whitelist-gated HTTP transport (stdlib default; fakeable).

Parsing is pure and unit-tested offline; the network only happens in the default transport.
"""

from darwin.sources.datasets import DataSource, DatasetRef, parse_dataset, parse_search
from darwin.sources.papers import PaperRef, PaperSource, normalize_arxiv_id, parse_atom
from darwin.sources.transport import Transport, UrllibTransport, build_url
from darwin.sources.whitelist import ALLOWED_HOSTS, EgressBlocked, check_url, host_allowed

__all__ = [
    "PaperRef",
    "PaperSource",
    "parse_atom",
    "normalize_arxiv_id",
    "DatasetRef",
    "DataSource",
    "parse_dataset",
    "parse_search",
    "Transport",
    "UrllibTransport",
    "build_url",
    "ALLOWED_HOSTS",
    "EgressBlocked",
    "check_url",
    "host_allowed",
]
