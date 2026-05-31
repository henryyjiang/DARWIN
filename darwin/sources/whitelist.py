"""Network egress whitelist (ARCHITECTURE.md §8.3).

DARWIN agents have **no general-purpose web/fetch tool**: web access is mediated *only* by the
MCP `paper.*` and `data.*` tools, and every request those tools make passes through this
whitelist. Default-deny: a request to any host not explicitly allowed raises `EgressBlocked`.
This is what blocks illegal scraping and arbitrary egress from inside the (sandboxed) container
— the container network policy is the hard boundary (§8.3), and this is the in-process guard so
the retrieval clients can't be pointed at an arbitrary URL.

Allowed hosts (§8.3):
- **Papers:** arXiv, Semantic Scholar.
- **Datasets/models:** Hugging Face Hub (+ CDN).
(PyPI for installs and the Anthropic API are container-level egress, not agent retrieval tools,
so they are not part of *this* tool whitelist.)
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Hosts the retrieval tools may reach. Kept deliberately small (§8.3 default-deny).
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        # papers
        "export.arxiv.org",
        "arxiv.org",
        "api.semanticscholar.org",
        # datasets / models (HF Hub + its CDN)
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs.hf.co",
        "hf.co",
    }
)


class EgressBlocked(RuntimeError):
    """Raised when a retrieval client tries to reach a non-whitelisted host (§8.3)."""


def host_allowed(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    if host in ALLOWED_HOSTS:
        return True
    # allow the *.hf.co / *.huggingface.co CDN subdomains without enumerating every shard host
    return host.endswith(".hf.co") or host.endswith(".huggingface.co")


def check_url(url: str) -> None:
    """Raise EgressBlocked unless `url`'s host is on the whitelist (§8.3)."""
    if not host_allowed(url):
        host = urlsplit(url).hostname or "(none)"
        raise EgressBlocked(
            f"egress to {host!r} is not on the DARWIN whitelist (§8.3); "
            f"allowed hosts: {sorted(ALLOWED_HOSTS)}"
        )
