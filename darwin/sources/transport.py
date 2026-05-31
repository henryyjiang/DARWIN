"""HTTP transport for the retrieval tools (ARCHITECTURE.md §8.3).

A tiny `Transport` interface (`get_text`) so the paper/dataset clients are decoupled from the
network and unit-testable offline with a fake. The default `UrllibTransport` uses the stdlib
(no new dependency) and routes every request through the §8.3 egress whitelist (`check_url`),
so the whitelist is enforced centrally regardless of which client builds the URL.
"""

from __future__ import annotations

from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from darwin.sources.whitelist import check_url

DEFAULT_TIMEOUT_S = 20.0
_USER_AGENT = "darwin-mcp/2.0 (+https://github.com/henryyjiang/DARWIN)"


def build_url(base: str, params: dict[str, str] | None = None) -> str:
    """Append query params to a base URL (stable ordering for testability)."""
    if not params:
        return base
    return f"{base}?{urlencode(params)}"


class Transport(Protocol):
    """Fetches the text body of a whitelisted URL."""

    def get_text(self, base: str, params: dict[str, str] | None = None) -> str: ...


class UrllibTransport:
    """Default transport: stdlib urllib, whitelist-gated (§8.3)."""

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.timeout_s = timeout_s

    def get_text(self, base: str, params: dict[str, str] | None = None) -> str:
        url = build_url(base, params)
        check_url(url)  # §8.3 default-deny egress guard
        req = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
        with urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 (host is whitelisted)
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
