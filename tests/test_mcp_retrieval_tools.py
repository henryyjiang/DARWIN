"""MCP paper.* / data.* retrieval tools (ARCHITECTURE.md §9.3 / §8.3 / §8.4)."""

import json

from darwin.mcp.tools import DataToolset, PaperToolset
from darwin.sources import DataSource, PaperSource
from darwin.sources.whitelist import EgressBlocked


class FakeTransport:
    def __init__(self, bodies):
        self.bodies = bodies

    def get_text(self, base, params=None):
        if base not in self.bodies:
            raise KeyError(base)
        return self.bodies[base]


class BlockedTransport:
    def get_text(self, base, params=None):
        raise EgressBlocked("blocked")


ARXIV_FEED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <title>A Paper</title>
    <summary>An abstract.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Ada L</name></author>
  </entry>
</feed>"""

HF_DATASET = json.dumps(
    {"id": "bigcode/the-stack", "sha": "abc", "tags": [], "cardData": {"license": "other"}}
)


def test_paper_toolset_search_returns_citation():
    tools = PaperToolset(PaperSource(FakeTransport({"http://export.arxiv.org/api/query": ARXIV_FEED})))
    out = tools.search("anything")
    assert out["ok"] is True
    assert out["results"][0]["arxiv_id"] == "2401.01234"
    assert "arXiv:2401.01234" in out["results"][0]["citation"]


def test_paper_toolset_fetch_not_found():
    empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    tools = PaperToolset(PaperSource(FakeTransport({"http://export.arxiv.org/api/query": empty})))
    out = tools.fetch("9999.99999")
    assert out["ok"] is False and "no arXiv paper" in out["error"]


def test_paper_toolset_egress_blocked_is_structured_error():
    tools = PaperToolset(PaperSource(BlockedTransport()))
    out = tools.search("x")
    assert out["ok"] is False and "blocked" in out["error"]


def test_data_toolset_fetch_returns_license():
    tools = DataToolset(DataSource(FakeTransport({"https://huggingface.co/api/datasets/bigcode/the-stack": HF_DATASET})))
    out = tools.fetch("bigcode/the-stack")
    assert out["ok"] is True
    assert out["license"] == "other"
    assert out["pinned_id"] == "bigcode/the-stack@abc"


def test_data_toolset_egress_blocked():
    tools = DataToolset(DataSource(BlockedTransport()))
    out = tools.fetch("x/y")
    assert out["ok"] is False


def test_server_registers_retrieval_tools():
    import asyncio
    from pathlib import Path
    import tempfile

    from darwin.mcp.server import create_server
    from darwin.memory import MemoryStore

    store = MemoryStore(Path(tempfile.mkdtemp()))
    server = create_server(store)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"paper_search", "paper_fetch", "data_search", "data_fetch"}.issubset(names)


def test_server_can_disable_retrieval():
    import asyncio
    from pathlib import Path
    import tempfile

    from darwin.mcp.server import create_server
    from darwin.memory import MemoryStore

    store = MemoryStore(Path(tempfile.mkdtemp()))
    server = create_server(store, enable_retrieval=False)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert "paper_search" not in names
