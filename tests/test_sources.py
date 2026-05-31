"""Whitelisted paper/dataset retrieval (ARCHITECTURE.md §8.3 / §8.4 / §9.3)."""

import json

import pytest

from darwin.sources import (
    DataSource,
    EgressBlocked,
    PaperSource,
    UrllibTransport,
    build_url,
    check_url,
    host_allowed,
    normalize_arxiv_id,
    parse_atom,
)


# ------------------------------------------------------------------ whitelist (§8.3)


def test_whitelist_allows_arxiv_and_hf():
    assert host_allowed("http://export.arxiv.org/api/query?x=1")
    assert host_allowed("https://huggingface.co/api/datasets")
    assert host_allowed("https://cdn-lfs.hf.co/some/shard")  # CDN subdomain


def test_whitelist_blocks_arbitrary_host():
    assert not host_allowed("https://evil.example.com/scrape")
    with pytest.raises(EgressBlocked):
        check_url("https://evil.example.com/scrape")


def test_urllib_transport_refuses_nonwhitelisted_before_network():
    # check_url runs before any socket call, so this raises without touching the network
    with pytest.raises(EgressBlocked):
        UrllibTransport().get_text("https://evil.example.com/x")


def test_build_url_encodes_params():
    assert build_url("http://h/q", {"a": "b c", "n": "2"}) == "http://h/q?a=b+c&n=2"


# ------------------------------------------------------------------ fake transport


class FakeTransport:
    """Returns canned bodies keyed by base URL; records requests."""

    def __init__(self, bodies: dict[str, str]):
        self.bodies = bodies
        self.calls: list[tuple[str, dict | None]] = []

    def get_text(self, base, params=None):
        self.calls.append((base, params))
        return self.bodies[base]


ARXIV_FEED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <title>LoRA:  Low-Rank
      Adaptation</title>
    <summary>We propose a   parameter-efficient method.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Edward Hu</name></author>
    <author><name>Yelong Shen</name></author>
    <link rel="alternate" href="http://arxiv.org/abs/2401.01234v2"/>
  </entry>
</feed>"""


# ------------------------------------------------------------------ papers (§8.4)


def test_normalize_arxiv_id_strips_version_and_url():
    assert normalize_arxiv_id("http://arxiv.org/abs/2401.01234v2") == "2401.01234"
    assert normalize_arxiv_id("arXiv:2312.99999") == "2312.99999"


def test_parse_atom_extracts_fields_and_citation():
    refs = parse_atom(ARXIV_FEED)
    assert len(refs) == 1
    ref = refs[0]
    assert ref.arxiv_id == "2401.01234"
    assert ref.title == "LoRA: Low-Rank Adaptation"  # whitespace normalized
    assert ref.authors == ["Edward Hu", "Yelong Shen"]
    assert ref.year == "2024"
    cite = ref.citation()
    assert "Edward Hu, Yelong Shen" in cite and "arXiv:2401.01234" in cite and "(2024)" in cite


def test_paper_source_search_and_fetch():
    transport = FakeTransport({"http://export.arxiv.org/api/query": ARXIV_FEED})
    source = PaperSource(transport)

    results = source.search("low rank adaptation", limit=3)
    assert results[0].arxiv_id == "2401.01234"
    # the query was sent to the arXiv endpoint with a max_results param
    base, params = transport.calls[0]
    assert "export.arxiv.org" in base and params["max_results"] == "3"

    fetched = source.fetch("2401.01234v2")
    assert fetched is not None and fetched.arxiv_id == "2401.01234"


def test_paper_search_empty_query_skips_network():
    transport = FakeTransport({})
    assert PaperSource(transport).search("   ") == []
    assert transport.calls == []


# ------------------------------------------------------------------ datasets (§8.3)


HF_DATASET = json.dumps(
    {
        "id": "bigcode/the-stack",
        "sha": "abc123",
        "downloads": 9001,
        "likes": 42,
        "gated": False,
        "tags": ["task_categories:text-generation", "license:other"],
        "cardData": {"license": "other", "pretty_name": "The Stack"},
    }
)
HF_SEARCH = json.dumps(
    [
        {"id": "bigcode/the-stack", "downloads": 9001, "cardData": {"license": "other"},
         "tags": []},
        {"id": "openai/gsm8k", "downloads": 5000, "cardData": {"license": "mit"}, "tags": []},
    ]
)


def test_data_source_fetch_returns_license_and_pin():
    transport = FakeTransport({"https://huggingface.co/api/datasets/bigcode/the-stack": HF_DATASET})
    ref = DataSource(transport).fetch("bigcode/the-stack")
    assert ref is not None
    assert ref.license == "other"
    assert ref.description == "The Stack"
    assert ref.revision == "abc123"  # main resolved to the commit sha
    assert ref.pinned_id == "bigcode/the-stack@abc123"


def test_data_source_search_lists_datasets():
    transport = FakeTransport({"https://huggingface.co/api/datasets": HF_SEARCH})
    refs = DataSource(transport).search("code", limit=5)
    assert [r.dataset_id for r in refs] == ["bigcode/the-stack", "openai/gsm8k"]
    assert refs[1].license == "mit"


def test_data_license_falls_back_to_tag():
    obj = json.dumps({"id": "x/y", "tags": ["license:apache-2.0"], "cardData": {}})
    transport = FakeTransport({"https://huggingface.co/api/datasets/x/y": obj})
    ref = DataSource(transport).fetch("x/y")
    assert ref.license == "apache-2.0"
