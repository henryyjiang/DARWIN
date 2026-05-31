"""Attribution-enforcement audit (ARCHITECTURE.md §8.4)."""

from darwin.memory.schema import IterationMemory
from darwin.observability import (
    audit_attribution,
    audit_iteration,
    find_arxiv_ids,
    render_attribution_markdown,
)


def _mem(**kw) -> IterationMemory:
    base = dict(
        model="o0", iteration=0, generation=1, parent_survivor="s0", mutator="s1",
        backend="claude", base_fitness=0.5, cost_usd=1.0,
        thesis="t", changes="c", smoke_results="g", outcome="o",
    )
    base.update(kw)
    return IterationMemory(**base)


def test_find_arxiv_ids_handles_formats():
    text = "see arXiv:2401.01234 and https://arxiv.org/abs/2312.99999v2"
    assert find_arxiv_ids(text) == {"2401.01234", "2312.99999"}


def test_clean_attribution_is_ok():
    mem = _mem(papers_cited=["arXiv:2401.01234"], datasets_used=["bigcode/the-stack@v1"])
    genome = "# implements idea from arXiv:2401.01234\ndata = load('bigcode/the-stack')\n"
    report = audit_attribution(mem, genome)
    assert report.ok
    assert report.findings == []


def test_uncited_paper_in_genome_is_error():
    mem = _mem(papers_cited=[])
    genome = "# borrowed from arXiv:2401.01234 without recording it\n"
    report = audit_attribution(mem, genome)
    assert not report.ok
    assert report.errors[0].kind == "uncited_paper"
    assert "2401.01234" in report.errors[0].detail


def test_cited_paper_without_inline_note_is_error():
    mem = _mem(papers_cited=["arXiv:2401.01234"])
    genome = "lora_rank = 32\n"  # no inline reference
    report = audit_attribution(mem, genome)
    assert not report.ok
    assert report.errors[0].kind == "missing_inline_paper"


def test_unrecorded_dataset_is_warning_only():
    mem = _mem(datasets_used=["bigcode/the-stack@v1"])
    genome = "lora_rank = 16\n"  # dataset not referenced
    report = audit_attribution(mem, genome)
    assert report.ok  # warnings don't fail the audit
    assert report.warnings[0].kind == "unrecorded_dataset"


def test_dataset_revision_pin_matches_bare_id():
    mem = _mem(datasets_used=["bigcode/the-stack@abc123"])
    genome = "data = load('bigcode/the-stack')\n"  # code uses the bare id, no pin
    report = audit_attribution(mem, genome)
    assert report.warnings == []


def test_render_markdown():
    mem = _mem(papers_cited=["arXiv:2401.01234"])
    md = render_attribution_markdown(audit_attribution(mem, "x = 1\n"))
    assert "Attribution audit" in md and "missing_inline_paper" in md
    clean = render_attribution_markdown(audit_attribution(_mem(), "x = 1\n"))
    assert "attribution OK" in clean


def test_audit_iteration_reads_genome_dir(tmp_path):
    g = tmp_path / "genome"
    g.mkdir()
    (g / "recipe.py").write_text("# from arXiv:2401.01234\nlora_rank = 16\n", encoding="utf-8")
    mem = _mem(papers_cited=["arXiv:2401.01234"])
    report = audit_iteration(mem, g)
    assert report.ok
