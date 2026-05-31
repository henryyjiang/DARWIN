"""Attribution-enforcement audit (ARCHITECTURE.md §8.4).

§8.4 requires that any idea taken from a paper is recorded **both** in the memory file's
`papers_cited` **and** as an inline note in the genome where the idea is implemented; datasets
acquired via the MCP `data.*` tools are recorded the same way (`datasets_used`, §8.3). This is
the Phase 7 *audit* that the enforcement actually held: it cross-checks a finished iteration's
recorded provenance against the genome source and reports mismatches.

Findings:
- `uncited_paper`        — an arXiv id appears in the genome but not in `papers_cited` (a paper
  idea used without recording it — the plagiarism-control surface §8.4 guards). **error**.
- `missing_inline_paper` — a `papers_cited` entry has no corresponding reference in the genome
  (recorded but the required inline note is absent). **error**.
- `unrecorded_dataset`   — a `datasets_used` entry isn't referenced anywhere in the genome
  (provenance recorded but not wired into the data scripts). **warning**.

Pure text analysis; the convenience `audit_iteration` reads the memory file + genome source via
the store / filesystem so the controller (or a CI audit job) can run it pre-finetune or after.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from darwin.memory.schema import IterationMemory

# arXiv ids as they appear in citations/code: arXiv:2401.01234, arxiv.org/abs/2401.01234v2, etc.
_ARXIV_RE = re.compile(r"(?:arxiv[:/]|abs/)\s*(\d{4}\.\d{4,5})", re.I)


def find_arxiv_ids(text: str) -> set[str]:
    """The set of arXiv ids referenced in `text` (version suffixes stripped)."""
    return {m.group(1) for m in _ARXIV_RE.finditer(text)}


def _arxiv_id(citation: str) -> str | None:
    """Extract the bare arXiv id from a `papers_cited` entry, if it is one."""
    m = _ARXIV_RE.search(citation)
    return m.group(1) if m else None


@dataclass(frozen=True)
class AttributionFinding:
    """One attribution-audit hit (§8.4)."""

    kind: str
    detail: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class AttributionReport:
    """Attribution findings for one iteration (§8.4)."""

    findings: list[AttributionFinding] = field(default_factory=list)

    @property
    def errors(self) -> list[AttributionFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[AttributionFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when there are no *error*-severity findings (warnings are advisory)."""
        return not self.errors


def audit_attribution(memory: IterationMemory, genome_source: str) -> AttributionReport:
    """Cross-check recorded provenance against the genome source (§8.4)."""
    report = AttributionReport()

    genome_ids = find_arxiv_ids(genome_source)
    cited_ids = {i for i in (_arxiv_id(c) for c in memory.papers_cited) if i}

    for gid in sorted(genome_ids - cited_ids):
        report.findings.append(
            AttributionFinding(
                kind="uncited_paper",
                detail=f"arXiv:{gid} is referenced in the genome but not in papers_cited",
            )
        )
    for cid in sorted(cited_ids - genome_ids):
        report.findings.append(
            AttributionFinding(
                kind="missing_inline_paper",
                detail=f"papers_cited entry arXiv:{cid} has no inline note in the genome",
            )
        )

    for dataset in memory.datasets_used:
        # match the bare id without an optional @revision pin (id@rev recorded; id used in code)
        bare = dataset.split("@", 1)[0]
        if bare and bare not in genome_source:
            report.findings.append(
                AttributionFinding(
                    kind="unrecorded_dataset",
                    detail=f"datasets_used entry {dataset!r} is not referenced in the genome",
                    severity="warning",
                )
            )

    return report


def audit_iteration(
    memory: IterationMemory, genome_dir: Path | str, *, source_reader=None
) -> AttributionReport:
    """Convenience: audit an iteration against its genome dir's source (§8.4)."""
    if source_reader is None:
        from darwin.controller.diversity import read_genome_source

        source_reader = read_genome_source
    return audit_attribution(memory, source_reader(Path(genome_dir)))


def render_attribution_markdown(report: AttributionReport) -> str:
    """A short audit block listing errors then warnings (§8.4)."""
    if not report.findings:
        return "✅ attribution OK — all paper/dataset provenance reconciles with the genome."
    lines = ["### Attribution audit (§8.4)"]
    for f in report.errors:
        lines.append(f"- ❌ **{f.kind}**: {f.detail}")
    for f in report.warnings:
        lines.append(f"- ⚠️ {f.kind}: {f.detail}")
    return "\n".join(lines)
