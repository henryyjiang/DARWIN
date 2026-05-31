"""Observability — run status dashboard (ARCHITECTURE.md §9.5).

Reads the controller's persisted run artifacts (`runs/gen_<n>/state.json` + the cost ledger)
and renders a per-generation / whole-run status summary (fitness table, spend, phase,
deferred/failed/flagged counts). Pure reads — usable live (mid-generation) or after the fact.

CLI: `python -m darwin.observability --runs runs [--cost cost.jsonl]`.
"""

from darwin.observability.dashboard import (
    GenerationSummary,
    OffspringRow,
    RunSummary,
    render_generation_markdown,
    render_run_markdown,
    summarize_generation,
    summarize_run,
)
from darwin.observability.attribution import (
    AttributionFinding,
    AttributionReport,
    audit_attribution,
    audit_iteration,
    find_arxiv_ids,
    render_attribution_markdown,
)

__all__ = [
    "OffspringRow",
    "GenerationSummary",
    "RunSummary",
    "summarize_generation",
    "summarize_run",
    "render_generation_markdown",
    "render_run_markdown",
    "AttributionFinding",
    "AttributionReport",
    "audit_attribution",
    "audit_iteration",
    "find_arxiv_ids",
    "render_attribution_markdown",
]
