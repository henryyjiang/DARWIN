"""Composed anti-gaming scan (ARCHITECTURE.md §6.4).

`run_antigaming_scan` runs the three §6.4 producers over one offspring's inputs and returns a
single `AntiGamingReport` whose `count` the controller feeds to fitness as `antigaming_flags`
(§6.3). Each check no-ops when its inputs are absent, so the scan degrades gracefully as the
live infra (host-only eval items, the OOD probe run) lands incrementally:

- contamination scan        — needs `data_texts` + `eval_items`
- genome-diff hack review   — needs `diff` (+ a `GenomeReviewer`)
- generalization-gap check  — needs `held_out_scores` + `ood_scores`

Pure orchestration given an injected reviewer; the reviewer choice (`claude` / `rule` / `none`)
and thresholds come from `AntiGamingConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.antigaming.contamination import contamination_scan
from darwin.antigaming.genome_review import (
    ClaudeGenomeReviewer,
    GenomeReviewer,
    RuleBasedGenomeReviewer,
)
from darwin.antigaming.plausibility import generalization_gap_flags
from darwin.antigaming.report import AntiGamingReport
from darwin.config import AntiGamingConfig


@dataclass
class AntiGamingScanInput:
    """The per-offspring inputs to the §6.4 scan (any may be empty -> that check no-ops)."""

    diff: str = ""  # genome diff (base..final), added lines reviewed (§6.4)
    data_texts: list[str] = field(default_factory=list)  # the genome's data scripts/manifests
    eval_items: list[str] = field(default_factory=list)  # held-out eval items (host-only, §6.2)
    held_out_scores: dict[str, float] = field(default_factory=dict)  # scored-slice scores
    ood_scores: dict[str, float] = field(default_factory=dict)  # OOD probe scores


def make_genome_reviewer(config: AntiGamingConfig) -> GenomeReviewer | None:
    """Resolve the configured genome-diff reviewer (§6.4 / §4.7), or None to skip it."""
    if config.genome_reviewer == "rule":
        return RuleBasedGenomeReviewer()
    if config.genome_reviewer == "claude":
        return ClaudeGenomeReviewer()
    return None


def run_antigaming_scan(
    inp: AntiGamingScanInput,
    *,
    config: AntiGamingConfig,
    reviewer: GenomeReviewer | None = None,
) -> AntiGamingReport:
    """Compose the three §6.4 checks into one report (`count` feeds fitness, §6.3)."""
    report = AntiGamingReport()
    if not config.enabled:
        return report

    if inp.data_texts and inp.eval_items:
        report.extend(
            contamination_scan(
                inp.data_texts,
                inp.eval_items,
                n=config.ngram_n,
                min_overlap=config.contamination_min_overlap,
                max_flags=config.contamination_max_flags,
            )
        )

    if inp.diff.strip() and reviewer is not None:
        report.extend(reviewer.review(inp.diff))

    if inp.held_out_scores and inp.ood_scores:
        report.extend(
            generalization_gap_flags(
                inp.held_out_scores,
                inp.ood_scores,
                max_gap=config.max_generalization_gap,
            )
        )

    return report
