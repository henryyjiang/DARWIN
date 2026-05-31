"""Anti-gaming heuristics (ARCHITECTURE.md §6.4).

A model can optimize *for the benchmark* rather than for genuine capability (reward hacking).
This package implements the §6.4 layered defenses as *producers* of fitness penalties:

- `contamination.py` — n-gram overlap of the genome's data scripts/datasets vs. the held-out
  eval items (catches pulling benchmark data into the training mix).
- `genome_review.py` — a lightweight (rule-based or Claude) review of the genome diff flagging
  benchmark-format special-casing, hardcoded answers, or eval-harness detection.
- `plausibility.py` — a generalization-gap check: a held-out score that wildly exceeds a quick
  out-of-distribution probe is flagged as suspected overfit/gaming.

Each emits `AntiGamingFlag`s collected into an `AntiGamingReport` (`report.py`); `scan.py`
composes the three into one report whose `count` feeds the §6.3 fitness reduction
(`lambda_penalty * antigaming_flags`). The controller wires the scan per offspring so gaming is
selected against, not merely logged.

The pure cores (n-gram math, rule patterns, gap math, prompt build/parse) run with no infra; the
*live* contamination eval items, the Claude reviewer call, and the OOD probe run are injected by
the controller (host-only eval data / API / a benchmark run), mirroring the deferred-infra split
used across the project.
"""

from darwin.antigaming.contamination import contamination_scan, word_ngrams
from darwin.antigaming.genome_review import (
    ClaudeGenomeReviewer,
    GenomeReviewer,
    RuleBasedGenomeReviewer,
    added_lines,
    build_review_prompt,
    parse_review,
)
from darwin.antigaming.plausibility import generalization_gap_flags
from darwin.antigaming.report import AntiGamingFlag, AntiGamingReport
from darwin.antigaming.scan import (
    AntiGamingScanInput,
    make_genome_reviewer,
    run_antigaming_scan,
)

__all__ = [
    "AntiGamingFlag",
    "AntiGamingReport",
    "contamination_scan",
    "word_ngrams",
    "generalization_gap_flags",
    "GenomeReviewer",
    "RuleBasedGenomeReviewer",
    "ClaudeGenomeReviewer",
    "added_lines",
    "build_review_prompt",
    "parse_review",
    "AntiGamingScanInput",
    "run_antigaming_scan",
    "make_genome_reviewer",
]
