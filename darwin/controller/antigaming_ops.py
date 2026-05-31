"""Concrete anti-gaming scanner: wire the §6.4 producers (ARCHITECTURE.md §6.4).

`LocalAntiGamingScanner` implements the controller's `AntiGamingScanner` seam against the real
`darwin.antigaming` cores. It assembles one offspring's scan inputs from the local filesystem
and the controller-held eval data, then runs `run_antigaming_scan`:

- **genome diff** — `root..HEAD` of the offspring's genome repo (the mutator's added code),
  fed to the rule-based or Claude reviewer (§6.4 / §4.7).
- **data texts** — the genome's source (where a leaked-data attempt would live), scanned for
  n-gram overlap with the held-out eval items (§6.4 contamination).
- **eval items** — supplied by an injected `eval_items_provider(generation, slice_id)`; these
  are **host-only** (§6.2) so the provider is the controller's, not the container's. Absent =>
  the contamination check no-ops (deferred until the live eval set lands).
- **OOD probe scores** — supplied by an injected `ood_probe(offspring, generation, slice_id)`;
  absent => the generalization-gap check no-ops (deferred with the eval-container infra).

With no providers injected (today's default), only the genome-diff review runs — exactly the
§6.4 check that needs no infra — matching the deferred-infra split used across the project.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from darwin.antigaming import (
    AntiGamingReport,
    AntiGamingScanInput,
    make_genome_reviewer,
    run_antigaming_scan,
)
from darwin.antigaming.genome_review import GenomeReviewer
from darwin.config import AntiGamingConfig
from darwin.controller.diversity import read_genome_source
from darwin.controller.population import Model
from darwin.controller.state import OffspringState

EvalItemsProvider = Callable[[int, int], list[str]]  # (generation, slice_id) -> eval items
OODProbe = Callable[[Model, int, int], dict[str, float]]  # (offspring, gen, slice) -> scores


def genome_mutation_diff(genome_dir: Path) -> str:
    """Unified diff of the mutator's changes: the genome repo's root commit -> HEAD.

    The root commit is the clone-of-S baseline (§4.4), so this is exactly the code the mutator
    added/changed this window. Returns "" if the dir isn't a git repo (e.g. an unmutated seed).
    """
    if not (genome_dir / ".git").exists():
        return ""
    try:
        root = subprocess.run(
            ["git", "-C", str(genome_dir), "rev-list", "--max-parents=0", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        if not root:
            return ""
        return subprocess.run(
            ["git", "-C", str(genome_dir), "diff", root[-1], "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return ""


@dataclass
class LocalAntiGamingScanner:
    """The controller's `AntiGamingScanner`, wiring the real §6.4 producers."""

    config: AntiGamingConfig
    reviewer: GenomeReviewer | None = None  # default resolved from config
    eval_items_provider: EvalItemsProvider | None = None  # host-only eval items (§6.2)
    ood_probe: OODProbe | None = None  # OOD probe scores (deferred infra)

    def scan(
        self, *, offspring: Model, state: OffspringState, slice_id: int, generation: int
    ) -> AntiGamingReport:
        eval_items = (
            self.eval_items_provider(generation, slice_id)
            if self.eval_items_provider is not None
            else []
        )
        ood = (
            self.ood_probe(offspring, generation, slice_id)
            if self.ood_probe is not None
            else {}
        )
        data_text = read_genome_source(offspring.genome_dir)
        inp = AntiGamingScanInput(
            diff=genome_mutation_diff(offspring.genome_dir),
            data_texts=[data_text] if data_text else [],
            eval_items=eval_items,
            held_out_scores=dict(state.scores),
            ood_scores=ood,
        )
        reviewer = self.reviewer or make_genome_reviewer(self.config)
        return run_antigaming_scan(inp, config=self.config, reviewer=reviewer)
