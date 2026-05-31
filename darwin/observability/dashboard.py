"""Run status dashboard (ARCHITECTURE.md §9.5).

A simple, dependency-free reader over the controller's persisted run artifacts — the resumable
`runs/gen_<n>/state.json` files (§2.3) and the append-only cost ledger (§5.4) — that produces a
per-generation status summary (fitness table, spend, phase, deferred/failed counts) and renders
it to markdown. This is the "simple run dashboard reading `runs/gen_<n>/` for live status" §9.5
calls for: it works on a completed *or* in-progress generation, so it doubles as a live monitor.

Pure reads, no controller dependency beyond the state/ledger data classes — it can run against a
run directory after the fact or while a generation is mid-flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.cost import CostLedger
from darwin.controller.state import GenerationState, GenerationStateStore, OffspringState


@dataclass
class OffspringRow:
    """One offspring's status line in the dashboard."""

    name: str
    parent_survivor: str
    mutator: str | None
    backend: str
    finetune_status: str | None
    fitness: float | None
    antigaming_flags: int
    mutation_failed: bool
    cost_usd: float

    @classmethod
    def from_state(cls, off: OffspringState) -> "OffspringRow":
        return cls(
            name=off.name,
            parent_survivor=off.parent_survivor,
            mutator=off.mutator,
            backend=off.backend,
            finetune_status=off.finetune_status,
            fitness=off.fitness,
            antigaming_flags=off.antigaming_flags,
            mutation_failed=off.mutation_failed,
            cost_usd=off.cost_usd,
        )


@dataclass
class GenerationSummary:
    """Status of one generation (§9.5)."""

    generation: int
    phase: str
    completed: bool
    survivors: list[str]
    offspring: list[OffspringRow] = field(default_factory=list)
    spend_usd: float = 0.0
    spend_by_kind: dict[str, float] = field(default_factory=dict)

    @property
    def best_fitness(self) -> float | None:
        scored = [o.fitness for o in self.offspring if o.fitness is not None]
        return max(scored) if scored else None

    @property
    def deferred_count(self) -> int:
        return sum(1 for o in self.offspring if o.finetune_status == "deferred")

    @property
    def failed_count(self) -> int:
        return sum(
            1 for o in self.offspring if o.finetune_status in ("finetune_failed", "infra_failed")
        )

    @property
    def flagged_count(self) -> int:
        return sum(1 for o in self.offspring if o.antigaming_flags > 0)


@dataclass
class RunSummary:
    """Status of a whole run (all persisted generations) (§9.5)."""

    generations: list[GenerationSummary] = field(default_factory=list)
    total_spend_usd: float = 0.0

    @property
    def latest_generation(self) -> int | None:
        return self.generations[-1].generation if self.generations else None


def summarize_generation(state: GenerationState, ledger: CostLedger | None = None) -> GenerationSummary:
    """Build a generation status summary from its persisted state (+ optional ledger)."""
    spend = ledger.total(state.generation) if ledger is not None else 0.0
    by_kind = ledger.totals_by_kind(state.generation) if ledger is not None else {}
    return GenerationSummary(
        generation=state.generation,
        phase=state.phase,
        completed=state.completed,
        survivors=list(state.survivors_after_cull or []),
        offspring=[OffspringRow.from_state(o) for o in state.offspring],
        spend_usd=spend,
        spend_by_kind=by_kind,
    )


def summarize_run(
    state_store: GenerationStateStore, ledger: CostLedger | None = None
) -> RunSummary:
    """Summarize every persisted generation in a run directory (§9.5)."""
    latest = state_store.latest_generation()
    summaries: list[GenerationSummary] = []
    if latest is not None:
        for gen in range(latest + 1):
            if state_store.exists(gen):
                summaries.append(summarize_generation(state_store.load(gen), ledger))
    return RunSummary(
        generations=summaries,
        total_spend_usd=ledger.total() if ledger is not None else 0.0,
    )


# ------------------------------------------------------------------ rendering


def _fmt_fitness(f: float | None) -> str:
    if f is None:
        return "—"
    if f == float("-inf"):
        return "floor"
    return f"{f:.4g}"


def render_generation_markdown(summary: GenerationSummary) -> str:
    """A per-generation status block: header line + offspring table (§9.5)."""
    status = "complete" if summary.completed else f"in-progress ({summary.phase})"
    head = (
        f"### Generation {summary.generation} — {status}\n"
        f"- spend: ${summary.spend_usd:.4g}"
        f" | best offspring fitness: {_fmt_fitness(summary.best_fitness)}"
        f" | deferred: {summary.deferred_count}"
        f" | failed: {summary.failed_count}"
        f" | flagged: {summary.flagged_count}"
    )
    table = [
        "| offspring | parent | mutator | backend | finetune | fitness | flags | cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for o in summary.offspring:
        table.append(
            f"| {o.name} | {o.parent_survivor} | {o.mutator or '—'} | {o.backend} "
            f"| {o.finetune_status or '—'} | {_fmt_fitness(o.fitness)} | {o.antigaming_flags} "
            f"| ${o.cost_usd:.4g} |"
        )
    survivors = ", ".join(summary.survivors) if summary.survivors else "—"
    return head + "\n\n" + "\n".join(table) + f"\n\n_survivors carried in: {survivors}_"


def render_run_markdown(run: RunSummary) -> str:
    """The whole-run dashboard (§9.5)."""
    if not run.generations:
        return "# DARWIN run status\n\n_no generations recorded yet._"
    blocks = [render_generation_markdown(g) for g in run.generations]
    return (
        f"# DARWIN run status\n\n"
        f"- generations recorded: {len(run.generations)} "
        f"(latest: {run.latest_generation})\n"
        f"- total spend: ${run.total_spend_usd:.4g}\n\n" + "\n\n".join(blocks)
    )
