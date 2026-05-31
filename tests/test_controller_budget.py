"""Hard per-generation budget cap wired into the controller (ARCHITECTURE.md §5.4).

Once the generation's `gen_budget_usd` is exhausted the controller launches no new finetunes;
already-launched (earlier) offspring finish, and offspring that never launched are carried as
`deferred` (unscored, not floor-fitness) for re-attempt next generation if budget frees.
"""

import random
from pathlib import Path

import pytest

from darwin.config import DarwinConfig
from darwin.controller import (
    Controller,
    FinetuneOutcomeView,
    GenerationStateStore,
    Model,
    MutateOutcome,
    Population,
)
from darwin.cost import BudgetGuard, CostLedger
from darwin.memory import GlobalMemory, MemoryStore


class LedgerOps:
    """Ops that charge the ledger per finetune so the BudgetGuard sees real spend."""

    def __init__(self, ledger, *, cost_per_job=10.0):
        self.ledger = ledger
        self.cost_per_job = cost_per_job
        self.finetuned: list[str] = []

    def spawn(self, *, offspring, parent, generation):
        return Model(name=offspring.name, genome_dir=Path(f"ws/{offspring.name}/genome"),
                     parent_survivor=offspring.parent_survivor, mutator=offspring.mutator,
                     backend=offspring.backend)

    def mutate(self, *, offspring, parent, mutator, state, generation):
        return MutateOutcome(final_commit="abc", mutation_failed=False)

    def finetune(self, *, offspring, state, generation):
        self.finetuned.append(offspring.name)
        self.ledger.record(generation=generation, kind="finetune",
                           amount_usd=self.cost_per_job, reason="ft", model=offspring.name)
        return FinetuneOutcomeView("ok", Path(f"ws/{offspring.name}/adapter.bin"),
                                   self.cost_per_job)

    def benchmark(self, *, offspring, state, slice_id, generation):
        return {"code": 0.6}


class FakeSynth:
    def synthesize(self, digest, current):
        return GlobalMemory(objectives="x")


def seed_population() -> Population:
    models = [Model(name="s0", genome_dir=Path("ws/s0/genome"), fitness=0.5,
                    scores={"code": 0.5}, is_survivor=True)]
    models += [Model(name=f"o{i}", genome_dir=Path(f"ws/o{i}/genome"), fitness=0.1)
               for i in range(5)]
    return Population(models)


def make_controller(tmp_path, ledger, budget, cfg):
    return Controller(
        config=cfg,
        store=MemoryStore(tmp_path / "store"),
        ledger=ledger,
        state_store=GenerationStateStore(tmp_path / "runs"),
        ops=LedgerOps(ledger),
        synthesizer=FakeSynth(),
        budget=budget,
        rng=random.Random(0),
    )


def test_budget_cap_defers_offspring_once_exhausted(tmp_path):
    ledger = CostLedger(tmp_path / "cost.jsonl")
    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    cfg.cost.gen_budget_usd = 25.0  # ~2 jobs at $10 before the 3rd sees spend >= cap
    ctrl = make_controller(tmp_path, ledger, BudgetGuard(ledger, cfg.cost), cfg)

    nxt = ctrl.run_generation(0, seed_population())
    state = ctrl.state_store.load(0)
    statuses = [o.finetune_status for o in state.offspring]

    # exactly the jobs that fit under the cap launched; the rest were deferred
    assert statuses.count("ok") == 3   # $10,$20,$30 -> 3rd job overshoots, then stop
    assert statuses.count("deferred") == 2
    assert ledger.total(0) == pytest.approx(30.0)  # in-flight overshoot allowed, not killed

    # deferred offspring are unscored (None) — not floor fitness (not a recipe failure)
    deferred = [o for o in state.offspring if o.finetune_status == "deferred"]
    assert all(o.fitness is None for o in deferred)
    # and they carry into the next population (size stays stable)
    assert len(nxt.models) == 6


def test_no_budget_guard_launches_all(tmp_path):
    ledger = CostLedger(tmp_path / "cost.jsonl")
    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    ctrl = make_controller(tmp_path, ledger, None, cfg)  # no guard
    ctrl.run_generation(0, seed_population())
    state = ctrl.state_store.load(0)
    assert all(o.finetune_status == "ok" for o in state.offspring)


def test_uncapped_guard_never_defers(tmp_path):
    ledger = CostLedger(tmp_path / "cost.jsonl")
    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    cfg.cost.gen_budget_usd = None  # capped guard but no cap value
    ctrl = make_controller(tmp_path, ledger, BudgetGuard(ledger, cfg.cost), cfg)
    ctrl.run_generation(0, seed_population())
    state = ctrl.state_store.load(0)
    assert all(o.finetune_status == "ok" for o in state.offspring)
