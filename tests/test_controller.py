"""Controller state machine end-to-end with fakes (ARCHITECTURE.md §2.3).

Drives whole generations without Docker/GPU/Claude: a fake `GenerationOps` returns scripted
mutate/finetune/benchmark outcomes so we can assert selection, pairing, fitness aggregation,
the cull, resumability, and the global-memory-pass trigger.
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
from darwin.cost import CostLedger
from darwin.memory import GlobalMemory, MemoryStore


# ------------------------------------------------------------------ fakes


class FakeOps:
    """Scripted GenerationOps. `score_fn(offspring_name, generation) -> dict|None`.

    Returns None to simulate finetune_failed; a dict of per-benchmark scores otherwise.
    Records the order offspring were processed for resume assertions.
    """

    def __init__(self, score_fn, *, cost_usd=1.0, write_memory=False):
        self.score_fn = score_fn
        self.cost_usd = cost_usd
        self.write_memory = write_memory
        self.spawned: list[str] = []
        self.mutated: list[str] = []
        self.benchmarked: list[str] = []
        self._store: MemoryStore | None = None

    def bind_store(self, store):
        self._store = store

    def spawn(self, *, offspring, parent, generation):
        self.spawned.append(offspring.name)
        return Model(
            name=offspring.name,
            genome_dir=Path(f"ws/{offspring.name}/genome"),
            generation_born=generation,
            parent_survivor=offspring.parent_survivor,
            mutator=offspring.mutator,
            backend=offspring.backend,
        )

    def mutate(self, *, offspring, parent, mutator, state, generation):
        self.mutated.append(offspring.name)
        if self.write_memory and self._store is not None:
            self._store.write_iteration(_mem(state, generation))
        return MutateOutcome(final_commit="deadbeef", mutation_failed=False)

    def finetune(self, *, offspring, state, generation):
        scores = self.score_fn(offspring.name, generation)
        if scores is None:
            return FinetuneOutcomeView("finetune_failed", None, self.cost_usd)
        return FinetuneOutcomeView("ok", Path(f"ws/{offspring.name}/adapter.bin"), self.cost_usd)

    def benchmark(self, *, offspring, state, slice_id, generation):
        self.benchmarked.append(offspring.name)
        return self.score_fn(offspring.name, generation)


def _mem(state, generation):
    from darwin.memory import IterationMemory

    return IterationMemory(
        model=state.name,
        iteration=state.iteration,
        generation=generation,
        parent_survivor=state.parent_survivor,
        mutator=state.mutator or "claude",
        backend=state.backend,
        base_fitness=0.5,
        cost_usd=1.0,
        thesis="t",
        changes="c",
        smoke_results="green",
        outcome="o",
    )


class FakeSynth:
    def __init__(self):
        self.calls = 0

    def synthesize(self, digest, current):
        self.calls += 1
        return GlobalMemory(objectives=f"gen {digest.generation}", whats_working="w")


def seed_population(n_survivors=5, n_offspring=5) -> Population:
    """A population where survivors carry cached scores and offspring slots are unscored."""
    models = []
    for i in range(n_survivors):
        models.append(
            Model(
                name=f"s{i}",
                genome_dir=Path(f"ws/s{i}/genome"),
                fitness=0.5 + i * 0.01,
                scores={"code": 0.5, "math": 0.5},
                is_survivor=True,
            )
        )
    for i in range(n_offspring):
        models.append(
            Model(name=f"o{i}", genome_dir=Path(f"ws/o{i}/genome"), fitness=0.1)
        )
    return Population(models)


def make_controller(tmp_path, ops, synth=None, config=None):
    store = MemoryStore(tmp_path / "store")
    if hasattr(ops, "bind_store"):
        ops.bind_store(store)
    return Controller(
        config=config or DarwinConfig(),
        store=store,
        ledger=CostLedger(tmp_path / "cost.jsonl"),
        state_store=GenerationStateStore(tmp_path / "runs"),
        ops=ops,
        synthesizer=synth,
        rng=random.Random(0),
    )


# ------------------------------------------------------------------ tests


def test_single_generation_selects_spawns_scores_and_culls(tmp_path):
    # offspring all beat the survivor baseline (0.5) -> they should dominate next population
    ops = FakeOps(lambda name, gen: {"code": 0.8, "math": 0.8})
    synth = FakeSynth()
    ctrl = make_controller(tmp_path, ops, synth)

    pop = seed_population()
    nxt = ctrl.run_generation(0, pop)

    # 5 offspring spawned/mutated/benchmarked (the non-survivor slots o0..o4)
    assert sorted(ops.spawned) == ["o0", "o1", "o2", "o3", "o4"]
    assert sorted(ops.benchmarked) == ["o0", "o1", "o2", "o3", "o4"]
    # next population is still 10; offspring fitness = normalized(1.6) - cost(0.05*1.0) = 1.55
    assert len(nxt.models) == 10
    offspring = nxt.offspring()
    assert len(offspring) == 5
    assert all(m.fitness == pytest.approx(1.55) for m in offspring)
    # global-memory pass ran exactly once
    assert synth.calls == 1
    # state persisted and complete
    assert ctrl.state_store.load(0).completed is True


def test_fitness_uses_survivor_baseline_and_penalties(tmp_path):
    ops = FakeOps(lambda name, gen: {"code": 0.6, "math": 0.7}, cost_usd=2.0)
    ctrl = make_controller(tmp_path, ops, FakeSynth())
    nxt = ctrl.run_generation(0, seed_population())
    off = nxt.get("o0")
    # baseline = {code:0.5, math:0.5}; normalized = (0.6/0.5 + 0.7/0.5)/2 weighted uniform
    # = 0.5*1.2 + 0.5*1.4 = 1.3 ; minus lambda_cost(0.05)*2.0 = 0.1 -> 1.2
    assert off.fitness == pytest.approx(1.3 - 0.1)


def test_finetune_failed_gets_floor_fitness_and_is_culled(tmp_path):
    # o0 fails finetune (floor); others are great -> o0 must not survive
    def score(name, gen):
        return None if name == "o0" else {"code": 0.9, "math": 0.9}

    ctrl = make_controller(tmp_path, FakeOps(score), FakeSynth())
    nxt = ctrl.run_generation(0, seed_population())
    o0 = nxt.get("o0")
    assert o0.finetune_failed is True
    assert o0.fitness == float("-inf")
    survivors = {m.name for m in nxt.survivors()}
    assert "o0" not in survivors


def test_memory_patched_with_final_fitness(tmp_path):
    ops = FakeOps(lambda name, gen: {"code": 0.6, "math": 0.6}, write_memory=True)
    ctrl = make_controller(tmp_path, ops, FakeSynth())
    ctrl.run_generation(0, seed_population())
    mem = ctrl.store.read_iteration("o0", 0)  # iteration 0 for a fresh slot
    assert mem.final_fitness is not None
    # normalized 1.2 minus default cost penalty (0.05 * cost_usd 1.0) = 1.15
    assert mem.final_fitness == pytest.approx(1.15)
    assert mem.mutation_failed is False


def test_resume_skips_completed_offspring(tmp_path):
    ops = FakeOps(lambda name, gen: {"code": 0.6})
    ctrl = make_controller(tmp_path, ops, FakeSynth())
    pop = seed_population()
    ctrl.run_generation(0, pop)
    first_mutated = list(ops.mutated)

    # a second controller resuming the completed generation must NOT re-run any offspring
    ops2 = FakeOps(lambda name, gen: {"code": 0.6})
    ctrl2 = make_controller(tmp_path, ops2, FakeSynth())
    nxt = ctrl2.run_generation(0, pop)
    assert ops2.mutated == []  # nothing re-mutated; state was already complete
    assert ops2.benchmarked == []
    assert len(first_mutated) == 5
    assert len(nxt.models) == 10


def test_two_generations_compound(tmp_path):
    # offspring keep beating the baseline; population fitness should not collapse
    ops = FakeOps(lambda name, gen: {"code": 0.7, "math": 0.7})
    synth = FakeSynth()
    ctrl = make_controller(tmp_path, ops, synth)

    pop = seed_population()
    final = ctrl.run(2, pop)
    assert synth.calls == 2  # one global-memory pass per generation
    assert ctrl.state_store.latest_generation() == 1
    assert len(final.models) == 10
    # every model in the final population has a fitness assigned
    assert all(m.fitness is not None for m in final.models)


def test_single_survivor_uses_claude_fallback_backend(tmp_path):
    # only one survivor -> mutator must be None -> backend forced to claude (§3.2)
    captured = {}

    class CapturingOps(FakeOps):
        def mutate(self, *, offspring, parent, mutator, state, generation):
            captured[offspring.name] = (state.mutator, state.backend)
            return super().mutate(
                offspring=offspring, parent=parent, mutator=mutator,
                state=state, generation=generation,
            )

    pop = Population(
        [Model(name="s0", genome_dir=Path("ws/s0/genome"), fitness=0.9, scores={"code": 0.5})]
        + [Model(name=f"o{i}", genome_dir=Path(f"ws/o{i}/genome"), fitness=0.1) for i in range(3)]
    )
    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.mutation.backend = "local"  # would normally be local, but fallback forces claude
    ops = CapturingOps(lambda name, gen: {"code": 0.6})
    ctrl = make_controller(tmp_path, ops, FakeSynth(), config=cfg)
    ctrl.run_generation(0, pop)
    for name, (mutator, backend) in captured.items():
        assert mutator is None
        assert backend == "claude"
