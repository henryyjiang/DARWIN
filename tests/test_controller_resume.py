"""Crash/resume hardening coverage (ARCHITECTURE.md §2.3 / Phase 7).

Each generation step persists `runs/gen_<n>/state.json` and is idempotent, so a crash
mid-generation resumes at the first incomplete stage without re-running completed work. These
tests crash at two boundaries — mid-offspring (a finetune raises) and during the global-memory
pass — then resume with a fresh controller and assert no completed offspring is recomputed.
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


class CrashOps:
    """Scripted ops that can raise once in finetune for a named offspring (§2.3 resume test)."""

    def __init__(self, *, crash_on=None):
        self.crash_on = crash_on
        self.crashed = False
        self.mutated: list[str] = []
        self.finetuned: list[str] = []
        self.benchmarked: list[str] = []

    def spawn(self, *, offspring, parent, generation):
        return Model(name=offspring.name, genome_dir=Path(f"ws/{offspring.name}/genome"),
                     parent_survivor=offspring.parent_survivor, mutator=offspring.mutator,
                     backend=offspring.backend)

    def mutate(self, *, offspring, parent, mutator, state, generation):
        self.mutated.append(offspring.name)
        return MutateOutcome(final_commit="abc", mutation_failed=False)

    def finetune(self, *, offspring, state, generation):
        if offspring.name == self.crash_on and not self.crashed:
            self.crashed = True
            raise RuntimeError(f"simulated crash finetuning {offspring.name}")
        self.finetuned.append(offspring.name)
        return FinetuneOutcomeView("ok", Path(f"ws/{offspring.name}/adapter.bin"), 1.0)

    def benchmark(self, *, offspring, state, slice_id, generation):
        self.benchmarked.append(offspring.name)
        return {"code": 0.6}


class CrashSynth:
    """Global-memory synthesizer that raises once, then succeeds on resume."""

    def __init__(self):
        self.calls = 0

    def synthesize(self, digest, current):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated global-memory pass crash")
        return GlobalMemory(objectives="recovered")


def seed_population() -> Population:
    models = [Model(name="s0", genome_dir=Path("ws/s0/genome"), fitness=0.5,
                    scores={"code": 0.5}, is_survivor=True)]
    models += [Model(name=f"o{i}", genome_dir=Path(f"ws/o{i}/genome"), fitness=0.1)
               for i in range(5)]
    return Population(models)


def make_controller(tmp_path, ops, synth):
    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    return Controller(
        config=cfg,
        store=MemoryStore(tmp_path / "store"),
        ledger=CostLedger(tmp_path / "cost.jsonl"),
        state_store=GenerationStateStore(tmp_path / "runs"),
        ops=ops,
        synthesizer=synth,
        rng=random.Random(0),
    )


def test_resume_after_midoffspring_crash(tmp_path):
    pop = seed_population()
    ops1 = CrashOps(crash_on="o2")
    ctrl1 = make_controller(tmp_path, ops1, _OkSynth())

    with pytest.raises(RuntimeError):
        ctrl1.run_generation(0, pop)

    # offspring before o2 were completed and persisted; o2 was not
    state = ctrl1.state_store.load(0)
    done = {o.name for o in state.offspring if o.benchmark_done}
    assert "o0" in done and "o1" in done
    assert "o2" not in done

    # resume with a fresh, non-crashing controller
    ops2 = CrashOps()
    ctrl2 = make_controller(tmp_path, ops2, _OkSynth())
    nxt = ctrl2.run_generation(0, pop)

    # completed offspring are NOT re-finetuned; only the incomplete tail runs
    assert "o0" not in ops2.finetuned and "o1" not in ops2.finetuned
    assert {"o2", "o3", "o4"}.issubset(set(ops2.finetuned))
    assert ctrl2.state_store.load(0).completed is True
    assert len(nxt.models) == 6


def test_resume_after_global_memory_crash(tmp_path):
    pop = seed_population()
    synth = CrashSynth()
    ops1 = CrashOps()
    ctrl1 = make_controller(tmp_path, ops1, synth)

    with pytest.raises(RuntimeError):
        ctrl1.run_generation(0, pop)

    # all offspring finished and the cull happened before the pass crashed
    state = ctrl1.state_store.load(0)
    assert state.at_least("culled")
    assert not state.at_least("global_memory")
    assert len(ops1.finetuned) == 5

    # resume: no offspring re-run; the pass is retried and the generation completes
    ops2 = CrashOps()
    ctrl2 = make_controller(tmp_path, ops2, synth)
    nxt = ctrl2.run_generation(0, pop)
    assert ops2.finetuned == []  # offspring stage already complete
    assert synth.calls == 2  # crashed once, succeeded on resume
    assert ctrl2.state_store.load(0).completed is True
    assert nxt is not None


class _OkSynth:
    def synthesize(self, digest, current):
        return GlobalMemory(objectives="ok")
