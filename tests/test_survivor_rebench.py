"""Survivor re-benchmark on eval-slice rotation (ARCHITECTURE.md §6.2)."""

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


class CountingOps:
    """Records which models were benchmarked and on which slice."""

    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.benchmarked: list[tuple[str, int]] = []

    def spawn(self, *, offspring, parent, generation):
        return Model(name=offspring.name, genome_dir=Path("x"))

    def mutate(self, **k):
        return MutateOutcome(final_commit="c", mutation_failed=False)

    def finetune(self, **k):
        return FinetuneOutcomeView("ok", Path("a"), 0.0)

    def benchmark(self, *, offspring, state, slice_id, generation):
        self.benchmarked.append((offspring.name, slice_id))
        return self.score_fn(offspring.name, slice_id)


class FakeSynth:
    def synthesize(self, digest, current):
        return GlobalMemory(objectives="x")


def _make(tmp_path, ops, cfg):
    return Controller(
        config=cfg, store=MemoryStore(tmp_path / "s"), ledger=CostLedger(tmp_path / "c.jsonl"),
        state_store=GenerationStateStore(tmp_path / "runs"), ops=ops,
        synthesizer=FakeSynth(), rng=random.Random(0),
    )


def _pop():
    return Population(
        [Model(name="s0", genome_dir=Path("x"), fitness=0.9, scores={"code": 0.5},
               is_survivor=True, scored_slice=0)]
        + [Model(name="o0", genome_dir=Path("x"), fitness=0.1)]
    )


def test_no_rotation_does_not_rebenchmark_survivors(tmp_path):
    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    cfg.benchmark.eval_rotation = False  # rotation off -> cached survivor scores reused
    ops = CountingOps(lambda name, sl: {"code": 0.6})
    _make(tmp_path, ops, cfg).run_generation(0, _pop())
    assert ops.benchmarked == [("o0", 0)]  # only the offspring


def test_rotation_rebenchmarks_survivors_on_current_slice(tmp_path):
    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    cfg.benchmark.eval_rotation = True
    cfg.benchmark.num_eval_slices = 3
    cfg.benchmark.eval_seed = 7

    from darwin.bench.rotation import held_out_slice

    slice0 = held_out_slice(0, 3, seed=7)
    ops = CountingOps(lambda name, sl: {"code": 0.55 if name == "s0" else 0.6})
    nxt = _make(tmp_path, ops, cfg).run_generation(0, _pop())

    names = {n for n, _ in ops.benchmarked}
    if slice0 != 0:  # the survivor's cached slice (0) differs from this gen's slice -> re-bench
        assert "s0" in names
        # the survivor was scored on the current slice, and its scores updated
        assert all(sl == slice0 for _, sl in ops.benchmarked)
        assert nxt.get("s0").scores == {"code": 0.55}
        assert nxt.get("s0").scored_slice == slice0
    # the offspring is always benchmarked on the current slice and tagged with it
    assert ("o0", slice0) in ops.benchmarked
    assert nxt.get("o0").scored_slice == slice0


def test_survivor_already_on_slice_is_not_rebenchmarked(tmp_path):
    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    cfg.benchmark.eval_rotation = True
    cfg.benchmark.num_eval_slices = 3
    cfg.benchmark.eval_seed = 7

    from darwin.bench.rotation import held_out_slice

    slice0 = held_out_slice(0, 3, seed=7)
    pop = Population(
        [Model(name="s0", genome_dir=Path("x"), fitness=0.9, scores={"code": 0.5},
               is_survivor=True, scored_slice=slice0)]  # already on this gen's slice
        + [Model(name="o0", genome_dir=Path("x"), fitness=0.1)]
    )
    ops = CountingOps(lambda name, sl: {"code": 0.6})
    _make(tmp_path, ops, cfg).run_generation(0, pop)
    assert "s0" not in {n for n, _ in ops.benchmarked}  # cached scores already on-slice


def test_scored_slice_round_trips_through_model_json():
    m = Model(name="s0", genome_dir=Path("x"), scores={"code": 0.5}, scored_slice=2)
    assert Model.from_dict(m.to_dict()).scored_slice == 2
