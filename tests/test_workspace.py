"""Workspace bootstrap / move-back / drop-slot reconciliation (ARCHITECTURE.md §3.1 / §9.1)."""

from pathlib import Path

from darwin.controller import bootstrap_population, materialize_model, reset_slot
from darwin.controller.workspace import adapter_path, genome_dir


def _base_genome(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    (base / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (base / "smoke_test.py").write_text("import recipe, sys; sys.exit(0)\n", encoding="utf-8")
    return base


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_creates_survivors_and_slots(tmp_path):
    ws = tmp_path / "models"
    pop = bootstrap_population(
        ws, _base_genome(tmp_path),
        num_survivors=5, num_offspring=5,
        survivor_scores={"s0": {"code": 0.5}}, survivor_fitness={"s0": 0.5},
    )
    assert len(pop.models) == 10
    survivors = pop.survivors()
    assert len(survivors) == 5
    # each survivor got a copy of the base genome + carries its cached score
    for s in survivors:
        assert (genome_dir(ws, s.name) / "recipe.py").exists()
    assert pop.get("s0").scores == {"code": 0.5}
    assert pop.get("s0").fitness == 0.5
    # offspring slots exist as dirs but have no genome yet (controller clones at SPAWN)
    assert not (genome_dir(ws, "o0")).exists()
    assert pop.get("o0").is_survivor is False


def test_bootstrap_missing_base_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        bootstrap_population(tmp_path / "models", tmp_path / "nope")


# ------------------------------------------------------------------ reset_slot (drop step)


def test_reset_slot_wipes_genome_and_adapter(tmp_path):
    ws = tmp_path / "models"
    g = genome_dir(ws, "o0")
    g.mkdir(parents=True)
    (g / "recipe.py").write_text("x = 1\n", encoding="utf-8")
    adapter_path(ws, "o0").write_text("adapter", encoding="utf-8")

    reset_slot(ws, "o0")
    assert not g.exists()
    assert not adapter_path(ws, "o0").exists()


def test_reset_slot_is_safe_when_absent(tmp_path):
    reset_slot(tmp_path / "models", "ghost")  # no error


# ------------------------------------------------------------------ materialize (move-back)


def test_materialize_copies_results_back(tmp_path):
    ws = tmp_path / "models"
    # an offspring that ran in a container workdir
    work = tmp_path / "container" / "o0"
    (work / "genome").mkdir(parents=True)
    (work / "genome" / "recipe.py").write_text("OK = True\n# mutated\n", encoding="utf-8")
    (work / "adapter.bin").write_text("trained-adapter", encoding="utf-8")
    (work / "memory").mkdir()
    (work / "memory" / "iter_0.md").write_text("notes", encoding="utf-8")

    materialize_model(
        ws, "o0",
        genome_src=work / "genome",
        adapter_src=work / "adapter.bin",
        memory_src=work / "memory",
    )
    assert "# mutated" in (genome_dir(ws, "o0") / "recipe.py").read_text(encoding="utf-8")
    assert adapter_path(ws, "o0").read_text(encoding="utf-8") == "trained-adapter"
    assert (ws / "o0" / "memory" / "iter_0.md").exists()


def test_materialize_noop_when_src_is_dst(tmp_path):
    ws = tmp_path / "models"
    g = genome_dir(ws, "o0")
    g.mkdir(parents=True)
    (g / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    # genome_src already IS the model's dir -> must not delete/copy onto itself
    materialize_model(ws, "o0", genome_src=g)
    assert (g / "recipe.py").exists()


# ------------------------------------------------------------------ controller wiring


def test_controller_resets_offspring_slots_once_per_generation(tmp_path):
    import random

    from darwin.config import DarwinConfig
    from darwin.controller import Controller, GenerationStateStore, Model, Population
    from darwin.cost import CostLedger
    from darwin.memory import MemoryStore

    reset_calls: list[str] = []

    class RecordingOps:
        def reset_offspring_slot(self, name):
            reset_calls.append(name)

        def spawn(self, **k):
            return Model(name=k["offspring"].name, genome_dir=Path("x"))

        def mutate(self, **k):
            from darwin.controller import MutateOutcome

            return MutateOutcome(final_commit="c", mutation_failed=False)

        def finetune(self, **k):
            from darwin.controller import FinetuneOutcomeView

            return FinetuneOutcomeView("ok", Path("a"), 0.0)

        def benchmark(self, **k):
            return {"code": 0.6}

    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    pop = Population(
        [Model(name="s0", genome_dir=Path("x"), fitness=0.9, scores={"code": 0.5}, is_survivor=True)]
        + [Model(name=f"o{i}", genome_dir=Path("x"), fitness=0.1) for i in range(3)]
    )

    class FakeSynth:
        def synthesize(self, digest, current):
            from darwin.memory import GlobalMemory

            return GlobalMemory(objectives="x")

    ctrl = Controller(
        config=cfg, store=MemoryStore(tmp_path / "s"), ledger=CostLedger(tmp_path / "c.jsonl"),
        state_store=GenerationStateStore(tmp_path / "runs"), ops=RecordingOps(),
        synthesizer=FakeSynth(), rng=random.Random(0),
    )
    ctrl._load_or_spawn(0, pop)  # fresh state -> resets the 3 offspring slots
    assert sorted(reset_calls) == ["o0", "o1", "o2"]

    # resuming the same generation must NOT reset again (work is preserved)
    reset_calls.clear()
    ctrl._load_or_spawn(0, pop)
    assert reset_calls == []
