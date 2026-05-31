"""Resumable generation state persistence (ARCHITECTURE.md §2.3)."""

import pytest

from darwin.controller.state import (
    GenerationState,
    GenerationStateStore,
    OffspringState,
    PHASE_ORDER,
)


def make_offspring(name="model6", **kw) -> OffspringState:
    defaults = dict(
        name=name, parent_survivor="model1", mutator="model2", backend="claude", iteration=3
    )
    defaults.update(kw)
    return OffspringState(**defaults)


def test_phase_advance_is_monotonic():
    st = GenerationState(generation=0)
    assert st.phase == "spawned"
    st.advance_to("aggregated")
    assert st.phase == "aggregated"
    st.advance_to("spawned")  # backward no-op (idempotent resume)
    assert st.phase == "aggregated"
    assert st.at_least("offspring_done") is True
    assert st.at_least("culled") is False


def test_phase_order_is_the_2_3_sequence():
    assert PHASE_ORDER[0] == "spawned"
    assert PHASE_ORDER[-1] == "checkpoint"


def test_offspring_finetune_failed_property():
    o = make_offspring(finetune_status="finetune_failed")
    assert o.finetune_failed is True
    assert make_offspring(finetune_status="ok").finetune_failed is False


def test_state_roundtrips_through_store(tmp_path):
    store = GenerationStateStore(tmp_path / "runs")
    st = GenerationState(
        generation=2,
        phase="offspring_done",
        population_in={"models": [{"name": "model1", "genome_dir": "x"}]},
        offspring=[
            make_offspring(
                "model6", mutation_done=True, finetune_done=True, finetune_status="ok",
                adapter_path="a.bin", benchmark_done=True, scores={"code": 0.6}, cost_usd=2.0,
            ),
            make_offspring("model7", mutator=None, mutation_done=True, mutation_failed=True),
        ],
        survivors_after_cull=["model1", "model6"],
    )
    path = store.save(st)
    assert path.exists()

    loaded = store.load(2)
    assert loaded.generation == 2
    assert loaded.phase == "offspring_done"
    assert len(loaded.offspring) == 2
    o6 = loaded.offspring_by_name()["model6"]
    assert o6.scores == {"code": 0.6}
    assert o6.finetune_status == "ok"
    o7 = loaded.offspring_by_name()["model7"]
    assert o7.mutator is None and o7.mutation_failed is True
    assert loaded.survivors_after_cull == ["model1", "model6"]


def test_store_exists_and_missing(tmp_path):
    store = GenerationStateStore(tmp_path / "runs")
    assert store.exists(0) is False
    with pytest.raises(FileNotFoundError):
        store.load(0)
    store.save(GenerationState(generation=0))
    assert store.exists(0) is True


def test_latest_generation_tracks_resume_point(tmp_path):
    store = GenerationStateStore(tmp_path / "runs")
    assert store.latest_generation() is None
    store.save(GenerationState(generation=0))
    store.save(GenerationState(generation=1))
    store.save(GenerationState(generation=3))
    assert store.latest_generation() == 3
