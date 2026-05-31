"""Tests for the memory store (ARCHITECTURE.md §7)."""

import pytest

from darwin.memory import IterationMemory, MemoryStore, GlobalMemory


def make_mem(model="model7", iteration=0, **overrides) -> IterationMemory:
    base = dict(
        model=model,
        iteration=iteration,
        generation=1,
        parent_survivor="model3",
        mutator=model,
        backend="claude",
        base_fitness=0.5,
        cost_usd=1.0,
        thesis=f"thesis {iteration}",
        changes=f"changes {iteration}",
        smoke_results="green",
        outcome=f"outcome {iteration}",
    )
    base.update(overrides)
    return IterationMemory(**base)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


def test_write_then_read_roundtrip(store):
    mem = make_mem(iteration=3)
    path = store.write_iteration(mem)
    assert path.exists()
    assert path == store.iter_path("model7", 3)
    assert store.read_iteration("model7", 3) == mem


def test_write_rejects_invalid(store):
    bad = make_mem(thesis="")  # empty body section
    with pytest.raises(Exception):
        store.write_iteration(bad)
    # nothing should have been written
    assert store.iteration_numbers("model7") == []


def test_iteration_numbers_sorted(store):
    for n in (2, 0, 10, 1):
        store.write_iteration(make_mem(iteration=n))
    assert store.iteration_numbers("model7") == [0, 1, 2, 10]


def test_recent_newest_first(store):
    for n in range(5):
        store.write_iteration(make_mem(iteration=n))
    recent = store.recent("model7", k=3)
    assert [m.iteration for m in recent] == [4, 3, 2]


def test_recent_empty_model(store):
    assert store.recent("ghost", k=3) == []


def test_search_ranks_by_relevance(store):
    store.write_iteration(make_mem(iteration=0, thesis="explore lora rank tuning"))
    store.write_iteration(
        make_mem(iteration=1, thesis="lora lora lora rank rank", changes="lora again")
    )
    store.write_iteration(make_mem(iteration=2, thesis="unrelated topic"))
    results = store.search("model7", "lora rank")
    # iter 1 has the most occurrences, iter 0 has some, iter 2 has none.
    assert [m.iteration for m in results] == [1, 0]


def test_search_empty_query(store):
    store.write_iteration(make_mem(iteration=0))
    assert store.search("model7", "   ") == []


def test_patch_iteration_sets_controller_fields(store):
    store.write_iteration(make_mem(iteration=4))
    patched = store.patch_iteration(
        "model7", 4, final_fitness=0.81, mutation_failed=True
    )
    assert patched.final_fitness == 0.81
    assert patched.mutation_failed is True
    # persisted to disk
    reloaded = store.read_iteration("model7", 4)
    assert reloaded.final_fitness == 0.81
    assert reloaded.mutation_failed is True


def test_patch_iteration_rejects_non_controller_field(store):
    store.write_iteration(make_mem(iteration=4))
    with pytest.raises(ValueError):
        store.patch_iteration("model7", 4, thesis="hijacked")


def test_read_missing_raises(store):
    with pytest.raises(FileNotFoundError):
        store.read_iteration("model7", 99)


def test_per_model_isolation(store):
    store.write_iteration(make_mem(model="model1", iteration=0))
    store.write_iteration(make_mem(model="model2", iteration=0))
    assert store.iteration_numbers("model1") == [0]
    assert store.iteration_numbers("model2") == [0]
    assert store.read_iteration("model1", 0).model == "model1"


# --- global memory ---

def test_global_roundtrip(store):
    gm = GlobalMemory(
        objectives="reach SOTA",
        whats_working="higher lora rank",
        todo="try alpha scaling",
        cost_ledger="$0.00",
    )
    store.write_global(gm)
    assert store.get_global() == gm


def test_global_missing_returns_empty(store):
    assert store.get_global() == GlobalMemory()
