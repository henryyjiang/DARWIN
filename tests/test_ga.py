"""Population model + GA selection/reproduction (ARCHITECTURE.md §3.2)."""

import random
from pathlib import Path

import pytest

from darwin.controller.ga import (
    OffspringPlan,
    pair_offspring,
    rank_models,
    select_survivors,
)
from darwin.controller.population import Model, Population


def model(name, fitness=None, **kw) -> Model:
    return Model(name=name, genome_dir=Path(f"models/{name}/genome"), fitness=fitness, **kw)


# ------------------------------------------------------------------ ranking & cull


def test_rank_orders_by_fitness_desc():
    ms = [model("a", 0.3), model("b", 0.9), model("c", 0.5)]
    assert [m.name for m in rank_models(ms)] == ["b", "c", "a"]


def test_rank_treats_none_and_floor_as_worst():
    ms = [model("a", 0.1), model("unscored", None), model("failed", float("-inf"))]
    ranked = [m.name for m in rank_models(ms)]
    assert ranked[0] == "a"
    assert set(ranked[1:]) == {"unscored", "failed"}  # both floor


def test_select_survivors_keeps_top_n():
    ms = [model(n, f) for n, f in [("a", 0.1), ("b", 0.9), ("c", 0.5), ("d", 0.7)]]
    survivors = select_survivors(ms, num_survivors=2)
    assert [m.name for m in survivors] == ["b", "d"]


def test_finetune_failed_floor_is_culled_first():
    ms = [
        model("good", 0.6),
        model("ok", 0.4),
        model("failed", float("-inf"), finetune_failed=True),
    ]
    survivors = select_survivors(ms, num_survivors=2)
    assert "failed" not in {m.name for m in survivors}


def test_select_survivors_handles_small_population():
    ms = [model("a", 0.5)]
    assert select_survivors(ms, num_survivors=5) == ms


def test_diversity_pick_reserves_a_slot_for_the_most_different():
    # fitness order: a(0.9) b(0.8) c(0.7) d(0.6). Greedy top-3 = a,b,c.
    # diversity_fn: distance keyed by a "family" tag; d is far from {a,b}, c is near.
    family = {"a": 0, "b": 0, "c": 0, "d": 9}
    ms = [model(n, f) for n, f in [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]]
    dist = lambda x, y: abs(family[x.name] - family[y.name])
    survivors = select_survivors(
        ms, num_survivors=3, diversity_pick=True, diversity_fn=dist
    )
    names = {m.name for m in survivors}
    assert {"a", "b"} <= names  # elites kept
    assert "d" in names and "c" not in names  # diversity slot took the far one


# ------------------------------------------------------------------ pairing (§3.2)


def test_pair_offspring_counts_and_constraints():
    survivors = [model(n, 0.5) for n in ["s1", "s2", "s3"]]
    rng = random.Random(0)
    plans = pair_offspring(survivors, num_offspring=5, rng=rng)
    assert len(plans) == 5
    for p in plans:
        assert p.parent_survivor in {"s1", "s2", "s3"}
        assert p.mutator in {"s1", "s2", "s3"}
        assert p.mutator != p.parent_survivor  # M != S


def test_pair_offspring_is_deterministic_under_seed():
    survivors = [model(n, 0.5) for n in ["s1", "s2", "s3"]]
    a = pair_offspring(survivors, 5, random.Random(42))
    b = pair_offspring(survivors, 5, random.Random(42))
    assert a == b


def test_pair_offspring_draws_with_replacement():
    # a single dominant pairing space; with replacement the same parent recurs across offspring
    survivors = [model(n, 0.5) for n in ["s1", "s2"]]
    plans = pair_offspring(survivors, 10, random.Random(1))
    parents = [p.parent_survivor for p in plans]
    assert len(set(parents)) <= 2 and len(parents) == 10  # drawn from the 2, repeats allowed


def test_single_survivor_falls_back_to_claude_mutator():
    survivors = [model("lonely", 0.5)]
    plans = pair_offspring(survivors, 3, random.Random(0))
    assert all(p.parent_survivor == "lonely" for p in plans)
    assert all(p.mutator is None for p in plans)  # no distinct M -> claude fallback (§3.2)


def test_pair_offspring_requires_a_survivor():
    with pytest.raises(ValueError):
        pair_offspring([], 5, random.Random(0))


# ------------------------------------------------------------------ population serialization


def test_population_roundtrips_through_dict():
    pop = Population(
        models=[
            model("a", 0.5, is_survivor=True, scores={"code": 0.5}),
            model("b", None, parent_survivor="a", mutator="c", backend="local"),
        ]
    )
    restored = Population.from_dict(pop.to_dict())
    assert restored.names() == ["a", "b"]
    assert restored.get("a").fitness == 0.5
    assert restored.get("a").scores == {"code": 0.5}
    assert restored.get("b").parent_survivor == "a"
    assert restored.get("b").adapter_path is None
    assert [m.name for m in restored.survivors()] == ["a"]
    assert [m.name for m in restored.offspring()] == ["b"]
