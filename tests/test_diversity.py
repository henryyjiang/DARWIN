"""Genome code-distance + diversity-pick wiring (ARCHITECTURE.md §3.4)."""

import random

from darwin.config import DarwinConfig
from darwin.controller import (
    Controller,
    GenerationStateStore,
    Model,
    Population,
    genome_code_distance,
    jaccard_distance,
    read_genome_source,
)
from darwin.controller.ga import select_survivors
from darwin.cost import CostLedger
from darwin.memory import MemoryStore


# ------------------------------------------------------------------ distance


def test_jaccard_identical_is_zero():
    assert jaccard_distance("a b c d e", "a b c d e", n=2) == 0.0


def test_jaccard_disjoint_is_one():
    assert jaccard_distance("alpha beta gamma", "delta epsilon zeta", n=2) == 1.0


def test_jaccard_both_empty_is_zero():
    assert jaccard_distance("", "", n=3) == 0.0


def test_read_genome_source_concatenates_code(tmp_path):
    g = tmp_path / "genome"
    g.mkdir()
    (g / "recipe.py").write_text("lora_rank = 16\n", encoding="utf-8")
    (g / "data.json").write_text('{"mix": "the-stack"}', encoding="utf-8")
    (g / "model.bin").write_bytes(b"\x00\x01")  # non-source: skipped
    src = read_genome_source(g)
    assert "lora_rank" in src and "the-stack" in src
    assert "\x00" not in src


def test_genome_code_distance_between_models(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "r.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    (b_dir / "r.py").write_text("p = 9\nq = 8\nr = 7\n", encoding="utf-8")
    a = Model(name="a", genome_dir=a_dir)
    b = Model(name="b", genome_dir=b_dir)
    assert genome_code_distance(a, b, n=2) > 0.5


# ------------------------------------------------------------------ select_survivors uses it


def _m(name, fitness, src, tmp_path):
    d = tmp_path / name
    d.mkdir()
    (d / "r.py").write_text(src, encoding="utf-8")
    return Model(name=name, genome_dir=d, fitness=fitness)


def test_diversity_pick_prefers_different_genome(tmp_path):
    # Two high-fitness near-duplicate elites + one slightly-lower but very different candidate.
    e1 = _m("e1", 0.9, "aa bb cc dd ee ff", tmp_path)
    e2 = _m("e2", 0.85, "aa bb cc dd ee ff", tmp_path)  # duplicate of e1
    near = _m("near", 0.7, "aa bb cc dd ee ff gg", tmp_path)  # similar, higher fitness
    diff = _m("diff", 0.6, "zz yy xx ww vv uu", tmp_path)  # different, lower fitness

    # greedy top-3 would take e1, e2, near
    greedy = select_survivors([e1, e2, near, diff], 3)
    assert [m.name for m in greedy] == ["e1", "e2", "near"]

    # with diversity pick, the last slot goes to the most-different genome instead
    diverse = select_survivors(
        [e1, e2, near, diff], 3, diversity_pick=True, diversity_fn=genome_code_distance
    )
    assert [m.name for m in diverse[:2]] == ["e1", "e2"]
    assert diverse[2].name == "diff"


# ------------------------------------------------------------------ controller wiring


def test_controller_passes_diversity_fn_when_enabled(tmp_path, monkeypatch):
    cfg = DarwinConfig()
    cfg.ga.diversity_pick = True
    cfg.ga.num_survivors = 2

    captured = {}

    import darwin.controller.controller as controller_mod

    def fake_select(models, n, *, diversity_pick=False, diversity_fn=None):
        captured["diversity_pick"] = diversity_pick
        captured["has_fn"] = diversity_fn is not None
        return models[:n]

    monkeypatch.setattr(controller_mod, "select_survivors", fake_select)

    ctrl = Controller(
        config=cfg,
        store=MemoryStore(tmp_path / "store"),
        ledger=CostLedger(tmp_path / "cost.jsonl"),
        state_store=GenerationStateStore(tmp_path / "runs"),
        ops=object(),
        rng=random.Random(0),
    )
    pop = Population(
        [Model(name=f"m{i}", genome_dir=tmp_path / f"m{i}") for i in range(4)]
    )
    ctrl._load_or_spawn(0, pop)
    assert captured["diversity_pick"] is True
    assert captured["has_fn"] is True
