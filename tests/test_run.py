"""Top-level run assembly: config load, bootstrap, build_controller, one generation (§2/§9.1)."""

import sys
import textwrap
from pathlib import Path

import pytest

from darwin.config import DarwinConfig
from darwin.run import (
    apply_overrides,
    bootstrap_or_load_population,
    build_controller,
    load_run_spec,
)


# ------------------------------------------------------------------ config parsing


def test_apply_overrides_sets_nested_fields():
    cfg = apply_overrides(DarwinConfig(), {
        "run_name": "x",
        "ga": {"num_survivors": 3, "diversity_pick": True},
        "mutation": {"backend": "local", "mutation_window_h": 2.0},
        "cost": {"gen_budget_usd": 100.0},
        "bogus": {"ignored": 1},  # unknown subconfig ignored
    })
    assert cfg.run_name == "x"
    assert cfg.ga.num_survivors == 3 and cfg.ga.diversity_pick is True
    assert cfg.mutation.backend == "local" and cfg.mutation.mutation_window_h == 2.0
    assert cfg.cost.gen_budget_usd == 100.0


def _write_base_genome(d: Path) -> Path:
    base = d / "base_genome"
    base.mkdir(parents=True, exist_ok=True)
    (base / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (base / "smoke_test.py").write_text("import recipe, sys; sys.exit(0)\n", encoding="utf-8")
    return base


def _write_config(d: Path) -> Path:
    _write_base_genome(d)
    cfg = d / "run.yaml"
    cfg.write_text(textwrap.dedent("""
        generations: 1
        paths:
          base_genome: %s
        seed_scores:
          s0: {code: 0.5}
        config:
          ga: {num_survivors: 1, population_size: 2}
          benchmark: {suite: ["code"]}
    """ % (d / "base_genome").as_posix()), encoding="utf-8")
    return cfg


def test_load_run_spec_reads_paths_and_overrides(tmp_path):
    spec = load_run_spec(_write_config(tmp_path))
    assert spec.generations == 1
    assert spec.config.ga.num_survivors == 1
    assert spec.config.benchmark.suite == ["code"]
    assert spec.paths.base_genome.name == "base_genome"


# ------------------------------------------------------------------ bootstrap / resume


def test_bootstrap_population_seeds_disk(tmp_path):
    spec = load_run_spec(_write_config(tmp_path))
    pop = bootstrap_or_load_population(spec)
    # population_size 2, num_survivors 1 -> 1 survivor + 1 offspring slot
    assert len(pop.models) == 2
    assert pop.get("s0").is_survivor and pop.get("s0").scores == {"code": 0.5}
    assert (spec.paths.workspace / "s0" / "genome" / "recipe.py").exists()


# ------------------------------------------------------------------ end-to-end via build_controller


FT = ("import os, pathlib; p = pathlib.Path(os.environ['DARWIN_ADAPTER_OUT']); "
      "p.parent.mkdir(parents=True, exist_ok=True); p.write_text('adapter')")
EVAL = ("import os, json, pathlib; "
        "pathlib.Path(os.environ['DARWIN_SCORES_OUT']).write_text(json.dumps({'code': 0.8}))")


class GreenBackend:
    def run(self, ctx, deadline):
        (ctx.genome_dir / "recipe.py").write_text("OK = True\n# mutated\n", encoding="utf-8")
        assert ctx.checkpoint("mutated") is True
        ctx.write_memory(thesis="t", changes="c", smoke_results="green", outcome="o", cost_usd=0.0)


class FakeSynth:
    def synthesize(self, digest, current):
        from darwin.memory import GlobalMemory

        return GlobalMemory(objectives="seeded")


def test_main_assembly_runs_one_generation(tmp_path):
    spec = load_run_spec(_write_config(tmp_path))
    spec.smoke_command = [sys.executable, "smoke_test.py"]

    from darwin.bench import SubprocessBenchmarkBackend
    from darwin.finetune import SubprocessFinetuneBackend

    controller, store, ledger = build_controller(
        spec,
        mutation_backend_factory=lambda name, ctx: GreenBackend(),
        finetune_backend=SubprocessFinetuneBackend(command=[sys.executable, "-c", FT]),
        benchmark_backend=SubprocessBenchmarkBackend(command=[sys.executable, "-c", EVAL]),
        synthesizer=FakeSynth(),
    )
    population = bootstrap_or_load_population(spec)
    nxt = controller.run(spec.generations, population)

    # the offspring was cloned, mutated, finetuned, benchmarked, and scored
    o0 = nxt.get("o0")
    assert (spec.paths.workspace / "o0" / "genome" / ".git").exists()
    assert "# mutated" in (spec.paths.workspace / "o0" / "genome" / "recipe.py").read_text(encoding="utf-8")
    assert o0.scores == {"code": 0.8}
    assert o0.fitness == pytest.approx(1.6, abs=0.02)  # 0.8 / 0.5 baseline, minus tiny cost
    # global-memory pass ran, generation state persisted complete
    assert store.get_global().objectives == "seeded"
    assert controller.state_store.load(0).completed is True


def test_resume_returns_persisted_population(tmp_path):
    spec = load_run_spec(_write_config(tmp_path))
    spec.smoke_command = [sys.executable, "smoke_test.py"]
    from darwin.bench import SubprocessBenchmarkBackend
    from darwin.finetune import SubprocessFinetuneBackend

    controller, _s, _l = build_controller(
        spec, mutation_backend_factory=lambda name, ctx: GreenBackend(),
        finetune_backend=SubprocessFinetuneBackend(command=[sys.executable, "-c", FT]),
        benchmark_backend=SubprocessBenchmarkBackend(command=[sys.executable, "-c", EVAL]),
        synthesizer=FakeSynth(),
    )
    controller.run(1, bootstrap_or_load_population(spec))
    # a fresh spec sees the persisted gen-0 state and loads its population_out
    resumed = bootstrap_or_load_population(load_run_spec(_write_config(tmp_path)))
    assert {m.name for m in resumed.models} == {"s0", "o0"}
