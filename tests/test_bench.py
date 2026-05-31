"""Benchmark runner, slice rotation, and fitness reduction (ARCHITECTURE.md §6)."""

import json
import sys
from pathlib import Path

import pytest

from darwin.config import FitnessConfig
from darwin.bench import (
    BenchmarkError,
    BenchmarkJob,
    SubprocessBenchmarkBackend,
    held_out_slice,
    normalize_scores,
    reduce_fitness,
    resolve_weights,
    rotation_schedule,
    survivor_baseline,
)


# ------------------------------------------------------------------ rotation (§6.4)


def test_rotation_is_deterministic_and_reproducible():
    a = held_out_slice(3, 5, seed=42)
    b = held_out_slice(3, 5, seed=42)
    assert a == b
    assert 0 <= a < 5


def test_rotation_covers_every_slice_within_a_cycle():
    # over one full cycle of num_slices generations, each slice is used exactly once
    sched = rotation_schedule(5, 5, seed=7)
    assert sorted(sched) == [0, 1, 2, 3, 4]


def test_rotation_seed_changes_order():
    assert rotation_schedule(5, 5, seed=1) != rotation_schedule(5, 5, seed=2)


def test_rotation_rejects_bad_args():
    with pytest.raises(ValueError):
        held_out_slice(0, 0)
    with pytest.raises(ValueError):
        held_out_slice(-1, 5)


# ------------------------------------------------------------------ fitness (§6.3)


def test_survivor_baseline_is_per_benchmark_mean():
    base = survivor_baseline(
        [{"code": 0.4, "math": 0.6}, {"code": 0.6, "math": 0.8}]
    )
    assert base["code"] == pytest.approx(0.5)
    assert base["math"] == pytest.approx(0.7)


def test_normalize_against_baseline_centers_on_one():
    norm = normalize_scores({"code": 0.5, "math": 0.7}, {"code": 0.5, "math": 0.7})
    assert norm["code"] == pytest.approx(1.0)
    assert norm["math"] == pytest.approx(1.0)


def test_normalize_zero_baseline_falls_back_to_raw():
    norm = normalize_scores({"code": 0.5}, {})  # gen 0, no survivors
    assert norm["code"] == 0.5


def test_resolve_weights_uniform_when_unconfigured():
    w = resolve_weights(["a", "b", "c"], {})
    assert w == {"a": pytest.approx(1 / 3), "b": pytest.approx(1 / 3), "c": pytest.approx(1 / 3)}


def test_resolve_weights_renormalizes_configured_subset():
    w = resolve_weights(["a", "b"], {"a": 3.0, "b": 1.0, "c": 99.0})
    assert w["a"] == pytest.approx(0.75)
    assert w["b"] == pytest.approx(0.25)


def test_reduce_fitness_improvement_over_survivors():
    cfg = FitnessConfig()  # lambda_cost 0.05, lambda_penalty 0.5, lambda_failed 0.1
    # offspring beats the survivor baseline by 20% on both -> normalized 1.2
    fit = reduce_fitness(
        scores={"code": 0.6, "math": 0.84},
        baseline={"code": 0.5, "math": 0.7},
        cost_usd=0.0,
        config=cfg,
    )
    assert fit == pytest.approx(1.2)


def test_reduce_fitness_applies_penalties():
    cfg = FitnessConfig()
    fit = reduce_fitness(
        scores={"code": 0.5},
        baseline={"code": 0.5},  # normalized 1.0
        cost_usd=4.0,            # -0.05*4 = -0.2
        antigaming_flags=1,      # -0.5
        mutation_failed=True,    # -0.1
        config=cfg,
    )
    assert fit == pytest.approx(1.0 - 0.2 - 0.5 - 0.1)


def test_reduce_fitness_finetune_failed_is_floor():
    cfg = FitnessConfig()
    fit = reduce_fitness(
        scores={"code": 9.9}, baseline={"code": 0.5}, finetune_failed=True, config=cfg
    )
    assert fit == cfg.finetune_failed_fitness == float("-inf")


# ------------------------------------------------------------------ subprocess backend (§6.2)


def make_job(tmp_path: Path, **overrides) -> BenchmarkJob:
    adapter = tmp_path / "out" / "adapter.bin"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_text("adapter")
    defaults = dict(
        offspring_id="7",
        model="model7",
        generation=2,
        base_model="qwen2.5-coder-32b",
        adapter_path=adapter,
        suite=["code", "math"],
        slice_id=3,
    )
    defaults.update(overrides)
    return BenchmarkJob(**defaults)


def write_eval_script(genome_dir: Path, body: str) -> Path:
    script = genome_dir / "eval.py"
    script.write_text(body, encoding="utf-8")
    return script


def test_subprocess_backend_reads_scores_file(tmp_path):
    job = make_job(tmp_path)
    write_eval_script(
        job.adapter_path.parent,
        "import os, json, pathlib\n"
        "assert os.environ['DARWIN_BASE_MODEL'] == 'qwen2.5-coder-32b'\n"
        "assert os.environ['DARWIN_EVAL_SLICE'] == '3'\n"
        "assert os.environ['DARWIN_SUITE'] == 'code,math'\n"
        "pathlib.Path(os.environ['DARWIN_SCORES_OUT']).write_text(json.dumps({'code':0.7,'math':0.5}))\n",
    )
    backend = SubprocessBenchmarkBackend(command=[sys.executable, "eval.py"])
    result = backend.run(job)
    assert result.scores == {"code": 0.7, "math": 0.5}
    assert result.slice_id == 3
    assert result.offspring_id == "7"


def test_subprocess_backend_raises_on_nonzero_exit(tmp_path):
    job = make_job(tmp_path)
    write_eval_script(job.adapter_path.parent, "import sys\nsys.exit(1)\n")
    backend = SubprocessBenchmarkBackend(command=[sys.executable, "eval.py"])
    with pytest.raises(BenchmarkError):
        backend.run(job)


def test_subprocess_backend_raises_when_no_scores_file(tmp_path):
    job = make_job(tmp_path)
    write_eval_script(job.adapter_path.parent, "print('ran but wrote nothing')\n")
    backend = SubprocessBenchmarkBackend(command=[sys.executable, "eval.py"])
    with pytest.raises(BenchmarkError):
        backend.run(job)


def test_end_to_end_bench_to_fitness(tmp_path):
    """Benchmark an offspring, then reduce its scores vs. a survivor baseline -> fitness."""
    job = make_job(tmp_path, suite=["code"])
    write_eval_script(
        job.adapter_path.parent,
        "import os, json, pathlib\n"
        "pathlib.Path(os.environ['DARWIN_SCORES_OUT']).write_text(json.dumps({'code':0.6}))\n",
    )
    result = SubprocessBenchmarkBackend(command=[sys.executable, "eval.py"]).run(job)
    baseline = survivor_baseline([{"code": 0.5}, {"code": 0.5}])
    fit = reduce_fitness(scores=result.scores, baseline=baseline, config=FitnessConfig())
    assert fit == pytest.approx(1.2)  # 0.6 / 0.5
