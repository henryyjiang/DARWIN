"""Finetune pipeline core: subprocess backend + runner failure policy (ARCHITECTURE.md §5.3)."""

import sys
from pathlib import Path

import pytest

from darwin.cost import CostLedger
from darwin.finetune import (
    FinetuneJob,
    FinetuneOutcome,
    SubprocessFinetuneBackend,
    run_finetune_job,
)


# ------------------------------------------------------------------ subprocess backend


def make_job(tmp_path: Path, **overrides) -> FinetuneJob:
    genome = tmp_path / "genome"
    genome.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        offspring_id="7",
        model="model7",
        generation=3,
        genome_dir=genome,
        adapter_out=tmp_path / "out" / "adapter.bin",
        gpu_rate_usd_per_h=2.0,
    )
    defaults.update(overrides)
    return FinetuneJob(**defaults)


def write_finetune_script(genome: Path, body: str) -> None:
    (genome / "finetune.py").write_text(body, encoding="utf-8")


def subprocess_backend() -> SubprocessFinetuneBackend:
    return SubprocessFinetuneBackend(command=[sys.executable, "finetune.py"])


def test_subprocess_success_materializes_adapter(tmp_path):
    job = make_job(tmp_path)
    write_finetune_script(
        job.genome_dir,
        "import os, pathlib\n"
        "p = pathlib.Path(os.environ['DARWIN_ADAPTER_OUT'])\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text('adapter')\n",
    )
    outcome = subprocess_backend().run(job)
    assert outcome.succeeded is True
    assert outcome.adapter_path == job.adapter_out
    assert job.adapter_out.read_text() == "adapter"
    assert outcome.gpu_hours >= 0
    assert outcome.failure_mode is None


def test_subprocess_oom_detected_from_log(tmp_path):
    job = make_job(tmp_path)
    write_finetune_script(
        job.genome_dir, "import sys\nprint('RuntimeError: CUDA out of memory')\nsys.exit(1)\n"
    )
    outcome = subprocess_backend().run(job)
    assert outcome.succeeded is False
    assert outcome.failure_mode == "oom"


def test_subprocess_nan_detected_from_log(tmp_path):
    job = make_job(tmp_path)
    write_finetune_script(
        job.genome_dir, "import sys\nprint('step 3: loss is nan')\nsys.exit(1)\n"
    )
    assert subprocess_backend().run(job).failure_mode == "nan_loss"


def test_subprocess_nonzero_without_known_marker(tmp_path):
    job = make_job(tmp_path)
    write_finetune_script(job.genome_dir, "import sys\nsys.exit(2)\n")
    assert subprocess_backend().run(job).failure_mode == "nonzero_exit"


def test_subprocess_green_but_no_adapter_is_recipe_failure(tmp_path):
    job = make_job(tmp_path)
    write_finetune_script(job.genome_dir, "print('done, but wrote no adapter')\n")
    outcome = subprocess_backend().run(job)
    assert outcome.succeeded is False
    assert outcome.failure_mode == "no_adapter"


def test_subprocess_passes_lora_config_via_env(tmp_path):
    job = make_job(tmp_path, lora_rank=8, lora_alpha=64)
    write_finetune_script(
        job.genome_dir,
        "import os, pathlib\n"
        "assert os.environ['DARWIN_LORA_RANK'] == '8'\n"
        "assert os.environ['DARWIN_LORA_ALPHA'] == '64'\n"
        "assert os.environ['DARWIN_SAFE_MODE'] == '0'\n"
        "pathlib.Path(os.environ['DARWIN_ADAPTER_OUT']).write_text('a')\n",
    )
    job.adapter_out.parent.mkdir(parents=True, exist_ok=True)
    assert subprocess_backend().run(job).succeeded is True


# ------------------------------------------------------------------ runner failure policy


class ScriptedBackend:
    """Yields a pre-scripted sequence of outcomes; records the safe_mode passed each call."""

    def __init__(self, outcomes: list[FinetuneOutcome]):
        self._outcomes = list(outcomes)
        self.safe_mode_calls: list[bool] = []

    def run(self, job, *, safe_mode: bool = False) -> FinetuneOutcome:
        self.safe_mode_calls.append(safe_mode)
        return self._outcomes.pop(0)


def ok(gpu=1.0):
    return FinetuneOutcome(True, gpu, adapter_path=Path("adapter.bin"))


def fail(mode, gpu=1.0):
    return FinetuneOutcome(False, gpu, failure_mode=mode)


def test_runner_success_records_cost(tmp_path):
    job = make_job(tmp_path, gpu_rate_usd_per_h=2.0)
    led = CostLedger(tmp_path / "cost.jsonl")
    res = run_finetune_job(job, ScriptedBackend([ok(gpu=1.5)]), led)
    assert res.status == "ok"
    assert res.attempts == 1
    assert res.gpu_hours == 1.5
    assert res.cost_usd == pytest.approx(3.0)  # 1.5h * $2/h
    assert led.total(3) == pytest.approx(3.0)
    assert led.entries()[0].kind == "finetune"


def test_runner_oom_retry_then_success(tmp_path):
    job = make_job(tmp_path, gpu_rate_usd_per_h=2.0)
    led = CostLedger(tmp_path / "cost.jsonl")
    backend = ScriptedBackend([fail("oom", gpu=1.0), ok(gpu=0.5)])
    res = run_finetune_job(job, backend, led)
    assert res.status == "ok"
    assert res.attempts == 2
    assert backend.safe_mode_calls == [False, True]  # retry was in safe mode
    assert res.gpu_hours == pytest.approx(1.5)
    assert res.cost_usd == pytest.approx(3.0)  # both attempts billed
    assert led.total(3) == pytest.approx(3.0)


def test_runner_oom_twice_is_finetune_failed(tmp_path):
    job = make_job(tmp_path)
    backend = ScriptedBackend([fail("oom"), fail("oom")])
    res = run_finetune_job(job, backend)
    assert res.status == "finetune_failed"
    assert res.failure_mode == "oom"
    assert res.attempts == 2
    assert res.finetune_failed is True
    assert res.adapter_path is None


def test_runner_infra_failure_is_not_recipe_failure(tmp_path):
    job = make_job(tmp_path)
    res = run_finetune_job(job, ScriptedBackend([fail("infra")]))
    assert res.status == "infra_failed"
    assert res.finetune_failed is False  # controller resumes; recipe not penalized


def test_runner_nan_fails_immediately_no_retry(tmp_path):
    job = make_job(tmp_path)
    backend = ScriptedBackend([fail("nan_loss")])
    res = run_finetune_job(job, backend)
    assert res.status == "finetune_failed"
    assert res.attempts == 1  # nan is not retried
    assert backend.safe_mode_calls == [False]


def test_runner_cost_cap_kills_runaway(tmp_path):
    # cap below the first attempt's cost -> killed after attempt 1, no OOM retry
    job = make_job(tmp_path, gpu_rate_usd_per_h=10.0, per_job_cap_usd=5.0)
    backend = ScriptedBackend([fail("oom", gpu=1.0)])  # 1h * $10 = $10 >= $5 cap
    res = run_finetune_job(job, backend)
    assert res.status == "finetune_failed"
    assert res.failure_mode == "cost_cap"
    assert res.attempts == 1  # cost cap pre-empts the OOM retry
