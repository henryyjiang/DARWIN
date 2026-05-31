"""Container finetune + eval backends (ARCHITECTURE.md §5/§6/§8).

These exercise the on-host container execution path with a **fake `ContainerRunner`** that never
shells out to Docker: it inspects the `ContainerSpec` the backend built (mounts + env), maps the
in-container artifact path back to the host bind mount, and simulates the entrypoint by writing
the adapter / scores file there — exactly what the real `darwin-finetune` / `darwin-eval` images
do. So the spec-building, path mapping, and classification are all tested without Docker/GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from darwin.bench.job import BenchmarkError, BenchmarkJob, EvalContainerBenchmarkBackend
from darwin.finetune.backend import ContainerFinetuneBackend
from darwin.finetune.job import FinetuneJob
from darwin.sandbox import ContainerResult, ContainerSpec


def _host_path_for(spec: ContainerSpec, container_path: str) -> str | None:
    """Map an in-container path to its host bind-mount path (longest-prefix match)."""
    best: tuple[int, str] | None = None
    for m in spec.mounts:
        if container_path == m.container_path or container_path.startswith(m.container_path + "/"):
            rel = container_path[len(m.container_path):].lstrip("/")
            host = str(Path(m.host_path) / rel) if rel else m.host_path
            if best is None or len(m.container_path) > best[0]:
                best = (len(m.container_path), host)
    return best[1] if best else None


@dataclass
class FakeRunner:
    """A `ContainerRunner` that simulates the entrypoint writing files through the bind mounts."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    writes: dict[str, str] = field(default_factory=dict)  # env var -> file contents to write
    specs: list[ContainerSpec] = field(default_factory=list)

    def run(self, spec: ContainerSpec, *, dry_run: bool = False) -> ContainerResult:
        self.specs.append(spec)
        if self.exit_code == 0:
            for env_key, contents in self.writes.items():
                container_path = spec.env[env_key]
                host = _host_path_for(spec, container_path)
                assert host is not None, f"{container_path} not under any mount"
                Path(host).parent.mkdir(parents=True, exist_ok=True)
                Path(host).write_text(contents, encoding="utf-8")
        return ContainerResult(self.exit_code, self.stdout, self.stderr, ["docker", "run"])


# ------------------------------------------------------------------ finetune


def _finetune_job(tmp_path: Path) -> FinetuneJob:
    genome = tmp_path / "genome"
    genome.mkdir()
    return FinetuneJob(
        offspring_id="o0", model="o0", generation=0, genome_dir=genome,
        adapter_out=tmp_path / "adapter.bin", method="qlora_4bit",
        lora_rank=16, lora_alpha=32, gpu_rate_usd_per_h=1.0,
    )


def test_container_finetune_success_materializes_adapter(tmp_path):
    job = _finetune_job(tmp_path)
    runner = FakeRunner(writes={"DARWIN_ADAPTER_OUT": "weights"})
    backend = ContainerFinetuneBackend(runner=runner, gpus=2, clock=iter([0.0, 3600.0]).__next__)

    out = backend.run(job)

    assert out.succeeded is True
    assert out.adapter_path == job.adapter_out and job.adapter_out.exists()
    assert out.gpu_hours == pytest.approx(2.0)  # 1h wall-clock x 2 GPUs
    spec = runner.specs[0]
    assert spec.image == "darwin-finetune"
    # genome mounted ro, adapter dir mounted rw; env points at the in-container adapter path
    ro = {m.container_path: m.read_only for m in spec.mounts}
    assert ro["/work/genome"] is True and ro["/work/adapter"] is False
    assert spec.env["DARWIN_ADAPTER_OUT"] == "/work/adapter/adapter.bin"
    assert spec.resources.gpus == 2


def test_container_finetune_green_exit_but_no_adapter_is_recipe_failure(tmp_path):
    job = _finetune_job(tmp_path)
    runner = FakeRunner(exit_code=0)  # exits clean but writes nothing
    out = ContainerFinetuneBackend(runner=runner).run(job)
    assert out.succeeded is False and out.failure_mode == "no_adapter"


def test_container_finetune_oom_classified_from_log(tmp_path):
    job = _finetune_job(tmp_path)
    runner = FakeRunner(exit_code=1, stderr="torch.cuda.OutOfMemoryError: CUDA out of memory")
    out = ContainerFinetuneBackend(runner=runner).run(job)
    assert out.succeeded is False and out.failure_mode == "oom"


def test_container_finetune_passes_base_model_and_safe_mode(tmp_path):
    job = _finetune_job(tmp_path)
    runner = FakeRunner(writes={"DARWIN_ADAPTER_OUT": "w"})
    ContainerFinetuneBackend(runner=runner, base_model="Qwen/Qwen2.5-Coder-7B").run(
        job, safe_mode=True
    )
    env = runner.specs[0].env
    assert env["DARWIN_BASE_MODEL"] == "Qwen/Qwen2.5-Coder-7B"
    assert env["DARWIN_SAFE_MODE"] == "1"


# ------------------------------------------------------------------ eval


def _bench_job(tmp_path: Path) -> BenchmarkJob:
    adapter = tmp_path / "adapter.bin"
    adapter.write_text("w", encoding="utf-8")
    slice_dir = tmp_path / "slice"
    slice_dir.mkdir()
    return BenchmarkJob(
        offspring_id="o0", model="o0", generation=0, base_model="base",
        adapter_path=adapter, suite=["humaneval+", "gsm8k"], slice_id=2,
        eval_data_dir=slice_dir,
    )


def test_eval_container_runs_zero_egress_and_reads_back_scores(tmp_path):
    job = _bench_job(tmp_path)
    runner = FakeRunner(writes={"DARWIN_SCORES_OUT": json.dumps({"humaneval+": 0.5, "gsm8k": 0.3})})
    backend = EvalContainerBenchmarkBackend(runner=runner)

    res = backend.run(job)

    assert res.scores == {"humaneval+": 0.5, "gsm8k": 0.3}
    assert res.slice_id == 2
    spec = runner.specs[0]
    assert spec.image == "darwin-eval"
    assert spec.network == "none"  # zero egress invariant (§6.2/§8.3)
    assert spec.env["DARWIN_SUITE"] == "humaneval+,gsm8k"
    assert spec.env["DARWIN_EVAL_SLICE"] == "2"


def test_eval_container_nonzero_exit_raises(tmp_path):
    job = _bench_job(tmp_path)
    runner = FakeRunner(exit_code=1, stderr="harness crashed")
    with pytest.raises(BenchmarkError, match="exited"):
        EvalContainerBenchmarkBackend(runner=runner).run(job)


def test_eval_container_missing_scores_raises(tmp_path):
    job = _bench_job(tmp_path)
    runner = FakeRunner(exit_code=0)  # clean exit but no scores file written
    with pytest.raises(BenchmarkError, match="no scores"):
        EvalContainerBenchmarkBackend(runner=runner).run(job)
