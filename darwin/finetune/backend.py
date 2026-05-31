"""Finetune backends (ARCHITECTURE.md §5.3).

Two implementations behind one `FinetuneBackend.run(...)` contract:

- `SubprocessFinetuneBackend` — runs the genome's declared finetune entrypoint as a
  subprocess in the genome dir, passing the LoRA config + adapter-output path via environment
  variables, and classifies the run from its exit code + log (OOM / NaN / non-zero / missing
  adapter). This is the CPU-runnable, fully testable core; on a single GPU it is exactly the
  job the §8.5 `darwin-finetune` image runs. GPU-hours are taken from wall-clock (one GPU).
- `LambdaFinetuneBackend` — the live path that provisions a Lambda GPU, runs the same
  entrypoint in the `darwin-finetune` image, and reports real GPU-hours. Scaffolded here;
  needs the Lambda API + image (deferred, like Phase 2's live container run).

`safe_mode` is the single OOM-retry lever (§5.3): the runner re-invokes with `safe_mode=True`
and the backend applies its own memory-frugal settings (smaller micro-batch, grad
checkpointing, QLoRA) — the backend owns the actual training config, not the runner.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from darwin.finetune.job import FinetuneJob, FinetuneOutcome

if TYPE_CHECKING:
    from darwin.finetune.lambda_api import LambdaClient, LambdaInstance
    from darwin.sandbox import ContainerRunner

# Default substrings that mark a recipe-level failure in a training log (§5.3). Case-insensitive.
_OOM_PATTERNS = ("out of memory", "cuda out of memory", "oom", "outofmemoryerror")
_NAN_PATTERNS = ("nan loss", "loss is nan", "inf loss", "loss became inf", "non-finite loss")


def finetune_env(job: FinetuneJob, adapter_out: str, *, safe_mode: bool) -> dict[str, str]:
    """The `DARWIN_*` env every finetune entrypoint reads (§5.3), shared across backends.

    `adapter_out` is given explicitly because the subprocess backend writes to the host path
    while the container backend writes to the in-container mount path — everything else is the
    same contract (the genome's entrypoint parses these, `finetune/entrypoint.py`)."""
    return {
        "DARWIN_ADAPTER_OUT": adapter_out,
        "DARWIN_LORA_RANK": str(job.lora_rank),
        "DARWIN_LORA_ALPHA": str(job.lora_alpha),
        "DARWIN_FINETUNE_METHOD": job.method,
        "DARWIN_SAFE_MODE": "1" if safe_mode else "0",
    }


def classify_finetune_outcome(
    *,
    exit_code: int,
    log: str,
    adapter_out: Path,
    gpu_hours: float,
    timed_out: bool = False,
    oom_patterns: tuple[str, ...] = _OOM_PATTERNS,
    nan_patterns: tuple[str, ...] = _NAN_PATTERNS,
) -> FinetuneOutcome:
    """Map an entrypoint run (exit code + log + did-the-adapter-materialize) to a `FinetuneOutcome`.

    Shared by every backend that runs the genome's entrypoint (subprocess / container) so the
    §5.3 failure taxonomy is identical regardless of where the job ran: timeout, OOM, NaN-loss,
    other non-zero exit (all recipe-level), and the green-exit-but-no-adapter case.
    """
    if timed_out:
        return FinetuneOutcome(False, gpu_hours, failure_mode="timeout", log=log)
    if exit_code == 0:
        if adapter_out.exists():
            return FinetuneOutcome(True, gpu_hours, adapter_path=adapter_out, log=log)
        # green exit but no adapter materialized -> recipe failure (§4.4.1 step 4 analogue)
        return FinetuneOutcome(False, gpu_hours, failure_mode="no_adapter", log=log)
    low = log.lower()
    if any(p in low for p in oom_patterns):
        mode = "oom"
    elif any(p in low for p in nan_patterns):
        mode = "nan_loss"
    else:
        mode = "nonzero_exit"
    return FinetuneOutcome(False, gpu_hours, failure_mode=mode, log=log)


class FinetuneBackend(Protocol):
    """Runs one finetune attempt for an offspring's genome."""

    def run(self, job: FinetuneJob, *, safe_mode: bool = False) -> FinetuneOutcome: ...


@dataclass
class SubprocessFinetuneBackend:
    """Runs the genome's finetune entrypoint as a subprocess; classifies from exit + log."""

    command: list[str]
    timeout_s: float = 3600.0
    oom_patterns: tuple[str, ...] = _OOM_PATTERNS
    nan_patterns: tuple[str, ...] = _NAN_PATTERNS
    env: dict[str, str] | None = None

    def run(self, job: FinetuneJob, *, safe_mode: bool = False) -> FinetuneOutcome:
        env = dict(os.environ if self.env is None else self.env)
        env.update(finetune_env(job, str(job.adapter_out), safe_mode=safe_mode))
        job.adapter_out.parent.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        timed_out = False
        exit_code = -1
        try:
            proc = subprocess.run(
                self.command,
                cwd=str(job.genome_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
            exit_code = proc.returncode
            log = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            log = f"finetune timed out after {self.timeout_s}s\n{exc.output or ''}"
        gpu_hours = (time.monotonic() - start) / 3600.0

        return classify_finetune_outcome(
            exit_code=exit_code,
            log=log,
            adapter_out=job.adapter_out,
            gpu_hours=gpu_hours,
            timed_out=timed_out,
            oom_patterns=self.oom_patterns,
            nan_patterns=self.nan_patterns,
        )


class LambdaJobRunner(Protocol):
    """Runs the finetune job on a provisioned instance and returns its raw outcome (§5.3).

    This is the one irreducibly-live seam: it SSHes to `instance.ip`, syncs the green genome,
    runs the genome's entrypoint in the `darwin-finetune` image, and fetches the adapter back to
    `job.adapter_out`. Injected so `LambdaFinetuneBackend`'s provision->run->terminate
    orchestration is testable with a fake; the default raises with the live-path note.
    """

    def __call__(
        self, instance: "LambdaInstance", job: FinetuneJob, *, safe_mode: bool
    ) -> FinetuneOutcome: ...


def _deferred_job_runner(instance, job, *, safe_mode):  # pragma: no cover - live path
    raise NotImplementedError(
        "LambdaFinetuneBackend needs a job_runner that SSHes to the instance, runs the "
        "darwin-finetune image, and fetches the adapter. Inject one; the API/lifecycle "
        "orchestration around it is implemented and tested."
    )


@dataclass
class LambdaFinetuneBackend:
    """Live Lambda Labs GPU finetune backend (§5.3).

    Orchestration (implemented + tested via injected fakes): launch a GPU instance via the
    Lambda API, poll until active, run the job on it (the injected `job_runner`), and **always
    terminate** the instance afterward (cost guard — a leaked instance bills indefinitely). The
    instance's active wall-clock x `num_gpus` is reported as GPU-hours when the runner doesn't
    measure it itself. Any Lambda API / provisioning failure surfaces as `failure_mode="infra"`
    so the runner classifies it `infra_failed` (resume/re-provision, §5.3), never penalizing the
    recipe.

    The only live piece is `job_runner` (SSH + image run + adapter fetch), injected so everything
    around it is testable; the default raises with the live-path note.
    """

    client: "LambdaClient"
    instance_type: str = "gpu_1x_a100"
    region: str = "us-east-1"
    ssh_key_names: tuple[str, ...] = ()
    num_gpus: int = 1
    job_runner: LambdaJobRunner = _deferred_job_runner
    wait_timeout_s: float = 1200.0
    clock: Callable[[], float] | None = None
    # GPU catalog for runtime sizing; when a job carries a RunSize the instance + GPU count are
    # chosen from this instead of the static `instance_type`/`num_gpus` defaults (§5.3 scaling).
    catalog: tuple = ()

    def _allocation(self, job: FinetuneJob) -> tuple[str, int]:
        """Pick (instance_type, num_gpus): dynamic from the run size, else the static default."""
        if job.run_size is None:
            return self.instance_type, self.num_gpus
        from darwin.finetune.sizing import LAMBDA_CATALOG, plan_instance

        plan = plan_instance(job.run_size, self.catalog or LAMBDA_CATALOG)
        return plan.instance_type, plan.num_gpus

    def run(self, job: FinetuneJob, *, safe_mode: bool = False) -> FinetuneOutcome:
        from darwin.finetune.lambda_api import LambdaApiError, wait_until_active

        clock = self.clock or time.monotonic
        instance_type, num_gpus = self._allocation(job)  # runtime GPU allocation (§5.3)
        instance_ids: list[str] = []
        start = clock()
        try:
            instance_ids = self.client.launch(
                instance_type=instance_type,
                region=self.region,
                ssh_key_names=list(self.ssh_key_names),
                name=f"darwin-{job.offspring_id}",
            )
            if not instance_ids:
                return FinetuneOutcome(False, 0.0, failure_mode="infra",
                                       log="Lambda launch returned no instance ids")
            instance = wait_until_active(
                self.client, instance_ids[0], timeout_s=self.wait_timeout_s, clock=self.clock
            )
        except LambdaApiError as exc:
            gpu_hours = (clock() - start) / 3600.0 * num_gpus
            self._safe_terminate(instance_ids)
            return FinetuneOutcome(False, gpu_hours, failure_mode="infra", log=str(exc))

        try:
            outcome = self.job_runner(instance, job, safe_mode=safe_mode)
        finally:
            self._safe_terminate(instance_ids)  # always terminate (cost guard, §5.4)

        if outcome.gpu_hours <= 0:  # bill active wall-clock x GPUs if the runner didn't
            gpu_hours = (clock() - start) / 3600.0 * num_gpus
            outcome = FinetuneOutcome(
                outcome.succeeded, gpu_hours, adapter_path=outcome.adapter_path,
                failure_mode=outcome.failure_mode, log=outcome.log,
            )
        return outcome

    def _safe_terminate(self, instance_ids: list[str]) -> None:
        if not instance_ids:
            return
        try:
            self.client.terminate(instance_ids)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


@dataclass
class ContainerFinetuneBackend:
    """Runs the finetune job inside the `darwin-finetune` image on the local Docker host (§5/§8).

    The on-host counterpart to `LambdaFinetuneBackend`: instead of provisioning a remote GPU it
    launches the §8.5 finetune container (genome mounted **ro**, the adapter-out dir mounted
    **rw**, GPUs, whitelist egress for HF Hub) via the injected `ContainerRunner` and classifies
    the run from its exit code + log with the *same* §5.3 taxonomy as the subprocess backend
    (`classify_finetune_outcome`). The genome's entrypoint is the image `CMD` by default
    (`python3 -m darwin.finetune.entrypoint`); pass `command` to override. GPU-hours are billed
    from wall-clock × GPU count.

    Path mapping: the adapter-out *directory* (`job.adapter_out.parent`) is bind-mounted rw at
    the canonical `/work/adapter`, and the entrypoint writes to `DARWIN_ADAPTER_OUT` =
    `/work/adapter/<filename>`, so the artifact lands back on the host at `job.adapter_out`.
    """

    runner: "ContainerRunner"
    image: str = "darwin-finetune"
    # default: use the image CMD (the reference entrypoint, python3 -m darwin.finetune.entrypoint)
    command: list[str] = field(default_factory=list)
    gpus: int = 1
    memory: str = "64g"
    base_model: str | None = None  # set DARWIN_BASE_MODEL when given (else the entrypoint default)
    env: dict[str, str] | None = None
    clock: Callable[[], float] | None = None
    oom_patterns: tuple[str, ...] = _OOM_PATTERNS
    nan_patterns: tuple[str, ...] = _NAN_PATTERNS

    def _num_gpus(self, job: FinetuneJob) -> int:
        """GPU count: from the run size (param-scaling, §5.3) when present, else the static default."""
        if job.run_size is None:
            return self.gpus
        from darwin.finetune.sizing import LAMBDA_CATALOG, plan_instance

        return plan_instance(job.run_size, LAMBDA_CATALOG).num_gpus

    def run(self, job: FinetuneJob, *, safe_mode: bool = False) -> FinetuneOutcome:
        from darwin.sandbox import ADAPTER_PATH, finetune_container

        clock = self.clock or time.monotonic
        num_gpus = self._num_gpus(job)
        job.adapter_out.parent.mkdir(parents=True, exist_ok=True)

        container_adapter_out = f"{ADAPTER_PATH}/{job.adapter_out.name}"
        env = dict(self.env or {})
        env.update(finetune_env(job, container_adapter_out, safe_mode=safe_mode))
        if self.base_model is not None:
            env["DARWIN_BASE_MODEL"] = self.base_model

        spec = finetune_container(
            offspring_id=job.offspring_id,
            genome_host=str(job.genome_dir),
            adapter_out_host=str(job.adapter_out.parent),
            command=list(self.command),
            env=env,
            gpus=num_gpus,
            memory=self.memory,
        )
        spec.image = self.image

        start = clock()
        result = self.runner.run(spec)
        gpu_hours = (clock() - start) / 3600.0 * num_gpus

        return classify_finetune_outcome(
            exit_code=result.exit_code,
            log=(result.stdout or "") + (result.stderr or ""),
            adapter_out=job.adapter_out,
            gpu_hours=gpu_hours,
            oom_patterns=self.oom_patterns,
            nan_patterns=self.nan_patterns,
        )
