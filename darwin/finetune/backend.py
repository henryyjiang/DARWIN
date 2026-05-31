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
from dataclasses import dataclass
from typing import Protocol

from darwin.finetune.job import FinetuneJob, FinetuneOutcome

# Default substrings that mark a recipe-level failure in a training log (§5.3). Case-insensitive.
_OOM_PATTERNS = ("out of memory", "cuda out of memory", "oom", "outofmemoryerror")
_NAN_PATTERNS = ("nan loss", "loss is nan", "inf loss", "loss became inf", "non-finite loss")


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
        env.update(
            DARWIN_ADAPTER_OUT=str(job.adapter_out),
            DARWIN_LORA_RANK=str(job.lora_rank),
            DARWIN_LORA_ALPHA=str(job.lora_alpha),
            DARWIN_FINETUNE_METHOD=job.method,
            DARWIN_SAFE_MODE="1" if safe_mode else "0",
        )
        job.adapter_out.parent.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        timed_out = False
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
            exit_code = -1
            log = f"finetune timed out after {self.timeout_s}s\n{exc.output or ''}"
        gpu_hours = (time.monotonic() - start) / 3600.0

        if timed_out:
            return FinetuneOutcome(False, gpu_hours, failure_mode="timeout", log=log)

        if exit_code == 0:
            if job.adapter_out.exists():
                return FinetuneOutcome(
                    True, gpu_hours, adapter_path=job.adapter_out, log=log
                )
            # green exit but no adapter materialized -> recipe failure (§4.4.1 step 4 analogue)
            return FinetuneOutcome(False, gpu_hours, failure_mode="no_adapter", log=log)

        low = log.lower()
        if any(p in low for p in self.oom_patterns):
            mode = "oom"
        elif any(p in low for p in self.nan_patterns):
            mode = "nan_loss"
        else:
            mode = "nonzero_exit"
        return FinetuneOutcome(False, gpu_hours, failure_mode=mode, log=log)


@dataclass
class LambdaFinetuneBackend:
    """Live Lambda Labs GPU finetune backend (§5.3) — scaffold.

    The live path: request a GPU instance via Lambda's API, push the green genome, run the
    genome's entrypoint inside the `darwin-finetune` image (CUDA + PEFT/LoRA + base model,
    §8.5), poll to completion, fetch the adapter, and report real GPU-hours (wall-clock x GPUs)
    for the cost ledger. Infra preemption surfaces as `failure_mode="infra"` so the runner
    classifies it `infra_failed` (resume/re-provision, §5.3) rather than penalizing the recipe.

    Deferred until the Lambda API client + image land; not importable infra is required to run
    the rest of Phase 3.
    """

    api_key: str | None = None
    instance_type: str = "gpu_1x_a100"
    region: str | None = None

    def run(self, job: FinetuneJob, *, safe_mode: bool = False) -> FinetuneOutcome:
        raise NotImplementedError(
            "LambdaFinetuneBackend is scaffolded; the live GPU path lands with the Lambda "
            "API client and the darwin-finetune image. Use SubprocessFinetuneBackend for the "
            "single-GPU / proxy path."
        )
