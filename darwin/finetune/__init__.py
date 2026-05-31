"""Finetuning Pipeline (ARCHITECTURE.md §5).

LoRA/QLoRA training jobs that execute an offspring's green genome (the finetuning recipe, §5.2)
to produce the LoRA adapter, report GPU-hours to the cost ledger (§5.4), and distinguish infra
failure (resume/re-provision) from recipe failure (`finetune_failed`, floor fitness, §5.3).
Starts with QLoRA 4-bit single-GPU to avoid sharding (§5.3, §10.1).

Backend-agnostic core (this phase): the `FinetuneJob`/`FinetuneOutcome`/`FinetuneResult`
contract (`job.py`), the `FinetuneBackend` protocol + a CPU-runnable `SubprocessFinetuneBackend`
and a scaffolded `LambdaFinetuneBackend` (`backend.py`), and `run_finetune_job` which applies
the §5.3 failure policy (one OOM safe-mode retry) and the §5.4 cost contract (`runner.py`).

Deferred (needs infra): the live Lambda GPU path + the `darwin-finetune` image.
"""

from darwin.finetune.job import (
    FINETUNE_STATUSES,
    FailureMode,
    FinetuneJob,
    FinetuneOutcome,
    FinetuneResult,
    FinetuneStatus,
)
from darwin.finetune.backend import (
    ContainerFinetuneBackend,
    FinetuneBackend,
    LambdaFinetuneBackend,
    LambdaJobRunner,
    SubprocessFinetuneBackend,
    classify_finetune_outcome,
    finetune_env,
)
from darwin.finetune.lambda_api import (
    LambdaApiError,
    LambdaClient,
    LambdaInstance,
    parse_instance,
    wait_until_active,
)
from darwin.finetune.sizing import (
    LAMBDA_CATALOG,
    MAX_TRAIN_TOKENS,
    GpuType,
    InstancePlan,
    RunSize,
    estimate_vram_gb,
    plan_instance,
)
from darwin.finetune.runner import run_finetune_job

__all__ = [
    "FinetuneJob",
    "FinetuneOutcome",
    "FinetuneResult",
    "FinetuneStatus",
    "FINETUNE_STATUSES",
    "FailureMode",
    "FinetuneBackend",
    "SubprocessFinetuneBackend",
    "LambdaFinetuneBackend",
    "ContainerFinetuneBackend",
    "classify_finetune_outcome",
    "finetune_env",
    "LambdaJobRunner",
    "LambdaClient",
    "LambdaInstance",
    "LambdaApiError",
    "parse_instance",
    "wait_until_active",
    "RunSize",
    "InstancePlan",
    "GpuType",
    "LAMBDA_CATALOG",
    "MAX_TRAIN_TOKENS",
    "estimate_vram_gb",
    "plan_instance",
    "run_finetune_job",
]
