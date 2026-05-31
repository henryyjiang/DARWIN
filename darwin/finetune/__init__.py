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
    FinetuneBackend,
    LambdaFinetuneBackend,
    LambdaJobRunner,
    SubprocessFinetuneBackend,
)
from darwin.finetune.lambda_api import (
    LambdaApiError,
    LambdaClient,
    LambdaInstance,
    parse_instance,
    wait_until_active,
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
    "LambdaJobRunner",
    "LambdaClient",
    "LambdaInstance",
    "LambdaApiError",
    "parse_instance",
    "wait_until_active",
    "run_finetune_job",
]
