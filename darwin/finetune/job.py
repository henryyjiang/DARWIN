"""Finetune job contract (ARCHITECTURE.md §5).

The genome a mutator edits *is* the finetuning recipe (§5.2); a finetune job executes that
green recipe to produce a LoRA adapter. This module defines the backend-agnostic contract —
the inputs (`FinetuneJob`), what a single backend attempt reports (`FinetuneOutcome`), and the
final per-offspring result (`FinetuneResult`) — so the controller treats the local subprocess
backend and the live Lambda backend identically.

Failure taxonomy (§5.3) is central and lives here as `FINETUNE_STATUSES` / failure-mode
strings:
- *infra failure* (preemption, transient GPU error) -> `infra_failed` -> controller
  resumes/re-provisions; **not** the offspring's fault.
- *recipe failure* (OOM that won't fit even after one safer-config retry, NaN/inf loss, job
  exits non-zero, no adapter materialized, or a per-job cost/time cap kill) -> `finetune_failed`
  -> floor fitness (§6.3); the recipe genuinely can't train at scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from darwin.config import FinetuneMethod

# Final classification of a finetune job (§5.3).
FinetuneStatus = Literal["ok", "finetune_failed", "infra_failed"]
FINETUNE_STATUSES: tuple[FinetuneStatus, ...] = ("ok", "finetune_failed", "infra_failed")

# Raw failure modes a backend attempt can report. "oom" is the only one the runner retries
# (once, in safe-mode). "infra" maps to infra_failed; everything else is a recipe failure.
FailureMode = Literal[
    "oom", "nan_loss", "nonzero_exit", "no_adapter", "timeout", "infra", "cost_cap"
]


@dataclass
class FinetuneJob:
    """Everything a backend needs to finetune one offspring's green genome (§5.3)."""

    offspring_id: str
    model: str
    generation: int
    genome_dir: Path  # the green genome (HEAD of offspring/<id>); finetuning only runs green
    adapter_out: Path  # where the produced LoRA adapter is written
    method: FinetuneMethod = "qlora_4bit"  # single-GPU default, avoid sharding early (§5.3)
    lora_rank: int = 16
    lora_alpha: int = 32
    gpu_rate_usd_per_h: float = 1.10  # $/hr for the provisioned instance (§5.4)
    per_job_cap_usd: float | None = None  # runaway-cost kill -> finetune_failed (§5.3)
    per_job_max_h: float | None = None  # runaway-time kill (backend-enforced timeout)


@dataclass
class FinetuneOutcome:
    """What one backend attempt reports — pre-classification, pre-costing.

    The runner (`run_finetune_job`) turns a sequence of these (with at most one OOM safe-mode
    retry) into the final `FinetuneResult`, computing cost from `gpu_hours x rate`.
    """

    succeeded: bool
    gpu_hours: float
    adapter_path: Path | None = None
    failure_mode: FailureMode | None = None
    log: str = ""


@dataclass
class FinetuneResult:
    """The controller-facing outcome of finetuning one offspring (§5.3)."""

    offspring_id: str
    model: str
    status: FinetuneStatus
    adapter_path: Path | None
    gpu_hours: float
    cost_usd: float
    failure_mode: FailureMode | None
    attempts: int
    log: str

    @property
    def finetune_failed(self) -> bool:
        """The §7.2 controller-patched flag: recipe couldn't train at scale."""
        return self.status == "finetune_failed"
