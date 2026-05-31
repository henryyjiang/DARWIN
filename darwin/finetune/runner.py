"""Finetune job orchestrator (ARCHITECTURE.md §5.3 / §5.4).

`run_finetune_job` drives a `FinetuneBackend` for one offspring and turns its raw attempt(s)
into a `FinetuneResult`, applying the §5.3 failure policy and §5.4 cost contract:

- **One OOM safe-mode retry.** An `oom` attempt is retried exactly once with `safe_mode=True`
  (the backend's memory-frugal config). A second OOM -> `finetune_failed` (§5.3).
- **Infra vs. recipe.** `infra` -> `infra_failed` (controller resumes/re-provisions; not the
  recipe's fault). Everything else non-`oom` (`nan_loss`, `nonzero_exit`, `no_adapter`,
  `timeout`) -> `finetune_failed` immediately.
- **Cost.** Each attempt's `gpu_hours x rate` is recorded to the cost ledger (§5.4); the
  cumulative cost is the result's `cost_usd`. If a `per_job_cap_usd` is set and cumulative cost
  reaches it, the job is treated as a runaway kill -> `finetune_failed` (`cost_cap`), with no
  further retry (cost guard, §5.3).

The controller (Phase 4) consumes the result: `ok` -> benchmark the adapter; `finetune_failed`
-> floor fitness + patch the memory file's `finetune_failed` flag (§7.2); `infra_failed` ->
re-provision and re-run.
"""

from __future__ import annotations

from darwin.cost import CostLedger
from darwin.finetune.backend import FinetuneBackend
from darwin.finetune.job import FinetuneJob, FinetuneResult


def run_finetune_job(
    job: FinetuneJob,
    backend: FinetuneBackend,
    ledger: CostLedger | None = None,
    *,
    max_oom_retries: int = 1,
) -> FinetuneResult:
    """Run one offspring's finetune end-to-end and return its classified result."""
    attempts = 0
    oom_retries = 0
    total_gpu_hours = 0.0
    cost_usd = 0.0
    safe_mode = False
    log = ""

    while True:
        attempts += 1
        outcome = backend.run(job, safe_mode=safe_mode)
        total_gpu_hours += outcome.gpu_hours
        log = outcome.log

        if ledger is not None and outcome.gpu_hours > 0:
            reason = f"finetune {job.offspring_id} attempt {attempts}"
            if safe_mode:
                reason += " (safe-mode retry)"
            entry = ledger.record_gpu(
                generation=job.generation,
                model=job.model,
                gpu_hours=outcome.gpu_hours,
                rate_usd_per_h=job.gpu_rate_usd_per_h,
                reason=reason,
            )
            cost_usd += entry.amount_usd
        else:
            cost_usd += outcome.gpu_hours * job.gpu_rate_usd_per_h

        if outcome.succeeded:
            return _result(job, "ok", outcome.adapter_path, total_gpu_hours, cost_usd,
                           None, attempts, log)

        # runaway cost kill (§5.3) — checked before any retry so we don't spend further
        if job.per_job_cap_usd is not None and cost_usd >= job.per_job_cap_usd:
            return _result(job, "finetune_failed", None, total_gpu_hours, cost_usd,
                           "cost_cap", attempts, log)

        if outcome.failure_mode == "infra":
            return _result(job, "infra_failed", None, total_gpu_hours, cost_usd,
                           "infra", attempts, log)

        if outcome.failure_mode == "oom" and oom_retries < max_oom_retries:
            oom_retries += 1
            safe_mode = True
            continue  # single safer-config retry (§5.3)

        return _result(job, "finetune_failed", None, total_gpu_hours, cost_usd,
                       outcome.failure_mode, attempts, log)


def _result(job, status, adapter_path, gpu_hours, cost_usd, failure_mode, attempts, log):
    return FinetuneResult(
        offspring_id=job.offspring_id,
        model=job.model,
        status=status,
        adapter_path=adapter_path,
        gpu_hours=gpu_hours,
        cost_usd=cost_usd,
        failure_mode=failure_mode,
        attempts=attempts,
        log=log,
    )
