"""Benchmark Runner & fitness (ARCHITECTURE.md §6).

Runs the eval suite (coding / math / reasoning) against a finetuned offspring (base + adapter)
in the zero-egress `darwin-eval` container, returns a per-benchmark score vector, and reduces
it to scalar fitness. Benchmarking is controller-driven and post-finetune — there is no
agent-callable scored-benchmark tool (§6.2 / §9.3).

This phase lands the backend-agnostic core:
- `job.py` — `BenchmarkJob`/`BenchmarkResult`/`BenchmarkBackend` contract + a CPU-runnable
  `SubprocessBenchmarkBackend` and a scaffolded `EvalContainerBenchmarkBackend`.
- `rotation.py` — the seeded held-out-slice rotation keyed by generation (§6.2/§6.4).
- `fitness.py` — the §6.3 fitness reduction (normalize vs. survivor baseline; floor on
  finetune_failed; cost / anti-gaming / mutation-failed penalties).

The salvaged v1 SWE-bench harness lives under `darwin/bench/swe_bench/` (§10.2) and feeds the
coding slice once the live eval container lands. Anti-gaming heuristics (§6.4) arrive in
Phase 6.
"""

from darwin.bench.job import (
    BenchmarkBackend,
    BenchmarkError,
    BenchmarkJob,
    BenchmarkResult,
    EvalContainerBenchmarkBackend,
    SubprocessBenchmarkBackend,
)
from darwin.bench.rotation import held_out_slice, rotation_schedule
from darwin.bench.fitness import (
    normalize_scores,
    reduce_fitness,
    resolve_weights,
    survivor_baseline,
)

__all__ = [
    "BenchmarkJob",
    "BenchmarkResult",
    "BenchmarkBackend",
    "BenchmarkError",
    "SubprocessBenchmarkBackend",
    "EvalContainerBenchmarkBackend",
    "held_out_slice",
    "rotation_schedule",
    "survivor_baseline",
    "normalize_scores",
    "resolve_weights",
    "reduce_fitness",
]
