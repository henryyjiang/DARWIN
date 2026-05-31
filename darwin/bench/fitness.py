"""Fitness reduction (ARCHITECTURE.md §6.3).

Reduces a per-benchmark score vector to the scalar the GA ranks on:

    fitness = w . normalized_benchmark_vector
              - lambda_cost   * cost_usd
              - lambda_penalty * antigaming_flags
              - lambda_failed  * mutation_failed

with two short-circuits:
- `finetune_failed` -> **floor fitness** (`config.finetune_failed_fitness`, below all valid
  offspring): a recipe that can't train at scale is strongly selected against (§5.3).
- normalization is **against the survivor baseline scored on the current generation's slice**
  (§6.2), so a score of "the survivor average" normalizes to 1.0 and improvements are measured
  relative to the current population.

Pure functions, no I/O — directly unit-testable. The controller assembles the inputs
(offspring scores from the benchmark runner, baseline from the survivors' current-slice
scores, cost from the ledger, flags from the anti-gaming pass) and calls `reduce_fitness`.
"""

from __future__ import annotations

from darwin.config import FitnessConfig


def survivor_baseline(survivor_scores: list[dict[str, float]]) -> dict[str, float]:
    """Per-benchmark mean across the survivors' current-slice score vectors (§6.2/§6.3)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for vec in survivor_scores:
        for bench, score in vec.items():
            sums[bench] = sums.get(bench, 0.0) + score
            counts[bench] = counts.get(bench, 0) + 1
    return {bench: sums[bench] / counts[bench] for bench in sums}


def normalize_scores(
    scores: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    """Normalize each benchmark by the survivor baseline (ratio; 1.0 == survivor average).

    A zero/absent baseline (e.g. gen 0 with no survivors yet) falls back to the raw score so
    normalization never divides by zero and gen-0 fitness is still ordered.
    """
    out: dict[str, float] = {}
    for bench, score in scores.items():
        base = baseline.get(bench, 0.0)
        out[bench] = score / base if base > 0 else score
    return out


def resolve_weights(
    benchmarks: list[str], configured: dict[str, float]
) -> dict[str, float]:
    """Weights for the benchmarks present, renormalized to sum 1 (§6.3 / §10.1 default uniform).

    Empty config => uniform. A config that names benchmarks is restricted to those present and
    renormalized; if that leaves no positive mass, fall back to uniform.
    """
    if not benchmarks:
        return {}
    if configured:
        w = {b: max(0.0, configured.get(b, 0.0)) for b in benchmarks}
        total = sum(w.values())
        if total > 0:
            return {b: v / total for b, v in w.items()}
    n = len(benchmarks)
    return {b: 1.0 / n for b in benchmarks}


def reduce_fitness(
    *,
    scores: dict[str, float],
    baseline: dict[str, float] | None = None,
    cost_usd: float = 0.0,
    antigaming_flags: int = 0,
    mutation_failed: bool = False,
    finetune_failed: bool = False,
    config: FitnessConfig,
) -> float:
    """Reduce a benchmark vector + penalties to the scalar fitness (§6.3)."""
    if finetune_failed:
        return config.finetune_failed_fitness  # floor: below all valid offspring (§5.3)

    baseline = baseline or {}
    if scores:
        norm = normalize_scores(scores, baseline)
        weights = resolve_weights(list(scores), config.benchmark_weights)
        benchmark_term = sum(weights[b] * norm[b] for b in scores)
    else:
        benchmark_term = 0.0

    return (
        benchmark_term
        - config.lambda_cost * cost_usd
        - config.lambda_penalty * antigaming_flags
        - config.lambda_failed * (1.0 if mutation_failed else 0.0)
    )
