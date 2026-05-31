"""DARWIN v2 configuration schema.

Default values come from ARCHITECTURE.md §10.1. Everything here is tunable; these are
sane starting points so nothing has to be invented at implementation time. The controller
(future phases) loads/overrides these from a run config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Backend = Literal["claude", "local", "mixed"]
FinetuneMethod = Literal["qlora_4bit", "lora", "full"]


@dataclass
class GAConfig:
    """Population & genetic-algorithm parameters (§3, §10.1)."""

    population_size: int = 10  # 5 survivors + 5 offspring
    num_survivors: int = 5
    num_culled: int = 5
    diversity_pick: bool = False  # enable after baseline understood (§3.4)


@dataclass
class MutationConfig:
    """Mutation-window timing & backend (§4, §10.1)."""

    backend: Backend = "claude"  # bootstrap with claude to prove gains, then -> local
    mutation_window_h: float = 3.0  # within the 2-4 h range
    soft_deadline_min: int = 15  # T-minus this many minutes: wrap-up nudge
    kill_grace_min: int = 5  # after hard deadline before force-stop


@dataclass
class FitnessConfig:
    """Fitness reduction weights & penalties (§6.3, §10.1).

    fitness = w * normalized_benchmark_vector
              - lambda_cost * cost
              - lambda_penalty * antigaming_flags
              - lambda_failed * mutation_failed
    """

    # benchmark weights: uniform across benchmarks, sum=1 (resolved at runtime once the
    # benchmark set is known). Empty mapping => uniform.
    benchmark_weights: dict[str, float] = field(default_factory=dict)
    lambda_cost: float = 0.05  # per $ (normalized); cost ~5-10% of fitness swing
    lambda_penalty: float = 0.5  # per anti-gaming flag (strong)
    lambda_failed: float = 0.1  # mutation-failed penalty (mild, not catastrophic)
    finetune_failed_fitness: float = float("-inf")  # floor: below all valid offspring


@dataclass
class CostConfig:
    """Budget enforcement (§5.4, §10.1). Caps must be set per run."""

    gen_budget_usd: float | None = None  # hard cap/generation
    per_job_cap_usd: float | None = None  # runaway finetune kill -> finetune_failed
    per_job_max_h: float | None = None


@dataclass
class FinetuneConfig:
    """Finetuning method & LoRA defaults (§5, §10.1)."""

    method: FinetuneMethod = "qlora_4bit"  # single GPU; avoid sharding until it works
    lora_rank: int = 16  # base genome default the agent may change
    lora_alpha: int = 32


@dataclass
class BenchmarkConfig:
    """Benchmark suite & eval rotation (§6, §10.1)."""

    suite: list[str] = field(default_factory=list)  # e.g. ["humaneval+", "gsm8k", ...]
    eval_rotation: bool = True  # held-out slice rotates per generation


@dataclass
class DarwinConfig:
    """Top-level run configuration. Groups the sub-configs above."""

    run_name: str = "darwin-run"
    ga: GAConfig = field(default_factory=GAConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict:
        return asdict(self)
