"""DARWIN v2 configuration schema.

Default values come from ARCHITECTURE.md §10.1. Everything here is tunable; these are
sane starting points so nothing has to be invented at implementation time. The controller
(future phases) loads/overrides these from a run config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Backend = Literal["claude", "local", "mixed", "mock"]
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
    # Test-profile knob (TEST_RUN_PLAN §3.5): N offspring per generation use the *real* Claude
    # mutator while the rest use `backend` (e.g. `mock`), so the SDK path can be validated while
    # capping API spend. 0 => every offspring uses `backend`.
    claude_sample: int = 0
    # Mutation directive style for the Claude/local agent: "full" = the real param-scaling mission
    # (§4.1); "small" = make one small, safe green code change (for validating the SDK path on a
    # trivial test genome without the agent attempting a full architecture redesign).
    directive_style: str = "full"
    # Claude Agent SDK session knobs (real-Claude path). Empty/0 => the SDK/CLI default.
    claude_model: str = ""  # e.g. a faster/cheaper model for the test; "" => account default
    claude_effort: str = ""  # "low"|"medium"|"high"|... ; "" => SDK default
    claude_max_budget_usd: float = 0.0  # hard per-session USD cap (SDK max_budget_usd); 0 => none


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
    # GPU $/hr used to convert finetune GPU-hours -> dollars in the cost ledger (§5.3).
    # Default ~ a Lambda single-GPU rate; override per run / per instance type.
    gpu_rate_usd_per_h: float = 1.10


Sharding = Literal["none", "tensor", "pipeline"]


@dataclass
class FinetuneConfig:
    """Finetuning method & LoRA defaults (§5, §10.1)."""

    method: FinetuneMethod = "qlora_4bit"  # single GPU; avoid sharding until it works
    lora_rank: int = 16  # base genome default the agent may change
    lora_alpha: int = 32
    # Scale-up knobs (§5.3 / Phase 6): the target base coder and how a >single-GPU model is
    # split. Consumed by the *live* Lambda finetune backend (deferred infra); QLoRA 4-bit on a
    # single GPU (`sharding="none"`, `num_gpus=1`) stays the default until Phase 4 is validated.
    base_model: str = "Qwen/Qwen2.5-Coder-32B"  # §5.1 target (cheapest strong coder at build)
    sharding: Sharding = "none"  # tensor/pipeline split when not using QLoRA single-GPU
    num_gpus: int = 1  # static fallback GPUs/offspring when run-size sizing isn't used (§5.3)
    # Parameter-scaling targets (§5.3): the population may grow the model via depth expansion
    # and MoE upcycling, so the *effective* param count after expansion drives runtime GPU
    # allocation (darwin/finetune/sizing.py), not a fixed instance. `target_params_b` seeds the
    # RunSize; mutators raise it when their genome expands the model.
    target_params_b: float = 32.0  # effective billions of params (base before expansion)
    max_train_tokens: float = 250_000_000_000  # per-run training-token ceiling (250B)
    dynamic_gpu_allocation: bool = True  # size the Lambda instance from the run size at launch


@dataclass
class BenchmarkConfig:
    """Benchmark suite & eval rotation (§6, §10.1)."""

    suite: list[str] = field(default_factory=list)  # e.g. ["humaneval+", "gsm8k", ...]
    eval_rotation: bool = True  # held-out slice rotates per generation
    num_eval_slices: int = 1  # how many private held-out slices exist to rotate over (§6.4)
    eval_seed: int = 0  # seeds the reproducible rotation permutation (§6.4)


@dataclass
class AntiGamingConfig:
    """Anti-gaming heuristics (§6.4). Producers of the `antigaming_flags` fed to fitness (§6.3).

    `enabled` gates the whole scan; individual checks additionally no-op when their inputs are
    absent (no eval items => no contamination scan; no OOD probe => no generalization-gap check),
    so the live infra (host-only eval data, the OOD probe run) can land incrementally.
    """

    enabled: bool = True
    ngram_n: int = 8  # word n-gram width for the contamination scan
    contamination_min_overlap: int = 1  # shared n-grams that trip a contamination flag
    contamination_max_flags: int = 20  # cap on contamination flags (bounded fitness hit)
    max_generalization_gap: float = 0.25  # held-out minus OOD probe that trips a gap flag
    # genome-diff reviewer (§6.4): "claude" (higher recall, default per §4.7), "rule"
    # (no API; used under strict-local), or "none" (skip the diff review).
    genome_reviewer: Literal["claude", "rule", "none"] = "claude"


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
    antigaming: AntiGamingConfig = field(default_factory=AntiGamingConfig)

    def to_dict(self) -> dict:
        return asdict(self)
