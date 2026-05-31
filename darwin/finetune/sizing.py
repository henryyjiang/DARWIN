"""Runtime GPU allocation / run sizing (ARCHITECTURE.md §5.3).

Because the population can **scale its parameter count** — depth expansion (stacking/duplicating
layers, LLaMA-Pro style) and MoE upcycling (dense → sparse mixture-of-experts) — a finetune job's
GPU footprint is no longer fixed at "32B QLoRA on one A100." This module sizes the Lambda
allocation **at runtime from the run's size**: estimate the VRAM the (possibly expanded) model
needs for the chosen method, pick the smallest/cheapest instance in the catalog that fits
(sharding across GPUs when one card can't hold it), and estimate wall-clock + GPU-hours + cost
from the target **training-token budget** (up to 250B tokens).

All pure + tunable: the catalog rates/VRAM and the throughput/VRAM heuristics are coarse
first-order models (clearly approximate — calibrate against real runs), and the planner is
unit-tested for *structure* (a bigger model picks more GPUs; more tokens costs more) rather than
exact magic numbers. `LambdaFinetuneBackend` calls `plan_instance` when a job carries a `RunSize`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Target training-token ceiling for a single run (§ user requirement). Runs may ask for fewer.
MAX_TRAIN_TOKENS = 250_000_000_000  # 250B


@dataclass(frozen=True)
class GpuType:
    """One Lambda instance type. Rates/VRAM are approximate — calibrate per region/availability."""

    name: str  # Lambda instance_type_name
    gpus: int
    vram_gb_per_gpu: float
    usd_per_h: float  # per-INSTANCE on-demand rate
    tok_s_per_gpu: float  # rough ~30B-QLoRA throughput on this card (scaled by size in the planner)

    @property
    def total_vram_gb(self) -> float:
        return self.gpus * self.vram_gb_per_gpu

    @property
    def usd_per_gpu_h(self) -> float:
        return self.usd_per_h / self.gpus


# Representative subset of the Lambda Cloud catalog (approximate on-demand pricing/VRAM).
LAMBDA_CATALOG: tuple[GpuType, ...] = (
    GpuType("gpu_1x_a10", 1, 24, 0.75, 1400),
    GpuType("gpu_1x_a100", 1, 40, 1.29, 1800),
    GpuType("gpu_1x_a100_sxm4", 1, 80, 1.79, 2000),
    GpuType("gpu_1x_h100_pcie", 1, 80, 2.49, 4000),
    GpuType("gpu_2x_a100", 2, 80, 2.58, 1800),
    GpuType("gpu_4x_a100", 4, 80, 5.16, 1800),
    GpuType("gpu_8x_a100_80gb_sxm4", 8, 80, 10.32, 2000),
    GpuType("gpu_8x_h100_sxm5", 8, 80, 23.92, 4000),
)

# Bytes per parameter held in VRAM, by method: weights + (optimizer states + grads for full).
# QLoRA freezes a 4-bit base (~0.5 B/param) + a tiny trainable adapter; full finetune also carries
# fp32 Adam m/v (8) + grads (4) on every param — which is why full is heavily penalized in sizing.
_WEIGHT_BYTES = {"qlora_4bit": 0.55, "lora": 2.1, "full": 2.1}
_OPT_BYTES = {"qlora_4bit": 0.0, "lora": 0.0, "full": 12.0}
# Throughput multiplier vs. the GPU's nominal ~30B-QLoRA rate.
_METHOD_TPUT = {"qlora_4bit": 1.0, "lora": 0.8, "full": 0.4}
_FIXED_OVERHEAD_GB = 6.0  # CUDA ctx + kv/activation slack (coarse)


@dataclass(frozen=True)
class RunSize:
    """The size of one finetune run — drives GPU allocation (§5.3)."""

    target_params_b: float  # effective param count AFTER expansion/upcycling (billions)
    train_tokens: float  # target training tokens (capped at MAX_TRAIN_TOKENS)
    method: str = "qlora_4bit"  # qlora_4bit | lora | full

    def capped_tokens(self) -> float:
        return min(max(self.train_tokens, 0.0), float(MAX_TRAIN_TOKENS))


@dataclass(frozen=True)
class InstancePlan:
    """The chosen Lambda allocation + estimates for a run (§5.3 / §5.4)."""

    instance_type: str
    num_gpus: int
    vram_needed_gb: float
    est_wall_h: float
    est_gpu_hours: float  # wall_h * num_gpus (what the cost ledger multiplies by the rate)
    est_cost_usd: float
    fits: bool  # False => even the largest instance is undersized (run anyway, flagged)


def estimate_vram_gb(rs: RunSize) -> float:
    """Coarse VRAM estimate (GB) for holding + training the model under `method`."""
    method = rs.method if rs.method in _WEIGHT_BYTES else "qlora_4bit"
    per_param = _WEIGHT_BYTES[method] + _OPT_BYTES[method]
    params = rs.target_params_b * 1e9
    return params * per_param / 1e9 + 0.25 * rs.target_params_b + _FIXED_OVERHEAD_GB


def _throughput(gpu: GpuType, rs: RunSize) -> float:
    """Rough tokens/sec across the instance, scaled by model size + method (tunable)."""
    method = rs.method if rs.method in _METHOD_TPUT else "qlora_4bit"
    per_gpu = gpu.tok_s_per_gpu * (32.0 / max(rs.target_params_b, 1.0)) * _METHOD_TPUT[method]
    return max(per_gpu, 1.0) * gpu.gpus


def plan_instance(
    rs: RunSize, catalog: tuple[GpuType, ...] = LAMBDA_CATALOG
) -> InstancePlan:
    """Pick the cheapest instance whose total VRAM fits the run; estimate time + cost (§5.3)."""
    if not catalog:
        raise ValueError("empty GPU catalog")
    vram = estimate_vram_gb(rs)
    fitting = [g for g in catalog if g.total_vram_gb >= vram]
    fits = bool(fitting)
    # cheapest-that-fits; if nothing fits, take the largest-VRAM instance and flag it
    chosen = (
        min(fitting, key=lambda g: g.usd_per_h)
        if fitting
        else max(catalog, key=lambda g: g.total_vram_gb)
    )
    tok_s = _throughput(chosen, rs)
    wall_h = rs.capped_tokens() / tok_s / 3600.0
    gpu_hours = wall_h * chosen.gpus
    return InstancePlan(
        instance_type=chosen.name,
        num_gpus=chosen.gpus,
        vram_needed_gb=round(vram, 1),
        est_wall_h=round(wall_h, 3),
        est_gpu_hours=round(gpu_hours, 3),
        est_cost_usd=round(wall_h * chosen.usd_per_h, 2),
        fits=fits,
    )
