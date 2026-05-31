"""Runtime GPU sizing for param-scaling runs (ARCHITECTURE.md §5.3)."""

import json

import pytest

from darwin.finetune import (
    LAMBDA_CATALOG,
    MAX_TRAIN_TOKENS,
    FinetuneJob,
    LambdaClient,
    LambdaFinetuneBackend,
    RunSize,
    estimate_vram_gb,
    plan_instance,
)


# ------------------------------------------------------------------ vram estimate


def test_qlora_32b_fits_a_single_80gb_card():
    vram = estimate_vram_gb(RunSize(target_params_b=32, train_tokens=1e9, method="qlora_4bit"))
    assert vram < 80  # 4-bit base fits one 80GB GPU


def test_full_finetune_needs_far_more_vram_than_qlora():
    rs_full = RunSize(target_params_b=32, train_tokens=1e9, method="full")
    rs_qlora = RunSize(target_params_b=32, train_tokens=1e9, method="qlora_4bit")
    assert estimate_vram_gb(rs_full) > 5 * estimate_vram_gb(rs_qlora)


# ------------------------------------------------------------------ planner (§5.3)


def test_small_qlora_picks_a_single_gpu_instance():
    plan = plan_instance(RunSize(target_params_b=7, train_tokens=1e9, method="qlora_4bit"))
    assert plan.num_gpus == 1
    assert plan.fits


def test_upcycled_model_scales_to_multi_gpu():
    # a depth-expanded / MoE-upcycled 150B QLoRA can't fit one 80GB card -> multi-GPU
    plan = plan_instance(RunSize(target_params_b=150, train_tokens=1e9, method="qlora_4bit"))
    assert plan.num_gpus >= 2
    assert plan.vram_needed_gb > 80


def test_more_tokens_costs_more_and_takes_longer():
    small = plan_instance(RunSize(target_params_b=32, train_tokens=1e9))
    big = plan_instance(RunSize(target_params_b=32, train_tokens=100e9))
    assert big.est_wall_h > small.est_wall_h
    assert big.est_cost_usd > small.est_cost_usd


def test_token_budget_capped_at_250b():
    over = plan_instance(RunSize(target_params_b=32, train_tokens=500e9))  # asks for 500B
    capped = plan_instance(RunSize(target_params_b=32, train_tokens=MAX_TRAIN_TOKENS))
    assert over.est_wall_h == capped.est_wall_h  # clamped to 250B


def test_oversized_run_flags_not_fits_but_still_plans():
    plan = plan_instance(RunSize(target_params_b=100_000, train_tokens=1e9, method="full"))
    assert plan.fits is False
    assert plan.num_gpus == max(g.gpus for g in LAMBDA_CATALOG)  # largest instance


# ------------------------------------------------------------------ dynamic Lambda allocation


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, json.loads(body) if body else None))
        for (m, suffix), resp in self.routes.items():
            if m == method and url.endswith(suffix):
                return resp
        return (404, "{}")


def _routes():
    d = lambda obj: (200, json.dumps({"data": obj}))
    return {
        ("POST", "/instance-operations/launch"): d({"instance_ids": ["i-1"]}),
        ("GET", "/instances/i-1"): d({"id": "i-1", "status": "active", "ip": "1.2.3.4"}),
        ("POST", "/instance-operations/terminate"): d({}),
    }


def test_backend_sizes_instance_from_run_size(tmp_path):
    http = FakeHttp(_routes())
    backend = LambdaFinetuneBackend(client=LambdaClient("k", http), clock=lambda: 0.0,
                                    job_runner=lambda inst, job, *, safe_mode: _ok(job))
    job = FinetuneJob(
        offspring_id="o0", model="o0", generation=1,
        genome_dir=tmp_path / "g", adapter_out=tmp_path / "a.bin",
        run_size=RunSize(target_params_b=150, train_tokens=1e9, method="qlora_4bit"),
    )
    backend.run(job)
    launch = next(p for m, u, p in http.calls if u.endswith("/launch"))
    # a 150B run must be sized onto a multi-GPU instance, not the static gpu_1x_a100 default
    assert "1x" not in launch["instance_type_name"]


def test_backend_uses_static_default_without_run_size(tmp_path):
    http = FakeHttp(_routes())
    backend = LambdaFinetuneBackend(client=LambdaClient("k", http), instance_type="gpu_1x_a100",
                                    clock=lambda: 0.0,
                                    job_runner=lambda inst, job, *, safe_mode: _ok(job))
    job = FinetuneJob(offspring_id="o0", model="o0", generation=1,
                      genome_dir=tmp_path / "g", adapter_out=tmp_path / "a.bin")  # no run_size
    backend.run(job)
    launch = next(p for m, u, p in http.calls if u.endswith("/launch"))
    assert launch["instance_type_name"] == "gpu_1x_a100"


def _ok(job):
    from darwin.finetune import FinetuneOutcome

    return FinetuneOutcome(True, 1.0, adapter_path=job.adapter_out, log="ok")
