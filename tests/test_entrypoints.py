"""Reference finetune + benchmark entrypoint pure cores (ARCHITECTURE.md §5 / §6.2).

Only the network-free config/kwarg/dispatch logic is tested here; the heavy training/eval
bodies in `main()` lazy-import torch/transformers and run only inside the CUDA images.
"""

import json

from darwin.bench.entrypoint import BenchRunConfig, parse_suite, run_suite, write_scores
from darwin.finetune.entrypoint import (
    FinetuneRunConfig,
    build_bnb_kwargs,
    build_lora_kwargs,
    build_training_kwargs,
)


# ------------------------------------------------------------------ finetune (§5)


def test_finetune_config_from_env():
    env = {
        "DARWIN_BASE_MODEL": "Qwen/Qwen2.5-Coder-7B",
        "DARWIN_ADAPTER_OUT": "/work/adapter",
        "DARWIN_FINETUNE_METHOD": "lora",
        "DARWIN_LORA_RANK": "32",
        "DARWIN_LORA_ALPHA": "64",
        "DARWIN_SAFE_MODE": "0",
        "DARWIN_DATASETS": "bigcode/the-stack, openai/gsm8k",
        "DARWIN_MAX_STEPS": "500",
    }
    cfg = FinetuneRunConfig.from_env(env)
    assert cfg.base_model.endswith("7B")
    assert cfg.method == "lora"
    assert cfg.lora_rank == 32 and cfg.lora_alpha == 64
    assert cfg.datasets == ["bigcode/the-stack", "openai/gsm8k"]
    assert cfg.max_steps == 500


def test_finetune_config_defaults_on_missing_env():
    cfg = FinetuneRunConfig.from_env({})
    assert cfg.method == "qlora_4bit"
    assert cfg.lora_rank == 16
    assert cfg.datasets == []


def test_build_lora_kwargs():
    kw = build_lora_kwargs(FinetuneRunConfig(lora_rank=8, lora_alpha=16))
    assert kw["r"] == 8 and kw["lora_alpha"] == 16
    assert kw["task_type"] == "CAUSAL_LM"
    assert "q_proj" in kw["target_modules"]


def test_build_bnb_kwargs_only_for_qlora():
    assert build_bnb_kwargs(FinetuneRunConfig(method="qlora_4bit"))["load_in_4bit"] is True
    assert build_bnb_kwargs(FinetuneRunConfig(method="lora")) is None
    assert build_bnb_kwargs(FinetuneRunConfig(method="full")) is None


def test_safe_mode_forces_frugal_qlora():
    cfg = FinetuneRunConfig(method="full", micro_batch_size=4, max_seq_len=4096, safe_mode=True)
    eff = cfg.effective()
    assert eff.method == "qlora_4bit"
    assert eff.micro_batch_size == 1
    assert eff.max_seq_len == 1024
    # non-safe-mode is unchanged
    assert FinetuneRunConfig(method="full", safe_mode=False).effective().method == "full"


def test_training_kwargs_enable_grad_checkpointing_for_qlora():
    kw = build_training_kwargs(FinetuneRunConfig(method="qlora_4bit", max_steps=123))
    assert kw["gradient_checkpointing"] is True
    assert kw["max_steps"] == 123
    assert kw["save_strategy"] == "no"


# ------------------------------------------------------------------ benchmark (§6.2)


def test_parse_suite():
    assert parse_suite("humaneval+, gsm8k ,math") == ["humaneval+", "gsm8k", "math"]
    assert parse_suite("") == []


def test_bench_config_from_env():
    env = {
        "DARWIN_BASE_MODEL": "base",
        "DARWIN_ADAPTER_PATH": "/work/adapter",
        "DARWIN_SUITE": "code,math",
        "DARWIN_EVAL_SLICE": "3",
        "DARWIN_SCORES_OUT": "/work/scores.json",
    }
    cfg = BenchRunConfig.from_env(env)
    assert cfg.suite == ["code", "math"]
    assert cfg.slice_id == 3
    assert cfg.scores_out.endswith("scores.json")


def test_run_suite_dispatches_each_benchmark():
    cfg = BenchRunConfig(suite=["code", "math"])
    scores = run_suite(cfg, runner=lambda b, c: {"code": 0.7, "math": 0.5}[b])
    assert scores == {"code": 0.7, "math": 0.5}


def test_write_scores_roundtrips(tmp_path):
    p = write_scores(tmp_path / "out" / "scores.json", {"code": 0.9})
    assert json.loads(p.read_text()) == {"code": 0.9}
