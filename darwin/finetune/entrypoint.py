"""Reference finetune entrypoint (ARCHITECTURE.md §5).

The default recipe the `darwin-finetune` image runs (§8.5) and the genome the mutator starts
from: read the LoRA/QLoRA config from the `DARWIN_*` env the finetune backend sets (§5.3
`SubprocessFinetuneBackend`), finetune the base model, and write a LoRA adapter to
`DARWIN_ADAPTER_OUT`. A mutator edits *this file* (and the data/objective) as its genome.

Design: the config parsing and the PEFT/BitsAndBytes/TrainingArguments kwarg assembly are
**pure functions** (`FinetuneRunConfig.from_env`, `build_lora_kwargs`, `build_bnb_kwargs`,
`build_training_kwargs`) so they're unit-tested with no GPU/torch. `main()` lazy-imports the
heavy stack (torch/transformers/peft/trl/datasets) and runs the actual training — exercised only
inside the CUDA image, like the other live paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(env, key, default):
    try:
        return float(env[key])
    except (KeyError, ValueError):
        return default


def _env_int(env, key, default):
    try:
        return int(env[key])
    except (KeyError, ValueError):
        return default


@dataclass
class FinetuneRunConfig:
    """Resolved finetune configuration (the genome may override any field via env, §5.2)."""

    base_model: str = "Qwen/Qwen2.5-Coder-32B"
    adapter_out: str = "adapter"
    method: str = "qlora_4bit"  # qlora_4bit | lora | full
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    safe_mode: bool = False
    datasets: list[str] = field(default_factory=list)  # HF dataset ids (§8.3)
    max_steps: int = 1000
    micro_batch_size: int = 4
    grad_accum: int = 8
    learning_rate: float = 2e-4
    max_seq_len: int = 2048
    seed: int = 0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "FinetuneRunConfig":
        env = dict(os.environ if env is None else env)
        datasets = [d.strip() for d in env.get("DARWIN_DATASETS", "").split(",") if d.strip()]
        return cls(
            base_model=env.get("DARWIN_BASE_MODEL", cls.base_model),
            adapter_out=env.get("DARWIN_ADAPTER_OUT", cls.adapter_out),
            method=env.get("DARWIN_FINETUNE_METHOD", cls.method),
            lora_rank=_env_int(env, "DARWIN_LORA_RANK", cls.lora_rank),
            lora_alpha=_env_int(env, "DARWIN_LORA_ALPHA", cls.lora_alpha),
            lora_dropout=_env_float(env, "DARWIN_LORA_DROPOUT", cls.lora_dropout),
            safe_mode=env.get("DARWIN_SAFE_MODE", "0") == "1",
            datasets=datasets,
            max_steps=_env_int(env, "DARWIN_MAX_STEPS", cls.max_steps),
            micro_batch_size=_env_int(env, "DARWIN_MICRO_BATCH", cls.micro_batch_size),
            grad_accum=_env_int(env, "DARWIN_GRAD_ACCUM", cls.grad_accum),
            learning_rate=_env_float(env, "DARWIN_LR", cls.learning_rate),
            max_seq_len=_env_int(env, "DARWIN_MAX_SEQ_LEN", cls.max_seq_len),
            seed=_env_int(env, "DARWIN_SEED", cls.seed),
        )

    def effective(self) -> "FinetuneRunConfig":
        """Apply the §5.3 safe-mode lever: memory-frugal settings for the single OOM retry."""
        if not self.safe_mode:
            return self
        return FinetuneRunConfig(
            **{
                **self.__dict__,
                "method": "qlora_4bit",  # force 4-bit even if the genome asked for full/lora
                "micro_batch_size": 1,
                "grad_accum": max(self.grad_accum, self.micro_batch_size * self.grad_accum),
                "max_seq_len": min(self.max_seq_len, 1024),
            }
        )


def build_lora_kwargs(cfg: FinetuneRunConfig) -> dict:
    """PEFT `LoraConfig` kwargs (pure)."""
    return {
        "r": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    }


def build_bnb_kwargs(cfg: FinetuneRunConfig) -> dict | None:
    """BitsAndBytes 4-bit kwargs for QLoRA, or None when not quantizing (pure)."""
    if cfg.method != "qlora_4bit":
        return None
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }


def build_training_kwargs(cfg: FinetuneRunConfig) -> dict:
    """`TrainingArguments` kwargs (pure)."""
    return {
        "output_dir": cfg.adapter_out,
        "per_device_train_batch_size": cfg.micro_batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "max_steps": cfg.max_steps,
        "gradient_checkpointing": cfg.safe_mode or cfg.method == "qlora_4bit",
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "no",
        "seed": cfg.seed,
        "report_to": [],
    }


def main(env: dict[str, str] | None = None) -> int:  # pragma: no cover - needs the GPU stack
    """Run the reference finetune. Lazy-imports the heavy training stack (CUDA image only)."""
    cfg = FinetuneRunConfig.from_env(env).effective()
    print(f"[darwin-finetune] base={cfg.base_model} method={cfg.method} "
          f"rank={cfg.lora_rank} steps={cfg.max_steps} safe_mode={cfg.safe_mode}")

    import torch  # noqa: F401
    from datasets import load_dataset, concatenate_datasets
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    bnb = build_bnb_kwargs(cfg)
    quant = None
    if bnb is not None:
        bnb = dict(bnb)
        bnb["bnb_4bit_compute_dtype"] = getattr(torch, bnb["bnb_4bit_compute_dtype"])
        quant = BitsAndBytesConfig(**bnb)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quant, device_map="auto"
    )
    if quant is not None:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**build_lora_kwargs(cfg)))

    parts = [load_dataset(d, split="train") for d in cfg.datasets] or [
        load_dataset("bigcode/the-stack-smol", split="train")
    ]
    data = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(**build_training_kwargs(cfg)),
        train_dataset=data,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_len,
    )
    trainer.train()
    model.save_pretrained(cfg.adapter_out)  # the LoRA adapter (§5.2)
    print(f"[darwin-finetune] adapter written to {cfg.adapter_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
