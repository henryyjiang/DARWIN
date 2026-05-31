"""Finetuning Pipeline (ARCHITECTURE.md §5) — scaffolding.

LoRA/QLoRA training jobs on Lambda Labs GPUs. Executes an offspring's green genome (the
finetuning recipe) to produce the LoRA adapter, reports GPU-hours to the cost ledger, and
distinguishes infra failure (resume/re-provision) from recipe failure (`finetune_failed`,
floor fitness, §5.3). Starts with QLoRA 4-bit single-GPU to avoid sharding (§5.3, §10.1).

Not yet implemented — landed as part of Phase 3 (§10).
"""
