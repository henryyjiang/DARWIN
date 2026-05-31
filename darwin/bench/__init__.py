"""Benchmark Runner (ARCHITECTURE.md §6) — scaffolding.

Runs the eval suite (coding / math / reasoning) against a finetuned offspring in the
zero-egress `darwin-eval` container, returns a fitness vector, and applies anti-gaming
checks (§6.4). Benchmarking is controller-driven and post-finetune (base + adapter at load
time) — there is no agent-callable scored-benchmark tool (§6.2 / §9.3).

The salvaged v1 SWE-bench harness lives under `darwin/bench/swe_bench/` (the one v1 piece
worth keeping per §10.2) and will be adapted into this runner.

Not yet implemented — landed as part of Phases 3 (single offspring) and 6 (anti-gaming) (§10).
"""
