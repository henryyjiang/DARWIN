"""Master Controller (ARCHITECTURE.md §2.3, §9.1) — scaffolding.

Owns the generation state machine (PROVISION → SELECT → SPAWN_OFFSPRING → MUTATE → FINETUNE
→ BENCHMARK → AGGREGATE_FITNESS → GA_CULL → GLOBAL_MEMORY_PASS → CHECKPOINT_GENERATION), the
GA (§3), container lifecycle, Lambda provisioning, deadline enforcement (§4.3), the cost
ledger, and the global-memory-pass trigger. Persists resumable state to `runs/gen_<n>/`.

Not yet implemented — landed as part of Phase 4 (§10).
"""
