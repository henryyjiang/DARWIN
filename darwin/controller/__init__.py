"""Master Controller (ARCHITECTURE.md §2.3, §9.1).

Owns the generation state machine (SELECT → SPAWN → per-offspring MUTATE → FINETUNE → BENCHMARK
→ AGGREGATE_FITNESS → form-next-population → GLOBAL_MEMORY_PASS → CHECKPOINT), the GA (§3), and
resumable state persisted to `runs/gen_<n>/state.json`. It composes the Phase 1-3 cores via an
injectable `GenerationOps` seam so it stays backend/infra-agnostic and testable end-to-end.

Modules:
- `population.py` — `Model` / `Population` (§3.1).
- `ga.py` — ranking, survivor selection (the GA cull), and (S, M) offspring pairing (§3.2).
- `state.py` — resumable `GenerationState` + `GenerationStateStore` (§2.3).
- `controller.py` — the `Controller` state machine + the `GenerationOps` protocol.
- `ops.py` — `LocalGenerationOps`, the concrete wiring of the real mutation/finetune/benchmark
  cores (live Docker/Lambda wrapping deferred).
"""

from darwin.controller.population import Model, Population
from darwin.controller.ga import (
    OffspringPlan,
    pair_offspring,
    rank_models,
    select_survivors,
)
from darwin.controller.state import (
    GenerationState,
    GenerationStateStore,
    OffspringState,
    PHASE_ORDER,
)
from darwin.controller.controller import (
    Controller,
    FinetuneOutcomeView,
    GenerationOps,
    MutateOutcome,
)
from darwin.controller.ops import LocalGenerationOps

__all__ = [
    "Model",
    "Population",
    "OffspringPlan",
    "pair_offspring",
    "rank_models",
    "select_survivors",
    "GenerationState",
    "GenerationStateStore",
    "OffspringState",
    "PHASE_ORDER",
    "Controller",
    "GenerationOps",
    "MutateOutcome",
    "FinetuneOutcomeView",
    "LocalGenerationOps",
]
