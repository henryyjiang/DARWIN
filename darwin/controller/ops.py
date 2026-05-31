"""Concrete generation ops: wire the real Phase 1-3 cores (ARCHITECTURE.md §2.3).

`LocalGenerationOps` implements the controller's `GenerationOps` seam against the actual
mutation / finetune / benchmark cores, with no Docker/Lambda dependency — it runs on the local
filesystem with subprocess (or injected) backends, which is exactly the seam the live
container/GPU wrapping will slot into later (deferred, as in Phases 2-3).

Model-directory layout (assumed; the caller seeds the initial population to match):
    <workspace>/<model>/genome      # the genome repo (the thing that mutates)
    <workspace>/<model>/adapter.bin # the LoRA adapter produced by finetuning

`spawn` clones survivor S's genome into the offspring's slot (idempotent — a re-spawn on
resume is a no-op once the offspring repo exists); `mutate` runs the §4.2 window;
`finetune`/`benchmark` build the §5/§6 jobs and run the injected backends. The mutation backend
is chosen per offspring via an injected factory so `claude` vs `local` (and the test fake) all
plug in identically.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from darwin.bench.job import BenchmarkBackend, BenchmarkJob
from darwin.config import DarwinConfig
from darwin.controller.controller import FinetuneOutcomeView, MutateOutcome
from darwin.controller.population import Model
from darwin.controller.state import OffspringState
from darwin.cost import CostLedger
from darwin.finetune import FinetuneBackend, FinetuneJob, run_finetune_job
from darwin.memory import MemoryStore
from darwin.mutation_agent import (
    DeadlineManager,
    GitCheckpointer,
    MutationBackend,
    SmokeTest,
    run_mutation_window,
)
from darwin.mutation_agent.backend import MutationContext
from darwin.mutation_agent.directive import build_directive

MutationBackendFactory = Callable[[str, MutationContext], MutationBackend]


@dataclass
class LocalGenerationOps:
    """Runs an offspring's stages against the real cores on the local filesystem (§2.3)."""

    config: DarwinConfig
    store: MemoryStore
    ledger: CostLedger
    workspace: Path
    mutation_backend_factory: MutationBackendFactory
    finetune_backend: FinetuneBackend
    benchmark_backend: BenchmarkBackend
    smoke_command: list[str]
    base_model: str = "base"
    deadline_factory: Callable[[], DeadlineManager] | None = None

    # ------------------------------------------------------------------ layout
    def _genome_dir(self, name: str) -> Path:
        return self.workspace / name / "genome"

    def _adapter_path(self, name: str) -> Path:
        return self.workspace / name / "adapter.bin"

    # ------------------------------------------------------------------ SPAWN (§3.2)
    def spawn(self, *, offspring: OffspringState, parent: Model, generation: int) -> Model:
        genome = self._genome_dir(offspring.name)
        if not (genome / ".git").exists():
            # clone S's genome into the offspring slot (genome files only, not S's git history)
            if genome.exists():
                shutil.rmtree(genome)
            genome.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                parent.genome_dir, genome, ignore=shutil.ignore_patterns(".git")
            )
        return Model(
            name=offspring.name,
            genome_dir=genome,
            adapter_path=self._adapter_path(offspring.name),
            generation_born=generation,
            parent_survivor=offspring.parent_survivor,
            mutator=offspring.mutator,
            backend=offspring.backend,
        )

    # ------------------------------------------------------------------ MUTATE (§4)
    def mutate(
        self,
        *,
        offspring: Model,
        parent: Model,
        mutator: Model | None,
        state: OffspringState,
        generation: int,
    ) -> MutateOutcome:
        mutator_name = state.mutator or "claude"  # schema needs a non-empty mutator (§7.2)
        ctx = MutationContext(
            offspring_id=offspring.name,
            genome_dir=offspring.genome_dir,
            model=offspring.name,
            parent_survivor=state.parent_survivor,
            mutator=mutator_name,
            generation=generation,
            iteration=state.iteration,
            backend_name=state.backend,
            base_fitness=parent.fitness if parent.fitness is not None else 0.0,
            directive=build_directive(
                offspring_id=offspring.name,
                model=offspring.name,
                parent_survivor=state.parent_survivor,
                mutator=mutator_name,
                generation=generation,
            ),
            checkpointer=GitCheckpointer(offspring.genome_dir),
            smoke=SmokeTest(command=self.smoke_command),
            store=self.store,
        )
        backend = self.mutation_backend_factory(state.backend, ctx)
        deadline = (
            self.deadline_factory()
            if self.deadline_factory is not None
            else DeadlineManager.from_config(self.config.mutation)
        )
        result = run_mutation_window(ctx, backend, deadline)
        return MutateOutcome(
            final_commit=result.final_commit, mutation_failed=result.mutation_failed
        )

    # ------------------------------------------------------------------ FINETUNE (§5)
    def finetune(
        self, *, offspring: Model, state: OffspringState, generation: int
    ) -> FinetuneOutcomeView:
        job = FinetuneJob(
            offspring_id=offspring.name,
            model=offspring.name,
            generation=generation,
            genome_dir=offspring.genome_dir,
            adapter_out=self._adapter_path(offspring.name),
            method=self.config.finetune.method,
            lora_rank=self.config.finetune.lora_rank,
            lora_alpha=self.config.finetune.lora_alpha,
            gpu_rate_usd_per_h=self.config.cost.gpu_rate_usd_per_h,
            per_job_cap_usd=self.config.cost.per_job_cap_usd,
            per_job_max_h=self.config.cost.per_job_max_h,
        )
        res = run_finetune_job(job, self.finetune_backend, self.ledger)
        return FinetuneOutcomeView(
            status=res.status, adapter_path=res.adapter_path, cost_usd=res.cost_usd
        )

    # ------------------------------------------------------------------ BENCHMARK (§6)
    def benchmark(
        self, *, offspring: Model, state: OffspringState, slice_id: int, generation: int
    ) -> dict[str, float]:
        job = BenchmarkJob(
            offspring_id=offspring.name,
            model=offspring.name,
            generation=generation,
            base_model=self.base_model,
            adapter_path=offspring.adapter_path,
            suite=list(self.config.benchmark.suite),
            slice_id=slice_id,
        )
        return self.benchmark_backend.run(job).scores
