"""Master Controller — the generation state machine (ARCHITECTURE.md §2.3, §9.1).

`Controller.run_generation` drives one generation end-to-end, composing the Phase 1-3 cores:

    SELECT survivors (GA cull, §3.2) → SPAWN offspring (clone S, mutator M, §3.2)
      → per offspring: MUTATE (§4) → FINETUNE (§5) → BENCHMARK (§6)
      → AGGREGATE_FITNESS (§6.3, normalized vs. the survivor baseline)
      → form next population (5 survivors + 5 offspring)
      → GLOBAL_MEMORY_PASS (§7.4) → CHECKPOINT (persist resumable state, §2.3)

Every step persists `runs/gen_<n>/state.json` and is idempotent: a crashed generation
**resumes at the first incomplete offspring stage** without re-running completed work (§2.3).

The per-offspring *execution* (clone the genome, run the mutation window, finetune, benchmark)
is delegated to an injectable **`GenerationOps`** seam so the controller stays backend- and
infra-agnostic and is testable end-to-end with fakes. The concrete `LocalGenerationOps`
(`ops.py`) wires the real cores; the live Docker/Lambda wrapping is deferred (as in Phases
2-3). The controller owns selection, pairing, fitness aggregation, the cull, the
controller-only memory patch (§7.2), and the global-memory pass trigger.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from darwin.bench.fitness import reduce_fitness, survivor_baseline
from darwin.bench.rotation import held_out_slice
from darwin.config import DarwinConfig
from darwin.controller.diversity import genome_code_distance
from darwin.controller.ga import OffspringPlan, pair_offspring, select_survivors
from darwin.controller.population import Model, Population
from darwin.controller.state import (
    GenerationState,
    GenerationStateStore,
    OffspringState,
)
from darwin.antigaming import AntiGamingReport
from darwin.cost import BudgetGuard, CostLedger
from darwin.global_memory import Synthesizer, run_global_memory_pass
from darwin.memory import MemoryStore


# --------------------------------------------------------------------------- ops seam


class GenerationOps(Protocol):
    """Executes one offspring's MUTATE / FINETUNE / BENCHMARK stages (§2.3).

    Implementations own the infra-specific details (genome cloning, the mutation window, the
    finetune job, the eval run). The controller calls these in order and records the results
    into the resumable state. `spawn` must be idempotent (safe to re-call on resume) and return
    the offspring's `Model` handle.
    """

    def spawn(self, *, offspring: OffspringState, parent: Model, generation: int) -> Model: ...

    def mutate(
        self,
        *,
        offspring: Model,
        parent: Model,
        mutator: Model | None,
        state: OffspringState,
        generation: int,
    ) -> "MutateOutcome": ...

    def finetune(
        self, *, offspring: Model, state: OffspringState, generation: int
    ) -> "FinetuneOutcomeView": ...

    def benchmark(
        self, *, offspring: Model, state: OffspringState, slice_id: int, generation: int
    ) -> dict[str, float]: ...


class AntiGamingScanner(Protocol):
    """Runs the §6.4 anti-gaming scan for one finetuned+benchmarked offspring.

    Returns the `AntiGamingReport` whose `count` the controller feeds to fitness as
    `antigaming_flags` (§6.3). Injected into the controller (default None => disabled, flags stay
    0); the concrete `LocalAntiGamingScanner` (`antigaming_ops.py`) wires the real producers.
    """

    def scan(
        self, *, offspring: Model, state: OffspringState, slice_id: int, generation: int
    ) -> AntiGamingReport: ...


@dataclass
class MutateOutcome:
    """What the controller records from the MUTATE stage."""

    final_commit: str | None
    mutation_failed: bool


@dataclass
class FinetuneOutcomeView:
    """What the controller records from the FINETUNE stage."""

    status: str  # "ok" | "finetune_failed" | "infra_failed"
    adapter_path: Path | None
    cost_usd: float


# --------------------------------------------------------------------------- controller


@dataclass
class Controller:
    """Owns the generation state machine and the GA (§2.3, §9.1)."""

    config: DarwinConfig
    store: MemoryStore
    ledger: CostLedger
    state_store: GenerationStateStore
    ops: GenerationOps
    synthesizer: Synthesizer | None = None  # global-memory pass writer (§7.4); injected/fake
    antigaming: "AntiGamingScanner | None" = None  # §6.4 scan; None => disabled (flags stay 0)
    budget: BudgetGuard | None = None  # §5.4 hard cap; None => uncapped (launch every offspring)
    enable_global_memory: bool = True
    # §3.4 diversity safeguard distance fn; used only when `ga.diversity_pick` is on. Defaults
    # to the code n-gram distance (`genome_code_distance`); inject an embedding-based one later.
    diversity_fn: Callable[[Model, Model], float] | None = None
    rng: random.Random = field(default_factory=random.Random)

    # ------------------------------------------------------------------ public loop
    def run(self, generations: int, population: Population) -> Population:
        """Run `generations` generations starting from `population`; return the final one."""
        for gen in range(generations):
            population = self.run_generation(gen, population)
        return population

    def run_generation(self, generation: int, population: Population) -> Population:
        """Run one generation end-to-end (resumable) and return the next population."""
        state = self._load_or_spawn(generation, population)
        survivors = [population.get(n) for n in (state.survivors_after_cull or [])]

        # ---- per-offspring pipeline: MUTATE -> FINETUNE -> BENCHMARK
        if not state.at_least("offspring_done"):
            slice_id = self._eval_slice(generation)
            by_name = population.by_name()
            for off in state.offspring:
                self._run_offspring(off, population, by_name, slice_id, generation)
                self.state_store.save(state)  # checkpoint after each offspring (resumable)
            state.advance_to("offspring_done")
            self.state_store.save(state)

        # ---- AGGREGATE_FITNESS
        if not state.at_least("aggregated"):
            self._aggregate_fitness(state, survivors)
            state.advance_to("aggregated")
            self.state_store.save(state)

        # ---- form next population (GA reproduction result: survivors + offspring)
        if not state.at_least("culled"):
            out = self._form_next_population(state, survivors, generation)
            state.population_out = out.to_dict()
            state.advance_to("culled")
            self.state_store.save(state)

        # ---- GLOBAL_MEMORY_PASS
        if not state.at_least("global_memory"):
            if self.enable_global_memory:
                run_global_memory_pass(self.store, generation, self.synthesizer)
            state.advance_to("global_memory")
            self.state_store.save(state)

        # ---- CHECKPOINT
        state.advance_to("checkpoint")
        state.completed = True
        self.state_store.save(state)
        return Population.from_dict(state.population_out)

    # ------------------------------------------------------------------ SELECT + SPAWN
    def _load_or_spawn(self, generation: int, population: Population) -> GenerationState:
        if self.state_store.exists(generation):
            return self.state_store.load(generation)

        survivors = select_survivors(
            population.models,
            self.config.ga.num_survivors,
            diversity_pick=self.config.ga.diversity_pick,
            diversity_fn=(self.diversity_fn or genome_code_distance)
            if self.config.ga.diversity_pick
            else None,
        )
        survivor_names = [s.name for s in survivors]
        # offspring fill the culled (non-survivor) slots, keeping population size + names stable
        offspring_slots = [m.name for m in population.models if m.name not in survivor_names]
        plans = pair_offspring(survivors, len(offspring_slots), self.rng)

        offspring_states = []
        for slot_name, plan in zip(offspring_slots, plans):
            offspring_states.append(
                OffspringState(
                    name=slot_name,
                    parent_survivor=plan.parent_survivor,
                    mutator=plan.mutator,
                    backend=self._mutation_backend(plan),
                    iteration=self._next_iteration(slot_name),
                )
            )

        state = GenerationState(
            generation=generation,
            phase="spawned",
            population_in=population.to_dict(),
            offspring=offspring_states,
            survivors_after_cull=survivor_names,
        )
        self.state_store.save(state)
        return state

    def _mutation_backend(self, plan: OffspringPlan) -> str:
        """Resolve the mutation backend for an offspring (§4.7 / §3.2 fallback)."""
        if plan.mutator is None:
            return "claude"  # degenerate <2-survivor case: no distinct local mutator (§3.2)
        return "local" if self.config.mutation.backend in ("local", "mixed") else "claude"

    def _next_iteration(self, model: str) -> int:
        nums = self.store.iteration_numbers(model)
        return nums[-1] + 1 if nums else 0

    def _eval_slice(self, generation: int) -> int:
        cfg = self.config.benchmark
        if not cfg.eval_rotation or cfg.num_eval_slices <= 1:
            return 0
        return held_out_slice(generation, cfg.num_eval_slices, seed=cfg.eval_seed)

    # ------------------------------------------------------------------ per offspring
    def _run_offspring(
        self,
        off: OffspringState,
        population: Population,
        by_name: dict[str, Model],
        slice_id: int,
        generation: int,
    ) -> None:
        parent = population.get(off.parent_survivor)
        mutator = by_name.get(off.mutator) if off.mutator else None
        offspring_model = self.ops.spawn(
            offspring=off, parent=parent, generation=generation
        )

        if not off.mutation_done:
            res = self.ops.mutate(
                offspring=offspring_model,
                parent=parent,
                mutator=mutator,
                state=off,
                generation=generation,
            )
            off.mutation_done = True
            off.mutation_failed = res.mutation_failed
            off.final_commit = res.final_commit

        if not off.finetune_done:
            # §5.4 hard cap: once the generation's budget is exhausted, launch no new finetunes.
            # In-flight (already-launched, earlier) offspring are never killed — this loop is
            # sequential, so checking before each launch models "let in-flight finish". A deferred
            # offspring is carried unscored for re-attempt next generation if budget frees.
            if self.budget is not None and self.budget.status(generation).exhausted:
                off.finetune_status = "deferred"
            else:
                fres = self.ops.finetune(
                    offspring=offspring_model, state=off, generation=generation
                )
                off.finetune_status = fres.status
                off.adapter_path = str(fres.adapter_path) if fres.adapter_path else None
                off.cost_usd += fres.cost_usd
                offspring_model.adapter_path = fres.adapter_path
            off.finetune_done = True

        if not off.benchmark_done:
            if off.finetune_status == "ok":
                off.scores = self.ops.benchmark(
                    offspring=offspring_model,
                    state=off,
                    slice_id=slice_id,
                    generation=generation,
                )
            else:
                # finetune_failed -> floor fitness (§5.3); infra_failed -> recipe not at fault,
                # but a full re-provision/retry is Phase 7 hardening — treated as no-score here.
                off.scores = {}
            off.benchmark_done = True

        if not off.antigaming_done:
            # Only scan offspring that will actually be ranked on merit; a finetune_failed recipe
            # already gets floor fitness (§5.3), so skip the (possibly Claude-backed) scan there.
            if self.antigaming is not None and off.finetune_status == "ok":
                report = self.antigaming.scan(
                    offspring=offspring_model,
                    state=off,
                    slice_id=slice_id,
                    generation=generation,
                )
                off.antigaming_flags = report.count
            off.antigaming_done = True

    # ------------------------------------------------------------------ AGGREGATE_FITNESS
    def _aggregate_fitness(self, state: GenerationState, survivors: list[Model]) -> None:
        baseline = survivor_baseline([s.scores for s in survivors if s.scores])
        for off in state.offspring:
            if off.finetune_status == "deferred":
                # never launched (budget cap, §5.4): unscored, not a recipe failure -> no floor.
                # Carried into the next population to be re-attempted if budget frees.
                off.fitness = None
                self._patch_memory(off)
                continue
            failed = off.finetune_status in ("finetune_failed", "infra_failed")
            fitness = reduce_fitness(
                scores=off.scores,
                baseline=baseline,
                cost_usd=off.cost_usd,
                antigaming_flags=off.antigaming_flags,
                mutation_failed=off.mutation_failed,
                finetune_failed=failed,
                config=self.config.fitness,
            )
            off.fitness = fitness
            self._patch_memory(off)

    def _patch_memory(self, off: OffspringState) -> None:
        """Patch the controller-owned post-benchmark fields into the memory file (§7.2)."""
        try:
            self.store.patch_iteration(
                off.name,
                off.iteration,
                final_fitness=off.fitness,
                mutation_failed=off.mutation_failed,
                finetune_failed=off.finetune_failed,
            )
        except FileNotFoundError:
            # No memory file (e.g. a mutation that never wrote one); the §4.3 transcript-based
            # synthesis fallback would fill it. Nothing to patch.
            pass

    # ------------------------------------------------------------------ next population
    def _form_next_population(
        self, state: GenerationState, survivors: list[Model], generation: int
    ) -> Population:
        models: list[Model] = []
        for s in survivors:
            s.is_survivor = True
            models.append(s)

        slot_models = Population.from_dict(state.population_in).by_name()
        for off in state.offspring:
            base = slot_models[off.name]
            models.append(
                Model(
                    name=off.name,
                    genome_dir=base.genome_dir,
                    adapter_path=Path(off.adapter_path) if off.adapter_path else None,
                    fitness=off.fitness,
                    scores=dict(off.scores),
                    generation_born=generation,
                    parent_survivor=off.parent_survivor,
                    mutator=off.mutator,
                    backend=off.backend,
                    is_survivor=False,
                    mutation_failed=off.mutation_failed,
                    finetune_failed=off.finetune_failed,
                )
            )
        return Population(models=models)
