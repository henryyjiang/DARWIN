"""Container generation ops: run each offspring stage *inside* the §8.5 images (ARCHITECTURE.md §8).

The container counterpart of `LocalGenerationOps`. `LocalGenerationOps` runs every stage in-process
on the host; `ContainerGenerationOps` runs them in the sandboxed Docker images (`darwin-agent`,
`darwin-finetune`, `darwin-eval`) that are the actual §8 security boundary for autonomous
`bypassPermissions` agents.

Only the **mutation window** needs new orchestration here: the finetune and benchmark stages are
already injected backends, so passing a `ContainerFinetuneBackend` / `EvalContainerBenchmarkBackend`
makes those run in their containers with no change to the controller. So this class composes a
`LocalGenerationOps` (with the container backends) for `spawn` / `finetune` / `benchmark` /
`reset_offspring_slot`, and overrides `mutate` to launch the `darwin-agent` container whose process
is `darwin.mutation_agent.entrypoint` (§4.5/§4.6).

Path mapping & memory handoff (the design points from IMPLEMENTATION.md §4):
- The offspring's `models/<name>/genome` dir is bind-mounted **rw** at `/work/genome`, so the
  agent's edits + Git checkpoints land directly on the host — no separate "move back" step.
- A per-offspring **scratch** dir is bind-mounted **rw** at `/work/scratch`; the in-container store
  + the result JSON are written under it. Before launch we **seed** the scratch store with this
  lineage's prior memory + the global memory (so the agent's ORIENT phase can read them); after the
  window we **ingest** the newly written iteration file back into the host store, and read the
  result JSON for `final_commit` / `mutation_failed`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from darwin.bench.job import BenchmarkBackend
from darwin.config import DarwinConfig
from darwin.controller.controller import FinetuneOutcomeView, MutateOutcome
from darwin.controller.ops import LocalGenerationOps
from darwin.controller.population import Model
from darwin.controller.state import OffspringState
from darwin.cost import CostLedger
from darwin.finetune import FinetuneBackend
from darwin.fsutil import force_rmtree
from darwin.memory import MemoryStore
from darwin.memory.store import GLOBAL_SECTIONS
from darwin.mutation_agent.checkpoint import GitCheckpointer
from darwin.sandbox import (
    GENOME_PATH,
    SCRATCH_PATH,
    DockerContainerRunner,
    agent_container,
)
from darwin.sandbox.runner import ContainerRunner


def _never(*_a, **_k):  # pragma: no cover - guard: container mutate doesn't use a host factory
    raise RuntimeError(
        "ContainerGenerationOps runs the mutation backend inside the agent container; the "
        "host-side mutation_backend_factory is not used."
    )


@dataclass
class ContainerGenerationOps:
    """Runs an offspring's stages inside the §8.5 containers via an injected `ContainerRunner`."""

    config: DarwinConfig
    store: MemoryStore
    ledger: CostLedger
    workspace: Path
    finetune_backend: FinetuneBackend  # a ContainerFinetuneBackend for the live path
    benchmark_backend: BenchmarkBackend  # an EvalContainerBenchmarkBackend for the live path
    smoke_command: list[str]
    container_runner: ContainerRunner = field(default_factory=DockerContainerRunner)
    agent_image: str = "darwin-agent"
    # Agent network policy (§8.3): "whitelist" (default, the egress-firewalled network) or "open"
    # (host network, dev-only — used by the test profile's real-Claude path to reach the API
    # without standing up the whitelist proxy; never for real autonomous runs, TEST_RUN_PLAN §6).
    agent_network: str = "whitelist"
    base_model: str = "base"
    agent_env: dict[str, str] = field(default_factory=dict)  # API keys etc. passed into the window
    # §6.2 host-only held-out slice provider: slice_id -> the slice dir bind-mounted read-only
    # into the eval container (the container eval backend requires it).
    eval_slice_dir: Callable[[int], Path | None] | None = None

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        # spawn / finetune / benchmark / reset are identical to the local path (the container
        # backends are injected), so delegate to a composed LocalGenerationOps.
        self._local = LocalGenerationOps(
            config=self.config,
            store=self.store,
            ledger=self.ledger,
            workspace=self.workspace,
            mutation_backend_factory=_never,  # unused on this path (mutate is overridden)
            finetune_backend=self.finetune_backend,
            benchmark_backend=self.benchmark_backend,
            smoke_command=self.smoke_command,
            base_model=self.base_model,
            eval_slice_dir=self.eval_slice_dir,
        )

    # ---------------------------------------------------------------- delegated stages
    def reset_offspring_slot(self, name: str) -> None:
        self._local.reset_offspring_slot(name)

    def spawn(self, *, offspring: OffspringState, parent: Model, generation: int) -> Model:
        return self._local.spawn(offspring=offspring, parent=parent, generation=generation)

    def finetune(
        self, *, offspring: Model, state: OffspringState, generation: int
    ) -> FinetuneOutcomeView:
        return self._local.finetune(offspring=offspring, state=state, generation=generation)

    def benchmark(
        self, *, offspring: Model, state: OffspringState, slice_id: int, generation: int
    ) -> dict[str, float]:
        return self._local.benchmark(
            offspring=offspring, state=state, slice_id=slice_id, generation=generation
        )

    # ---------------------------------------------------------------- MUTATE (in-container, §4)
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
        scratch = self.workspace / offspring.name / "scratch"
        if scratch.exists():
            force_rmtree(scratch)  # fresh window: no stale result/store from a prior attempt
        scratch.mkdir(parents=True, exist_ok=True)

        container_store_root = f"{SCRATCH_PATH}/store"
        host_scratch_store = scratch / "store"
        self._seed_container_store(host_scratch_store, offspring.name)

        base_fitness = parent.fitness if parent.fitness is not None else 0.0
        env = dict(self.agent_env)
        env.update(
            self._window_env(
                offspring, state, mutator_name, generation, container_store_root, base_fitness
            )
        )

        spec = agent_container(
            offspring_id=offspring.name,
            genome_host=str(offspring.genome_dir),
            memory_host=str(self._ensure_model_memory_dir(offspring.name)),
            scratch_host=str(scratch),
            command=[],  # the image CMD is `python -m darwin.mutation_agent.entrypoint`
            env=env,
        )
        spec.image = self.agent_image
        spec.network = self.agent_network  # §8.3: whitelist (default) or open (dev-only, §6)

        result = self.container_runner.run(spec)

        # Always persist the agent container's full output (entrypoint + `[claude]` session lines:
        # start/end, ResultMessage, tool-use count) to a stable per-model log for auditing.
        log = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        log_path = self.workspace / offspring.name / "agent.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(log, encoding="utf-8")
        except OSError:  # pragma: no cover - best-effort logging
            pass

        payload = self._read_result(scratch / "result.json")
        mutation_failed = (
            True if payload is None else bool(payload.get("mutation_failed", True))
        ) or (not result.ok)
        # One-line summary per offspring + (on failure) the log tail, so a window that "finished
        # instantly with no work" tells you why instead of being silent.
        print(
            f"[darwin] {offspring.name} window: exit={result.exit_code} "
            f"mutation_failed={mutation_failed} log={log_path}"
        )
        if mutation_failed and log:
            print(f"[darwin] --- {offspring.name} container log tail ---\n{log[-2500:]}\n[darwin] --- end ---")
        if not result.ok and payload is None:
            # the container crashed and wrote nothing: the genome is whatever the agent left;
            # treat as a failed mutation and report the current HEAD (the clone of S, or a green
            # checkpoint if one was made before the crash).
            return MutateOutcome(final_commit=self._safe_head(offspring), mutation_failed=True)

        # surface the newly written memory file from the scratch store into the host store (§7.2)
        self._ingest_memory(host_scratch_store, offspring.name, state.iteration)

        if payload is None:
            return MutateOutcome(final_commit=self._safe_head(offspring), mutation_failed=True)
        return MutateOutcome(
            final_commit=payload.get("final_commit"),
            mutation_failed=bool(payload.get("mutation_failed", True)),
        )

    # ---------------------------------------------------------------- helpers
    def _window_env(
        self,
        offspring: Model,
        state: OffspringState,
        mutator_name: str,
        generation: int,
        container_store_root: str,
        base_fitness: float,
    ) -> dict[str, str]:
        m = self.config.mutation
        return {
            "DARWIN_OFFSPRING_ID": offspring.name,
            "DARWIN_MODEL": offspring.name,
            "DARWIN_PARENT_SURVIVOR": state.parent_survivor,
            "DARWIN_MUTATOR": mutator_name,
            "DARWIN_GENERATION": str(generation),
            "DARWIN_ITERATION": str(state.iteration),
            "DARWIN_BACKEND": state.backend,
            "DARWIN_BASE_FITNESS": str(base_fitness),
            "DARWIN_GENOME_DIR": GENOME_PATH,
            "DARWIN_STORE_ROOT": container_store_root,
            "DARWIN_RESULT_OUT": f"{SCRATCH_PATH}/result.json",
            "DARWIN_SMOKE_CMD": json.dumps(list(self.smoke_command)),
            "DARWIN_WINDOW_H": str(m.mutation_window_h),
            "DARWIN_SOFT_DEADLINE_MIN": str(m.soft_deadline_min),
            "DARWIN_KILL_GRACE_MIN": str(m.kill_grace_min),
            # directive style + Claude session knobs (real-Claude path; ignored by mock/local)
            "DARWIN_DIRECTIVE_STYLE": m.directive_style,
            "DARWIN_CLAUDE_MODEL": m.claude_model,
            "DARWIN_CLAUDE_EFFORT": m.claude_effort,
            "DARWIN_CLAUDE_MAX_BUDGET_USD": str(m.claude_max_budget_usd),
        }

    def _ensure_model_memory_dir(self, model: str) -> Path:
        d = self.store.model_memory_dir(model)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _seed_container_store(self, scratch_store_root: Path, model: str) -> None:
        """Copy this lineage's prior memory + the global memory into the scratch store (ORIENT)."""
        scratch = MemoryStore(scratch_store_root)
        # prior per-model iterations
        src_mem = self.store.model_memory_dir(model)
        if src_mem.exists():
            dst_mem = scratch.model_memory_dir(model)
            dst_mem.mkdir(parents=True, exist_ok=True)
            for f in src_mem.glob("iter_*.md"):
                shutil.copy2(f, dst_mem / f.name)
        # global memory (read-only context for the agent)
        if self.store.global_dir.exists():
            scratch.global_dir.mkdir(parents=True, exist_ok=True)
            for filename in GLOBAL_SECTIONS.values():
                src = self.store.global_dir / filename
                if src.exists():
                    shutil.copy2(src, scratch.global_dir / filename)

    def _ingest_memory(self, scratch_store_root: Path, model: str, iteration: int) -> None:
        """Copy the iteration file the agent wrote in the container back to the host store (§7.2)."""
        produced = MemoryStore(scratch_store_root).iter_path(model, iteration)
        if not produced.exists():
            return  # no memory written; the controller's §4.3 synthesis fallback handles it
        dst = self.store.iter_path(model, iteration)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, dst)

    def _safe_head(self, offspring: Model) -> str | None:
        try:
            return GitCheckpointer(offspring.genome_dir).head()
        except Exception:  # pragma: no cover - genome may not be a repo on a hard crash
            return None

    @staticmethod
    def _read_result(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:  # pragma: no cover - corrupt handoff
            return None
