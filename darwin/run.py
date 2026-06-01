"""Top-level run entrypoint — assemble and drive the generational loop (ARCHITECTURE.md §2 / §9.1).

This is the `main` the operator runs. It:
1. loads a run config (YAML) into a `DarwinConfig` + filesystem paths,
2. **bootstraps** the gen-0 population on disk (5 survivor seeds + 5 offspring slots, §3.1) or
   **resumes** from the latest persisted generation state (§2.3),
3. assembles the controller — `LocalGenerationOps` wiring the mutation/finetune/benchmark
   backends, the `darwin-mcp` attachment, the cost ledger + `BudgetGuard`, the §6.4 anti-gaming
   scanner, and the §7.4 Claude global-memory pass — and
4. runs `Controller.run(generations)`.

`build_controller` is the assembly seam: `main` wires the live defaults (Claude/local mutation,
subprocess finetune/eval from the genome's declared commands, `ClaudeSynthesizer`), while tests
inject fakes to exercise the whole wiring without GPU/Docker/Claude.

    python -m darwin --config run.yaml --generations 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from darwin.bench import SubprocessBenchmarkBackend
from darwin.config import DarwinConfig
from darwin.controller import (
    Controller,
    GenerationStateStore,
    LocalGenerationOps,
    Population,
    bootstrap_population,
)
from darwin.controller.state import GenerationStateStore as _GSS
from darwin.cost import BudgetGuard, CostLedger
from darwin.finetune import SubprocessFinetuneBackend
from darwin.memory import MemoryStore

_SUBCONFIGS = ("ga", "mutation", "fitness", "cost", "finetune", "benchmark", "antigaming")


@dataclass
class RunPaths:
    """Filesystem layout for a run (cross-platform; defaults under the run root)."""

    root: Path
    workspace: Path  # models/ (population dirs)
    runs: Path  # runs/ (gen_<n>/state.json)
    memory: Path  # memory/ (global + per-model store root)
    base_genome: Path  # the shared starting genome template (cloned into seeds)
    cost_ledger: Path
    eval_slices: Path  # host-only held-out slices: eval_slices/<slice_id>/ (container mode, §6.2)

    @classmethod
    def from_dict(cls, root: Path, d: dict[str, Any]) -> "RunPaths":
        # Resolve every path to ABSOLUTE (relative entries are taken relative to the config-file
        # dir). Container mode bind-mounts these into Docker, and `docker -v` requires an absolute
        # host path — a relative one is read as a *named volume* (e.g. `models\s0` →
        # "invalid characters for a local volume name"). Absolute paths also make the run
        # independent of the process working directory.
        root = Path(root).resolve()

        def g(key: str, default: str) -> Path:
            raw = Path(d[key]) if key in d else Path(default)
            return raw if raw.is_absolute() else (root / raw).resolve()

        return cls(
            root=root,
            workspace=g("workspace", "models"),
            runs=g("runs", "runs"),
            memory=g("memory", "memory"),
            base_genome=g("base_genome", "base_genome"),
            cost_ledger=g("cost_ledger", "runs/cost.jsonl"),
            eval_slices=g("eval_slices", "eval_slices"),
        )


@dataclass
class RunSpec:
    """A fully-parsed run configuration."""

    config: DarwinConfig
    paths: RunPaths
    generations: int = 1
    mode: str = "local"  # "local" (in-process) | "container" (run stages in the §8.5 images)
    # "" (live defaults) | "test" (the budget-free mock profile, TEST_RUN_PLAN): mock global
    # synthesizer by default, and the loop is wired to the mock entrypoints via images/commands.
    profile: str = ""
    smoke_command: list[str] = field(default_factory=list)
    finetune_command: list[str] = field(default_factory=list)
    benchmark_command: list[str] = field(default_factory=list)
    seed_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-role container images (container mode). Defaults are the three §8.5 images; the test
    # profile points all three at the one slim `darwin-agent` image (TEST_RUN_PLAN §3.5/§4).
    images: dict[str, str] = field(default_factory=dict)
    # Agent network policy for container mode (§8.3): "whitelist" (default) | "open" (dev-only).
    agent_network: str = "whitelist"
    # Opt-in: run the *real* Claude global-memory pass (capped) instead of the mock synthesizer,
    # when a profile would otherwise mock it and an ANTHROPIC_API_KEY is present (TEST_RUN_PLAN §3.4).
    real_global_memory: bool = False
    global_memory_model: str = ""  # override the global-pass model (e.g. a cheaper one for testing)
    global_memory_effort: str = ""  # override the global-pass effort ("low"/"medium"/"high")


def apply_overrides(config: DarwinConfig, overrides: dict[str, Any]) -> DarwinConfig:
    """Apply a nested `{subconfig: {field: value}}` mapping onto a DarwinConfig (in place)."""
    if "run_name" in overrides:
        config.run_name = overrides["run_name"]
    for name in _SUBCONFIGS:
        sub = overrides.get(name)
        if not isinstance(sub, dict):
            continue
        target = getattr(config, name)
        for key, value in sub.items():
            if hasattr(target, key):
                setattr(target, key, value)
    return config


def load_run_spec(path: Path | str) -> RunSpec:
    """Parse a run-config YAML into a RunSpec."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = apply_overrides(DarwinConfig(), data.get("config", {}))
    paths = RunPaths.from_dict(path.parent, data.get("paths", {}))
    commands = data.get("commands", {}) or {}
    return RunSpec(
        config=config,
        paths=paths,
        generations=int(data.get("generations", 1)),
        mode=str(data.get("mode", "local")),
        profile=str(data.get("profile", "")),
        # `commands.finetune`/`commands.benchmark` are aliases for the top-level *_command keys;
        # the test profile's YAML uses the `commands:` block (TEST_RUN_PLAN §4).
        smoke_command=list(data.get("smoke_command", [])),
        finetune_command=list(commands.get("finetune", data.get("finetune_command", []))),
        benchmark_command=list(commands.get("benchmark", data.get("benchmark_command", []))),
        seed_scores=dict(data.get("seed_scores", {})),
        images=dict(data.get("images", {})),
        agent_network=str(data.get("agent_network", "whitelist")),
        real_global_memory=bool(data.get("real_global_memory", False)),
        global_memory_model=str(data.get("global_memory_model", "")),
        global_memory_effort=str(data.get("global_memory_effort", "")),
    )


def bootstrap_or_load_population(spec: RunSpec) -> Population:
    """Resume the latest generation's population if present, else bootstrap gen-0 (§2.3/§3.1)."""
    store = _GSS(spec.paths.runs)
    latest = store.latest_generation()
    if latest is not None and store.exists(latest):
        state = store.load(latest)
        # the next population formed by the latest generation (resume point)
        if state.population_out:
            return Population.from_dict(state.population_out)
        return Population.from_dict(state.population_in)
    ga = spec.config.ga
    return bootstrap_population(
        spec.paths.workspace,
        spec.paths.base_genome,
        num_survivors=ga.num_survivors,
        num_offspring=ga.population_size - ga.num_survivors,
        survivor_scores=spec.seed_scores,
    )


def build_controller(
    spec: RunSpec,
    *,
    mutation_backend_factory,
    finetune_backend=None,
    benchmark_backend=None,
    synthesizer=None,
    antigaming=None,
    enable_global_memory: bool = True,
    base_model: str | None = None,
    container_runner=None,
) -> tuple[Controller, MemoryStore, CostLedger]:
    """Assemble the controller from a RunSpec + injected backends (the testable seam).

    `spec.mode` selects the execution path: `local` runs every stage in-process via
    `LocalGenerationOps` (default); `container` runs the window/finetune/eval inside the §8.5
    images via `ContainerGenerationOps` + the container backends (a real Docker host is needed
    at run time, but the assembly is identical and testable).
    """
    # The MemoryStore is rooted at the RUN ROOT, not the `memory/` dir: it derives the per-model
    # notebooks at `<root>/models/<model>/memory/iter_<n>.md` (co-located with the population
    # genomes under `models/`) and global memory at `<root>/memory/global/` (ARCHITECTURE §7.2/§7.3
    # — both repo-top-level). Passing `paths.memory` here would double it to `memory/memory/global`
    # and orphan the seeded global templates the run is meant to read.
    store = MemoryStore(spec.paths.root)
    ledger = CostLedger(spec.paths.cost_ledger)
    if spec.mode == "container":
        ops = _build_container_ops(
            spec, store, ledger, finetune_backend, benchmark_backend, base_model,
            container_runner,
        )
    else:
        ops = LocalGenerationOps(
            config=spec.config,
            store=store,
            ledger=ledger,
            workspace=spec.paths.workspace,
            mutation_backend_factory=mutation_backend_factory,
            finetune_backend=finetune_backend
            or SubprocessFinetuneBackend(command=spec.finetune_command or ["true"]),
            benchmark_backend=benchmark_backend
            or SubprocessBenchmarkBackend(command=spec.benchmark_command or ["true"]),
            smoke_command=spec.smoke_command or ["true"],
            base_model=base_model or spec.config.finetune.base_model,
        )
    budget = BudgetGuard(ledger, spec.config.cost) if spec.config.cost.gen_budget_usd else None
    controller = Controller(
        config=spec.config,
        store=store,
        ledger=ledger,
        state_store=GenerationStateStore(spec.paths.runs),
        ops=ops,
        synthesizer=synthesizer,
        antigaming=antigaming,
        budget=budget,
        enable_global_memory=enable_global_memory,
    )
    return controller, store, ledger


def _build_container_ops(
    spec: RunSpec, store, ledger, finetune_backend, benchmark_backend, base_model,
    container_runner=None,
):
    """Assemble `ContainerGenerationOps` wiring the §8.5 images (container mode).

    `container_runner` is injectable so the assembly is testable with a fake; `main` leaves it
    None to get the live `DockerContainerRunner`."""
    import os

    from darwin.bench import EvalContainerBenchmarkBackend
    from darwin.controller import ContainerGenerationOps
    from darwin.finetune import ContainerFinetuneBackend
    from darwin.sandbox import DockerContainerRunner

    runner = container_runner or DockerContainerRunner()
    resolved_base = base_model or spec.config.finetune.base_model
    # forward the agent's API key into the sandboxed window (the only secret it needs, §8.2)
    agent_env = {
        k: os.environ[k] for k in ("ANTHROPIC_API_KEY",) if os.environ.get(k)
    }
    # Per-role images (TEST_RUN_PLAN §3.5): default to the three §8.5 images; the test profile
    # points all three at the one slim darwin-agent image and overrides finetune/eval commands
    # with the mock entrypoints.
    images = spec.images
    agent_image = images.get("agent", "darwin-agent")
    finetune_image = images.get("finetune", "darwin-finetune")
    eval_image = images.get("eval", "darwin-eval")

    ft_backend = finetune_backend or ContainerFinetuneBackend(
        runner=runner,
        base_model=resolved_base,
        image=finetune_image,
        command=list(spec.finetune_command),  # [] => the image CMD (reference entrypoint)
    )
    ev_backend = benchmark_backend or EvalContainerBenchmarkBackend(
        runner=runner,
        image=eval_image,
        command=list(spec.benchmark_command),  # [] => the image CMD (reference entrypoint)
    )

    # In the test profile the eval slice dirs are synthetic (the mock eval ignores their contents,
    # TEST_RUN_PLAN §4); auto-create them so the run is self-contained. Real runs ship real slices.
    slices_root = spec.paths.eval_slices

    def slice_dir(sid: int) -> Path:
        d = slices_root / str(sid)
        if spec.profile == "test":
            d.mkdir(parents=True, exist_ok=True)
        return d

    return ContainerGenerationOps(
        config=spec.config,
        store=store,
        ledger=ledger,
        workspace=spec.paths.workspace,
        finetune_backend=ft_backend,
        benchmark_backend=ev_backend,
        smoke_command=spec.smoke_command or ["python", "smoke_test.py"],
        container_runner=runner,
        agent_image=agent_image,
        agent_network=spec.agent_network,
        base_model=resolved_base,
        agent_env=agent_env,
        eval_slice_dir=slice_dir,
    )


def build_synthesizer(spec: RunSpec):
    """Pick the global-memory synthesizer for a run (TEST_RUN_PLAN §3.4).

    Live runs use `ClaudeSynthesizer` (the §7.4 default). The `test` profile uses the offline
    `MockSynthesizer` unless the operator opts into the real pass (`real_global_memory: true`) and
    an `ANTHROPIC_API_KEY` is present — in which case the real pass runs under a wall-clock cap
    (`CappedSynthesizer`) so the budget-free smoke test can never stall on it.
    """
    import os

    if spec.profile == "test":
        if spec.real_global_memory and os.environ.get("ANTHROPIC_API_KEY"):
            from darwin.global_memory import CappedSynthesizer, ClaudeSynthesizer
            from darwin.global_memory.synthesizer import DEFAULT_EFFORT, MODEL

            inner = ClaudeSynthesizer(
                model=spec.global_memory_model or MODEL,
                effort=spec.global_memory_effort or DEFAULT_EFFORT,
            )
            return CappedSynthesizer(inner, timeout_s=300.0)
        from darwin.global_memory import MockSynthesizer

        return MockSynthesizer()
    from darwin.global_memory import ClaudeSynthesizer

    return ClaudeSynthesizer()


def _default_mutation_factory(spec: RunSpec):
    """The live default factory: Claude / local backends, attaching darwin-mcp (§9.3)."""
    from darwin.mutation_agent import make_mutation_backend_factory

    serve = None
    if spec.config.mutation.backend in ("local", "mixed"):
        from darwin.mutation_agent import VLLMServeConfig

        serve = VLLMServeConfig(base_model=spec.config.finetune.base_model)
    return make_mutation_backend_factory(
        serve_config=serve,
        claude_transcript_dir=spec.paths.runs / "transcripts",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darwin", description=__doc__)
    parser.add_argument("--config", required=True, help="run-config YAML")
    parser.add_argument("--generations", type=int, default=None, help="override generations")
    args = parser.parse_args(argv)

    spec = load_run_spec(args.config)
    if args.generations is not None:
        spec.generations = args.generations

    controller, _store, _ledger = build_controller(
        spec,
        mutation_backend_factory=_default_mutation_factory(spec),
        synthesizer=build_synthesizer(spec),
    )
    population = bootstrap_or_load_population(spec)
    print(f"[darwin] running {spec.generations} generation(s) from {len(population.models)} models")
    final = controller.run(spec.generations, population)
    print(f"[darwin] done. final population: {[m.name for m in final.models]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
