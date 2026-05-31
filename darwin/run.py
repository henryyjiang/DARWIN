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

    @classmethod
    def from_dict(cls, root: Path, d: dict[str, Any]) -> "RunPaths":
        root = Path(root)
        g = lambda k, default: Path(d.get(k, root / default))  # noqa: E731
        return cls(
            root=root,
            workspace=g("workspace", "models"),
            runs=g("runs", "runs"),
            memory=g("memory", "memory"),
            base_genome=Path(d["base_genome"]) if "base_genome" in d else root / "base_genome",
            cost_ledger=g("cost_ledger", "runs/cost.jsonl"),
        )


@dataclass
class RunSpec:
    """A fully-parsed run configuration."""

    config: DarwinConfig
    paths: RunPaths
    generations: int = 1
    smoke_command: list[str] = field(default_factory=list)
    finetune_command: list[str] = field(default_factory=list)
    benchmark_command: list[str] = field(default_factory=list)
    seed_scores: dict[str, dict[str, float]] = field(default_factory=dict)


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
    return RunSpec(
        config=config,
        paths=paths,
        generations=int(data.get("generations", 1)),
        smoke_command=list(data.get("smoke_command", [])),
        finetune_command=list(data.get("finetune_command", [])),
        benchmark_command=list(data.get("benchmark_command", [])),
        seed_scores=dict(data.get("seed_scores", {})),
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
) -> tuple[Controller, MemoryStore, CostLedger]:
    """Assemble the controller from a RunSpec + injected backends (the testable seam)."""
    store = MemoryStore(spec.paths.memory)
    ledger = CostLedger(spec.paths.cost_ledger)
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

    from darwin.global_memory import ClaudeSynthesizer

    controller, _store, _ledger = build_controller(
        spec,
        mutation_backend_factory=_default_mutation_factory(spec),
        synthesizer=ClaudeSynthesizer(),
    )
    population = bootstrap_or_load_population(spec)
    print(f"[darwin] running {spec.generations} generation(s) from {len(population.models)} models")
    final = controller.run(spec.generations, population)
    print(f"[darwin] done. final population: {[m.name for m in final.models]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
