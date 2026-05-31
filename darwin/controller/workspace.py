"""Workspace materialization — the on-disk `models/` layout (ARCHITECTURE.md §3.1 / §9.1).

The controller reasons over an in-memory `Population`; this module reconciles that with the
on-disk model directories so the loop is actually runnable and the GA's keep-5/drop-5 shows up
on disk:

- `bootstrap_population` — seed the gen-0 population: N **survivor** dirs (each a copy of a base
  genome template, optionally carrying cached benchmark scores) + N **offspring slot** dirs, and
  return the 2N-model `Population` the controller starts from. (Solves "5 Qwen models loaded in
  models/ with their benchmark logs" + the 5 empty slots the GA fills.)
- `reset_slot` — wipe one model's dir (genome + adapter) so the next clone is clean. This is the
  "**remove the models the GA dropped**" step: a culled/recycled offspring slot is cleared before
  a fresh offspring is cloned into it. **Survivor dirs are never touched** — they persist with
  their genome, adapter, and memory, so the resting set is the 5 survivors.
- `materialize_model` — copy an offspring's produced genome + adapter (+ memory) **back into**
  `models/<name>/` from wherever it ran (a container workdir for `ContainerGenerationOps`; a no-op
  when it already ran in place). This is the "**move the models back to the models folder from the
  docker containers**" step.

Pure filesystem helpers, no controller dependency beyond `Model`/`Population`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from darwin.controller.population import Model, Population

GENOME = "genome"
ADAPTER = "adapter.bin"
MEMORY = "memory"


def model_dir(workspace: Path | str, name: str) -> Path:
    return Path(workspace) / name


def genome_dir(workspace: Path | str, name: str) -> Path:
    return model_dir(workspace, name) / GENOME


def adapter_path(workspace: Path | str, name: str) -> Path:
    return model_dir(workspace, name) / ADAPTER


def bootstrap_population(
    workspace: Path | str,
    base_genome: Path | str,
    *,
    num_survivors: int = 5,
    num_offspring: int = 5,
    survivor_names: list[str] | None = None,
    offspring_names: list[str] | None = None,
    survivor_scores: dict[str, dict[str, float]] | None = None,
    survivor_fitness: dict[str, float] | None = None,
) -> Population:
    """Seed the gen-0 population on disk and return it (§3.1).

    Each survivor gets a copy of `base_genome` (the shared starting recipe) under
    `models/<name>/genome`; cached `survivor_scores`/`survivor_fitness` (from the seeds'
    pre-run benchmark, §6.2) travel with them so the gen-0 fitness baseline exists. Offspring
    slots are created empty (no genome yet) — the controller clones a survivor into each at SPAWN.
    """
    ws = Path(workspace)
    base = Path(base_genome)
    if not base.exists():
        raise FileNotFoundError(f"base genome template not found: {base}")
    survivor_names = survivor_names or [f"s{i}" for i in range(num_survivors)]
    offspring_names = offspring_names or [f"o{i}" for i in range(num_offspring)]
    scores = survivor_scores or {}
    fitness = survivor_fitness or {}

    models: list[Model] = []
    for name in survivor_names:
        g = genome_dir(ws, name)
        if g.exists():
            shutil.rmtree(g)
        g.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, g, ignore=shutil.ignore_patterns(".git"))
        models.append(
            Model(
                name=name,
                genome_dir=g,
                adapter_path=adapter_path(ws, name),
                fitness=fitness.get(name),
                scores=dict(scores.get(name, {})),
                is_survivor=True,
            )
        )
    for name in offspring_names:
        model_dir(ws, name).mkdir(parents=True, exist_ok=True)
        models.append(
            Model(name=name, genome_dir=genome_dir(ws, name),
                  adapter_path=adapter_path(ws, name), is_survivor=False)
        )
    return Population(models=models)


def reset_slot(workspace: Path | str, name: str) -> None:
    """Wipe a model's genome + adapter so the next spawn re-clones fresh (drop step)."""
    g = genome_dir(workspace, name)
    if g.exists():
        shutil.rmtree(g)
    a = adapter_path(workspace, name)
    if a.exists():
        a.unlink()


def materialize_model(
    workspace: Path | str,
    name: str,
    *,
    genome_src: Path | str,
    adapter_src: Path | str | None = None,
    memory_src: Path | str | None = None,
) -> None:
    """Copy an offspring's results back into `models/<name>/` (move-back step).

    Used by `ContainerGenerationOps` to pull the produced genome/adapter/memory out of the
    container workdir. A no-op-equivalent when `genome_src` already *is* the model's dir.
    """
    ws = Path(workspace)
    dst_genome = genome_dir(ws, name)
    if Path(genome_src).resolve() != dst_genome.resolve():
        if dst_genome.exists():
            shutil.rmtree(dst_genome)
        dst_genome.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(genome_src, dst_genome)
    if adapter_src is not None:
        dst_adapter = adapter_path(ws, name)
        if Path(adapter_src).resolve() != dst_adapter.resolve():
            dst_adapter.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(adapter_src, dst_adapter)
    if memory_src is not None:
        dst_mem = model_dir(ws, name) / MEMORY
        if Path(memory_src).resolve() != dst_mem.resolve():
            if dst_mem.exists():
                shutil.rmtree(dst_mem)
            shutil.copytree(memory_src, dst_mem)
