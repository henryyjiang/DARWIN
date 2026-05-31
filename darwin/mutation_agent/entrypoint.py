"""In-container mutation entrypoint (ARCHITECTURE.md §4 / §8.5).

The default command the `darwin-agent` image runs: it is the in-container counterpart of the
finetune/eval entrypoints. `LocalGenerationOps` runs the §4.2 mutation window *in-process*;
`ContainerGenerationOps` instead launches the agent container, whose process is **this module** —
it reads the window parameters from the `DARWIN_*` env the container backend set, assembles the
`MutationContext` against the mounted genome, resolves the backend (the live Claude/OpenHands
session, §4.5/§4.6), runs `run_mutation_window` (which guarantees an always-green final genome),
and writes a small JSON result the host reads back (`final_commit`, `mutation_failed`, and the
model/iteration so the produced memory file can be materialized back into the host store).

Design mirrors the other entrypoints: the config parsing (`MutationRunConfig.from_env`,
`parse_command`) and the result handoff (`write_result`/`read_result`) are pure and unit-tested;
`run_window` is exercised end-to-end with a fake backend over a real temp git genome (no
Claude/Docker), and `main()` only adds the live backend-factory resolution.
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from darwin.memory import MemoryStore
from darwin.mutation_agent.backend import MutationBackend, MutationContext, MutationResult
from darwin.mutation_agent.checkpoint import GitCheckpointer
from darwin.mutation_agent.deadline import DeadlineManager
from darwin.mutation_agent.directive import build_directive
from darwin.mutation_agent.runner import run_mutation_window
from darwin.mutation_agent.smoke import SmokeTest

# Canonical in-container defaults (match darwin.sandbox.roles); overridable via env.
DEFAULT_GENOME_DIR = "/work/genome"
DEFAULT_STORE_ROOT = "/work/scratch/store"
DEFAULT_RESULT_OUT = "/work/scratch/result.json"

# A factory mapping (backend_name, ctx) -> MutationBackend (the §4.7 router / live seam).
BackendFactory = Callable[[str, MutationContext], MutationBackend]


def _env_float(env, key, default):
    try:
        return float(env[key])
    except (KeyError, ValueError):
        return default


def _env_int(env, key, default):
    try:
        return int(env[key])
    except (KeyError, ValueError):
        return default


def parse_command(raw: str) -> list[str]:
    """Parse a command string into argv: a JSON array if it is one, else POSIX shell-split."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except ValueError:
            pass
    return shlex.split(raw)


@dataclass
class MutationRunConfig:
    """Resolved mutation-window parameters (the controller/container backend sets these, §4.2)."""

    offspring_id: str = ""
    model: str = ""
    parent_survivor: str = ""
    mutator: str = "claude"  # the schema requires a non-empty mutator (§7.2)
    generation: int = 0
    iteration: int = 0
    backend_name: str = "claude"  # "claude" | "local"
    base_fitness: float = 0.0
    genome_dir: str = DEFAULT_GENOME_DIR
    store_root: str = DEFAULT_STORE_ROOT
    result_out: str = DEFAULT_RESULT_OUT
    smoke_command: list[str] = field(default_factory=list)
    window_h: float = 3.0
    soft_deadline_min: float = 20.0
    kill_grace_min: float = 5.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "MutationRunConfig":
        env = dict(os.environ if env is None else env)
        return cls(
            offspring_id=env.get("DARWIN_OFFSPRING_ID", ""),
            model=env.get("DARWIN_MODEL", ""),
            parent_survivor=env.get("DARWIN_PARENT_SURVIVOR", ""),
            mutator=env.get("DARWIN_MUTATOR", "") or "claude",
            generation=_env_int(env, "DARWIN_GENERATION", 0),
            iteration=_env_int(env, "DARWIN_ITERATION", 0),
            backend_name=env.get("DARWIN_BACKEND", "claude"),
            base_fitness=_env_float(env, "DARWIN_BASE_FITNESS", 0.0),
            genome_dir=env.get("DARWIN_GENOME_DIR", DEFAULT_GENOME_DIR),
            store_root=env.get("DARWIN_STORE_ROOT", DEFAULT_STORE_ROOT),
            result_out=env.get("DARWIN_RESULT_OUT", DEFAULT_RESULT_OUT),
            smoke_command=parse_command(env.get("DARWIN_SMOKE_CMD", "")),
            window_h=_env_float(env, "DARWIN_WINDOW_H", 3.0),
            soft_deadline_min=_env_float(env, "DARWIN_SOFT_DEADLINE_MIN", 20.0),
            kill_grace_min=_env_float(env, "DARWIN_KILL_GRACE_MIN", 5.0),
        )


def build_context(cfg: MutationRunConfig, *, store: MemoryStore | None = None) -> MutationContext:
    """Assemble the `MutationContext` for the window against the mounted genome (§4.2)."""
    genome = Path(cfg.genome_dir)
    return MutationContext(
        offspring_id=cfg.offspring_id,
        genome_dir=genome,
        model=cfg.model,
        parent_survivor=cfg.parent_survivor,
        mutator=cfg.mutator,
        generation=cfg.generation,
        iteration=cfg.iteration,
        backend_name=cfg.backend_name,
        base_fitness=cfg.base_fitness,
        directive=build_directive(
            offspring_id=cfg.offspring_id,
            model=cfg.model,
            parent_survivor=cfg.parent_survivor,
            mutator=cfg.mutator,
            generation=cfg.generation,
        ),
        checkpointer=GitCheckpointer(genome),
        smoke=SmokeTest(command=cfg.smoke_command),
        store=store if store is not None else MemoryStore(cfg.store_root),
    )


def build_deadline(cfg: MutationRunConfig) -> DeadlineManager:
    return DeadlineManager(
        window_s=cfg.window_h * 3600.0,
        soft_lead_s=cfg.soft_deadline_min * 60.0,
        kill_grace_s=cfg.kill_grace_min * 60.0,
    )


def write_result(path: str | Path, result: MutationResult, *, model: str, iteration: int) -> Path:
    """Write the window result as JSON for the host to read back (the §4.2 handoff)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(result), "model": model, "iteration": iteration}
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def read_result(path: str | Path) -> dict:
    """Read back a result JSON written by `write_result`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_window(
    cfg: MutationRunConfig,
    backend: MutationBackend,
    *,
    deadline: DeadlineManager | None = None,
    store: MemoryStore | None = None,
) -> MutationResult:
    """Run the window and write the result handoff; returns the `MutationResult`."""
    ctx = build_context(cfg, store=store)
    result = run_mutation_window(ctx, backend, deadline or build_deadline(cfg))
    write_result(cfg.result_out, result, model=cfg.model, iteration=cfg.iteration)
    return result


def _default_backend_factory() -> BackendFactory:  # pragma: no cover - live seam
    """The live default: route claude/local, serving the local model for `local` (§4.6)."""
    from darwin.mutation_agent.local_backend import make_mutation_backend_factory

    serve = None
    if os.environ.get("DARWIN_BACKEND") == "local":
        from darwin.mutation_agent.vllm_serving import VLLMServeConfig

        serve = VLLMServeConfig(base_model=os.environ.get("DARWIN_BASE_MODEL", ""))
    return make_mutation_backend_factory(serve_config=serve)


def main(
    env: dict[str, str] | None = None,
    *,
    backend_factory: BackendFactory | None = None,
) -> int:
    """Drive the in-container mutation window. `backend_factory` is the live seam (injectable)."""
    cfg = MutationRunConfig.from_env(env)
    print(f"[darwin-agent] offspring={cfg.offspring_id} model={cfg.model} "
          f"backend={cfg.backend_name} gen={cfg.generation} iter={cfg.iteration}")
    ctx = build_context(cfg)
    factory = backend_factory or _default_backend_factory()
    backend = factory(cfg.backend_name, ctx)
    result = run_mutation_window(ctx, backend, build_deadline(cfg))
    write_result(cfg.result_out, result, model=cfg.model, iteration=cfg.iteration)
    print(f"[darwin-agent] final_commit={result.final_commit} "
          f"mutation_failed={result.mutation_failed} -> {cfg.result_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
