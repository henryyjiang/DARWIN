"""Resumable generation state (ARCHITECTURE.md §2.3).

The controller persists each generation's state to `runs/gen_<n>/state.json` so a crash
mid-generation resumes **without re-running completed offspring**. Each transition is
idempotent: the controller advances the top-level `phase` and, within the per-offspring
pipeline, flips `mutation_done` / `finetune_done` / `benchmark_done` as each stage lands, so a
resume re-enters at the first incomplete stage of the first incomplete offspring.

This module is pure data + JSON I/O; the transition *logic* lives in the controller and is
tested there. `PHASE_ORDER` gives the canonical generation phase sequence (the §2.3 state
machine, minus the infra-only PROVISION which the controller owns separately).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Canonical generation phases, in order (§2.3). `at_least` compares against this.
PHASE_ORDER: tuple[str, ...] = (
    "spawned",          # offspring (S, M) pairings assigned
    "offspring_done",   # all offspring finished mutate -> finetune -> benchmark
    "aggregated",       # fitness reduced for every offspring
    "culled",           # GA cull done; next-gen population formed
    "global_memory",    # global-memory pass written
    "checkpoint",       # generation finalized
)


def _phase_index(phase: str) -> int:
    return PHASE_ORDER.index(phase)


@dataclass
class OffspringState:
    """Per-offspring progress through the mutate -> finetune -> benchmark pipeline (§2.3).

    The `*_done` flags are the resume gates; the result fields are filled as each stage lands
    so a resumed run reuses completed work rather than recomputing it.
    """

    name: str
    parent_survivor: str
    mutator: str | None  # None => Claude fallback (degenerate <2-survivor case, §3.2)
    backend: str
    iteration: int

    # MUTATE (§4)
    mutation_done: bool = False
    mutation_failed: bool = False
    final_commit: str | None = None

    # FINETUNE (§5)
    finetune_done: bool = False
    finetune_status: str | None = None  # "ok" | "finetune_failed" | "infra_failed"
    adapter_path: str | None = None

    # BENCHMARK (§6)
    benchmark_done: bool = False
    scores: dict[str, float] = field(default_factory=dict)

    # ANTI-GAMING (§6.4) — `antigaming_flags` is the penalty weight fed to fitness (§6.3)
    antigaming_done: bool = False
    antigaming_flags: int = 0

    # AGGREGATE_FITNESS (§6.3)
    fitness: float | None = None
    cost_usd: float = 0.0

    @property
    def finetune_failed(self) -> bool:
        return self.finetune_status == "finetune_failed"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OffspringState":
        return cls(**d)


@dataclass
class GenerationState:
    """The full persisted state of one generation (§2.3)."""

    generation: int
    phase: str = "spawned"
    population_in: dict = field(default_factory=dict)  # Population.to_dict() at gen start
    offspring: list[OffspringState] = field(default_factory=list)
    survivors_after_cull: list[str] | None = None
    population_out: dict | None = None  # Population.to_dict() after cull + reproduce
    completed: bool = False

    # ------------------------------------------------------------------ phase helpers
    def at_least(self, phase: str) -> bool:
        """True once the generation has reached (or passed) `phase`."""
        return _phase_index(self.phase) >= _phase_index(phase)

    def advance_to(self, phase: str) -> None:
        """Move the top-level phase forward (never backward — idempotent on resume)."""
        if _phase_index(phase) > _phase_index(self.phase):
            self.phase = phase

    def offspring_by_name(self) -> dict[str, OffspringState]:
        return {o.name: o for o in self.offspring}

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "phase": self.phase,
            "population_in": self.population_in,
            "offspring": [o.to_dict() for o in self.offspring],
            "survivors_after_cull": self.survivors_after_cull,
            "population_out": self.population_out,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationState":
        return cls(
            generation=d["generation"],
            phase=d.get("phase", "spawned"),
            population_in=d.get("population_in", {}),
            offspring=[OffspringState.from_dict(o) for o in d.get("offspring", [])],
            survivors_after_cull=d.get("survivors_after_cull"),
            population_out=d.get("population_out"),
            completed=d.get("completed", False),
        )


class GenerationStateStore:
    """Reads/writes `runs/gen_<n>/state.json` under a run root (§2.3)."""

    def __init__(self, runs_root: Path | str):
        self.runs_root = Path(runs_root)

    def gen_dir(self, generation: int) -> Path:
        return self.runs_root / f"gen_{generation}"

    def state_path(self, generation: int) -> Path:
        return self.gen_dir(generation) / "state.json"

    def exists(self, generation: int) -> bool:
        return self.state_path(generation).exists()

    def save(self, state: GenerationState) -> Path:
        path = self.state_path(state.generation)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        return path

    def load(self, generation: int) -> GenerationState:
        path = self.state_path(generation)
        if not path.exists():
            raise FileNotFoundError(f"no generation state at {path}")
        return GenerationState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def latest_generation(self) -> int | None:
        """Highest generation number with a persisted state, or None (for run resume)."""
        if not self.runs_root.exists():
            return None
        gens = []
        for entry in self.runs_root.iterdir():
            if entry.is_dir() and entry.name.startswith("gen_") and (entry / "state.json").exists():
                try:
                    gens.append(int(entry.name[len("gen_"):]))
                except ValueError:
                    continue
        return max(gens) if gens else None
