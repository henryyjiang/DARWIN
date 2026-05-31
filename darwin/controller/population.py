"""Population model (ARCHITECTURE.md §3.1).

A model = `(genome, weights)`: the **genome is the code+config** the mutator edits, the
**weights are the LoRA adapter** produced by finetuning that genome. Each model is a directory
under `models/<name>/` holding the genome, the adapter, the finetune config, and the model's
memory folder; this dataclass is the in-memory handle the GA + controller pass around, and it
round-trips to JSON for the resumable generation state (§2.3).

`fitness` is the scalar the GA ranks on (§6.3); `scores` is the per-benchmark vector it was
reduced from (kept so survivors form the normalization baseline and can be cheaply re-scored
when the eval slice rotates, §6.2). Survivors carry cached `fitness`/`scores`; freshly mutated
offspring get theirs filled in after finetune+benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Model:
    """One population member (§3.1)."""

    name: str
    genome_dir: Path
    adapter_path: Path | None = None
    fitness: float | None = None  # last scored fitness (§6.3); None => not yet scored
    scores: dict[str, float] = field(default_factory=dict)  # last per-benchmark vector
    generation_born: int = 0
    parent_survivor: str | None = None  # S this was cloned from (None for gen-0 seeds)
    mutator: str | None = None  # M that edited it (None => claude fallback / seed)
    backend: str = "claude"  # backend that drove its mutation
    is_survivor: bool = False  # carried over with a cached score vs. freshly mutated
    mutation_failed: bool = False  # fell back to the unchanged clone of S (§4.3)
    finetune_failed: bool = False  # recipe couldn't train at scale (§5.3)
    scored_slice: int | None = None  # the held-out slice `scores` were computed on (§6.2)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "genome_dir": str(self.genome_dir),
            "adapter_path": None if self.adapter_path is None else str(self.adapter_path),
            "fitness": self.fitness,
            "scores": dict(self.scores),
            "generation_born": self.generation_born,
            "parent_survivor": self.parent_survivor,
            "mutator": self.mutator,
            "backend": self.backend,
            "is_survivor": self.is_survivor,
            "mutation_failed": self.mutation_failed,
            "finetune_failed": self.finetune_failed,
            "scored_slice": self.scored_slice,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        adapter = d.get("adapter_path")
        return cls(
            name=d["name"],
            genome_dir=Path(d["genome_dir"]),
            adapter_path=None if adapter is None else Path(adapter),
            fitness=d.get("fitness"),
            scores=dict(d.get("scores", {})),
            generation_born=d.get("generation_born", 0),
            parent_survivor=d.get("parent_survivor"),
            mutator=d.get("mutator"),
            backend=d.get("backend", "claude"),
            is_survivor=d.get("is_survivor", False),
            mutation_failed=d.get("mutation_failed", False),
            finetune_failed=d.get("finetune_failed", False),
            scored_slice=d.get("scored_slice"),
        )


@dataclass
class Population:
    """The set of models alive at a point in the loop (10 by default, §3.1)."""

    models: list[Model] = field(default_factory=list)

    def by_name(self) -> dict[str, Model]:
        return {m.name: m for m in self.models}

    def get(self, name: str) -> Model:
        return self.by_name()[name]

    def names(self) -> list[str]:
        return [m.name for m in self.models]

    def survivors(self) -> list[Model]:
        return [m for m in self.models if m.is_survivor]

    def offspring(self) -> list[Model]:
        return [m for m in self.models if not m.is_survivor]

    def to_dict(self) -> dict:
        return {"models": [m.to_dict() for m in self.models]}

    @classmethod
    def from_dict(cls, d: dict) -> "Population":
        return cls(models=[Model.from_dict(m) for m in d.get("models", [])])
