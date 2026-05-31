"""Gather a generation's inputs for the global-memory pass (ARCHITECTURE.md §7.4 steps 1-2).

Pure, dependency-light: scans the per-model memory store for the iteration files written
*this generation* and renders them — plus a fitness table derived from the controller-patched
`final_fitness` fields — into prompt-ready text. No LLM involved here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin.memory import IterationMemory, MemoryStore


@dataclass
class GenerationDigest:
    """The inputs the global-memory pass reasons over for one generation."""

    generation: int
    memories: list[IterationMemory] = field(default_factory=list)

    def fitness_table(self) -> str:
        """Markdown fitness/cost table across this generation's models (§7.4 step 2)."""
        header = (
            "| model | mutator | backend | base_fitness | final_fitness | cost_usd "
            "| mutation_failed | finetune_failed |\n"
            "|---|---|---|---|---|---|---|---|"
        )
        if not self.memories:
            return header + "\n| _no models this generation_ |"
        rows = []
        for m in sorted(self.memories, key=lambda x: x.model):
            final = "—" if m.final_fitness is None else f"{m.final_fitness:.4g}"
            rows.append(
                f"| {m.model} | {m.mutator} | {m.backend} | {m.base_fitness:.4g} "
                f"| {final} | {m.cost_usd:.4g} | {m.mutation_failed} "
                f"| {m.finetune_failed} |"
            )
        return header + "\n" + "\n".join(rows)

    def render_memories(self) -> str:
        """Concatenate this generation's per-model memory files for the prompt."""
        if not self.memories:
            return "_No per-model memory files were written this generation._"
        blocks = []
        for m in sorted(self.memories, key=lambda x: x.model):
            blocks.append(
                f"### {m.model} — iter {m.iteration} (gen {m.generation})\n\n"
                + m.to_markdown()
            )
        return "\n\n---\n\n".join(blocks)


def collect_generation(store: MemoryStore, generation: int) -> GenerationDigest:
    """Collect every per-model iteration memory written for `generation`.

    A model contributes its iteration file(s) whose frontmatter `generation` matches — i.e.
    the offspring mutated this generation (and any survivor re-touched this generation).
    """
    memories: list[IterationMemory] = []
    for model in store.list_models():
        for n in store.iteration_numbers(model):
            mem = store.read_iteration(model, n)
            if mem.generation == generation:
                memories.append(mem)
    return GenerationDigest(generation=generation, memories=memories)
