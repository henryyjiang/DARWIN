"""Memory store: filesystem I/O for per-model and global memory (ARCHITECTURE.md §7).

Canonical layout (§7 "Canonical paths"):
    <root>/models/<model>/memory/iter_<n>.md   # per-model, agent-written
    <root>/memory/global/                       # global, global-memory-pass-written

This is the host-side library. The MCP server (future) wraps the per-model read methods
(`recent`, `search`) and the schema-validated `write_iteration` for agents; the controller
uses `patch_iteration` (post-benchmark) and the global-memory helpers. Population/mutation
agents only ever *read* global memory — they never write it (§7.3 invariant).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

from darwin.memory.schema import (
    IterationMemory,
    MemoryValidationError,
    BODY_SECTIONS,
)

# Global memory files (§7.3). Maps attribute name -> filename.
GLOBAL_SECTIONS: dict[str, str] = {
    "objectives": "objectives.md",
    "whats_working": "whats_working.md",
    "todo": "todo.md",
    "cost_ledger": "cost_ledger.md",
}

# Controller-owned fields that patch_iteration is allowed to set post-benchmark (§7.2).
_CONTROLLER_PATCH_FIELDS = {"final_fitness", "mutation_failed", "finetune_failed"}

_ITER_RE = re.compile(r"^iter_(\d+)\.md$")


@dataclass
class GlobalMemory:
    """The four-section global memory store (§7.3)."""

    objectives: str = ""
    whats_working: str = ""
    todo: str = ""
    cost_ledger: str = ""


class MemoryStore:
    """Filesystem-backed store rooted at the repo root."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    # ------------------------------------------------------------------ paths
    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def global_dir(self) -> Path:
        return self.root / "memory" / "global"

    def model_memory_dir(self, model: str) -> Path:
        return self.models_dir / model / "memory"

    def list_models(self) -> list[str]:
        """Sorted names of all models present under models/ (each a model directory)."""
        if not self.models_dir.exists():
            return []
        return sorted(p.name for p in self.models_dir.iterdir() if p.is_dir())

    def iter_path(self, model: str, iteration: int) -> Path:
        return self.model_memory_dir(model) / f"iter_{iteration}.md"

    # ------------------------------------------------------------------ per-model writes
    def write_iteration(self, mem: IterationMemory) -> Path:
        """Validate (with body required) and write a per-model iteration file.

        This is the schema gate behind the MCP `memory.write_iteration` tool: a record that
        does not satisfy the §7.2 schema is rejected before anything touches disk.
        """
        mem.validate(require_body=True)
        path = self.iter_path(mem.model, mem.iteration)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mem.to_markdown(), encoding="utf-8")
        return path

    def patch_iteration(self, model: str, iteration: int, **updates) -> IterationMemory:
        """Controller-side patch of the post-benchmark fields (§7.2).

        Only `final_fitness`, `mutation_failed`, `finetune_failed` may be patched — the
        agent owns everything else. The file is rewritten in place.
        """
        bad = set(updates) - _CONTROLLER_PATCH_FIELDS
        if bad:
            raise ValueError(
                f"patch_iteration may only set {sorted(_CONTROLLER_PATCH_FIELDS)}, "
                f"got {sorted(bad)}"
            )
        mem = self.read_iteration(model, iteration)
        for key, value in updates.items():
            setattr(mem, key, value)
        mem.validate(require_body=True)
        self.iter_path(model, iteration).write_text(mem.to_markdown(), encoding="utf-8")
        return mem

    # ------------------------------------------------------------------ per-model reads
    def read_iteration(self, model: str, iteration: int) -> IterationMemory:
        path = self.iter_path(model, iteration)
        if not path.exists():
            raise FileNotFoundError(f"no memory file at {path}")
        return IterationMemory.from_markdown(path.read_text(encoding="utf-8"))

    def iteration_numbers(self, model: str) -> list[int]:
        """Sorted (ascending) iteration numbers present for a model."""
        mem_dir = self.model_memory_dir(model)
        if not mem_dir.exists():
            return []
        nums = []
        for entry in mem_dir.iterdir():
            match = _ITER_RE.match(entry.name)
            if match:
                nums.append(int(match.group(1)))
        return sorted(nums)

    def all_iterations(self, model: str) -> list[IterationMemory]:
        """All of a model's iterations, oldest first."""
        return [self.read_iteration(model, n) for n in self.iteration_numbers(model)]

    def recent(self, model: str, k: int = 5) -> list[IterationMemory]:
        """The k most recent iterations, newest first (MCP `memory.recent(k)`)."""
        if k < 0:
            raise ValueError("k must be non-negative")
        nums = self.iteration_numbers(model)[::-1][:k]
        return [self.read_iteration(model, n) for n in nums]

    def search(self, model: str, query: str) -> list[IterationMemory]:
        """Keyword search over a model's own memory history (MCP `memory.search`).

        Simple, dependency-free relevance: case-insensitive count of query-term occurrences
        across the body sections and provenance. Returns matches (score > 0) ranked by score
        then recency (newest first).
        """
        terms = [t.lower() for t in query.split() if t.strip()]
        if not terms:
            return []

        scored: list[tuple[int, int, IterationMemory]] = []
        for mem in self.all_iterations(model):
            haystack = "\n".join(
                [getattr(mem, attr) for attr in BODY_SECTIONS]
                + mem.datasets_used
                + mem.papers_cited
            ).lower()
            score = sum(haystack.count(term) for term in terms)
            if score > 0:
                scored.append((score, mem.iteration, mem))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [mem for _, _, mem in scored]

    # ------------------------------------------------------------------ global memory
    def get_global(self) -> GlobalMemory:
        """Read the global memory store (MCP `memory.get_global()`). Missing files -> ''."""
        kwargs = {}
        for attr, filename in GLOBAL_SECTIONS.items():
            path = self.global_dir / filename
            kwargs[attr] = path.read_text(encoding="utf-8") if path.exists() else ""
        return GlobalMemory(**kwargs)

    def write_global(self, memory: GlobalMemory) -> None:
        """Write the global memory store. **Global-memory-pass / controller only** (§7.3) —
        never called from a population/mutation agent path."""
        self.global_dir.mkdir(parents=True, exist_ok=True)
        for f in fields(GlobalMemory):
            filename = GLOBAL_SECTIONS[f.name]
            content = getattr(memory, f.name)
            (self.global_dir / filename).write_text(content, encoding="utf-8")
