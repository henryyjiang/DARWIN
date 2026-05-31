"""Per-model memory file schema (ARCHITECTURE.md §7.2).

A memory file is markdown with a YAML frontmatter block followed by four required body
sections. The schema is enforced here so the MCP `memory.write_iteration` tool cannot write
a malformed or empty memory (§4.8: "you can't write junk that doesn't fit the schema").

The agent writes everything *except* the controller-owned post-benchmark fields
(`final_fitness`, `mutation_failed`, `finetune_failed`), which the controller patches in
host-side after benchmarking (§7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import yaml

# The four required body sections, in canonical order. Keys are attribute names; values are
# the markdown headings they serialize to.
BODY_SECTIONS: dict[str, str] = {
    "thesis": "Thesis",
    "changes": "Changes implemented",
    "smoke_results": "Smoke-test / validation results",
    "outcome": "Outcome & reflection",
}

# Frontmatter field order for stable serialization.
_FRONTMATTER_ORDER = [
    "model",
    "iteration",
    "generation",
    "parent_survivor",
    "mutator",
    "backend",
    "base_fitness",
    "final_fitness",
    "mutation_failed",
    "finetune_failed",
    "cost_usd",
    "datasets_used",
    "papers_cited",
]

_VALID_BACKENDS = {"local", "claude"}


class MemoryValidationError(ValueError):
    """Raised when an IterationMemory fails schema validation."""


@dataclass
class IterationMemory:
    """One iteration's per-model memory record (§7.2).

    Agent-written fields are required at write time. Controller-owned fields
    (`final_fitness`, `mutation_failed`, `finetune_failed`) default to unset/False and are
    patched in by the controller after benchmarking.
    """

    # --- agent-written frontmatter ---
    model: str
    iteration: int
    generation: int
    parent_survivor: str  # the genome this offspring was cloned from
    mutator: str  # who edited it (self if local backend)
    backend: str  # "local" | "claude"
    base_fitness: float  # parent's fitness at clone time
    cost_usd: float

    # --- agent-written body sections ---
    thesis: str = ""
    changes: str = ""
    smoke_results: str = ""
    outcome: str = ""

    # --- agent-written provenance (§8.3 / §8.4) ---
    datasets_used: list[str] = field(default_factory=list)
    papers_cited: list[str] = field(default_factory=list)

    # --- controller-owned, patched post-benchmark (§7.2) ---
    final_fitness: float | None = None
    mutation_failed: bool = False
    finetune_failed: bool = False

    # ------------------------------------------------------------------ validation
    def validate(self, require_body: bool = True) -> None:
        """Validate the record. Raises MemoryValidationError on the first problem.

        require_body=True (the default, used for agent writes) additionally requires the
        four body sections to be non-empty.
        """
        errors: list[str] = []

        def _nonempty_str(name: str, value: Any) -> None:
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} must be a non-empty string")

        _nonempty_str("model", self.model)
        _nonempty_str("parent_survivor", self.parent_survivor)
        _nonempty_str("mutator", self.mutator)

        for name in ("iteration", "generation"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{name} must be a non-negative int")

        if self.backend not in _VALID_BACKENDS:
            errors.append(f"backend must be one of {sorted(_VALID_BACKENDS)}")

        if not _is_number(self.base_fitness):
            errors.append("base_fitness must be a number")

        if not _is_number(self.cost_usd) or self.cost_usd < 0:
            errors.append("cost_usd must be a non-negative number")

        if self.final_fitness is not None and not _is_number(self.final_fitness):
            errors.append("final_fitness must be a number or None")

        for name in ("mutation_failed", "finetune_failed"):
            if not isinstance(getattr(self, name), bool):
                errors.append(f"{name} must be a bool")

        for name in ("datasets_used", "papers_cited"):
            value = getattr(self, name)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                errors.append(f"{name} must be a list of strings")

        if require_body:
            for attr, heading in BODY_SECTIONS.items():
                if not getattr(self, attr).strip():
                    errors.append(f"body section '{heading}' must be non-empty")

        if errors:
            raise MemoryValidationError(
                "invalid IterationMemory:\n  - " + "\n  - ".join(errors)
            )

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly mapping of all fields (for MCP tool responses)."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Render to the canonical markdown-with-frontmatter file format (§7.2)."""
        front = {key: getattr(self, key) for key in _FRONTMATTER_ORDER}
        frontmatter = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()

        parts = [f"---\n{frontmatter}\n---", ""]
        for attr, heading in BODY_SECTIONS.items():
            parts.append(f"## {heading}\n{getattr(self, attr).strip()}\n")
        return "\n".join(parts).rstrip() + "\n"

    @classmethod
    def from_markdown(cls, text: str) -> "IterationMemory":
        """Parse a memory file back into an IterationMemory (inverse of to_markdown)."""
        frontmatter, body = _split_frontmatter(text)
        front = yaml.safe_load(frontmatter) or {}
        if not isinstance(front, dict):
            raise MemoryValidationError("frontmatter must be a YAML mapping")

        sections = _parse_sections(body)
        known = {attr: front.get(attr) for attr in _FRONTMATTER_ORDER if attr in front}
        unknown = set(front) - set(_FRONTMATTER_ORDER)
        if unknown:
            raise MemoryValidationError(f"unknown frontmatter keys: {sorted(unknown)}")

        kwargs: dict[str, Any] = dict(known)
        for attr, heading in BODY_SECTIONS.items():
            kwargs[attr] = sections.get(heading, "")

        try:
            return cls(**kwargs)
        except TypeError as exc:  # missing required frontmatter field
            raise MemoryValidationError(str(exc)) from exc


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a `---`-delimited frontmatter block from the body."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        raise MemoryValidationError("missing '---' frontmatter delimiter")
    # Drop the opening delimiter line, then split on the next line that is exactly '---'.
    after_open = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    lines = after_open.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "---":
            frontmatter = "\n".join(lines[:i])
            body = "\n".join(lines[i + 1 :])
            return frontmatter, body
    raise MemoryValidationError("unterminated frontmatter block (no closing '---')")


def _parse_sections(body: str) -> dict[str, str]:
    """Parse `## Heading` sections from the markdown body into {heading: content}."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in body.split("\n"):
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections
