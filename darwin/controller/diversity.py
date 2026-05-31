"""Genome code-distance for the diversity safeguard (ARCHITECTURE.md §3.4).

Pure greedy top-N selection can collapse the population into near-duplicates. The §3.4
safeguard reserves one survivor slot for the highest-fitness model whose genome is most
*different* from the already-selected elites, where "different" is a code-distance.

The spec names a **code-embedding** distance; a learned embedding model is deferred (Appendix A
open question). This provides a dependency-free default: a Jaccard distance over the genome
source's token n-grams (0.0 == identical code, 1.0 == no shared n-grams). It is a reasonable
stand-in — it cleanly separates a copy of S from a substantially-rewritten genome — and the
`select_survivors` seam takes any `(Model, Model) -> float` callable, so an embedding-based
distance drops in later without touching the GA.
"""

from __future__ import annotations

import re
from pathlib import Path

from darwin.controller.population import Model

# Source files that make up the genome's "code" (the thing the mutator edits, §3.1).
_CODE_SUFFIXES = (".py", ".json", ".yaml", ".yml", ".toml", ".cfg", ".txt", ".md")
_TOKEN = re.compile(r"\w+")
_MAX_BYTES_PER_FILE = 1_000_000  # guard against a pathological genome file


def read_genome_source(genome_dir: Path, *, suffixes: tuple[str, ...] = _CODE_SUFFIXES) -> str:
    """Concatenate the genome's source files (sorted for determinism), skipping `.git`."""
    if not genome_dir.exists():
        return ""
    parts: list[str] = []
    for path in sorted(genome_dir.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in suffixes:
            continue
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore")[:_MAX_BYTES_PER_FILE])
        except OSError:
            continue
    return "\n".join(parts)


def _token_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _TOKEN.findall(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard_distance(a: str, b: str, *, n: int = 3) -> float:
    """1 - |A∩B|/|A∪B| over token n-grams. 0.0 == identical, 1.0 == disjoint."""
    ga, gb = _token_ngrams(a, n), _token_ngrams(b, n)
    union = ga | gb
    if not union:
        return 0.0  # both empty -> treat as identical (no diversity signal)
    return 1.0 - len(ga & gb) / len(union)


def genome_code_distance(a: Model, b: Model, *, n: int = 3) -> float:
    """Code-distance between two models' genomes (§3.4 default; embedding-based later)."""
    return jaccard_distance(
        read_genome_source(a.genome_dir), read_genome_source(b.genome_dir), n=n
    )
