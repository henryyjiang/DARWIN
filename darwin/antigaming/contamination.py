"""Contamination scan (ARCHITECTURE.md §6.4).

Before finetuning, scan the genome's data scripts/datasets for n-gram overlap with the held-out
eval items and flag matches — this catches a model that tries to pull benchmark data into its
training mix (train/eval separation is *also* enforced by the container, §6.4, but a recipe can
still try to *fetch* eval-shaped data into its mix). Each contaminated eval item raises one
`AntiGamingFlag`, so the penalty scales with how much leaked.

Pure index math over word-level n-grams — no I/O. The controller is the only place that holds
the held-out eval items (host-only, §6.2); it passes their text in. The genome's data text is
read from the offspring's data scripts/dataset manifests by the caller (the scanner seam).
"""

from __future__ import annotations

import re
from typing import Iterable

from darwin.antigaming.report import AntiGamingFlag

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """The set of word-level n-grams in `text` (lowercased, punctuation-split)."""
    toks = _tokens(text)
    if n <= 0:
        raise ValueError("n must be positive")
    if len(toks) < n:
        # short text: treat the whole token sequence as a single gram so tiny eval items
        # (e.g. a one-line answer) can still match verbatim.
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def contamination_scan(
    data_texts: Iterable[str],
    eval_items: Iterable[str],
    *,
    n: int = 8,
    min_overlap: int = 1,
    max_flags: int = 20,
) -> list[AntiGamingFlag]:
    """Flag eval items whose word n-grams overlap the genome's data text (§6.4).

    `n` is the n-gram width (8 words ≈ a verbatim phrase, low false-positive rate);
    `min_overlap` is how many shared n-grams trip a flag; `max_flags` caps the penalty so a
    massively-contaminated mix doesn't produce an unbounded fitness hit (it's already strongly
    penalized well before the cap).
    """
    data_grams: set[tuple[str, ...]] = set()
    for text in data_texts:
        data_grams |= word_ngrams(text, n)
    if not data_grams:
        return []

    flags: list[AntiGamingFlag] = []
    for idx, item in enumerate(eval_items):
        shared = word_ngrams(item, n) & data_grams
        if len(shared) >= min_overlap:
            sample = " ".join(next(iter(shared)))
            flags.append(
                AntiGamingFlag(
                    kind="contamination",
                    detail=f"eval item {idx} shares {len(shared)} {n}-gram(s) with data mix: "
                    f"…{sample}…",
                )
            )
            if len(flags) >= max_flags:
                break
    return flags
