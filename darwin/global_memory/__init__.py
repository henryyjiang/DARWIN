"""The global-memory pass (ARCHITECTURE.md §7.4).

After benchmarking each generation, the controller invokes a focused reasoning pass that
reads all of this generation's per-model memory files + the fitness table + the cost ledger,
and *rewrites* global memory (`objectives`, `whats_working`, `todo`, `cost_ledger`).

Invariant (§7.3): global memory is written **only** by this dedicated meta-pass — never by
the population/mutation agents. By default the pass runs on Claude (§7.4); a local model may
drive it under the non-default `strict-local` flag, which is why the writer is abstracted
behind the `Synthesizer` interface rather than hardwired to the Claude SDK.

Layering (mirrors the "two interchangeable AI backends" principle, §1.4):
- `digest.py` — pure gathering of a generation's inputs into a `GenerationDigest`.
- `synthesizer.py` — the `Synthesizer` interface + `ClaudeSynthesizer` (Anthropic API).
- `pass_run.py` — `run_global_memory_pass`: gather → synthesize → write.
"""

from darwin.global_memory.digest import GenerationDigest, collect_generation
from darwin.global_memory.synthesizer import (
    Synthesizer,
    CappedSynthesizer,
    ClaudeSynthesizer,
    MockSynthesizer,
    parse_global_memory,
)
from darwin.global_memory.pass_run import run_global_memory_pass

__all__ = [
    "GenerationDigest",
    "collect_generation",
    "Synthesizer",
    "ClaudeSynthesizer",
    "MockSynthesizer",
    "CappedSynthesizer",
    "parse_global_memory",
    "run_global_memory_pass",
]
