"""Run the global-memory pass end-to-end (ARCHITECTURE.md §7.4).

gather (digest) → synthesize (Claude or strict-local) → write global memory. This is the
controller-invoked entry point and the *only* sanctioned writer of global memory besides
direct controller edits (§7.3 invariant).
"""

from __future__ import annotations

from darwin.global_memory.digest import collect_generation
from darwin.global_memory.synthesizer import Synthesizer, ClaudeSynthesizer
from darwin.memory import GlobalMemory, MemoryStore


def run_global_memory_pass(
    store: MemoryStore,
    generation: int,
    synthesizer: Synthesizer | None = None,
) -> GlobalMemory:
    """Rewrite global memory from `generation`'s per-model memories + the current state.

    `synthesizer` defaults to `ClaudeSynthesizer` (the §7.4 default); inject another impl for
    `strict-local` runs or for tests. Returns the newly written GlobalMemory.
    """
    if synthesizer is None:
        synthesizer = ClaudeSynthesizer()

    digest = collect_generation(store, generation)
    current = store.get_global()
    new_global = synthesizer.synthesize(digest, current)
    store.write_global(new_global)
    return new_global
