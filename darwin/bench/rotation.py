"""Held-out eval-slice rotation (ARCHITECTURE.md §6.2 / §6.4).

The full eval set lives **only on the controller host**; each generation the controller picks
one private held-out slice and mounts *only* that slice read-only into the zero-egress
`darwin-eval` container. Rotation is a **seeded permutation keyed by generation number** so it
is reproducible and auditable: the same `(seed, num_slices, generation)` always yields the
same slice, and over `num_slices` generations every slice is used once (in a shuffled order)
before the cycle repeats.

This is pure index math — it does not touch the eval data itself (that handoff is the
controller's, §6.2). When `eval_rotation` is off (config), the controller simply fixes the
slice instead of calling this.
"""

from __future__ import annotations

import random


def held_out_slice(generation: int, num_slices: int, *, seed: int = 0) -> int:
    """Return the private held-out slice index for `generation` (§6.4).

    Deterministic in `(seed, num_slices, generation)`. The permutation is reshuffled each full
    cycle (keyed by the cycle number) so rotation order isn't identical every cycle while
    staying fully reproducible.
    """
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    if generation < 0:
        raise ValueError("generation must be non-negative")
    cycle, within = divmod(generation, num_slices)
    rng = random.Random(f"{seed}:{cycle}")
    perm = list(range(num_slices))
    rng.shuffle(perm)
    return perm[within]


def rotation_schedule(num_generations: int, num_slices: int, *, seed: int = 0) -> list[int]:
    """The slice chosen for generations 0..num_generations-1 (handy for auditing/tests)."""
    return [held_out_slice(g, num_slices, seed=seed) for g in range(num_generations)]
