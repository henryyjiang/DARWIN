"""Genetic-algorithm operations (ARCHITECTURE.md §3.2).

Pure transforms over `Model`s with an injectable RNG so selection/reproduction are
reproducible and unit-testable:

- `rank_models` / `select_survivors` — rank all 10 by fitness, keep top `num_survivors`
  (the **GA cull**, §3.2 step 2). `finetune_failed` recipes carry floor fitness (§6.3) so they
  sort last and are culled first; `None` fitness (unscored) is treated as the floor too.
- `pair_offspring` — the crossover analogue (§3.2 step 3): for each of the 5 offspring, draw a
  starting survivor **S with replacement** and a mutator **M with replacement, M != S**. With
  fewer than 2 survivors there is no distinct mutator, so M falls back to the Claude backend
  (`mutator=None`), regardless of the configured backend (§3.2 degenerate case).

The optional diversity safeguard (§3.4) reserves one survivor slot for the highest-fitness
model whose genome is most different from the already-selected elites; **off by default** until
baseline behavior is understood, and it needs a code-distance callable to be supplied.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from darwin.controller.population import Model

# A model's sort key: its fitness, with None (unscored) treated as the floor so it ranks last.
FLOOR = float("-inf")


def fitness_key(model: Model) -> float:
    return FLOOR if model.fitness is None else model.fitness


def rank_models(models: list[Model]) -> list[Model]:
    """All models, best fitness first. Stable on ties (preserves input order)."""
    return sorted(models, key=fitness_key, reverse=True)


def select_survivors(
    models: list[Model],
    num_survivors: int,
    *,
    diversity_pick: bool = False,
    diversity_fn: Callable[[Model, Model], float] | None = None,
) -> list[Model]:
    """Keep the top `num_survivors` by fitness (the GA cull, §3.2 step 2).

    With `diversity_pick` (and a `diversity_fn(a, b) -> distance`), the last slot is given to
    the candidate maximizing the minimum code-distance to the already-selected elites — the
    §3.4 diversity safeguard. Defaults to pure greedy top-N.
    """
    ranked = rank_models(models)
    if num_survivors >= len(ranked):
        return ranked
    if not diversity_pick or diversity_fn is None or num_survivors < 2:
        return ranked[:num_survivors]

    elites = ranked[: num_survivors - 1]
    remaining = ranked[num_survivors - 1 :]
    # most-different-from-elites; ties broken by fitness (remaining is already fitness-ordered)
    diverse = max(
        remaining,
        key=lambda c: min(diversity_fn(c, e) for e in elites),
    )
    return elites + [diverse]


@dataclass
class OffspringPlan:
    """One offspring's parentage for this generation (§3.2 step 3).

    The offspring's starting genome is `clone(parent_survivor)`; `mutator` is the survivor M
    that reshapes it (or None => no distinct mutator exists, use the Claude backend, §3.2).
    """

    parent_survivor: str
    mutator: str | None


def pair_offspring(
    survivors: list[Model],
    num_offspring: int,
    rng: random.Random,
) -> list[OffspringPlan]:
    """Assign (S, M) for each offspring per §3.2's pairing rules."""
    if not survivors:
        raise ValueError("need at least one survivor to spawn offspring")
    names = [s.name for s in survivors]
    plans: list[OffspringPlan] = []
    for _ in range(num_offspring):
        parent = rng.choice(names)  # S, with replacement
        if len(names) < 2:
            mutator = None  # degenerate: no distinct M -> claude fallback (§3.2)
        else:
            mutator = rng.choice([n for n in names if n != parent])  # M != S, with replacement
        plans.append(OffspringPlan(parent_survivor=parent, mutator=mutator))
    return plans
