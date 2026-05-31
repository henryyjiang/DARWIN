# Objectives

Central research question (ARCHITECTURE.md §1.1): can an AI system independently improve its
own code and architecture — discovering, testing, and keeping ideas that move it toward
state-of-the-art — with **no human writing the improvements**?

## Current high-level direction

**Primary thrust — grow a strong base model's capacity, then specialize it.** The population
should treat **parameter scaling of an already-trained model** as a first-class lever for
capability gains, not just LoRA tuning of a fixed-size base. Two seeded, well-supported families
to start from (mutators: use `paper.fetch` to pull the methods and **cite them**, §8.4):

- **Depth expansion** — add capacity by stacking/duplicating transformer blocks and continuing
  training (e.g. block expansion / depth up-scaling: arXiv:2401.02415, arXiv:2312.15166). Cheap
  relative to from-scratch; preserves the base model's learned features.
- **MoE upcycling** — convert a dense checkpoint into a sparse Mixture-of-Experts by replicating
  MLPs into experts + adding a router, then continue training (sparse upcycling:
  arXiv:2212.05055). Raises parameter count and capacity while keeping per-token FLOPs bounded.

**Invent better methods, don't just copy.** The two families above are *starting points*. A core
objective is for mutators to **reason about and devise improved param-scaling methods of their
own** — novel expansion schedules, hybrid depth+MoE, smarter expert initialization, growth
curricula — and test them. Genuine method innovation that moves benchmarks is the highest-value
outcome.

**Performance is the target signal.** Optimize for real capability gains on the coding/math/
reasoning suite (held-out), driven by **what has actually worked across the population** (see
`whats_working.md`, synthesized from the per-model lab notebooks). Concrete avenues, in addition
to scaling: implementing and testing **methods from papers** (`paper.*`), and **optimizing the
data mix** — curating/weighting existing license-clear datasets (`data.*`, never scraping, §8.3).

**Cost and train time are first-class constraints.** Scaling params + training on large token
budgets (up to 250B tokens/run) is expensive and slow; the GPU allocation is sized at runtime to
the run (`darwin/finetune/sizing.py`). Prefer the cheapest scaling that yields the gain: weigh
expected benchmark improvement against GPU-hours and wall-clock, and favor parameter-efficient
growth (QLoRA on the expanded model, bounded token budgets, MoE's bounded active FLOPs) unless a
thesis justifies more. See `cost_ledger.md` for the running budget picture.

_This file is rewritten each generation by the global-memory pass (§7.4); the above is the seed
direction for generation 0's mutators to ORIENT on._
