# To-do / open problems for next-gen mutators

Open experiments for mutators to pick up. Rewritten each generation by the global-memory pass
(§7.4); seeded with the priority directions from `objectives.md`.

## Parameter scaling (priority)
- **Depth expansion** — implement block/layer expansion of the base (duplicate or interleave
  transformer blocks, zero-init the residual adds where appropriate) and continue-train. Start
  from arXiv:2401.02415 (block expansion) and arXiv:2312.15166 (depth up-scaling); `paper.fetch`
  + cite (§8.4). Sweep how many blocks / where to insert / freeze-vs-train schedule.
- **MoE upcycling** — convert the dense base into a sparse MoE (replicate MLP → N experts, add a
  top-k router) and continue-train; start from arXiv:2212.05055 (sparse upcycling). Sweep expert
  count, top-k, router init, load-balancing loss, and which layers to upcycle.
- **Invent improved scaling methods** — don't stop at the seeded recipes: try hybrid depth+MoE,
  growth curricula (expand mid-training), smarter expert/router initialization from the dense
  weights, partial expansion of only the most-utilized layers, etc. Novel methods that beat the
  baselines are the highest-value result. Record the method clearly so the global pass can spread
  what works.

## Performance (driven by what's working across the population)
- **Test paper implementations** — pull a promising method (objective/loss change, attention or
  optimizer trick, data-curation scheme) via `paper.*`, implement it in the genome, smoke-test,
  cite it. Prefer methods correlated with gains in `whats_working.md`.
- **Optimize the data mix** — curate/weight existing license-clear datasets via `data.*` (record
  `id@revision` + license, §8.3); try domain up-weighting, dedup, quality filtering, curriculum
  ordering. No scrapers.

## Always weigh cost & train time
Every experiment competes on **fitness per dollar and per GPU-hour** (§5.4/§6.3). A scaling run on
a large token budget (up to 250B) is expensive — justify it against expected gain, prefer the
cheapest method that yields the improvement, and bound the token budget unless a thesis argues for
more.
