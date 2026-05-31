# Cost ledger

Running spend, per-generation budget, and cost guidance (§5.4 / §7.4). **Cost and train time are
first-class constraints** surfaced here so the global-memory pass can steer the population toward
gains that justify their spend and away from runs that don't.

| Generation | API cost (USD) | GPU cost (USD) | Total (USD) |
|---|---|---|---|
| _none yet_ | | | |

Running total: $0.00

## Cost guidance for mutators
- **Parameter-scaling runs are the expensive ones.** Depth expansion / MoE upcycling raise the
  effective param count, so the GPU instance is sized to the run at launch
  (`darwin/finetune/sizing.py`): bigger models → more/larger GPUs → higher $/hr, and a larger
  token budget (up to 250B) → longer wall-clock. Both multiply.
- **Optimize fitness per dollar and per GPU-hour**, not raw fitness. Prefer the cheapest method
  that yields the gain: QLoRA on the expanded model over full finetune; bounded token budgets;
  MoE's bounded *active* FLOPs over pure dense growth; partial expansion over full when it
  suffices.
- A budget cap (`gen_budget_usd`, §5.4) stops *new* launches once a generation's spend is hit
  (in-flight jobs finish); skipped offspring are deferred. Don't design a generation that can only
  pay for one giant run.
