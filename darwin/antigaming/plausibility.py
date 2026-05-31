"""Plausibility / generalization-gap check (ARCHITECTURE.md §6.4).

If an offspring's held-out benchmark score *wildly exceeds* a quick out-of-distribution (OOD)
probe — a small set of held-out-style items drawn from a different distribution than the scored
slice — that's the signature of overfitting to the benchmark rather than a real capability gain,
so it is flagged as suspected gaming. A genuinely-improved model should generalize: its OOD
score tracks its held-out score, leaving a small gap.

Pure gap math, per benchmark. The controller supplies both vectors: the held-out slice scores
(already produced post-finetune, §6.2) and the OOD probe scores (a separate, cheap eval run —
the *live* probe is deferred infra, like the rest of the eval-container work, §6.2).
"""

from __future__ import annotations

from darwin.antigaming.report import AntiGamingFlag


def generalization_gap_flags(
    held_out: dict[str, float],
    ood: dict[str, float],
    *,
    max_gap: float = 0.25,
) -> list[AntiGamingFlag]:
    """Flag benchmarks where `held_out - ood` exceeds `max_gap` (§6.4).

    Only benchmarks present in *both* vectors are compared (no probe score => no claim). A
    larger gap raises a higher-severity flag (one extra `lambda_penalty` unit per `max_gap`
    overshoot) so a blatant generalization failure is penalized harder than a marginal one.
    """
    if max_gap <= 0:
        raise ValueError("max_gap must be positive")
    flags: list[AntiGamingFlag] = []
    for bench, ho in held_out.items():
        if bench not in ood:
            continue
        gap = ho - ood[bench]
        if gap > max_gap:
            severity = int(gap // max_gap)  # >=1 once gap > max_gap
            flags.append(
                AntiGamingFlag(
                    kind="generalization_gap",
                    detail=f"{bench}: held-out {ho:.3f} exceeds OOD probe {ood[bench]:.3f} "
                    f"by {gap:.3f} (> {max_gap}) — suspected overfit/gaming",
                    severity=severity,
                )
            )
    return flags
