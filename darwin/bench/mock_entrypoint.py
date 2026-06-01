"""Mock eval entrypoint — the test-profile substitute for the GPU eval (TEST_RUN_PLAN §3.2).

The default `darwin-eval` entrypoint (`entrypoint.py`) loads `base + adapter` on a GPU and runs
the real benchmark harnesses. For the budget-free full-run test we swap in *this* module: it runs
in a **real `--network none` container** through the same `EvalContainerBenchmarkBackend` /
`SubprocessBenchmarkBackend` contract and the same `DARWIN_*` env, but instead of running models it
reads the genome fingerprint the mock finetune wrote into the adapter (§3.1) and turns it into a
**genome-dependent score vector with per-slice drift**.

The score model (per benchmark):

    score = clamp01( base(genome_sha) + IMPROVE_STEP * markers + drift(slice_id, benchmark) )

- `base(genome_sha)` is deterministic in a narrow band around 0.5 — so changing the genome jiggles
  the score (real, if noisy, selection pressure) but does not swamp the improvement signal;
- each accumulated §3.3 marker adds `IMPROVE_STEP`, so a lineage that keeps making green mutations
  trends **up** across generations — a watchable demo and real selection (the up-trending lineages
  survive);
- `drift(slice_id, benchmark)` is a small per-slice delta, so when the held-out slice rotates
  (§6.2/§6.4) survivors must be re-benchmarked and their scores shift — exercising that path too.

The eval container sees only the adapter (never the genome), so all genome signal travels inside
the adapter JSON — mirroring the real train/eval separation. `score_vector` is a pure function,
unit-tested with no Docker; `main` reads the env, the adapter JSON, and writes the scores JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from darwin.bench.entrypoint import BenchRunConfig, parse_suite, write_scores

# Defaults chosen so the marker signal is visible above base noise but selection is still real:
# base in [0.5 - BASE_SPREAD, 0.5 + BASE_SPREAD]; each marker adds IMPROVE_STEP; slice drift small.
BASE_CENTER = 0.5
BASE_SPREAD = 0.05
IMPROVE_STEP = 0.03
DRIFT_SPREAD = 0.015


def _unit_hash(*parts: str) -> float:
    """A deterministic float in [0, 1) from the given string parts (stable across runs/machines)."""
    h = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    # take 8 bytes -> integer -> [0, 1)
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def base_score(genome_sha: str) -> float:
    """Deterministic base score for a genome in [BASE_CENTER ± BASE_SPREAD]."""
    return BASE_CENTER + (2.0 * _unit_hash("base", genome_sha) - 1.0) * BASE_SPREAD


def slice_drift(slice_id: int, benchmark: str) -> float:
    """Small per-slice, per-benchmark delta in [±DRIFT_SPREAD] (drives eval-rotation re-bench)."""
    return (2.0 * _unit_hash("drift", str(slice_id), benchmark) - 1.0) * DRIFT_SPREAD


def score_vector(
    suite: list[str],
    genome_sha: str,
    markers: int,
    slice_id: int,
    *,
    improve_step: float = IMPROVE_STEP,
) -> dict[str, float]:
    """The genome-dependent score vector for one offspring on one slice (pure)."""
    base = base_score(genome_sha)
    return {
        bench: clamp01(base + improve_step * markers + slice_drift(slice_id, bench))
        for bench in suite
    }


def read_adapter(path: Path | str) -> dict:
    """Read the mock adapter JSON (genome fingerprint) the mock finetune wrote (§3.1)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def main(env: dict[str, str] | None = None) -> int:
    """Read the fingerprint adapter and write the genome-dependent score vector (the mock eval)."""
    env = dict(os.environ if env is None else env)
    cfg = BenchRunConfig.from_env(env)
    suite = cfg.suite or parse_suite(env.get("DARWIN_SUITE", ""))

    adapter = read_adapter(cfg.adapter_path)
    genome_sha = str(adapter.get("genome_sha", ""))
    markers = int(adapter.get("markers", 0) or 0)

    scores = score_vector(suite, genome_sha, markers, cfg.slice_id)
    write_scores(cfg.scores_out, scores)
    print(
        f"[mock-eval] genome_sha={genome_sha[:12]} markers={markers} slice={cfg.slice_id} "
        f"-> {cfg.scores_out}: {scores}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
