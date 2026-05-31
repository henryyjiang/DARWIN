"""Reference benchmark entrypoint (ARCHITECTURE.md §6.2).

The default eval the `darwin-eval` image runs (§8.5): read the job from the `DARWIN_*` env the
benchmark backend sets (§6.2 `SubprocessBenchmarkBackend`), load `base + adapter`, run each
benchmark in the suite on the mounted private slice, and write `{benchmark: score}` JSON to
`DARWIN_SCORES_OUT`. Runs with **zero egress** (§6.2/§8.3): the base weights are baked into the
image and the slice arrives via a read-only bind mount.

Design mirrors the finetune entrypoint: config parsing + suite dispatch/aggregation are pure
(`BenchRunConfig.from_env`, `parse_suite`, `run_suite`, `write_scores`) and unit-tested with a
fake per-benchmark runner; `main()` lazy-imports the model + harnesses (CUDA image only). The
salvaged `darwin/bench/swe_bench/` harness feeds the coding slice.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# A per-benchmark runner: (benchmark_id, cfg) -> score in [0, 1].
BenchRunner = Callable[[str, "BenchRunConfig"], float]


@dataclass
class BenchRunConfig:
    """Resolved benchmark job (§6.2)."""

    base_model: str = ""
    adapter_path: str = ""
    suite: list[str] = field(default_factory=list)
    slice_id: int = 0
    eval_data_dir: str = ""
    scores_out: str = "scores.json"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "BenchRunConfig":
        env = dict(os.environ if env is None else env)
        return cls(
            base_model=env.get("DARWIN_BASE_MODEL", ""),
            adapter_path=env.get("DARWIN_ADAPTER_PATH", ""),
            suite=parse_suite(env.get("DARWIN_SUITE", "")),
            slice_id=int(env.get("DARWIN_EVAL_SLICE", "0") or 0),
            eval_data_dir=env.get("DARWIN_EVAL_DATA_DIR", ""),
            scores_out=env.get("DARWIN_SCORES_OUT", "scores.json"),
        )


def parse_suite(raw: str) -> list[str]:
    """Parse the comma-separated `DARWIN_SUITE` into benchmark ids (pure)."""
    return [b.strip() for b in raw.split(",") if b.strip()]


def run_suite(cfg: BenchRunConfig, runner: BenchRunner) -> dict[str, float]:
    """Run each benchmark in the suite via `runner`, returning the score vector (pure)."""
    return {bench: float(runner(bench, cfg)) for bench in cfg.suite}


def write_scores(path: str | Path, scores: dict[str, float]) -> Path:
    """Write the score vector as JSON to `path` (the §6.2 handoff file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scores), encoding="utf-8")
    return p


def main(env: dict[str, str] | None = None) -> int:  # pragma: no cover - needs the GPU stack
    """Run the reference eval suite. Lazy-imports the model + harnesses (eval image only)."""
    cfg = BenchRunConfig.from_env(env)
    print(f"[darwin-eval] base={cfg.base_model} adapter={cfg.adapter_path} "
          f"suite={cfg.suite} slice={cfg.slice_id}")
    runner = _build_default_runner(cfg)
    scores = run_suite(cfg, runner)
    write_scores(cfg.scores_out, scores)
    print(f"[darwin-eval] scores -> {cfg.scores_out}: {scores}")
    return 0


def _build_default_runner(cfg: "BenchRunConfig") -> BenchRunner:  # pragma: no cover
    """Load base+adapter once, return a runner dispatching benchmark ids to harnesses."""
    import torch  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, device_map="auto")
    if cfg.adapter_path:
        model = PeftModel.from_pretrained(model, cfg.adapter_path)  # base + adapter (§6.2)
    model.eval()

    def runner(bench: str, _cfg: "BenchRunConfig") -> float:
        b = bench.lower()
        if b.startswith("humaneval") or b in ("mbpp", "livecodebench"):
            from darwin.bench.swe_bench import harness  # salvaged coding harness

            return harness.evaluate(model, tokenizer, bench, _cfg.eval_data_dir)
        # math / reasoning suites via lm-eval-harness
        from lm_eval import simple_evaluate  # type: ignore

        out = simple_evaluate(model=model, tasks=[bench], limit=None)
        return float(next(iter(out["results"].values())).get("acc", 0.0))

    return runner


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
