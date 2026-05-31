"""Benchmark runner contract (ARCHITECTURE.md §6.2).

Benchmarking is **controller-driven and post-finetune**: it operates on a *finetuned
offspring* (base + adapter), never on a bare genome, and there is no agent-callable
scored-benchmark tool (§9.3). The handoff (§6.2): the finetune job outputs the LoRA adapter;
the controller applies it on top of the base **at load time in the zero-egress eval
container**, mounts only the current generation's private held-out slice read-only, runs the
suite, and gets back a per-benchmark score vector that feeds fitness (§6.3).

This module defines the backend-agnostic contract (`BenchmarkJob` / `BenchmarkResult` /
`BenchmarkBackend`) plus a CPU-runnable `SubprocessBenchmarkBackend` for the testable core and
a scaffolded `EvalContainerBenchmarkBackend` for the live `darwin-eval` path.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class BenchmarkError(RuntimeError):
    """The eval run failed operationally (non-zero exit, missing/invalid scores file).

    Distinct from a low *score*: a low score is a valid result that feeds fitness; a
    `BenchmarkError` means the eval didn't produce a usable vector at all (an infra/harness
    problem the controller handles — e.g. re-run — not a recipe penalty)."""


@dataclass
class BenchmarkJob:
    """Inputs to evaluate one finetuned offspring on the current held-out slice (§6.2)."""

    offspring_id: str
    model: str
    generation: int
    base_model: str  # baked into the darwin-eval image; named for provenance/serving
    adapter_path: Path  # the small artifact mounted in per offspring (base+adapter at load)
    suite: list[str]  # benchmark ids, e.g. ["humaneval+", "gsm8k", ...]
    slice_id: int  # the private held-out slice for this generation (§6.4)
    eval_data_dir: Path | None = None  # the slice, mounted read-only (subprocess backend)


@dataclass
class BenchmarkResult:
    """A finetuned offspring's per-benchmark score vector on the current slice (§6.2/§6.3)."""

    offspring_id: str
    model: str
    generation: int
    slice_id: int
    scores: dict[str, float] = field(default_factory=dict)
    log: str = ""


class BenchmarkBackend(Protocol):
    """Runs the eval suite for one finetuned offspring, returning a score vector."""

    def run(self, job: BenchmarkJob) -> BenchmarkResult: ...


@dataclass
class SubprocessBenchmarkBackend:
    """Runs an eval entrypoint as a subprocess; reads back a JSON scores file.

    Contract with the entrypoint (mirrors §8.5 darwin-eval semantics): the controller passes
    `base_model`, `adapter_path`, the suite, the slice id, the mounted slice dir, and an output
    path via environment variables; the entrypoint loads `base + adapter`, runs the suite, and
    writes `{benchmark: score}` JSON to `DARWIN_SCORES_OUT`. Exit 0 + a parseable file => the
    score vector; anything else raises `BenchmarkError`.
    """

    command: list[str]
    timeout_s: float = 3600.0
    env: dict[str, str] | None = None

    def run(self, job: BenchmarkJob) -> BenchmarkResult:
        scores_out = job.adapter_path.parent / f"scores_{job.offspring_id}.json"
        env = dict(os.environ if self.env is None else self.env)
        env.update(
            DARWIN_BASE_MODEL=job.base_model,
            DARWIN_ADAPTER_PATH=str(job.adapter_path),
            DARWIN_SUITE=",".join(job.suite),
            DARWIN_EVAL_SLICE=str(job.slice_id),
            DARWIN_EVAL_DATA_DIR=str(job.eval_data_dir or ""),
            DARWIN_SCORES_OUT=str(scores_out),
        )
        scores_out.parent.mkdir(parents=True, exist_ok=True)
        cwd = job.eval_data_dir if job.eval_data_dir else job.adapter_path.parent

        try:
            proc = subprocess.run(
                self.command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )
            log = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            raise BenchmarkError(
                f"benchmark timed out after {self.timeout_s}s\n{exc.output or ''}"
            ) from exc

        if proc.returncode != 0:
            raise BenchmarkError(f"benchmark exited {proc.returncode}\n{log[-4000:]}")
        if not scores_out.exists():
            raise BenchmarkError(f"benchmark produced no scores file at {scores_out}")
        try:
            raw = json.loads(scores_out.read_text(encoding="utf-8"))
            scores = {str(k): float(v) for k, v in raw.items()}
        except (ValueError, TypeError) as exc:
            raise BenchmarkError(f"invalid scores file: {exc}") from exc

        return BenchmarkResult(
            offspring_id=job.offspring_id,
            model=job.model,
            generation=job.generation,
            slice_id=job.slice_id,
            scores=scores,
            log=log,
        )


@dataclass
class EvalContainerBenchmarkBackend:
    """Live zero-egress `darwin-eval` container backend (§6.2 / §8) — scaffold.

    The live path: spin up the `darwin-eval` image (benchmark harnesses only, base-model
    weights baked in, **zero egress**, §8.5); bind-mount the small adapter and *only* the
    current generation's private slice read-only (the slice arrives via local bind mount, not
    the network — train/eval separation, §6.4); load `base + adapter`; run the suite; return
    the score vector. Deferred until the eval image + harness adapters land (the salvaged
    `darwin/bench/swe_bench/` harness feeds the coding slice).
    """

    image: str = "darwin-eval"

    def run(self, job: BenchmarkJob) -> BenchmarkResult:
        raise NotImplementedError(
            "EvalContainerBenchmarkBackend is scaffolded; the live zero-egress eval-container "
            "path lands with the darwin-eval image. Use SubprocessBenchmarkBackend for the "
            "local/proxy path."
        )
