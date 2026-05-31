"""LocalGenerationOps wiring the real cores end-to-end (ARCHITECTURE.md §2.3).

No Docker/GPU/Claude: a fake mutation backend drives a real Git mutation window on a cloned
trivial genome, then the real subprocess finetune + benchmark backends run, and the controller
aggregates fitness, patches memory, and forms the next population — proving the Phase 1-3 cores
compose through `LocalGenerationOps`.
"""

import random
import sys
from pathlib import Path

import pytest

from darwin.config import DarwinConfig
from darwin.controller import (
    Controller,
    GenerationStateStore,
    LocalGenerationOps,
    Model,
    Population,
)
from darwin.cost import CostLedger
from darwin.memory import MemoryStore
from darwin.mutation_agent import DeadlineManager


FT_SCRIPT = (
    "import os, pathlib; p = pathlib.Path(os.environ['DARWIN_ADAPTER_OUT']); "
    "p.parent.mkdir(parents=True, exist_ok=True); p.write_text('adapter')"
)
EVAL_SCRIPT = (
    "import os, json, pathlib; "
    "pathlib.Path(os.environ['DARWIN_SCORES_OUT']).write_text(json.dumps({'code': 0.7}))"
)


class GreenBackend:
    """Fake mutation backend: makes one green edit, checkpoints it, writes its memory."""

    def run(self, ctx, deadline):
        (ctx.genome_dir / "recipe.py").write_text("OK = True\n# mutated\n", encoding="utf-8")
        assert ctx.checkpoint("mutated recipe") is True
        ctx.write_memory(
            thesis="t", changes="c", smoke_results="green", outcome="o", cost_usd=0.0
        )


def seed_survivor_genome(genome: Path) -> None:
    genome.mkdir(parents=True, exist_ok=True)
    (genome / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (genome / "smoke_test.py").write_text(
        "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )


def test_local_ops_compose_one_generation(tmp_path):
    ws = tmp_path / "ws"
    # survivor s0 with a green genome; offspring slot o0 to be filled
    seed_survivor_genome(ws / "s0" / "genome")
    population = Population(
        [
            Model(name="s0", genome_dir=ws / "s0" / "genome", fitness=0.5,
                  scores={"code": 0.5}, is_survivor=True),
            Model(name="o0", genome_dir=ws / "o0" / "genome", fitness=0.1),
        ]
    )

    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]

    store = MemoryStore(tmp_path / "store")
    ledger = CostLedger(tmp_path / "cost.jsonl")

    from darwin.finetune import SubprocessFinetuneBackend
    from darwin.bench import SubprocessBenchmarkBackend

    ops = LocalGenerationOps(
        config=cfg,
        store=store,
        ledger=ledger,
        workspace=ws,
        mutation_backend_factory=lambda name, ctx: GreenBackend(),
        finetune_backend=SubprocessFinetuneBackend(command=[sys.executable, "-c", FT_SCRIPT]),
        benchmark_backend=SubprocessBenchmarkBackend(command=[sys.executable, "-c", EVAL_SCRIPT]),
        smoke_command=[sys.executable, "smoke_test.py"],
        deadline_factory=lambda: DeadlineManager(
            window_s=100, soft_lead_s=20, kill_grace_s=10, clock=lambda: 0.0, start=0.0
        ),
    )

    class FakeSynth:
        def synthesize(self, digest, current):
            from darwin.memory import GlobalMemory

            return GlobalMemory(objectives="x")

    ctrl = Controller(
        config=cfg,
        store=store,
        ledger=ledger,
        state_store=GenerationStateStore(tmp_path / "runs"),
        ops=ops,
        synthesizer=FakeSynth(),
        rng=random.Random(0),
    )

    nxt = ctrl.run_generation(0, population)

    # the offspring genome was cloned into its slot and is a real git repo
    assert (ws / "o0" / "genome" / ".git").exists()
    assert (ws / "o0" / "genome" / "recipe.py").exists()
    # the mutation's edit survived to the green final genome
    assert "# mutated" in (ws / "o0" / "genome" / "recipe.py").read_text(encoding="utf-8")
    # finetune produced the adapter
    assert (ws / "o0" / "adapter.bin").read_text(encoding="utf-8") == "adapter"

    o0 = nxt.get("o0")
    assert o0.finetune_failed is False
    assert o0.adapter_path == ws / "o0" / "adapter.bin"
    assert o0.scores == {"code": 0.7}
    # fitness ~ normalized 0.7/0.5 = 1.4 minus a tiny GPU cost penalty
    assert o0.fitness == pytest.approx(1.4, abs=0.01)

    # per-model memory written (iteration 0) and patched with final fitness (§7.2)
    mem = store.read_iteration("o0", 0)
    assert mem.final_fitness == pytest.approx(o0.fitness)
    assert mem.mutator == "claude"  # single survivor -> claude fallback

    # finetune cost was recorded to the ledger
    assert any(e.kind == "finetune" for e in ledger.entries())

    # global memory was written by the pass
    assert store.get_global().objectives == "x"
