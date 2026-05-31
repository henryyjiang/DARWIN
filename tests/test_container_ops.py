"""ContainerGenerationOps end-to-end through the controller (ARCHITECTURE.md §2.3 / §8).

No Docker/GPU/Claude: a single dispatching **fake `ContainerRunner`** stands in for all three
§8.5 images. By image it: drives the real in-container mutation entrypoint (`run_window`) with a
fake backend over the bind-mounted genome (`darwin-agent`); writes the adapter through the
adapter mount (`darwin-finetune`); writes the score vector through the writable scores mount
(`darwin-eval`). This proves the whole container loop — spec building, host↔container path
mapping, the memory seed/ingest, and the result handoff — composes through the controller exactly
like the local path does, with the security-boundary specs actually built.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from darwin.config import DarwinConfig
from darwin.controller import (
    Controller,
    ContainerGenerationOps,
    GenerationStateStore,
    Model,
    Population,
)
from darwin.cost import CostLedger
from darwin.finetune import ContainerFinetuneBackend
from darwin.bench import EvalContainerBenchmarkBackend
from darwin.memory import MemoryStore
from darwin.sandbox import ContainerResult, ContainerSpec
from darwin.mutation_agent.deadline import DeadlineManager
from darwin.mutation_agent.entrypoint import MutationRunConfig, run_window


def _host_path_for(spec: ContainerSpec, container_path: str) -> str:
    best = None
    for m in spec.mounts:
        if container_path == m.container_path or container_path.startswith(m.container_path + "/"):
            rel = container_path[len(m.container_path):].lstrip("/")
            host = str(Path(m.host_path) / rel) if rel else m.host_path
            if best is None or len(m.container_path) > best[0]:
                best = (len(m.container_path), host)
    assert best is not None, f"{container_path} not under any mount of {spec.image}"
    return best[1]


class GreenBackend:
    """Fake in-container mutation backend: one green edit, checkpoint, memory write."""

    def run(self, ctx, deadline):
        (ctx.genome_dir / "recipe.py").write_text("OK = True\n# mutated\n", encoding="utf-8")
        assert ctx.checkpoint("mutated recipe") is True
        ctx.write_memory(thesis="t", changes="c", smoke_results="green", outcome="o", cost_usd=0.0)


@dataclass
class DispatchRunner:
    """One fake ContainerRunner standing in for all three §8.5 images."""

    specs: list[ContainerSpec] = field(default_factory=list)

    def run(self, spec: ContainerSpec, *, dry_run: bool = False) -> ContainerResult:
        self.specs.append(spec)
        if spec.image == "darwin-agent":
            self._run_agent(spec)
        elif spec.image == "darwin-finetune":
            host = _host_path_for(spec, spec.env["DARWIN_ADAPTER_OUT"])
            Path(host).parent.mkdir(parents=True, exist_ok=True)
            Path(host).write_text("adapter", encoding="utf-8")
        elif spec.image == "darwin-eval":
            host = _host_path_for(spec, spec.env["DARWIN_SCORES_OUT"])
            Path(host).parent.mkdir(parents=True, exist_ok=True)
            Path(host).write_text(json.dumps({"code": 0.7}), encoding="utf-8")
        return ContainerResult(0, "", "", ["docker", "run"])

    def _run_agent(self, spec: ContainerSpec) -> None:
        # Translate the container paths back to host bind-mount paths (what the kernel does for a
        # real mount), then run the entrypoint window exactly as it would inside the image.
        cfg = MutationRunConfig.from_env(spec.env)
        cfg.genome_dir = _host_path_for(spec, spec.env["DARWIN_GENOME_DIR"])
        cfg.store_root = _host_path_for(spec, spec.env["DARWIN_STORE_ROOT"])
        cfg.result_out = _host_path_for(spec, spec.env["DARWIN_RESULT_OUT"])
        deadline = DeadlineManager(window_s=100, soft_lead_s=20, kill_grace_s=10,
                                   clock=lambda: 0.0, start=0.0)
        run_window(cfg, GreenBackend(), deadline=deadline)


def seed_survivor_genome(genome: Path) -> None:
    genome.mkdir(parents=True, exist_ok=True)
    (genome / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (genome / "smoke_test.py").write_text(
        "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )


class FakeSynth:
    def synthesize(self, digest, current):
        from darwin.memory import GlobalMemory

        return GlobalMemory(objectives="x")


def test_container_ops_compose_one_generation(tmp_path):
    ws = tmp_path / "ws"
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

    slice_dir = tmp_path / "slices" / "0"
    slice_dir.mkdir(parents=True)

    runner = DispatchRunner()
    ops = ContainerGenerationOps(
        config=cfg,
        store=store,
        ledger=ledger,
        workspace=ws,
        finetune_backend=ContainerFinetuneBackend(runner=runner),
        benchmark_backend=EvalContainerBenchmarkBackend(runner=runner),
        smoke_command=["python", "smoke_test.py"],
        container_runner=runner,
        eval_slice_dir=lambda sid: slice_dir,
    )

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

    # all three images were launched with the right security policy
    images = {s.image: s for s in runner.specs}
    assert set(images) == {"darwin-agent", "darwin-finetune", "darwin-eval"}
    assert images["darwin-eval"].network == "none"        # zero egress (§6.2/§8.3)
    assert images["darwin-finetune"].network == "whitelist"
    assert images["darwin-agent"].network == "whitelist"

    # the mutation ran in-container and its edit survived to the green final genome on the host
    assert (ws / "o0" / "genome" / ".git").exists()
    assert "# mutated" in (ws / "o0" / "genome" / "recipe.py").read_text(encoding="utf-8")
    # finetune produced the adapter through the rw adapter mount
    assert (ws / "o0" / "adapter.bin").read_text(encoding="utf-8") == "adapter"

    o0 = nxt.get("o0")
    assert o0.finetune_failed is False
    assert o0.scores == {"code": 0.7}
    assert o0.fitness == pytest.approx(1.4, abs=0.01)

    # the memory the agent wrote inside the container was ingested into the host store (§7.2)
    mem = store.read_iteration("o0", 0)
    assert mem.changes == "c"
    assert mem.final_fitness == pytest.approx(o0.fitness)  # controller patched it post-benchmark
    assert any(e.kind == "finetune" for e in ledger.entries())
    assert store.get_global().objectives == "x"  # global pass still ran


def test_container_ops_seeds_prior_and_global_memory_into_scratch(tmp_path):
    """The agent's scratch store is seeded with prior lineage memory + global memory (ORIENT)."""
    ws = tmp_path / "ws"
    seed_survivor_genome(ws / "s0" / "genome")
    store = MemoryStore(tmp_path / "store")
    # a prior iteration for o0 + some global memory on the host store
    from darwin.memory import GlobalMemory, IterationMemory

    store.write_iteration(IterationMemory(
        model="o0", iteration=0, generation=0, parent_survivor="s0", mutator="s0",
        backend="claude", base_fitness=0.1, cost_usd=0.0, thesis="prior", changes="prior-change",
        smoke_results="green", outcome="o",
    ))
    store.write_global(GlobalMemory(objectives="seek depth expansion"))

    captured = {}

    @dataclass
    class CaptureRunner:
        def run(self, spec, *, dry_run=False):
            store_host = _host_path_for(spec, spec.env["DARWIN_STORE_ROOT"])
            seeded = MemoryStore(store_host)
            captured["prior"] = seeded.iteration_numbers("o0")
            captured["global"] = seeded.get_global().objectives
            # write a fresh iter_1 result so the window looks successful
            Path(_host_path_for(spec, spec.env["DARWIN_RESULT_OUT"])).write_text(
                json.dumps({"final_commit": "abc", "mutation_failed": False,
                            "produced_green": True, "memory_written": True,
                            "model": "o0", "iteration": 1}), encoding="utf-8")
            return ContainerResult(0, "", "", [])

    ops = ContainerGenerationOps(
        config=DarwinConfig(), store=store, ledger=CostLedger(tmp_path / "c.jsonl"),
        workspace=ws,
        finetune_backend=ContainerFinetuneBackend(runner=CaptureRunner()),
        benchmark_backend=EvalContainerBenchmarkBackend(runner=CaptureRunner()),
        smoke_command=["python", "smoke_test.py"], container_runner=CaptureRunner(),
    )
    parent = Model(name="s0", genome_dir=ws / "s0" / "genome", fitness=0.5)
    offspring = ops.spawn(
        offspring=_offspring_state("o0"), parent=parent, generation=1
    )
    ops.mutate(offspring=offspring, parent=parent, mutator=None,
               state=_offspring_state("o0", iteration=1), generation=1)

    assert captured["prior"] == [0]  # prior iteration seeded
    assert captured["global"] == "seek depth expansion"  # global memory seeded


def _offspring_state(name, iteration=0):
    from darwin.controller.state import OffspringState

    return OffspringState(name=name, parent_survivor="s0", mutator=None,
                          backend="claude", iteration=iteration)
