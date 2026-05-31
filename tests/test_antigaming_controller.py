"""Anti-gaming scan wired into the controller / fitness (ARCHITECTURE.md §6.4 + §6.3)."""

import random
import subprocess
import sys
from pathlib import Path

import pytest

from darwin.antigaming import AntiGamingFlag, AntiGamingReport
from darwin.antigaming.genome_review import RuleBasedGenomeReviewer
from darwin.config import AntiGamingConfig, DarwinConfig
from darwin.controller import (
    Controller,
    GenerationStateStore,
    LocalAntiGamingScanner,
    LocalGenerationOps,
    Model,
    Population,
    genome_mutation_diff,
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


# ------------------------------------------------------------------ genome diff helper


def test_genome_mutation_diff_captures_added_lines(tmp_path):
    g = tmp_path / "genome"
    g.mkdir()
    (g / "r.py").write_text("OK = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(g), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(g), "config", "user.email", "a@b.c"], check=True)
    subprocess.run(["git", "-C", str(g), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(g), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(g), "commit", "-qm", "base"], check=True)
    (g / "r.py").write_text("OK = True\nif dataset == 'humaneval': cheat()\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(g), "commit", "-aqm", "mutate"], check=True)

    diff = genome_mutation_diff(g)
    assert "humaneval" in diff
    assert "+if dataset == 'humaneval'" in diff


def test_genome_mutation_diff_no_repo_is_empty(tmp_path):
    g = tmp_path / "genome"
    g.mkdir()
    (g / "r.py").write_text("x = 1\n", encoding="utf-8")
    assert genome_mutation_diff(g) == ""


# ------------------------------------------------------------------ scanner unit


def test_local_scanner_flags_genome_hack(tmp_path):
    g = tmp_path / "genome"
    g.mkdir()
    (g / "r.py").write_text("OK = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(g), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(g), "config", "user.email", "a@b.c"], check=True)
    subprocess.run(["git", "-C", str(g), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(g), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(g), "commit", "-qm", "base"], check=True)
    (g / "r.py").write_text("OK = True\nANSWERS = {'q': 1}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(g), "commit", "-aqm", "mutate"], check=True)

    scanner = LocalAntiGamingScanner(
        config=AntiGamingConfig(), reviewer=RuleBasedGenomeReviewer()
    )
    from darwin.controller.state import OffspringState

    report = scanner.scan(
        offspring=Model(name="o0", genome_dir=g),
        state=OffspringState(name="o0", parent_survivor="s0", mutator="s1", backend="claude", iteration=0),
        slice_id=0,
        generation=1,
    )
    assert "genome_hack" in report.kinds


def test_local_scanner_contamination_via_eval_provider(tmp_path):
    g = tmp_path / "genome"
    g.mkdir()
    leaked = "the gold answer to hidden eval problem forty two"
    (g / "data.py").write_text(f"mix = ['{leaked}']\n", encoding="utf-8")
    scanner = LocalAntiGamingScanner(
        config=AntiGamingConfig(ngram_n=4, genome_reviewer="none"),
        eval_items_provider=lambda gen, sl: [leaked],
    )
    from darwin.controller.state import OffspringState

    report = scanner.scan(
        offspring=Model(name="o0", genome_dir=g),
        state=OffspringState(name="o0", parent_survivor="s0", mutator="s1", backend="claude", iteration=0),
        slice_id=0,
        generation=1,
    )
    assert "contamination" in report.kinds


# ------------------------------------------------------------------ controller integration


def seed_survivor_genome(genome: Path) -> None:
    genome.mkdir(parents=True, exist_ok=True)
    (genome / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (genome / "smoke_test.py").write_text(
        "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )


class HackBackend:
    """Fake mutation backend that writes benchmark-gaming code, checkpoints, writes memory."""

    def run(self, ctx, deadline):
        # valid, importable (OK stays True) but contains a benchmark-name special-case the
        # genome-diff reviewer flags as gaming (§6.4)
        (ctx.genome_dir / "recipe.py").write_text(
            "OK = True\ndef hook(dataset):\n    if dataset == 'humaneval':\n        return 1\n",
            encoding="utf-8",
        )
        assert ctx.checkpoint("hacked recipe") is True
        ctx.write_memory(thesis="t", changes="c", smoke_results="green", outcome="o", cost_usd=0.0)


def _make_controller(tmp_path, antigaming):
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

    from darwin.finetune import SubprocessFinetuneBackend
    from darwin.bench import SubprocessBenchmarkBackend

    ops = LocalGenerationOps(
        config=cfg, store=store, ledger=ledger, workspace=ws,
        mutation_backend_factory=lambda name, ctx: HackBackend(),
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

    return Controller(
        config=cfg, store=store, ledger=ledger,
        state_store=GenerationStateStore(tmp_path / "runs"),
        ops=ops, synthesizer=FakeSynth(), antigaming=antigaming, rng=random.Random(0),
    ), population


def test_hacked_offspring_penalized_when_scan_enabled(tmp_path):
    scanner = LocalAntiGamingScanner(config=AntiGamingConfig(), reviewer=RuleBasedGenomeReviewer())
    ctrl, pop = _make_controller(tmp_path, antigaming=scanner)
    nxt = ctrl.run_generation(0, pop)
    o0 = nxt.get("o0")
    # normalized 0.7/0.5 = 1.4, minus lambda_penalty(0.5)*1 flag, minus tiny GPU cost
    assert o0.fitness == pytest.approx(1.4 - 0.5, abs=0.02)


def test_no_scanner_means_no_flags(tmp_path):
    ctrl, pop = _make_controller(tmp_path, antigaming=None)
    nxt = ctrl.run_generation(0, pop)
    o0 = nxt.get("o0")
    assert o0.fitness == pytest.approx(1.4, abs=0.02)  # no penalty


def test_flags_persist_in_state(tmp_path):
    scanner = LocalAntiGamingScanner(config=AntiGamingConfig(), reviewer=RuleBasedGenomeReviewer())
    ctrl, pop = _make_controller(tmp_path, antigaming=scanner)
    ctrl.run_generation(0, pop)
    state = ctrl.state_store.load(0)
    off = state.offspring_by_name()["o0"]
    assert off.antigaming_done is True
    assert off.antigaming_flags >= 1
