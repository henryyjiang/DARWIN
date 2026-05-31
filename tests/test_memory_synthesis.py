"""Memory-file synthesis fallback (ARCHITECTURE.md §4.3 / §7.2)."""

import subprocess
from pathlib import Path

from darwin.mutation_agent.memory_synthesis import (
    SynthesisContext,
    build_synthesis_prompt,
    git_log,
    parse_synthesis,
    read_transcript_excerpt,
)


def _ctx(**kw):
    base = dict(model="o0", iteration=0, generation=1, parent_survivor="s0", mutator="s1",
                backend="local", base_fitness=0.5)
    base.update(kw)
    return SynthesisContext(**base)


# ------------------------------------------------------------------ pure helpers


def test_git_log_reads_commits(tmp_path):
    g = tmp_path / "genome"
    g.mkdir()
    (g / "r.py").write_text("OK = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(g), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(g), "config", "user.email", "a@b.c"], check=True)
    subprocess.run(["git", "-C", str(g), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(g), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(g), "commit", "-qm", "darwin-green: raised lora rank"], check=True)
    log = git_log(g)
    assert "raised lora rank" in log


def test_git_log_empty_without_repo(tmp_path):
    assert git_log(tmp_path / "nope") == ""


def test_read_transcript_excerpt_tails(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("x" * 20000, encoding="utf-8")
    assert len(read_transcript_excerpt(p, max_chars=8000)) == 8000
    assert read_transcript_excerpt(None) == ""
    assert read_transcript_excerpt(tmp_path / "missing.jsonl") == ""


def test_build_prompt_includes_log_and_transcript():
    prompt = build_synthesis_prompt(_ctx(git_log="abc raised rank", transcript_excerpt="ran smoke"))
    assert "raised rank" in prompt and "ran smoke" in prompt and "o0" in prompt


def test_parse_synthesis_fills_iteration_memory():
    mem = parse_synthesis(
        {"thesis": "t", "changes": "c", "smoke_results": "green", "outcome": "o",
         "papers_cited": ["arXiv:2401.02415"], "datasets_used": ["bigcode/the-stack@v1"]},
        _ctx(),
    )
    mem.validate()  # passes the §7.2 schema
    assert mem.model == "o0" and mem.thesis == "t"
    assert mem.papers_cited == ["arXiv:2401.02415"]


def test_parse_synthesis_placeholders_when_sparse():
    mem = parse_synthesis({"thesis": "", "changes": "", "smoke_results": "", "outcome": ""}, _ctx())
    mem.validate()  # placeholders keep the body non-empty so validation passes
    assert "synthesized" in mem.outcome


# ------------------------------------------------------------------ controller wiring


def test_controller_synthesizes_missing_memory(tmp_path):
    import random

    from darwin.config import DarwinConfig
    from darwin.controller import (
        Controller,
        FinetuneOutcomeView,
        GenerationStateStore,
        Model,
        MutateOutcome,
        Population,
    )
    from darwin.cost import CostLedger
    from darwin.memory import GlobalMemory, MemoryStore

    class SilentOps:
        """Mutates without writing a memory file -> the controller must synthesize it."""

        def spawn(self, *, offspring, parent, generation):
            return Model(name=offspring.name, genome_dir=tmp_path / offspring.name)

        def mutate(self, **k):
            return MutateOutcome(final_commit="abc", mutation_failed=False)

        def finetune(self, **k):
            return FinetuneOutcomeView("ok", Path("a"), 0.0)

        def benchmark(self, **k):
            return {"code": 0.6}

    class FakeMemSynth:
        def __init__(self):
            self.calls = 0

        def synthesize(self, ctx):
            self.calls += 1
            return parse_synthesis(
                {"thesis": "synth", "changes": "c", "smoke_results": "g", "outcome": "o"}, ctx
            )

    class FakeSynth:
        def synthesize(self, digest, current):
            return GlobalMemory(objectives="x")

    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    mem_synth = FakeMemSynth()
    store = MemoryStore(tmp_path / "store")
    pop = Population(
        [Model(name="s0", genome_dir=tmp_path / "s0", fitness=0.5, scores={"code": 0.5}, is_survivor=True)]
        + [Model(name="o0", genome_dir=tmp_path / "o0", fitness=0.1)]
    )
    ctrl = Controller(
        config=cfg, store=store, ledger=CostLedger(tmp_path / "c.jsonl"),
        state_store=GenerationStateStore(tmp_path / "runs"), ops=SilentOps(),
        synthesizer=FakeSynth(), memory_synthesizer=mem_synth, rng=random.Random(0),
    )
    ctrl.run_generation(0, pop)

    # the agent wrote nothing, so the controller synthesized the memory and patched it
    assert mem_synth.calls == 1
    mem = store.read_iteration("o0", 0)
    assert mem.thesis == "synth"
    assert mem.final_fitness is not None  # patched post-benchmark (§7.2)


def test_controller_skips_synthesis_when_no_synthesizer(tmp_path):
    # default (no memory_synthesizer) keeps the prior behavior: no file, no error
    import random

    from darwin.config import DarwinConfig
    from darwin.controller import (
        Controller, FinetuneOutcomeView, GenerationStateStore, Model, MutateOutcome, Population,
    )
    from darwin.cost import CostLedger
    from darwin.memory import GlobalMemory, MemoryStore

    class SilentOps:
        def spawn(self, *, offspring, parent, generation):
            return Model(name=offspring.name, genome_dir=tmp_path / offspring.name)

        def mutate(self, **k):
            return MutateOutcome(final_commit="abc", mutation_failed=False)

        def finetune(self, **k):
            return FinetuneOutcomeView("ok", Path("a"), 0.0)

        def benchmark(self, **k):
            return {"code": 0.6}

    class FakeSynth:
        def synthesize(self, digest, current):
            return GlobalMemory(objectives="x")

    cfg = DarwinConfig()
    cfg.ga.num_survivors = 1
    cfg.benchmark.suite = ["code"]
    store = MemoryStore(tmp_path / "store")
    pop = Population(
        [Model(name="s0", genome_dir=tmp_path / "s0", fitness=0.5, scores={"code": 0.5}, is_survivor=True)]
        + [Model(name="o0", genome_dir=tmp_path / "o0", fitness=0.1)]
    )
    ctrl = Controller(
        config=cfg, store=store, ledger=CostLedger(tmp_path / "c.jsonl"),
        state_store=GenerationStateStore(tmp_path / "runs"), ops=SilentOps(),
        synthesizer=FakeSynth(), rng=random.Random(0),
    )
    ctrl.run_generation(0, pop)
    assert store.iteration_numbers("o0") == []  # no memory file, no crash
