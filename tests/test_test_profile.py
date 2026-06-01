"""End-to-end test-profile assembly (TEST_RUN_PLAN §3.5/§3.8).

The whole budget-free loop, offline: `build_controller(profile=test, mode=container)` wires
`ContainerGenerationOps` + the container backends pointed at the slim image with the mock
commands, and one dispatching fake `ContainerRunner` stands in for the three §8.5 container roles —
running the *real* mock entrypoints (`darwin.finetune.mock_entrypoint` / `darwin.bench.mock_entrypoint`)
and the *real* in-container mutation window with `MockMutationBackend`. We drive multiple
generations and assert genome drift → score drift → real selection, plus crash-free resume.
"""

import textwrap
from pathlib import Path

import pytest

from darwin.finetune.mock_entrypoint import count_markers
from darwin.global_memory import MockSynthesizer
from darwin.run import bootstrap_or_load_population, build_controller, load_run_spec
from darwin.sandbox import ContainerResult, ContainerSpec


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


class DispatchRunner:
    """One fake ContainerRunner for all three test-profile roles (dispatched by the command)."""

    def __init__(self) -> None:
        self.specs: list[ContainerSpec] = []

    @property
    def agent_runs(self) -> int:
        return sum(1 for s in self.specs if not s.command)  # agent = image CMD (empty command)

    def run(self, spec: ContainerSpec, *, dry_run: bool = False) -> ContainerResult:
        self.specs.append(spec)
        cmd = " ".join(spec.command)
        if "darwin.finetune.mock_entrypoint" in cmd:
            from darwin.finetune.mock_entrypoint import main as ft_main

            ft_main(self._translated_env(spec, with_genome=True))
        elif "darwin.bench.mock_entrypoint" in cmd:
            from darwin.bench.mock_entrypoint import main as ev_main

            ev_main(self._translated_env(spec))
        else:
            self._run_agent(spec)
        return ContainerResult(0, "", "", ["docker", "run"])

    def _translated_env(self, spec: ContainerSpec, *, with_genome: bool = False) -> dict:
        """Map the container-path env vars back to host paths (what a real bind mount does)."""
        env = dict(spec.env)
        env["DARWIN_MOCK_SLEEP_S"] = "0"  # keep the test instant
        path_keys = (
            "DARWIN_ADAPTER_OUT", "DARWIN_ADAPTER_PATH", "DARWIN_SCORES_OUT",
            "DARWIN_EVAL_DATA_DIR", "DARWIN_GENOME_DIR", "DARWIN_STORE_ROOT", "DARWIN_RESULT_OUT",
        )
        for key in path_keys:
            if env.get(key):
                try:
                    env[key] = _host_path_for(spec, env[key])
                except AssertionError:
                    pass
        if with_genome and not env.get("DARWIN_GENOME_DIR"):
            # the finetune entrypoint fingerprints cwd (=/work/genome in the real container)
            env["DARWIN_GENOME_DIR"] = _host_path_for(spec, "/work/genome")
        return env

    def _run_agent(self, spec: ContainerSpec) -> None:
        from darwin.mutation_agent.deadline import DeadlineManager
        from darwin.mutation_agent.entrypoint import MutationRunConfig, run_window
        from darwin.mutation_agent.mock_backend import MockMutationBackend

        cfg = MutationRunConfig.from_env(spec.env)
        cfg.genome_dir = _host_path_for(spec, spec.env["DARWIN_GENOME_DIR"])
        cfg.store_root = _host_path_for(spec, spec.env["DARWIN_STORE_ROOT"])
        cfg.result_out = _host_path_for(spec, spec.env["DARWIN_RESULT_OUT"])
        deadline = DeadlineManager(window_s=100, soft_lead_s=20, kill_grace_s=10,
                                   clock=lambda: 0.0, start=0.0)
        run_window(cfg, MockMutationBackend(), deadline=deadline)


def _write_test_config(tmp_path: Path, generations: int = 3) -> Path:
    base = tmp_path / "base_genome"
    base.mkdir(parents=True, exist_ok=True)
    (base / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (base / "smoke_test.py").write_text(
        "import recipe, sys; sys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )
    cfg = tmp_path / "run.test.yaml"
    cfg.write_text(textwrap.dedent(f"""
        generations: {generations}
        mode: container
        profile: test
        paths:
          base_genome: {base.as_posix()}
        images:
          agent: darwin-agent
          finetune: darwin-agent
          eval: darwin-agent
        commands:
          finetune: ["python", "-m", "darwin.finetune.mock_entrypoint"]
          benchmark: ["python", "-m", "darwin.bench.mock_entrypoint"]
        smoke_command: ["python", "smoke_test.py"]
        seed_scores:
          s0: {{humaneval+: 0.50, gsm8k: 0.42}}
          s1: {{humaneval+: 0.49, gsm8k: 0.41}}
          s2: {{humaneval+: 0.51, gsm8k: 0.43}}
          s3: {{humaneval+: 0.48, gsm8k: 0.40}}
          s4: {{humaneval+: 0.50, gsm8k: 0.44}}
        config:
          ga: {{population_size: 10, num_survivors: 5, diversity_pick: false}}
          mutation: {{backend: mock, mutation_window_h: 0.083, soft_deadline_min: 1}}
          cost: {{gen_budget_usd: 0, gpu_rate_usd_per_h: 1.79}}
          benchmark: {{suite: ["humaneval+", "gsm8k"], eval_rotation: true, num_eval_slices: 3}}
          antigaming: {{enabled: false}}
    """), encoding="utf-8")
    return cfg


def _build(spec, runner):
    controller, store, ledger = build_controller(
        spec,
        mutation_backend_factory=lambda name, ctx: None,  # unused in container mode
        synthesizer=MockSynthesizer(),
        container_runner=runner,
    )
    return controller, store, ledger


def test_test_profile_multi_generation_drift_and_selection(tmp_path):
    spec = load_run_spec(_write_test_config(tmp_path, generations=3))
    assert spec.profile == "test" and spec.mode == "container"
    runner = DispatchRunner()
    controller, store, ledger = _build(spec, runner)

    final = controller.run(spec.generations, bootstrap_or_load_population(spec))

    # all 3 generations completed without intervention
    for g in range(3):
        assert controller.state_store.load(g).completed is True

    # all three container roles actually launched, with the right network policy (§6.2/§8.3)
    by_cmd = {" ".join(s.command): s for s in runner.specs}
    finetune_spec = next(s for s in runner.specs if "finetune.mock_entrypoint" in " ".join(s.command))
    eval_spec = next(s for s in runner.specs if "bench.mock_entrypoint" in " ".join(s.command))
    agent_spec = next(s for s in runner.specs if not s.command)
    assert eval_spec.network == "none"          # zero-egress eval
    assert finetune_spec.network == "whitelist"
    assert agent_spec.network == "whitelist"
    assert {finetune_spec.image, eval_spec.image, agent_spec.image} == {"darwin-agent"}  # one image

    # population names stay stable across generations: 5 survivors + 5 offspring slots (§3.1)
    assert {m.name for m in final.models} == {f"s{i}" for i in range(5)} | {f"o{i}" for i in range(5)}

    # genome drift: later offspring build on prior generations' genomes, so markers accumulate
    # beyond the single marker one mutation adds (a lineage carried forward across generations).
    marker_counts = [count_markers(m.genome_dir) for m in final.models]
    assert max(marker_counts) >= 2

    # real selection: at least one evolved (mutated) genome was promoted into the survivor set —
    # the seed lineage did not simply persist unchanged (a flat, no-selection run).
    survivors = [m for m in final.models if m.is_survivor]
    assert len(survivors) == 5
    assert sum(count_markers(s.genome_dir) for s in survivors) >= 1

    # fitness is not a flat line across the run (genome-dependent scores + drift)
    fitnesses = []
    for g in range(3):
        for off in controller.state_store.load(g).offspring:
            if off.fitness is not None:
                fitnesses.append(round(off.fitness, 6))
    assert len(set(fitnesses)) > 1

    # the global-memory pass ran each generation (mock synth seeded objectives + appended a line)
    gm = store.get_global()
    assert gm.objectives
    assert gm.whats_working.count("best=") == 3

    # the cost ledger accrued mock finetune entries (§5.4)
    assert any(e.kind == "finetune" for e in ledger.entries())


def test_test_profile_resume_does_not_recompute(tmp_path):
    cfg = _write_test_config(tmp_path, generations=1)

    # gen 0
    spec1 = load_run_spec(cfg)
    runner1 = DispatchRunner()
    controller1, _s1, _l1 = _build(spec1, runner1)
    controller1.run(1, bootstrap_or_load_population(spec1))
    assert runner1.agent_runs == 5  # 5 offspring windows
    gen0_markers = [count_markers(m.genome_dir)
                    for m in bootstrap_or_load_population(load_run_spec(cfg)).models]

    # resume: a fresh controller over the same dirs, asked for 2 generations
    spec2 = load_run_spec(cfg)
    spec2.generations = 2
    runner2 = DispatchRunner()
    controller2, _s2, _l2 = _build(spec2, runner2)
    final = controller2.run(2, bootstrap_or_load_population(spec2))

    # gen 0 was NOT recomputed — only gen 1's 5 offspring windows ran on the second pass
    assert runner2.agent_runs == 5
    assert controller2.state_store.load(0).completed is True
    assert controller2.state_store.load(1).completed is True
    # gen-0 survivors' genomes were not re-mutated (marker counts didn't double)
    assert max(count_markers(m.genome_dir) for m in final.models) == 2  # gen0 + gen1
    assert max(gen0_markers) == 1
