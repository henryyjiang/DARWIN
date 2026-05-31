"""In-container mutation entrypoint (ARCHITECTURE.md §4 / §8.5).

Exercises the module the `darwin-agent` image runs: env parsing, the result handoff, and a full
window driven by a fake backend over a real temp git genome (no Claude/Docker), mirroring
test_mutation_runner.
"""

import sys
from pathlib import Path

from darwin.memory import MemoryStore
from darwin.mutation_agent.backend import MutationContext
from darwin.mutation_agent.deadline import DeadlineManager
from darwin.mutation_agent.entrypoint import (
    MutationRunConfig,
    parse_command,
    read_result,
    run_window,
)


def make_genome(path: Path, ok: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "recipe.py").write_text(f"OK = {ok}\n", encoding="utf-8")
    (path / "smoke_test.py").write_text(
        "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )


def never_ending_deadline() -> DeadlineManager:
    return DeadlineManager(window_s=100, soft_lead_s=20, kill_grace_s=10,
                           clock=lambda: 0.0, start=0.0)


class GreenBackend:
    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        (ctx.genome_dir / "recipe.py").write_text("OK = True\n# improved\n", encoding="utf-8")
        assert ctx.checkpoint("improved recipe") is True
        ctx.write_memory(thesis="t", changes="c", smoke_results="green", outcome="o", cost_usd=1.0)


# ------------------------------------------------------------------ pure parsing


def test_parse_command_json_and_shell():
    assert parse_command('["python", "smoke.py"]') == ["python", "smoke.py"]
    assert parse_command("python smoke_test.py") == ["python", "smoke_test.py"]
    assert parse_command("") == []


def test_from_env_reads_darwin_vars():
    cfg = MutationRunConfig.from_env({
        "DARWIN_OFFSPRING_ID": "o0", "DARWIN_MODEL": "o0", "DARWIN_PARENT_SURVIVOR": "s1",
        "DARWIN_MUTATOR": "s2", "DARWIN_GENERATION": "3", "DARWIN_ITERATION": "7",
        "DARWIN_BACKEND": "local", "DARWIN_BASE_FITNESS": "0.4",
        "DARWIN_SMOKE_CMD": '["python", "smoke_test.py"]', "DARWIN_WINDOW_H": "2",
    })
    assert cfg.model == "o0" and cfg.mutator == "s2" and cfg.generation == 3
    assert cfg.iteration == 7 and cfg.backend_name == "local"
    assert cfg.base_fitness == 0.4 and cfg.smoke_command == ["python", "smoke_test.py"]
    assert cfg.window_h == 2.0


def test_from_env_defaults_mutator_when_blank():
    assert MutationRunConfig.from_env({}).mutator == "claude"  # schema needs non-empty (§7.2)


# ------------------------------------------------------------------ full window


def test_run_window_drives_window_and_writes_result(tmp_path):
    genome = tmp_path / "genome"
    make_genome(genome)
    cfg = MutationRunConfig(
        offspring_id="o0", model="o0", parent_survivor="s1", mutator="s2",
        generation=1, iteration=0, backend_name="claude",
        genome_dir=str(genome), store_root=str(tmp_path / "store"),
        result_out=str(tmp_path / "scratch" / "result.json"),
        smoke_command=[sys.executable, "smoke_test.py"],
    )
    store = MemoryStore(tmp_path / "store")

    result = run_window(cfg, GreenBackend(), deadline=never_ending_deadline(), store=store)

    assert result.produced_green is True and result.mutation_failed is False
    # the final genome is green and the edit survived
    assert "# improved" in (genome / "recipe.py").read_text(encoding="utf-8")
    # the agent's memory file landed in the (container-local) store
    assert store.read_iteration("o0", 0).changes == "c"
    # the result handoff JSON the host reads back
    payload = read_result(cfg.result_out)
    assert payload["final_commit"] == result.final_commit
    assert payload["mutation_failed"] is False
    assert payload["model"] == "o0" and payload["iteration"] == 0
