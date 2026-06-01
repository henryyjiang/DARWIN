"""Real-Claude mutation path wiring (TEST_RUN_PLAN real-Claude opt-in).

Pure/offline coverage of the pieces that make the in-container Claude session functional: the
small-changes directive, the agent options (MCP allow-listing + model/effort/budget), the darwin-mcp
stdio config, the backend builder, the env config, and the env-built agent MCP server. The live SDK
session itself is exercised only by an actual run with an API key.
"""

import sys
from pathlib import Path

from darwin.config import MutationConfig
from darwin.mutation_agent.claude_backend import ALLOWED_TOOLS, build_agent_options
from darwin.mutation_agent.directive import (
    DIRECTIVE_SYSTEM_PROMPT,
    SMALL_DIRECTIVE_SYSTEM_PROMPT,
    build_directive,
)
from darwin.mutation_agent.entrypoint import (
    MutationRunConfig,
    _build_claude_backend,
    mcp_servers_config,
)


def test_small_directive_is_compact_and_green_focused():
    full = build_directive(offspring_id="o0", model="o0", parent_survivor="s0", mutator="s1",
                           generation=0, style="full")
    small = build_directive(offspring_id="o0", model="o0", parent_survivor="s0", mutator="s1",
                            generation=0, style="small")
    assert small != full
    assert "small" in small.lower()
    assert "smoke.run" in small and "memory.write_iteration" in small
    # the small mission must not drag in the heavy param-scaling framing
    assert "data mix" not in small.lower() and "architectural adapter" not in small.lower()


def test_config_defaults_for_claude_knobs():
    m = MutationConfig()
    assert m.directive_style == "full"
    assert m.claude_model == "" and m.claude_effort == "" and m.claude_max_budget_usd == 0.0
    assert m.claude_sample == 0


def test_build_agent_options_allowlists_mcp_and_threads_knobs():
    class _Ctx:
        genome_dir = "/work/genome"
        directive = "task"

    opts = build_agent_options(
        _Ctx(), mcp_servers={"darwin": {"type": "stdio"}},
        system_prompt=SMALL_DIRECTIVE_SYSTEM_PROMPT, model="m", effort="low", max_budget_usd=1.0,
    )
    assert opts["system_prompt"] == SMALL_DIRECTIVE_SYSTEM_PROMPT
    assert "mcp__darwin" in opts["allowed_tools"]          # MCP tools allow-listed
    assert set(ALLOWED_TOOLS).issubset(set(opts["allowed_tools"]))
    assert opts["model"] == "m" and opts["effort"] == "low" and opts["max_budget_usd"] == 1.0
    assert opts["permission_mode"] == "bypassPermissions"


def test_build_agent_options_omits_unset_knobs():
    class _Ctx:
        genome_dir = "/g"
        directive = "t"

    opts = build_agent_options(_Ctx(), mcp_servers={})
    assert "model" not in opts and "effort" not in opts and "max_budget_usd" not in opts
    assert opts["system_prompt"] == DIRECTIVE_SYSTEM_PROMPT  # default preserved


def test_mcp_servers_config_is_stdio_darwin_server():
    cfg = mcp_servers_config({"DARWIN_STORE_ROOT": "/work/scratch/store", "PATH": "x"})
    assert set(cfg) == {"darwin"}
    d = cfg["darwin"]
    assert d["type"] == "stdio" and d["command"] == sys.executable
    assert d["args"] == ["-m", "darwin.mcp.server"]
    assert d["env"]["DARWIN_STORE_ROOT"] == "/work/scratch/store"  # full env forwarded


def test_mutation_run_config_reads_claude_env():
    cfg = MutationRunConfig.from_env({
        "DARWIN_BACKEND": "claude", "DARWIN_DIRECTIVE_STYLE": "small",
        "DARWIN_CLAUDE_MODEL": "m", "DARWIN_CLAUDE_EFFORT": "low",
        "DARWIN_CLAUDE_MAX_BUDGET_USD": "2.5",
    })
    assert cfg.directive_style == "small" and cfg.claude_model == "m"
    assert cfg.claude_effort == "low" and cfg.claude_max_budget_usd == 2.5


def test_build_claude_backend_wires_small_prompt_and_mcp():
    from darwin.mutation_agent.backend import MutationContext
    from darwin.mutation_agent.checkpoint import GitCheckpointer
    from darwin.mutation_agent.smoke import SmokeTest
    from darwin.memory import MemoryStore

    ctx = MutationContext(
        offspring_id="o0", genome_dir=Path("/work/genome"), model="o0", parent_survivor="s0",
        mutator="claude", generation=0, iteration=0, backend_name="claude", base_fitness=0.0,
        directive="t", checkpointer=GitCheckpointer("/work/genome"),
        smoke=SmokeTest(command=[]), store=MemoryStore("/work/scratch/store"),
    )
    backend = _build_claude_backend(
        {"DARWIN_DIRECTIVE_STYLE": "small", "DARWIN_CLAUDE_EFFORT": "low",
         "DARWIN_CLAUDE_MAX_BUDGET_USD": "1.0"}, ctx,
    )
    assert backend.system_prompt == SMALL_DIRECTIVE_SYSTEM_PROMPT
    assert "darwin" in backend.mcp_servers
    assert backend.effort == "low" and backend.max_budget_usd == 1.0


def test_build_agent_server_from_env(tmp_path):
    """With an offspring id set, the env-built MCP server binds to the window context (no crash)."""
    from darwin.mcp.server import build_agent_server_from_env

    genome = tmp_path / "genome"
    genome.mkdir()
    (genome / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    env = {
        "DARWIN_OFFSPRING_ID": "o0", "DARWIN_MODEL": "o0",
        "DARWIN_GENOME_DIR": str(genome), "DARWIN_STORE_ROOT": str(tmp_path / "store"),
        "DARWIN_DIRECTIVE_STYLE": "small", "DARWIN_SMOKE_CMD": '["python","smoke_test.py"]',
    }
    server = build_agent_server_from_env(env)
    assert server is not None
    # memory-only server when no offspring id (original behavior)
    bare = build_agent_server_from_env({"DARWIN_STORE_ROOT": str(tmp_path / "s2")})
    assert bare is not None


def test_window_env_emits_directive_and_claude_knobs(tmp_path):
    """ContainerGenerationOps forwards the directive style + Claude knobs into the container env."""
    from darwin.bench import EvalContainerBenchmarkBackend
    from darwin.config import DarwinConfig
    from darwin.controller import ContainerGenerationOps, Model
    from darwin.controller.state import OffspringState
    from darwin.cost import CostLedger
    from darwin.finetune import ContainerFinetuneBackend
    from darwin.memory import MemoryStore
    from darwin.sandbox import ContainerResult

    class _R:
        def run(self, spec, *, dry_run=False):
            return ContainerResult(0, "", "", [])

    cfg = DarwinConfig()
    cfg.mutation.directive_style = "small"
    cfg.mutation.claude_model = "m"
    cfg.mutation.claude_effort = "low"
    cfg.mutation.claude_max_budget_usd = 1.5
    ops = ContainerGenerationOps(
        config=cfg, store=MemoryStore(tmp_path / "store"), ledger=CostLedger(tmp_path / "c.jsonl"),
        workspace=tmp_path / "ws", finetune_backend=ContainerFinetuneBackend(runner=_R()),
        benchmark_backend=EvalContainerBenchmarkBackend(runner=_R()),
        smoke_command=["python", "smoke_test.py"], container_runner=_R(),
    )
    off = Model(name="o0", genome_dir=tmp_path / "ws" / "o0" / "genome")
    state = OffspringState(name="o0", parent_survivor="s0", mutator="s1", backend="claude", iteration=0)
    env = ops._window_env(off, state, "claude", 0, "/work/scratch/store", 0.0)
    assert env["DARWIN_DIRECTIVE_STYLE"] == "small"
    assert env["DARWIN_CLAUDE_MODEL"] == "m" and env["DARWIN_CLAUDE_EFFORT"] == "low"
    assert env["DARWIN_CLAUDE_MAX_BUDGET_USD"] == "1.5"
