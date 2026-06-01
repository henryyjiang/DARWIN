"""Local-model mutation backend + backend factory (ARCHITECTURE.md §4.6 / §4.7)."""

import sys
from pathlib import Path

import pytest

from darwin.memory import MemoryStore
from darwin.mutation_agent import (
    DeadlineManager,
    GitCheckpointer,
    LocalMutationBackend,
    SmokeTest,
    VLLMServeConfig,
    build_harness_config,
    make_mutation_backend_factory,
    run_mutation_window,
)
from darwin.mutation_agent.backend import MutationContext
from darwin.mutation_agent.claude_backend import ClaudeMutationBackend
from darwin.mutation_agent.directive import DIRECTIVE_SYSTEM_PROMPT
from darwin.mutation_agent.local_backend import build_llm_kwargs, to_openhands_mcp_config


def make_ctx(tmp_path: Path) -> MutationContext:
    genome = tmp_path / "genome"
    genome.mkdir(parents=True, exist_ok=True)
    (genome / "recipe.py").write_text("OK = True\n", encoding="utf-8")
    (genome / "smoke_test.py").write_text(
        "import recipe, sys\nsys.exit(0 if recipe.OK else 1)\n", encoding="utf-8"
    )
    return MutationContext(
        offspring_id="7", genome_dir=genome, model="model7", parent_survivor="model3",
        mutator="model2", generation=5, iteration=0, backend_name="local",
        base_fitness=0.5, directive="DIRECTIVE-TEXT", checkpointer=GitCheckpointer(genome),
        smoke=SmokeTest(command=[sys.executable, "smoke_test.py"]),
        store=MemoryStore(tmp_path / "store"),
    )


def never_ending_deadline() -> DeadlineManager:
    return DeadlineManager(window_s=100, soft_lead_s=20, kill_grace_s=10,
                           clock=lambda: 0.0, start=0.0)


def test_build_harness_config_maps_fields(tmp_path):
    ctx = make_ctx(tmp_path)
    serve = VLLMServeConfig(base_model="qwen", served_model_name="m7", port=8001, api_key="k")
    cfg = build_harness_config(ctx, serve, mcp_servers={"darwin": {}}, max_iterations=42)
    assert cfg.workspace == str(ctx.genome_dir)
    assert cfg.base_url == "http://127.0.0.1:8001/v1"
    assert cfg.api_key == "k"
    assert cfg.model == "m7"
    assert cfg.task == "DIRECTIVE-TEXT"  # the same directive the Claude backend gets (§4.6)
    assert cfg.system_prompt == DIRECTIVE_SYSTEM_PROMPT
    assert cfg.max_iterations == 42
    assert cfg.mcp_servers == {"darwin": {}}


def test_local_backend_runs_through_window_with_fake_harness(tmp_path):
    ctx = make_ctx(tmp_path)
    captured = {}

    def fake_harness(config, c, deadline):
        captured["config"] = config
        # a weak-but-working agent: one green edit, checkpoint, write memory
        (c.genome_dir / "recipe.py").write_text("OK = True\n# local-mutated\n", encoding="utf-8")
        assert c.checkpoint("local edit") is True
        c.write_memory(thesis="t", changes="c", smoke_results="green", outcome="o", cost_usd=0.0)

    backend = LocalMutationBackend(
        serve_config=VLLMServeConfig(base_model="qwen", served_model_name="m7"),
        harness_runner=fake_harness,
    )
    result = run_mutation_window(ctx, backend, never_ending_deadline())

    assert result.produced_green is True
    assert result.mutation_failed is False
    assert result.memory_written is True
    assert "# local-mutated" in (ctx.genome_dir / "recipe.py").read_text(encoding="utf-8")
    # the backend handed the harness the right model/endpoint
    assert captured["config"].model == "m7"
    assert captured["config"].base_url.endswith("/v1")


def test_local_backend_default_runner_needs_openhands(tmp_path):
    # The default runner is the live OpenHands V1-SDK session; without the `local` extra installed
    # it fails at the lazy `import openhands.sdk` — the genuinely-deferred part (needs GPU + deps).
    ctx = make_ctx(tmp_path)
    ctx.checkpointer.init_offspring("7")
    backend = LocalMutationBackend(serve_config=VLLMServeConfig(base_model="qwen"))
    with pytest.raises(ImportError):
        backend.run(ctx, never_ending_deadline())


# ------------------------------------------------------------ OpenHands config (pure, §4.6)


def test_build_llm_kwargs_targets_openai_compatible_endpoint(tmp_path):
    ctx = make_ctx(tmp_path)
    serve = VLLMServeConfig(base_model="qwen", served_model_name="m7", port=8001, api_key="k")
    kwargs = build_llm_kwargs(build_harness_config(ctx, serve))
    # litellm routes to the OpenAI-compatible vLLM endpoint via the `openai/` model prefix + base_url
    assert kwargs == {"model": "openai/m7", "base_url": "http://127.0.0.1:8001/v1", "api_key": "k"}


def test_to_openhands_mcp_config_translates_stdio_shape():
    ours = {
        "darwin": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "darwin.mcp.server"],
            "env": {"DARWIN_OFFSPRING_ID": "7"},
        }
    }
    assert to_openhands_mcp_config(ours) == {
        "mcpServers": {
            "darwin": {
                "command": "python",
                "args": ["-m", "darwin.mcp.server"],
                "env": {"DARWIN_OFFSPRING_ID": "7"},
            }
        }
    }


def test_to_openhands_mcp_config_empty_and_non_stdio():
    assert to_openhands_mcp_config({}) == {}
    # a non-stdio entry (e.g. sse) is skipped; an all-skipped input yields an empty config
    assert to_openhands_mcp_config({"x": {"type": "sse", "url": "http://h/sse"}}) == {}


# ------------------------------------------------------------------ factory (§4.7)


def test_factory_routes_local_to_local_backend(tmp_path):
    serve = VLLMServeConfig(base_model="qwen")
    factory = make_mutation_backend_factory(serve_config=serve)
    backend = factory("local", make_ctx(tmp_path))
    assert isinstance(backend, LocalMutationBackend)


def test_factory_routes_claude_and_fallback_to_claude_backend(tmp_path):
    factory = make_mutation_backend_factory()  # no serve_config needed for claude
    ctx = make_ctx(tmp_path)
    assert isinstance(factory("claude", ctx), ClaudeMutationBackend)
    # the degenerate <2-survivor fallback resolves to backend name "claude" upstream
    assert isinstance(factory("claude", ctx), ClaudeMutationBackend)


def test_factory_local_without_serve_config_raises(tmp_path):
    factory = make_mutation_backend_factory()
    with pytest.raises(ValueError):
        factory("local", make_ctx(tmp_path))


def test_factory_writes_claude_transcript_path(tmp_path):
    factory = make_mutation_backend_factory(claude_transcript_dir=tmp_path / "transcripts")
    backend = factory("claude", make_ctx(tmp_path))
    assert backend.transcript_path == tmp_path / "transcripts" / "7.jsonl"
