"""Backend B — local-model harness (ARCHITECTURE.md §4.6).

When the mutator M is the population model itself, M's LoRA-merged weights are served by vLLM
(`vllm_serving.py`) behind an OpenAI-compatible endpoint, and the **OpenHands** agentic harness
(the V1 `openhands-sdk`) turns that raw chat model into a tool-using coder driving the **same**
`darwin-mcp` tools + directive as the Claude backend. OpenHands speaks OpenAI tool-calling and
supports MCP servers natively, so the `darwin-mcp` server attaches directly (no translation shim
needed, §9.3). That tool/directive parity is what makes §3.2's "M's identity drives the mutation"
hold meaningfully and keeps results comparable across backends. OpenHands is also the deliberate
forward-looking choice: as the population scales up in parameters across generations, the mutator
grows into OpenHands' full planning/recovery/sub-agent feature suite rather than outgrowing a
minimal hand-rolled loop.

This implements the §4.2 `MutationBackend.run(ctx, deadline)` contract, so it plugs into the
controller's `mutation_backend_factory` seam exactly where `backend="local"` routes (the
controller already does that routing, §4.7). The pure, testable surfaces are `build_harness_config`,
`build_llm_kwargs`, and `to_openhands_mcp_config`; the live OpenHands session needs a GPU-served
endpoint + the harness package (optional `local` extra) and is delegated to an injectable
`harness_runner` (the real default is `_run_openhands`), mirroring the deferral of the Claude SDK
session and the Lambda finetune. The always-green finalization + the kill-and-recover guarantee are
backend-agnostic and handled by `run_mutation_window` / the checkpointer (§9.3), so pausing the
harness mid-step on the deadline is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from darwin.mutation_agent.backend import MutationContext
from darwin.mutation_agent.deadline import DeadlineManager, Phase
from darwin.mutation_agent.directive import DIRECTIVE_SYSTEM_PROMPT
from darwin.mutation_agent.vllm_serving import VLLMServeConfig

# A larger turn budget than the Claude backend, within the same wall-clock window: a 32B coder
# is materially weaker at long-horizon agentic work, so it gets more, smaller steps (§4.6).
DEFAULT_MAX_ITERATIONS = 250


@dataclass
class HarnessConfig:
    """What the agentic harness needs to drive one mutation window (§4.6)."""

    workspace: str  # the offspring genome repo (cwd)
    base_url: str  # vLLM OpenAI-compatible endpoint
    api_key: str
    model: str  # the served model name (M's weights)
    system_prompt: str  # the shared directive system prompt (tool/directive parity, §4.6)
    task: str  # the predetermined mutation directive
    mcp_servers: dict[str, Any]  # darwin-mcp attach (OpenHands native MCP support, §9.3)
    max_iterations: int


# A harness runner drives the live session; injected so the backend is testable with a fake.
HarnessRunner = Callable[[HarnessConfig, MutationContext, DeadlineManager], None]


def build_harness_config(
    ctx: MutationContext,
    serve: VLLMServeConfig,
    *,
    mcp_servers: dict[str, Any] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> HarnessConfig:
    """Build the harness config for an offspring's window (pure → testable)."""
    return HarnessConfig(
        workspace=str(ctx.genome_dir),
        base_url=serve.base_url,
        api_key=serve.api_key,
        model=serve.served_model_name,
        system_prompt=DIRECTIVE_SYSTEM_PROMPT,
        task=ctx.directive,
        mcp_servers=dict(mcp_servers or {}),
        max_iterations=max_iterations,
    )


class LocalMutationBackend:
    """Drives the population model as the mutator via an agentic harness (§4.6)."""

    def __init__(
        self,
        serve_config: VLLMServeConfig,
        mcp_servers: dict[str, Any] | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        harness_runner: HarnessRunner | None = None,
    ):
        self.serve_config = serve_config
        self.mcp_servers = mcp_servers or {}
        self.max_iterations = max_iterations
        self.harness_runner = harness_runner

    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        config = build_harness_config(
            ctx,
            self.serve_config,
            mcp_servers=self.mcp_servers,
            max_iterations=self.max_iterations,
        )
        runner = self.harness_runner or _run_openhands
        runner(config, ctx, deadline)


# litellm (which the OpenHands SDK's LLM wraps) routes to an OpenAI-compatible endpoint when the
# model id is prefixed `openai/` and a `base_url` is supplied; vLLM serves M under its
# `served_model_name`, so M's litellm model id is `openai/<served_model_name>`.
OPENAI_COMPAT_PREFIX = "openai"


def build_llm_kwargs(config: HarnessConfig) -> dict[str, Any]:
    """The `openhands.sdk.LLM(...)` kwargs pointing at M's vLLM endpoint (pure → testable)."""
    return {
        "model": f"{OPENAI_COMPAT_PREFIX}/{config.model}",
        "base_url": config.base_url,
        "api_key": config.api_key,
    }


def to_openhands_mcp_config(mcp_servers: dict[str, Any]) -> dict[str, Any]:
    """Translate our darwin-mcp stdio config into OpenHands' `mcp_config` shape (pure → testable).

    Our config (the Claude-SDK shape built by `entrypoint.mcp_servers_config`) is
    `{"darwin": {"type": "stdio", "command": ..., "args": [...], "env": {...}}}`; the OpenHands SDK
    Agent wants `{"mcpServers": {"darwin": {"command": ..., "args": [...], "env": {...}}}}` (no
    `type` key). Only stdio servers are forwarded — the only kind we attach — and an empty input
    yields an empty config (no MCP).
    """
    servers: dict[str, Any] = {}
    for name, spec in mcp_servers.items():
        if not isinstance(spec, dict) or spec.get("type", "stdio") != "stdio":
            continue
        entry: dict[str, Any] = {"command": spec.get("command")}
        if spec.get("args"):
            entry["args"] = list(spec["args"])
        if spec.get("env"):
            entry["env"] = dict(spec["env"])
        servers[name] = entry
    return {"mcpServers": servers} if servers else {}


def _run_openhands(
    config: HarnessConfig, ctx: MutationContext, deadline: DeadlineManager
) -> None:
    """Live OpenHands V1-SDK session against M's vLLM endpoint (§4.6).

    Builds an `openhands.sdk` Agent — terminal + file-editor tools (the local-model equivalent of
    the Claude backend's Bash/Edit/Write/Read built-ins) plus `darwin-mcp` (memory/smoke/finalize/
    cost/paper/data) attached via the SDK's native `mcp_config` — driven by M's weights over the
    OpenAI-compatible vLLM endpoint, and runs the predetermined directive against the mounted
    genome. A wall-clock watcher injects the soft-deadline nudge (best-effort, §9.3) and `pause()`s
    the run on the hard/kill deadline; `pause()` is the SDK's thread-safe stop. The always-green
    finalize + recover-last-green guarantee is `run_mutation_window`'s job (§9.3), so pausing
    mid-step loses no green work. Needs a GPU-served endpoint + the `local` extra (openhands-sdk /
    openhands-tools), so the SDK is lazy-imported; inject a `harness_runner` to drive the contract
    without it.
    """
    import threading

    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    # Reuse the Claude backend's tested wall-clock injection policy for backend parity (§9.3).
    from darwin.mutation_agent.claude_backend import next_injection

    llm = LLM(**build_llm_kwargs(config))
    agent = Agent(
        llm=llm,
        tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
        mcp_config=to_openhands_mcp_config(config.mcp_servers),
        system_prompt=config.system_prompt,  # the shared directive (tool/directive parity, §4.6)
    )
    conversation = Conversation(
        agent=agent,
        workspace=config.workspace,
        max_iteration_per_run=config.max_iterations,
    )
    print(f"[openhands] session start: model={config.model} base_url={config.base_url} "
          f"mcp={list(config.mcp_servers)} max_iter={config.max_iterations}", flush=True)
    try:  # best-effort: confirm the darwin-mcp tools attached alongside terminal/file-editor
        tool_names = sorted(getattr(agent, "tools_map", {}) or {})
        if tool_names:
            print(f"[openhands] tools loaded: {tool_names}", flush=True)
    except Exception:  # pragma: no cover - live SDK only
        pass

    error: list[BaseException] = []

    def _drive() -> None:
        try:
            conversation.send_message(config.task)
            conversation.run()
        except BaseException as exc:  # re-raised on the main thread after join
            error.append(exc)

    worker = threading.Thread(target=_drive, name=f"openhands-{ctx.offspring_id}", daemon=True)
    worker.start()

    soft_sent = False
    while worker.is_alive():
        worker.join(timeout=5.0)
        if not worker.is_alive():
            break
        phase = deadline.phase()
        injection, soft_sent = next_injection(phase, deadline.remaining(), soft_sent)
        if injection is not None:
            try:  # best-effort mid-run nudge; the real guarantee is the runner's finalize (§9.3)
                conversation.send_message(injection)
            except Exception as exc:  # pragma: no cover - live SDK only
                print(f"[openhands] inject failed: {type(exc).__name__}: {exc}", flush=True)
        if phase in (Phase.HARD, Phase.KILL):
            try:
                conversation.pause()  # thread-safe; takes effect at the next agent step
            except Exception as exc:  # pragma: no cover - live SDK only
                print(f"[openhands] pause failed: {type(exc).__name__}: {exc}", flush=True)
            break

    worker.join(timeout=30.0)
    if error:
        print(f"[openhands] session ERROR: {type(error[0]).__name__}: {error[0]}", flush=True)
        raise error[0]
    print("[openhands] session end", flush=True)


def make_mutation_backend_factory(
    *,
    serve_config: VLLMServeConfig | None = None,
    mcp_servers: dict[str, Any] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    harness_runner: HarnessRunner | None = None,
    claude_transcript_dir: Path | str | None = None,
) -> Callable[[str, MutationContext], Any]:
    """A default `mutation_backend_factory` for `LocalGenerationOps` (§4.7 routing).

    Maps the per-offspring backend name to a concrete backend: `mock` → `MockMutationBackend` (the
    offline test-profile mutator, §3.3); `local` → `LocalMutationBackend` (requires a
    `serve_config`); anything else (`claude`, and the degenerate <2-survivor fallback) →
    `ClaudeMutationBackend`. The live backends attach the same `darwin-mcp` so the directive and
    tools are identical across them (§9.4).
    """
    # Lazy import to avoid a hard dependency cycle / keep import light.
    from darwin.mutation_agent.claude_backend import ClaudeMutationBackend

    def factory(backend_name: str, ctx: MutationContext) -> Any:
        if backend_name == "mock":
            from darwin.mutation_agent.mock_backend import MockMutationBackend

            return MockMutationBackend()
        if backend_name == "local":
            if serve_config is None:
                raise ValueError("local backend requires a VLLMServeConfig (serve_config)")
            return LocalMutationBackend(
                serve_config=serve_config,
                mcp_servers=mcp_servers,
                max_iterations=max_iterations,
                harness_runner=harness_runner,
            )
        transcript = None
        if claude_transcript_dir is not None:
            transcript = Path(claude_transcript_dir) / f"{ctx.offspring_id}.jsonl"
        return ClaudeMutationBackend(mcp_servers=mcp_servers or {}, transcript_path=transcript)

    return factory
