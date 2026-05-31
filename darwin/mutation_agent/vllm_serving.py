"""vLLM serving for the local mutation backend (ARCHITECTURE.md §4.6).

When the mutator M is the population model itself, M's LoRA-merged weights are served by
**vLLM** behind an **OpenAI-compatible** `/v1/chat/completions` endpoint, so any harness that
speaks the OpenAI tool-calling schema (OpenHands, or the §9.3 shim) drives it unmodified. M's
adapter is either pre-merged into the base or loaded dynamically via vLLM's `--enable-lora`
(the pre-merge is what §6.2 notes serving backends need).

This module builds the launch command (pure → testable) and scaffolds the process launcher
(`VLLMServer`). Actually starting vLLM needs a GPU + the `vllm` package (optional `local`
extra), so the live launch is lazy/deferred — mirroring the deferral of the Claude SDK session
(§4.5) and the Lambda finetune (§5.3).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLLMServeConfig:
    """Configuration for serving M's weights via vLLM (§4.6)."""

    base_model: str  # base path/id; if the adapter is pre-merged, this is the merged model
    served_model_name: str = "darwin-local"
    adapter_path: str | None = None  # LoRA adapter to load dynamically (when enable_lora)
    enable_lora: bool = True  # False => base_model is already the merged model
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = "darwin-local"  # vLLM requires a key; harness must present the same one
    tool_call_parser: str = "hermes"  # enables OpenAI tool-call parsing for the served model
    dtype: str = "auto"
    max_model_len: int | None = None
    extra_args: tuple[str, ...] = ()  # escape hatch for instance-specific flags

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL a harness/shim points at."""
        return f"http://{self.host}:{self.port}/v1"


def build_serve_command(cfg: VLLMServeConfig) -> list[str]:
    """The `vllm serve ...` argv for an OpenAI-compatible, tool-calling endpoint (§4.6)."""
    cmd = [
        "vllm",
        "serve",
        cfg.base_model,
        "--served-model-name",
        cfg.served_model_name,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--api-key",
        cfg.api_key,
        "--dtype",
        cfg.dtype,
        # tool-call parsing so the OpenAI tools=[...] schema works for the local model
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        cfg.tool_call_parser,
    ]
    if cfg.max_model_len is not None:
        cmd += ["--max-model-len", str(cfg.max_model_len)]
    if cfg.enable_lora and cfg.adapter_path:
        cmd += ["--enable-lora", "--lora-modules", f"{cfg.served_model_name}={cfg.adapter_path}"]
    cmd += list(cfg.extra_args)
    return cmd


class VLLMServer:
    """Launches/stops a vLLM server for one mutation window (§4.6) — scaffold.

    The live path: `subprocess.Popen(build_serve_command(cfg))` on the GPU host, poll
    `{base_url}/models` until ready, hand `base_url`/`api_key`/`served_model_name` to the
    harness, and terminate on window end. Needs a GPU + the `vllm` package (optional `local`
    extra), so it is deferred here like the other live-infra entry points.
    """

    def __init__(self, config: VLLMServeConfig):
        self.config = config

    @property
    def base_url(self) -> str:
        return self.config.base_url

    def start(self) -> None:
        raise NotImplementedError(
            "VLLMServer.start is scaffolded; the live launch needs a GPU + the vllm package "
            "(optional `local` extra). Use build_serve_command(config) to get the argv."
        )

    def stop(self) -> None:  # pragma: no cover - scaffold
        raise NotImplementedError("VLLMServer.stop is scaffolded; lands with the live path.")
