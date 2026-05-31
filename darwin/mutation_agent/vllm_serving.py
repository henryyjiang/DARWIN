"""vLLM serving for the local mutation backend (ARCHITECTURE.md §4.6).

When the mutator M is the population model itself, M's LoRA-merged weights are served by
**vLLM** behind an **OpenAI-compatible** `/v1/chat/completions` endpoint, so any harness that
speaks the OpenAI tool-calling schema (OpenHands, or the §9.3 shim) drives it unmodified. M's
adapter is either pre-merged into the base or loaded dynamically via vLLM's `--enable-lora`
(the pre-merge is what §6.2 notes serving backends need).

This module builds the launch command (pure → testable) and the process launcher
(`VLLMServer`): `subprocess.Popen` the command, poll the OpenAI-compatible `/models` endpoint
until ready, hand `base_url`/`api_key` to the harness, terminate on window end. The launch/poll
orchestration is testable via injected `popen`/`readiness_check`/`sleep`/`clock` seams; the
defaults shell out to real `vllm` + localhost HTTP, which needs a GPU + the `vllm` package
(optional `local` extra) — the only genuinely-live part.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Callable
from urllib.request import Request, urlopen


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


def _http_ready(base_url: str, api_key: str) -> bool:  # pragma: no cover - real localhost HTTP
    """Default readiness check: GET {base_url}/models returns 200."""
    req = Request(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req, timeout=5) as resp:  # noqa: S310 (localhost, controller-owned)
            return resp.status == 200
    except Exception:
        return False


class VLLMServerError(RuntimeError):
    """vLLM failed to launch or become ready (§4.6)."""


class VLLMServer:
    """Launches/stops a vLLM server for one mutation window (§4.6).

    `start()` Popens `build_serve_command(config)`, polls `{base_url}/models` until ready (or the
    process exits / the timeout elapses), and returns `base_url`; `stop()` terminates it. The
    `popen` / `readiness_check` / `sleep` / `clock` seams are injected so this orchestration is
    unit-tested without a GPU; the defaults are real `subprocess.Popen` + localhost HTTP (needs
    the `vllm` package + a GPU). Usable as a context manager.
    """

    def __init__(
        self,
        config: VLLMServeConfig,
        *,
        popen: Callable[[list[str]], "subprocess.Popen"] | None = None,
        readiness_check: Callable[[str, str], bool] | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.config = config
        self._popen = popen or (lambda cmd: subprocess.Popen(cmd))
        self._ready = readiness_check or _http_ready
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._proc: "subprocess.Popen | None" = None

    @property
    def base_url(self) -> str:
        return self.config.base_url

    def start(self, *, timeout_s: float = 600.0, poll_s: float = 5.0) -> str:
        """Launch vLLM and block until it serves `/models`; return `base_url` (§4.6)."""
        if self._proc is not None:
            return self.base_url
        self._proc = self._popen(build_serve_command(self.config))
        start = self._clock()
        while True:
            if self._ready(self.base_url, self.config.api_key):
                return self.base_url
            rc = self._proc.poll()
            if rc is not None:
                raise VLLMServerError(f"vllm exited with code {rc} before becoming ready")
            if self._clock() - start > timeout_s:
                self.stop()
                raise VLLMServerError(f"vllm not ready after {timeout_s}s")
            self._sleep(poll_s)

    def stop(self) -> None:
        """Terminate the vLLM process (window end / failure cleanup)."""
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except Exception:  # pragma: no cover - escalate to kill
            self._proc.kill()
        self._proc = None

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
